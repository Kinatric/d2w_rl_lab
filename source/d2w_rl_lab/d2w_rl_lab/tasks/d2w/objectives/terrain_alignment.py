"""Soft loaded-wheel hip alignment on locally ascending/descending terrain.

The height-scan fit is a local trend estimate, not a traversability certificate.
No action projection, motor change, fixed pitch target, or swing-foot target.
"""
import math
import torch


def smooth(value, low, high):
    t = ((value - low) / (high - low)).clamp(0., 1.)
    return t.square() * (3. - 2. * t)


def height_trend(rays, position, quaternion):
    """Least-squares ground trend in yaw-aligned coordinates, invalid fits released."""
    x, y, z, w = quaternion.unbind(-1)
    heading = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z)), -1)
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    finite = torch.isfinite(rays).all(-1)
    safe = torch.where(finite[..., None], rays, position[:, None, :])
    delta = safe - position[:, None, :]
    u = (delta[..., :2] * heading[:, None, :]).sum(-1)
    v = -delta[..., 0] * heading[:, None, 1] + delta[..., 1] * heading[:, None, 0]
    valid = finite & (u.abs() <= .7) & (v.abs() <= .45)
    weight = valid.to(rays.dtype)
    count = weight.sum(-1).clamp_min(1.)
    def center(a):
        return (a - (a * weight).sum(-1, keepdim=True) / count[:, None]) * weight
    u, v, h = center(u), center(v), center(delta[..., 2])
    xx, yy, xy = (u*u).sum(-1), (v*v).sum(-1), (u*v).sum(-1)
    xh, yh = (u*h).sum(-1), (v*h).sum(-1)
    determinant = xx*yy - xy*xy
    fit_valid = (count >= 8) & (determinant > 1.e-6)
    slope = (xh*yy - yh*xy) / determinant.clamp_min(1.e-6)
    return torch.where(fit_valid, slope, 0.), fit_valid


def alignment_cost(hip, command, gravity_b, load, reference_load, slope_x,
                   fit_valid, stairs, slopes, terminated):
    """FL/FR/RL/RR; cost only outward excess, weighted by real ground support."""
    forward = command[:, 0] > 0
    signed_trend = slope_x * command[:, 0].sign()
    ascending = stairs * smooth(signed_trend, .03, .12)
    descending = (stairs | slopes) * smooth(-signed_trend, .03, .12)
    front = hip.new_tensor([1., 1., 0., 0.])[None, :]
    leading = torch.where(forward[:, None], front, 1. - front)
    scope = ascending[:, None] * leading + descending[:, None]
    moving = smooth(command[:, 0].abs(), .05, .20)
    straight = (1. - smooth(command[:, 1].abs(), .02, .10)) * (1. - smooth(command[:, 2].abs(), .05, .20))
    roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2]).abs()
    balance = (1. - smooth(roll, .15, .35)) * smooth(-gravity_b[:, 2], math.cos(1.), .8)
    gate = moving * straight * balance * fit_valid * (~terminated)
    support = smooth(load, 1., (.5 * reference_load[:, None]).clamp_min(2.))
    outward = hip * hip.new_tensor([1., -1., 1., -1.])
    excess = (outward - .12).clamp_min(0.) / .15
    penalty = excess.square() / (1. + excess.square())
    # Normalize by selected legs, not supported legs: unloading receives no bonus.
    # Contact gating can still be exploited; behavior must be checked after training.
    # Strengthen ascent only. The term weight is tripled; divide descent by
    # three to retain its previous effective cost, deadband and load dependence.
    old_support = smooth(load, 1., reference_load[:, None].clamp_min(2.))
    old_excess = (outward - .15).clamp_min(0.) / .15
    old_penalty = old_excess.square() / (1. + old_excess.square())
    weighted = torch.where(descending[:, None] > 0., old_support * old_penalty / 3., support * penalty)
    # Add only excessive inward abduction on ascent, including the trailing pair.
    # [-0.08, +0.12] rad is a free band, not a fixed posture target.
    # Opposite-sign excess terms have disjoint gradients. Swing/recovery remains free.
    inward_excess = (-outward - .08).clamp_min(0.) / .15
    inward_penalty = inward_excess.square() / (1. + inward_excess.square())
    inward_cost = ascending * (support * inward_penalty).mean(-1)
    cost = gate * ((scope * weighted).sum(-1) / scope.sum(-1).clamp_min(1.) + inward_cost)
    return cost, dict(gate=gate, scope=scope, support=support, outward=outward,
                      ascending=ascending, descending=descending, inward_cost=inward_cost)


def reward(env, asset_cfg, sensor_cfg):
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    from .standing_finetune import METRICS
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.normal_force_matrix_w.torch[:, sensor_cfg.body_ids]
    forces = torch.where(torch.isfinite(forces).all(-1, keepdim=True), forces, 0.)
    gravity = forces.new_tensor(env.cfg.sim.gravity)
    up = -gravity / gravity.norm().clamp_min(1.e-8)
    load = (forces.sum(-2) * up).sum(-1).clamp_min(0.)
    reference = robot.data.body_mass.torch.sum(-1) * gravity.norm() / 4.
    slope, valid = height_trend(env.scene.sensors['height_scanner'].data.ray_hits_w.torch,
                                robot.data.root_pos_w.torch, robot.data.root_quat_w.torch)
    masks = get_terrain_masks(env)
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    stairs = masks.get('stairs_up', zero) | masks.get('stairs_down', zero)
    slopes = masks.get('slope_up', zero) | masks.get('slope_down', zero) | masks.get('rough_slope', zero)
    cost, diagnostics = alignment_cost(robot.data.joint_pos.torch[:, asset_cfg.joint_ids],
        env.command_manager.get_command('base_velocity'), robot.data.projected_gravity_b.torch,
        load, reference, slope, valid, stairs, slopes, env.termination_manager.terminated)
    env._d2w_alignment_diagnostics = {k: v.detach() for k, v in diagnostics.items()}
    env._d2w_alignment_diagnostics.update(cost=cost.detach(), slope=slope.detach(), load=load.detach(),
        rear_load_fraction=(load[:, 2:].sum(-1) / load.sum(-1).clamp_min(1.e-6)).detach())
    METRICS['terrain_hip_alignment'] = cost.mean().detach()
    return cost


def configure(cfg):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    cfg.rewards.terrain_hip_alignment = RewardTermCfg(func=reward, weight=-.30, params=dict(
        asset_cfg=SceneEntityCfg('robot', joint_names=['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint'], preserve_order=True),
        sensor_cfg=SceneEntityCfg('stair_contacts', body_names=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'], preserve_order=True)))
    return cfg
