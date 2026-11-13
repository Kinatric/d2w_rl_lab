"""D2W support-aware relief of persistent wheel blockage at an ascending riser.

Reward-only candidate: no gait clock, action projection, extra actor input, or
motor change. Constants are initial shaping scales, not safety certificates.
Normal-force support is a conservative proxy: tangential wall support is not
counted. Contact arrays may aggregate contacts for each ground shape.
"""
import math

import torch


def smooth(value, low, high):
    t = ((value - low) / (high - low)).clamp(0., 1.)
    return t.square() * (3. - 2. * t)


def command_scope(command, upright, stairs, terminated):
    """Forward ascent practice only; retain turn/reverse/recovery freedom."""
    forward = smooth(command[:, 0], .05, .20)
    straight = (1. - smooth(command[:, 1].abs(), .02, .10))
    straight *= 1. - smooth(command[:, 2].abs(), .05, .20)
    healthy = smooth(upright, math.cos(1.), .8) * (~terminated)
    return forward * straight * healthy * stairs


def ground_support(points, forces, up, reference_load):
    """Ground-filtered upward normal load; a vertical wall is not support."""
    finite = torch.isfinite(points).all(-1) & torch.isfinite(forces).all(-1)
    force = torch.where(finite[..., None], forces, 0.)
    magnitude = force.norm(dim=-1)
    vertical = (force * up).sum(-1)
    supporting = finite & (magnitude > 1.) & (vertical > .5 * magnitude)
    weights = torch.where(supporting, vertical, 0.)
    load = weights.sum(-1)
    surface = (torch.nan_to_num(points[..., 2]) * weights).sum(-1)
    surface /= load.clamp_min(1.e-8)
    # At least 10% of a nominal quarter-weight distinguishes real support from
    # a marginal touch; this is a training label, not a support-polygon proof.
    bearing = load > (.1 * reference_load[:, None]).clamp_min(1.)
    wall = (force - vertical[..., None] * up).norm(dim=-1)
    wall = (wall - .6 * vertical.abs()).clamp_min(0.).sum(-1)
    return load, bearing, surface, wall


def higher_tread_ahead(center, ground, rays, direction, ground_valid):
    """Per-wheel higher tread within 20 cm; no cross-leg symmetry assumption.

    The nearest higher sample is used rather than the highest distant sample.
    Sparse scans can miss a riser; missing/invalid geometry disables the term.
    """
    finite = torch.isfinite(rays).all(-1)
    safe = torch.where(finite[..., None], rays, 0.)
    delta = safe[:, None, :, :2] - center[:, :, None, :2]
    along = (delta * direction[:, None, None, :2]).sum(-1)
    lateral = (-delta[..., 0] * direction[:, None, None, 1]
               + delta[..., 1] * direction[:, None, None, 0])
    rise = safe[:, None, :, 2] - ground[:, :, None]
    nearby = finite[:, None, :] & (along > .025) & (along < .20)
    nearby &= (lateral.abs() < .09) & (rise > .03) & (rise < .30)
    nearby &= ground_valid[:, :, None]
    distance, index = along.masked_fill(~nearby, float('inf')).min(-1)
    present = nearby.any(-1)
    height = safe[:, :, 2].gather(1, index)
    return present, torch.where(present, height, ground), distance


def restep_cost(command, forward_velocity, obstacle, active, wheel_slip,
                wall, load, reference_load, bearing, gate, valid,
                previous_age, dt):
    """Discourage continued loaded pushing/spinning only when unloading is viable.

    Unloading reduces this cost monotonically. It earns no positive reward;
    unchanged verified supported-placement terms reward actual stair progress.
    The same-pair condition releases this cost when the counterpart loses
    support. It cannot guarantee against simultaneous lift on the next step,
    and does not force the next side or prohibit corrective re-steps.
    """
    if dt <= 0.:
        raise ValueError('dt must be positive')
    others = bearing.sum(-1, keepdim=True) - bearing.to(torch.long)
    counterpart = bearing[:, [1, 0, 3, 2]]
    can_unload = (others >= 2) & counterpart
    stalled = 1. - smooth(forward_velocity / command[:, 0].clamp_min(.1), .25, .75)
    loaded = smooth(load, .1 * reference_load[:, None], .6 * reference_load[:, None])
    wall_signal = smooth(wall, .15 * reference_load[:, None], .6 * reference_load[:, None])
    # contact_quality returns bounded material-point tangential slip. Preserve
    # that physical signal; never substitute wheel-center speed or raw R*omega.
    evidence = torch.maximum(wall_signal, wheel_slip.clamp(0., 1.))
    blocked = obstacle & active & (loaded > 0.) & (evidence > 1.e-6)
    legal = valid[:, None] & (gate[:, None] > 0.) & (stalled[:, None] > 0.)
    old = torch.zeros_like(load) if previous_age is None else previous_age.clamp(0., .4)
    # Learn persistence from wheel blockage, independently of whether unloading
    # is safe in this exact sample. Brief contact dropouts decay the evidence
    # instead of erasing it; at most .8 s without new evidence clears it fully.
    age = torch.where(blocked, old + dt, old - .5 * dt).clamp(0., .4)
    age = torch.where(legal, age, 0.)
    context = blocked & can_unload & legal
    # Time-based rather than frame-count based; invalid/reset/unsafe boundaries
    # still clear immediately. The timer itself never produces a reward or cost.
    persistence = smooth(age, .12, .40)
    per_wheel = gate[:, None] * stalled[:, None] * loaded * evidence * persistence
    per_wheel = torch.where(context, per_wheel, 0.)
    # Fixed four-wheel normalization avoids rewarding dilution by extra contacts.
    value = per_wheel.mean(-1)
    return value, age, dict(can_unload=can_unload, context=context, blocked=blocked,
                           loaded=loaded, evidence=evidence, per_wheel=per_wheel,
                           stalled=stalled, persistence=persistence)


def reward(env, asset_cfg, sensor_cfg):
    """Read true contact-point kinematics once at each reward boundary."""
    from isaaclab.utils.math import quat_apply
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    from .rollclimb import contact_quality

    step = int(env.common_step_counter)
    old = getattr(env, '_d2w_traction_state', None)
    if old is not None and old['step'] == step:
        return old['value']
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    ids, sensor_ids = asset_cfg.body_ids, sensor_cfg.body_ids
    command = env.command_manager.get_command('base_velocity')
    episode = env.episode_length_buf.clone()
    points = sensor.data.contact_pos_w.torch[:, sensor_ids]
    forces = sensor.data.normal_force_matrix_w.torch[:, sensor_ids]
    gravity = command.new_tensor(env.cfg.sim.gravity)
    up = -gravity / gravity.norm().clamp_min(1.e-8)
    reference = robot.data.body_mass.torch.sum(-1) * gravity.norm() / 4.
    active, _, slip = contact_quality(
        robot.data.body_pos_w.torch[:, ids], robot.data.body_link_lin_vel_w.torch[:, ids],
        robot.data.body_link_ang_vel_w.torch[:, ids], points, forces, up)
    load, bearing, surface, wall = ground_support(points, forces, up, reference)
    quat = robot.data.body_quat_w.torch[:, ids]
    offset = command.new_zeros(env.num_envs, 4, 3)
    offset[:, :, 1] = command.new_tensor([1., -1., 1., -1.]) * .0455
    center = robot.data.body_pos_w.torch[:, ids] + quat_apply(quat, offset)
    direction = command.new_zeros(env.num_envs, 3)
    direction[:, 0] = 1.
    direction = quat_apply(robot.data.root_quat_w.torch, direction)
    direction[:, 2] = 0.
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    obstacle, ahead, distance = higher_tread_ahead(
        center, surface, env.scene.sensors['height_scanner'].data.ray_hits_w.torch,
        direction, bearing)
    masks = get_terrain_masks(env)
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    stairs = masks.get('stairs_up', zero) | masks.get('stairs_down', zero)
    gate = command_scope(command, -robot.data.projected_gravity_b.torch[:, 2],
                         stairs, env.termination_manager.terminated)
    valid = torch.zeros_like(episode, dtype=torch.bool)
    if old is not None:
        valid = (episode == old['episode'] + 1) & (episode > 1) & (step == old['step'] + 1)
        valid &= (command - old['command']).abs().amax(-1) < 1.e-5
        valid &= (direction * old['direction']).sum(-1) > .866
    value, age, diagnostics = restep_cost(
        command, robot.data.root_lin_vel_b.torch[:, 0], obstacle, active, slip,
        wall, load, reference, bearing, gate, valid,
        None if old is None else old['age'], env.step_dt)
    # Shared current geometry for optional task tracking refinements. This is
    # reward bookkeeping only and deliberately calls no other stateful reward.
    unresolved = obstacle.any(-1) & (bearing.sum(-1) >= 2)
    env._d2w_traction_state = dict(
        step=step, episode=episode, command=command.detach().clone(),
        direction=direction.detach().clone(), age=age.detach().clone(),
        value=value, gate=gate.detach(), obstacle=obstacle.detach(),
        scope=(gate[:, None] * obstacle).detach(),
        unresolved_ahead=unresolved.detach(),
        slip=slip.detach(), wall=wall.detach(), load=load.detach(),
        ahead=ahead.detach(), distance=distance.detach(), bearing=bearing.detach(),
        **{k: v.detach() for k, v in diagnostics.items()})
    return value


def configure(cfg):
    """Replace overlapping approach costs while retaining supported placements."""
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

    for name in ('stair_blockage', 'stair_unload', 'stair_swing_clearance', 'stair_pair_lag'):
        setattr(cfg.rewards, name, None)
    wheels = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    cfg.rewards.stair_restep = RewardTermCfg(func=reward, weight=-.12, params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names=wheels, preserve_order=True),
        sensor_cfg=SceneEntityCfg('stair_contacts', body_names=wheels, preserve_order=True)))
    return cfg
