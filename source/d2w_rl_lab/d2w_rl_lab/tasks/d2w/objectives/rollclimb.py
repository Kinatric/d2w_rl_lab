"""D2W stair progress and ground-filtered contact quality, independent of gait choice."""
import math
import torch


def smooth(value, low, high):
    t = ((value - low) / (high - low)).clamp(0., 1.)
    return t.square() * (3. - 2. * t)


def bound_square(value):
    x = value.square()
    return x / (1. + x)


def contact_kinematics(body_pos, linear, angular, points, forces):
    """Force-weighted mean contact point and resultant normal, not per-contact slip."""
    finite = torch.isfinite(points).all(-1) & torch.isfinite(forces).all(-1)
    weights = forces.norm(dim=-1) * finite
    force = torch.where(finite[..., None], forces, 0.).sum(-2)
    point = (torch.nan_to_num(points) * weights[..., None]).sum(-2) / weights.sum(-1).clamp_min(1.e-8)[..., None]
    active = (force.norm(dim=-1) > 1.) & (weights.sum(-1) > 1.)
    point = torch.where(active[..., None], point, body_pos)
    normal = force / force.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    velocity = linear + torch.cross(angular, point - body_pos, dim=-1)
    tangent = velocity - (velocity * normal).sum(-1, keepdim=True) * normal
    return active, tangent, point, normal


def contact_quality(body_pos, linear, angular, points, forces, up):
    """Evaluate each ground-shape entry before reduction; entries may aggregate contacts."""
    finite = torch.isfinite(points).all(-1) & torch.isfinite(forces).all(-1)
    force = torch.where(finite[..., None], forces, 0.)
    magnitude = force.norm(dim=-1)
    valid_contact = finite & (magnitude > 1.)
    point = torch.where(valid_contact[..., None], points, body_pos[..., None, :])
    normal = force / magnitude[..., None].clamp_min(1.e-8)
    velocity = linear[..., None, :] + torch.cross(
        angular[..., None, :].expand_as(point), point-body_pos[..., None, :], dim=-1)
    tangent = velocity-(velocity*normal).sum(-1, keepdim=True)*normal
    weight = magnitude*valid_contact
    per_wheel = (bound_square(tangent.norm(dim=-1)/.25)*weight).sum(-1)/weight.sum(-1).clamp_min(1.e-8)
    active = valid_contact.any(-1)
    # A vertical riser can provide traction, but its normal force alone is not weight support.
    bearing = active & ((force.sum(-2)*up).sum(-1) > 1.)
    return active, bearing, per_wheel


def active_slip_cost(wheel_slip, active):
    """Average ground-contact slip without dilution from wheels in free swing."""
    value = torch.where(active, wheel_slip, 0.)
    return value.sum(-1) / active.sum(-1).clamp_min(1)


def progress_state(position, command, world_command, valid, previous, dt):
    """New path progress; a straight command retains its initial world direction.

    Turning/lateral commands keep the original instantaneous direction. This
    prevents unintended yaw from earning circular-path progress under a straight
    command. This retains a historical direction in the reward; the actor has
    no new heading observation and no explicit heading-error penalty is added.
    """
    direction = world_command / world_command.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    straight = (command[:, 0].abs() > 1.e-6) & (command[:, 1:].abs().amax(-1) <= 1.e-6)
    if previous is None:
        changed = torch.ones_like(valid)
        previous_position = position
        coordinate = torch.zeros_like(position[:, 0])
        highwater = torch.zeros_like(coordinate)
    else:
        changed = (~valid) | ((command - previous['command']).abs().amax(-1) > 1.e-5)
        previous_position = torch.where(changed[:, None], position, previous['position'])
        coordinate = torch.where(changed, 0., previous['coordinate'])
        highwater = torch.where(changed, 0., previous['highwater'])
    anchor = direction if previous is None else torch.where(
        changed[:, None], direction, previous['anchor'])
    projection = torch.where(straight[:, None], anchor, direction)
    # A fixed straight-command axis makes lateral arcs and circles stop paying.
    # Backtracking cannot repay the same frontier. Reset/recommand starts afresh.
    coordinate = coordinate + ((position - previous_position) * projection).sum(-1)
    delta = (coordinate - highwater).clamp_min(0.)
    allowance = world_command.norm(dim=-1) * dt
    score = torch.minimum(delta, allowance) / allowance.clamp_min(1.e-8)
    score = torch.where(valid & (~changed) & (allowance > 1.e-8), score, 0.)
    state = dict(command=command.clone(), position=position.clone(), coordinate=coordinate.clone(),
                 highwater=torch.maximum(highwater, coordinate).clone(), anchor=anchor.clone())
    return score, state


def continuous_progress(value):
    """Bounded concave progress: prefer sustained advancement at equal total gain.

    Finite slope at zero; no standstill, reverse, or revisited-frontier payout.
    This is a reward preference, not a guarantee of continuous physical motion.
    """
    return ((.01 + .99 * value.clamp(0., 1.)).sqrt() - .1) / .9


def force_rise_cost(history, reference_load, physics_dt):
    """Penalize abrupt load acquisition; steady support and load release cost zero."""
    # IsaacLab histories are newest-first; include the previous control boundary.
    magnitude = history.norm(dim=-1)
    rises = (magnitude[:, :-1] - magnitude[:, 1:]).clamp_min(0.)
    scale = reference_load[:, None, None].clamp_min(1.e-6) * physics_dt / .02
    return bound_square(rises / scale).amax(1).mean(1)


def stair_terms(env, asset_cfg, sensor_cfg):
    from isaaclab.utils.math import quat_apply
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    from .standing_finetune import METRICS

    step = int(env.common_step_counter)
    previous = getattr(env, '_d2w_rollclimb_state', None)
    if previous is not None and previous['step'] == step:
        return previous['values']
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    ids, sensor_ids = asset_cfg.body_ids, sensor_cfg.body_ids
    points = data.contact_pos_w.torch[:, sensor_ids]
    forces = data.normal_force_matrix_w.torch[:, sensor_ids]
    gravity_vector = torch.tensor(env.cfg.sim.gravity, device=env.device)
    up = -gravity_vector/gravity_vector.norm().clamp_min(1.e-8)
    active, bearing, wheel_slip = contact_quality(
        robot.data.body_pos_w.torch[:, ids], robot.data.body_link_lin_vel_w.torch[:, ids],
        robot.data.body_link_ang_vel_w.torch[:, ids], points, forces, up)
    command = env.command_manager.get_command('base_velocity')
    world_command = quat_apply(robot.data.root_quat_w.torch,
                               torch.cat([command[:, :2], torch.zeros_like(command[:, :1])], -1))[:, :2]
    episode = env.episode_length_buf.clone()
    valid = torch.zeros_like(episode, dtype=torch.bool)
    if previous is not None:
        valid = (episode == previous['episode'] + 1) & (episode > 1) & (step == previous['step'] + 1)
    position = robot.data.root_pos_w.torch[:, :2]
    progress, progress_memory = progress_state(position, command, world_command, valid,
                                               None if previous is None else previous['progress'], env.step_dt)
    masks = get_terrain_masks(env)
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    stairs = masks.get('stairs_up', zero) | masks.get('stairs_down', zero)
    moving = smooth(command[:, :2].norm(dim=-1), .05, .20)
    gate = stairs * moving * valid
    upright = -robot.data.projected_gravity_b.torch[:, 2]
    # Fade over the existing recoverable interval; original 1 rad failure stays intact.
    healthy = smooth(upright, math.cos(1.), .8) * (~env.termination_manager.terminated)
    supported = bearing.sum(-1) >= 2
    slip = active_slip_cost(wheel_slip, active)
    gyro = robot.data.root_ang_vel_w.torch
    shock = torch.zeros_like(progress)
    if previous is not None:
        acceleration = (gyro[:, :2] - previous['gyro'][:, :2]) / env.step_dt
        x = acceleration.square().mean(-1) / 25.**2
        shock = x / (1. + x)
    # All counterparts are ground shapes: self/robot contacts cannot count as support.
    history = data.normal_force_matrix_w_history.torch[:, :, sensor_ids].sum(-2)
    gravity = torch.tensor(env.cfg.sim.gravity, device=env.device).norm()
    reference_load = robot.data.body_mass.torch.sum(-1) * gravity / 4.
    impact = force_rise_cost(history, reference_load, env.physics_dt)
    values = dict(progress=gate * supported * healthy * continuous_progress(progress), slip=gate * slip,
                  shock=gate * shock, impact=gate * impact)
    env._d2w_rollclimb_state = dict(step=step, episode=episode, progress=progress_memory,
                                  gyro=gyro.clone(), values=values, gate=gate.clone(), bearing=bearing.clone(),
                                  active_contacts=active.clone(), wheel_slip=wheel_slip.clone(),
                                  linear_progress=progress.clone(), progress_multiplier=(gate*supported*healthy).clone())
    den = gate.sum().clamp_min(1.)
    METRICS.update({f'rollclimb_{k}':v.sum().detach()/den for k, v in values.items()})
    METRICS.update(rollclimb_support_fraction=(gate*supported).sum().detach()/den,
                   rollclimb_contact_fraction=(gate[:, None]*active).sum().detach()/den/4.)
    return values


def reward(env, asset_cfg, sensor_cfg, component):
    return stair_terms(env, asset_cfg, sensor_cfg)[component]


def configure(cfg):
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    wheels = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    cfg.scene.stair_contacts = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/.*_foot',
        filter_shape_prim_expr=[cfg.scene.terrain.prim_path + '/.*'],
        track_contact_points=True, history_length=cfg.decimation + 1,
        update_period=cfg.sim.dt, debug_vis=False)
    params = dict(asset_cfg=SceneEntityCfg('robot', body_names=wheels, preserve_order=True),
                  sensor_cfg=SceneEntityCfg('stair_contacts', body_names=wheels, preserve_order=True))
    cfg.rewards.stair_touchdown = None
    for name, component, weight in [('stair_progress', 'progress', .20), ('stair_slip', 'slip', -.15),
                                    ('stair_shock', 'shock', -.10), ('stair_contact_impact', 'impact', -.05)]:
        setattr(cfg.rewards, name, RewardTermCfg(func=reward, weight=weight,
                                               params=dict(params, component=component)))
    return cfg
