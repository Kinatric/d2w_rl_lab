"""Conservative load relief during blocked lateral motion; observe all terrain.

Does not alter PD, envelope, observations, commands, or existing stair rewards.
Current/proper temperature measurements and compliant-grass dynamics are absent.
"""
import torch
from .lateral_load_math import lateral_scope, relief, support_margins


def reward(env, asset_cfg, sensor_cfg):
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    from isaaclab.utils.math import quat_apply_inverse
    from .stair_traction import ground_support

    step = int(env.common_step_counter)
    old = getattr(env, '_d2w_lateral_load_state', None)
    if old is not None and old['step'] == step:
        return old['value']
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    observer = robot.actuators['RS06'].calf_load
    ratio_sq, interval_s = observer.consume_interval()
    ema = observer.values['ratio_sq_ema'].clone()
    command = env.command_manager.get_command('base_velocity')
    episode = env.episode_length_buf.clone()
    points = sensor.data.contact_pos_w.torch[:, sensor_cfg.body_ids]
    forces = sensor.data.normal_force_matrix_w.torch[:, sensor_cfg.body_ids]
    gravity = command.new_tensor(env.cfg.sim.gravity)
    up = -gravity/gravity.norm().clamp_min(1.e-8)
    mass = robot.data.body_mass.torch
    reference = mass.sum(-1)*gravity.norm()/4.
    load, bearing, _, _ = ground_support(points, forces, up, reference)
    finite = torch.isfinite(points).all(-1) & torch.isfinite(forces).all(-1)
    safe_force = torch.where(finite[..., None], forces, 0.)
    vertical = (safe_force*up).sum(-1)
    support = finite & (vertical > .5*safe_force.norm(dim=-1)) & (vertical > 1.)
    weights = torch.where(support, vertical, 0.)
    contact = (torch.nan_to_num(points)*weights[..., None]).sum(-2)/weights.sum(-1).clamp_min(1.e-8)[..., None]
    com = (robot.data.body_com_pos_w.torch*mass[..., None]).sum(1)/mass.sum(-1, keepdim=True)
    representative=torch.where(bearing[...,None],contact,robot.data.body_pos_w.torch[:,asset_cfg.body_ids])
    offset=representative-robot.data.root_pos_w.torch[:,None]
    rotation=robot.data.root_quat_w.torch[:,None,:].expand(-1,4,-1)
    wheel_xy=quat_apply_inverse(rotation.reshape(-1,4),offset.reshape(-1,3)).reshape(-1,4,3)[...,:2]
    actual_twist=torch.cat((robot.data.root_lin_vel_b.torch[:,:2],robot.data.root_ang_vel_b.torch[:,2:3]),-1)
    # Current tasks have vertical gravity; fail closed if this contract changes.
    vertical_gravity = (up[:2].abs().amax() < 1.e-6) & (up[2] > .999)
    margins = support_margins(contact[..., :2], com[:, :2])
    masks = get_terrain_masks(env)
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    stairs = masks.get('stairs_up', zero) | masks.get('stairs_down', zero)
    gate = lateral_scope(command, -robot.data.projected_gravity_b.torch[:, 2],
        robot.data.root_ang_vel_b.torch[:, :2], stairs, env.termination_manager.terminated)
    valid = torch.zeros_like(episode, dtype=torch.bool)
    if old is not None:
        valid = (episode == old['episode']+1) & (episode > 1) & (step == old['step']+1)
        valid &= (command-old['command']).abs().amax(-1) < 1.e-5
    valid &= vertical_gravity & torch.isfinite(ratio_sq).all(-1) & torch.isfinite(ema).all(-1)
    valid &= (interval_s >= env.step_dt*.99).all(-1)
    value, memory, diag = relief(command, actual_twist, wheel_xy,
        gate, load, reference, bearing, margins, ema, ratio_sq,
        observer.values['filtered_above_s'], valid,
        None if old is None else old['memory'], env.step_dt)
    # Aggregate physical samples across resets until the training logger reads
    # and clears them. This avoids treating a single control-boundary torque as
    # an RMS statistic for an entire rollout.
    sample = dict(torque_sq_s=(ratio_sq*observer.reference_nm**2*interval_s).sum(0),
        duration_s=interval_s.sum(0), power_j=observer.last_interval_power_j.sum(0),
        above_s=observer.last_interval_above_s.sum(0),
        envsteps=value.new_tensor(float(env.num_envs)), scope=(gate>0).sum().to(value.dtype),
        active=(value>0).sum().to(value.dtype), cost=value.sum(),
        load_active=(diag['load_cost']>0).sum().to(value.dtype),
        unload_active=(diag['unload_cost']>0).sum().to(value.dtype),
        load_cost=diag['load_cost'].sum(), unload_cost=diag['unload_cost'].sum(),
        blocked=diag['blocked'].sum().to(value.dtype),
        safe=(diag['can_unload'] & (gate[:,None]>0)).sum().to(value.dtype))
    totals = getattr(env, '_d2w_load_totals', None)
    if totals is None:
        totals = {k:torch.zeros_like(v) for k,v in sample.items()}
        env._d2w_load_totals = totals
    for k,v in sample.items():
        totals[k].add_(v.detach())
    env._d2w_lateral_load_state = dict(step=step, episode=episode,
        command=command.detach().clone(), value=value, memory=memory,
        diagnostics=diag, ratio_sq=ratio_sq.detach(), ema=ema.detach(),
        actual_twist=actual_twist.detach().clone(),wheel_xy=wheel_xy.detach().clone(),
        load_age=observer.values['filtered_above_s'].detach().clone(),
        interval_s=interval_s.detach(), load=load.detach(), bearing=bearing.detach(),
        contact=contact.detach(), com=com.detach(), valid=valid.detach())
    return value


def configure(cfg):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from .load_observer import LoadObservedRobStrideActuator
    cfg.scene.robot.actuators['RS06'].class_type = LoadObservedRobStrideActuator
    wheels = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    cfg.rewards.lateral_load_relief = RewardTermCfg(func=reward, weight=-.12, params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names=wheels, preserve_order=True),
        sensor_cfg=SceneEntityCfg('stair_contacts', body_names=wheels, preserve_order=True)))
    return cfg
