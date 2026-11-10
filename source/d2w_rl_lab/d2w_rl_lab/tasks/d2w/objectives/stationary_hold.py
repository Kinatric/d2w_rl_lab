"""Native D2W zero-command hold: independent velocity costs, no anchor chasing."""
import torch
from .stand_balance import standing_balance, mirror_pairs
from .consistent_gate import designated_flat
from .standing_finetune_v1 import smooth


def motion_cost(velocity_xy, yaw_rate):
    # Each error contributes at most one: large braking errors cannot create
    # an arbitrarily large per-step debt compared with a termination event.
    xy=(velocity_xy/.05).square().sum(-1)
    yaw=(yaw_rate/.15).square()
    return xy/(1+xy)+yaw/(1+yaw)


def braking_blend(age):
    # Two seconds is the existing evaluation settling interval. Smoothly
    # introduce the hold cost while normal zero-velocity tracking brakes.
    return smooth(age,0.,2.)


def zero_mask(command):
    # Do not oppose commanded low-speed driving, turning, or stair traversal.
    return command.abs().amax(-1)<1.e-6


def hold_quality(env,asset_cfg,wheels_cfg,sensor_cfg,component='quality'):
    value=standing_balance(env,asset_cfg,wheels_cfg,sensor_cfg,'quality')
    return value*zero_mask(env.command_manager.get_command('base_velocity'))


def hold_residual(env,asset_cfg,wheels_cfg,sensor_cfg,component='residual'):
    r=env.scene[asset_cfg.name]
    c=env.command_manager.get_command('base_velocity')
    h=env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2]
    valid=torch.isfinite(h).all(1)
    safe=torch.where(torch.isfinite(h),h,torch.zeros_like(h))
    gate=designated_flat(env)*valid*(1-smooth(safe.amax(1)-safe.amin(1),.04,.08))*zero_mask(c)
    d=r.data.joint_pos.torch[:,asset_cfg.joint_ids]-r.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    pairs=mirror_pairs(d)/.1**2
    residual=motion_cost(r.data.root_lin_vel_b.torch[:,:2],r.data.root_ang_vel_b.torch[:,2])+(pairs/(1+pairs)).mean(1)
    age=env.command_manager.get_term('base_velocity').quiet_age
    return gate*braking_blend(age)*residual


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    params=dict(cfg.rewards.standing_quality.params)
    cfg.rewards.standing_quality=RewardTermCfg(func=hold_quality,weight=2.,params=dict(params,component='quality'))
    cfg.rewards.standing_balance_residual=RewardTermCfg(func=hold_residual,weight=-.5,params=dict(params,component='residual'))
    # Keep anchor bookkeeping/metrics, but do not demand return to an old world
    # pose which the actor cannot observe; current speed/yaw targets stay zero.
    cfg.rewards.standing_parking_position=None
    cfg.rewards.standing_parking_heading=None
    return cfg
