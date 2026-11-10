"""Bound soft D2W preferences before gating; preserve physical safety costs."""
import torch
from .consistent_gate import designated_flat
from .standing_finetune_v1 import smooth
from .stationary_hold import braking_blend, zero_mask
from . import stance_math, motion_math, reward_coherent
from ..mdp_base import locomotion


def bounded(value):
    return value/(1+value)


def gated_bound(value,gate):
    # Do not saturate an already-gated cost: that would defeat a small gate.
    raw=torch.where(gate>0,value/gate.clamp_min(1.e-12),torch.zeros_like(value))
    return gate*bounded(raw)


def quiet_tracking_cost(command,velocity,gyro,leg_velocity):
    v=velocity.clone();g=gyro.clone()
    v[:,:2]-=command[:,:2];g[:,2]-=command[:,2]
    return locomotion.quiet_cost(command,v,g,leg_velocity)


def quiet_tracking(env,command_name,asset_cfg):
    r=env.scene[asset_cfg.name];c=env.command_manager.get_command(command_name)
    return quiet_tracking_cost(c,r.data.root_lin_vel_b.torch,r.data.root_ang_vel_b.torch,r.data.joint_vel.torch[:,asset_cfg.joint_ids])


def stance_bounded(env,asset_cfg,component,reference,target_height):
    r=env.scene[asset_cfg.name];c=env.command_manager.get_command('base_velocity')
    out=stance_math.costs(c,env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2],r.data.root_pos_w.torch[:,2],r.data.joint_pos.torch[:,asset_cfg.joint_ids],c.new_tensor(reference),target_height)
    return designated_flat(env)*gated_bound(out[component],out['gate'])


def straight_bounded(env,asset_cfg,component):
    r=env.scene[asset_cfg.name];c=env.command_manager.get_command('base_velocity')
    d=r.data.joint_pos.torch[:,asset_cfg.joint_ids]-r.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    v=r.data.root_lin_vel_b.torch.clone();g=r.data.root_ang_vel_b.torch.clone()
    v[:,1]-=c[:,1];g[:,2]-=c[:,2]
    gate,_,mirror,_,tilt,lateral=motion_math.costs(c,env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2],d,r.data.projected_gravity_b.torch,v,g)
    raw={'mirror':mirror,'tilt':tilt,'lateral':lateral}[component]
    return designated_flat(env)*gate*(tilt if component=='tilt' else bounded(raw))


def fast_bounded(env,command_name,sensor_cfg,height_cfg,settings,asset_cfg,component):
    from ..mdp import _locomotion_inputs
    _,_,contact,weights=_locomotion_inputs(env,command_name,sensor_cfg,height_cfg,settings)
    r=env.scene[asset_cfg.name]
    if component=='stability':
        raw=locomotion.stability_cost(r.data.projected_gravity_b.torch,r.data.root_ang_vel_b.torch,r.data.root_lin_vel_b.torch[:,2],settings);gate=weights[3]
    else:
        dev=r.data.joint_pos.torch[:,asset_cfg.joint_ids]-r.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
        raw=locomotion.straight_coordination_cost(dev,contact,settings);gate=weights[2]
    return gate*bounded(raw)


def parking_growth_blended(env,asset_cfg,component):
    # Restore the ACTUAL old bounded outward-growth implementation.
    # Stationary away from anchor: zero. Returning toward anchor: zero, no bonus.
    c=env.command_manager.get_command('base_velocity')
    age=env.command_manager.get_term('base_velocity').quiet_age
    return zero_mask(c)*braking_blend(age)*reward_coherent.parking_growth(env,asset_cfg,component)


def configure(cfg):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    for name in ['stance_hip','stance_pose','stance_height']:
        getattr(cfg.rewards,name).func=stance_bounded
    for name in ['straight_mirror','straight_lateral']:
        getattr(cfg.rewards,name).func=straight_bounded
    cfg.rewards.quiet_stability.func=quiet_tracking
    for name,component in [('fast_motion_stability','stability'),('fast_straight_coordination','coordination')]:
        term=getattr(cfg.rewards,name);term.func=fast_bounded
        term.params['component']=component
        term.params.setdefault('asset_cfg',SceneEntityCfg('robot'))
    asset=cfg.rewards.standing_quality.params['asset_cfg']
    for name,weight in [('parking_position',-.5),('parking_heading',-.25)]:
        setattr(cfg.rewards,'standing_'+name,RewardTermCfg(func=parking_growth_blended,weight=weight,params={'asset_cfg':asset,'component':name}))
    return cfg
