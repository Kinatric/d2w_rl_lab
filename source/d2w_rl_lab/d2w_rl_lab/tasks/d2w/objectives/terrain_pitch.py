"""Restore D2W terrain/command distribution; local flat-standing attitude cost."""
from copy import deepcopy
import math
import torch
from . import reward_coherent as parent
from .standing_finetune import METRICS
from .standing_finetune_v1 import smooth

EXPECTED={'wave':.05,'slope_up':.1,'slope_down':.1,'rough_slope':.05,'stairs_up':.25,'stairs_down':.1,'obstacles':.2,'stepping_stones':0.,'gap':0.,'flat':.15}

def attitude_cost(gravity):
    scale=math.sin(math.radians(10))**2
    x=gravity[:,:2].square()/scale
    return x[:,0]/(1+x[:,0])+.5*x[:,1]/(1+x[:,1])

def flat_stand_attitude(env):
    r=env.scene['robot'];c=env.command_manager.get_command('base_velocity')
    gate=parent.flat_gate(env)*(1-smooth(c.abs().amax(1),.02,.2))
    g=r.data.projected_gravity_b.torch
    den=gate.sum().clamp_min(1)
    pitch=torch.atan2(g[:,0],-g[:,2])*180/torch.pi
    roll=torch.atan2(-g[:,1],-g[:,2])*180/torch.pi
    METRICS.update({'stand_pitch_signed_deg':(pitch*gate).sum().detach()/den,
        'stand_pitch_abs_deg':(pitch.abs()*gate).sum().detach()/den,
        'stand_roll_abs_deg':(roll.abs()*gate).sum().detach()/den,
        'attitude_gate_fraction':gate.mean().detach()})
    return gate*attitude_cost(g)

def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    from d2w_rl_lab.tasks.d2w.mdp import D2WRLGymCommand
    from .parking_reference import ParkingReference
    # The subclass only records parking references; command values, durations,
    # terrain curricula and original zero-command sampling remain untouched.
    from .default_sampling import DefaultCommandWithParking
    default_terrain=deepcopy(cfg.scene.terrain.terrain_generator)
    default_command=deepcopy(cfg.commands.base_velocity)
    assert {k:v.proportion for k,v in default_terrain.sub_terrains.items()}==EXPECTED
    cfg=parent.configure(cfg)
    cfg.scene.terrain.terrain_generator=default_terrain
    cfg.commands.base_velocity=default_command
    cfg.commands.base_velocity.class_type=DefaultCommandWithParking
    cfg.rewards.flat_stand_attitude=RewardTermCfg(func=flat_stand_attitude,weight=-1.5)
    return cfg

def self_test():
    angles=torch.tensor([0.,5.,8.,10.,-8.])*torch.pi/180
    g=torch.stack([angles.sin(),torch.zeros_like(angles),-angles.cos()],1)
    cost=attitude_cost(g)
    assert cost[0]==0 and cost[0]<cost[1]<cost[2]<cost[3] and cost[2]==cost[4]
    assert torch.isfinite(cost).all() and (cost<=1.5).all()
    return {'status':'PASS','parent':parent.self_test(),'checks':['level_zero_cost','pitch_monotonic_0_10deg','pitch_sign_symmetry','bounded_attitude_cost'],'learning_updates':0}

def runtime_preflight(env):
    from d2w_rl_lab.tasks.d2w.env_cfg import D2WEnvCfg
    from d2w_rl_lab.assets.d2w_metadata import D2W_LEG_JOINT_ORDER
    from .parking_reference import yaw_of
    r=env.scene['robot'];t=env.command_manager.get_term('base_velocity')
    actual={k:v.proportion for k,v in env.cfg.scene.terrain.terrain_generator.sub_terrains.items()}
    assert actual==EXPECTED
    baseline=D2WEnvCfg().commands.base_velocity.to_dict();current=env.cfg.commands.base_velocity.to_dict()
    baseline.pop('class_type');current.pop('class_type');assert baseline==current
    p=env.reward_manager.get_term_cfg('standing_quality').params
    assert [r.joint_names[i] for i in p['asset_cfg'].joint_ids]==list(D2W_LEG_JOINT_ORDER)
    names=['FL_foot','FR_foot','RL_foot','RR_foot']
    assert [r.body_names[i] for i in p['wheels_cfg'].body_ids]==names
    sensor=env.scene.sensors['contact_forces'];assert [sensor.body_names[i] for i in p['sensor_cfg'].body_ids]==names
    quiet=t.commands.abs().amax(1)<=.02
    assert torch.equal(quiet,t.parking.active)
    assert torch.allclose(t.parking.xy[quiet],r.data.root_pos_w.torch[quiet,:2])
    assert torch.allclose(t.parking.yaw[quiet],yaw_of(r.data.root_quat_w.torch[quiet]))
    for name in ['standing_quality','standing_balance_residual','hip_pos_penalty_l1','joint_pos_penalty_l1','standing_parking_position','standing_parking_heading','flat_stand_attitude']:
        c=env.reward_manager.get_term_cfg(name);v=c.func(env,**c.params)
        assert v.shape==(env.num_envs,) and torch.isfinite(v).all()
    fractions={k:(t.terrain_idxs==i).float().mean().item() for k,i in t.terrain_type2idx.items()}
    assert abs(fractions['flat']-.15)<.01 and abs(fractions['stairs_up']-.25)<.01
    return {'status':'PASS','terrain_proportions':actual,'actual_environment_fractions':fractions,'default_command_config_equal_except_bookkeeping_subclass':True,'zero_command_reset_fraction':quiet.float().mean().item(),'mapping_and_finite_rewards':True,'learning_updates':0,'explicit_physics_steps':0}
