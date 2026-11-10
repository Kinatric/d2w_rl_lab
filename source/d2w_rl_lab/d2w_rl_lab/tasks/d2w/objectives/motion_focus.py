from .consistent_gate import designated_flat
import torch
from . import standing_finetune as parent
from .standing_finetune import METRICS
from .motion_math import costs,self_test as motion_test

WEIGHTS={'mirror':-8.,'pose':-2.,'tilt':-8.,'lateral':-1.}
def motion_cost(env,asset_cfg,component):
    robot=env.scene[asset_cfg.name];c=env.command_manager.get_command('base_velocity')
    d=robot.data.joint_pos.torch[:,asset_cfg.joint_ids]-robot.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    gate,pg,mirror,pose,tilt,lateral=costs(c,env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2],d,
        robot.data.projected_gravity_b.torch,robot.data.root_lin_vel_b.torch,robot.data.root_ang_vel_b.torch)
    if component=='mirror':
        denom=gate.sum().clamp_min(1)
        METRICS.update({'straight_flat_fraction':gate.mean().detach(),
            'straight_mirror_rms_rad':torch.sqrt((gate*mirror).sum()/denom).detach(),
            'straight_pose_excess':(gate*pose).sum().detach()/denom,
            'straight_tilt_deg':(torch.acos((-robot.data.projected_gravity_b.torch[:,2]).clamp(-1,1))*180/torch.pi*gate).sum().detach()/denom,
            'straight_vx_error_m_s':((robot.data.root_lin_vel_b.torch[:,0]-c[:,0]).abs()*gate).sum().detach()/denom})
    if component=='pose':return pg*pose
    return designated_flat(env)*gate*{'mirror':mirror,'tilt':tilt,'lateral':lateral}[component]
def configure(cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from d2w_rl_lab.assets.d2w import D2W_LEG_JOINT_ORDER
    cfg=parent.configure(cfg)
    for name,weight in WEIGHTS.items():
        setattr(cfg.rewards,'straight_'+name,RewardTermCfg(func=motion_cost,weight=weight,
            params={'asset_cfg':SceneEntityCfg('robot',joint_names=list(D2W_LEG_JOINT_ORDER),preserve_order=True),'component':name}))
    return cfg
def self_test():return {'status':'PASS','standing':parent.self_test(),'motion':motion_test(),'learning_updates':0}
def runtime_preflight(env):
    from .parking_reference import yaw_of
    term=env.command_manager.get_term('base_velocity');c=term.command;robot=env.scene['robot']
    quiet=c.abs().amax(1)<=.02
    straight=(c[:,0].abs()>=.2)&(c[:,1:].abs().amax(1)<1e-6)
    f=quiet.float().mean().item();sf=straight.float().mean().item()
    assert .3<f<.55 and .35<sf<.65,(f,sf)
    assert torch.equal(quiet,term.parking.active)
    assert torch.allclose(term.parking.xy[quiet],robot.data.root_pos_w.torch[quiet,:2])
    assert torch.allclose(term.parking.yaw[quiet],yaw_of(robot.data.root_quat_w.torch[quiet]))
    for key in WEIGHTS:assert 'straight_'+key in env.reward_manager.active_terms
    return {'status':'PASS','zero_command_fraction':f,'straight_command_fraction':sf,
        'flat_fraction':(term.terrain_idxs==term.terrain_type2idx['flat']).float().mean().item(),
        'parking_references_valid':True,'new_reward_weights':WEIGHTS,'learning_updates_at_check':0}
