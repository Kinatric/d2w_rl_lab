import torch
from .standing_finetune_v1 import components, METRICS, self_test as base_self_test
from .parking_reference import self_test as parking_self_test

WEIGHTS={'tilt':-24.,'mirror':-8.,'drift':-1.,'parking_position':-2.,'parking_heading':-1.}

def standing_cost(env,asset_cfg,component):
    robot=env.scene[asset_cfg.name]
    term=env.command_manager.get_term('base_velocity')
    heights=env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2]
    deviation=robot.data.joint_pos.torch[:,asset_cfg.joint_ids]-robot.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    gate,tilt,mirror,drift=components(term.command,heights,robot.data.projected_gravity_b.torch,
        robot.data.root_lin_vel_b.torch,robot.data.root_ang_vel_b.torch,deviation)
    displacement_sq,heading=term.parking.errors(robot.data.root_pos_w.torch,robot.data.root_quat_w.torch)
    parking_gate=gate*term.parking.active
    if component=='tilt':
        denom=gate.sum().clamp_min(1);pd=parking_gate.sum().clamp_min(1)
        METRICS.update({
            'quiet_flat_fraction':gate.mean().detach(),
            'zero_command_fraction':(term.command.abs().amax(dim=1)<=.02).float().mean().detach(),
            'tilt_deg':(torch.acos((-robot.data.projected_gravity_b.torch[:,2]).clamp(-1,1))*180/torch.pi*gate).sum().detach()/denom,
            'mirror_rms_rad':torch.sqrt((mirror*gate).sum()/denom).detach(),
            'drift_speed_m_s':(robot.data.root_lin_vel_b.torch[:,:2].norm(dim=1)*gate).sum().detach()/denom,
            'yaw_speed_rad_s':(robot.data.root_ang_vel_b.torch[:,2].abs()*gate).sum().detach()/denom,
            'parking_displacement_rms_m':torch.sqrt((displacement_sq*parking_gate).sum()/pd).detach(),
            'parking_heading_abs_deg':(heading.abs()*180/torch.pi*parking_gate).sum().detach()/pd,
        })
    if component=='parking_position':return parking_gate*displacement_sq/.1**2
    if component=='parking_heading':return parking_gate*heading.square()/(10*torch.pi/180)**2
    return gate*{'tilt':tilt,'mirror':mirror,'drift':drift}[component]

def configure(cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from d2w_rl_lab.assets.d2w import D2W_LEG_JOINT_ORDER
    from .standing_sampling import StandingFocusCommand
    cfg.commands.base_velocity.class_type=StandingFocusCommand
    sub=cfg.scene.terrain.terrain_generator.sub_terrains
    rest=sum(v.proportion for k,v in sub.items() if k!='flat')
    for k,v in sub.items():v.proportion=.7 if k=='flat' else .3*v.proportion/rest
    for name,weight in WEIGHTS.items():
        setattr(cfg.rewards,'standing_'+name,RewardTermCfg(func=standing_cost,weight=weight,
            params={'asset_cfg':SceneEntityCfg('robot',joint_names=list(D2W_LEG_JOINT_ORDER),preserve_order=True),'component':name}))
    return cfg

def self_test():
    return {'status':'PASS','base':base_self_test(),'parking':parking_self_test(),'learning_updates':0}

def runtime_preflight(env):
    from .parking_reference import yaw_of
    term=env.command_manager.get_term('base_velocity');robot=env.scene['robot']
    quiet=term.command.abs().amax(dim=1)<=.02
    fraction=quiet.float().mean().item()
    flat=(term.terrain_idxs==term.terrain_type2idx['flat']).float().mean().item()
    assert fraction>.75 and .65<flat<.75,(fraction,flat)
    assert torch.equal(quiet,term.parking.active)
    assert torch.allclose(term.parking.xy[quiet],robot.data.root_pos_w.torch[quiet,:2])
    assert torch.allclose(term.parking.yaw[quiet],yaw_of(robot.data.root_quat_w.torch[quiet]))
    return {'status':'PASS','zero_command_fraction':fraction,'flat_terrain_fraction':flat,
        'parking_anchors_match_reset_state':True,'weights':WEIGHTS,'learning_updates_at_check':0,
        'scope':'Real reset/configuration checks only, no separate physical evaluation.'}
