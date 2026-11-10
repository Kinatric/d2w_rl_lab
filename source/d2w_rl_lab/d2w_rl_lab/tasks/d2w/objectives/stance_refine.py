from .consistent_gate import designated_flat
import torch
from . import motion_focus as parent
from .standing_finetune import METRICS
from .stance_math import costs,self_test as refine_test

WEIGHTS={'hip':-1.,'pose':-2.,'rear':-1.,'height':-1.}

def stance_cost(env,asset_cfg,component,reference,target_height):
    r=env.scene[asset_cfg.name];c=env.command_manager.get_command('base_velocity')
    q=r.data.joint_pos.torch[:,asset_cfg.joint_ids]
    out=costs(c,env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2],r.data.root_pos_w.torch[:,2],q,q.new_tensor(reference),target_height)
    if component=='hip':
        gate=out['gate'];den=gate.sum().clamp_min(1);quiet=out['quiet_gate'];qd=quiet.sum().clamp_min(1)
        d=out['deviation']
        METRICS.update({'posture_gate_fraction':gate.mean().detach(),
            'posture_relative_height_m':(out['relative_height']*gate).sum().detach()/den,
            'posture_height_error_rms_m':torch.sqrt(((out['relative_height']-target_height).square()*gate).sum()/den).detach(),
            'posture_hip_rms_rad':torch.sqrt((d[:,:,0].square().mean(1)*gate).sum()/den).detach(),
            'standing_RL_pose_rms_rad':torch.sqrt((d[:,2].square().mean(1)*quiet).sum()/qd).detach(),
            'standing_RR_pose_rms_rad':torch.sqrt((d[:,3].square().mean(1)*quiet).sum()/qd).detach(),
            'standing_worst_leg_rms_rad':(d.square().mean(2).sqrt().amax(1)*quiet).sum().detach()/qd,
            'height_ray_valid_fraction':out['valid'].float().mean().detach()})
    return designated_flat(env)*out[component]

def configure(cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from d2w_rl_lab.assets.d2w import D2W_FIXSTAND_BASE_HEIGHT_M
    from d2w_rl_lab.assets.d2w_metadata import D2W_LEG_JOINT_ORDER,D2W_FIXSTAND_JOINT_POSITIONS
    cfg=parent.configure(cfg)
    # Replace old default-pose objective, retaining action/observation default offsets.
    cfg.rewards.straight_pose=None
    reference=[D2W_FIXSTAND_JOINT_POSITIONS[n] for n in D2W_LEG_JOINT_ORDER]
    for name,weight in WEIGHTS.items():
        setattr(cfg.rewards,'stance_'+name,RewardTermCfg(func=stance_cost,weight=weight,params={
            'asset_cfg':SceneEntityCfg('robot',joint_names=list(D2W_LEG_JOINT_ORDER),preserve_order=True),
            'component':name,'reference':reference,'target_height':D2W_FIXSTAND_BASE_HEIGHT_M}))
    return cfg

def self_test():return {'status':'PASS','parent':parent.self_test(),'refine':refine_test(),'learning_updates':0}

def runtime_preflight(env):
    from .parking_reference import yaw_of
    from d2w_rl_lab.assets.d2w_metadata import D2W_LEG_JOINT_ORDER,D2W_FIXSTAND_JOINT_POSITIONS
    r=env.scene['robot'];term=env.command_manager.get_term('base_velocity');c=term.command
    quiet=c.abs().amax(1)<=.02;straight=(c[:,0].abs()>=.2)&(c[:,1:].abs().amax(1)<1e-6)
    f=quiet.float().mean().item();s=straight.float().mean().item();assert .40<f<.65 and .35<s<.60,(f,s)
    assert torch.equal(quiet,term.parking.active)
    assert torch.allclose(term.parking.xy[quiet],r.data.root_pos_w.torch[quiet,:2])
    assert torch.allclose(term.parking.yaw[quiet],yaw_of(r.data.root_quat_w.torch[quiet]))
    assert 'straight_pose' not in env.reward_manager.active_terms
    details={}
    for name,w in WEIGHTS.items():
        cfg=env.reward_manager.get_term_cfg('stance_'+name);p=cfg.params
        assert cfg.weight==w
        names=[r.joint_names[i] for i in p['asset_cfg'].joint_ids]
        assert names==list(D2W_LEG_JOINT_ORDER),names
        assert p['reference']==[D2W_FIXSTAND_JOINT_POSITIONS[n] for n in names]
        value=cfg.func(env,**p);assert value.shape==(env.num_envs,) and torch.isfinite(value).all()
        details[name]={'weight':w,'reference':p['reference'],'target_height_m':p['target_height'],'joint_names':names}
    return {'status':'PASS','zero_command_fraction':f,'straight_command_fraction':s,
        'flat_fraction':(term.terrain_idxs==term.terrain_type2idx['flat']).float().mean().item(),
        'parking_references_valid':True,'rewards':details,'learning_updates_at_check':0,
        'scope':'Real reset, resolved joint names and finite reward calls; not a behavior evaluation'}
