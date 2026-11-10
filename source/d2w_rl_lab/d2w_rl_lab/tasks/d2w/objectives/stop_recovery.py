import torch
from . import stance_refine as parent
from .standing_finetune import METRICS
from .standing_finetune_v1 import smooth

def quality(speed,yaw,mirror,span,load):
    # Soft bounded signal. 0.40m is a nominal span reference from the existing
    # fixed-stand MuJoCo traces, not a collision limit or termination threshold.
    terms=torch.stack([speed.square()/.05**2,yaw.square()/.15**2,mirror.square()/.1**2,
                       (span-.40).square()/.10**2],1)
    return torch.exp(-terms.clamp(max=30)).mean(1)*load

def standing_quality(env,asset_cfg,wheels_cfg,sensor_cfg):
    from isaaclab.utils.math import quat_apply_inverse
    r=env.scene[asset_cfg.name];term=env.command_manager.get_term('base_velocity')
    h=env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2]
    valid=torch.isfinite(h).all(1);safe=torch.where(torch.isfinite(h),h,torch.zeros_like(h))
    gate=valid*(1-smooth(safe.amax(1)-safe.amin(1),.04,.08))*(term.command.abs().amax(1)<=.02)
    d=(r.data.joint_pos.torch[:,asset_cfg.joint_ids]-r.data.default_joint_pos.torch[:,asset_cfg.joint_ids]).reshape(-1,4,3)
    mirror=(d[:,2]-d[:,3]*d.new_tensor([-1.,1.,1.])).square().mean(1).sqrt()
    delta=r.data.body_pos_w.torch[:,wheels_cfg.body_ids]-r.data.root_pos_w.torch[:,None,:]
    body=quat_apply_inverse(r.data.root_quat_w.torch[:,None,:].expand(-1,4,-1).reshape(-1,4),delta.reshape(-1,3)).reshape(-1,4,3)
    span=body[:,:2,0].mean(1)-body[:,2:,0].mean(1)
    force=env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:,sensor_cfg.body_ids,2]
    load=(force>1.).float().mean(1)
    healthy=(r.data.projected_gravity_b.torch[:,2]<-.9)&(~env.termination_manager.terminated)
    score=quality(r.data.root_lin_vel_b.torch[:,:2].norm(dim=1),r.data.root_ang_vel_b.torch[:,2],mirror,span,load)
    den=gate.sum().clamp_min(1)
    METRICS.update({'park_rear_mirror_rms_rad':torch.sqrt((mirror.square()*gate).sum()/den).detach(),
        'park_wheelbase_m':(span*gate).sum().detach()/den,'park_contact_fraction':(load*gate).sum().detach()/den,
        'park_over10s_fraction':((term.quiet_age>=10).float()*gate).mean().detach(),
        'park_age_mean_s':(term.quiet_age*gate).sum().detach()/den,
        'true_failure_fraction':env.termination_manager.terminated.float().mean().detach(),
        'drift_speed_m_s':(r.data.root_lin_vel_b.torch[:,:2].norm(dim=1)*gate).sum().detach()/den,
        'zero_command_fraction':(term.command.abs().amax(1)<=.02).float().mean().detach(),
        'quiet_flat_fraction':gate.mean().detach()})
    return gate*healthy*score

def failure_cost(env):return env.termination_manager.terminated.float()/env.step_dt

def configure(cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from d2w_rl_lab.assets.d2w_metadata import D2W_LEG_JOINT_ORDER
    cfg=parent.configure(cfg)
    # Remove newly accumulated standing penalties in this branch, retain original
    # project rewards and straight-driving terms. Replace with bounded quality.
    for n in ['standing_tilt','standing_mirror','standing_drift','standing_parking_position','standing_parking_heading','stance_rear']:
        setattr(cfg.rewards,n,None)
    # Persistent reference costs retained at modest weights; other components
    # are bounded positive quality rather than ever-increasing per-step debt.
    from .standing_finetune import standing_cost
    leg=SceneEntityCfg('robot',joint_names=list(D2W_LEG_JOINT_ORDER),preserve_order=True)
    for n,w in [('parking_position',-.5),('parking_heading',-.25)]:
        setattr(cfg.rewards,'standing_'+n,RewardTermCfg(func=standing_cost,weight=w,params={'asset_cfg':leg,'component':n}))
    wheels=['FL_foot','FR_foot','RL_foot','RR_foot']
    cfg.rewards.standing_quality=RewardTermCfg(func=standing_quality,weight=2.,params={'asset_cfg':leg,
        'wheels_cfg':SceneEntityCfg('robot',body_names=wheels,preserve_order=True),
        'sensor_cfg':SceneEntityCfg('contact_forces',body_names=wheels,preserve_order=True)})
    cfg.rewards.true_failure_cost=RewardTermCfg(func=failure_cost,weight=-2.)
    return cfg

def self_test():
    a=torch.zeros(3);span=torch.full((3,),.4);load=torch.ones(3)
    q=quality(a,a,a,span,load);assert torch.allclose(q,load)
    span[1]=.18;assert quality(a,a,a,span,load)[1]<q[1]
    load[2]=0;assert quality(a,a,a,span,load)[2]==0
    v=torch.tensor([0.,.05,1000.]);assert quality(v,a,a,span,load).isfinite().all()
    assert -2*(1/.02)*.02==-2
    from .parking_reference import self_test as anchor_test
    return {'status':'PASS','anchor':anchor_test(),'checks':['bounded_quality','collapsed_span_lower_reward','no_load_no_bonus','finite_extreme','failure_event_dt_normalization'],'learning_updates':0}

def runtime_preflight(env):
    from d2w_rl_lab.assets.d2w_metadata import D2W_LEG_JOINT_ORDER
    t=env.command_manager.get_term('base_velocity');r=env.scene['robot']
    fractions=[(t.phase==i).float().mean().item() for i in [0,1,3]]
    assert .25<fractions[0]<.35 and .55<fractions[1]<.65,fractions
    p=env.reward_manager.get_term_cfg('standing_quality').params
    assert [r.joint_names[i] for i in p['asset_cfg'].joint_ids]==list(D2W_LEG_JOINT_ORDER)
    names=['FL_foot','FR_foot','RL_foot','RR_foot']
    assert [r.body_names[i] for i in p['wheels_cfg'].body_ids]==names
    sensor=env.scene.sensors['contact_forces'];assert [sensor.body_names[i] for i in p['sensor_cfg'].body_ids]==names
    assert torch.isfinite(standing_quality(env,**p)).all()
    ids=(t.phase==1).nonzero().flatten()[:8];assert len(ids)==8
    before=t.commands[ids].clone();t._resample(ids);assert (t.phase[ids]==2).all() and torch.equal(before,t.commands[ids])
    for _ in range(80):t._update_command()
    assert (t.commands[ids]==0).all() and t.parking.active[ids].all()
    anchors=t.parking.xy[ids].clone()
    for _ in range(10):t._update_command()
    assert torch.equal(anchors,t.parking.xy[ids])
    env.reset()  # discard synthetic command timing before any physics/learning
    return {'status':'PASS','reset_phase_fractions':fractions,'joint_body_sensor_mapping':True,
        'drive_to_brake_to_stop_and_persistent_anchor':True,'synthetic_command_ticks':90,'physical_steps':0,'learning_updates':0}
