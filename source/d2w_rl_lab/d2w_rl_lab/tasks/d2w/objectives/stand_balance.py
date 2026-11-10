from .consistent_gate import designated_flat,span_excess
NOMINAL_SPAN=0.3868
"""Stage-local D2W standing reward correction; no deployment/action changes."""
import torch
from . import stop_recovery as parent
from .standing_finetune import METRICS
from .standing_finetune_v1 import smooth


def mirror_pairs(deviation):
    d=deviation.reshape(-1,4,3)
    return (d[:,[0,2]]-d[:,[1,3]]*d.new_tensor([-1.,1.,1.])).square().mean(2)


def scores(speed,yaw,pair_sq,span,load,pose_sq=None):
    # Preserve prior physical scales. Sum costs BEFORE mapping to a bounded
    # score, so good yaw/span cannot average away sustained drift/asymmetry.
    drift=speed.square()/.02**2
    pairs=pair_sq/.1**2
    total=drift+yaw.square()/.15**2+pairs.sum(1)+span_excess(span,NOMINAL_SPAN).square()/.10**2
    # Same fixed standing reference as stance_pose; worst leg cannot hide in the mean.
    # No new reward term, joint target, control clamp, or reward for returning to an anchor.
    if pose_sq is not None:
        total = total + pose_sq.mean(1) + pose_sq.amax(1)
    quality=load/(1+total)
    # Still charges drift/asymmetry if contact is lost; bounded by 2, unlike
    # accumulating large per-step penalties. Both pairs contribute explicitly.
    residual=drift/(1+drift)+(pairs/(1+pairs)).mean(1)
    return quality,residual


def standing_balance(env,asset_cfg,wheels_cfg,sensor_cfg,component):
    from isaaclab.utils.math import quat_apply_inverse
    r=env.scene[asset_cfg.name];t=env.command_manager.get_term('base_velocity')
    h=env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2]
    valid=torch.isfinite(h).all(1);safe=torch.where(torch.isfinite(h),h,torch.zeros_like(h))
    gate=designated_flat(env)*valid*(1-smooth(safe.amax(1)-safe.amin(1),.04,.08))*(1-smooth(t.command.abs().amax(1),.02,.2))
    d=r.data.joint_pos.torch[:,asset_cfg.joint_ids]-r.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    pair_sq=mirror_pairs(d)
    delta=r.data.body_pos_w.torch[:,wheels_cfg.body_ids]-r.data.root_pos_w.torch[:,None,:]
    body=quat_apply_inverse(r.data.root_quat_w.torch[:,None,:].expand(-1,4,-1).reshape(-1,4),delta.reshape(-1,3)).reshape(-1,4,3)
    span=body[:,:2,0].mean(1)-body[:,2:,0].mean(1)
    force=env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:,sensor_cfg.body_ids,2]
    load=(force>1.).float().mean(1)
    speed=r.data.root_lin_vel_b.torch[:,:2].norm(dim=1)
    from d2w_rl_lab.assets.d2w_metadata import D2W_FIXSTAND_JOINT_POSITIONS
    ref=d.new_tensor([D2W_FIXSTAND_JOINT_POSITIONS[r.joint_names[i]] for i in asset_cfg.joint_ids])
    deviation=(r.data.joint_pos.torch[:,asset_cfg.joint_ids]-ref).reshape(-1,4,3)
    tolerance=deviation.new_tensor([.05,.12,.12])
    scale=deviation.new_tensor([.15,.30,.30])
    pose_sq=((deviation.abs()-tolerance).clamp_min(0)/scale).square().mean(2)
    quality,residual=scores((r.data.root_lin_vel_b.torch[:,:2]-t.command[:,:2]).norm(dim=1),r.data.root_ang_vel_b.torch[:,2]-t.command[:,2],pair_sq,span,load,pose_sq)
    healthy=(r.data.projected_gravity_b.torch[:,2]<-.9)&(~env.termination_manager.terminated)
    if component=='residual':return gate*residual
    assert component=='quality'
    den=gate.sum().clamp_min(1)
    mean=lambda x:(x*gate).sum().detach()/den
    pos_sq,heading=t.parking.errors(r.data.root_pos_w.torch,r.data.root_quat_w.torch)
    METRICS.update({'park_front_mirror_rms_rad':mean(pair_sq[:,0]).sqrt(),
        'park_rear_mirror_rms_rad':mean(pair_sq[:,1]).sqrt(),
        'park_worst_pair_rms_rad':mean(pair_sq.amax(1).sqrt()),
        'park_front_wheel_x_mismatch_m':mean((body[:,0,0]-body[:,1,0]).abs()),
        'park_rear_wheel_x_mismatch_m':mean((body[:,2,0]-body[:,3,0]).abs()),
        'park_wheelbase_m':mean(span),'park_contact_fraction':mean(load),
        'park_over10s_fraction':((t.quiet_age>=10).float()*gate).mean().detach(),
        'park_age_mean_s':mean(t.quiet_age),'drift_speed_m_s':mean(speed),
        'parking_displacement_rms_m':mean(pos_sq).sqrt(),
        'parking_heading_abs_deg':mean(heading.abs()*180/torch.pi),
        'park_tilt_deg':mean(torch.acos((-r.data.projected_gravity_b.torch[:,2]).clamp(-1,1))*180/torch.pi),
        'park_quality':mean(quality),'park_residual_cost':mean(residual),
        'true_failure_fraction':env.termination_manager.terminated.float().mean().detach(),
        'zero_command_fraction':(t.command.abs().amax(1)<=.02).float().mean().detach(),
        'quiet_flat_fraction':gate.mean().detach()})
    return gate*healthy*quality


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    cfg=parent.configure(cfg)
    params=dict(cfg.rewards.standing_quality.params)
    cfg.rewards.standing_quality=RewardTermCfg(func=standing_balance,weight=2.,params=dict(params,component='quality'))
    cfg.rewards.standing_balance_residual=RewardTermCfg(func=standing_balance,weight=-.5,params=dict(params,component='residual'))
    return cfg


def self_test():
    a=torch.zeros(5);pairs=torch.zeros(5,2);span=torch.full((5,),NOMINAL_SPAN);load=torch.ones(5)
    speed=torch.tensor([0.,.11,0.,0.,0.],requires_grad=True)
    pairs[2,0]=.2**2;pairs[3,1]=.2**2;span[4]=.18
    q,c=scores(speed,a,pairs,span,load)
    assert q[0]==1 and c[0]==0
    assert q[1]<.18 and q[2]==q[3] and q[2]<.21 and q[4]<1
    assert c[1]>.8 and c[2]==c[3] and c[2]>0
    assert torch.autograd.grad((2*q-.5*c).sum(),speed)[0][1]<0
    p=torch.tensor([[.02,.03]],requires_grad=True)
    pq,pc=scores(a[:1],a[:1],p,span[:1],load[:1])
    assert (torch.autograd.grad((2*pq-.5*pc).sum(),p)[0]<0).all()
    q0,c0=scores(speed.detach(),a,pairs,span,load*0)
    assert (q0==0).all() and torch.equal(c0,c)
    d=torch.tensor([[.1,.2,.3,-.1,.2,.3,.2,.4,.6,-.2,.4,.6]])
    assert (mirror_pairs(d)==0).all()
    for joint in range(12):
        changed=d.clone();changed[0,joint]+=.2
        assert mirror_pairs(changed).sum()>0
    extreme=torch.full((5,),1e6)
    eq,ec=scores(extreme,extreme,torch.full((5,2),1e12),extreme,load)
    assert torch.isfinite(eq).all() and torch.isfinite(ec).all() and (ec<=2).all()
    return {'status':'PASS','parent':parent.self_test(),
        'checks':['no_average_compensation','front_rear_equal_treatment','all_12_joints_detected',
                  'mirror_hip_signs','negative_speed_and_both_pair_score_derivatives',
                  'residual_not_removed_by_contact_loss','bounded_finite_cost'],
        'ideal_vs_011mps_quality':[q[0].item(),q[1].item()],
        'learning_updates':0,'optimizer_steps':0}


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
    assert torch.isfinite(standing_balance(env,**p)).all()
    ids=(t.phase==1).nonzero().flatten()[:8];assert len(ids)==8
    before=t.commands[ids].clone();t._resample(ids);assert (t.phase[ids]==2).all() and torch.equal(before,t.commands[ids])
    for _ in range(80):t._update_command()
    assert (t.commands[ids]==0).all() and t.parking.active[ids].all()
    anchors=t.parking.xy[ids].clone()
    for _ in range(10):t._update_command()
    assert torch.equal(anchors,t.parking.xy[ids])
    env.reset()  # discard synthetic timing before learning
    for name in ['standing_quality','standing_balance_residual']:
        c=env.reward_manager.get_term_cfg(name)
        value=c.func(env,**c.params)
        assert value.shape==(env.num_envs,) and torch.isfinite(value).all()
    return {'status':'PASS','reset_phase_fractions':fractions,'joint_body_sensor_mapping':True,
        'drive_to_brake_to_stop_and_persistent_anchor':True,'synthetic_command_ticks':90,'physical_steps':0,'learning_updates':0}
