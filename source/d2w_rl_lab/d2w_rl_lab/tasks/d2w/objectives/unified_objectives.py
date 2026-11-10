import torch
from . import terrain_pitch as parent
from .standing_finetune import METRICS
from .consistent_gate import designated_flat,span_excess

def configure(cfg):
    cfg=parent.configure(cfg)
    # Superseded by stance_hip and unified_reference, with a different old
    # default-centred front-hip target. Remove this duplicate, not motor limits.
    cfg.rewards.front_stance=None
    return cfg

def self_test():
    result=parent.self_test()
    from .stand_balance import NOMINAL_SPAN
    x=torch.tensor([.5,.85,1.,1.15,1.5])*NOMINAL_SPAN
    y=span_excess(x,NOMINAL_SPAN)
    assert (y[1:4]<1e-7).all() and y[0]>0 and y[-1]>0
    return {'status':'PASS','parent':result,'geometry_span_m':NOMINAL_SPAN,'checks':['compatible_nominal_span_envelope','no_exact_front_rear_angle_equality'],'learning_updates':0}

def runtime_preflight(env):
    r=parent.runtime_preflight(env)
    assert 'front_stance' not in env.reward_manager.active_terms
    # At runtime evaluate each gated term on the real mixed-terrain reset.
    # All designated nonflat environments must have zero extra posture costs.
    nonflat=~designated_flat(env)
    gated=['standing_quality','standing_balance_residual','standing_parking_position','standing_parking_heading','stance_hip','stance_pose','stance_height','straight_mirror','straight_tilt','straight_lateral','flat_stand_attitude']
    for name in gated:
        c=env.reward_manager.get_term_cfg(name);v=c.func(env,**c.params)
        assert torch.isfinite(v).all() and (v[nonflat]==0).all(),name
    r['nonflat_special_posture_rewards_zero']=gated
    r['conflicting_front_stance_removed']=True
    return r
