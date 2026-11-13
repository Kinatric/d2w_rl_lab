from .consistent_gate import designated_flat
"""Stage-local reward reconciliation. Observations/actions/actuators unchanged."""
import torch
from . import stand_balance as parent
from .standing_finetune import METRICS
from .standing_finetune_v1 import smooth

def flat_gate(env):
    h=env.scene.sensors['height_scanner'].data.ray_hits_w.torch[...,2]
    valid=torch.isfinite(h).all(1);safe=torch.where(torch.isfinite(h),h,torch.zeros_like(h))
    return designated_flat(env)*valid*(1-smooth(safe.amax(1)-safe.amin(1),.04,.08))

def unified_reference(env,asset_cfg,command_name,stand_still_scale,command_threshold,kind,velocity_threshold=None):
    from d2w_rl_lab.tasks.d2w.mdp_base import rewards
    from d2w_rl_lab.assets.d2w_metadata import D2W_TRAINING_JOINT_POSITIONS
    c=env.command_manager.get_command(command_name);r=env.scene[asset_cfg.name]
    quiet=(1-smooth(c[:,:2].norm(dim=1),.02,.08))*(1-smooth(c[:,2].abs(),.02,.08))
    straight=(1-smooth(c[:,1].abs(),.02,.1))*(1-smooth(c[:,2].abs(),.02,.15))*smooth(c[:,0].abs(),.05,.2)
    g=flat_gate(env)*torch.maximum(quiet,straight)
    kw=dict(asset_cfg=asset_cfg,command_name=command_name,stand_still_scale=stand_still_scale,command_threshold=command_threshold)
    if kind=='hip':old=rewards.hip_pos_penalty_l1(env,**kw)
    else:old=rewards.joint_pos_penalty_l1(env,velocity_threshold=velocity_threshold,**kw)
    ref=r.data.joint_pos.torch.new_tensor([D2W_TRAINING_JOINT_POSITIONS[r.joint_names[i]] for i in asset_cfg.joint_ids])
    fixed=(r.data.joint_pos.torch[:,asset_cfg.joint_ids]-ref).abs().sum(1)
    return (1-g)*old+g*fixed

def growth_cost(delta,velocity,scale):
    # Cost of growing cumulative displacement. Returning is optional, not a
    # rewarded command to drive back through the parking point.
    distance=delta.norm(dim=1)
    outward=(delta*velocity).sum(1)/distance.clamp_min(1e-6)
    x=outward.clamp_min(0)/scale
    return x.square()/(1+x.square())

def parking_growth(env,asset_cfg,component):
    r=env.scene[asset_cfg.name];t=env.command_manager.get_term('base_velocity')
    gate=flat_gate(env)*t.parking.active
    if component=='parking_position':
        return gate*growth_cost(r.data.root_pos_w.torch[:,:2]-t.parking.xy,r.data.root_lin_vel_w.torch[:,:2],.05)
    _,angle=t.parking.errors(r.data.root_pos_w.torch,r.data.root_quat_w.torch)
    # Exact derivative of quaternion yaw for world-frame angular velocity.
    q=r.data.root_quat_w.torch;x,y,z,w=q.unbind(1)
    wx,wy,wz=r.data.root_ang_vel_w.torch.unbind(1)
    dw=-(wx*x+wy*y+wz*z)/2;dx=(wx*w+wy*z-wz*y)/2
    dy=(-wx*z+wy*w+wz*x)/2;dz=(wx*y-wy*x+wz*w)/2
    a=2*(w*z+x*y);b=1-2*(y*y+z*z)
    da=2*(dw*z+w*dz+dx*y+x*dy);db=-4*(y*dy+z*dz)
    yaw_rate=(b*da-a*db)/(a*a+b*b).clamp_min(1e-6)
    return gate*growth_cost(angle[:,None],yaw_rate[:,None],.15)

def configure(cfg):
    cfg=parent.configure(cfg)
    for name,kind in [('hip_pos_penalty_l1','hip'),('joint_pos_penalty_l1','leg')]:
        term=getattr(cfg.rewards,name);term.func=unified_reference;term.params['kind']=kind
    for name in ['standing_parking_position','standing_parking_heading']:
        getattr(cfg.rewards,name).func=parking_growth
    return cfg

def self_test():
    d=torch.tensor([[.2,0],[.2,0],[.2,0],[0.,0],[100.,0]])
    v=torch.tensor([[0.,0],[.1,0],[-.1,0],[0.,0],[1e6,0]])
    c=growth_cost(d,v,.05)
    assert c[0]==0 and c[1]>0 and c[2]==0 and c[3]==0 and torch.isfinite(c).all() and (c<=1).all()
    # A stationary displaced robot pays no continuing anchor debt; moving
    # toward its anchor earns no bonus. No incentive to reset just to clear it.
    assert torch.equal(growth_cost(-d,-v,.05),c)
    cmd=torch.linspace(0,.2,1001)
    g=1-smooth(cmd,.02,.2)
    assert g[0]==1 and g[-1]==0 and (g[1:]-g[:-1]).abs().max()<.002
    return {'status':'PASS','parent':parent.self_test(),'checks':['bounded_outward_displacement_growth','stationary_offset_zero_debt','returning_no_bonus','left_right_equal','continuous_stop_blend'], 'learning_updates':0}

def runtime_preflight(env):
    result=parent.runtime_preflight(env)
    for name in ['hip_pos_penalty_l1','joint_pos_penalty_l1','standing_parking_position','standing_parking_heading']:
        c=env.reward_manager.get_term_cfg(name);v=c.func(env,**c.params)
        assert v.shape==(env.num_envs,) and torch.isfinite(v).all()
    result['reconciled_rewards_finite']=True
    return result
