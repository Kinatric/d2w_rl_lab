import torch
from .standing_finetune_v1 import smooth

def costs(command,heights,deviation,gravity,velocity,gyro):
    finite=torch.isfinite(heights).all(dim=1)
    h=torch.where(torch.isfinite(heights),heights,torch.zeros_like(heights))
    flat=finite*(1-smooth(h.max(1).values-h.min(1).values,.04,.08))
    straight=(1-smooth(command[:,1].abs(),.02,.1))*(1-smooth(command[:,2].abs(),.02,.15))
    moving=smooth(command[:,0].abs(),.05,.2)
    gate=flat*straight*moving
    quiet=(1-smooth(command[:,:2].norm(dim=1),.02,.08))*(1-smooth(command[:,2].abs(),.02,.08))
    posture_gate=flat*torch.maximum(straight*moving,quiet)
    pose=deviation.reshape(-1,4,3)
    mirror=(pose[:,[0,2]]-pose[:,[1,3]]*pose.new_tensor([-1.,1.,1.])).square().mean((1,2))
    allowances=pose.new_tensor([.15,.35,.35])
    excess=((pose.abs()-allowances).clamp_min(0)/allowances).square().mean((1,2))
    tilt=gravity[:,:2].square().sum(1)
    lateral=velocity[:,1].square()/.2**2+gyro[:,2].square()/.5**2
    return gate,posture_gate,mirror,excess,tilt,lateral

def sample_mix(old,ranges,draw,u,reverse,maximum):
    # Mutually exclusive nominal groups: 35% stop, 45% longitudinal, 20% original.
    new=old.clone();stop=draw<.35;straight=(draw>=.35)&(draw<.8)
    new[stop]=0
    speed=.2+1.8*u
    speed=torch.where(maximum,torch.full_like(speed,2.),speed)
    lim=torch.where(reverse,(-ranges[:,0]).clamp_min(0),ranges[:,1].clamp_min(0))
    speed=torch.minimum(speed,lim)*torch.where(reverse,-1.,1.)
    new[straight]=0;new[straight,0]=speed[straight]
    return new,stop,straight

def self_test():
    c=torch.tensor([[0.,0,0],[.3,0,0],[2.,0,0],[-2.,0,0],[2.,0,1.],[2.,.5,0],[2.,0,0]])
    h=torch.zeros(7,8);h[6,0]=.1;d=torch.zeros(7,12);g=torch.tensor([[0.,0,-1.]]).repeat(7,1);v=torch.zeros(7,3)
    gate,pg,*_=costs(c,h,d,g,v,v)
    assert torch.equal(gate,torch.tensor([0.,1,1,1,0,0,0]))
    assert pg[0]==1
    d[2,6]=-.6
    out=costs(c,h,d,g,v,v);assert out[2][2]>0 and out[3][2]>0
    d[:]=0;d[2]=torch.tensor([.05,.1,.1,-.05,.1,.1,.05,.1,.1,-.05,.1,.1])
    out=costs(c,h,d,g,v,v);assert out[2][2]==out[3][2]==0
    g[2,0]=.2;v[2,1]=.1;out=costs(c,h,d,g,v,v);assert out[4][2]>0 and out[5][2]>0
    draws=(torch.arange(1000)+.5)/1000;old=torch.ones(1000,3);ranges=torch.tensor([[-2.,2.]]).repeat(1000,1)
    new,stop,straight=sample_mix(old,ranges,draws,torch.linspace(0,1,1000),torch.arange(1000)%2==0,torch.ones(1000,dtype=torch.bool))
    assert stop.sum()==350 and straight.sum()==450
    assert (new[stop]==0).all() and (new[straight,1:]==0).all()
    assert (new[straight,0]==2).sum()==225 and (new[straight,0]==-2).sum()==225
    assert torch.equal(new[draws>=.8],old[draws>=.8])
    return {'status':'PASS','checks':['stop_low_high_reverse_gates','turn_and_terrain_exclusion','rear_abnormal_pose_penalized','mirror_signs','nominal_pose_allowance','tilt_lateral_costs','35_45_20_mixture','forward_reverse_balance_and_2mps','other_commands_preserved'],'learning_updates':0}
