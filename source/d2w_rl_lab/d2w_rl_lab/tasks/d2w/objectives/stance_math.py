import torch
from .standing_finetune_v1 import smooth

def sample_mix(old,ranges,draw,u,reverse,maximum):
    new=old.clone();stop=draw<.45;straight=(draw>=.45)&(draw<.90)
    new[stop]=0
    speed=torch.where(maximum,torch.full_like(u,2.),.2+1.8*u)
    limit=torch.where(reverse,(-ranges[:,0]).clamp_min(0),ranges[:,1].clamp_min(0))
    speed=torch.minimum(speed,limit)*torch.where(reverse,-1.,1.)
    new[straight]=0;new[straight,0]=speed[straight]
    return new,stop,straight

def costs(command,ray_z,base_z,joints,reference,target_height):
    valid=torch.isfinite(ray_z).all(1)&(ray_z.abs().amax(1)<1e6)
    h=torch.where(torch.isfinite(ray_z),ray_z,torch.zeros_like(ray_z))
    flat=valid*(1-smooth(h.amax(1)-h.amin(1),.04,.08))
    quiet=(1-smooth(command[:,:2].norm(dim=1),.02,.08))*(1-smooth(command[:,2].abs(),.02,.08))
    straight=(1-smooth(command[:,1].abs(),.02,.1))*(1-smooth(command[:,2].abs(),.02,.15))*smooth(command[:,0].abs(),.05,.2)
    gate=flat*torch.maximum(quiet,straight);qgate=flat*quiet
    d=(joints-reference).reshape(-1,4,3)
    # Explicit absolute hip target: mirror symmetry alone permits symmetric splay.
    hip=((d[:,:,0].abs()-.05).clamp_min(0)/.10).square()
    hip_cost=.5*(hip.mean(1)+hip.amax(1))
    # Mean + worst leg prevents one folded leg from being hidden by the other three.
    tol=d.new_tensor([.05,.25,.25]);scale=d.new_tensor([.15,.30,.30])
    per_leg=((d.abs()-tol).clamp_min(0)/scale).square().mean(2)
    pose=.5*(per_leg.mean(1)+per_leg.amax(1))
    rear=((d[:,2:].abs()-d.new_tensor([.05,.12,.12])).clamp_min(0)/scale).square().mean(2).amax(1)
    height=torch.where(valid,base_z-h.mean(1),torch.full_like(base_z,target_height))
    height_cost=((height-target_height)/.05).square()
    return {'gate':gate,'quiet_gate':qgate,'hip':gate*hip_cost,'pose':gate*pose,
            'rear':qgate*rear,'height':gate*height_cost,'relative_height':height,'deviation':d,'valid':valid}

def self_test():
    c=torch.tensor([[0.,0,0],[2.,0,0],[-2.,0,0],[2.,0,1.],[2.,.5,0],[0.,0,0],[0.,0,0]])
    h=torch.zeros(7,8);h[5,0]=.1;h[6,0]=float('nan')
    ref=torch.tensor([0.,.8,-1.5]*4);j=ref.repeat(7,1);z=torch.full((7,),.5)
    a=costs(c,h,z,j,ref,.5)
    assert torch.equal(a['gate'],torch.tensor([1.,1,1,0,0,0,0]))
    assert all(torch.equal(a[k],torch.zeros(7)) for k in ['hip','pose','rear','height'])
    j[1,[0,3,6,9]]=torch.tensor([.4,-.4,.4,-.4]);a=costs(c,h,z,j,ref,.5)
    assert a['hip'][1]>0 and a['pose'][1]>0
    assert torch.equal(a['deviation'][1,0],a['deviation'][1,1]*torch.tensor([-1.,1.,1.]))
    j=ref.repeat(7,1);j[0,10]+=.6;a=costs(c,h,z,j,ref,.5)
    assert a['rear'][0]>0 and a['pose'][0]>0
    per=((.6-.25)/.30)**2/3
    assert abs(a['pose'][0].item()-.625*per)<1e-5
    j=ref.repeat(7,1);j[0,7]+=.6;b=costs(c,h,z,j,ref,.5)
    assert torch.allclose(a['rear'],b['rear']) and torch.allclose(a['pose'],b['pose'])
    a=costs(c,h,z-.05,j,ref,.5);b=costs(c,h+3,z+3-.05,j,ref,.5)
    assert torch.allclose(a['height'],b['height'],atol=1e-5) and abs(a['height'][1].item()-1)<1e-5
    for key in ['hip','pose','rear','height']:assert torch.isfinite(a[key]).all() and a[key][6]==0
    d=(torch.arange(1000)+.5)/1000;r=torch.tensor([[-2.,2.]]).repeat(1000,1);old=torch.ones(1000,3)
    new,stop,straight=sample_mix(old,r,d,d,torch.arange(1000)%2==0,torch.ones(1000,dtype=torch.bool))
    assert stop.sum()==450 and straight.sum()==450 and (new[stop]==0).all()
    assert (new[straight,0]==2).sum()==225 and (new[straight,0]==-2).sum()==225
    assert torch.equal(new[d>=.9],old[d>=.9])
    return {'status':'PASS','checks':['fixed_reference_zero_cost','symmetric_splay_nonzero_cost',
        'single_rear_leg_worst_aggregation','left_right_rear_equal_treatment','terrain_translation_invariance',
        'invalid_ray_finite_zero_cost','turn_lateral_rough_exclusion','45_45_10_sampling','forward_reverse_balance'],
        'learning_updates':0,'optimizer_steps':0,'physics_steps':0}
