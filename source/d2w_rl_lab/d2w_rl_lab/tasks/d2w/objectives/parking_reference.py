import torch

def yaw_of(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

class ParkingReference:
    def __init__(self,n,device):
        self.active=torch.zeros(n,device=device,dtype=torch.bool)
        self.xy=torch.zeros(n,2,device=device)
        self.yaw=torch.zeros(n,device=device)

    def update(self,ids,commands,position,quaternion,reset=False):
        # Match stationary_hold.zero_mask exactly. A small requested motion
        # must exit parking so the next true stop captures its own reference.
        quiet=commands.abs().amax(dim=1)<1.e-6
        capture=quiet & (~self.active[ids] | reset)
        selected=ids[capture]
        self.xy[selected]=position[capture,:2]
        self.yaw[selected]=yaw_of(quaternion[capture])
        self.active[ids]=quiet

    def errors(self,position,quaternion):
        delta=position[:,:2]-self.xy
        yaw_delta=yaw_of(quaternion)-self.yaw
        angle=torch.atan2(torch.sin(yaw_delta),torch.cos(yaw_delta))
        return delta.square().sum(dim=1)*self.active, angle*self.active

def self_test():
    ref=ParkingReference(2,'cpu');ids=torch.arange(2)
    zero=torch.zeros(2,3);p=zero.clone();q=torch.tensor([[0.,0,0,1.]]).repeat(2,1)
    ref.update(ids,zero,p,q,True);assert ref.active.all()
    p[0,0]=.1
    ref.update(ids,zero,p,q);assert abs(ref.errors(p,q)[0][0].item()-.01)<1e-7
    # Repeated zero resampling must retain the initial parking reference.
    ref.update(ids,zero,p,q);assert ref.xy[0,0]==0
    moving=zero.clone();moving[0,0]=1
    ref.update(ids,moving,p,q);assert not ref.active[0]
    ref.update(ids,zero,p,q);assert ref.xy[0,0]==.1
    p[0,0]=2;ref.update(ids,zero,p,q,True);assert ref.xy[0,0]==2
    # One environment reset must not reset its neighbour's parking anchor.
    p[1,0]=.2;ref.update(ids[:1],zero[:1],p[:1],q[:1],True)
    assert abs(ref.errors(p,q)[0][1].item()-.04)<1e-7
    def quat(deg):
        a=torch.tensor(deg*torch.pi/180/2);return torch.tensor([[0,0,torch.sin(a),torch.cos(a)]])
    ref.update(ids[:1],zero[:1],p[:1],quat(179),True)
    _,angle=ref.errors(p,torch.cat([quat(-179),q[1:]]))
    assert abs(angle[0].item()-2*torch.pi/180)<1e-6
    assert abs(yaw_of(-quat(179)).item()-yaw_of(quat(179)).item())<1e-6
    return {'status':'PASS','checks':['stop_capture','persistent_zero_anchor','move_deactivates','reenter_captures','reset_recaptures','per_env_reset_isolation','yaw_wrap','quaternion_sign']}
