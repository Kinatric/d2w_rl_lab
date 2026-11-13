"""Current-state stair objectives; no last-side target or gait clock."""
import torch
from .terrain_alignment import smooth


def sequence_step(center, ground, bearing, near_surface, wall, reference_load,
                  direction, command, velocity, gate, valid, previous, dt):
    """Pay new supported placements once; keep only anti-repeat frontier memory."""
    if previous is None:
        previous=dict(center=center.clone(),ground=ground.clone(),last_ground=ground.clone(),
                      support=torch.zeros_like(ground))
        valid=torch.zeros_like(valid)
    active=gate & valid
    frontier=torch.where(valid[:,None,None],previous['center'],center)
    floor=torch.where(valid[:,None],previous['ground'],ground)
    stable=bearing & near_surface & active[:,None]
    stable &= (ground-previous['last_ground']).abs()<.015
    support=torch.where(stable,previous['support']+1.,0.)
    others=(bearing.sum(-1)[:,None]-bearing.to(torch.long))>=2
    advance=((center-frontier)*direction[:,None,:]).sum(-1)
    events=(support>=3)&others&((ground-floor)>.03)&(advance>.06)&active[:,None]
    # Immediate state-dependent cost: no remembered blocked timer or repeat side.
    stalled=1.-smooth(velocity[:,0]/command[:,0].clamp_min(.1),.25,.75)
    strength=smooth(wall,.15*reference_load[:,None],.6*reference_load[:,None])
    values=dict(front=events[:,:2].to(ground.dtype).mean(-1)/dt,
                rear=events[:,2:].to(ground.dtype).mean(-1)/dt,
                blockage=active*stalled*(strength*others).mean(-1))
    memory=dict(center=torch.where(events[:,:,None],center,frontier).detach().clone(),
                ground=torch.where(events,torch.maximum(floor,ground),floor).detach().clone(),
                last_ground=ground.detach().clone(),support=support.detach().clone())
    return values,memory,events


def approach_costs(center, bottom, ground, rays, direction, bearing, load,
                   wall, reference, command, velocity, gate):
    """Bounded unloading, low swing-clearance and excessive pair-lag costs.

    No reward for staying airborne. Normal one-tread offsets are allowed.
    Terrain is a privileged reward label, not an extra deployment observation.
    """
    finite=torch.isfinite(rays).all(-1)
    safe=torch.where(finite[:,:,None],rays,0.)
    delta=safe[:,None,:,:2]-center[:,:,None,:2]
    along=(delta*direction[:,None,None,:2]).sum(-1)
    lateral=-delta[...,0]*direction[:,None,None,1]+delta[...,1]*direction[:,None,None,0]
    scan=finite[:,None,:]&(along>.025)&(along<.20)&(lateral.abs()<.09)
    ahead=safe[:,None,:,2].expand(-1,4,-1).masked_fill(~scan,-float('inf')).amax(-1)
    has_scan=scan.any(-1)
    ahead=torch.where(has_scan,ahead,ground)
    riser=(ahead-ground)
    obstacle=has_scan&(riser>.03)&(riser<.30)
    others=(bearing.sum(-1)[:,None]-bearing.to(torch.long))>=2
    scope=gate[:,None]&obstacle&others
    stalled=1.-smooth(velocity[:,0]/command[:,0].clamp_min(.1),.25,.75)
    wall_strength=smooth(wall,.15*reference[:,None],.6*reference[:,None])
    loaded=smooth(load,.1*reference[:,None],.6*reference[:,None])
    unload=(scope*stalled[:,None]*wall_strength*loaded).mean(-1)
    # Only a currently unloaded limb near a higher tread gets this clearance cost.
    deficit=((ahead+.02-bottom)/.12).clamp(0.,1.)
    clearance=(scope*(1.-loaded)*deficit).mean(-1)
    forward=(center*direction[:,None,:]).sum(-1)
    pair_delta=torch.stack((forward[:,0]-forward[:,1],forward[:,2]-forward[:,3]),-1).abs()
    both=torch.stack((bearing[:,0]&bearing[:,1],bearing[:,2]&bearing[:,3]),-1)
    lag=(smooth(pair_delta,.34,.55)*both).mean(-1)*gate*stalled
    return dict(unload=unload,clearance=clearance,pair_lag=lag)


def reward(env,asset_cfg,sensor_cfg,component):
    from .stair_coordination import reward as collect_geometry
    from isaaclab.utils.math import quat_apply
    step=int(env.common_step_counter)
    old=getattr(env,'_d2w_sequence_state',None)
    if old is not None and old['step']==step:return old['values'][component]
    # Reuse the existing contact ordering and curriculum observations, once/step.
    collect_geometry(env,asset_cfg,sensor_cfg)
    geometry=env._d2w_transfer_state
    robot=env.scene[asset_cfg.name];c=env.command_manager.get_command('base_velocity')
    episode=env.episode_length_buf.clone()
    direction=c.new_zeros(env.num_envs,3);direction[:,0]=1.
    direction=quat_apply(robot.data.root_quat_w.torch,direction)
    direction[:,2]=0.
    direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1.e-8)
    valid=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    descent_age=c.new_zeros(env.num_envs)
    if old is not None:
        valid=(episode==old['episode']+1)&(episode>1)&(step==old['step']+1)
        valid &= (direction*old['anchor_direction']).sum(-1)>.866
        # Speed changes alone do not erase placements; reversal does.
        valid &= (c[:,0]*old['command'][:,0])>0
        descent_age=torch.where(geometry['slope']<-.03,old['descent_age']+env.step_dt,0.)
        valid &= geometry['stairs'] & (descent_age<=.5)
    anchor_direction=direction if old is None else torch.where(valid[:,None],old['anchor_direction'],direction)
    forces=env.scene.sensors[sensor_cfg.name].data.normal_force_matrix_w.torch[:,sensor_cfg.body_ids]
    forces=torch.where(torch.isfinite(forces).all(-1,keepdim=True),forces,0.)
    up=forces.new_tensor(env.cfg.sim.gravity);gravity=up.norm();up=-up/gravity.clamp_min(1.e-8)
    vertical=(forces*up).sum(-1).abs()
    horizontal=(forces-(forces*up).sum(-1,keepdim=True)*up).norm(dim=-1)
    # Remove the modest horizontal normal component of inclined support.
    wall=(horizontal-.6*vertical).clamp_min(0.).sum(-1)
    reference=robot.data.body_mass.torch.sum(-1)*gravity/4.
    load=(forces.sum(-2)*up).sum(-1).clamp_min(0.)
    values,memory,events=sequence_step(geometry['center'],geometry['ground'],geometry['bearing'],
        geometry['near_surface'],wall,reference,direction,c,robot.data.root_lin_vel_b.torch,
        geometry['placement_gate'],valid,None if old is None else old['memory'],env.step_dt)
    # Only verified front/rear placement rewards use the broader per-wheel
    # geometry scope. Keep the legacy blockage and approach components on
    # their original scope, even though the active config removes them.
    values['blockage']=values['blockage']*geometry['gate']
    values.update(approach_costs(geometry['center'],geometry['bottom'],geometry['ground'],
        env.scene.sensors['height_scanner'].data.ray_hits_w.torch,direction,geometry['bearing'],load,
        wall,reference,c,robot.data.root_lin_vel_b.torch,geometry['gate'] & valid))
    env._d2w_sequence_state=dict(step=step,episode=episode,command=c.clone(),direction=direction.clone(),
        anchor_direction=anchor_direction.clone(),descent_age=descent_age.clone(),
        values=values,memory=memory,events=events,wall=wall,gate=geometry['gate'].clone(),
        placement_gate=geometry['placement_gate'].clone())
    return values[component]


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    params=dict(cfg.rewards.stair_repeat_transfer.params)
    for name in ['stair_repeat_transfer','stair_alternate_placement','stair_repeat_placement','stair_side_dominance']:
        setattr(cfg.rewards,name,None)
    for name,component,weight in [('stair_front_placement','front',.08),('stair_rear_placement','rear',.08),
            ('stair_blockage','blockage',-.06),('stair_unload','unload',-.04),
            ('stair_swing_clearance','clearance',-.04),('stair_pair_lag','pair_lag',-.04)]:
        setattr(cfg.rewards,name,RewardTermCfg(func=reward,weight=weight,params=dict(params,component=component)))
    cfg.rewards.stair_straight_tracking.weight=-.15
    return cfg
