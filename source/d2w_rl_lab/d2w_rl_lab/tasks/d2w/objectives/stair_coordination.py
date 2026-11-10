"""D2W preference for alternating verified ascending placements, including rolling; no clocked gait."""
import torch
from .transfer_math import transfer_repeat_cost


def practice_commands(command, stairs, elapsed, random_select, random_speed, performance_cap=None):
    """Keep zero samples; focused speed is bounded by time and training readiness."""
    nonzero = command.abs().amax(-1) > 1.e-6
    eligible = stairs & nonzero
    selected = eligible & (random_select < .25)
    cap = 1. + (elapsed / 4000.).clamp(0., 1.)
    if performance_cap is not None:
        if not .6 <= float(performance_cap) <= 2.:
            raise ValueError("Invalid stair practice performance cap")
        cap = torch.minimum(cap, torch.full_like(cap, float(performance_cap)))
    result = command.clone()
    result[:, 0] = torch.where(eligible, result[:, 0] * .5, result[:, 0])
    # Half the focused samples cover slow entry and rear-leg follow-through.
    speed = torch.where(random_speed<.5, .15+.5*random_speed,
                        .5+(2.*random_speed-1.)*(cap-.5))
    result[:, 0] = torch.where(selected, speed, result[:, 0])
    result[:, 1:] = torch.where(selected[:, None], 0., result[:, 1:])
    return result, selected, cap


def placement_events(gain, advance, support, other_support, gate, valid, progressing):
    return (gain>.025)&(advance>.04)&support&other_support&gate[:,None]&valid[:,None]&progressing[:,None]


def reward(env, asset_cfg, sensor_cfg):
    from isaaclab.utils.math import quat_apply
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    from .terrain_alignment import height_trend
    robot = env.scene[asset_cfg.name]
    step = int(env.common_step_counter)
    old = getattr(env, '_d2w_transfer_state', None)
    if old is not None and old['step'] == step:
        return old['cost'] / env.step_dt
    n = env.num_envs;device = env.device;ids = asset_cfg.body_ids
    command = env.command_manager.get_command('base_velocity')
    episode = env.episode_length_buf.clone()
    valid = torch.zeros(n, dtype=torch.bool, device=device)
    if old is not None:
        valid = (episode == old['episode'] + 1) & (episode > 1) & (step == old['step'] + 1)
        valid &= (command - old['command']).abs().amax(-1) < 1.e-5
    quat = robot.data.body_quat_w.torch[:, ids]
    offset = command.new_zeros(n,4,3);offset[:,:,1] = command.new_tensor([1.,-1.,1.,-1.])*.0455
    center = robot.data.body_pos_w.torch[:, ids] + quat_apply(quat, offset)
    axis = torch.zeros_like(center);axis[:,:,1] = 1.;axis = quat_apply(quat, axis)
    rays = env.scene.sensors['height_scanner'].data.ray_hits_w.torch
    finite = torch.isfinite(rays).all(-1);safe = torch.where(finite[:,:,None], rays, 0.)
    distance, nearest = ((center[:,:,None,:2]-safe[:,None,:,:2]).square().sum(-1).masked_fill(~finite[:,None,:],float('inf'))).min(-1)
    ground = safe[:,:,2].gather(1,nearest)
    extent = .0813*torch.sqrt((1-axis[:,:,2].square()).clamp_min(0)) + .024675*axis[:,:,2].abs()
    clearance = center[:,:,2]-extent-ground
    geometry_valid = (distance < .15**2).all(-1) & finite.sum(-1).ge(8)
    force = env.scene.sensors[sensor_cfg.name].data.normal_force_matrix_w.torch[:,sensor_cfg.body_ids].sum(-2)
    gravity = force.new_tensor(env.cfg.sim.gravity);up = -gravity/gravity.norm().clamp_min(1.e-8)
    bearing = ((force*up).sum(-1)>1.)
    support = bearing[:,:2] & (clearance[:,:2].abs()<.05)
    slope, fit_valid = height_trend(rays,robot.data.root_pos_w.torch,robot.data.root_quat_w.torch)
    masks = get_terrain_masks(env);zero = torch.zeros(n,dtype=torch.bool,device=device)
    stairs = masks.get('stairs_up',zero)|masks.get('stairs_down',zero)
    # Compare left/right scan strips at comparable forward positions, not feet
    # which legitimately stand on different steps during a transfer.
    q=robot.data.root_quat_w.torch;x,y,z,w=q.unbind(-1)
    heading=torch.stack([1-2*(y*y+z*z),2*(x*y+w*z)],-1);heading/=heading.norm(dim=-1,keepdim=True).clamp_min(1.e-8)
    delta=safe[:,:,:2]-robot.data.root_pos_w.torch[:,None,:2]
    local_x=(delta*heading[:,None,:]).sum(-1)
    local_y=-delta[:,:,0]*heading[:,None,1]+delta[:,:,1]*heading[:,None,0]
    strips=[];counts=[]
    for sign in [1.,-1.]:
        mask=finite&(local_x>.1)&(local_x<.65)&(local_y*sign>.08)&(local_y*sign<.4)
        count=mask.sum(-1);counts.append(count)
        strips.append((safe[:,:,2]*mask).sum(-1)/count.clamp_min(1))
    symmetric=(counts[0]>=2)&(counts[1]>=2)&((strips[0]-strips[1]).abs()<.06)
    healthy=(-robot.data.projected_gravity_b.torch[:,2]>.8)&(~env.termination_manager.terminated)
    healthy &= robot.data.root_ang_vel_b.torch[:,:2].norm(dim=-1)<2.
    gate=stairs&(slope>.03)&fit_valid&symmetric&geometry_valid&healthy
    gate &= (command[:,0]>.1)&(command[:,1].abs()<.1)&(command[:,2].abs()<.2)
    frontier_center=center[:,:2].clone();frontier_ground=ground[:,:2].clone()
    events=torch.zeros(n,2,dtype=torch.bool,device=device)
    direction=command.new_zeros(n,3);direction[:,0]=1.;direction=quat_apply(robot.data.root_quat_w.torch,direction)
    if old is not None:
        # Freeze geometry over the same short dropout tolerated by side memory.
        gap = old['memory'].get('inactive_age', command.new_zeros(n))
        keep=valid & (gap <= .2)
        frontier_center=torch.where(keep[:,None,None],old['frontier_center'],frontier_center)
        frontier_ground=torch.where(keep[:,None],old['frontier_ground'],frontier_ground)
        advance=((center[:,:2]-frontier_center)*direction[:,None,:]).sum(-1)
        gain=ground[:,:2]-frontier_ground
        other_support=(bearing.sum(-1)[:,None]-bearing[:,:2].to(torch.long))>=2
        progress_state=getattr(env,'_d2w_rollclimb_state',None)
        progressing=torch.zeros(n,dtype=torch.bool,device=device)
        if progress_state is not None and progress_state['step']==step:
            progressing=progress_state['values']['progress']>0
        events=placement_events(gain,advance,support,other_support,gate,valid,progressing)
        frontier_ground=torch.where(events,torch.maximum(frontier_ground,ground[:,:2]),frontier_ground)
        frontier_center=torch.where(events[:,:,None],center[:,:2],frontier_center)
    cost,memory=transfer_repeat_cost(events,robot.data.root_pos_w.torch[:,:2],gate,valid,None if old is None else old['memory'],env.step_dt)
    env._d2w_transfer_state=dict(step=step,episode=episode,command=command.clone(),frontier_center=frontier_center.clone(),frontier_ground=frontier_ground.clone(),memory=memory,cost=cost.clone(),events=events.clone(),gate=gate.clone(),ascent=slope>.03)
    env._d2w_transfer_state.update(center=center.clone(),ground=ground.clone(),bearing=bearing.clone(),
        near_surface=(clearance.abs()<.05),bottom=(center[:,:,2]-extent).clone(),stairs=stairs.clone(),slope=slope.clone())
    sampler=env.command_manager.get_term('base_velocity')
    if hasattr(sampler,'observe_ascent'):
        sampler.observe_ascent(robot,stairs,slope,valid)
    return cost/env.step_dt


def tracking_cost(command, velocity, gyro):
    y=((velocity[:,1]-command[:,1])/.15).square()
    w=((gyro[:,2]-command[:,2])/.10).square()
    return .5*(y/(1+y)+w/(1+w))


def stair_tracking(env,asset_cfg):
    import math
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    from .rollclimb import smooth
    robot=env.scene[asset_cfg.name];c=env.command_manager.get_command('base_velocity')
    masks=get_terrain_masks(env);zero=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    stairs=masks.get('stairs_up',zero)|masks.get('stairs_down',zero)
    gate=stairs*smooth(c[:,0].abs(),.05,.2)*(1-smooth(c[:,1].abs(),.02,.10))*(1-smooth(c[:,2].abs(),.05,.2))
    gate*=smooth(-robot.data.projected_gravity_b.torch[:,2],math.cos(1.),.8)*(~env.termination_manager.terminated)
    value=gate*tracking_cost(c,robot.data.root_lin_vel_b.torch,robot.data.root_ang_vel_b.torch)
    env._d2w_stair_tracking_diagnostics=dict(gate=gate.clone(),value=value.clone())
    return value


def configure(cfg, start_index=64800):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from ..stair_practice_command import StairPracticeCommand
    cfg.commands.base_velocity.class_type=StairPracticeCommand
    cfg.commands.base_velocity.stair_practice_start_index=start_index
    for name in ['stairs_up','stairs_down']:
        cfg.commands.base_velocity.terrain_max_command_ranges[name]['lin_vel_x']=(-2.,2.)
    wheels=['FL_foot','FR_foot','RL_foot','RR_foot']
    cfg.rewards.stair_repeat_transfer=RewardTermCfg(func=reward,weight=-.02,params=dict(asset_cfg=SceneEntityCfg('robot',body_names=wheels,preserve_order=True),sensor_cfg=SceneEntityCfg('stair_contacts',body_names=wheels,preserve_order=True)))
    cfg.rewards.stair_straight_tracking=RewardTermCfg(func=stair_tracking,weight=-.10,params=dict(asset_cfg=SceneEntityCfg('robot')))
    return cfg
