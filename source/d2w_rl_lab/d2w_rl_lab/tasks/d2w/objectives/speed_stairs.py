"""D2W precision tracking and supported stair transitions; no controller changes."""
import torch
from .reward_coherent import flat_gate
from .standing_finetune_v1 import smooth
from .standing_finetune import METRICS


def precise_tracking(command, velocity, flat, std=.5):
    error = (command[:, :2] - velocity[:, :2]).square().sum(1)
    straight = (1-smooth(command[:, 1].abs(), .02, .10)) * (1-smooth(command[:, 2].abs(), .02, .15))
    gate = flat * straight * smooth(command[:, 0].abs(), .2, 1.)
    return (1-gate)*torch.exp(-error/std**2) + gate*torch.exp(-error/.35**2)


def track_linear(env, std, command_name, asset_cfg):
    robot = env.scene[asset_cfg.name]
    command=env.command_manager.get_command(command_name)
    velocity=robot.data.root_lin_vel_b.torch
    value=precise_tracking(command,velocity,flat_gate(env),std)
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    masks=get_terrain_masks(env);zero=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    stairs=masks.get('stairs_up',zero)|masks.get('stairs_down',zero)
    speed=command[:,:2].norm(dim=-1)
    gate=stairs*smooth(speed,.05,.15)*(1.-smooth(speed,.35,.6))
    error=(command[:,:2]-velocity[:,:2]).square().sum(-1)
    return (1.-gate)*value+gate*torch.exp(-error/.25**2)


def flat_fast_preference(env,command_name,sensor_cfg,height_cfg,settings,asset_cfg,component):
    from .consistent_gate import designated_flat
    from .recovery_boundaries import fast_bounded
    # A locally level stair tread must not re-enable flat high-speed posture.
    return designated_flat(env)*fast_bounded(env,command_name,sensor_cfg,height_cfg,settings,asset_cfg,component)


def rolling_contact_velocity(axis, angular, center_velocity, radius=.0813, half_width=.024675):
    up = torch.zeros_like(axis);up[..., 2] = 1.
    radial = up-axis*axis[..., 2:3]
    radial = radial / radial.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    bottom = -radius*radial-half_width*axis[..., 2:3].sign()*axis
    return center_velocity + torch.cross(angular,bottom,dim=-1)


def transition_score(gain, advance, air_time, landed, continuous_support):
    height = smooth(gain, .025, .075)
    progress = smooth(advance, .025, .10)
    duration = smooth(air_time, .02, .06) * (1-smooth(air_time,.5,1.))
    return (height*progress*duration*landed*continuous_support).mean(1)


def stair_terms(env, asset_cfg, sensor_cfg):
    """Cache once per environment step; reset/discontinuity cannot earn touchdown."""
    from isaaclab.utils.math import quat_apply
    from d2w_rl_lab.utils.terrains import get_terrain_masks
    robot=env.scene[asset_cfg.name];ids=asset_cfg.body_ids
    step=int(env.common_step_counter)
    previous=getattr(env,'_d2w_stair_reward_state',None)
    if previous is not None and previous['step']==step:
        return previous['values']
    device=env.device;n=env.num_envs
    command=env.command_manager.get_command('base_velocity')
    masks=get_terrain_masks(env)
    stair=masks.get('stairs_up',torch.zeros(n,device=device,dtype=torch.bool)) | masks.get('stairs_down',torch.zeros(n,device=device,dtype=torch.bool))
    moving=smooth(command[:,:2].norm(dim=1),.05,.20)
    gate=stair*moving
    quat=robot.data.body_quat_w.torch[:,ids]
    offset=torch.zeros((n,4,3),device=device);offset[:,:,1]=offset.new_tensor([1.,-1.,1.,-1.])*.0455
    offset=quat_apply(quat,offset)
    center=robot.data.body_pos_w.torch[:,ids]+offset
    axis=torch.zeros_like(center);axis[:,:,1]=1.;axis=quat_apply(quat,axis)
    angular=robot.data.body_link_ang_vel_w.torch[:,ids]
    velocity=robot.data.body_link_lin_vel_w.torch[:,ids]+torch.cross(angular,offset,dim=-1)
    force=env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:,sensor_cfg.body_ids]
    rays=env.scene.sensors['height_scanner'].data.ray_hits_w.torch
    finite=torch.isfinite(rays).all(2);safe=torch.where(torch.isfinite(rays),rays,torch.zeros_like(rays))
    distances=(center[:,:,None,:2]-safe[:,None,:,:2]).square().sum(-1).masked_fill(~finite[:,None,:],float('inf'))
    distance,nearest=distances.min(-1);ground=safe[:,:,2].gather(1,nearest)
    extent=.0813*torch.sqrt((1-axis[:,:,2].square()).clamp_min(0))+.024675*axis[:,:,2].abs()
    near_surface=(distance<.15**2)&((center[:,:,2]-extent-ground).abs()<.04)
    support=(force[:,:,2]>1.)&near_surface
    # Horizontal material-point motion, not wheel rotation or link velocity alone.
    slip=rolling_contact_velocity(axis,angular,velocity)[:,:,:2].square().sum(-1)/.25**2
    slip_cost=(slip/(1+slip)*support).mean(1)
    gyro=robot.data.root_ang_vel_w.torch
    episode=env.episode_length_buf.clone()
    zero=command.new_zeros(n)
    values={'touchdown':zero.clone(),'slip':gate*slip_cost,'shock':zero.clone()}
    local_dir=torch.cat([command[:,:2],torch.zeros((n,1),device=device)],1)
    local_dir=local_dir/local_dir.norm(dim=1,keepdim=True).clamp_min(1.e-6)
    direction=quat_apply(robot.data.root_quat_w.torch,local_dir)
    origin=center.clone();origin_ground=ground.clone();origin_dir=direction[:,None,:].expand(-1,4,-1).clone()
    air=torch.zeros((n,4),device=device);continuous=torch.zeros((n,4),dtype=torch.bool,device=device)
    if previous is not None:
        valid=(episode==previous['episode']+1)&(episode>1)&(step==previous['step']+1)
        valid4=valid[:,None]
        takeoff=previous['support']&(~support)&valid4
        airborne=(~support)&valid4
        ongoing_support=(support.sum(1)>=2)[:,None]
        origin=torch.where(takeoff[:,:,None],previous['center'],previous['origin'])
        origin_ground=torch.where(takeoff,previous['ground'],previous['origin_ground'])
        origin_dir=torch.where(takeoff[:,:,None],previous['direction'][:,None,:].expand(-1,4,-1),previous['origin_dir'])
        air=torch.where(takeoff,torch.full_like(air,env.step_dt),torch.where(airborne,previous['air']+env.step_dt,air))
        continuous=torch.where(takeoff,ongoing_support,previous['continuous']&ongoing_support)&valid4
        landed=support&(~previous['support'])&valid4
        advance=((center-origin)*origin_dir).sum(-1)
        aligned=(origin_dir*direction[:,None,:]).sum(-1)>.5
        contact_score=transition_score(ground-origin_ground,advance,previous['air'],landed&aligned,continuous)
        healthy=(robot.data.projected_gravity_b.torch[:,2]<-.8)&(~env.termination_manager.terminated)
        values['touchdown']=gate*healthy*contact_score
        # Penalize abrupt angular changes, never absolute uphill pitch or v_z.
        acceleration=(gyro[:,:2]-previous['gyro'][:,:2])/env.step_dt
        shock=acceleration.square().mean(1)/25.**2
        values['shock']=gate*valid*shock/(1+shock)
        origin=torch.where(valid4[:,:,None],origin,center)
        origin_ground=torch.where(valid4,origin_ground,ground)
        origin_dir=torch.where(valid4[:,:,None],origin_dir,direction[:,None,:])
    state=dict(step=step,episode=episode,support=support.clone(),center=center.clone(),ground=ground.clone(),
               gyro=gyro.clone(),direction=direction.clone(),origin=origin,origin_ground=origin_ground,
               origin_dir=origin_dir,air=air,continuous=continuous,values=values)
    env._d2w_stair_reward_state=state
    den=gate.sum().clamp_min(1)
    METRICS.update({f'stair_{k}':v.sum().detach()/den for k,v in values.items()})
    return values


def stair_reward(env, asset_cfg, sensor_cfg, component):
    return stair_terms(env,asset_cfg,sensor_cfg)[component]


def configure(cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from d2w_rl_lab.tasks.mixed.terrain import MIXED_TERRAIN_CFG
    # Restore default weights from this checkout, not historical RobotLab constants.
    current=cfg.scene.terrain.terrain_generator.sub_terrains
    assert set(current)==set(MIXED_TERRAIN_CFG.sub_terrains)
    for name,term in current.items():term.proportion=MIXED_TERRAIN_CFG.sub_terrains[name].proportion
    cfg.rewards.track_lin_vel_xy_exp.func=track_linear
    cfg.rewards.track_lin_vel_xy_exp.params['asset_cfg']=SceneEntityCfg('robot')
    for name in ['fast_motion_stability','fast_straight_coordination']:
        getattr(cfg.rewards,name).func=flat_fast_preference
    wheels=['FL_foot','FR_foot','RL_foot','RR_foot']
    params={'asset_cfg':SceneEntityCfg('robot',body_names=wheels,preserve_order=True),
            'sensor_cfg':SceneEntityCfg('contact_forces',body_names=wheels,preserve_order=True)}
    for component,weight in [('touchdown',1.),('slip',-.15),('shock',-.10)]:
        setattr(cfg.rewards,'stair_'+component,RewardTermCfg(func=stair_reward,weight=weight,params=dict(params,component=component)))
    return cfg
