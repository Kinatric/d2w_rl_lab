"""External D2W standing reward experiment. No action/actuator modifications."""
import torch

METRICS = {}

def smooth(x, a, b):
    t = ((x-a)/(b-a)).clamp(0, 1)
    return t.square()*(3-2*t)

def components(command, heights, gravity, velocity, gyro, deviation):
    quiet = (1-smooth(command[:, :2].norm(dim=1), .02, .08))*(1-smooth(command[:, 2].abs(), .02, .08))
    valid = torch.isfinite(heights).all(dim=1)
    safe = torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))
    flat = valid*(1-smooth(safe.max(dim=1).values-safe.min(dim=1).values, .04, .08))
    gate = quiet*flat
    pose = deviation.reshape(-1, 4, 3)
    mirror = (pose[:, [0, 2]]-pose[:, [1, 3]]*pose.new_tensor([-1., 1., 1.])).square().mean(dim=(1, 2))
    tilt = gravity[:, :2].square().sum(dim=1)
    drift = velocity[:, :2].square().sum(dim=1)/.2**2 + gyro[:, 2].square()/.5**2
    return gate, tilt, mirror, drift

def standing_cost(env, asset_cfg, component):
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command('base_velocity')
    heights = env.scene.sensors['height_scanner'].data.ray_hits_w.torch[..., 2]
    deviation = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]-robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    gate, tilt, mirror, drift = components(command, heights, robot.data.projected_gravity_b.torch,
        robot.data.root_lin_vel_b.torch, robot.data.root_ang_vel_b.torch, deviation)
    if component == 'tilt':
        denom = gate.sum().clamp_min(1)
        METRICS.update({
            'quiet_flat_fraction': gate.mean().detach(),
            'quiet_flat_weight_sum': gate.sum().detach(),
            'tilt_deg': (torch.acos((-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1, 1))*180/torch.pi*gate).sum().detach()/denom,
            'mirror_rms_rad': torch.sqrt((mirror*gate).sum()/denom).detach(),
            'drift_speed_m_s': (robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)*gate).sum().detach()/denom,
            'yaw_speed_rad_s': (robot.data.root_ang_vel_b.torch[:, 2].abs()*gate).sum().detach()/denom,
        })
    return gate*{'tilt': tilt, 'mirror': mirror, 'drift': drift}[component]

def configure(cfg):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from d2w_rl_lab.assets.d2w import D2W_LEG_JOINT_ORDER
    for name, weight in [('tilt', -8.), ('mirror', -2.), ('drift', -1.)]:
        setattr(cfg.rewards, 'standing_'+name, RewardTermCfg(func=standing_cost, weight=weight,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=list(D2W_LEG_JOINT_ORDER), preserve_order=True), 'component': name}))
    return cfg

def self_test():
    command = torch.zeros(7, 3); heights = torch.zeros(7, 6)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(7, 1)
    velocity = torch.zeros(7, 3); gyro = torch.zeros(7, 3); deviation = torch.zeros(7, 12)
    gravity[1] = torch.tensor([.286, 0., -.95823]); deviation[2, 6] = -.6
    velocity[3, 0] = .1; command[4, 0] = .1; command[5, 2] = .1; heights[6, 0] = .1
    gate, tilt, mirror, drift = components(command, heights, gravity, velocity, gyro, deviation)
    assert torch.equal(gate, torch.tensor([1., 1., 1., 1., 0., 0., 0.]))
    assert tilt[0] == mirror[0] == drift[0] == 0
    assert tilt[1] > 0 and mirror[2] > 0 and drift[3] > 0
    d = torch.tensor([[.1,.2,.3,-.1,.2,.3,.2,.4,.6,-.2,.4,.6]])
    assert components(command[:1], heights[:1], gravity[:1], velocity[:1], gyro[:1], d)[2].item() == 0
    heights[0, 0] = float('nan')
    assert components(command, heights, gravity, velocity, gyro, deviation)[0][0] == 0
    return {'status':'PASS', 'checks':['ideal_zero','tilt_cost','rear_asymmetry_cost','drift_cost','moving_and_turning_disabled','rough_and_invalid_ground_disabled','mirrored_pose_zero'],
        'learning_updates':0, 'scope':'Reward math only; not policy improvement or feedback stability validation.'}
