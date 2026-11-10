"""D2W locomotion reward math; Torch only, no simulator or D2 dependency.

Modes use commands rather than achieved speed, so overspeed cannot turn off a
penalty. All new posture/contact preferences fade out on non-flat local ground.
These are learning rewards, not action clamps, gait controllers or terminations.
"""
from __future__ import annotations
import torch

DEFAULT_MODE_SETTINGS = {
    "linear_slow": 0.5, "linear_fast": 1.2,
    "yaw_slow": 0.6, "yaw_fast": 1.0,
    "turn_start": 0.15, "turn_full": 0.35,
    "straight_yaw_start": 0.1, "straight_yaw_end": 0.3,
    "straight_lateral_start": 0.05, "straight_lateral_end": 0.2,
    "flat_height_start": 0.04, "flat_height_end": 0.08,
    "contact_force": 2.0,
    "wheel_radius": 0.0813, "wheel_center_offset_y": 0.0455,
    "rolling_std": 0.25,
    "air_time_min": 0.06, "air_time_peak": 0.15, "air_time_max": 0.30,
    "yaw_tracking_std": 0.5,
    "tilt_scale": 4.0, "angular_scale": 0.25, "vertical_scale": 1.0,
    "stance_pose_scale": 1.0, "support_balance_scale": 0.25,
}


def smooth(value, start, end):
    x = ((value - start) / (end - start)).clamp(0.0, 1.0)
    return x.square() * (3.0 - 2.0 * x)


def mode_weights(command, heights, settings):
    s = settings
    linear = smooth(command[:, :2].norm(dim=1), s["linear_slow"], s["linear_fast"])
    yaw = command[:, 2].abs()
    angular = smooth(yaw, s["yaw_slow"], s["yaw_fast"])
    high = 1.0 - (1.0 - linear) * (1.0 - angular)
    turn = smooth(yaw, s["turn_start"], s["turn_full"])
    straight = (1.0 - smooth(yaw, s["straight_yaw_start"], s["straight_yaw_end"]))
    straight *= 1.0 - smooth(command[:, 1].abs(), s["straight_lateral_start"], s["straight_lateral_end"])
    rolling_compatible = 1.0 - smooth(command[:, 1].abs(), s["straight_lateral_start"], s["straight_lateral_end"])
    valid = torch.isfinite(heights).all(dim=1)
    safe = torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))
    span = safe.max(dim=1).values - safe.min(dim=1).values
    flat = (1.0 - smooth(span, s["flat_height_start"], s["flat_height_end"])) * valid
    return (1.0 - high) * turn * flat * rolling_compatible, high * turn * flat, linear * straight * flat, high * flat


def rolling_score(command, contact, wheel_speed, wheel_y, settings):
    # Positive yaw: left wheels slower, right wheels faster. Wheel axes all +Y.
    target = command[:, :1] - command[:, 2:3] * wheel_y
    error = (settings["wheel_radius"] * wheel_speed - target) / settings["rolling_std"]
    match = (torch.exp(-error.square()) * contact).mean(dim=1)
    return 0.5 * match - (1.0 - contact).mean(dim=1)


def turn_step_score(contact, first_contact, last_air, current_air, yaw_error, gravity_z, settings):
    s = settings
    # A short completed swing earns one event, never reward for hovering.
    quality = torch.minimum((last_air - s["air_time_min"]) / (s["air_time_peak"] - s["air_time_min"]),
                            (s["air_time_max"] - last_air) / (s["air_time_max"] - s["air_time_peak"])).clamp(0, 1)
    supported = (contact.sum(dim=1) >= 2).to(last_air.dtype)
    upright = (-gravity_z).clamp(0, 1)
    tracking = torch.exp(-(yaw_error / s["yaw_tracking_std"]).square())
    touchdown = (quality * first_contact).mean(dim=1) * supported * upright * tracking
    unsupported = ((2.0 - contact.sum(dim=1)) / 2.0).clamp_min(0)
    prolonged_air = ((current_air - s["air_time_max"]) / s["air_time_max"]).clamp(0, 1).mean(dim=1)
    return touchdown - unsupported - prolonged_air


def stability_cost(gravity, angular_velocity, vertical_velocity, settings):
    return (settings["tilt_scale"] * gravity[:, :2].square().sum(dim=1)
            + settings["angular_scale"] * angular_velocity[:, :2].square().sum(dim=1)
            + settings["vertical_scale"] * vertical_velocity.square())


def straight_coordination_cost(joint_deviation, contact, settings):
    # FL, FR, RL, RR x hip, thigh, calf. Only compare a pair while both
    # wheels support the robot; never impose identical swing phases or front/rear angles.
    pose = joint_deviation.reshape(-1, 4, 3)
    mirror = pose.new_tensor([-1.0, 1.0, 1.0])
    pair_error = (pose[:, [0, 2]] - pose[:, [1, 3]] * mirror).square().mean(dim=2)
    pair_support = contact[:, [0, 2]] * contact[:, [1, 3]]
    imbalance = (contact[:, [0, 2]].sum(dim=1) - contact[:, [1, 3]].sum(dim=1)) / 2.0
    return settings["stance_pose_scale"] * (pair_error * pair_support).mean(dim=1) + settings["support_balance_scale"] * imbalance.square()


def sample_flat_modes(commands, ranges, eligible, probability, settings):
    """Replace a configurable share of moving flat samples, preserving zeros/bounds.

    Moving samples: 30% slow turn, 30% fast turn, 40% straight; balanced forward/reverse.
    An independent 20% of selected samples teach a complete stop.
    Returns new commands; the manager accounts for changed accumulated distance.
    """
    result = commands.clone()
    n = len(commands); device = commands.device
    selected = eligible & (commands.abs().sum(dim=1) > 0) & (torch.rand(n, device=device) < probability)
    draw = torch.rand(n, device=device)
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    yaw_limit = torch.minimum(-ranges["ang_vel_yaw"][:, 0], ranges["ang_vel_yaw"][:, 1]).clamp_min(0)
    reverse_limit = (-ranges["lin_vel_x"][:, 0]).clamp_min(0)
    positive_limit = ranges["lin_vel_x"][:, 1].clamp_min(0)
    reverse = ((torch.rand(n, device=device) < .5) | (positive_limit == 0)) & (reverse_limit > 0)
    forward_limit = torch.where(reverse, reverse_limit, positive_limit)
    slow = selected & (draw < .3)
    fast = selected & (draw >= .3) & (draw < .6)
    straight = selected & (draw >= .6)
    u = torch.rand(n, device=device); v = torch.rand(n, device=device)
    result[slow, 0] = u[slow] * forward_limit[slow].clamp(max=settings["linear_slow"])
    slow_in_place = slow & (torch.rand(n, device=device) < .5)
    result[slow_in_place, 0] = 0
    slow_yaw_max = yaw_limit.clamp(max=settings["yaw_slow"])
    result[slow, 2] = (sign * slow_yaw_max * (.35 + .65 * v))[slow]
    moving_turn = fast & (torch.rand(n, device=device) < .5)
    in_place = fast & ~moving_turn
    fast_min = forward_limit.clamp(max=settings["linear_fast"])
    result[moving_turn | straight, 0] = (fast_min + u * (forward_limit - fast_min))[moving_turn | straight]
    turn_min = yaw_limit.clamp(max=settings["turn_full"])
    result[moving_turn, 2] = (sign * (turn_min + v * (yaw_limit.clamp(max=settings["yaw_fast"]) - turn_min)))[moving_turn]
    result[in_place, 0] = 0
    yaw_min = yaw_limit.clamp(max=settings["yaw_fast"])
    result[in_place, 2] = (sign * (yaw_min + v * (yaw_limit - yaw_min)))[in_place]
    result[straight, 2] = 0
    result[selected, 1] = 0
    result[selected, 0] *= torch.where(reverse[selected], -1., 1.)
    stopping = selected & (torch.rand(n, device=device) < .2)
    result[stopping] = 0.
    # Retain curriculum and per-terrain bounds, including asymmetric test ranges.
    for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
        result[selected, axis] = result[selected, axis].clamp(ranges[name][selected, 0], ranges[name][selected, 1])
    return result


def quiet_cost(command, velocity, gyro, leg_velocity):
    gate = (1.-smooth(command[:,:2].norm(dim=1), .02, .08)) * (1.-smooth(command[:,2].abs(), .02, .08))
    error = velocity[:,:2].square().sum(dim=1)/.04**2 + gyro[:,2].square()/.10**2
    # Bounded so large initial errors cannot dominate motion learning. No action filtering.
    return gate * (error/(1.+error) + .05*torch.tanh(leg_velocity.square().mean(dim=1)))


def front_stance_cost(command, heights, deviation, contact, settings):
    moving = smooth(command[:,:2].norm(dim=1), .10, .50)
    straight = 1.-smooth(command[:,2].abs(), .10, .30)
    straight *= 1.-smooth(command[:,1].abs(), .05, .20)
    valid = torch.isfinite(heights).all(dim=1)
    safe = torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))
    flat = (1.-smooth(safe.max(dim=1).values-safe.min(dim=1).values,.04,.08))*valid
    front = deviation.reshape(-1,4,3)[:,:2,0]
    excess = ((front.abs()-.08).clamp_min(0)/.15).square().mean(dim=1)
    return moving*straight*flat*contact[:,0]*contact[:,1]*excess


def swing_clearance_score(command, clearance, valid, contact, air, gravity, settings):
    _,turn,_,_ = mode_weights(command, torch.zeros(len(command),1,device=command.device), settings)
    # Local wheel-to-ground clearance, not body height; no hovering or all-feet jumping reward.
    quality = (clearance/.06).clamp(0,1) * (1.-smooth(clearance,.10,.18))
    short = 1.-smooth(air,.15,.30)
    swing = (1.-contact)*valid*(air>0.02)*short*quality
    support = (contact.sum(dim=1)>=2)*(-gravity[:,2]).clamp(0,1)
    return turn*support*swing.mean(dim=1)
