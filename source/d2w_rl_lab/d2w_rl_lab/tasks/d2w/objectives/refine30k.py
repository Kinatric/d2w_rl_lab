"""Target slow hold drift and correctly scaled stair events; D2W only."""
import torch
from .reward_coherent import flat_gate
from .stationary_hold import braking_blend,zero_mask
from .stop_mirror import pair_balance_cost
from .stand_balance import mirror_pairs
from .speed_stairs import stair_terms


def precise_hold_motion(velocity_xy,yaw_rate):
    # Same two bounded components and zero optimum; improve small-drift resolution.
    xy=velocity_xy.square().sum(-1)/.02**2
    yaw=yaw_rate.square()/.15**2
    return xy/(1+xy)+yaw/(1+yaw)


def hold_residual(env,asset_cfg,wheels_cfg,sensor_cfg,component='residual'):
    robot=env.scene[asset_cfg.name];command=env.command_manager.get_command('base_velocity')
    gate=flat_gate(env)*zero_mask(command)*braking_blend(env.command_manager.get_term('base_velocity').quiet_age)
    deviation=robot.data.joint_pos.torch[:,asset_cfg.joint_ids]-robot.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    return gate*(precise_hold_motion(robot.data.root_lin_vel_b.torch[:,:2],robot.data.root_ang_vel_b.torch[:,2])+pair_balance_cost(mirror_pairs(deviation)))


def touchdown_rate(env,asset_cfg,sensor_cfg,component='touchdown'):
    assert component=='touchdown'
    # RewardManager multiplies rates by step_dt. Each qualifying wheel earns .05
    # once, independent of the control timestep; no continuous airborne payout.
    return stair_terms(env,asset_cfg,sensor_cfg)['touchdown']/env.step_dt


def configure(cfg):
    cfg.rewards.standing_balance_residual.func=hold_residual
    assert cfg.rewards.standing_balance_residual.weight==-.5
    cfg.rewards.stair_touchdown.func=touchdown_rate
    cfg.rewards.stair_touchdown.weight=.20
    return cfg
