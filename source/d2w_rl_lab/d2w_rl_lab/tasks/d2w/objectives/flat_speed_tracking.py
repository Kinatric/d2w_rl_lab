"""Strengthen only the existing flat straight-speed objective, without overspeed pay.

An additional bounded error cost inside the same tracking term supplies a
non-exponential tail and strengthens the observed 0.10--0.25 m/s deficits.
The 0.5 coefficient is an initial shaping choice, not a measured optimal weight.
"""
import torch


def tracking_adjustment(command, velocity, flat):
    from .terrain_alignment import smooth
    straight = (1.-smooth(command[:, 1].abs(), .02, .10))
    straight *= 1.-smooth(command[:, 2].abs(), .02, .15)
    gate = flat * straight * smooth(command[:, 0].abs(), .2, 1.)
    error = (command[:, :2]-velocity[:, :2]).square().sum(-1)
    return -.5*gate*error/(.35**2+error)


def track_linear(env, std, command_name, asset_cfg):
    from .speed_stairs import track_linear as previous_tracking
    from .reward_coherent import flat_gate
    command = env.command_manager.get_command(command_name)
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b.torch
    return previous_tracking(env, std, command_name, asset_cfg) + tracking_adjustment(
        command, velocity, flat_gate(env))


def configure(cfg):
    cfg.rewards.track_lin_vel_xy_exp.func = track_linear
    return cfg
