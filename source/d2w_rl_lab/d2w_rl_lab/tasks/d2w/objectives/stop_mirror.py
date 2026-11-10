"""Flat zero-command pair balancing, within the existing residual cost budget."""
import torch
from .stand_balance import mirror_pairs
from .reward_coherent import flat_gate
from .stationary_hold import motion_cost, braking_blend, zero_mask
from .standing_finetune import METRICS


def pair_balance_cost(pair_sq):
    # Same mirrored optimum for front and rear. A bad pair cannot disappear
    # into a mean over both pairs; do not force an absolute joint position.
    scaled = pair_sq / .05**2
    bounded = scaled / (1.0 + scaled)
    return .5 * (bounded.mean(-1) + bounded.amax(-1))


def hold_residual(env, asset_cfg, wheels_cfg, sensor_cfg, component='residual'):
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command('base_velocity')
    term = env.command_manager.get_term('base_velocity')
    gate = flat_gate(env) * zero_mask(command) * braking_blend(term.quiet_age)
    deviation = (robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
                 - robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids])
    pair_cost = pair_balance_cost(mirror_pairs(deviation))
    motion = motion_cost(robot.data.root_lin_vel_b.torch[:, :2],
                         robot.data.root_ang_vel_b.torch[:, 2])
    METRICS['stop_pair_balance_cost'] = (gate * pair_cost).sum().detach() / gate.sum().clamp_min(1)
    return gate * (motion + pair_cost)


def configure(cfg):
    cfg.rewards.standing_balance_residual.func = hold_residual
    assert cfg.rewards.standing_balance_residual.weight == -.5
    from ..stop_mirror_command import StopMirrorCommand
    cfg.commands.base_velocity.class_type = StopMirrorCommand
    return cfg
