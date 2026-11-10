"""D2W-specific MDP terms."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from d2w_rl_lab.tasks.d2w.mdp_base import locomotion as locomotion_math
from d2w_rl_lab.tasks.d2w.mdp_base.locomotion import DEFAULT_MODE_SETTINGS
from d2w_rl_lab.assets.d2w_robstride_actuator import RobStrideDiagnosticSnapshot, load_motor_spec
from d2w_rl_lab.tasks.d2w.mdp_base.commands import ReferenceRLGymCommand


class D2WRLGymCommand(ReferenceRLGymCommand):
    """Go2-compatible command term that also supports a true plane terrain."""

    def _init_terrain_infos(self) -> None:
        terrain_generator = self._env.scene.terrain.cfg.terrain_generator
        if terrain_generator is not None:
            super()._init_terrain_infos()
            return
        self.terrain_types = ["flat"]
        self.terrain_type2idx = {"flat": 0}
        self.terrain_idxs = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_length = 0.0


    def _resample(self, env_ids) -> None:
        super()._resample(env_ids)
        if len(env_ids) == 0:
            return
        old = self.commands[env_ids].clone()
        flat_index = self.terrain_type2idx.get("flat", -2)
        selected_flat = self.terrain_idxs[env_ids] == flat_index
        ranges = {name: values[env_ids] for name, values in self.env_command_ranges.items()}
        new = locomotion_math.sample_flat_modes(old, ranges, selected_flat,
                                                self.cfg.flat_mode_probability, self.cfg.locomotion_settings)
        self.commands[env_ids] = new
        # The base sampler already added old; account exactly once for the replacement.
        self.commands_xy_accumulation[env_ids] += new[:, :2] - old[:, :2]
        changed = torch.any(new != old, dim=1)
        ids = torch.as_tensor(env_ids, device=self.device)
        self.last_is_limit_vel[ids[changed]] = False


@configclass
class D2WCommandCfg(CommandTermCfg):
    """Go2-derived command configuration with explicit D2W standing samples."""

    class_type: type = D2WRLGymCommand
    flat_mode_probability: float = 0.5
    locomotion_settings: dict = DEFAULT_MODE_SETTINGS.copy()
    asset_name: str = "robot"
    dynamic_resample_commands: bool = True
    limit_vel_invert_when_continuous: bool = True
    zero_command_curriculum: dict = {
        "start_iter": 0,
        "end_iter": 1500,
        "start_value": 0.1,
        "end_value": 0.2,
    }
    zero_command_reset_yaw: bool = True
    """Reset all three commands before the 50% turn-in-place mix within this group."""
    limit_vel: dict = {"lin_vel_x": [-1, 0, 1], "lin_vel_y": [-1, 0, 1], "ang_vel_yaw": [-1, 0, 1]}
    command_range_curriculum: list[dict] = [
        {
            "iter": 2000,
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-1.5, 1.5],
        },
        {
            "iter": 5000,
            "lin_vel_x": [-2.0, 2.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-2.0, 2.0],
        },
        {
            "iter": 10000,
            "lin_vel_x": [-2.0, 2.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-2.0, 2.0],
        },
    ]
    terrain_max_command_ranges: dict[str, dict] = {
        "wave": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "slope_up": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "slope_down": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "rough_slope": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "stairs_up": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "stairs_down": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "obstacles": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "stepping_stones": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "gap": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "flat": {"lin_vel_x": [-2.0, 2.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-2.0, 2.0]},
        "random_rough": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "hf_pyramid_slope": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "hf_pyramid_slope_inv": {"lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "pyramid_stairs": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "pyramid_stairs_inv": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
        "boxes": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-1.5, 1.5]},
    }
    resampling_time: float = 5.0
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    limit_ang_vel_at_zero_command_prob: float = 0.5
    limit_vel_prob: float = 0.2
    num_steps_per_iter: int = 24

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = [-0.5, 0.5]
        lin_vel_y: tuple[float, float] = [-0.5, 0.5]
        ang_vel_yaw: tuple[float, float] = [-1.0, 1.0]

    ranges: Ranges = Ranges()
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )


def reset_joints_by_offset(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset selected joints with a small additive perturbation.

    Go2's multiplicative 0.5--1.5 reset is inappropriate for D2W because it
    magnifies the non-zero crouched pose.  Additive noise has a direct angular
    interpretation and is clamped to the articulation's soft limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.default_joint_pos.torch[env_ids][:, joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[env_ids][:, joint_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(*position_range)
    joint_vel += torch.empty_like(joint_vel).uniform_(*velocity_range)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids][:, joint_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def motor_torque_feasibility_values(
    requested: torch.Tensor,
    applied: torch.Tensor,
    peak_torque_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized excess, clipped-joint ratio, and max excess in N.m."""
    excess_nm = torch.relu(torch.abs(requested) - torch.abs(applied))
    normalized_excess = excess_nm / float(peak_torque_nm)
    clip_ratio = torch.mean((excess_nm > 0.0).to(requested.dtype), dim=-1)
    max_exceed_nm = torch.amax(excess_nm, dim=-1)
    return normalized_excess, clip_ratio, max_exceed_nm


def motor_torque_feasibility(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize RS06 envelope exceedance normalized by its output peak torque."""
    asset = env.scene[asset_cfg.name]
    requested = asset.actuators.computed_effort.torch[:, asset_cfg.joint_ids]
    applied = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    normalized_excess, _, _ = motor_torque_feasibility_values(
        requested,
        applied,
        load_motor_spec().peak_torque_nm,
    )
    return torch.sum(normalized_excess, dim=1)


def motor_torque_clip_ratio(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the instantaneous fraction of joints clipped by the motor envelope."""
    asset = env.scene[asset_cfg.name]
    requested = asset.actuators.computed_effort.torch[:, asset_cfg.joint_ids]
    applied = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    _, clip_ratio, _ = motor_torque_feasibility_values(requested, applied, load_motor_spec().peak_torque_nm)
    return clip_ratio


def motor_torque_max_exceed_nm(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return instantaneous maximum requested torque above the envelope in N.m."""
    asset = env.scene[asset_cfg.name]
    requested = asset.actuators.computed_effort.torch[:, asset_cfg.joint_ids]
    applied = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    _, _, maximum = motor_torque_feasibility_values(requested, applied, load_motor_spec().peak_torque_nm)
    return maximum


def rs06_episode_diagnostics(
    env,
    env_ids: torch.Tensor | slice | None = None,
    *,
    reset: bool = False,
    actuator_name: str = "RS06",
) -> RobStrideDiagnosticSnapshot:
    """Read cumulative RS06 diagnostics for episode logging/reset hooks."""
    actuator = env.scene["robot"].actuators[actuator_name]
    return actuator.diagnostics(env_ids, reset=reset)


def motor_overload(env, actuator_name: str = "RS06") -> torch.Tensor:
    """Terminate environments latched by the RS06 hard-safety monitor."""

    actuator = env.scene["robot"].actuators[actuator_name]
    overload_tripped = getattr(actuator, "overload_tripped", None)
    if overload_tripped is None:
        raise RuntimeError(
            f"D2W actuator {actuator_name!r} does not expose overload_tripped; "
            "the RS06 hard-safety contract is incomplete."
        )
    if not isinstance(overload_tripped, torch.Tensor):
        raise RuntimeError("RS06 overload_tripped must be a torch.Tensor.")
    if overload_tripped.ndim == 2:
        overload_tripped = torch.any(overload_tripped, dim=-1)
    if overload_tripped.ndim != 1 or overload_tripped.shape[0] != env.num_envs:
        raise RuntimeError(
            "RS06 overload_tripped must provide one value per environment; "
            f"got shape={tuple(overload_tripped.shape)}, num_envs={env.num_envs}."
        )
    return overload_tripped.to(dtype=torch.bool)


# D2 action/observation interface adapted only for the four continuous joints.
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from d2w_rl_lab.assets.d2w_metadata import D2W_CANONICAL_JOINT_ORDER, D2W_WHEEL_INDICES
from d2w_rl_lab.assets.d2w_wheel_motor import load_wheel_motor_spec

class D2WHybridAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if tuple(self._joint_names) != D2W_CANONICAL_JOINT_ORDER:
            raise ValueError("D2W hybrid action joint order differs from the asset contract")
        self._wheel_columns = list(D2W_WHEEL_INDICES)
        self._wheel_speed_limit = load_wheel_motor_spec().target_speed_rad_s

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions[:, self._wheel_columns] = (self._raw_actions[:, self._wheel_columns] * self._scale[:, self._wheel_columns]).clamp(-self._wheel_speed_limit, self._wheel_speed_limit)

    def apply_actions(self):
        positions = self._processed_actions.clone()
        velocities = torch.zeros_like(positions)
        velocities[:, self._wheel_columns] = positions[:, self._wheel_columns]
        positions[:, self._wheel_columns] = 0.
        self._asset.actuators.target_command.set_position_index(value=positions, joint_ids=self._joint_ids)
        self._asset.actuators.target_command.set_velocity_index(value=velocities, joint_ids=self._joint_ids)

@configclass
class D2WHybridActionCfg(JointPositionActionCfg):
    class_type: type = D2WHybridAction

def joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    values = (asset.data.joint_pos.torch - asset.data.default_joint_pos.torch).clone()
    ids, _ = asset.find_joints(".*_foot_joint")
    values[:, ids] = 0.
    return values[:, asset_cfg.joint_ids]



def _locomotion_inputs(env, command_name, sensor_cfg, height_cfg, settings):
    sensor = env.scene.sensors[sensor_cfg.name]
    # Current contact avoids rewarding a touchdown while a different foot's old
    # contact is still present in the force-history buffer.
    contact = (sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > settings["contact_force"]).float()
    heights = env.scene.sensors[height_cfg.name].data.ray_hits_w.torch[..., 2]
    command = env.command_manager.get_command(command_name)
    weights = locomotion_math.mode_weights(command, heights, settings)
    return command, sensor, contact, weights


def low_turn_rolling(env, command_name, sensor_cfg, height_cfg, asset_cfg, settings):
    command, _, contact, weights = _locomotion_inputs(env, command_name, sensor_cfg, height_cfg, settings)
    robot = env.scene[asset_cfg.name]
    # Use collision-cylinder centres, not link origins. All four axes are +Y.
    offsets = torch.zeros_like(robot.data.body_pos_w.torch[:, asset_cfg.body_ids])
    offsets[:, :, 1] = offsets.new_tensor([1., -1., 1., -1.]) * settings["wheel_center_offset_y"]
    centres = robot.data.body_pos_w.torch[:, asset_cfg.body_ids] + quat_apply(robot.data.body_quat_w.torch[:, asset_cfg.body_ids], offsets)
    root_quat = robot.data.root_quat_w.torch[:, None, :].expand(-1, 4, -1)
    wheel_y = quat_apply_inverse(root_quat, centres - robot.data.root_pos_w.torch[:, None, :])[:, :, 1]
    score = locomotion_math.rolling_score(command, contact, robot.data.joint_vel.torch[:, asset_cfg.joint_ids], wheel_y, settings)
    return weights[0] * score


def fast_turn_steps(env, command_name, sensor_cfg, height_cfg, settings, asset_cfg=SceneEntityCfg("robot")):
    command, sensor, contact, weights = _locomotion_inputs(env, command_name, sensor_cfg, height_cfg, settings)
    robot = env.scene[asset_cfg.name]
    score = locomotion_math.turn_step_score(contact,
        sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids],
        sensor.data.last_air_time.torch[:, sensor_cfg.body_ids], sensor.data.current_air_time.torch[:, sensor_cfg.body_ids],
        robot.data.root_ang_vel_b.torch[:, 2] - command[:, 2], robot.data.projected_gravity_b.torch[:, 2], settings)
    return weights[1] * score


def fast_motion_stability(env, command_name, sensor_cfg, height_cfg, settings, asset_cfg=SceneEntityCfg("robot")):
    _, _, _, weights = _locomotion_inputs(env, command_name, sensor_cfg, height_cfg, settings)
    robot = env.scene[asset_cfg.name]
    return weights[3] * locomotion_math.stability_cost(robot.data.projected_gravity_b.torch,
        robot.data.root_ang_vel_b.torch, robot.data.root_lin_vel_b.torch[:, 2], settings)


def fast_straight_coordination(env, command_name, sensor_cfg, height_cfg, asset_cfg, settings):
    _, _, contact, weights = _locomotion_inputs(env, command_name, sensor_cfg, height_cfg, settings)
    robot = env.scene[asset_cfg.name]
    deviation = robot.data.joint_pos.torch[:, asset_cfg.joint_ids] - robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return weights[2] * locomotion_math.straight_coordination_cost(deviation, contact, settings)


def quiet_stability(env, command_name, asset_cfg):
    robot=env.scene[asset_cfg.name]
    return locomotion_math.quiet_cost(env.command_manager.get_command(command_name),
        robot.data.root_lin_vel_b.torch,robot.data.root_ang_vel_b.torch,robot.data.joint_vel.torch[:,asset_cfg.joint_ids])


def front_stance(env,command_name,sensor_cfg,height_cfg,asset_cfg,settings):
    command,_,contact,_ = _locomotion_inputs(env,command_name,sensor_cfg,height_cfg,settings)
    robot=env.scene[asset_cfg.name]
    heights=env.scene.sensors[height_cfg.name].data.ray_hits_w.torch[...,2]
    deviation=robot.data.joint_pos.torch[:,asset_cfg.joint_ids]-robot.data.default_joint_pos.torch[:,asset_cfg.joint_ids]
    return locomotion_math.front_stance_cost(command,heights,deviation,contact,settings)


def turn_clearance(env,command_name,sensor_cfg,height_cfg,asset_cfg,settings,half_width):
    command,sensor,contact,_ = _locomotion_inputs(env,command_name,sensor_cfg,height_cfg,settings)
    robot=env.scene[asset_cfg.name]
    position=robot.data.body_pos_w.torch[:,asset_cfg.body_ids];rotation=robot.data.body_quat_w.torch[:,asset_cfg.body_ids]
    offset=torch.zeros_like(position);offset[:,:,1]=position.new_tensor([1.,-1.,1.,-1.])*settings['wheel_center_offset_y']
    centres=position+quat_apply(rotation,offset)
    axis=torch.zeros_like(position);axis[:,:,1]=1.
    axis_z=quat_apply(rotation,axis)[:,:,2].abs().clamp(0,1)
    extent=settings['wheel_radius']*torch.sqrt((1.-axis_z.square()).clamp_min(0))+half_width*axis_z
    rays=env.scene.sensors[height_cfg.name].data.ray_hits_w.torch
    finite=torch.isfinite(rays).all(dim=2)
    safe=torch.where(torch.isfinite(rays),rays,torch.zeros_like(rays))
    dist=(centres[:,:,None,:2]-safe[:,None,:,:2]).square().sum(dim=-1)
    dist=dist.masked_fill(~finite[:,None,:],float('inf'))
    distance,index=dist.min(dim=2)
    ground=safe[:,:,2].gather(1,index)
    clearance=centres[:,:,2]-extent-ground
    return locomotion_math.swing_clearance_score(command,clearance,distance<.15**2,contact,
        sensor.data.current_air_time.torch[:,sensor_cfg.body_ids],robot.data.projected_gravity_b.torch,settings)
