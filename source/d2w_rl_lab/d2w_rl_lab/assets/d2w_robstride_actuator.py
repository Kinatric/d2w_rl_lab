"""RobStride actuators driven by the canonical motor YAML model.

Values in :mod:`resources/d2w_description/config/robstride06_motor.yaml` use the integrated actuator
output reference. The provisional armature is an unidentified simulation
estimate, applied once on the output side without an additional 9:1 conversion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

from .d2w_robstride_motor import INSTANTANEOUS_ENVELOPE, RobStrideMotorSpec, load_motor_spec as _load_motor_spec


ROBSTRIDE06_MOTOR_CONFIG = (
    Path(__file__).resolve().parents[4] / "resources" / "d2w_description" / "config" / "robstride06_motor.yaml"
)


class RobStrideDiagnosticSnapshot(NamedTuple):
    """Cumulative per-environment diagnostics since the last reset."""

    requested_count: torch.Tensor
    clipped_count: torch.Tensor
    clip_ratio: torch.Tensor
    max_exceed_nm: torch.Tensor
    max_overload_damage: torch.Tensor
    hard_safety_joint_count: torch.Tensor


def load_motor_spec(config_path: str | Path = ROBSTRIDE06_MOTOR_CONFIG) -> RobStrideMotorSpec:
    """Load the canonical 48 V output-side RS06 contract."""

    return _load_motor_spec(config_path)


def interpolate_torque_limits(
    joint_velocity: torch.Tensor,
    velocity_points: torch.Tensor,
    torque_points: torch.Tensor,
) -> torch.Tensor:
    """Interpolate the symmetric output torque limit at each absolute joint speed."""
    speed = joint_velocity.abs()
    upper = torch.bucketize(speed.contiguous(), velocity_points)
    lower = (upper - 1).clamp(0, velocity_points.numel() - 1)
    upper = upper.clamp(0, velocity_points.numel() - 1)

    velocity_lower = velocity_points[lower]
    velocity_upper = velocity_points[upper]
    torque_lower = torque_points[lower]
    torque_upper = torque_points[upper]
    fraction = torch.where(
        velocity_upper > velocity_lower,
        (speed - velocity_lower) / (velocity_upper - velocity_lower),
        torch.zeros_like(speed),
    )
    interpolated = torque_lower + fraction * (torque_upper - torque_lower)
    return torch.where(speed > velocity_points[-1], torch.zeros_like(interpolated), interpolated).clamp(min=0.0)


def interpolate_continuous_torque_limits(
    joint_velocity: torch.Tensor,
    peak_limits: torch.Tensor,
    velocity_points: torch.Tensor,
    torque_points: torch.Tensor,
    *,
    switch_speed_rad_s: float,
    stall_continuous_torque_nm: float,
) -> torch.Tensor:
    """Apply engineering speed derating and the configured low-speed assumption."""

    speed = joint_velocity.abs()
    upper = torch.bucketize(speed.contiguous(), velocity_points)
    lower = (upper - 1).clamp(0, velocity_points.numel() - 1)
    upper = upper.clamp(0, velocity_points.numel() - 1)
    velocity_lower = velocity_points[lower]
    velocity_upper = velocity_points[upper]
    torque_lower = torque_points[lower]
    torque_upper = torque_points[upper]
    fraction = torch.where(
        velocity_upper > velocity_lower,
        (speed - velocity_lower) / (velocity_upper - velocity_lower),
        torch.zeros_like(speed),
    )
    rotating = torque_lower + fraction * (torque_upper - torque_lower)
    rotating = torch.where(speed > velocity_points[-1], torque_points[-1], rotating)
    thermal = torch.where(
        speed < float(switch_speed_rad_s),
        torch.full_like(speed, float(stall_continuous_torque_nm)),
        rotating,
    )
    return torch.minimum(thermal, peak_limits).clamp(min=0.0)


def _damage_rate_from_table(
    delivered_effort: torch.Tensor,
    continuous_effort: torch.Tensor,
    torque_points: torch.Tensor,
    inverse_duration_points: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear inverse-duration interpolation from a continuous anchor."""

    rate = torch.zeros_like(delivered_effort)
    assigned = delivered_effort <= continuous_effort
    previous_torque = continuous_effort
    previous_rate = torch.zeros_like(delivered_effort)
    for index in range(torque_points.numel()):
        torque = torque_points[index]
        next_rate = inverse_duration_points[index]
        in_interval = (~assigned) & (delivered_effort <= torque)
        denominator = torch.clamp(torque - previous_torque, min=torch.finfo(delivered_effort.dtype).eps)
        fraction = torch.clamp((delivered_effort - previous_torque) / denominator, min=0.0, max=1.0)
        interpolated = previous_rate + fraction * (next_rate - previous_rate)
        rate = torch.where(in_interval, interpolated, rate)
        assigned = assigned | in_interval
        previous_torque = torch.ones_like(delivered_effort) * torque
        previous_rate = torch.ones_like(delivered_effort) * next_rate
    return torch.where(assigned, rate, previous_rate)


def overload_damage_rates(
    joint_velocity: torch.Tensor,
    delivered_effort: torch.Tensor,
    continuous_effort: torch.Tensor,
    *,
    switch_speed_rad_s: float,
    stall_torque_points: torch.Tensor,
    stall_inverse_duration_points: torch.Tensor,
    rotating_torque_points: torch.Tensor,
    rotating_inverse_duration_points: torch.Tensor,
) -> torch.Tensor:
    """Return per-joint irreversible overload damage rate in inverse seconds."""

    stall_rate = _damage_rate_from_table(
        delivered_effort, continuous_effort, stall_torque_points, stall_inverse_duration_points
    )
    rotating_rate = _damage_rate_from_table(
        delivered_effort, continuous_effort, rotating_torque_points, rotating_inverse_duration_points
    )
    return torch.where(joint_velocity.abs() < float(switch_speed_rad_s), stall_rate, rotating_rate)


def torque_feasibility_metrics(
    requested_effort: torch.Tensor,
    available_effort: torch.Tensor,
    peak_torque_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized joint exceedance, clipped-joint ratio, and maximum exceedance.

    The first tensor retains one value per joint so a reward term can choose a
    sum or mean.  The other two tensors are reduced per environment and are
    intended for training diagnostics.
    """
    excess_nm = torch.relu(torch.abs(requested_effort) - available_effort)
    normalized_excess = excess_nm / float(peak_torque_nm)
    clip_ratio = torch.mean((excess_nm > 0.0).to(requested_effort.dtype), dim=-1)
    max_excess_nm = torch.amax(excess_nm, dim=-1)
    return normalized_excess, clip_ratio, max_excess_nm


class RobStrideActuator(DelayedPDActuator):
    """Delayed PD actuator limited by an arbitrary output-side torque-speed curve."""

    cfg: "RobStrideActuatorCfg"

    @property
    def overload_tripped(self) -> torch.Tensor:
        """Per-joint hard-safety latch consumed by D2W termination logic."""

        return self.hard_safety_active.clone()

    def __init__(self, cfg: "RobStrideActuatorCfg", *args, **kwargs):
        spec = load_motor_spec(cfg.motor_config_path)

        super().__init__(cfg, *args, **kwargs)
        self.motor_spec = spec
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._velocity_points = torch.tensor(spec.velocity_points, dtype=torch.float32, device=self._device)
        self._torque_points = torch.tensor(spec.torque_points, dtype=torch.float32, device=self._device)
        self._continuous_velocity_points = torch.tensor(
            spec.continuous_velocity_points, dtype=torch.float32, device=self._device
        )
        self._continuous_torque_points = torch.tensor(
            spec.continuous_torque_points, dtype=torch.float32, device=self._device
        )

        def finite_damage_nodes(table):
            nodes = [
                (torque, 1.0 / duration)
                for torque, duration in zip(table.torque_points_nm, table.duration_points_s)
                if math.isfinite(duration)
            ]
            return (
                torch.tensor([node[0] for node in nodes], dtype=torch.float32, device=self._device),
                torch.tensor([node[1] for node in nodes], dtype=torch.float32, device=self._device),
            )

        self._stall_damage_torque_points, self._stall_inverse_duration_points = finite_damage_nodes(
            spec.stall_overload
        )
        self._rotating_damage_torque_points, self._rotating_inverse_duration_points = finite_damage_nodes(
            spec.rotating_overload
        )
        self.available_peak_effort = torch.full_like(self.computed_effort, spec.peak_torque_nm)
        self.continuous_effort = torch.full_like(self.computed_effort, spec.stall_continuous_torque_nm)
        self.available_effort = torch.full_like(self.computed_effort, spec.peak_torque_nm)
        self.overload_damage = torch.zeros_like(self.computed_effort)
        self.overload_damage_rate = torch.zeros_like(self.computed_effort)
        self.hard_safety_active = torch.zeros_like(self.computed_effort, dtype=torch.bool)
        self.normalized_torque_excess = torch.zeros_like(self.computed_effort)
        self.torque_clip_ratio = torch.zeros(self.computed_effort.shape[:-1], device=self._device)
        self.max_torque_excess_nm = torch.zeros(self.computed_effort.shape[:-1], device=self._device)
        diagnostic_shape = self.computed_effort.shape[:-1]
        self._diagnostic_requested_count = torch.zeros(diagnostic_shape, dtype=torch.int64, device=self._device)
        self._diagnostic_clipped_count = torch.zeros(diagnostic_shape, dtype=torch.int64, device=self._device)
        self._diagnostic_max_exceed_nm = torch.zeros(diagnostic_shape, device=self._device)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # Mirror UnitreeActuator's explicit-effort output contract used by Go2.
        # DelayedPDActuator already computes the effort, but keeping these fields
        # explicit prevents position drives from being reintroduced downstream.
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        self._joint_vel[:] = joint_vel
        self.available_peak_effort = interpolate_torque_limits(
            self._joint_vel,
            self._velocity_points,
            self._torque_points,
        )
        if self.motor_spec.protection_mode == INSTANTANEOUS_ENVELOPE:
            self.available_effort = self.available_peak_effort
        else:
            self.continuous_effort = interpolate_continuous_torque_limits(
                self._joint_vel,
                self.available_peak_effort,
                self._continuous_velocity_points,
                self._continuous_torque_points,
                switch_speed_rad_s=self.motor_spec.overload_mode_switch_speed_rad_s,
                stall_continuous_torque_nm=self.motor_spec.stall_continuous_torque_nm,
            )
            delivered_before_safety = torch.minimum(torch.abs(effort), self.available_peak_effort)
            self.overload_damage_rate = overload_damage_rates(
                self._joint_vel,
                delivered_before_safety,
                self.continuous_effort,
                switch_speed_rad_s=self.motor_spec.overload_mode_switch_speed_rad_s,
                stall_torque_points=self._stall_damage_torque_points,
                stall_inverse_duration_points=self._stall_inverse_duration_points,
                rotating_torque_points=self._rotating_damage_torque_points,
                rotating_inverse_duration_points=self._rotating_inverse_duration_points,
            )
            hard_before_step = self.hard_safety_active.clone()
            self.overload_damage = torch.clamp(
                self.overload_damage + self.overload_damage_rate * float(self.cfg.physics_dt_s),
                min=0.0,
                max=1.0,
            )
            self.overload_damage = torch.where(
                self.overload_damage >= 1.0 - 1.0e-6,
                torch.ones_like(self.overload_damage),
                self.overload_damage,
            )
            self.hard_safety_active |= self.overload_damage >= 1.0
            # The sample that consumes the final budget is still permitted.  From
            # the following physics step onward the latch limits the joint to its
            # speed-dependent continuous curve for the remainder of the episode.
            self.available_effort = torch.where(
                hard_before_step, self.continuous_effort, self.available_peak_effort
            )
        (
            self.normalized_torque_excess,
            self.torque_clip_ratio,
            self.max_torque_excess_nm,
        ) = torque_feasibility_metrics(effort, self.available_effort, self.motor_spec.peak_torque_nm)
        excess_nm = self.normalized_torque_excess * self.motor_spec.peak_torque_nm
        self._diagnostic_requested_count += effort.shape[-1]
        self._diagnostic_clipped_count += torch.sum(excess_nm > 0.0, dim=-1)
        self._diagnostic_max_exceed_nm = torch.maximum(
            self._diagnostic_max_exceed_nm,
            self.max_torque_excess_nm,
        )
        return torch.clamp(effort, min=-self.available_effort, max=self.available_effort)

    def reset(self, env_ids):
        """Start a new episode with an empty budget; never recover inside one."""

        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.overload_damage[ids] = 0.0
        self.overload_damage_rate[ids] = 0.0
        self.hard_safety_active[ids] = False
        self.reset_diagnostics(env_ids)

    def reset_diagnostics(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset cumulative torque-envelope diagnostics for selected environments."""
        ids = slice(None) if env_ids is None else env_ids
        self._diagnostic_requested_count[ids] = 0
        self._diagnostic_clipped_count[ids] = 0
        self._diagnostic_max_exceed_nm[ids] = 0.0

    def diagnostics(
        self,
        env_ids: torch.Tensor | slice | None = None,
        *,
        reset: bool = False,
    ) -> RobStrideDiagnosticSnapshot:
        """Read cumulative per-environment diagnostics, optionally resetting them.

        Counts are joint-command samples, not policy steps: with 12 joints one
        physical actuator update contributes 12 to ``requested_count``.
        Returned tensors are clones so a subsequent reset cannot mutate a
        logger's snapshot.
        """
        ids = slice(None) if env_ids is None else env_ids
        requested = self._diagnostic_requested_count[ids].clone()
        clipped = self._diagnostic_clipped_count[ids].clone()
        ratio = clipped.to(torch.float32) / torch.clamp(requested, min=1).to(torch.float32)
        maximum = self._diagnostic_max_exceed_nm[ids].clone()
        damage = torch.amax(self.overload_damage[ids], dim=-1).clone()
        hard_count = torch.sum(self.hard_safety_active[ids], dim=-1).clone()
        snapshot = RobStrideDiagnosticSnapshot(requested, clipped, ratio, maximum, damage, hard_count)
        if reset:
            self.reset_diagnostics(env_ids)
        return snapshot


@configclass
class RobStrideActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a RobStride piecewise torque-speed actuator."""

    class_type: type = RobStrideActuator
    motor_config_path: str = ""
    physics_dt_s: float = 0.005


@configclass
class RobStride06ActuatorCfg(RobStrideActuatorCfg):
    """RobStride 06 output envelope measured at 48 V with the integrated 9:1 gearbox."""

    motor_config_path: str = str(ROBSTRIDE06_MOTOR_CONFIG)

    def __post_init__(self):
        # Canonicalize both actuator-model and PhysX-solver limits from the
        # same output-side model.  Explicit values supplied at a call site are
        # intentionally replaced so a stale task config cannot diverge.
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        spec = load_motor_spec(self.motor_config_path)
        self.actuator_effort_limit = spec.peak_torque_nm
        self.actuator_velocity_limit = spec.max_speed_rad_s
        self.joint_effort_limit = spec.peak_torque_nm
        self.joint_velocity_limit = spec.max_speed_rad_s
        self.armature = spec.output_equivalent_inertia_kg_m2
        self.friction = spec.output_joint_friction_nm
        self.physics_dt_s = spec.actuator_delay_physics_dt_s
        self.min_delay = spec.actuator_delay_min_physics_steps
        self.max_delay = spec.actuator_delay_max_physics_steps


class WheelDiagnosticSnapshot(NamedTuple):
    requested_count: torch.Tensor
    clipped_count: torch.Tensor
    clip_ratio: torch.Tensor
    max_exceed_nm: torch.Tensor


class RobStride02Actuator(DelayedPDActuator):
    """D2W wheel velocity PD with the manual-derived instantaneous envelope."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from d2w_rl_lab.assets.d2w_wheel_motor import load_wheel_motor_spec
        spec = load_wheel_motor_spec()
        points = torch.tensor(spec.envelope, device=self._device, dtype=torch.float32)
        self._wheel_velocity_points, self._wheel_torque_points = points[:, 0].contiguous(), points[:, 1].contiguous()
        shape = self.computed_effort.shape[:-1]
        self._diagnostic_requested_count = torch.zeros(shape, dtype=torch.int64, device=self._device)
        self._diagnostic_clipped_count = torch.zeros(shape, dtype=torch.int64, device=self._device)
        self._diagnostic_max_exceed_nm = torch.zeros(shape, device=self._device)

    def compute(self, control_action, joint_pos, joint_vel):
        result = super().compute(control_action, joint_pos, joint_vel)
        available = interpolate_torque_limits(joint_vel, self._wheel_velocity_points, self._wheel_torque_points)
        # Count every physical actuator update using the same pre-integration
        # speed and requested torque as the limiter, including delayed commands.
        excess = torch.relu(self.computed_effort.abs() - available)
        self._diagnostic_requested_count += self.computed_effort.shape[-1]
        self._diagnostic_clipped_count += (excess > 0.).sum(dim=-1)
        self._diagnostic_max_exceed_nm = torch.maximum(self._diagnostic_max_exceed_nm, excess.amax(dim=-1))
        self.applied_effort = torch.clamp(self.computed_effort, min=-available, max=available)
        result.joint_positions = None
        result.joint_velocities = None
        result.joint_efforts = self.applied_effort
        return result


    def reset(self, env_ids):
        super().reset(env_ids)
        self.reset_diagnostics(env_ids)

    def reset_diagnostics(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._diagnostic_requested_count[ids] = 0
        self._diagnostic_clipped_count[ids] = 0
        self._diagnostic_max_exceed_nm[ids] = 0.

    def diagnostics(self, env_ids=None, *, reset=False):
        """Joint-command samples at physics frequency; returned values are copies."""
        ids = slice(None) if env_ids is None else env_ids
        requested = self._diagnostic_requested_count[ids].clone()
        clipped = self._diagnostic_clipped_count[ids].clone()
        ratio = clipped.float() / requested.clamp_min(1).float()
        snapshot = WheelDiagnosticSnapshot(requested, clipped, ratio, self._diagnostic_max_exceed_nm[ids].clone())
        if reset:
            self.reset_diagnostics(env_ids)
        return snapshot


@configclass
class RobStride02ActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = RobStride02Actuator
