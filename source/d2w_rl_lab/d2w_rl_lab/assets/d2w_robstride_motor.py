"""Pure-Python RobStride motor contract and overload-budget helpers.

This module deliberately has no Torch or Isaac Lab imports.  The simulator
actuator, asset generator, deployment parity tests, and lightweight CPU tests
can therefore validate the same output-side RS06 data without starting Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Iterable

import yaml


RPM_TO_RAD_S = 2.0 * math.pi / 60.0
DAMAGE_EXHAUSTION_EPS = 1.0e-6
RS06_SCHEMA_VERSION = 1
RS06_CONTRACT_VERSION = "rs06-48v-integrated-output-v1"
RS06_MANUAL_REVISION = "official_manual_260428_v2"
RS06_MANUAL_CONTRACT_VERSION = "rs06-48v-integrated-output-manual260428-v2"
RS06_INSTANTANEOUS_CONTRACT_VERSION = "rs06-48v-integrated-output-manual260428-instantaneous-v3"
CUMULATIVE_OVERLOAD = "cumulative_overload"
INSTANTANEOUS_ENVELOPE = "instantaneous_envelope"
RS06_MANUAL_SHA256 = "28972fe4ddb6e2731856536e16cd95d0d5b1f4556c9cd3389fa0bef5e6c53497"
RS06_MODEL = "RobStride_06"
RS06_RATED_VOLTAGE_V = 48.0
RS06_REFERENCE_SIDE = "integrated_actuator_output_after_9_to_1_reduction"
RS06_INERTIA_REFERENCE_SIDE = "integrated_actuator_output_low_speed_equivalent"
RS06_OUTPUT_TORQUE_UNIT = "N.m"
RS06_OUTPUT_SPEED_UNIT = "rad/s"
RS06_OUTPUT_INERTIA_UNIT = "kg.m^2"
RS06_PRODUCT_MANUAL_SOURCE = "not_distributed_with_runtime_assets"
RS06_PRODUCT_REPOSITORY = "https://github.com/RobStride/Product_Information"


@dataclass(frozen=True)
class TorqueDurationTable:
    """Monotone output-torque nodes and their maximum operating durations."""

    torque_points_nm: tuple[float, ...]
    duration_points_s: tuple[float, ...]


@dataclass(frozen=True)
class RobStrideMotorSpec:
    """Validated 48 V, integrated-output RS06 motor contract."""

    config_path: Path
    schema_version: int
    contract_version: str
    model: str
    rated_voltage_v: float
    reference_side: str
    output_torque_unit: str
    output_speed_unit: str
    output_inertia_unit: str
    inertia_reference_side: str
    product_manual_source: str
    official_product_repository: str
    peak_curve_source: str
    continuous_curve_source: str
    stall_overload_source: str
    rotating_overload_source: str
    velocity_points: tuple[float, ...]
    torque_points: tuple[float, ...]
    continuous_velocity_points: tuple[float, ...]
    continuous_torque_points: tuple[float, ...]
    rotating_overload: TorqueDurationTable
    stall_overload: TorqueDurationTable
    rated_torque_nm: float
    peak_torque_nm: float
    no_load_speed_rad_s: float
    protocol_speed_limit_rad_s: float
    max_speed_rad_s: float
    overload_mode_switch_speed_rad_s: float
    stall_continuous_torque_nm: float
    output_equivalent_inertia_kg_m2: float
    output_joint_friction_nm: float
    output_joint_friction_provenance: str
    actuator_delay_min_physics_steps: int
    actuator_delay_max_physics_steps: int
    actuator_delay_physics_dt_s: float
    actuator_delay_provenance: str
    actuator_delay_sampling: str
    overload_damage_accumulation: str
    overload_damage_recovery_within_episode: bool
    overload_hard_safety_action: str
    rated_load_phase_current_apk: float
    peak_phase_current_apk: float
    firmware_iq_reference_limit_a: float
    phase_current_to_firmware_iq_mapping: str
    output_equivalent_inertia_provenance: str
    overload_mode_switch_provenance: str
    overload_model_provenance: str
    product_manual_version: str | None
    product_manual_sha256: str | None
    firmware_iq_reference_range_a: tuple[float, float] | None
    firmware_current_limit_range_a: tuple[float, float] | None
    protection_mode: str = CUMULATIVE_OVERLOAD


def motor_protection_mode(raw: dict) -> str:
    """Missing mode preserves historical behavior; unknown modes never fall back."""
    mode = raw.get("protection_mode", CUMULATIVE_OVERLOAD)
    if mode not in (CUMULATIVE_OVERLOAD, INSTANTANEOUS_ENVELOPE):
        raise ValueError(f"Unsupported RS06 protection mode: {mode!r}")
    if mode == INSTANTANEOUS_ENVELOPE and raw.get("data_revision") != RS06_MANUAL_REVISION:
        raise ValueError("RS06 instantaneous mode requires official manual 260428 data.")
    return mode


def motor_protection_identity(raw: dict) -> dict:
    """Only explicit modes extend identities, preserving old receipt serialization."""
    mode = motor_protection_mode(raw)
    if "protection_mode" not in raw:
        return {}
    version = (RS06_INSTANTANEOUS_CONTRACT_VERSION if mode == INSTANTANEOUS_ENVELOPE else
               RS06_MANUAL_CONTRACT_VERSION if raw.get("data_revision") == RS06_MANUAL_REVISION else
               RS06_CONTRACT_VERSION)
    return {"protection_mode": mode, "contract_version": version}


@dataclass(frozen=True)
class OverloadStep:
    """One scalar overload-budget update used by CPU contract tests."""

    peak_limit_nm: float
    continuous_limit_nm: float
    damage_rate_per_s: float
    damage: float
    hard_safety_active: bool
    available_limit_nm: float


def _finite_nonnegative(values: Iterable[float], field: str) -> None:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{field} values must be finite and non-negative.")


def _curve(raw: object, field: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"{field} needs at least two [x, y] points.")
    try:
        x = tuple(float(point[0]) for point in raw)
        y = tuple(float(point[1]) for point in raw)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{field} contains an invalid point.") from exc
    _finite_nonnegative((*x, *y), field)
    if any(right <= left for left, right in zip(x, x[1:])):
        raise ValueError(f"{field} x points must be strictly increasing.")
    return x, y


def _duration_table(raw: object, field: str) -> TorqueDurationTable:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"{field} needs at least two [torque, duration] points.")
    try:
        torque = tuple(float(point[0]) for point in raw)
        duration = tuple(math.inf if point[1] is None else float(point[1]) for point in raw)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{field} contains an invalid point.") from exc
    _finite_nonnegative(torque, field)
    if any(right <= left for left, right in zip(torque, torque[1:])):
        raise ValueError(f"{field} torque points must be strictly increasing.")
    if any((not math.isinf(value)) and (not math.isfinite(value) or value <= 0.0) for value in duration):
        raise ValueError(f"{field} durations must be positive or null/continuous.")
    if not math.isinf(duration[0]):
        raise ValueError(f"{field} must begin with its continuous-torque node.")
    return TorqueDurationTable(torque, duration)


@lru_cache(maxsize=None)
def load_motor_spec(config_path: str | Path) -> RobStrideMotorSpec:
    """Load the canonical output-side RS06 YAML without unit conversion guesses."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != RS06_SCHEMA_VERSION:
        raise ValueError(f"Unsupported RobStride motor schema in {path}.")
    revision = raw.get("data_revision")
    if revision not in (None, RS06_MANUAL_REVISION):
        raise ValueError(f"Unsupported RS06 data revision: {revision!r}")
    manual_260428 = revision == RS06_MANUAL_REVISION
    protection_mode = motor_protection_mode(raw)

    try:
        peak_velocity, peak_torque = _curve(
            raw["peak_torque_speed_envelope_rad_s"], "peak_torque_speed_envelope_rad_s"
        )
        thermal_rpm, continuous_torque = _curve(
            raw["thermal_equilibrium_rotating"], "thermal_equilibrium_rotating"
        )
        continuous_velocity = tuple(rpm * RPM_TO_RAD_S for rpm in thermal_rpm)
        rotating_overload = _duration_table(raw["rotating_overload_duration"], "rotating_overload_duration")
        stall_overload = _duration_table(raw["stall_overload_duration"], "stall_overload_duration")
        output_limits = raw["output_limits"]
        mechanical = raw["mechanical"]
        electrical = raw["electrical"]
        firmware = raw["firmware_private_protocol"]
        sources = raw["sources"]
        model = str(raw["model"])
        rated_voltage_v = float(electrical["rated_voltage_v"])
        reduction_ratio = float(mechanical["reduction_ratio"])
        manual_torque, manual_speed_rpm = _curve(
            raw["torque_speed_curve_48v"], "torque_speed_curve_48v"
        )
        manual_torque_speed_curve = tuple(zip(manual_torque, manual_speed_rpm))
        heatsink_mm = tuple(float(value) for value in output_limits["test_heatsink_mm"])
        product_manual_source = str(sources["local_product_sheet"])
        official_product_repository = str(sources["official_product_repository"])

        rated_torque_nm = float(output_limits["rated_torque_nm"])
        peak_torque_nm = float(output_limits["peak_torque_nm"])
        no_load_speed_rad_s = float(output_limits["no_load_speed_rad_s"])
        protocol_speed_limit_rad_s = float(output_limits["protocol_speed_limit_rad_s"])
        switch_speed_rpm = float(
            raw["engineering_assumptions"]["overload_mode_switch_speed_rpm"]
            if manual_260428 else output_limits["rated_torque_speed_rpm"]
        )
        switch_speed_rad_s = switch_speed_rpm * RPM_TO_RAD_S
        stall_continuous_torque_nm = float(stall_overload.torque_points_nm[0])
        armature = float(mechanical["output_equivalent_inertia_kg_m2"])
        friction = float(mechanical["output_joint_friction_nm"])
        rated_current_apk = float(electrical["rated_load_phase_current_apk"])
        peak_current_apk = float(electrical["peak_phase_current_apk"])
        firmware_iq_limit_a = float(firmware[
            "iq_ref_range_a" if manual_260428 else "limit_cur_iq_ref_range_a"
        ][1])
        firmware_velocity_mapping = tuple(float(value) for value in firmware["velocity_mapping_rad_s"])
        firmware_speed_limit = tuple(float(value) for value in firmware[
            "limit_spd_range_rad_s" if manual_260428 else "limit_spd_and_spd_ref_range_rad_s"
        ])
        firmware_torque_mapping = tuple(float(value) for value in firmware["torque_mapping_nm"])
        if manual_260428:
            iq_range = tuple(float(v) for v in firmware["iq_ref_range_a"])
            current_range = tuple(float(v) for v in firmware["limit_cur_range_a"])
            speed_reference_range = tuple(float(v) for v in firmware["spd_ref_range_rad_s"])
            standard_heatsink = tuple(float(v) for v in output_limits["standard_use_heatsink_mm"])
            torque_constant = float(electrical["torque_constant_nm_per_arms"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Incomplete RobStride motor model: {path}") from exc

    scalar_values = (
        rated_torque_nm,
        peak_torque_nm,
        no_load_speed_rad_s,
        protocol_speed_limit_rad_s,
        switch_speed_rad_s,
        switch_speed_rpm,
        stall_continuous_torque_nm,
        armature,
        friction,
        0.005,
        rated_current_apk,
        peak_current_apk,
        firmware_iq_limit_a,
    )
    _finite_nonnegative(scalar_values, "RS06 scalar")
    if min(rated_torque_nm, peak_torque_nm, no_load_speed_rad_s, protocol_speed_limit_rad_s) <= 0.0:
        raise ValueError("RS06 rated/peak torque, speed, and physics dt must be positive.")
    if model != RS06_MODEL:
        raise ValueError(f"Motor contract is for {model!r}, expected {RS06_MODEL!r}.")
    if not math.isclose(rated_voltage_v, RS06_RATED_VOLTAGE_V, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"RS06 motor contract must use the 48 V product-sheet curves, got {rated_voltage_v} V.")
    if not math.isclose(reduction_ratio, 9.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("RS06 integrated-output contract requires the built-in 9:1 reduction.")
    if not math.isclose(armature, 0.012, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("This RS06 model version pins provisional output-side inertia to 0.012 kg.m^2; this is not a manufacturer measurement and must not be multiplied by 81.")
    expected_manual_source = "official_RS06_manual_260428" if manual_260428 else RS06_PRODUCT_MANUAL_SOURCE
    if product_manual_source != expected_manual_source:
        raise ValueError("RS06 local product-manual provenance is missing or unexpected.")
    if official_product_repository != RS06_PRODUCT_REPOSITORY:
        raise ValueError("RS06 official product-repository provenance is missing or unexpected.")
    if heatsink_mm != ((130.0, 155.0) if manual_260428 else (200.0, 200.0)):
        raise ValueError("RS06 overload test heatsink differs from the selected data revision.")
    if manual_260428:
        if (str(sources.get("manual_version")) != "260428"
                or sources.get("manual_sha256") != RS06_MANUAL_SHA256
                or sources.get("manual_git_blob") != "0fe792deba72296a79f1a3a7ed3ce80b0aa94cfa"
                or sources.get("manual_git_tag") != "V26.05.19"):
            raise ValueError("RS06 manual 260428 source identity differs.")
        if iq_range != (-57.0, 57.0) or current_range != (0.0, 57.0):
            raise ValueError("RS06 manual 260428 requires separate signed Iq and unsigned current-limit ranges.")
        if speed_reference_range != (-50.0, 50.0):
            raise ValueError("RS06 speed reference must retain both rotation directions.")
        if standard_heatsink != (130.0, 160.0) or torque_constant != 1.09:
            raise ValueError("RS06 manual 260428 standard conditions/torque constant differ.")
        if (output_limits.get("overload_test_ambient_temperature_c") != 25.0
                or output_limits.get("overload_test_winding_constraint_temperature_c") != 145.0):
            raise ValueError("RS06 manual 260428 overload test temperatures differ.")
        if (mechanical.get("output_equivalent_inertia_provenance") != "engineering_prior_pending_identification"
                or raw["engineering_assumptions"].get("continuous_curve_provenance")
                != "engineering_prior_with_manual_11Nm_100rpm_anchor"
                or raw["engineering_assumptions"].get("overload_mode_switch_provenance")
                != "engineering_prior_not_manufacturer_stall_definition"
                or raw["engineering_assumptions"].get("overload_model_provenance")
                != "engineering_budget_not_firmware_temperature_estimator"):
            raise ValueError("RS06 provisional model assumptions must not claim manufacturer verification.")
        if thermal_rpm[0] != 100.0 or continuous_torque[0] != rated_torque_nm or rated_torque_nm != 11.0:
            raise ValueError("RS06 continuous-curve anchor must match 11 N.m at 100 rpm.")

    # Check conversion of the graph-digitization nodes, not their accuracy
    # against the printed raster plot. The endpoint extensions are assumptions.
    if len(manual_torque_speed_curve) != 5:
        raise ValueError("This RS06 version requires five graph-digitization nodes.")
    for manual_torque_nm, manual_speed_rpm in manual_torque_speed_curve:
        manual_speed_rad_s = manual_speed_rpm * RPM_TO_RAD_S
        if not any(
            math.isclose(runtime_speed, manual_speed_rad_s, rel_tol=0.0, abs_tol=5.0e-6)
            and math.isclose(runtime_torque, manual_torque_nm, rel_tol=0.0, abs_tol=1.0e-12)
            for runtime_speed, runtime_torque in zip(peak_velocity, peak_torque)
        ):
            raise ValueError(
                "RS06 peak envelope is not an output-side rad/s conversion of torque_speed_curve_48v."
            )

    if not math.isclose(peak_velocity[0], 0.0, abs_tol=1.0e-12) or not math.isclose(
        peak_torque[0], peak_torque_nm, abs_tol=1.0e-9
    ):
        raise ValueError("RS06 peak envelope must start at [0, peak torque].")
    if not math.isclose(peak_velocity[-1], no_load_speed_rad_s, abs_tol=1.0e-6) or not math.isclose(
        peak_torque[-1], 0.0, abs_tol=1.0e-12
    ):
        raise ValueError("RS06 peak envelope must end at [no-load speed, 0].")
    if protocol_speed_limit_rad_s > no_load_speed_rad_s:
        raise ValueError("RS06 protocol speed cap cannot exceed no-load speed.")
    if firmware_velocity_mapping != (-protocol_speed_limit_rad_s, protocol_speed_limit_rad_s):
        raise ValueError("RS06 protocol velocity mapping must match the declared output-side protocol speed cap.")
    if firmware_speed_limit != (0.0, protocol_speed_limit_rad_s):
        raise ValueError("RS06 protocol speed-reference limit must match the declared output-side speed cap.")
    if firmware_torque_mapping != (-peak_torque_nm, peak_torque_nm):
        raise ValueError("RS06 protocol torque mapping must match the declared output-side peak torque.")
    if not math.isclose(stall_overload.torque_points_nm[0], stall_continuous_torque_nm, abs_tol=1.0e-12):
        raise ValueError("RS06 stall overload table must start at the stall-continuous torque.")
    if rotating_overload.torque_points_nm[-1] != peak_torque_nm or stall_overload.torque_points_nm[-1] != peak_torque_nm:
        raise ValueError("RS06 overload tables must end at peak torque.")

    return RobStrideMotorSpec(
        config_path=path,
        schema_version=RS06_SCHEMA_VERSION,
        contract_version=(RS06_INSTANTANEOUS_CONTRACT_VERSION
                          if protection_mode == INSTANTANEOUS_ENVELOPE else
                          RS06_MANUAL_CONTRACT_VERSION if manual_260428 else RS06_CONTRACT_VERSION),
        protection_mode=protection_mode,
        model=model,
        rated_voltage_v=rated_voltage_v,
        reference_side=RS06_REFERENCE_SIDE,
        output_torque_unit=RS06_OUTPUT_TORQUE_UNIT,
        output_speed_unit=RS06_OUTPUT_SPEED_UNIT,
        output_inertia_unit=RS06_OUTPUT_INERTIA_UNIT,
        inertia_reference_side=RS06_INERTIA_REFERENCE_SIDE,
        product_manual_source=product_manual_source,
        official_product_repository=official_product_repository,
        peak_curve_source="manual260428.p7.graph_digitization_with_assumed_extensions@48V" if manual_260428 else "product_manual.torque_speed_curve_48v@48V",
        continuous_curve_source="engineering_prior_with_manual_11Nm_100rpm_anchor" if manual_260428 else "product_manual.thermal_equilibrium_rotating@200x200mm_heatsink",
        stall_overload_source="manual260428.p8.stall_overload_duration" if manual_260428 else "product_manual.stall_overload_duration@200x200mm_heatsink",
        rotating_overload_source="manual260428.p7-p8.rotating_overload_duration@130x155mm_heatsink_25C" if manual_260428 else "product_manual.rotating_overload_duration@200x200mm_heatsink",
        velocity_points=peak_velocity,
        torque_points=peak_torque,
        continuous_velocity_points=continuous_velocity,
        continuous_torque_points=continuous_torque,
        rotating_overload=rotating_overload,
        stall_overload=stall_overload,
        rated_torque_nm=rated_torque_nm,
        peak_torque_nm=peak_torque_nm,
        no_load_speed_rad_s=no_load_speed_rad_s,
        protocol_speed_limit_rad_s=protocol_speed_limit_rad_s,
        max_speed_rad_s=min(no_load_speed_rad_s, protocol_speed_limit_rad_s),
        overload_mode_switch_speed_rad_s=switch_speed_rad_s,
        stall_continuous_torque_nm=stall_continuous_torque_nm,
        output_equivalent_inertia_kg_m2=armature,
        output_joint_friction_nm=friction,
        # Friction and the inherited Go2 delay remain unidentified estimates.
        output_joint_friction_provenance="engineering_prior",
        actuator_delay_min_physics_steps=0,
        actuator_delay_max_physics_steps=4,
        actuator_delay_physics_dt_s=0.005,
        actuator_delay_provenance="engineering_prior",
        actuator_delay_sampling="uniform_per_environment_on_reset",
        overload_damage_accumulation=("not_applicable" if protection_mode == INSTANTANEOUS_ENVELOPE
                                      else "piecewise_linear_inverse_duration"),
        overload_damage_recovery_within_episode=False,
        overload_hard_safety_action=("not_applicable" if protection_mode == INSTANTANEOUS_ENVELOPE
                                    else "clamp_to_speed_dependent_continuous_torque"),
        rated_load_phase_current_apk=rated_current_apk,
        peak_phase_current_apk=peak_current_apk,
        firmware_iq_reference_limit_a=firmware_iq_limit_a,
        phase_current_to_firmware_iq_mapping="unresolved_no_conversion",
        output_equivalent_inertia_provenance="engineering_prior_pending_identification",
        overload_mode_switch_provenance="engineering_prior_not_manufacturer_stall_definition",
        overload_model_provenance="engineering_budget_not_firmware_temperature_estimator",
        product_manual_version="260428" if manual_260428 else None,
        product_manual_sha256=RS06_MANUAL_SHA256 if manual_260428 else None,
        firmware_iq_reference_range_a=iq_range if manual_260428 else None,
        firmware_current_limit_range_a=current_range if manual_260428 else None,
    )


def interpolate_piecewise(value: float, x: tuple[float, ...], y: tuple[float, ...], *, right: float) -> float:
    """Scalar linear interpolation with constant left and explicit right fill."""

    if value <= x[0]:
        return y[0]
    for left, upper in enumerate(x[1:]):
        if value <= upper:
            x0, x1 = x[left], upper
            fraction = (value - x0) / (x1 - x0)
            return y[left] + fraction * (y[left + 1] - y[left])
    return right


def peak_torque_limit_nm(spec: RobStrideMotorSpec, speed_rad_s: float) -> float:
    return max(
        0.0,
        interpolate_piecewise(
            abs(float(speed_rad_s)), spec.velocity_points, spec.torque_points, right=0.0
        ),
    )


def continuous_torque_limit_nm(spec: RobStrideMotorSpec, speed_rad_s: float) -> float:
    """Return the model's continuous limit; speed derating is an engineering prior."""

    speed = abs(float(speed_rad_s))
    peak = peak_torque_limit_nm(spec, speed)
    if speed < spec.overload_mode_switch_speed_rad_s:
        thermal = spec.stall_continuous_torque_nm
    else:
        thermal = interpolate_piecewise(
            speed,
            spec.continuous_velocity_points,
            spec.continuous_torque_points,
            right=spec.continuous_torque_points[-1],
        )
    return min(peak, thermal)


def _damage_rate_from_table(
    delivered_torque_nm: float,
    continuous_torque_nm: float,
    table: TorqueDurationTable,
) -> float:
    if delivered_torque_nm <= continuous_torque_nm:
        return 0.0
    finite_nodes = [
        (torque, 1.0 / duration)
        for torque, duration in zip(table.torque_points_nm, table.duration_points_s)
        if math.isfinite(duration) and torque > continuous_torque_nm
    ]
    if not finite_nodes:
        return 0.0
    previous_torque, previous_rate = continuous_torque_nm, 0.0
    for torque, rate in finite_nodes:
        if delivered_torque_nm <= torque:
            fraction = (delivered_torque_nm - previous_torque) / (torque - previous_torque)
            return previous_rate + fraction * (rate - previous_rate)
        previous_torque, previous_rate = torque, rate
    return finite_nodes[-1][1]


def overload_damage_rate_per_s(spec: RobStrideMotorSpec, speed_rad_s: float, effort_nm: float) -> float:
    """Return irreversible-in-episode overload damage rate for delivered effort."""

    speed = abs(float(speed_rad_s))
    peak = peak_torque_limit_nm(spec, speed)
    continuous = continuous_torque_limit_nm(spec, speed)
    delivered = min(abs(float(effort_nm)), peak)
    table = spec.stall_overload if speed < spec.overload_mode_switch_speed_rad_s else spec.rotating_overload
    return _damage_rate_from_table(delivered, continuous, table)


def step_overload_damage(
    spec: RobStrideMotorSpec,
    *,
    damage: float,
    speed_rad_s: float,
    effort_nm: float,
    dt_s: float,
) -> OverloadStep:
    """Advance one scalar budget; a latched joint cannot recover until reset."""

    if not math.isfinite(damage) or damage < 0.0 or not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("Overload damage must be non-negative and dt must be positive.")
    prior_damage = min(float(damage), 1.0)
    peak = peak_torque_limit_nm(spec, speed_rad_s)
    if not math.isfinite(speed_rad_s) or not math.isfinite(effort_nm):
        raise ValueError("RS06 motor input must be finite.")
    if spec.protection_mode == INSTANTANEOUS_ENVELOPE:
        return OverloadStep(peak, peak, 0.0, 0.0, False, peak)
    continuous = continuous_torque_limit_nm(spec, speed_rad_s)
    rate = overload_damage_rate_per_s(spec, speed_rad_s, effort_nm)
    updated = min(1.0, prior_damage + rate * dt_s)
    if updated >= 1.0 - DAMAGE_EXHAUSTION_EPS:
        updated = 1.0
    hard_before_step = prior_damage >= 1.0
    return OverloadStep(
        peak_limit_nm=peak,
        continuous_limit_nm=continuous,
        damage_rate_per_s=rate,
        damage=updated,
        hard_safety_active=updated >= 1.0,
        available_limit_nm=continuous if hard_before_step else peak,
    )
