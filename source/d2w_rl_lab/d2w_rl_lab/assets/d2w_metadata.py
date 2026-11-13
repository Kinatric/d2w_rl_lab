"""Fail-closed metadata loader for the source/runtime separated D2W asset."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


D2W_CANONICAL_JOINT_ORDER = ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_joint')
D2W_LEG_JOINT_ORDER = ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint')
D2W_WHEEL_JOINT_ORDER = ('FL_foot_joint', 'FR_foot_joint', 'RL_foot_joint', 'RR_foot_joint')
D2W_FIXED_FOOT_JOINTS = ()
D2W_FIXSTAND_JOINT_POSITIONS = dict(zip(D2W_CANONICAL_JOINT_ORDER, (0., .8, -1.5, 0.) * 4, strict=True))
# Shared training reference: Go2W leg axes verified against the D2W URDF.
# RL reset/action/observation offsets and zero-command pose rewards use this
# same mapping. The separate FixStand calibration pose above is unchanged.
D2W_TRAINING_JOINT_POSITIONS = dict(zip(D2W_CANONICAL_JOINT_ORDER, [
    -0.1, 0.8, -1.5, 0.0, 0.1, 0.8, -1.5, 0.0,
    -0.1, 0.8, -1.5, 0.0, 0.1, 0.8, -1.5, 0.0,
], strict=True))
D2W_RL_DEFAULT_JOINT_POSITIONS = D2W_TRAINING_JOINT_POSITIONS
# Explicit RL reward target; independent of FixStand FK and spawn clearance.
D2W_RL_STANDING_HEIGHT_M = 0.40
# PD baseline from D2W gait training P51B (2026-10-22), not a motor identification.
D2W_LEG_KP = 30.0
D2W_LEG_KD = 2.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MASS_METHOD = "preserve_source_urdf_mass_com_inertia"
_TOTAL_MASS_KG = 19.83718907


@dataclass(frozen=True)
class D2WAssetIdentity:
    manifest_path: Path
    canonical_asset_sha256: str
    joint_order: tuple[str, ...]
    control: dict[str, object]
    isaac_usd_path: Path
    isaac_usd_sha256: str
    mujoco_xml_path: Path
    mujoco_xml_sha256: str
    manifest: dict[str, object]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: object, field: str) -> str:
    value = str(value).lower()
    if not _SHA256.fullmatch(value):
        raise RuntimeError(f"D2W manifest {field} is not a SHA-256 digest.")
    return value


def _resolve(root: Path, item: object, field: str, *, allow_source: bool = False) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise RuntimeError(f"D2W manifest {field} must contain path/sha256.")
    relative = Path(item["path"])
    if relative.is_absolute() or (".." in relative.parts and not allow_source):
        raise RuntimeError(f"D2W manifest {field} path is unsafe: {relative}")
    path = (root / relative).resolve()
    allowed = root.parent.resolve() if allow_source else root.resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise RuntimeError(f"D2W manifest {field} is missing or escapes its package: {path}")
    expected = _require_hash(item.get("sha256"), f"{field}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"D2W manifest {field} hash differs: expected={expected}, actual={actual}")
    return path


@lru_cache(maxsize=1)
def get_d2w_asset_identity() -> D2WAssetIdentity:
    repo = Path(__file__).resolve().parents[4]
    runtime = repo / "resources/d2w_runtime"
    manifest_path = runtime / "asset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read D2W runtime manifest: {manifest_path}") from exc
    if manifest.get("schema") != "robotlab.d2w.runtime.v1" or manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported D2W runtime manifest schema.")
    if manifest.get("validation", {}).get("status") != "ready":
        raise RuntimeError("D2W runtime has not passed independent validation.")
    identity = _require_hash(manifest.get("asset_identity_sha256"), "asset_identity_sha256")
    if manifest.get("canonical_asset_sha256") != identity:
        raise RuntimeError("D2W canonical/runtime identity fields differ.")
    mass_properties = manifest.get("mass_properties", {})
    if mass_properties.get("schema") != "robotlab.d2w.measured-mass.v1":
        raise RuntimeError("D2W manifest has no supported measured-mass contract.")
    if mass_properties.get("method") != _MASS_METHOD:
        raise RuntimeError("D2W measured-mass inertia derivation method differs.")
    if not math.isclose(float(mass_properties.get("total_mass_kg", math.nan)), _TOTAL_MASS_KG, abs_tol=1.0e-12):
        raise RuntimeError("D2W measured total mass differs.")
    topology = manifest.get("topology")
    expected_topology = {
        "links": 17, "revolute_joints": 12, "fixed_joints": 0, "continuous_joints": 4, "collisions": 33,
        "joint_order": list(D2W_CANONICAL_JOINT_ORDER), "fixed_foot_joints": list(D2W_FIXED_FOOT_JOINTS),
    }
    if topology != expected_topology:
        raise RuntimeError(f"D2W runtime topology differs: {topology}")
    urdf_path = _resolve(runtime, manifest["inputs"]["urdf"], "inputs.urdf", allow_source=True)
    source_root = urdf_path.parents[1]
    for index, item in enumerate(manifest.get("source", {}).get("files", [])):
        _resolve(source_root, item, f"source.files[{index}]")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("D2W runtime manifest has no outputs.")
    usd = outputs.get("isaac_usd")
    mjcf = outputs.get("mujoco_xml")
    usd_path = _resolve(runtime, usd, "outputs.isaac_usd")
    mjcf_path = _resolve(runtime, mjcf, "outputs.mujoco_xml")
    files = usd.get("files") if isinstance(usd, dict) else None
    if not isinstance(files, list) or not files:
        raise RuntimeError("D2W runtime manifest has no USD package inventory.")
    for index, item in enumerate(files):
        _resolve(usd_path.parent, item, f"outputs.isaac_usd.files[{index}]")
    actual_files = {
        path.relative_to(usd_path.parent).as_posix() for path in usd_path.parent.rglob("*") if path.is_file()
    }
    if actual_files != {str(item["path"]) for item in files}:
        raise RuntimeError("D2W USD package contains an undeclared or missing dependency.")
    receipt_path = _resolve(runtime, manifest["validation"].get("receipt"), "validation.receipt")
    report = json.loads(receipt_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("asset_identity_sha256") != identity:
        raise RuntimeError("D2W validation receipt is stale or not passed.")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") != "ready":
        raise RuntimeError("D2W standing calibration is not ready.")
    joint_order = tuple(manifest.get("joint_order", ()))
    if joint_order != D2W_CANONICAL_JOINT_ORDER:
        raise RuntimeError("D2W manifest joint order differs from the policy contract.")
    control = manifest.get("control")
    if control != {"physics_dt": 0.005, "decimation": 4, "policy_hz": 50.0}:
        raise RuntimeError(f"D2W control timing differs: {control}")
    return D2WAssetIdentity(
        manifest_path=manifest_path, canonical_asset_sha256=identity, joint_order=joint_order,
        control=dict(control), isaac_usd_path=usd_path,
        isaac_usd_sha256=_require_hash(usd["sha256"], "outputs.isaac_usd.sha256"),
        mujoco_xml_path=mjcf_path,
        mujoco_xml_sha256=_require_hash(mjcf["sha256"], "outputs.mujoco_xml.sha256"), manifest=manifest,
    )


def _vector(value: str | None, length: int, default: tuple[float, ...]) -> np.ndarray:
    result = np.asarray(default if value is None else [float(item) for item in value.split()], dtype=np.float64)
    if result.shape != (length,):
        raise RuntimeError(f"Expected {length} URDF values, got {value!r}")
    return result


def _rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.asarray([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def _transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3], value[:3, 3] = rotation, xyz
    return value


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, d = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray([[c+x*x*d, x*y*d-z*s, x*z*d+y*s], [y*x*d+z*s, c+y*y*d, y*z*d-x*s], [z*x*d-y*s, z*y*d+x*s, c+z*z*d]])


def _pose_height(joint_positions: dict[str, float]) -> float:
    identity = get_d2w_asset_identity()
    urdf = _resolve(identity.manifest_path.parent, identity.manifest["inputs"]["urdf"], "inputs.urdf", allow_source=True)
    root = ET.parse(urdf).getroot()
    transforms = {'base_link': np.eye(4)}
    unresolved = list(root.findall("joint"))
    while unresolved:
        progress = False
        for joint in unresolved[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            transform = transforms[parent] @ _transform(
                _vector(None if origin is None else origin.get("xyz"), 3, (0, 0, 0)),
                _rotation(_vector(None if origin is None else origin.get("rpy"), 3, (0, 0, 0))),
            )
            if joint.get("type") in ("revolute", "continuous"):
                axis = joint.find("axis")
                transform = transform @ _transform(
                    np.zeros(3), _axis_angle(_vector(None if axis is None else axis.get("xyz"), 3, (1, 0, 0)), joint_positions[joint.get("name")])
                )
            transforms[child] = transform
            unresolved.remove(joint)
            progress = True
        if not progress:
            raise RuntimeError("D2W URDF is not a resolvable tree.")
    lowest = []
    for link in root.findall("link"):
        name = link.get("name")
        if not name.endswith("_foot"):
            continue
        collision = link.find("collision")
        origin, sphere = collision.find("origin"), collision.find("geometry/cylinder")
        transform = transforms[name] @ _transform(
            _vector(None if origin is None else origin.get("xyz"), 3, (0, 0, 0)),
            _rotation(_vector(None if origin is None else origin.get("rpy"), 3, (0, 0, 0))),
        )
        lowest.append(float(transform[2, 3]) - (float(sphere.get("radius")) * np.sqrt(max(0., 1. - transform[2, 2] ** 2)) + float(sphere.get("length")) * abs(transform[2, 2]) / 2.))
    if len(lowest) != 4:
        raise RuntimeError("D2W must have four cylindrical wheel collisions.")
    return -min(lowest)


@lru_cache(maxsize=1)
def get_d2w_kinematic_heights() -> tuple[float, float]:
    height = _pose_height(D2W_FIXSTAND_JOINT_POSITIONS)
    return height, height + 0.03


@lru_cache(maxsize=1)
def get_d2w_rl_reset_kinematic_height() -> float:
    return _pose_height(D2W_RL_DEFAULT_JOINT_POSITIONS)


def get_d2w_standing_height_target(*, require_ready: bool = False) -> float:
    calibration = get_d2w_asset_identity().manifest.get("calibration", {})
    if calibration.get("status") == "ready":
        value = float(calibration["standing_base_height_m"])
        if 0.15 <= value <= 0.60:
            return value
    if require_ready:
        raise RuntimeError("D2W runtime has no ready 30 second standing validation.")
    return get_d2w_kinematic_heights()[0]


def get_d2w_calibration_standing_pd_command_positions() -> dict[str, float]:
    return dict(D2W_FIXSTAND_JOINT_POSITIONS)


# Derived from names, never from a second manually maintained index list.
D2W_WHEEL_INDICES = tuple(D2W_CANONICAL_JOINT_ORDER.index(n) for n in D2W_WHEEL_JOINT_ORDER)
D2W_LEG_INDICES = tuple(D2W_CANONICAL_JOINT_ORDER.index(n) for n in D2W_LEG_JOINT_ORDER)
D2W_OBSERVATION_TERMS = (("base_ang_vel", 3), ("projected_gravity", 3),
                         ("velocity_commands", 3), ("joint_pos", 16),
                         ("joint_vel", 16), ("actions", 16))


def d2w_motor_contract(leg_spec, wheel_spec):
    """Portable motor identity; callers must validate their effective parameters."""
    wheel_path = Path(__file__).resolve().parents[4] / 'resources/d2w_description/config/robstride02_motor.yaml'
    return {
        "RS06": {"actuator": "d2w_robstride_actuator.RobStrideActuator",
                 "joint_names": list(D2W_LEG_JOINT_ORDER),
                 "model_sha256": sha256_file(leg_spec.config_path),
                 "protection_mode": leg_spec.protection_mode,
                 "peak_torque_nm": leg_spec.peak_torque_nm,
                 "max_speed_rad_s": leg_spec.max_speed_rad_s,
                 "armature": leg_spec.output_equivalent_inertia_kg_m2,
                 "friction": leg_spec.output_joint_friction_nm,
                 "envelope": [list(p) for p in zip(leg_spec.velocity_points, leg_spec.torque_points)],
                 "physics_dt": leg_spec.actuator_delay_physics_dt_s},
        "RS02": {"actuator": "d2w_robstride_actuator.RobStride02Actuator",
                 "joint_names": list(D2W_WHEEL_JOINT_ORDER),
                 "model_sha256": sha256_file(wheel_path),
                 "protection_mode": "instantaneous_envelope",
                 "peak_torque_nm": wheel_spec.peak_torque_nm,
                 "max_speed_rad_s": wheel_spec.max_speed_rad_s,
                 "armature": wheel_spec.armature, "friction": wheel_spec.friction,
                 "envelope": [list(p) for p in wheel_spec.envelope]},
    }


def d2w_policy_interface(*, dt, decimation, defaults, action_scale, history,
                         observation_scales, observation_clips, kp, kd,
                         delay_min, delay_max, wheel_spec, motor_contract):
    """D2W-only physical policy interface shared by export and deployment.

    Noise/randomization are training distributions, not deployment preprocessing.
    Values are supplied by the caller's actual config, not inferred from hashes.
    """
    return {
        "schema_version": 3,
        "motor_contract": motor_contract,
        "joint_order": list(D2W_CANONICAL_JOINT_ORDER),
        "control": {"physics_dt": float(dt), "decimation": int(decimation)},
        "default_joint_positions": [float(x) for x in defaults],
        "action_scale": [float(x) for x in action_scale],
        "action_modes": ["velocity" if i in D2W_WHEEL_INDICES else "position" for i in range(16)],
        "leg_processed_position_clip": [-100.0, 100.0],
        "wheel_target_speed_limit": float(wheel_spec.target_speed_rad_s),
        "kp": [float(x) for x in kp], "kd": [float(x) for x in kd],
        "delay": {"unit": "physics_steps", "min": int(delay_min), "max": int(delay_max),
                  "groups": [list(D2W_LEG_INDICES), list(D2W_WHEEL_INDICES)]},
        "policy_observation_history": int(history), "single_observation_dimension": 57,
        "history_layout": "term_major_oldest_first", "wheel_position_observation": "zero",
        "joint_velocity_reference": "zero", "last_action": "raw_policy_output",
        "observation_processing": "clip_then_scale",
        "observation_terms": [
            {"name": name, "width": width, "scale": float(scale),
             "clip": [float(x) for x in clip]}
            for (name, width), scale, clip in zip(D2W_OBSERVATION_TERMS, observation_scales, observation_clips, strict=True)
        ],
        "wheel_motor": {"max_speed_rad_s": wheel_spec.max_speed_rad_s,
                        "peak_torque_nm": wheel_spec.peak_torque_nm,
                        "envelope": [list(p) for p in wheel_spec.envelope],
                        "armature": wheel_spec.armature, "friction": wheel_spec.friction},
    }
