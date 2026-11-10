"""Pure-Python, fail-closed D2W joint-adapter identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import yaml


D2W_JOINT_ADAPTER_SCHEMA_VERSION = 2
D2W_SIMULATION_MAPPING_SCHEMA_VERSION = 1
D2W_SIMULATION_MAPPING_VERSION = "d2w_simulation_joint_adapter_v1"
D2W_HARDWARE_MAPPING_SCHEMA_VERSION = 2
D2W_HARDWARE_MAPPING_VERSION = "d2w_hardware_joint_adapter_v2"
D2W_MODEL_JOINT_ORDER = ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_joint')


def _default_adapter_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "resources" / "d2w_description" / "config" / "joint_adapter.yaml"
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().with_name("joint_adapter.yaml")


D2W_JOINT_ADAPTER_PATH = _default_adapter_path()


@dataclass(frozen=True)
class D2WHardwareJointIdentity:
    joint_name: str
    bus: str | None
    can_id: int | None
    mcu_uuid: str | None
    firmware: str | None
    zero_offset_rad: float | None
    direction_sign: int | None

    def canonical_dict(self) -> dict[str, object]:
        return {
            "bus": self.bus,
            "can_id": self.can_id,
            "mcu_uuid": self.mcu_uuid,
            "firmware": self.firmware,
            "zero_offset_rad": self.zero_offset_rad,
            "direction_sign": self.direction_sign,
        }


@dataclass(frozen=True)
class D2WJointAdapterIdentity:
    path: Path
    file_sha256: str
    schema_version: int
    simulation_schema_version: int
    simulation_mapping_version: str
    simulation_sha256: str
    model_order: tuple[str, ...]
    mujoco_order: tuple[str, ...]
    simulation_zero_offset_rad: tuple[float, ...]
    simulation_direction_sign: tuple[int, ...]
    hardware_schema_version: int
    hardware_mapping_version: str
    hardware_sha256: str
    hardware_status: str
    hardware_protocol: str | None
    hardware_joints: tuple[D2WHardwareJointIdentity, ...]

    def simulation_manifest_identity(self) -> dict[str, object]:
        """Return only policy-relevant model-to-simulation semantics."""
        return {
            "schema_version": self.simulation_schema_version,
            "mapping_version": self.simulation_mapping_version,
            "sha256": self.simulation_sha256,
        }

    def hardware_manifest_identity(self) -> dict[str, object]:
        """Return the separately versioned physical commissioning identity."""
        return {
            "schema_version": self.hardware_schema_version,
            "mapping_version": self.hardware_mapping_version,
            "sha256": self.hardware_sha256,
            "status": self.hardware_status,
        }

    @property
    def model_to_mujoco(self) -> tuple[int, ...]:
        return tuple(self.model_order.index(name) for name in self.mujoco_order)

    @property
    def mujoco_to_model(self) -> tuple[int, ...]:
        return tuple(self.mujoco_order.index(name) for name in self.model_order)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _order(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 16 or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"D2W joint adapter {field} must be a 16-name list.")
    result = tuple(value)
    if len(set(result)) != 16 or set(result) != set(D2W_MODEL_JOINT_ORDER):
        raise RuntimeError(f"D2W joint adapter {field} is not a permutation of the canonical model joints.")
    return result


def _simulation_zero_offsets(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 16:
        raise RuntimeError("D2W joint adapter simulation.zero_offset_rad must contain 16 values.")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("D2W simulation zero offsets contain a non-numeric value.") from exc
    if not all(math.isfinite(item) for item in result) or any(item != 0.0 for item in result):
        raise RuntimeError("D2W simulation zero offsets must remain the commissioned all-zero transform.")
    return result


def _simulation_direction_signs(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != 16:
        raise RuntimeError("D2W joint adapter simulation.direction_sign must contain 16 values.")
    if any(isinstance(item, bool) or item != 1 for item in value):
        raise RuntimeError("D2W simulation direction signs must remain the commissioned all-+1 transform.")
    return tuple(1 for _ in value)


def _hardware_joint(
    joint_name: str,
    value: object,
    *,
    commissioned: bool,
) -> D2WHardwareJointIdentity:
    fields = ("bus", "can_id", "mcu_uuid", "firmware", "zero_offset_rad", "direction_sign")
    if not isinstance(value, dict) or set(value) != set(fields):
        raise RuntimeError(f"D2W hardware joint {joint_name} must define exactly {fields}.")
    if not commissioned:
        if any(value[field] is not None for field in fields):
            raise RuntimeError(f"Uncommissioned D2W hardware joint {joint_name} must keep all identity fields null.")
        return D2WHardwareJointIdentity(joint_name, None, None, None, None, None, None)

    bus = value["bus"]
    can_id = value["can_id"]
    mcu_uuid = value["mcu_uuid"]
    firmware = value["firmware"]
    zero_offset = value["zero_offset_rad"]
    direction = value["direction_sign"]
    if not isinstance(bus, str) or not bus.strip():
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} has no CAN bus.")
    if isinstance(can_id, bool) or not isinstance(can_id, int) or not 0 <= can_id <= 255:
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} has an invalid 8-bit CAN ID.")
    if not isinstance(mcu_uuid, str) or not mcu_uuid.strip():
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} has no MCU UUID.")
    if not isinstance(firmware, str) or not firmware.strip():
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} has no firmware identity.")
    try:
        zero = float(zero_offset)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} has an invalid zero offset.") from exc
    if not math.isfinite(zero):
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} has a non-finite zero offset.")
    if isinstance(direction, bool) or direction not in (-1, 1):
        raise RuntimeError(f"Commissioned D2W hardware joint {joint_name} direction must be +/-1.")
    return D2WHardwareJointIdentity(
        joint_name,
        bus.strip(),
        can_id,
        mcu_uuid.strip(),
        firmware.strip(),
        zero,
        int(direction),
    )


def load_d2w_joint_adapter(
    path: str | Path = D2W_JOINT_ADAPTER_PATH,
    *,
    mode: str = "simulation",
) -> D2WJointAdapterIdentity:
    """Load both identities; hardware mode rejects uncommissioned mappings."""

    if mode not in {"simulation", "hardware"}:
        raise ValueError(f"Unsupported D2W joint-adapter mode: {mode!r}")
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != D2W_JOINT_ADAPTER_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported D2W joint-adapter schema: {resolved}")

    simulation = raw.get("simulation")
    hardware = raw.get("hardware")
    if not isinstance(simulation, dict) or not isinstance(hardware, dict):
        raise RuntimeError("D2W joint adapter is missing simulation/hardware contracts.")
    if simulation.get("schema_version") != D2W_SIMULATION_MAPPING_SCHEMA_VERSION:
        raise RuntimeError("Unsupported D2W simulation joint-adapter schema.")
    if simulation.get("mapping_version") != D2W_SIMULATION_MAPPING_VERSION:
        raise RuntimeError(f"Unsupported D2W simulation mapping version: {simulation.get('mapping_version')!r}")
    model_order = _order(simulation.get("model_order"), "simulation.model_order")
    if model_order != D2W_MODEL_JOINT_ORDER:
        raise RuntimeError("D2W simulation model order differs from FL/FR/RL/RR policy order.")
    if simulation.get("mujoco_status") != "commissioned":
        raise RuntimeError("D2W MuJoCo joint adapter is not commissioned.")
    mujoco_order = _order(simulation.get("mujoco_order"), "simulation.mujoco_order")
    zero = _simulation_zero_offsets(simulation.get("zero_offset_rad"))
    signs = _simulation_direction_signs(simulation.get("direction_sign"))
    simulation_contract = {
        "schema_version": D2W_SIMULATION_MAPPING_SCHEMA_VERSION,
        "mapping_version": D2W_SIMULATION_MAPPING_VERSION,
        "model_order": list(model_order),
        "mujoco_status": "commissioned",
        "mujoco_order": list(mujoco_order),
        "zero_offset_rad": list(zero),
        "direction_sign": list(signs),
    }

    if hardware.get("schema_version") != D2W_HARDWARE_MAPPING_SCHEMA_VERSION:
        raise RuntimeError("Unsupported D2W hardware joint-adapter schema.")
    if hardware.get("mapping_version") != D2W_HARDWARE_MAPPING_VERSION:
        raise RuntimeError(f"Unsupported D2W hardware mapping version: {hardware.get('mapping_version')!r}")
    hardware_status = str(hardware.get("status", ""))
    if hardware_status not in {"uncommissioned", "commissioned"}:
        raise RuntimeError(f"Unknown D2W hardware mapping status: {hardware_status!r}")
    commissioned = hardware_status == "commissioned"
    protocol = hardware.get("protocol")
    if commissioned:
        if not isinstance(protocol, str) or not protocol.strip():
            raise RuntimeError("Commissioned D2W hardware mapping must name its CAN protocol.")
        protocol = protocol.strip()
    elif protocol is not None:
        raise RuntimeError("Uncommissioned D2W hardware protocol must remain null.")
    joints = hardware.get("joints")
    if not isinstance(joints, dict) or set(joints) != set(D2W_MODEL_JOINT_ORDER):
        raise RuntimeError("D2W hardware mapping must provide exactly the 16 canonical joint names.")
    hardware_joints = tuple(
        _hardware_joint(name, joints[name], commissioned=commissioned) for name in D2W_MODEL_JOINT_ORDER
    )
    if commissioned:
        bus_ids = [(joint.bus, joint.can_id) for joint in hardware_joints]
        uuids = [joint.mcu_uuid for joint in hardware_joints]
        if len(set(bus_ids)) != 16:
            raise RuntimeError("Commissioned D2W hardware mapping has duplicate (bus, CAN ID) identities.")
        if len(set(uuids)) != 16:
            raise RuntimeError("Commissioned D2W hardware mapping has duplicate MCU UUID identities.")
    hardware_contract = {
        "schema_version": D2W_HARDWARE_MAPPING_SCHEMA_VERSION,
        "mapping_version": D2W_HARDWARE_MAPPING_VERSION,
        "status": hardware_status,
        "protocol": protocol,
        "joints": {joint.joint_name: joint.canonical_dict() for joint in hardware_joints},
    }

    if mode == "hardware" and not commissioned:
        raise RuntimeError(
            "D2W hardware joint adapter is uncommissioned; bus/CAN ID/UUID/firmware/zero/sign are unknown."
        )
    return D2WJointAdapterIdentity(
        path=resolved,
        file_sha256=_sha256_file(resolved),
        schema_version=D2W_JOINT_ADAPTER_SCHEMA_VERSION,
        simulation_schema_version=D2W_SIMULATION_MAPPING_SCHEMA_VERSION,
        simulation_mapping_version=D2W_SIMULATION_MAPPING_VERSION,
        simulation_sha256=_semantic_sha256(simulation_contract),
        model_order=model_order,
        mujoco_order=mujoco_order,
        simulation_zero_offset_rad=zero,
        simulation_direction_sign=signs,
        hardware_schema_version=D2W_HARDWARE_MAPPING_SCHEMA_VERSION,
        hardware_mapping_version=D2W_HARDWARE_MAPPING_VERSION,
        hardware_sha256=_semantic_sha256(hardware_contract),
        hardware_status=hardware_status,
        hardware_protocol=protocol,
        hardware_joints=hardware_joints,
    )
