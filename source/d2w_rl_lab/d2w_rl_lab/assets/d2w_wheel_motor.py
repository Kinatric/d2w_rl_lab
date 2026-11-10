"""D2W RS02 manual parameters and approximate 48 V envelope; no thermal model."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import yaml
import numpy as np

@dataclass(frozen=True)
class WheelMotorSpec:
    peak_torque_nm: float
    max_speed_rad_s: float
    target_speed_rad_s: float
    kp: float
    kd: float
    armature: float
    friction: float
    envelope: tuple[tuple[float, float], ...]

@lru_cache(maxsize=1)
def load_wheel_motor_spec() -> WheelMotorSpec:
    path=Path(__file__).resolve().parents[4]/'resources/d2w_description/config/robstride02_motor.yaml'
    raw=yaml.safe_load(path.read_text())
    if raw.get('schema_version') != 1 or raw.get('model') != 'RobStride_02' or raw.get('protection_mode') != 'instantaneous_envelope':
        raise ValueError('Unsupported D2W RS02 simulation contract')
    limits=raw['output_limits'];assumptions=raw['assumptions']
    spec=WheelMotorSpec(*(float(v) for v in (
        limits['peak_torque_Nm'],limits['sim_velocity_limit_rad_s'],
        limits['initial_target_velocity_clip_rad_s'],limits['kp_sim_initial'],limits['kd_sim_initial'],
        assumptions['armature'],assumptions['friction'])), envelope=tuple(
        (float(rpm)*2*math.pi/60, float(torque)) for rpm,torque in raw['peak_torque_speed_envelope_rpm']))
    if not all(math.isfinite(v) and v>=0 for k,v in vars(spec).items() if k != "envelope"):
        raise ValueError('RS02 parameters must be finite and nonnegative')
    if not (0 < spec.target_speed_rad_s <= spec.max_speed_rad_s) or spec.peak_torque_nm<=0 or spec.kp!=0:
        raise ValueError('RS02 requires a bounded velocity-controlled wheel interface')
    points=np.asarray(spec.envelope)
    if points.ndim != 2 or points.shape[1] != 2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('RS02 envelope must contain finite speed/torque pairs')
    if (points[0,0] != 0 or points[0,1] != spec.peak_torque_nm or points[-1,1] != 0
        or not np.isclose(points[-1,0],spec.max_speed_rad_s)
        or np.any(np.diff(points[:,0])<=0) or np.any(np.diff(points[:,1])>0) or np.any(points<0)):
        raise ValueError('RS02 envelope must decrease from peak to zero at no-load speed')
    return spec


def wheel_torque_limits(speed_rad_s, spec=None):
    """Same symmetric magnitude-envelope approximation as D2; no thermal model."""
    spec=spec or load_wheel_motor_spec()
    points=np.asarray(spec.envelope)
    return np.interp(np.abs(speed_rad_s), points[:,0], points[:,1], right=0.0)
