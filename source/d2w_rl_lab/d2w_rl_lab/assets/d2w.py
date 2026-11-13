# Copyright (c) 2024-2026 Ziqi Fan and contributors
# SPDX-License-Identifier: Apache-2.0

"""D2W quadruped articulation configuration."""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass

from d2w_rl_lab.assets.d2w_metadata import (
    D2W_CANONICAL_JOINT_ORDER,
    D2W_LEG_KP,
    D2W_LEG_KD,
    D2W_LEG_JOINT_ORDER,
    D2W_WHEEL_JOINT_ORDER,
    D2W_FIXSTAND_JOINT_POSITIONS,
    D2W_RL_DEFAULT_JOINT_POSITIONS,
    D2W_RL_STANDING_HEIGHT_M,
    get_d2w_asset_identity,
    get_d2w_kinematic_heights,
    get_d2w_rl_reset_kinematic_height,
)
from d2w_rl_lab.assets.d2w_robstride_actuator import RobStride06ActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from d2w_rl_lab.assets.d2w_robstride_actuator import RobStride02ActuatorCfg
from d2w_rl_lab.assets.d2w_wheel_motor import load_wheel_motor_spec
WHEEL_SPEC = load_wheel_motor_spec()


D2W_JOINT_NAMES = list(D2W_CANONICAL_JOINT_ORDER)
D2W_ASSET_IDENTITY = get_d2w_asset_identity()
D2W_FIXSTAND_BASE_HEIGHT_M, D2W_SPAWN_HEIGHT_M = get_d2w_kinematic_heights()
D2W_NOMINAL_BASE_HEIGHT_M = D2W_FIXSTAND_BASE_HEIGHT_M
D2W_RL_RESET_BASE_HEIGHT_M = get_d2w_rl_reset_kinematic_height()


# Isaac's rigid-body angular-velocity setting is degrees/s. The inherited
# 1000 deg/s ceiling is only 17.45 rad/s and spuriously brakes a rolling wheel
# before the RS02 joint limit (42.94 rad/s). Keep every non-wheel body unchanged.
D2W_LINK_ANGULAR_LIMIT_DEG_S = 1000.0
D2W_WHEEL_BODY_ANGULAR_LIMIT_DEG_S = D2W_LINK_ANGULAR_LIMIT_DEG_S + math.degrees(WHEEL_SPEC.max_speed_rad_s)


def spawn_d2w_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Allow wheel spin plus carrier rotation, retaining all motor protections.

    Apply before outer cloning so every environment inherits the same settings.
    This is a rigid-body numerical ceiling, not the relative joint/motor limit.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        sim_utils.modify_rigid_body_properties(
            f"{prim_path}/{name}",
            sim_utils.RigidBodyPropertiesCfg(max_angular_velocity=D2W_WHEEL_BODY_ANGULAR_LIMIT_DEG_S),
        )
    # D2W motor housings have millimetre-scale gaps. The Go2 terrain margin
    # (1 cm outward inflation) creates artificial body/thigh self-penetration.
    # Preserve exact URDF non-wheel collision surfaces; retain the upstream
    # wheel/terrain margins and all self-collision pairs and termination tests.
    # Overrides are authored on the transient scene layer, not the source USD.
    from pxr import Sdf, Usd, UsdPhysics
    for branch in list(Usd.PrimRange(prim)):
        if branch.GetName() == 'collisions' and not branch.GetParent().GetName().endswith('_foot'):
            branch.SetInstanceable(False)
            for collider in Usd.PrimRange(branch):
                if collider.HasAPI(UsdPhysics.CollisionAPI):
                    collider.AddAppliedSchema('NewtonCollisionAPI')
                    collider.CreateAttribute('newton:contactMargin', Sdf.ValueTypeNames.Float, custom=False).Set(0.0)
    if not cfg.visuals_enabled:
        from d2w_rl_lab.assets.usd_utils import _deactivate_visual_branches
        _deactivate_visual_branches(prim)
    return prim


@configclass
class D2WUsdFileCfg(sim_utils.UsdFileCfg):
    visuals_enabled: bool = True


D2W_CFG = ArticulationCfg(
    spawn=D2WUsdFileCfg(
        func=spawn_d2w_usd,
        # The manifest points at usd/<canonical-hash-prefix>/d2w.usd and verifies
        # its digest before this configuration is constructed.  Never fall back
        # to the archived model_99 USD or an unversioned importer output.
        usd_path=str(D2W_ASSET_IDENTITY.isaac_usd_path),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=D2W_LINK_ANGULAR_LIMIT_DEG_S,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn height is the preserved FixStand FK height plus 3 cm.  The
        # articulation itself resets at the exact Go2 RL default pose.
        pos=(0.0, 0.0, D2W_SPAWN_HEIGHT_M),
        joint_pos=dict(D2W_RL_DEFAULT_JOINT_POSITIONS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "RS02": RobStride02ActuatorCfg(
            joint_names_expr=list(D2W_WHEEL_JOINT_ORDER),
            stiffness=WHEEL_SPEC.kp, damping=WHEEL_SPEC.kd, actuator_effort_limit=WHEEL_SPEC.peak_torque_nm, joint_effort_limit=WHEEL_SPEC.peak_torque_nm,
            actuator_velocity_limit=WHEEL_SPEC.max_speed_rad_s, joint_velocity_limit=WHEEL_SPEC.max_speed_rad_s,
            armature=WHEEL_SPEC.armature, friction=WHEEL_SPEC.friction, min_delay=0, max_delay=4,
        ),
        "RS06": RobStride06ActuatorCfg(
            joint_names_expr=list(D2W_LEG_JOINT_ORDER),
            # Match the later D2W gait-training PD baseline; retain the
            # independent motor envelope and existing delay distribution.
            stiffness=D2W_LEG_KP,
            damping=D2W_LEG_KD,
            min_delay=0,
            max_delay=4,
        ),
    },
)
