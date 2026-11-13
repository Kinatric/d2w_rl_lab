import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab_physx.sensors import ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.sim import SimulationCfg
from isaaclab_tasks.utils import PresetCfg

@configclass
class D2WPhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=1048576, gpu_collision_stack_size=536870912, enable_external_forces_every_iteration=True)
    default = isaacsim_physx
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import d2w_rl_lab.tasks.d2w.mdp as d2w_mdp
import d2w_rl_lab.tasks.d2w.mdp_base as mdp
from d2w_rl_lab.assets.d2w import (
    D2W_CFG,
    D2W_RL_STANDING_HEIGHT_M,
    D2W_FIXSTAND_JOINT_POSITIONS,
    D2W_JOINT_NAMES,
    D2W_LEG_JOINT_ORDER,
)
from d2w_rl_lab.tasks.d2w.terrains import D2W_TERRAIN_CFG

JOINT_NAMES = D2W_JOINT_NAMES

BASE_LINK_NAME = 'base_link'
FOOT_LINK_NAME = ".*_foot"
BASE_HEIGHT_TARGET = D2W_RL_STANDING_HEIGHT_M

##
# Scene definition
##

@configclass
class D2WSceneCfg(InteractiveSceneCfg):
    """Go2 task scene with the canonical D2W articulation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=D2W_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False
    )

    robot: ArticulationCfg = D2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_small = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 灯光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = d2w_mdp.D2WCommandCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 腿部关节：位置控制
    joint_pos = d2w_mdp.D2WHybridActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={".*_(hip|thigh|calf)_joint": 0.25, ".*_foot_joint": 5.0},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=d2w_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=d2w_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1e-4,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.01,
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
            clip=(-100.0, 100.0),
            scale=1e-3,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=2.5,
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    single_obs: SingleObsCfg = SingleObsCfg() # Used to obtain the current-timestep observation for the MoE CTS model

@configclass
class EventCfg:
    """Configuration for events."""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",
            "offset_range": (-0.035, 0.035),
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6)
            }
        }
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # The joint_acc reward is not computed on the same scale in Gym and Lab.
    # In Gym, it is computed at the policy-step level,
    # while in Lab, it is computed at the physics-step level.
    # In Lab, the reward calculation is more precise, and because the L2 term is more sensitive to outliers.
    # Thus, the reward value is overall higher, so we need to decrease the weights to be suitable for Lab.
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(D2W_LEG_JOINT_ORDER))}
    )

    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(D2W_LEG_JOINT_ORDER))}
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(D2W_LEG_JOINT_ORDER))}
    )
    # Wheel-only initial weights: fan-ziqi/robot_lab Go2W, commit
    # 26acb225aad129e222da7a34ba11f39e29b73e39. D2 leg weights stay unchanged.
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-9,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")},
    )
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")},
    )
    joint_power_wheel = RewTerm(
        func=mdp.joint_power, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint")},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        }
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 5.0},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(D2W_LEG_JOINT_ORDER))},
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        # A rolling wheel body translates without contact slip.
        weight=0.0,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.hip_pos_penalty_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 1.0,
            "command_threshold": 0.1,
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,
        },
    )


    # D2W-specific soft preferences, using existing sensors only. These stay
    # active late in training but fade out near steps/slopes/obstacles.
    low_turn_rolling = RewTerm(
        func=d2w_mdp.low_turn_rolling, weight=0.6,
        params={"command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
            "height_cfg": SceneEntityCfg("height_scanner"),
            "settings": d2w_mdp.DEFAULT_MODE_SETTINGS.copy(),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"], body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
        },
    )
    fast_turn_steps = RewTerm(
        func=d2w_mdp.fast_turn_steps, weight=0.6,
        params={"command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
            "height_cfg": SceneEntityCfg("height_scanner"),
            "settings": d2w_mdp.DEFAULT_MODE_SETTINGS.copy(),
        },
    )
    fast_motion_stability = RewTerm(
        func=d2w_mdp.fast_motion_stability, weight=-0.5,
        params={"command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
            "height_cfg": SceneEntityCfg("height_scanner"),
            "settings": d2w_mdp.DEFAULT_MODE_SETTINGS.copy(),
        },
    )
    fast_straight_coordination = RewTerm(
        func=d2w_mdp.fast_straight_coordination, weight=-0.15,
        params={"command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
            "height_cfg": SceneEntityCfg("height_scanner"),
            "settings": d2w_mdp.DEFAULT_MODE_SETTINGS.copy(),
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(D2W_LEG_JOINT_ORDER), preserve_order=True),
        },
    )


    # Zero-command accuracy, front stance and effective high-turn swings.
    # Learning objectives only; action and motor limits are unchanged.
    quiet_stability = RewTerm(func=d2w_mdp.quiet_stability,weight=-1.0,
        params={"command_name":"base_velocity","asset_cfg":SceneEntityCfg("robot",joint_names=list(D2W_LEG_JOINT_ORDER),preserve_order=True)})
    front_stance = RewTerm(func=d2w_mdp.front_stance,weight=-0.2,
        params={"command_name":"base_velocity","sensor_cfg":SceneEntityCfg("contact_forces",body_names=["FL_foot","FR_foot","RL_foot","RR_foot"],preserve_order=True),
        "height_cfg":SceneEntityCfg("height_scanner"),"asset_cfg":SceneEntityCfg("robot",joint_names=list(D2W_LEG_JOINT_ORDER),preserve_order=True),"settings":d2w_mdp.DEFAULT_MODE_SETTINGS.copy()})
    turn_clearance = RewTerm(func=d2w_mdp.turn_clearance,weight=0.6,
        params={"command_name":"base_velocity","sensor_cfg":SceneEntityCfg("contact_forces",body_names=["FL_foot","FR_foot","RL_foot","RR_foot"],preserve_order=True),
        "height_cfg":SceneEntityCfg("height_scanner"),"asset_cfg":SceneEntityCfg("robot",body_names=["FL_foot","FR_foot","RL_foot","RR_foot"],preserve_order=True),"settings":d2w_mdp.DEFAULT_MODE_SETTINGS.copy(),"half_width":0.024675})

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Count true excessive tilt as a failed episode before post-fall action feedback grows.
    # Keep raw actions, rewards, body-contact termination and motor protection unchanged.
    fallen_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "threshold": 1.0
        },
    )
    motor_overload = DoneTerm(
        func=d2w_mdp.motor_overload,
        params={"actuator_name": "RS06"},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_gym)
    base_linear_velocity = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "lin_vel_z_l2", "initial_weight": -2.0, "final_weight": -0.0, "start_it": 0, "end_it": 1500
        })
    base_height_l2 = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "base_height_l2", "initial_weight": -1.0, "final_weight": -10.0, "start_it": 0, "end_it": 5000
        })

##
# Environment configuration
##

@configclass
class D2WEnvCfg(ManagerBasedRLEnvCfg):
    """D2W rough-terrain task frozen to the go2_rl_robotlab Go2 contract."""

    sim: SimulationCfg = SimulationCfg(physics=D2WPhysicsCfg(), use_newton_actuators=False)

    # Scene settings
    scene: D2WSceneCfg = D2WSceneCfg(num_envs=4096, env_spacing=0.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def configure_motor_protection(self):
        """Resolve the effective motor file before managers or identities are built."""
        from d2w_rl_lab.assets.d2w_robstride_motor import CUMULATIVE_OVERLOAD, load_motor_spec
        spec = load_motor_spec(self.scene.robot.actuators["RS06"].motor_config_path)
        self.terminations.motor_overload = (
            DoneTerm(func=d2w_mdp.motor_overload, params={"actuator_name": "RS06"})
            if spec.protection_mode == CUMULATIVE_OVERLOAD else None
        )

    def __post_init__(self):
        """Post initialization."""
        self.configure_motor_protection()
        # General settings
        self.decimation = 4
        self.episode_length_s = 25.0
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Physics material settings from subclass
        self.sim.physics_material = self.scene.terrain.physics_material

        # Update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner_small is not None:
            self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Handle curriculum for terrain generator
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class D2WCurrentEnvCfg(D2WEnvCfg):
    """Current effective objectives and original D2W terrain/command distribution."""
    def __post_init__(self):
        super().__post_init__()
        from .objectives.unified_objectives import configure
        configure(self)
        self.scene.num_envs = 8192
