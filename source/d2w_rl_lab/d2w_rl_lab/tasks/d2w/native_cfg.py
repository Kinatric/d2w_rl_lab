"""D2W on the upstream Lab 3.0 terrain, command, physics, and learning stack.

Only robot-specific IO/control and selected current posture/parking objectives differ.
No legacy checkpoint or old iteration/terrain counter is loaded.
"""
from copy import deepcopy
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from d2w_rl_lab.assets.d2w import D2W_CFG, D2W_FIXSTAND_BASE_HEIGHT_M, D2W_LEG_JOINT_ORDER
from d2w_rl_lab.tasks.mixed.env_cfg import RobotEnvCfg, RobotSceneCfg, TerrainHeightScannerCfg
from .env_cfg import D2WCurrentEnvCfg, ObservationsCfg, ActionsCfg
from .native_command import D2WNativeCommand

@configclass
class D2WNativeSceneCfg(RobotSceneCfg):
    robot = D2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class D2WNativeEnvCfg(RobotEnvCfg):
    d2w_upstream_profile: bool = True
    scene: D2WNativeSceneCfg = D2WNativeSceneCfg(num_envs=8192, env_spacing=.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # The actor has 16 actions; use the corresponding D2W observation and motor contract.
        self.commands.base_velocity.class_type = D2WNativeCommand
        # Focus on true zero commands, including sustained stand and stop recovery.
        self.commands.base_velocity.zero_command_curriculum = {
            'start_iter': 0, 'end_iter': 1500, 'start_value': 0.30, 'end_value': 0.30,
        }
        self.commands.base_velocity.limit_ang_vel_at_zero_command_prob = 0.0
        # Explicit user-requested global range; retain terrain-specific lower caps.
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-2.0, 2.0)
        self.commands.base_velocity.command_range_curriculum = []
        self.commands.base_velocity.terrain_max_command_ranges['flat'] = {
            'lin_vel_x': (-2.0, 2.0), 'lin_vel_y': (-1.0, 1.0), 'ang_vel_yaw': (-2.0, 2.0),
        }
        current = D2WCurrentEnvCfg()
        current_rewards = vars(current.rewards)
        preserve_upstream = {
            'track_lin_vel_xy_exp','track_ang_vel_z_exp','lin_vel_z_l2','ang_vel_xy_l2',
            'action_rate_l2','action_smoothness_l2','joint_acc_l2','joint_power',
            'joint_torques_l2','base_height_l2','undesired_contacts','joint_pos_limits',
            'feet_regulation',
        }
        for name, term in current_rewards.items():
            if not name.startswith('_') and name not in preserve_upstream:
                setattr(self.rewards, name, deepcopy(term))
        for name in ('joint_acc_l2','joint_power','joint_torques_l2','joint_pos_limits'):
            getattr(self.rewards, name).params['asset_cfg'] = SceneEntityCfg('robot', joint_names=list(D2W_LEG_JOINT_ORDER), preserve_order=True)
        self.rewards.base_height_l2.params['target_height'] = D2W_FIXSTAND_BASE_HEIGHT_M
        self.rewards.base_height_l2.params['asset_cfg'] = SceneEntityCfg('robot', body_names='base_link')
        # Wheel-body translation is not contact slip; upstream foot-speed regularization is inappropriate.
        self.rewards.feet_regulation = None
        self.rewards.front_stance = None
        self.scene.robot.actuators = deepcopy(D2W_CFG.actuators)
        self.terminations.illegal_contact.params['sensor_cfg'] = SceneEntityCfg('contact_forces', body_names='base_link')
        self.terminations.bad_orientation.params['limit_angle'] = 1.0
        self.terminations.motor_overload = deepcopy(getattr(current.terminations, 'motor_overload', None))
        for name in ('randomize_rigid_body_mass_base','randomize_com_positions'):
            term = getattr(self.events, name, None)
            variants = [term] + [getattr(term, k, None) for k in ('default', 'newton_mjwarp', 'isaacsim_physx')]
            for variant in variants:
                if variant is not None and 'asset_cfg' in getattr(variant, 'params', {}):
                    variant.params['asset_cfg'] = SceneEntityCfg('robot', body_names='base_link')
        # Adapt only the body selector in the upstream Newton COM bucket implementation.
        physics = self.sim.physics
        choices = [physics] + [getattr(physics,k,None) for k in ('newton_mjwarp','isaacsim_physx','default')]
        for choice in choices:
            solver = getattr(choice, 'solver_cfg', None)
            buckets = getattr(solver, 'body_com_buckets', None)
            if buckets is not None:
                buckets.body_name_pattern = r'(?:^|/)Robot/base_link$'

        from .objectives.stationary_hold import configure
        configure(self)
        from .objectives.recovery_boundaries import configure as reconcile_boundaries
        reconcile_boundaries(self)
        from .objectives.stop_mirror import configure as refine_stop_mirror
        refine_stop_mirror(self)
        from .objectives.speed_stairs import configure as refine_speed_stairs
        refine_speed_stairs(self)
        from .objectives.refine30k import configure as refine_long_training
        refine_long_training(self)
        from .objectives.rollclimb import configure as refine_rollclimb
        refine_rollclimb(self)
        from .objectives.terrain_alignment import configure as refine_terrain_alignment
        refine_terrain_alignment(self)
        from .objectives.stair_coordination import configure as refine_stair_coordination
        refine_stair_coordination(self,start_index=64800)
        from .objectives.stair_sequence import configure as refine_stair_sequence
        refine_stair_sequence(self)

    def configure_motor_protection(self):
        # Configuration already selected above; keep the D2W env reset diagnostics hook.
        pass


@configclass
class D2WNativePlayEnvCfg(D2WNativeEnvCfg):
    """Upstream deterministic mixed-terrain playback with D2W hardware."""
    def __post_init__(self):
        super().__post_init__()
        from d2w_rl_lab.tasks.mixed.env_cfg import RobotPlayEnvCfg
        reference = RobotPlayEnvCfg()
        self.scene.num_envs = reference.scene.num_envs
        self.scene.terrain = deepcopy(reference.scene.terrain)
        # Retain D2W ranges and sampler. Upstream playback requests 3 m/s on
        # flat ground, outside this task's +/-2 m/s training distribution.
        self.curriculum.terrain_levels = None
