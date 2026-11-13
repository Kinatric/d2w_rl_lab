"""D2W-only passive calf diagnostics; inherited actuator output is untouched."""
from d2w_rl_lab.assets.d2w_robstride_actuator import RobStrideActuator
from .lateral_load_math import LoadAccumulator


class LoadObservedRobStrideActuator(RobStrideActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        names = [f'{leg}_calf_joint' for leg in ('FL', 'FR', 'RL', 'RR')]
        self.calf_local_ids = [self.joint_names.index(name) for name in names]
        self.calf_joint_names = names
        # Fixed low-speed reference avoids incentivizing spinning the calf to
        # cross the existing speed-derating breakpoint. Source is the canonical
        # YAML's conservative stall reference, not a measured D2W heat limit.
        self.calf_load = LoadAccumulator((self.computed_effort.shape[0], 4),
            self._device, reference_nm=self.motor_spec.stall_continuous_torque_nm)

    def compute(self, control_action, joint_pos, joint_vel):
        result = super().compute(control_action, joint_pos, joint_vel)
        self.calf_load.update(self.applied_effort[:, self.calf_local_ids],
                              joint_vel[:, self.calf_local_ids], float(self.cfg.physics_dt_s))
        return result

    def reset(self, env_ids):
        super().reset(env_ids)
        if hasattr(self, 'calf_load'):
            self.calf_load.reset(env_ids)
