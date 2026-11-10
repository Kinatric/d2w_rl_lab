"""D2W environment reusing the verified Go2 action-manager behavior."""

from contextlib import nullcontext

import torch

from d2w_rl_lab.assets.d2w_robstride_motor import INSTANTANEOUS_ENVELOPE
from d2w_rl_lab.tasks.d2w import mdp as d2w_mdp
from d2w_rl_lab.tasks.d2w.env.base_env import D2WBaseEnv


class D2WEnv(D2WBaseEnv):
    """D2W quadruped locomotion environment."""

    def __init__(self, cfg, *args, **kwargs):
        if not getattr(cfg, "_d2w_preserve_disabled_terminations", False):
            cfg.configure_motor_protection()
        super().__init__(cfg, *args, **kwargs)

    def step(self, action):
        # Count the command paired with this action, before reset/resampling.
        # Metrics do not change commands, rewards, actions, or RNG draws.
        native = getattr(self.cfg, 'd2w_upstream_profile', False)
        if native:
            command = self.command_manager.get_command('base_velocity')
            pure_zero = (command == 0).all(dim=1).float().mean()
            fast_straight = ((command[:, 0] >= 1.6) & (command[:, 1:].abs().amax(dim=1) < 1.e-6)).float().mean()
        result = super().step(action)
        if native:
            self._zero_sample_sum = getattr(self, '_zero_sample_sum', 0.0) + pure_zero
            self._zero_sample_steps = getattr(self, '_zero_sample_steps', 0) + 1
            log = self.extras.setdefault('log', {})
            log['Sampling/pure_zero_fraction'] = pure_zero
            log['Sampling/pure_zero_fraction_cumulative'] = self._zero_sample_sum / self._zero_sample_steps
            log['Sampling/fast_straight_fraction'] = fast_straight
        return result

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Log and clear cumulative RS06 clipping metrics per completed episode."""
        diagnostics = d2w_mdp.rs06_episode_diagnostics(self, env_ids, reset=True)
        wheel_diagnostics = self.scene["robot"].actuators["RS02"].diagnostics(env_ids, reset=True)
        mode = self.scene["robot"].actuators["RS06"].motor_spec.protection_mode
        # A captured cold-start timeout entered NVRTC from the shared quaternion
        # reset math. Keep the same scripted formulas and random draws, but avoid
        # optimizing a new reset batch shape in the instantaneous runtime. The
        # context restores the previous executor setting, including on errors.
        reset_execution = torch.jit.optimized_execution(False) if mode == INSTANTANEOUS_ENVELOPE else nullcontext()
        with reset_execution:
            super()._reset_idx(env_ids)
        log = self.extras.setdefault("log", {})
        log["Metrics/motor_torque_clip_ratio"] = torch.mean(diagnostics.clip_ratio)
        log["Metrics/motor_torque_max_exceed_nm"] = torch.mean(diagnostics.max_exceed_nm)
        log["Metrics/wheel_torque_clip_ratio"] = torch.mean(wheel_diagnostics.clip_ratio)
        log["Metrics/wheel_torque_max_exceed_nm"] = torch.mean(wheel_diagnostics.max_exceed_nm)
