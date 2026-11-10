"""A subset of flat straight samples explicitly transitions into a sustained stop."""
import torch
from .native_command import D2WNativeCommand
from .objectives.consistent_gate import designated_flat
from .objectives.stationary_hold import zero_mask


class StopMirrorCommand(D2WNativeCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stop_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.paired_stop_events = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if not hasattr(self, 'stop_pending'):
            return
        ids = self._ids(env_ids)
        if not len(ids):
            return
        flat = designated_flat(self._env)[ids]
        due = self.stop_pending[ids].clone() & flat
        if self._parking_reset:
            due.zero_()
        self.stop_pending[ids] = False
        due_ids = ids[due]
        if len(due_ids):
            old = self.command[due_ids].clone()
            self._requested_velocity_command_b[due_ids] = 0.
            self._apply_terrain_command_limits(due_ids)
            self.commands_xy_accumulation[due_ids] += self.command[due_ids, :2] - old[:, :2]
            self.last_is_limit_vel[due_ids] = False
            self.quiet_age[due_ids] = 0.
            remaining = ((self._env.max_episode_length - self._env.episode_length_buf[due_ids])
                         * self._env.step_dt).clamp_min(self._env.step_dt)
            self.time_left[due_ids] = torch.minimum(8. + 4. * torch.rand(len(due_ids), device=self.device), remaining)
            self.parking.update(due_ids, self.command[due_ids], self.robot.data.root_pos_w.torch[due_ids],
                                self.robot.data.root_quat_w.torch[due_ids], False)
            self.paired_stop_events[due_ids] += 1
        c = self.command[ids]
        remaining = (self._env.max_episode_length - self._env.episode_length_buf[ids]) * self._env.step_dt
        eligible = flat & ~due & (c[:, 0].abs() >= 1.6) & (c[:, 1:].abs().amax(1) < 1.e-6) & (remaining >= 12.)
        selected = eligible & (torch.rand(len(ids), device=self.device) < .5)
        selected_ids = ids[selected]
        self.stop_pending[selected_ids] = True
        self.time_left[selected_ids] = 4. + 2. * torch.rand(len(selected_ids), device=self.device)

    def _update_command(self):
        previous_age = self.quiet_age.clone() if hasattr(self, 'quiet_age') else None
        super()._update_command()
        if previous_age is not None:
            # The hold penalty uses strict zero; a slow nonzero command must
            # not pre-age its two-second braking blend.
            self.quiet_age = torch.where(zero_mask(self.command), previous_age + self._env.step_dt,
                                         torch.zeros_like(previous_age))
