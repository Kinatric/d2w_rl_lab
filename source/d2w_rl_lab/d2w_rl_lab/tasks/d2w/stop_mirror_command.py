"""A subset of moving flat samples continuously transitions into a longer stop.

Preserve sampled command directions/ranges and all already-zero samples. Motion
lasts 4--6 s, then strict zero lasts 12--20 s, capped by the remaining episode.
No policy history, action delay, or physical state is reset at the transition.
"""
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
            self.time_left[due_ids] = torch.minimum(12. + 8. * torch.rand(len(due_ids), device=self.device), remaining)
            self.parking.update(due_ids, self.command[due_ids], self.robot.data.root_pos_w.torch[due_ids],
                                self.robot.data.root_quat_w.torch[due_ids], False)
            self.paired_stop_events[due_ids] += 1
        c = self.command[ids]
        remaining = (self._env.max_episode_length - self._env.episode_length_buf[ids]) * self._env.step_dt
        # Eighteen seconds reserves the longest 6 s drive plus a 12 s hold.
        # Include reverse, lateral and turn commands, while retaining whatever
        # command the parent sampler chose and never replacing a zero sample.
        eligible = flat & ~due & ~zero_mask(c) & (remaining >= 18.)
        selected = eligible & (torch.rand(len(ids), device=self.device) < .5)
        selected_ids = ids[selected]
        self.stop_pending[selected_ids] = True
        self.time_left[selected_ids] = 4. + 2. * torch.rand(len(selected_ids), device=self.device)

    # D2WNativeCommand now owns the same strict-zero quiet_age update. Keeping
    # one implementation avoids applying different stop boundaries twice.
