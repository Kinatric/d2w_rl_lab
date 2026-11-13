"""Upstream command sampler with passive parking reference bookkeeping."""
import torch
from d2w_rl_lab.cmd_utils.rl_gym_cmd import RLGymVelCommand
from .objectives.parking_reference import ParkingReference
from .objectives.focus_sampling import focus_commands

class D2WNativeCommand(RLGymVelCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.parking = ParkingReference(self.num_envs, self.device)
        self.quiet_age = torch.zeros(self.num_envs, device=self.device)
        self._parking_reset = False

    @property
    def commands(self):
        return self.command

    def _ids(self, ids):
        if ids is None or isinstance(ids, slice):
            return torch.arange(self.num_envs, device=self.device)[ids or slice(None)]
        return torch.as_tensor(ids, device=self.device, dtype=torch.long)

    def reset(self, env_ids=None):
        self._parking_reset = True
        try:
            return super().reset(env_ids)
        finally:
            self._parking_reset = False

    def _resample(self, env_ids):
        super()._resample(env_ids)
        ids = self._ids(env_ids)
        if len(ids) and hasattr(self, 'parking'):
            # Reserve some nonzero flat samples for high-speed straight motion;
            # never replace sampled zero commands or exceed terrain limits.
            if self._terrain_lookup is None:
                flat = torch.ones(len(ids), dtype=torch.bool, device=self.device)
            elif 'flat' in self._terrain_lookup.names:
                flat_index = self._terrain_lookup.names.index('flat')
                flat = self.current_terrain_ids[ids] == flat_index
            else:
                flat = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
            old = self.command[ids].clone()
            new, selected = focus_commands(old, flat, self.env_command_ranges['lin_vel_x'][ids, 1])
            selected_ids = ids[selected]
            if len(selected_ids):
                self._requested_velocity_command_b[selected_ids] = new[selected]
                self._apply_terrain_command_limits(selected_ids)
                self.commands_xy_accumulation[selected_ids] += self.command[selected_ids, :2] - old[selected, :2]
                self.last_is_limit_vel[selected_ids] = False
            # A subset of zero samples persists long enough to expose slow drift.
            zero = flat & (self.command[ids].abs().amax(dim=1) < 1.e-6)
            long_zero = zero & (torch.rand(len(ids), device=self.device) < .5)
            duration_u = torch.rand(len(ids), device=self.device)
            # Split the existing long-zero branch: half remain 8-12 s, half are 12-20 s.
            # Episode reset remains authoritative; do not force a long episode.
            hold_seconds = torch.where(duration_u < .5, 8. + 8. * duration_u, 12. + 16. * (duration_u - .5))
            self.time_left[ids[long_zero]] = hold_seconds[long_zero]
            if self._parking_reset:
                self.quiet_age[ids] = 0
            self.parking.update(ids, self.command[ids], self.robot.data.root_pos_w.torch[ids],
                self.robot.data.root_quat_w.torch[ids], self._parking_reset)

    def _update_command(self):
        super()._update_command()
        if not hasattr(self, 'parking'):
            return
        ids = self._ids(None)
        self.parking.update(ids, self.command, self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch)
        # Use the same strict-zero boundary as hold rewards and parking anchors.
        quiet = self.command.abs().amax(1) < 1.e-6
        self.quiet_age = torch.where(quiet, self.quiet_age+self._env.step_dt, torch.zeros_like(self.quiet_age))
