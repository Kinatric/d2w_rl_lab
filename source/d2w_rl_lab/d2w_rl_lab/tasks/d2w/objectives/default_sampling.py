import torch
from d2w_rl_lab.tasks.d2w.mdp import D2WRLGymCommand
from .parking_reference import ParkingReference

class DefaultCommandWithParking(D2WRLGymCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.parking=ParkingReference(self.num_envs,self.device)
        self.quiet_age=torch.zeros(self.num_envs,device=self.device)
        self._parking_reset=False
    def reset(self,env_ids=None):
        self._parking_reset=True
        try:return super().reset(env_ids)
        finally:self._parking_reset=False
    def _resample(self,env_ids):
        super()._resample(env_ids)
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if len(ids):
            if self._parking_reset:self.quiet_age[ids]=0
            self.parking.update(ids,self.commands[ids],self.robot.data.root_pos_w.torch[ids],self.robot.data.root_quat_w.torch[ids],self._parking_reset)
    def _update_command(self):
        super()._update_command()
        ids=torch.arange(self.num_envs,device=self.device)
        self.parking.update(ids,self.commands,self.robot.data.root_pos_w.torch,self.robot.data.root_quat_w.torch)
        quiet=self.commands.abs().amax(1)<=.02
        self.quiet_age=torch.where(quiet,self.quiet_age+self._env.step_dt,torch.zeros_like(self.quiet_age))
