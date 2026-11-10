import torch
from d2w_rl_lab.tasks.d2w.mdp import D2WRLGymCommand
from .parking_reference import ParkingReference

class StandingFocusCommand(D2WRLGymCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.phase=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.quiet_age=torch.zeros(self.num_envs,device=self.device)
        self.parking=ParkingReference(self.num_envs,self.device);self._parking_reset=False
    def reset(self,env_ids=None):
        self._parking_reset=True
        try:return super().reset(env_ids)
        finally:self._parking_reset=False
    def _resample(self,env_ids):
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if not len(ids):return
        prior=self.commands[ids].clone();phase=self.phase[ids].clone()
        super()._resample(ids);old=self.commands[ids].clone()
        if self._parking_reset:
            draw=torch.rand(len(ids),device=self.device)
            phase=torch.where(draw<.30,0,torch.where(draw<.90,1,3))
            self.quiet_age[ids]=0
        else:
            phase=torch.where(phase==1,2,phase)
        self.phase[ids]=phase
        hold=phase==0;drive=phase==1;brake=phase==2
        self.commands[ids[hold]]=0;self.time_left[ids[hold]]=25.
        if drive.any():
            speed=.3+1.7*torch.rand(int(drive.sum()),device=self.device)
            speed=torch.minimum(speed,self.env_command_ranges['lin_vel_x'][ids[drive],1].clamp_min(0))
            self.commands[ids[drive]]=0;self.commands[ids[drive],0]=speed;self.time_left[ids[drive]]=5.
        self.commands[ids[brake]]=prior[brake];self.time_left[ids[brake]]=20.
        self.last_is_limit_vel[ids[hold|drive|brake]]=False
        self.commands_xy_accumulation[ids]+=self.commands[ids,:2]-old[:,:2]
        self.parking.update(ids,self.commands[ids],self.robot.data.root_pos_w.torch[ids],self.robot.data.root_quat_w.torch[ids],self._parking_reset)
    def _update_command(self):
        super()._update_command()
        ids=(self.phase==2).nonzero().flatten()
        if len(ids):
            old=self.commands[ids].clone();rate=old.new_tensor([1.5,1.5,2.])
            self.commands[ids]=old.sign()*(old.abs()-rate*self._env.step_dt).clamp_min(0)
        all_ids=torch.arange(self.num_envs,device=self.device)
        self.parking.update(all_ids,self.commands,self.robot.data.root_pos_w.torch,self.robot.data.root_quat_w.torch)
        quiet=self.commands.abs().amax(1)<=.02
        self.quiet_age=torch.where(quiet,self.quiet_age+self._env.step_dt,torch.zeros_like(self.quiet_age))
