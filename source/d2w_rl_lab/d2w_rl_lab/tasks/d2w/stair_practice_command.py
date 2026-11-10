"""Stage-local stair practice within truthful +/-2m/s task command limits."""
import torch
from .stop_mirror_command import StopMirrorCommand
from .objectives.stair_coordination import practice_commands
from .objectives.stair_probe import add_probe
from d2w_rl_lab.utils.terrains import get_terrain_masks



def next_practice_cap(cap, count, success):
    """Adjust only after enough overall and student training windows exist."""
    if count[0] < 128 or count[1] < 64:
        return cap
    overall, student = success[0] / count[0], success[1] / count[1]
    if overall >= .8 and student >= .8:
        return min(2., round(cap + .1, 6))
    if student < .2:
        return max(.6, round(cap - .1, 6))
    return cap


class StairPracticeCommand(StopMirrorCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.practice_samples=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.practice_selected=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.probe_selected=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.performance_cap = .8
        self.stage_step = int(env.common_step_counter)
        self.student_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.window_steps = torch.zeros(self.num_envs, device=self.device)
        self.window_good = torch.zeros_like(self.window_steps)
        self.window_ascent = torch.zeros_like(self.window_steps)
        self.window_height = torch.zeros_like(self.window_steps)
        self.evidence_count = torch.zeros(2, device=self.device)
        self.evidence_success = torch.zeros(2, device=self.device)

    def curriculum_state(self):
        return dict(cap=self.performance_cap, stage_step=self.stage_step,
                    count=self.evidence_count.cpu().tolist(), success=self.evidence_success.cpu().tolist())

    def restore_curriculum(self, state):
        cap=float(state['cap'])
        if not .6 <= cap <= 2.: raise ValueError('Invalid D2W practice cap')
        self.performance_cap=cap;self.stage_step=int(state['stage_step'])
        self.evidence_count.copy_(self.evidence_count.new_tensor(state['count']))
        self.evidence_success.copy_(self.evidence_success.new_tensor(state['success']))

    def observe_ascent(self, robot, stairs, slope, valid):
        """Two-second upper-speed windows, separately counted for student rollouts.

        This is a training readiness proxy, not a summit-success certificate.
        Called from the reward path before reset; falls cannot disappear on reset.
        """
        c=self.command
        eligible=self.practice_selected & stairs & (c[:,0]>=self.performance_cap-.15) & valid
        start=eligible & (self.window_steps==0)
        z=robot.data.root_pos_w.torch[:,2]
        self.window_height=torch.where(start,z,self.window_height)
        v=robot.data.root_lin_vel_b.torch;g=robot.data.root_ang_vel_b.torch
        good=(v[:,0]>=.75*c[:,0]) & (v[:,1].abs()<.15) & (g[:,2].abs()<.2)
        good &= (-robot.data.projected_gravity_b.torch[:,2]>.8253356)
        good &= g[:,:2].norm(dim=-1)<2.
        self.window_steps=torch.where(eligible,self.window_steps+1,0.)
        self.window_good=torch.where(eligible,self.window_good+good,0.)
        self.window_ascent=torch.where(eligible,self.window_ascent+(slope>.03),0.)
        failed=self._env.termination_manager.terminated
        finished=eligible & ((self.window_steps*self._env.step_dt>=2.) | failed)
        # Only actual ascending windows qualify; early falls on ascent are failures.
        evidence=finished & (self.window_ascent>=.5*self.window_steps)
        success=evidence & (~failed) & (self.window_good>=.9*self.window_steps) & (z-self.window_height>.05)
        self.evidence_count += torch.stack((evidence.sum(),(evidence & self.student_mask).sum()))
        self.evidence_success += torch.stack((success.sum(),(success & self.student_mask).sum()))
        for name in ['window_steps','window_good','window_ascent']:
            value=getattr(self,name);value[finished]=0.
        self.metrics['stair_practice_performance_cap']=c.new_full((self.num_envs,),self.performance_cap)
        rate=self.evidence_success/self.evidence_count.clamp_min(1)
        self.metrics['stair_practice_readiness']=rate[0].expand(self.num_envs).clone()
        self.metrics['stair_practice_student_readiness']=rate[1].expand(self.num_envs).clone()

    def _advance_performance_cap(self):
        step=int(self._env.common_step_counter)
        # Never advance based only on elapsed iterations. The iteration ceiling
        # still applies in practice_commands; use the configured rollout length.
        if step-self.stage_step < 500*self.cfg.num_steps_per_iter:return
        self.stage_step=step
        count=self.evidence_count.tolist();success=self.evidence_success.tolist()
        if count[0]<128 or count[1]<64:return
        previous_cap=self.performance_cap
        self.performance_cap=next_practice_cap(previous_cap,count,success)
        if self.performance_cap != previous_cap:
            self.window_steps.zero_();self.window_good.zero_();self.window_ascent.zero_()
        self.stage_step=step;self.evidence_count.zero_();self.evidence_success.zero_()

    def _resample(self,env_ids):
        super()._resample(env_ids)
        if not hasattr(self,'practice_samples'):return
        self._advance_performance_cap()
        ids=self._ids(env_ids)
        if not len(ids):return
        masks=get_terrain_masks(self._env);zero=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        stairs=(masks.get('stairs_up',zero)|masks.get('stairs_down',zero))[ids]
        old=self.command[ids].clone()
        elapsed=old.new_full((len(ids),),max(0.,self._env.common_step_counter/self.cfg.num_steps_per_iter-float(self.cfg.stair_practice_start_index)))
        new,selected,cap=practice_commands(old,stairs,elapsed,torch.rand(len(ids),device=self.device),torch.rand(len(ids),device=self.device),performance_cap=self.performance_cap)
        new,probe=add_probe(old,new,stairs,selected,torch.rand(len(ids),device=self.device),torch.rand(len(ids),device=self.device))
        self.probe_selected[ids]=probe
        self.metrics['stair_highspeed_probe_fraction']=self.probe_selected.float()
        changed=stairs&old.abs().amax(-1).gt(1.e-6);changed_ids=ids[changed]
        if len(changed_ids):
            self._requested_velocity_command_b[changed_ids]=new[changed]
            self._apply_terrain_command_limits(changed_ids)
            self.commands_xy_accumulation[changed_ids]+=self.command[changed_ids,:2]-old[changed,:2]
            self.last_is_limit_vel[changed_ids]=False
        self.practice_selected[ids]=selected;self.practice_samples[ids[selected]]+=1
        self.window_steps[ids]=0.;self.window_good[ids]=0.;self.window_ascent[ids]=0.
        self.metrics['stair_practice_selected']=self.practice_selected.float()
        self.metrics['stair_practice_speed_cap']=self.command[:,0].new_full((self.num_envs,),float(cap[0]))
