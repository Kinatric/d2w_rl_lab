from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from d2w_rl_lab.tasks.d2w.manager.action_manager import ActionManagerD2W

class D2WBaseEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        super().load_managers()
        # override action manager
        self.action_manager = ActionManagerD2W(self.cfg.actions, self)
        print("[D2WBaseEnv-INFO] Overriding action manager with ActionManagerD2W: ", self.action_manager)
