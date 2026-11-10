"""Upstream MoE-CTS defaults with a validated 16-action D2W mapping."""
from isaaclab.utils.configclass import configclass
from d2w_rl_lab.tasks.mixed.rsl_rl_cfg import ReferenceMoECTSRunnerCfg
from rsl_rl.moe_cts.config import MoeCtsSymmetryCfg

@configclass
class D2WMoECTSRunnerCfg(ReferenceMoECTSRunnerCfg):
    class_name = 'd2w_rl_lab.tasks.d2w.runner:D2WRunner'
    retention_path: str | None = None
    retention_sha256: str | None = None
    retention_coefficient: float = .2
    experiment_name = 'd2w_moe_cts_lab30'
    save_interval = 200
    def __post_init__(self):
        super().__post_init__()
        # Resume moments and noise; allow moderate exploration of stair behaviors.
        self.algorithm.class_name = 'd2w_rl_lab.tasks.d2w.learning:D2WRetentionMoECTS'
        self.algorithm.entropy_coef = .002
        # Preserve optimizer rates on resume; cap adaptive PPO LR without warmup.
        self.algorithm.learning_rate = 1.e-4
        self.algorithm.student_encoder_learning_rate = 1.e-4
        self.algorithm.symmetry_cfg = MoeCtsSymmetryCfg(
            use_symmetric_augmentation=True,
            symmetry_class='d2w_rl_lab.tasks.d2w.symmetry:D2WSymmetry')
