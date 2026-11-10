"""D2W on the native Lab 3.0 stack; fresh initialization by default."""
import gymnasium as gym
gym.register(
    id="d2w-mixed-moe-cts",
    entry_point="d2w_rl_lab.tasks.d2w.env.d2w_env:D2WEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":"d2w_rl_lab.tasks.d2w.native_cfg:D2WNativeEnvCfg",
        "play_env_cfg_entry_point":"d2w_rl_lab.tasks.d2w.native_cfg:D2WNativePlayEnvCfg",
        "rsl_rl_cfg_entry_point":"d2w_rl_lab.tasks.d2w.rsl_rl_cfg:D2WMoECTSRunnerCfg",
    },
)
