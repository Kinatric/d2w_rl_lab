# RSL-RL for D2W

Vendored RSL-RL 3.3.0 retains PPO, MoE-CTS, symmetry, export and logging. Install this version with `python -m pip install --no-deps -e source/rsl_rl`.

The D2W task selects `d2w_rl_lab.tasks.d2w.runner:D2WRunner` and `d2w_rl_lab.tasks.d2w.learning:D2WRetentionMoECTS`. See the root README for runtime and continuation commands. PPO remains available as implementation code but has no standalone registered task here.

RoboGauge is disabled by default. Enabling it requires an explicitly registered D2W-compatible server task. The former Go2 task name must not be silently reused.

Based on [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl), extended by [wty-yy/go2_rl_lab](https://github.com/wty-yy/go2_rl_lab). Preserve LICENSE, CONTRIBUTORS.md, CITATION.cff and licenses/.

## Kinatric modifications

Kinatric-owned modifications are GPL-3.0-or-later; upstream RSL-RL and Tianyang Wu contributions retain their original notices in LICENSE and licenses/. See [package coverage](LICENSING.md), [GPL text](COPYING), and [third-party notices](THIRD_PARTY_NOTICES.md).
