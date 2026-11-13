# D2W training code map

All paths below are relative to `source/d2w_rl_lab/d2w_rl_lab/`.

| Entry | Responsibility |
|---|---|
| `tasks/d2w/native_cfg.py` | Effective Newton/PhysX environment; inherited terrain settings and ordered objective composition |
| `tasks/d2w/env_cfg.py` | Robot observation/action groups, base reward/event/termination definitions |
| `tasks/d2w/native_command.py` | Native command curriculum and extended zero-command intervals |
| `tasks/d2w/stair_practice_command.py` | Focused stair sampling, performance cap, student-specific curriculum evidence |
| `tasks/d2w/objectives/stair_probe.py` | Eligible high-speed probe sampling, excluded from promotion evidence |
| `tasks/d2w/objectives/stair_sequence.py` | Front/rear placement and support-related objective composition |
| `tasks/d2w/objectives/terrain_alignment.py` | Terrain-conditioned limb alignment and support diagnostics |
| `tasks/d2w/objectives/stair_coordination.py` | Stair transfer, command practice and coordination building blocks |
| `tasks/d2w/objectives/stair_traction.py` | Persistent loaded stair blockage and support-aware unloading cost |
| `tasks/d2w/objectives/flat_speed_tracking.py`, `flat_speed_extension.py` | Flat speed deficit shaping and straight-only forward exploration to 2.6 m/s |
| `tasks/d2w/objectives/focus_sampling.py`, `tasks/d2w/stop_mirror_command.py` | Focused flat commands and continuous move-to-stop episodes |
| `tasks/d2w/objectives/lateral_load.py`, `lateral_load_math.py`, `load_observer.py` | Passive delivered-torque integration and support-aware lateral load relief |
| `tasks/d2w/objectives/stationary_hold.py` | Stationary hold objectives |
| `tasks/d2w/objectives/recovery_boundaries.py` | Reward applicability and recovery boundaries |
| `tasks/d2w/learning.py` | Teacher/student parking-retention losses and adaptive PPO LR ceiling |
| `tasks/d2w/runner.py` | Continuation metadata, two optimizer states, replay provenance/RNG, finite checkpoint checks and budget |
| `tasks/d2w/rsl_rl_cfg.py` | Active algorithm, network, entropy, mirror augmentation and save interval |
| `tasks/d2w/symmetry.py` | 16-action D2W reflection mapping |
| `tasks/d2w/env/`, `manager/` | Environment step/reset behavior and hybrid action execution |
| `assets/d2w_metadata.py` | Canonical ordering, resource identity and interface definitions |
| `assets/d2w_robstride_actuator.py` | Actual delayed leg/wheel effort implementation |
| `assets/d2w_robstride_motor.py`, `assets/d2w_wheel_motor.py` | RS06/RS02 envelopes and protection contracts |

Other files in `objectives/` remain dependencies of the configuration chain. A reward definition existing in a file does not imply it is active: later `configure()` calls can replace it or set it to `None`. Consult the resolved `params/env.yaml` and reward-manager active terms for a particular run.

The packaging migration preserves effective training/playback configurations after normalizing Python module names and repository paths. It adds a reusable runner for lifecycle features previously applied by the local batch controller. No files under the original `tests/` directory or their datasets are published.

Migration does not establish universal policy safety, conflict-free rewards on every reachable state, or improvement over the existing checkpoint. Source equality and CPU contracts have a narrower meaning than a new full training/behavior validation.

The November 11 source sync brings the current D2W objective chain and training reference from the working training project into this independent package. The independent D2W runner, replay/optimizer restoration and adaptive PPO learning-rate ceiling remain in place. Generated runtime assets and experiment scripts are not imported by this sync.
