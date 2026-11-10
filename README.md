# D2W_rl_lab

Reinforcement learning for the D2W wheel-legged robot on Isaac Lab 3.0. The project provides mixed-terrain training, concurrent teacher/student learning, native simulation playback and student policy export. Training covers stationary holding, walking/stopping transitions, velocity tracking and stair locomotion with hybrid leg-position and wheel-velocity control.

## Packages and task

| Path | Responsibility |
|---|---|
| `source/d2w_rl_lab` | D2W assets, motors, environment, observations/actions, rewards, commands and curricula |
| `source/d2w_rl_lab/d2w_rl_lab/tasks/mixed` | Shared terrain, MDP functions and base configurations |
| `source/rsl_rl` | Project PPO/MoE-CTS algorithms, networks, logging and export |
| `resources/d2w_description` | Source URDF, meshes, joint mappings and motor parameters |
| `scripts` | Training, playback, asset validation and environment utilities |

| Task | Purpose |
|---|---|
| `d2w-mixed-moe-cts` | Mixed-terrain concurrent teacher/student learning with eight student experts |

D2W uses 16 actions: 12 leg-position targets and four wheel-velocity targets. Policy observations have 57 single-frame dimensions and 570 dimensions of term-major ten-frame history; critic input has 295 dimensions. Physics runs at 200 Hz and policy control at 50 Hz. Legs use the RS06 motor model and wheels use RS02.

## Installation and assets

Use Python 3.12 and an already configured Isaac Lab 3.0 environment with Newton/PhysX plugins, PyTorch, Warp, MuJoCo and USD. Install in an independent environment so the package paths of active training jobs remain intact.

```bash
git clone http://192.168.0.100:3002/huangchao/D2W_rl_lab.git
cd D2W_rl_lab
git lfs pull
# Activate the configured Isaac Lab environment first.
./d2w_rl_lab.sh --install
./d2w_rl_lab.sh --list
```

Local RSL-RL and task packages install with `--no-deps`; the simulator stack is not automatically replaced. Optional video dependencies use the helper. Editable installation resolves assets relative to the repository root.

Source meshes use Git LFS. Generated USD/MJCF files, asset validation receipts, checkpoints and parking replay data are not distributed through Git. Place the complete validated runtime package in `resources/d2w_runtime/` before training:

```bash
# Replace this example with the actual validated runtime package location.
cp -a /absolute/path/to/d2w_runtime resources/
python scripts/tools/verify_d2w_assets.py
```

The runtime package includes mass/inertia, collision settings, joint mapping, standing calibration and a dependency inventory. A generic URDF conversion alone does not replace this preparation. Missing files or mismatched hashes are rejected.

## Training and continuation

```bash
# Fresh network: 1000 additional updates.
./d2w_rl_lab.sh --train --task d2w-mixed-moe-cts \
  --physics newton_mjwarp --viz none --num_envs 8192 \
  --max_iterations 1000 --seed 42 --logger tensorboard

# Continue a trusted checkpoint containing parking-retention state: 40000 updates.
./d2w_rl_lab.sh --train --task d2w-mixed-moe-cts \
  --physics newton_mjwarp --viz none --num_envs 8192 \
  --resume --checkpoint /absolute/path/model.pt \
  --max_iterations 40000 --seed 42 --logger tensorboard \
  agent.retention_path=/absolute/path/parking_replay.pt \
  agent.retention_sha256=YOUR_REPLAY_SHA256
```

`--max_iterations` specifies additional updates in this invocation. Continuation begins at checkpoint `iter + 1`, preserving the model, both teacher/student Adam states and rates, stair curriculum and dedicated parking replay RNG. Physics, general simulation RNG and observation history are reinitialized. Replay RNG restoration requires a compatible device type.

A checkpoint with parking retention requires the same replay SHA-256 and parameters; missing data is rejected before learning. Fresh training does not require replay. Current settings cap adaptive PPO LR at `1e-4`, with entropy coefficient `0.002` and parking retention coefficient `0.2`.

Hydra overrides are supported, for example `agent.algorithm.num_mini_batches=8`. Distributed training uses `--train-distributed` with explicit `CUDA_VISIBLE_DEVICES`.

## Checkpoints and training curves

By default, checkpoints are saved every 200 global update indices and at completion:

```text
logs/rsl_rl/d2w_moe_cts_lab30/<timestamp_run>/
├── model_<index>.pt
├── params/
├── budget.json
├── source_inventory.json
└── latest_checkpoint.json
```

`params/` contains resolved environment and learning settings. Other metadata records the budget, source hashes and latest checkpoint.

```bash
tensorboard --logdir logs/rsl_rl/d2w_moe_cts_lab30 \
  --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006` on the TensorBoard host to view rewards, losses, teacher/student metrics and sampling statistics.

## Sampling and locomotion objectives

Global command bounds are ±2 m/s forward, ±1 m/s lateral and ±2 rad/s yaw, subject to terrain-specific limits. Sampling includes strict zero commands, walking/stopping transitions, straight motion and mixed terrain.

Some flat zero-command intervals last 12–20 seconds and may be truncated by the 25-second episode. Among eligible nonzero, non-focused stair resamples, a 2% branch probes straight `[1,2)` m/s commands. This is not a fraction of total training time and does not contribute to focused curriculum promotion.

Objectives cover velocity tracking, standing/parking, terrain-conditioned alignment, support transfer and front/rear wheel placement. Consult resolved run configurations for active terms; a function's presence in a file does not mean it is enabled. See the [training code map](docs/training_map.md).

## Playback and export

```bash
./d2w_rl_lab.sh --play --task d2w-mixed-moe-cts \
  --checkpoint /absolute/path/model.pt --num_envs 1 \
  --physics newton_mjwarp --viz newton_gl
```

Playback supports student policy export and `--video` recording. Training checkpoints and exported student policies serve different purposes; exported policies require matching observation, action and motor settings.

This repository provides training and native playback. The independently maintained D2W MuJoCo sim2sim controller is not included. Behavior must be evaluated in specific standing, tracking and stair scenarios; source and loading checks do not replace behavioral validation.

## Licenses and local files

See [LICENSE](LICENSE) and `docs/licenses/`. Validation stays under ignored `tests/`. Models, replay datasets, logs, videos, caches, environments, credentials and generated runtime assets are excluded from commits.
