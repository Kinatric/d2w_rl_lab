# D2W repository guide

This is the D2W 16-action training project. Preserve robot joint order, observation layout, motor protection and the teacher/student split. The only registered task is `d2w-mixed-moe-cts`. Read README.md and docs/training_map.md before changing training.

Keep `tests/`, checkpoints, replay datasets, logs, videos, generated runtime assets, caches and secrets local-only. Do not stage or upload files under `tests/`. Keep `/tests/` ignored without exceptions. Preserve source asset meshes and Git LFS. Preserve upstream licenses and copyright notices.

Use independent workspaces/environments for migration; never modify the source inventory or editable installation of a running training process. Do not install runtime dependencies blindly. Publication does not authorize starting extra learning or hardware deployment.

Validate changes at their actual boundary. For source renames, compare the fully composed old/new configurations and import/string entry points; check trusted checkpoint and both optimizers. Keep GPU tests sequential and do not compete with a live training job. CPU tests and config parity are not physical behavior or convergence certificates.

Run `git diff --check`, `bash -n d2w_rl_lab.sh`, task discovery and focused local-only tests before publishing. Verify `git ls-files tests` is empty and no models, generated resources or credentials are staged. Document model/runtime dependencies explicitly instead of silently downgrading missing requirements.
