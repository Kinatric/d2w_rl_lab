"""D2W continuation state on the vendored MoE-CTS rollout loop.

Checkpoints contain tensors, optimizer states and curriculum metadata. Physics,
command history and the general simulation RNG are reset when a run is resumed.
Parking replay is an external, hash-verified input and is never embedded in Git.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from rsl_rl.moe_cts.runner import MoeCtsRunner


class D2WRunner(MoeCtsRunner):
    def _load_checkpoint_state(self, loaded_dict, load_optimizer):
        result = super()._load_checkpoint_state(loaded_dict, load_optimizer)
        self._resume_state = {
            name: loaded_dict.get(name) for name in (
                'd2w_practice_curriculum', 'd2w_parking_retention',
                'd2w_parking_retention_rng',
            )
        }
        self._resume_index = int(loaded_dict['iter'])
        if load_optimizer:
            rates = [float(g['lr']) for g in self.alg.optimizer.param_groups]
            if not rates or any(v != rates[0] or not math.isfinite(v) or v <= 0 for v in rates):
                raise ValueError('D2W resume requires equal finite positive PPO group learning rates')
            self.alg.learning_rate = rates[0]
            if any(not math.isfinite(float(g['lr'])) or g['lr'] <= 0
                   for g in self.alg.optimizer_stu_enc.param_groups):
                raise ValueError('Invalid student optimizer learning rate')
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations < 1:
            raise ValueError('Training budget must be positive')
        start = getattr(self, '_resume_index', -1) + 1
        self.current_learning_iteration = start
        self._budget = dict(first_index=start, last_index=start + num_learning_iterations - 1,
                            additional_updates=num_learning_iterations)
        self._completed_updates = 0
        root = Path(__file__).resolve().parents[5]
        self._source_inventory = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for base in ('source', 'scripts') for p in (root / base).rglob('*.py')
        }
        if self.logger.log_dir:
            directory = Path(self.logger.log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'budget.json').write_text(json.dumps(self._budget, indent=2))
            (directory / 'source_inventory.json').write_text(json.dumps(self._source_inventory, indent=2))
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def _before_learn(self):
        super()._before_learn()
        base = self.env.unwrapped
        base.common_step_counter = self.current_learning_iteration * self.cfg['num_steps_per_env']
        sampler = base.command_manager.get_term('base_velocity')
        sampler.student_mask.zero_()
        sampler.student_mask[self.alg.student_env_idxs] = True
        saved = getattr(self, '_resume_state', {})
        curriculum = saved.get('d2w_practice_curriculum')
        if curriculum is not None:
            sampler.restore_curriculum(curriculum)
        else:
            sampler.stage_step = base.common_step_counter

        path, digest = self.cfg.get('retention_path'), self.cfg.get('retention_sha256')
        previous = saved.get('d2w_parking_retention')
        if bool(path) != bool(digest):
            raise ValueError('Specify both retention_path and retention_sha256')
        if previous and not path:
            raise ValueError('This checkpoint requires its parking replay; supply retention_path and retention_sha256')
        if path:
            if previous and digest != previous['sha256']:
                raise ValueError('Resume parking replay differs from the saved source')
            self.alg.load_retention(path, digest, coefficient=self.cfg['retention_coefficient'])
            if previous:
                for key in ('coefficient', 'scale', 'batch_size'):
                    if previous[key] != self.alg.retention_provenance[key]:
                        raise ValueError(f'Parking rehearsal {key} changed on resume')
                rng = saved.get('d2w_parking_retention_rng')
                if rng is None:
                    raise ValueError('Missing parking replay RNG state')
                try:
                    self.alg.retention_rng.set_state(rng.cpu())
                except RuntimeError as exc:
                    raise ValueError('Parking RNG is incompatible with this device; resume on the original device type') from exc

        update = self.alg.update
        def checked_update():
            losses = update()
            self._completed_updates += 1
            if not all(math.isfinite(float(value)) for value in losses.values()):
                raise FloatingPointError('Nonfinite D2W training loss')
            if self._completed_updates > self._budget['additional_updates']:
                raise RuntimeError('Training budget exceeded')
            return losses
        self.alg.update = checked_update

    def _checkpoint_state(self, infos):
        state = super()._checkpoint_state(infos)
        state['d2w_practice_curriculum'] = self.env.unwrapped.command_manager.get_term('base_velocity').curriculum_state()
        if self.alg.retention is not None:
            state['d2w_parking_retention'] = self.alg.retention_provenance
            state['d2w_parking_retention_rng'] = self.alg.retention_rng.get_state()
        state['d2w_budget'] = self._budget
        state['d2w_relearn_updates'] = self._completed_updates
        state['d2w_source_inventory'] = self._source_inventory
        return state

    def save(self, path, infos=None, *, iteration=None, final=False):
        index = self.current_learning_iteration
        if not self._budget['first_index'] <= index <= self._budget['last_index']:
            raise RuntimeError('Checkpoint iteration outside training budget')
        for value in self.alg.policy.state_dict().values():
            if not torch.isfinite(value).all():
                raise FloatingPointError('Refusing a nonfinite policy checkpoint')
        for optimizer in (self.alg.optimizer, self.alg.optimizer_stu_enc):
            for values in optimizer.state.values():
                if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                       for value in values.values()):
                    raise FloatingPointError('Refusing a nonfinite optimizer checkpoint')
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        torch.save(self._checkpoint_state(infos), temporary)
        os.replace(temporary, path)
        receipt = dict(index=index, path=str(path), final=final,
                       updates_this_process=self._completed_updates)
        destination = path.parent / 'latest_checkpoint.json'
        tmp_receipt = destination.with_suffix('.tmp')
        tmp_receipt.write_text(json.dumps(receipt, indent=2))
        os.replace(tmp_receipt, destination)
        self._after_save(index, final)
