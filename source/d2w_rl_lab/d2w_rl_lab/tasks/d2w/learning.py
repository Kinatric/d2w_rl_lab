"""D2W-only zero-command rehearsal for late-stage stair fine-tuning.

Deployment networks and inference are unchanged. Rehearsal is explicitly loaded
by a training controller; ordinary training without a replay uses upstream MoECTS.
The two update loops retain the vendored loss, clipping and optimizer-step order.
"""
from collections.abc import Generator
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.func import functional_call
from rsl_rl.moe_cts.algorithm import MoECTS


class D2WRetentionMoECTS(MoECTS):
    retention = None

    def _adapt_learning_rate(self, *args, **kwargs):
        super()._adapt_learning_rate(*args, **kwargs)
        self.learning_rate = min(1.e-4, self.learning_rate)
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate

    def load_retention(self, path, expected_sha256, *, coefficient=1.0, scale=0.01, batch_size=256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Parking replay provenance mismatch")
        data = torch.load(path, map_location=self.device, weights_only=True)
        n = data["single_obs"].shape[0]
        for name, width in [("single_obs", 57), ("policy", 570), ("action", 16)]:
            if data[name].shape != (n, width) or not torch.isfinite(data[name]).all():
                raise ValueError("Invalid parking replay " + name)
        if n < batch_size or data["single_obs"][:, 6:9].abs().max() != 0:
            raise ValueError("Parking replay requires zero-command observations")
        if data["policy"][:, 60:90].abs().max() != 0:
            raise ValueError("Parking replay history contains nonzero commands")
        if not coefficient > 0 or not scale > 0:
            raise ValueError("Parking rehearsal scales must be positive")
        self.retention = data
        self.retention_coefficient = float(coefficient)
        self.retention_scale = float(scale)
        self.retention_batch_size = int(batch_size)
        self.retention_rng = torch.Generator(device=self.device).manual_seed(20261106)
        self.retention_provenance = dict(path=str(path), sha256=expected_sha256,
                                         reference_sha256=data["reference_sha256"],
                                         coefficient=coefficient, scale=scale, batch_size=batch_size)

    def _retention_loss(self, *, student):
        data = self.retention
        idx = torch.randint(len(data["action"]), (self.retention_batch_size,),
                            device=self.device, generator=self.retention_rng)
        history = self.policy.actor_obs_normalizer(data["policy"][idx])
        single = self.policy.single_obs_normalizer(data["single_obs"][idx])
        if student:
            latent, _ = self.policy.student_moe_encoder(history)
            # Detach actor parameters, retaining the gradient to the student input.
            state = {k: v.detach() for k, v in self.policy.actor.named_parameters()}
            state.update({k: v.detach() for k, v in self.policy.actor.named_buffers()})
            prediction = functional_call(self.policy.actor, state, (torch.cat((latent, single), -1),))
        else:
            with torch.no_grad():
                latent, _ = self.policy.student_moe_encoder(history)
            prediction = self.policy.actor(torch.cat((latent, single), -1))
        return self.retention_coefficient * ((prediction - data["action"][idx]) / self.retention_scale).square().mean()

    def _update_ppo_batches(
        self,
        generator: Generator,
        teacher_samples: int,
        student_samples: int,
        original_mask: torch.Tensor,
    ) -> tuple[dict[str, float], int]:
        if self.retention is None:
            return super()._update_ppo_batches(generator, teacher_samples, student_samples, original_mask)
        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "parking_actor": 0.0}
        num_updates = 0
        for batch in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            teacher_results = self._batch_policy_results(
                obs_batch[:teacher_samples], actions_batch[:teacher_samples], True
            )
            student_stop = teacher_samples + student_samples
            student_results = self._batch_policy_results(
                obs_batch[teacher_samples:student_stop],
                actions_batch[teacher_samples:student_stop],
                False,
            )
            actions_log_prob_batch, value_batch, mu_batch, sigma_batch, entropy_batch = [
                torch.cat([teacher, student], dim=0)
                for teacher, student in zip(teacher_results, student_results)
            ]

            self._adapt_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch, original_mask)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            surrogate_loss = (
                surrogate_losses[:teacher_samples].mean() + surrogate_losses[teacher_samples:student_stop].mean()
            )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch[original_mask].mean()
            )
            retention_loss = self._retention_loss(student=False)
            loss = loss + retention_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            parameters = (parameter for group in self.optimizer.param_groups for parameter in group["params"])
            nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()

            totals["parking_actor"] += retention_loss.item()
            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy_batch.mean().item()
            num_updates += 1
        return totals, num_updates

    def _update_student_batches(
        self,
        generator: Generator,
        teacher_samples: int,
    ) -> tuple[dict[str, float], int]:
        if self.retention is None:
            return super()._update_student_batches(generator, teacher_samples)
        totals = {"mean_latent_loss": 0.0, "mean_load_balance_loss": 0.0, "parking_student": 0.0}
        student_updates = 0
        for batch in generator:
            obs_batch = batch[0]
            actor_obs = self.policy.actor_obs_normalizer(self.policy.get_actor_obs(obs_batch))
            student_latent, gating_weights = self.policy.student_moe_encoder(actor_obs[teacher_samples:])
            with torch.no_grad():
                critic_obs = self.policy.critic_obs_normalizer(self.policy.get_critic_obs(obs_batch))
                teacher_latent = self.policy.teacher_encoder(critic_obs[teacher_samples:])
            latent_loss = (teacher_latent - student_latent).pow(2).mean()
            mean_usage = gating_weights.mean(dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = (mean_usage - target_usage).pow(2).mean()
            student_loss = latent_loss + self.load_balance_coef * load_balance_loss

            retention_loss = self._retention_loss(student=True)
            student_loss = student_loss + retention_loss
            self.optimizer_stu_enc.zero_grad()
            student_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer_stu_enc.step()
            totals["parking_student"] += retention_loss.item()
            totals["mean_latent_loss"] += latent_loss.item()
            totals["mean_load_balance_loss"] += load_balance_loss.item()
            student_updates += 1
        return totals, student_updates
