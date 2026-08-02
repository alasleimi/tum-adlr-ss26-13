from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from typing import Any

import numpy as np
import torch

from cleanrl_utils.buffers import ReplayBuffer as CleanRLReplayBuffer
from cleanrl_utils.buffers import ReplayBufferSamples

from last_nine_rl.envs import UprightDetector


class SACNReplaySamples(NamedTuple):
    observations: Any
    actions: Any
    trajectory_observations: Any
    trajectory_actions: Any
    trajectory_next_observations: Any
    trajectory_rewards: Any
    trajectory_dones: Any
    trajectory_action_log_probs: Any
    reference_actions: Any | None = None
    reference_critic_actions: Any | None = None
    replay_indices: Any | None = None
    sampling_probabilities: Any | None = None
    importance_weights: Any | None = None


class InstrumentedReplaySamples(NamedTuple):
    """Ordinary replay samples plus teacher labels captured at collection time."""

    observations: Any
    actions: Any
    next_observations: Any
    dones: Any
    rewards: Any
    reference_actions: Any | None = None
    reference_critic_actions: Any | None = None
    replay_indices: Any | None = None
    sampling_probabilities: Any | None = None
    importance_weights: Any | None = None


def _optional_action_batch(batch: Any, name: str) -> torch.Tensor:
    value = getattr(batch, name, None)
    if value is None:
        return torch.full_like(batch.actions, float("nan"))
    return value


def _optional_scalar_batch(batch: Any, name: str, fill_value: float) -> torch.Tensor:
    value = getattr(batch, name, None)
    if value is None:
        return torch.full(
            (batch.actions.shape[0], 1),
            fill_value,
            dtype=batch.actions.dtype,
            device=batch.actions.device,
        )
    return value


def concatenate_replay_samples(*batches: ReplayBufferSamples) -> InstrumentedReplaySamples:
    if len(batches) == 0:
        raise ValueError("At least one replay batch is required.")
    if len(batches) == 1:
        return batches[0]
    return InstrumentedReplaySamples(
        observations=torch.cat([batch.observations for batch in batches], dim=0),
        actions=torch.cat([batch.actions for batch in batches], dim=0),
        next_observations=torch.cat([batch.next_observations for batch in batches], dim=0),
        dones=torch.cat([batch.dones for batch in batches], dim=0),
        rewards=torch.cat([batch.rewards for batch in batches], dim=0),
        reference_actions=torch.cat(
            [_optional_action_batch(batch, "reference_actions") for batch in batches], dim=0
        ),
        reference_critic_actions=torch.cat(
            [_optional_action_batch(batch, "reference_critic_actions") for batch in batches], dim=0
        ),
        replay_indices=torch.cat(
            [_optional_scalar_batch(batch, "replay_indices", -1.0) for batch in batches], dim=0
        ),
        sampling_probabilities=torch.cat(
            [_optional_scalar_batch(batch, "sampling_probabilities", float("nan")) for batch in batches], dim=0
        ),
        importance_weights=torch.cat(
            [_optional_scalar_batch(batch, "importance_weights", 1.0) for batch in batches], dim=0
        ),
    )


def concatenate_sacn_replay_samples(*batches: SACNReplaySamples) -> SACNReplaySamples:
    if len(batches) == 0:
        raise ValueError("At least one SACn replay batch is required.")
    if len(batches) == 1:
        return batches[0]
    return SACNReplaySamples(
        observations=torch.cat([batch.observations for batch in batches], dim=0),
        actions=torch.cat([batch.actions for batch in batches], dim=0),
        trajectory_observations=torch.cat([batch.trajectory_observations for batch in batches], dim=0),
        trajectory_actions=torch.cat([batch.trajectory_actions for batch in batches], dim=0),
        trajectory_next_observations=torch.cat([batch.trajectory_next_observations for batch in batches], dim=0),
        trajectory_rewards=torch.cat([batch.trajectory_rewards for batch in batches], dim=0),
        trajectory_dones=torch.cat([batch.trajectory_dones for batch in batches], dim=0),
        trajectory_action_log_probs=torch.cat([batch.trajectory_action_log_probs for batch in batches], dim=0),
        reference_actions=torch.cat(
            [_optional_action_batch(batch, "reference_actions") for batch in batches], dim=0
        ),
        reference_critic_actions=torch.cat(
            [_optional_action_batch(batch, "reference_critic_actions") for batch in batches], dim=0
        ),
        replay_indices=torch.cat(
            [_optional_scalar_batch(batch, "replay_indices", -1.0) for batch in batches], dim=0
        ),
        sampling_probabilities=torch.cat(
            [_optional_scalar_batch(batch, "sampling_probabilities", float("nan")) for batch in batches], dim=0
        ),
        importance_weights=torch.cat(
            [_optional_scalar_batch(batch, "importance_weights", 1.0) for batch in batches], dim=0
        ),
    )


class InstrumentedReplayBuffer(CleanRLReplayBuffer):
    """CleanRL replay buffer with Week 1 inspection telemetry."""

    steps: np.ndarray
    episode_ids: np.ndarray
    sample_counts: np.ndarray
    action_log_probs: np.ndarray
    reference_actions: np.ndarray
    reference_critic_actions: np.ndarray
    priorities: np.ndarray

    def __init__(
        self,
        *args: Any,
        swd_linear_decay_steps: int = 0,
        swd_min_weight: float = 0.1,
        priority_mode: str = "none",
        priority_alpha: float = 0.6,
        priority_beta_initial: float = 0.4,
        priority_beta_final: float = 1.0,
        priority_beta_anneal_steps: int = 100_000,
        priority_uniform_fraction: float = 0.5,
        priority_epsilon: float = 1e-3,
        priority_clip: float = 10.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.sample_counts = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.action_log_probs = np.full((self.buffer_size, self.n_envs), np.nan, dtype=np.float32)
        self.reference_actions = np.full_like(self.actions, np.nan, dtype=np.float32)
        self.reference_critic_actions = np.full_like(self.actions, np.nan, dtype=np.float32)
        self.priorities = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self.swd_linear_decay_steps = int(swd_linear_decay_steps)
        self.swd_min_weight = float(swd_min_weight)
        self.priority_mode = str(priority_mode)
        self.priority_alpha = float(priority_alpha)
        self.priority_beta_initial = float(priority_beta_initial)
        self.priority_beta_final = float(priority_beta_final)
        self.priority_beta_anneal_steps = int(priority_beta_anneal_steps)
        self.priority_uniform_fraction = float(priority_uniform_fraction)
        self.priority_epsilon = float(priority_epsilon)
        self.priority_clip = float(priority_clip)
        self._last_priority_sample_metrics: dict[str, float] = {}
        self._current_step = 0
        self._pendulum_hard_cache_params: tuple[float, float, float] | None = None
        self._pendulum_hard_index_positions = np.full(self.buffer_size, -1, dtype=np.int64)
        self._pendulum_hard_indices = np.empty(self.buffer_size, dtype=np.int64)
        self._pendulum_hard_count = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        step: int | None = None,
        episode_id: int | None = None,
        action_log_prob: np.ndarray | float | None = None,
        reference_action: np.ndarray | None = None,
        reference_critic_action: np.ndarray | None = None,
    ) -> None:
        write_pos = self.pos
        current_size = self.size()
        new_priority = (
            float(np.max(self.priorities[:current_size, 0]))
            if self.priority_mode != "none" and current_size > 0
            else 1.0
        )
        super().add(obs, next_obs, action, reward, done, infos)
        self.sample_counts[write_pos, :] = 0
        self.priorities[write_pos, :] = new_priority
        if action_log_prob is None:
            self.action_log_probs[write_pos, :] = np.nan
        else:
            self.action_log_probs[write_pos, :] = np.asarray(action_log_prob, dtype=np.float32).reshape(self.n_envs)
        if reference_action is None:
            self.reference_actions[write_pos, :] = np.nan
        else:
            self.reference_actions[write_pos, :] = np.asarray(reference_action, dtype=np.float32).reshape(
                self.n_envs, -1
            )
        if reference_critic_action is None:
            self.reference_critic_actions[write_pos, :] = np.nan
        else:
            self.reference_critic_actions[write_pos, :] = np.asarray(
                reference_critic_action, dtype=np.float32
            ).reshape(self.n_envs, -1)
        if step is not None:
            self.steps[write_pos, :] = int(step)
            self._current_step = max(self._current_step, int(step))
        else:
            self._current_step += 1
        if episode_id is not None:
            self.episode_ids[write_pos, :] = int(episode_id)
        if self._pendulum_hard_cache_params is not None and self.n_envs == 1:
            low, high, velocity_limit = self._pendulum_hard_cache_params
            hard = bool(
                pendulum_hard_state_mask(
                    np.asarray(obs).reshape(1, -1),
                    abs_theta_low=low,
                    abs_theta_high=high,
                    velocity_limit=velocity_limit,
                )[0]
            )
            self._set_pendulum_hard_index(int(write_pos), hard)

    def sample(self, batch_size: int, count: bool = True):
        if self.n_envs != 1 or self.optimize_memory_usage:
            return super().sample(batch_size)
        upper_bound = self.buffer_size if self.full else self.pos
        sampling_probabilities = None
        importance_weights = None
        if self.priority_mode != "none":
            pool = np.arange(upper_bound, dtype=np.int64)
            batch_inds, sampling_probabilities, importance_weights = self._sample_prioritized_from_pool(
                pool, batch_size
            )
        elif self.swd_linear_decay_steps == 0:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        else:
            batch_inds = self._sample_swd_indices(upper_bound, batch_size)
        if count:
            np.add.at(self.sample_counts[:, 0], batch_inds, 1)
        return self._get_samples(
            batch_inds,
            sampling_probabilities=sampling_probabilities,
            importance_weights=importance_weights,
        )

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        sampling_probabilities: np.ndarray | None = None,
        importance_weights: np.ndarray | None = None,
    ) -> InstrumentedReplaySamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]
        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
            self.reference_actions[batch_inds, env_indices, :],
            self.reference_critic_actions[batch_inds, env_indices, :],
            np.asarray(batch_inds, dtype=np.float32).reshape(-1, 1),
            (
                np.asarray(sampling_probabilities, dtype=np.float32).reshape(-1, 1)
                if sampling_probabilities is not None
                else np.full((len(batch_inds), 1), np.nan, dtype=np.float32)
            ),
            (
                np.asarray(importance_weights, dtype=np.float32).reshape(-1, 1)
                if importance_weights is not None
                else np.ones((len(batch_inds), 1), dtype=np.float32)
            ),
        )
        return InstrumentedReplaySamples(*tuple(map(self.to_torch, data)))

    def sample_sacn(
        self,
        batch_size: int,
        n_step: int,
        count: bool = True,
        max_age_steps: int = 0,
        require_action_log_probs: bool = True,
    ) -> SACNReplaySamples:
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise NotImplementedError("SACn sequence sampling currently supports n_envs=1 without memory optimization.")
        n_step = int(n_step)
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        sampling_probabilities = None
        importance_weights = None
        if self.priority_mode != "none":
            upper_bound = self.buffer_size if self.full else self.pos
            valid = self._valid_sacn_indices(
                n_step=n_step,
                upper_bound=upper_bound,
                max_age_steps=max_age_steps,
                require_action_log_probs=require_action_log_probs,
            )
            batch_inds, sampling_probabilities, importance_weights = self._sample_prioritized_from_pool(
                valid, batch_size
            )
        else:
            batch_inds = self._sample_valid_indices(
                batch_size,
                n_step=n_step,
                max_age_steps=max_age_steps,
                require_action_log_probs=require_action_log_probs,
            )
        if count:
            np.add.at(self.sample_counts[:, 0], batch_inds, 1)
        return self._get_sacn_samples(
            batch_inds,
            n_step=n_step,
            sampling_probabilities=sampling_probabilities,
            importance_weights=importance_weights,
        )

    def sample_sacn_pendulum_hard_states(
        self,
        batch_size: int,
        n_step: int,
        fraction: float,
        abs_theta_low: float,
        abs_theta_high: float,
        velocity_limit: float,
        count: bool = True,
        max_age_steps: int = 0,
        require_action_log_probs: bool = True,
    ) -> SACNReplaySamples:
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise NotImplementedError("SACn hard-state sampling currently supports n_envs=1 without memory optimization.")
        n_step = int(n_step)
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        hard_count = int(round(batch_size * max(0.0, min(1.0, fraction))))
        if hard_count <= 0:
            return self.sample_sacn(
                batch_size,
                n_step=n_step,
                count=count,
                max_age_steps=max_age_steps,
                require_action_log_probs=require_action_log_probs,
            )

        upper_bound = self.buffer_size if self.full else self.pos
        valid = self._valid_sacn_indices(
            n_step=n_step,
            upper_bound=upper_bound,
            max_age_steps=max_age_steps,
            require_action_log_probs=require_action_log_probs,
        )
        hard_mask = pendulum_hard_state_mask(
            self.observations[valid, 0],
            abs_theta_low=abs_theta_low,
            abs_theta_high=abs_theta_high,
            velocity_limit=velocity_limit,
        )
        hard_valid = valid[hard_mask]
        if hard_valid.size == 0:
            return self.sample_sacn(
                batch_size,
                n_step=n_step,
                count=count,
                max_age_steps=max_age_steps,
                require_action_log_probs=require_action_log_probs,
            )

        hard_inds = self._sample_from_sacn_pool(hard_valid, min(hard_count, batch_size), upper_bound)
        uniform_count = batch_size - len(hard_inds)
        if uniform_count > 0:
            uniform_inds = self._sample_from_sacn_pool(valid, uniform_count, upper_bound)
            batch_inds = np.concatenate([hard_inds, uniform_inds])
            np.random.shuffle(batch_inds)
        else:
            batch_inds = hard_inds

        if count:
            np.add.at(self.sample_counts[:, 0], batch_inds, 1)
        return self._get_sacn_samples(batch_inds, n_step=n_step)

    def _sample_valid_indices(
        self,
        batch_size: int,
        n_step: int,
        max_age_steps: int = 0,
        require_action_log_probs: bool = True,
    ) -> np.ndarray:
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound < n_step:
            raise ValueError(f"Replay buffer has {upper_bound} transitions, fewer than n_step={n_step}.")

        if self.swd_linear_decay_steps != 0:
            all_candidates = np.arange(upper_bound, dtype=np.int64)
            valid = all_candidates[
                self._valid_sacn_start_mask(
                    all_candidates,
                    n_step=n_step,
                    upper_bound=upper_bound,
                    max_age_steps=max_age_steps,
                    require_action_log_probs=require_action_log_probs,
                )
            ]
            if valid.size == 0:
                raise ValueError(f"No valid contiguous replay trajectories of length n_step={n_step} are available.")
            probabilities = self._swd_probabilities_for_indices(valid, upper_bound)
            return np.random.choice(valid, size=batch_size, replace=True, p=probabilities).astype(np.int64, copy=False)

        selected: list[np.ndarray] = []
        selected_count = 0
        attempts = 0
        while selected_count < batch_size and attempts < 20:
            draw_count = max(64, (batch_size - selected_count) * 4)
            candidates = np.random.randint(0, upper_bound, size=draw_count)
            valid = candidates[
                self._valid_sacn_start_mask(
                    candidates,
                    n_step=n_step,
                    upper_bound=upper_bound,
                    max_age_steps=max_age_steps,
                    require_action_log_probs=require_action_log_probs,
                )
            ]
            if valid.size:
                take = valid[: batch_size - selected_count]
                selected.append(take)
                selected_count += int(take.size)
            attempts += 1

        if selected_count < batch_size:
            all_candidates = np.arange(upper_bound, dtype=np.int64)
            valid = all_candidates[
                self._valid_sacn_start_mask(
                    all_candidates,
                    n_step=n_step,
                    upper_bound=upper_bound,
                    max_age_steps=max_age_steps,
                    require_action_log_probs=require_action_log_probs,
                )
            ]
            if valid.size == 0:
                raise ValueError(f"No valid contiguous replay trajectories of length n_step={n_step} are available.")
            selected.append(np.random.choice(valid, size=batch_size - selected_count, replace=True))

        return np.concatenate(selected).astype(np.int64, copy=False)

    def _valid_sacn_indices(
        self,
        n_step: int,
        upper_bound: int,
        max_age_steps: int = 0,
        require_action_log_probs: bool = True,
    ) -> np.ndarray:
        if upper_bound < n_step:
            raise ValueError(f"Replay buffer has {upper_bound} transitions, fewer than n_step={n_step}.")
        all_candidates = np.arange(upper_bound, dtype=np.int64)
        valid = all_candidates[
            self._valid_sacn_start_mask(
                all_candidates,
                n_step=n_step,
                upper_bound=upper_bound,
                max_age_steps=max_age_steps,
                require_action_log_probs=require_action_log_probs,
            )
        ]
        if valid.size == 0:
            raise ValueError(f"No valid contiguous replay trajectories of length n_step={n_step} are available.")
        return valid

    def _sample_from_sacn_pool(self, valid: np.ndarray, batch_size: int, upper_bound: int) -> np.ndarray:
        if valid.size == 0:
            raise ValueError("Cannot sample SACn trajectories from an empty valid index pool.")
        probabilities = (
            self._swd_probabilities_for_indices(valid, upper_bound)
            if self.swd_linear_decay_steps != 0
            else None
        )
        return np.random.choice(valid, size=batch_size, replace=True, p=probabilities).astype(np.int64, copy=False)

    def _valid_sacn_start_mask(
        self,
        indices: np.ndarray,
        n_step: int,
        upper_bound: int,
        max_age_steps: int = 0,
        require_action_log_probs: bool = True,
    ) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        offsets = np.arange(n_step, dtype=np.int64).reshape(1, -1)
        positions = (indices.reshape(-1, 1) + offsets) % int(upper_bound)
        episode_ids = self.episode_ids[:upper_bound, 0][positions]
        steps = self.steps[:upper_bound, 0][positions]
        same_episode = np.all(episode_ids == episode_ids[:, :1], axis=1)
        sequential_steps = np.all(steps == steps[:, :1] + offsets, axis=1)
        valid = same_episode & sequential_steps
        if require_action_log_probs:
            log_probs = self.action_log_probs[:upper_bound, 0][positions]
            finite_log_probs = np.all(np.isfinite(log_probs), axis=1)
            valid = valid & finite_log_probs
        max_age_steps = int(max_age_steps)
        if max_age_steps > 0:
            current_step = self._current_step if self._current_step > 0 else int(np.max(self.steps[:upper_bound, 0]))
            ages = np.maximum(current_step - steps[:, 0], 0)
            valid = valid & (ages <= max_age_steps)
        return valid

    def _get_sacn_samples(
        self,
        batch_inds: np.ndarray,
        n_step: int,
        sampling_probabilities: np.ndarray | None = None,
        importance_weights: np.ndarray | None = None,
    ) -> SACNReplaySamples:
        upper_bound = self.buffer_size if self.full else self.pos
        offsets = np.arange(n_step, dtype=np.int64).reshape(1, -1)
        positions = (np.asarray(batch_inds, dtype=np.int64).reshape(-1, 1) + offsets) % int(upper_bound)

        observations = self.observations[batch_inds, 0, :]
        actions = self.actions[batch_inds, 0, :]
        trajectory_observations = self.observations[positions, 0, :]
        trajectory_actions = self.actions[positions, 0, :]
        trajectory_next_observations = self.next_observations[positions, 0, :]
        trajectory_rewards = self.rewards[positions, 0]
        trajectory_dones = self.dones[positions, 0] * (1 - self.timeouts[positions, 0])
        trajectory_action_log_probs = self.action_log_probs[positions, 0]
        reference_actions = self.reference_actions[batch_inds, 0, :]
        reference_critic_actions = self.reference_critic_actions[batch_inds, 0, :]

        data = (
            observations,
            actions,
            trajectory_observations,
            trajectory_actions,
            trajectory_next_observations,
            trajectory_rewards[..., None],
            trajectory_dones[..., None],
            trajectory_action_log_probs[..., None],
            reference_actions,
            reference_critic_actions,
            np.asarray(batch_inds, dtype=np.float32).reshape(-1, 1),
            (
                np.asarray(sampling_probabilities, dtype=np.float32).reshape(-1, 1)
                if sampling_probabilities is not None
                else np.full((len(batch_inds), 1), np.nan, dtype=np.float32)
            ),
            (
                np.asarray(importance_weights, dtype=np.float32).reshape(-1, 1)
                if importance_weights is not None
                else np.ones((len(batch_inds), 1), dtype=np.float32)
            ),
        )
        return SACNReplaySamples(*tuple(map(self.to_torch, data)))

    def _priority_beta(self) -> float:
        if self.priority_beta_anneal_steps <= 0:
            return self.priority_beta_final
        fraction = min(max(float(self._current_step) / float(self.priority_beta_anneal_steps), 0.0), 1.0)
        return self.priority_beta_initial + fraction * (
            self.priority_beta_final - self.priority_beta_initial
        )

    def _sample_prioritized_from_pool(
        self,
        pool: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample from a uniform/PER mixture and return exact draw probabilities.

        The explicit uniform component keeps every eligible transition reachable
        and bounds the inverse-probability correction. No observation coordinate
        or task-specific range is consulted.
        """

        pool = np.asarray(pool, dtype=np.int64).reshape(-1)
        if pool.size == 0:
            raise ValueError("Cannot sample priorities from an empty replay pool.")
        raw_priorities = np.clip(
            self.priorities[pool, 0].astype(np.float64, copy=False),
            self.priority_epsilon,
            self.priority_clip,
        )
        scaled_priorities = np.power(raw_priorities, self.priority_alpha)
        scaled_sum = float(np.sum(scaled_priorities))
        if not np.isfinite(scaled_sum) or scaled_sum <= 0.0:
            priority_probabilities = np.full(pool.size, 1.0 / float(pool.size), dtype=np.float64)
        else:
            priority_probabilities = scaled_priorities / scaled_sum
        uniform_probability = 1.0 / float(pool.size)
        mixture_probabilities = (
            self.priority_uniform_fraction * uniform_probability
            + (1.0 - self.priority_uniform_fraction) * priority_probabilities
        )
        selected_positions = np.random.choice(
            pool.size,
            size=batch_size,
            replace=True,
            p=mixture_probabilities,
        )
        batch_inds = pool[selected_positions]
        draw_probabilities = mixture_probabilities[selected_positions]
        beta = self._priority_beta()
        importance_weights = np.power(pool.size * draw_probabilities, -beta)
        max_importance_weight = np.power(
            pool.size * float(np.min(mixture_probabilities)),
            -beta,
        )
        importance_weights = importance_weights / max(max_importance_weight, 1e-12)
        self._last_priority_sample_metrics = {
            "priority_beta": float(beta),
            "priority_pool_size": float(pool.size),
            "priority_sampling_probability_mean": float(np.mean(draw_probabilities)),
            "priority_sampling_probability_min": float(np.min(draw_probabilities)),
            "priority_sampling_probability_max": float(np.max(draw_probabilities)),
            "priority_importance_weight_mean": float(np.mean(importance_weights)),
            "priority_importance_weight_min": float(np.min(importance_weights)),
            "priority_importance_weight_max": float(np.max(importance_weights)),
        }
        return (
            batch_inds.astype(np.int64, copy=False),
            draw_probabilities.astype(np.float32, copy=False),
            importance_weights.astype(np.float32, copy=False),
        )

    def update_priorities(self, indices: Any, values: Any) -> None:
        """Update raw priorities, merging duplicate sampled indices by maximum."""

        if self.priority_mode == "none":
            return
        resolved_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        resolved_values = np.asarray(values, dtype=np.float64).reshape(-1)
        if resolved_indices.shape != resolved_values.shape:
            raise ValueError("Priority indices and values must have the same shape.")
        upper_bound = self.buffer_size if self.full else self.pos
        if np.any((resolved_indices < 0) | (resolved_indices >= upper_bound)):
            raise IndexError("Priority update contains an index outside the populated replay buffer.")
        resolved_values = np.nan_to_num(
            resolved_values,
            nan=0.0,
            posinf=self.priority_clip,
            neginf=0.0,
        )
        resolved_values = np.clip(
            np.abs(resolved_values) + self.priority_epsilon,
            self.priority_epsilon,
            self.priority_clip,
        ).astype(np.float32, copy=False)
        # Duplicate indices can appear with replacement; keep their strongest
        # fresh signal, while replacing (rather than monotonically increasing)
        # an older priority so transitions can become less prominent again.
        unique_indices = np.unique(resolved_indices)
        for index in unique_indices:
            duplicate_values = resolved_values[resolved_indices == index]
            self.priorities[index, 0] = float(np.max(duplicate_values))

    def _sample_swd_indices(self, upper_bound: int, batch_size: int) -> np.ndarray:
        indices = np.arange(upper_bound, dtype=np.int64)
        probabilities = self._swd_probabilities_for_indices(indices, upper_bound)
        if probabilities is None:
            return np.random.randint(0, upper_bound, size=batch_size)
        return np.random.choice(indices, size=batch_size, replace=True, p=probabilities)

    def _swd_probabilities_for_indices(self, indices: np.ndarray, upper_bound: int) -> np.ndarray | None:
        decay_steps = abs(self.swd_linear_decay_steps)
        if decay_steps <= 0:
            return None
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return None
        steps = self.steps[:upper_bound, 0]
        current_step = self._current_step if self._current_step > 0 else int(np.max(steps))
        ages = np.maximum(current_step - steps[indices], 0)
        if self.swd_linear_decay_steps > 0:
            weights = np.maximum(self.swd_min_weight, 1.0 - ages / decay_steps)
        else:
            weights = np.minimum(1.0, self.swd_min_weight + ages / decay_steps)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return None
        return weights / weight_sum

    def sample_pendulum_hard_states(
        self,
        batch_size: int,
        fraction: float,
        abs_theta_low: float,
        abs_theta_high: float,
        velocity_limit: float,
        count: bool = True,
    ):
        if self.n_envs != 1 or self.optimize_memory_usage:
            return super().sample(batch_size)

        upper_bound = self.buffer_size if self.full else self.pos
        hard_count = int(round(batch_size * max(0.0, min(1.0, fraction))))
        if upper_bound <= 0 or hard_count <= 0:
            return self.sample(batch_size, count=count)

        self._ensure_pendulum_hard_cache(
            abs_theta_low=abs_theta_low,
            abs_theta_high=abs_theta_high,
            velocity_limit=velocity_limit,
        )
        if self._pendulum_hard_count == 0:
            return self.sample(batch_size, count=count)

        hard_pool = self._pendulum_hard_indices[: self._pendulum_hard_count]
        hard_inds = np.random.choice(hard_pool, size=min(hard_count, batch_size), replace=True)
        uniform_count = batch_size - len(hard_inds)
        if uniform_count > 0:
            uniform_inds = np.random.randint(0, upper_bound, size=uniform_count)
            batch_inds = np.concatenate([hard_inds, uniform_inds])
            np.random.shuffle(batch_inds)
        else:
            batch_inds = hard_inds

        if count:
            np.add.at(self.sample_counts[:, 0], batch_inds, 1)
        return self._get_samples(batch_inds)

    def _ensure_pendulum_hard_cache(
        self,
        abs_theta_low: float,
        abs_theta_high: float,
        velocity_limit: float,
    ) -> None:
        params = (float(abs_theta_low), float(abs_theta_high), float(velocity_limit))
        if self._pendulum_hard_cache_params == params:
            return

        self._pendulum_hard_cache_params = params
        self._pendulum_hard_index_positions.fill(-1)
        self._pendulum_hard_count = 0
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            return

        hard_mask = pendulum_hard_state_mask(
            self.observations[:upper_bound, 0],
            abs_theta_low=abs_theta_low,
            abs_theta_high=abs_theta_high,
            velocity_limit=velocity_limit,
        )
        hard_indices = np.flatnonzero(hard_mask).astype(np.int64, copy=False)
        count = len(hard_indices)
        if count == 0:
            return
        self._pendulum_hard_indices[:count] = hard_indices
        self._pendulum_hard_index_positions[hard_indices] = np.arange(count, dtype=np.int64)
        self._pendulum_hard_count = count

    def _set_pendulum_hard_index(self, index: int, hard: bool) -> None:
        current_pos = int(self._pendulum_hard_index_positions[index])
        if hard:
            if current_pos >= 0:
                return
            self._pendulum_hard_indices[self._pendulum_hard_count] = index
            self._pendulum_hard_index_positions[index] = self._pendulum_hard_count
            self._pendulum_hard_count += 1
            return

        if current_pos < 0:
            return
        last_pos = self._pendulum_hard_count - 1
        last_index = int(self._pendulum_hard_indices[last_pos])
        self._pendulum_hard_indices[current_pos] = last_index
        self._pendulum_hard_index_positions[last_index] = current_pos
        self._pendulum_hard_index_positions[index] = -1
        self._pendulum_hard_count -= 1

    def summary(
        self,
        detector: UprightDetector,
        current_step: int,
        action_high: np.ndarray | None = None,
    ) -> dict[str, float]:
        size = self.size()
        if size == 0:
            return {"size": 0.0}

        metrics = summarize_transitions(
            observations=self.observations[:size, 0],
            next_observations=self.next_observations[:size, 0],
            actions=self.actions[:size, 0],
            rewards=self.rewards[:size, 0],
            dones=self.dones[:size, 0],
            steps=self.steps[:size, 0],
            sample_counts=self.sample_counts[:size, 0],
            detector=detector,
            current_step=current_step,
            capacity=self.buffer_size,
            action_high=action_high,
        )
        if self.priority_mode != "none":
            populated_priorities = self.priorities[:size, 0]
            metrics.update(
                {
                    "priority_enabled": 1.0,
                    "priority_mean": float(np.mean(populated_priorities)),
                    "priority_std": float(np.std(populated_priorities)),
                    "priority_min": float(np.min(populated_priorities)),
                    "priority_max": float(np.max(populated_priorities)),
                    "priority_uniform_fraction": float(self.priority_uniform_fraction),
                    **self._last_priority_sample_metrics,
                }
            )
        else:
            metrics["priority_enabled"] = 0.0
        return metrics

    def save_npz(self, path: str | Path) -> None:
        size = self.size()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations[:size, 0],
            actions=self.actions[:size, 0],
            rewards=self.rewards[:size, 0, None],
            next_observations=self.next_observations[:size, 0],
            dones=self.dones[:size, 0, None],
            steps=self.steps[:size, 0],
            episode_ids=self.episode_ids[:size, 0],
            sample_counts=self.sample_counts[:size, 0],
            action_log_probs=self.action_log_probs[:size, 0],
            reference_actions=self.reference_actions[:size, 0],
            reference_critic_actions=self.reference_critic_actions[:size, 0],
            priorities=self.priorities[:size, 0],
        )


def summarize_transitions(
    observations: np.ndarray,
    next_observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    steps: np.ndarray,
    sample_counts: np.ndarray,
    detector: UprightDetector,
    current_step: int,
    capacity: int | None = None,
    action_high: np.ndarray | None = None,
) -> dict[str, float]:
    observations = np.asarray(observations, dtype=np.float32).reshape(len(observations), -1)
    next_observations = np.asarray(next_observations, dtype=np.float32).reshape(len(next_observations), -1)
    actions = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    steps = np.asarray(steps, dtype=np.int64).reshape(-1)
    sample_counts = np.asarray(sample_counts, dtype=np.int64).reshape(-1)
    size = len(rewards)

    if size == 0:
        return {"size": 0.0}

    near_obs = detector.near_upright(observations)
    near_next = detector.near_upright(next_observations)
    ages = current_step - steps
    resolved_capacity = int(capacity or size)

    out: dict[str, float] = {
        "size": float(size),
        "capacity": float(resolved_capacity),
        "fill_fraction": float(size / resolved_capacity),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "done_fraction": float(np.mean(dones)),
        "near_upright_obs_fraction": float(np.mean(near_obs)),
        "near_upright_next_obs_fraction": float(np.mean(near_next)),
        "near_upright_any_transition_fraction": float(np.mean(near_obs | near_next)),
        "transition_age_mean": float(np.mean(ages)),
        "transition_age_max": float(np.max(ages)),
        "sample_count_mean": float(np.mean(sample_counts)),
        "sample_count_max": float(np.max(sample_counts)),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "action_abs_max": float(np.max(np.abs(actions))),
    }

    if action_high is not None:
        high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
        high = np.maximum(np.abs(high), 1e-6)
        saturated = np.abs(actions) >= 0.95 * high
        out["action_saturation_fraction"] = float(np.mean(saturated))

    hard_mask = pendulum_hard_state_mask(observations)
    out["pendulum_hard_state_fraction"] = float(np.mean(hard_mask))
    if np.any(hard_mask):
        out["pendulum_hard_state_sample_count_mean"] = float(np.mean(sample_counts[hard_mask]))
        out["pendulum_hard_state_sample_count_max"] = float(np.max(sample_counts[hard_mask]))

    return out


def summarize_saved_replay(
    path: str | Path,
    detector: UprightDetector,
    current_step: int | None = None,
    action_high: np.ndarray | None = None,
) -> dict[str, float]:
    data = np.load(path)
    steps = data["steps"]
    resolved_step = int(current_step if current_step is not None else np.max(steps) if len(steps) else 0)
    return summarize_transitions(
        observations=data["observations"],
        next_observations=data["next_observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
        steps=steps,
        sample_counts=data["sample_counts"],
        detector=detector,
        current_step=resolved_step,
        capacity=len(steps),
        action_high=action_high,
    )


def pendulum_hard_state_mask(
    observations: np.ndarray,
    abs_theta_low: float = 2.0943951023931953,
    abs_theta_high: float = 2.356194490192345,
    velocity_limit: float = 1.0,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    obs = obs.reshape(obs.shape[0], -1)
    if obs.shape[1] != 3:
        return np.zeros(obs.shape[0], dtype=bool)

    theta = np.arctan2(obs[:, 1], obs[:, 0])
    abs_theta = np.abs(theta)
    abs_velocity = np.abs(obs[:, 2])
    return (abs_theta >= abs_theta_low) & (abs_theta <= abs_theta_high) & (abs_velocity <= velocity_limit)
