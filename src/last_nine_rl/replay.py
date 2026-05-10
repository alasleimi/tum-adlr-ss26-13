from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cleanrl_utils.buffers import ReplayBuffer as CleanRLReplayBuffer

from last_nine_rl.envs import UprightDetector


class InstrumentedReplayBuffer(CleanRLReplayBuffer):
    """CleanRL replay buffer with Week 1 inspection telemetry."""

    steps: np.ndarray
    episode_ids: np.ndarray
    sample_counts: np.ndarray

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.sample_counts = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

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
    ) -> None:
        write_pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.sample_counts[write_pos, :] = 0
        if step is not None:
            self.steps[write_pos, :] = int(step)
        if episode_id is not None:
            self.episode_ids[write_pos, :] = int(episode_id)

    def sample(self, batch_size: int, count: bool = True):
        if self.n_envs != 1 or self.optimize_memory_usage:
            return super().sample(batch_size)
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        if count:
            np.add.at(self.sample_counts[:, 0], batch_inds, 1)
        return self._get_samples(batch_inds)

    def summary(
        self,
        detector: UprightDetector,
        current_step: int,
        action_high: np.ndarray | None = None,
    ) -> dict[str, float]:
        size = self.size()
        if size == 0:
            return {"size": 0.0}

        return summarize_transitions(
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
