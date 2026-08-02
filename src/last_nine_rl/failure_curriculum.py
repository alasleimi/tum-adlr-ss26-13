from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np

from last_nine_rl.envs import make_env


class DeterministicPolicy(Protocol):
    """Minimal policy surface needed by reward-only failure discovery."""

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray: ...


@dataclass(frozen=True)
class FailureDiscoveryResult:
    """Candidate returns and the automatically selected worst initial states."""

    candidate_states: np.ndarray
    candidate_returns: np.ndarray
    selected_states: np.ndarray
    selected_returns: np.ndarray
    environment_steps: int

    def scalar_metrics(self) -> dict[str, float]:
        candidate_returns = np.asarray(self.candidate_returns, dtype=np.float64)
        selected_returns = np.asarray(self.selected_returns, dtype=np.float64)
        candidate_states = np.asarray(self.candidate_states, dtype=np.float64)
        selected_states = np.asarray(self.selected_states, dtype=np.float64)
        return {
            "candidate_count": float(len(candidate_returns)),
            "selected_count": float(len(selected_returns)),
            "discovery_environment_steps": float(self.environment_steps),
            "candidate_return_mean": float(np.mean(candidate_returns)),
            "candidate_return_std": float(np.std(candidate_returns)),
            "candidate_return_min": float(np.min(candidate_returns)),
            "candidate_return_q10": float(np.quantile(candidate_returns, 0.10)),
            "candidate_return_q25": float(np.quantile(candidate_returns, 0.25)),
            "candidate_return_median": float(np.median(candidate_returns)),
            "candidate_return_max": float(np.max(candidate_returns)),
            "selected_return_mean": float(np.mean(selected_returns)),
            "selected_return_min": float(np.min(selected_returns)),
            "selected_return_max": float(np.max(selected_returns)),
            "candidate_abs_theta_mean": float(np.mean(np.abs(candidate_states[:, 0]))),
            "candidate_abs_theta_dot_mean": float(np.mean(np.abs(candidate_states[:, 1]))),
            "selected_abs_theta_mean": float(np.mean(np.abs(selected_states[:, 0]))),
            "selected_abs_theta_dot_mean": float(np.mean(np.abs(selected_states[:, 1]))),
        }


class PendulumFailureStartCurriculum:
    """Discover difficult Pendulum starts from reward-only policy rollouts.

    Discovery uses a dedicated environment and never exposes its transitions to
    the learner.  The resulting state bank is consumed only by training resets.
    Consequently ``environment_steps`` is an explicit auxiliary-simulation cost,
    not a replay or learning-transition count.
    """

    def __init__(
        self,
        *,
        env_id: str,
        max_episode_steps: int | None,
        seed: int,
        start_step: int,
        refresh_interval_steps: int,
        candidate_count: int,
        worst_fraction: float,
        rollouts_per_candidate: int,
        rollout_horizon: int,
    ) -> None:
        if not env_id.startswith("Pendulum"):
            raise ValueError("Failure-start curriculum currently supports only Pendulum environments.")
        self.start_step = int(start_step)
        self.refresh_interval_steps = int(refresh_interval_steps)
        self.candidate_count = int(candidate_count)
        self.worst_fraction = float(worst_fraction)
        self.rollouts_per_candidate = int(rollouts_per_candidate)
        self.rollout_horizon = int(rollout_horizon)
        self.discovery_environment_steps = 0
        self.refresh_count = 0
        self._env = make_env(
            env_id,
            seed=int(seed),
            max_episode_steps=max_episode_steps,
        )
        # Seed once. Subsequent ordinary resets advance a private candidate RNG.
        self._env.reset(seed=int(seed))
        episode_limit = getattr(self._env.spec, "max_episode_steps", None)
        effective_horizon = self.rollout_horizon
        if episode_limit is not None:
            effective_horizon = min(effective_horizon, int(episode_limit))
        self.environment_steps_per_refresh_upper_bound = int(
            self.candidate_count * self.rollouts_per_candidate * effective_horizon
        )

    def should_refresh(self, learning_step: int) -> bool:
        step = int(learning_step)
        return (
            step >= self.start_step
            and (step - self.start_step) % self.refresh_interval_steps == 0
        )

    def refresh(
        self,
        policy: DeterministicPolicy,
        *,
        learning_step: int,
    ) -> FailureDiscoveryResult:
        result = discover_pendulum_failure_starts(
            policy,
            self._env,
            candidate_count=self.candidate_count,
            worst_fraction=self.worst_fraction,
            rollouts_per_candidate=self.rollouts_per_candidate,
            rollout_horizon=self.rollout_horizon,
        )
        self.discovery_environment_steps += int(result.environment_steps)
        self.refresh_count += 1
        return result

    def planned_refresh_steps(self, total_learning_steps: int) -> tuple[int, ...]:
        """Learning steps that will trigger useful pre-terminal refreshes."""

        total = int(total_learning_steps)
        first_step = self.start_step
        if first_step < 1:
            first_step += int(np.ceil((1 - first_step) / self.refresh_interval_steps)) * (
                self.refresh_interval_steps
            )
        if first_step >= total:
            return ()
        return tuple(range(first_step, total, self.refresh_interval_steps))

    def planned_environment_steps_upper_bound(self, total_learning_steps: int) -> int:
        return int(
            len(self.planned_refresh_steps(total_learning_steps))
            * self.environment_steps_per_refresh_upper_bound
        )

    def close(self) -> None:
        self._env.close()


def discover_pendulum_failure_starts(
    policy: DeterministicPolicy,
    env: gym.Env,
    *,
    candidate_count: int,
    worst_fraction: float,
    rollouts_per_candidate: int,
    rollout_horizon: int,
) -> FailureDiscoveryResult:
    """Sample full-support resets and retain states with the lowest actor return."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not (0.0 < worst_fraction <= 1.0):
        raise ValueError("worst_fraction must be in (0, 1]")
    if rollouts_per_candidate <= 0:
        raise ValueError("rollouts_per_candidate must be positive")
    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")
    if not getattr(env.unwrapped.spec, "id", "").startswith("Pendulum"):
        raise ValueError("Failure-start discovery requires a Pendulum environment.")

    # Sampling via the environment's own reset distribution avoids imposing a
    # hand-written angle band.  It also tracks future reset-support changes.
    candidate_states = np.empty((candidate_count, 2), dtype=np.float64)
    for candidate_index in range(candidate_count):
        env.reset()
        candidate_states[candidate_index] = np.asarray(env.unwrapped.state, dtype=np.float64)

    returns = np.empty((candidate_count, rollouts_per_candidate), dtype=np.float64)
    environment_steps = 0
    for candidate_index, state in enumerate(candidate_states):
        for rollout_index in range(rollouts_per_candidate):
            observation = reset_pendulum_to_state(env, state)
            episode_return = 0.0
            for _ in range(rollout_horizon):
                action = policy.act(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                environment_steps += 1
                episode_return += float(reward)
                if bool(terminated or truncated):
                    break
            returns[candidate_index, rollout_index] = episode_return

    mean_returns = np.mean(returns, axis=1)
    selected_count = max(1, int(np.ceil(candidate_count * worst_fraction)))
    # Stable sorting makes ties reproducible in original candidate order.
    selected_indices = np.argsort(mean_returns, kind="stable")[:selected_count]
    return FailureDiscoveryResult(
        candidate_states=candidate_states.copy(),
        candidate_returns=mean_returns.copy(),
        selected_states=candidate_states[selected_indices].copy(),
        selected_returns=mean_returns[selected_indices].copy(),
        environment_steps=environment_steps,
    )


def reset_pendulum_to_state(env: gym.Env, state: np.ndarray) -> np.ndarray:
    """Reset wrapper episode counters, then install an exact Pendulum state."""

    env.reset()
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.shape != (2,) or not np.all(np.isfinite(state_array)):
        raise ValueError("Pendulum state must contain finite [theta, theta_dot].")
    unwrapped = env.unwrapped
    unwrapped.state = state_array.copy()
    unwrapped.last_u = None
    return np.asarray(unwrapped._get_obs(), dtype=np.float32)
