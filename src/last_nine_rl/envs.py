from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Float32Observation(gym.ObservationWrapper):
    """Cast flat observations to float32 so PyTorch tensors are stable."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(f"Float32Observation requires Box observations, got {env.observation_space}")
        self.observation_space = spaces.Box(
            low=np.asarray(env.observation_space.low, dtype=np.float32),
            high=np.asarray(env.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)


class CastAction(gym.ActionWrapper):
    """Cast actions back to the wrapped environment action dtype."""

    def action(self, action: Any) -> np.ndarray:
        return np.asarray(action, dtype=self.env.action_space.dtype)


class PendulumHardReset(gym.Wrapper):
    """Bias a fraction of Pendulum training resets toward a configured hard angle band."""

    def __init__(
        self,
        env: gym.Env,
        probability: float,
        abs_theta_low: float,
        abs_theta_high: float,
        velocity_limit: float,
    ):
        super().__init__(env)
        self.probability = float(probability)
        self.abs_theta_low = float(abs_theta_low)
        self.abs_theta_high = float(abs_theta_high)
        self.velocity_limit = float(velocity_limit)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.probability <= 0.0:
            return obs, info

        rng = self.env.unwrapped.np_random
        hard = bool(rng.random() < self.probability)
        if not hard:
            return obs, {**info, "hard_reset": False}

        abs_theta = rng.uniform(self.abs_theta_low, self.abs_theta_high)
        theta = -abs_theta if rng.random() < 0.5 else abs_theta
        theta_dot = rng.uniform(-self.velocity_limit, self.velocity_limit)

        unwrapped = self.env.unwrapped
        unwrapped.state = np.asarray([theta, theta_dot], dtype=np.float64)
        unwrapped.last_u = None
        obs = np.asarray(unwrapped._get_obs())
        return obs, {
            **info,
            "hard_reset": True,
            "hard_reset_theta": float(theta),
            "hard_reset_theta_dot": float(theta_dot),
        }


class PendulumFailureReset(gym.Wrapper):
    """Draw a configured fraction of training resets from a learned state bank."""

    def __init__(self, env: gym.Env, probability: float):
        super().__init__(env)
        self.probability = float(probability)
        self._states = np.empty((0, 2), dtype=np.float64)
        self.total_resets = 0
        self.curriculum_resets = 0

    @property
    def state_count(self) -> int:
        return int(len(self._states))

    def replace_states(self, states: np.ndarray) -> None:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != (2,):
            raise ValueError("Pendulum failure-reset states must have shape (N, 2).")
        if not np.all(np.isfinite(values)):
            raise ValueError("Pendulum failure-reset states must be finite.")
        self._states = values.copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.total_resets += 1
        if self.probability <= 0.0 or self.state_count == 0:
            return obs, {
                **info,
                "failure_curriculum_reset": False,
                "failure_curriculum_bank_size": self.state_count,
            }

        rng = self.env.unwrapped.np_random
        use_curriculum = bool(rng.random() < self.probability)
        if not use_curriculum:
            return obs, {
                **info,
                "failure_curriculum_reset": False,
                "failure_curriculum_bank_size": self.state_count,
            }

        index = int(rng.integers(0, self.state_count))
        theta, theta_dot = self._states[index]
        unwrapped = self.env.unwrapped
        unwrapped.state = np.asarray([theta, theta_dot], dtype=np.float64)
        unwrapped.last_u = None
        self.curriculum_resets += 1
        obs = np.asarray(unwrapped._get_obs())
        return obs, {
            **info,
            "failure_curriculum_reset": True,
            "failure_curriculum_bank_size": self.state_count,
            "failure_curriculum_state_index": index,
            "failure_curriculum_theta": float(theta),
            "failure_curriculum_theta_dot": float(theta_dot),
        }


def set_pendulum_hard_reset_probability(env: gym.Env, probability: float) -> bool:
    """Update PendulumHardReset probability through a wrapper stack."""

    current: gym.Env | None = env
    while current is not None:
        if isinstance(current, PendulumHardReset):
            current.probability = float(probability)
            return True
        current = getattr(current, "env", None)
    return False


def replace_pendulum_failure_reset_states(env: gym.Env, states: np.ndarray) -> bool:
    """Replace the automatic failure-start bank through a wrapper stack."""

    current: gym.Env | None = env
    while current is not None:
        if isinstance(current, PendulumFailureReset):
            current.replace_states(states)
            return True
        current = getattr(current, "env", None)
    return False


def pendulum_failure_reset_stats(env: gym.Env) -> dict[str, int] | None:
    """Return reset-bank usage counters through a wrapper stack."""

    current: gym.Env | None = env
    while current is not None:
        if isinstance(current, PendulumFailureReset):
            return {
                "state_count": current.state_count,
                "total_resets": int(current.total_resets),
                "curriculum_resets": int(current.curriculum_resets),
            }
        current = getattr(current, "env", None)
    return None


def make_env(
    env_id: str,
    seed: int,
    max_episode_steps: int | None = None,
    pendulum_hard_reset_prob: float = 0.0,
    pendulum_hard_reset_enabled: bool | None = None,
    pendulum_hard_reset_abs_theta_low: float = 2.0943951023931953,
    pendulum_hard_reset_abs_theta_high: float = 2.356194490192345,
    pendulum_hard_reset_velocity_limit: float = 1.0,
    pendulum_failure_reset_prob: float = 0.0,
    pendulum_failure_reset_enabled: bool | None = None,
) -> gym.Env:
    if env_id.startswith("dm_control/"):
        import shimmy  # noqa: F401  Registers dm_control/... Gymnasium IDs.

    hard_reset_enabled = bool(
        pendulum_hard_reset_prob > 0.0 if pendulum_hard_reset_enabled is None else pendulum_hard_reset_enabled
    )
    if hard_reset_enabled and not env_id.startswith("Pendulum"):
        raise ValueError("Pendulum hard-reset curriculum only supports Pendulum environments.")
    failure_reset_enabled = bool(
        pendulum_failure_reset_prob > 0.0
        if pendulum_failure_reset_enabled is None
        else pendulum_failure_reset_enabled
    )
    if failure_reset_enabled and not env_id.startswith("Pendulum"):
        raise ValueError("Pendulum failure-reset curriculum only supports Pendulum environments.")

    env = gym.make(env_id)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if isinstance(env.observation_space, spaces.Dict | spaces.Tuple):
        env = gym.wrappers.FlattenObservation(env)

    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError(f"SAC requires Box observations, got {env.observation_space}")
    if not isinstance(env.action_space, spaces.Box):
        raise TypeError(f"SAC requires Box actions, got {env.action_space}")

    if hard_reset_enabled:
        env = PendulumHardReset(
            env,
            probability=pendulum_hard_reset_prob,
            abs_theta_low=pendulum_hard_reset_abs_theta_low,
            abs_theta_high=pendulum_hard_reset_abs_theta_high,
            velocity_limit=pendulum_hard_reset_velocity_limit,
        )

    if failure_reset_enabled:
        env = PendulumFailureReset(env, probability=pendulum_failure_reset_prob)

    env = Float32Observation(env)
    env = CastAction(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


@dataclass(frozen=True)
class UprightDetector:
    env_id: str
    cos_threshold: float = 0.95
    abs_velocity_threshold: float = 1.0

    def near_upright(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape[1] < 3:
            return np.zeros(obs.shape[0], dtype=bool)

        if self.env_id.startswith("Pendulum"):
            cos_theta = obs[:, 0]
            angular_velocity = np.abs(obs[:, 2])
            return (cos_theta >= self.cos_threshold) & (angular_velocity <= self.abs_velocity_threshold)

        if self.env_id.startswith("dm_control/cartpole"):
            # Shimmy flattens DMC cartpole observations as:
            # position[cart_x, pole_cos, pole_sin], velocity[cart_v, pole_v].
            if obs.shape[1] < 5:
                return np.zeros(obs.shape[0], dtype=bool)
            pole_cos = obs[:, 1]
            pole_velocity = np.abs(obs[:, 4])
            return (pole_cos >= self.cos_threshold) & (pole_velocity <= self.abs_velocity_threshold)

        return np.zeros(obs.shape[0], dtype=bool)
