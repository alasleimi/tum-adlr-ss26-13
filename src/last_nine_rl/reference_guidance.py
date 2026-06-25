from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from last_nine_rl.pendulum_dp import (
    DPSolution,
    PendulumDPParams,
    greedy_actions,
    pendulum_step_model,
    query_value,
)
from last_nine_rl.reference import PendulumEnergySwingupController


DEFAULT_DP_SOLUTION_PATH = Path(
    "reports/pendulum_investigation_20260509/"
    "pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_solution.npz"
)


class PendulumReferenceGuidance:
    """Reference action source for Pendulum replay injection and mixed-behavior data collection."""

    def __init__(
        self,
        policy: str,
        dp_solution_path: str | Path | None = None,
        horizon: int = 200,
    ):
        if policy not in {"controller", "dp", "best"}:
            raise ValueError(f"Unknown reference guidance policy: {policy}")
        self.policy = policy
        self.controller = PendulumEnergySwingupController()
        self.horizon = int(horizon)
        self.solution = None
        self.dp_solution_path: Path | None = None
        if policy in {"dp", "best"}:
            self.dp_solution_path = Path(dp_solution_path) if dp_solution_path else DEFAULT_DP_SOLUTION_PATH
            self.solution = load_dp_solution(self.dp_solution_path)

    def act(self, observation: np.ndarray, remaining_steps: int | None = None) -> np.ndarray:
        return self.act_batch(np.asarray(observation, dtype=np.float32).reshape(1, -1), remaining_steps)[0]

    def act_batch(self, observations: np.ndarray, remaining_steps: int | None = None) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[1] < 3:
            raise ValueError("Pendulum reference guidance expects observations [cos(theta), sin(theta), theta_dot].")
        remaining = self._remaining(remaining_steps)
        if self.policy == "controller":
            return self._controller_actions_from_obs(obs)

        theta, theta_dot = pendulum_states_from_obs(obs)
        dp_actions = self._dp_actions(theta, theta_dot, remaining)
        if self.policy == "dp":
            return dp_actions

        dp_return = self._dp_values(theta, theta_dot, remaining)
        controller_return = self._controller_returns(theta, theta_dot, remaining)
        controller_actions = self._controller_actions_from_obs(obs)
        use_controller = (controller_return > dp_return).reshape(-1, 1)
        return np.where(use_controller, controller_actions, dp_actions).astype(np.float32)

    def model_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        theta, theta_dot = pendulum_state_from_obs(observation)
        reward, next_theta, next_theta_dot = pendulum_step_model(
            np.asarray([theta], dtype=np.float64),
            np.asarray([theta_dot], dtype=np.float64),
            np.asarray(action, dtype=np.float64).reshape(-1)[:1],
            self._params,
        )
        next_obs = np.asarray(
            [math.cos(float(next_theta[0])), math.sin(float(next_theta[0])), float(next_theta_dot[0])],
            dtype=np.float32,
        )
        return next_obs, float(reward[0])

    @property
    def _params(self) -> PendulumDPParams:
        if self.solution is not None:
            return self.solution.params
        return PendulumDPParams(horizon=self.horizon)

    def _remaining(self, remaining_steps: int | None) -> int:
        if remaining_steps is None:
            return max(1, self.horizon)
        return max(1, min(int(remaining_steps), self.horizon))

    def _dp_action(self, theta: float, theta_dot: float, remaining_steps: int) -> float:
        return float(
            self._dp_actions(
                np.asarray([theta], dtype=np.float64),
                np.asarray([theta_dot], dtype=np.float64),
                remaining_steps,
            )[0, 0]
        )

    def _dp_actions(self, theta: np.ndarray, theta_dot: np.ndarray, remaining_steps: int) -> np.ndarray:
        if self.solution is None:
            raise RuntimeError("DP solution is required for DP reference guidance.")
        next_values = self.solution.values_by_remaining[max(remaining_steps - 1, 0)]
        actions = greedy_actions(
            np.asarray(theta, dtype=np.float64),
            np.asarray(theta_dot, dtype=np.float64),
            next_values,
            self.solution,
        )
        return actions.astype(np.float32).reshape(-1, 1)

    def _dp_value(self, theta: float, theta_dot: float, remaining_steps: int) -> float:
        return float(
            self._dp_values(
                np.asarray([theta], dtype=np.float64),
                np.asarray([theta_dot], dtype=np.float64),
                remaining_steps,
            )[0]
        )

    def _dp_values(self, theta: np.ndarray, theta_dot: np.ndarray, remaining_steps: int) -> np.ndarray:
        if self.solution is None:
            raise RuntimeError("DP solution is required for DP reference guidance.")
        values = self.solution.values_by_remaining[remaining_steps]
        return query_value(
            values,
            self.solution.params,
            np.asarray(theta, dtype=np.float64),
            np.asarray(theta_dot, dtype=np.float64),
        )

    def _controller_return(self, theta: float, theta_dot: float, remaining_steps: int) -> float:
        return float(
            self._controller_returns(
                np.asarray([theta], dtype=np.float64),
                np.asarray([theta_dot], dtype=np.float64),
                remaining_steps,
            )[0]
        )

    def _controller_returns(self, theta: np.ndarray, theta_dot: np.ndarray, remaining_steps: int) -> np.ndarray:
        current_theta = np.asarray(theta, dtype=np.float64).copy()
        current_theta_dot = np.asarray(theta_dot, dtype=np.float64).copy()
        total_return = np.zeros_like(current_theta, dtype=np.float64)
        for _ in range(remaining_steps):
            action = self._controller_actions_from_state(current_theta, current_theta_dot).reshape(-1)
            reward, next_theta, next_theta_dot = pendulum_step_model(
                current_theta,
                current_theta_dot,
                action,
                self._params,
            )
            total_return += reward
            current_theta = next_theta
            current_theta_dot = next_theta_dot
        return total_return

    def _controller_actions_from_obs(self, observations: np.ndarray) -> np.ndarray:
        theta, theta_dot = pendulum_states_from_obs(observations)
        return self._controller_actions_from_state(theta, theta_dot)

    def _controller_actions_from_state(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        theta_dot = np.asarray(theta_dot, dtype=np.float64)
        local = (np.abs(theta) <= self.controller.switch_angle) & (
            np.abs(theta_dot) <= self.controller.switch_velocity
        )
        pd_torque = -self.controller.kp * theta - self.controller.kd * theta_dot
        energy = 0.5 * theta_dot * theta_dot + self.controller.upright_energy * np.cos(theta)
        swing_torque = -self.controller.energy_gain * (energy - self.controller.upright_energy) * theta_dot
        small_velocity = np.abs(theta_dot) < 0.1
        swing_torque = swing_torque + np.where(
            small_velocity,
            -0.5 * np.sign(np.where(theta != 0.0, theta, 1.0)),
            0.0,
        )
        torque = np.where(local, pd_torque, swing_torque)
        return np.clip(torque, -self.controller.max_torque, self.controller.max_torque).astype(np.float32).reshape(-1, 1)

    def metadata(self) -> dict[str, Any]:
        return {
            "reference_policy": self.policy,
            "horizon": self.horizon,
            "dp_solution_path": str(self.dp_solution_path) if self.dp_solution_path is not None else None,
        }


def load_dp_solution(path: str | Path) -> DPSolution:
    solution_path = Path(path)
    if not solution_path.is_file():
        raise FileNotFoundError(
            f"DP solution not found at {solution_path}. Run last_nine_rl.pendulum_dp with --save-solution first."
        )
    with np.load(solution_path, allow_pickle=False) as data:
        params_raw = data["params"]
        if isinstance(params_raw, np.ndarray):
            params_json = str(params_raw.item())
        else:
            params_json = str(params_raw)
        params = PendulumDPParams(**json.loads(params_json))
        return DPSolution(
            params=params,
            theta_values=np.asarray(data["theta_values"], dtype=np.float64),
            velocity_values=np.asarray(data["velocity_values"], dtype=np.float64),
            actions=np.asarray(data["actions"], dtype=np.float64),
            values_by_remaining=np.asarray(data["values_by_remaining"], dtype=np.float32),
        )


def pendulum_state_from_obs(observation: np.ndarray) -> tuple[float, float]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.shape[0] < 3:
        raise ValueError("Pendulum reference guidance expects observation [cos(theta), sin(theta), theta_dot].")
    return float(math.atan2(float(obs[1]), float(obs[0]))), float(obs[2])


def pendulum_states_from_obs(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] < 3:
        raise ValueError("Pendulum reference guidance expects observations [cos(theta), sin(theta), theta_dot].")
    theta = np.arctan2(obs[:, 1], obs[:, 0]).astype(np.float64)
    theta_dot = obs[:, 2].astype(np.float64)
    return theta, theta_dot
