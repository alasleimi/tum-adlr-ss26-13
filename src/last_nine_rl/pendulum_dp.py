from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.config import ReliabilityConfig


@dataclass(frozen=True)
class PendulumDPParams:
    horizon: int = 200
    theta_bins: int = 241
    velocity_bins: int = 161
    action_bins: int = 81
    eval_theta_bins: int = 61
    eval_velocity_bins: int = 41
    eval_velocity_limit: float = 1.0
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0

    def validate(self) -> None:
        errors: list[str] = []
        for name in (
            "horizon",
            "theta_bins",
            "velocity_bins",
            "action_bins",
            "eval_theta_bins",
            "eval_velocity_bins",
        ):
            if getattr(self, name) <= 1:
                errors.append(f"{name} must be greater than 1")
        if self.max_speed <= 0.0:
            errors.append("max_speed must be positive")
        if self.max_torque <= 0.0:
            errors.append("max_torque must be positive")
        if self.eval_velocity_limit <= 0.0 or self.eval_velocity_limit > self.max_speed:
            errors.append("eval_velocity_limit must be in (0, max_speed]")
        if errors:
            raise ValueError("; ".join(errors))


@dataclass
class DPSolution:
    params: PendulumDPParams
    theta_values: np.ndarray
    velocity_values: np.ndarray
    actions: np.ndarray
    values_by_remaining: np.ndarray


@dataclass
class TransitionTable:
    rewards: np.ndarray
    i00: np.ndarray
    i01: np.ndarray
    i10: np.ndarray
    i11: np.ndarray
    w00: np.ndarray
    w01: np.ndarray
    w10: np.ndarray
    w11: np.ndarray


def main() -> None:
    args = parse_args()
    params = PendulumDPParams(
        horizon=args.horizon,
        theta_bins=args.theta_bins,
        velocity_bins=args.velocity_bins,
        action_bins=args.action_bins,
        eval_theta_bins=args.eval_theta_bins,
        eval_velocity_bins=args.eval_velocity_bins,
        eval_velocity_limit=args.eval_velocity_limit,
    )
    reliability = ReliabilityConfig(success_return_threshold=args.success_return_threshold)
    result = run_pendulum_dp_report(
        output_dir=Path(args.out),
        params=params,
        reliability=reliability,
        sac_grid_path=Path(args.sac_grid) if args.sac_grid else None,
        save_solution=args.save_solution,
    )
    print(json.dumps(result["summary"], allow_nan=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-horizon dynamic-programming calibration for Pendulum-v1.")
    parser.add_argument("--out", required=True, help="Output directory for DP calibration artifacts.")
    parser.add_argument("--sac-grid", default=None, help="Optional checkpoint grid summary CSV to join against.")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--theta-bins", type=int, default=241)
    parser.add_argument("--velocity-bins", type=int, default=161)
    parser.add_argument("--action-bins", type=int, default=81)
    parser.add_argument("--eval-theta-bins", type=int, default=61)
    parser.add_argument("--eval-velocity-bins", type=int, default=41)
    parser.add_argument("--eval-velocity-limit", type=float, default=1.0)
    parser.add_argument("--success-return-threshold", type=float, default=-200.0)
    parser.add_argument("--save-solution", action="store_true", help="Save value tables as a compressed NPZ file.")
    return parser.parse_args()


def run_pendulum_dp_report(
    output_dir: Path,
    params: PendulumDPParams,
    reliability: ReliabilityConfig,
    sac_grid_path: Path | None = None,
    save_solution: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    solution = solve_finite_horizon_dp(params)
    solve_seconds = time.perf_counter() - start

    eval_theta_values = np.linspace(-math.pi, math.pi, params.eval_theta_bins, endpoint=False, dtype=np.float64)
    eval_velocity_values = np.linspace(
        -params.eval_velocity_limit,
        params.eval_velocity_limit,
        params.eval_velocity_bins,
        dtype=np.float64,
    )
    dp_rows = evaluate_dp_on_grid(solution, eval_theta_values, eval_velocity_values, reliability)
    write_csv(output_dir / "pendulum_dp_grid.csv", dp_rows)

    comparison_rows: list[dict[str, Any]] | None = None
    if sac_grid_path is not None:
        comparison_rows = join_sac_grid(dp_rows, read_csv(sac_grid_path), reliability)
        write_csv(output_dir / "pendulum_dp_sac_comparison.csv", comparison_rows)

    if save_solution:
        np.savez_compressed(
            output_dir / "pendulum_dp_solution.npz",
            theta_values=solution.theta_values,
            velocity_values=solution.velocity_values,
            actions=solution.actions,
            values_by_remaining=solution.values_by_remaining,
            params=json.dumps(asdict(params), sort_keys=True),
        )

    figures = write_report_figures(
        output_dir=output_dir,
        dp_rows=dp_rows,
        comparison_rows=comparison_rows,
        theta_values=eval_theta_values,
        velocity_values=eval_velocity_values,
        success_return_threshold=reliability.success_return_threshold,
    )
    summary = summarize_report(
        params=params,
        reliability=reliability,
        dp_rows=dp_rows,
        comparison_rows=comparison_rows,
        solve_seconds=solve_seconds,
        sac_grid_path=sac_grid_path,
    )
    (output_dir / "pendulum_dp_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_index(output_dir, figures, summary)
    return {"summary": summary, "dp_rows": dp_rows, "comparison_rows": comparison_rows}


def solve_finite_horizon_dp(params: PendulumDPParams) -> DPSolution:
    params.validate()
    theta_values = np.linspace(-math.pi, math.pi, params.theta_bins, endpoint=False, dtype=np.float64)
    velocity_values = np.linspace(-params.max_speed, params.max_speed, params.velocity_bins, dtype=np.float64)
    actions = np.linspace(-params.max_torque, params.max_torque, params.action_bins, dtype=np.float64)
    transitions = precompute_transitions(params, theta_values, velocity_values, actions)

    num_states = params.theta_bins * params.velocity_bins
    values = np.zeros((params.horizon + 1, num_states), dtype=np.float32)
    for remaining in range(1, params.horizon + 1):
        values[remaining] = bellman_backup(values[remaining - 1], transitions)
    return DPSolution(
        params=params,
        theta_values=theta_values,
        velocity_values=velocity_values,
        actions=actions,
        values_by_remaining=values,
    )


def precompute_transitions(
    params: PendulumDPParams,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    actions: np.ndarray,
) -> TransitionTable:
    theta_mesh, velocity_mesh = np.meshgrid(theta_values, velocity_values)
    flat_theta = theta_mesh.ravel()
    flat_velocity = velocity_mesh.ravel()
    num_actions = len(actions)
    num_states = len(flat_theta)

    rewards = np.empty((num_actions, num_states), dtype=np.float32)
    i00 = np.empty((num_actions, num_states), dtype=np.int32)
    i01 = np.empty((num_actions, num_states), dtype=np.int32)
    i10 = np.empty((num_actions, num_states), dtype=np.int32)
    i11 = np.empty((num_actions, num_states), dtype=np.int32)
    w00 = np.empty((num_actions, num_states), dtype=np.float32)
    w01 = np.empty((num_actions, num_states), dtype=np.float32)
    w10 = np.empty((num_actions, num_states), dtype=np.float32)
    w11 = np.empty((num_actions, num_states), dtype=np.float32)

    for action_idx, action in enumerate(actions):
        reward, next_theta, next_velocity = pendulum_step_model(flat_theta, flat_velocity, action, params)
        indices, weights = interpolation_weights(
            next_theta,
            next_velocity,
            theta_bins=params.theta_bins,
            velocity_bins=params.velocity_bins,
            max_speed=params.max_speed,
        )
        rewards[action_idx] = reward.astype(np.float32)
        i00[action_idx], i01[action_idx], i10[action_idx], i11[action_idx] = indices
        w00[action_idx], w01[action_idx], w10[action_idx], w11[action_idx] = weights

    return TransitionTable(
        rewards=rewards,
        i00=i00,
        i01=i01,
        i10=i10,
        i11=i11,
        w00=w00,
        w01=w01,
        w10=w10,
        w11=w11,
    )


def bellman_backup(next_values: np.ndarray, transitions: TransitionTable) -> np.ndarray:
    best = np.full(next_values.shape, -np.inf, dtype=np.float32)
    for action_idx in range(transitions.rewards.shape[0]):
        future = interpolate_precomputed(next_values, transitions, action_idx)
        q_value = transitions.rewards[action_idx] + future
        np.maximum(best, q_value, out=best)
    return best


def interpolate_precomputed(values: np.ndarray, transitions: TransitionTable, action_idx: int) -> np.ndarray:
    return (
        transitions.w00[action_idx] * values[transitions.i00[action_idx]]
        + transitions.w01[action_idx] * values[transitions.i01[action_idx]]
        + transitions.w10[action_idx] * values[transitions.i10[action_idx]]
        + transitions.w11[action_idx] * values[transitions.i11[action_idx]]
    )


def evaluate_dp_on_grid(
    solution: DPSolution,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    reliability: ReliabilityConfig,
) -> list[dict[str, Any]]:
    theta_mesh, velocity_mesh = np.meshgrid(theta_values, velocity_values)
    initial_theta = theta_mesh.ravel()
    initial_velocity = velocity_mesh.ravel()
    rollout = rollout_greedy_policy(solution, initial_theta, initial_velocity, reliability)
    dp_value = query_value(
        solution.values_by_remaining[solution.params.horizon],
        solution.params,
        initial_theta,
        initial_velocity,
    )

    rows: list[dict[str, Any]] = []
    for idx, (theta, velocity) in enumerate(zip(initial_theta, initial_velocity, strict=True)):
        rows.append(
            {
                "theta": float(theta),
                "theta_degrees": float(np.degrees(theta)),
                "theta_dot": float(velocity),
                "dp_value": float(dp_value[idx]),
                "dp_policy_return": float(rollout["return"][idx]),
                "dp_value_return_success": float(dp_value[idx] >= reliability.success_return_threshold),
                "dp_policy_return_success": float(rollout["return_success"][idx]),
                "dp_policy_strict_success": float(rollout["strict_success"][idx]),
                "dp_near_upright_fraction": float(rollout["near_upright_fraction"][idx]),
                "dp_not_near_upright_streak": float(rollout["not_near_upright_streak"][idx]),
            }
        )
    return rows


def rollout_greedy_policy(
    solution: DPSolution,
    initial_theta: np.ndarray,
    initial_velocity: np.ndarray,
    reliability: ReliabilityConfig,
) -> dict[str, np.ndarray]:
    params = solution.params
    theta = angle_normalize(np.asarray(initial_theta, dtype=np.float64))
    velocity = np.asarray(initial_velocity, dtype=np.float64)
    returns = np.zeros(theta.shape, dtype=np.float64)
    near_count = np.zeros(theta.shape, dtype=np.int32)
    current_not_near_streak = np.zeros(theta.shape, dtype=np.int32)
    longest_not_near_streak = np.zeros(theta.shape, dtype=np.int32)

    for remaining in range(params.horizon, 0, -1):
        values_next = solution.values_by_remaining[remaining - 1]
        action = greedy_actions(theta, velocity, values_next, solution)
        reward, theta, velocity = pendulum_step_model(theta, velocity, action, params)
        theta = angle_normalize(theta)
        returns += reward

        near = (np.cos(theta) >= reliability.near_upright_cos_threshold) & (
            np.abs(velocity) <= reliability.near_upright_abs_velocity_threshold
        )
        near_count += near.astype(np.int32)
        current_not_near_streak = np.where(near, 0, current_not_near_streak + 1)
        longest_not_near_streak = np.maximum(longest_not_near_streak, current_not_near_streak)

    near_fraction = near_count / max(params.horizon, 1)
    return_success = returns >= reliability.success_return_threshold
    stability_success = near_fraction >= reliability.success_near_upright_fraction_threshold
    streak_success = longest_not_near_streak <= reliability.success_max_not_near_upright_streak
    return {
        "return": returns,
        "near_upright_fraction": near_fraction,
        "not_near_upright_streak": longest_not_near_streak.astype(np.float64),
        "return_success": return_success.astype(np.float64),
        "strict_success": (return_success & stability_success & streak_success).astype(np.float64),
    }


def greedy_actions(
    theta: np.ndarray,
    velocity: np.ndarray,
    next_values: np.ndarray,
    solution: DPSolution,
) -> np.ndarray:
    best_q = np.full(theta.shape, -np.inf, dtype=np.float64)
    best_action = np.zeros(theta.shape, dtype=np.float64)
    for action in solution.actions:
        reward, next_theta, next_velocity = pendulum_step_model(theta, velocity, action, solution.params)
        future = query_value(next_values, solution.params, next_theta, next_velocity)
        q_value = reward + future
        improved = q_value > best_q
        best_q = np.where(improved, q_value, best_q)
        best_action = np.where(improved, action, best_action)
    return best_action


def pendulum_step_model(
    theta: np.ndarray | float,
    theta_dot: np.ndarray | float,
    action: np.ndarray | float,
    params: PendulumDPParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_array = np.asarray(theta, dtype=np.float64)
    theta_dot_array = np.asarray(theta_dot, dtype=np.float64)
    torque = np.clip(np.asarray(action, dtype=np.float64), -params.max_torque, params.max_torque)
    reward = -(
        angle_normalize(theta_array) ** 2
        + 0.1 * theta_dot_array**2
        + 0.001 * torque**2
    )
    next_theta_dot = theta_dot_array + (
        3.0 * params.gravity / (2.0 * params.length) * np.sin(theta_array)
        + 3.0 / (params.mass * params.length**2) * torque
    ) * params.dt
    next_theta_dot = np.clip(next_theta_dot, -params.max_speed, params.max_speed)
    next_theta = theta_array + next_theta_dot * params.dt
    return reward, next_theta, next_theta_dot


def query_value(
    values_flat: np.ndarray,
    params: PendulumDPParams,
    theta: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    indices, weights = interpolation_weights(
        theta,
        velocity,
        theta_bins=params.theta_bins,
        velocity_bins=params.velocity_bins,
        max_speed=params.max_speed,
    )
    i00, i01, i10, i11 = indices
    w00, w01, w10, w11 = weights
    return w00 * values_flat[i00] + w01 * values_flat[i01] + w10 * values_flat[i10] + w11 * values_flat[i11]


def interpolation_weights(
    theta: np.ndarray,
    velocity: np.ndarray,
    theta_bins: int,
    velocity_bins: int,
    max_speed: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    theta_step = 2.0 * math.pi / theta_bins
    theta_pos = (angle_normalize(theta) + math.pi) / theta_step
    theta_floor = np.floor(theta_pos)
    theta0 = np.mod(theta_floor.astype(np.int64), theta_bins)
    theta1 = (theta0 + 1) % theta_bins
    theta_frac = (theta_pos - theta_floor).astype(np.float32)

    velocity_clipped = np.clip(velocity, -max_speed, max_speed)
    velocity_step = 2.0 * max_speed / (velocity_bins - 1)
    velocity_pos = (velocity_clipped + max_speed) / velocity_step
    velocity0 = np.floor(velocity_pos).astype(np.int64)
    velocity0 = np.clip(velocity0, 0, velocity_bins - 2)
    velocity1 = velocity0 + 1
    velocity_frac = (velocity_pos - velocity0).astype(np.float32)

    i00 = (velocity0 * theta_bins + theta0).astype(np.int32)
    i01 = (velocity0 * theta_bins + theta1).astype(np.int32)
    i10 = (velocity1 * theta_bins + theta0).astype(np.int32)
    i11 = (velocity1 * theta_bins + theta1).astype(np.int32)

    w00 = ((1.0 - velocity_frac) * (1.0 - theta_frac)).astype(np.float32)
    w01 = ((1.0 - velocity_frac) * theta_frac).astype(np.float32)
    w10 = (velocity_frac * (1.0 - theta_frac)).astype(np.float32)
    w11 = (velocity_frac * theta_frac).astype(np.float32)
    return (i00, i01, i10, i11), (w00, w01, w10, w11)


def angle_normalize(x: np.ndarray | float) -> np.ndarray:
    return ((np.asarray(x) + math.pi) % (2.0 * math.pi)) - math.pi


def join_sac_grid(
    dp_rows: list[dict[str, Any]],
    sac_rows: list[dict[str, Any]],
    reliability: ReliabilityConfig,
) -> list[dict[str, Any]]:
    if len(dp_rows) != len(sac_rows):
        raise ValueError(f"DP grid has {len(dp_rows)} rows but SAC grid has {len(sac_rows)} rows.")
    joined: list[dict[str, Any]] = []
    for dp_row, sac_row in zip(dp_rows, sac_rows, strict=True):
        if not (
            math.isclose(float(dp_row["theta"]), float(sac_row["theta"]), abs_tol=1e-10)
            and math.isclose(float(dp_row["theta_dot"]), float(sac_row["theta_dot"]), abs_tol=1e-10)
        ):
            raise ValueError("DP and SAC grids are not ordered on the same initial states.")
        sac_mean_return = float(sac_row["mean_return"])
        sac_return_success_rate = float(sac_row["return_success_rate"])
        sac_strict_success_rate = float(sac_row["strict_success_rate"])
        dp_policy_return = float(dp_row["dp_policy_return"])
        dp_feasible = float(dp_row["dp_policy_return_success"])
        dp_strict_feasible = float(dp_row["dp_policy_strict_success"])
        joined.append(
            {
                **dp_row,
                "sac_mean_return": sac_mean_return,
                "sac_return_success_rate": sac_return_success_rate,
                "sac_strict_success_rate": sac_strict_success_rate,
                "sac_mean_regret_to_dp_policy": dp_policy_return - sac_mean_return,
                "sac_failure_rate_on_dp_feasible": dp_feasible * (1.0 - sac_return_success_rate),
                "sac_strict_failure_rate_on_dp_strict_feasible": dp_strict_feasible * (1.0 - sac_strict_success_rate),
                "sac_all_seed_failure_on_dp_feasible": float(dp_feasible and sac_return_success_rate == 0.0),
                "sac_all_seed_strict_failure_on_dp_strict_feasible": float(
                    dp_strict_feasible and sac_strict_success_rate == 0.0
                ),
                "sac_return_success_gap_to_threshold": sac_mean_return - reliability.success_return_threshold,
            }
        )
    return joined


def summarize_report(
    params: PendulumDPParams,
    reliability: ReliabilityConfig,
    dp_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]] | None,
    solve_seconds: float,
    sac_grid_path: Path | None,
) -> dict[str, Any]:
    dp_return_success = np.asarray([row["dp_policy_return_success"] for row in dp_rows], dtype=np.float64)
    dp_value_success = np.asarray([row["dp_value_return_success"] for row in dp_rows], dtype=np.float64)
    dp_strict_success = np.asarray([row["dp_policy_strict_success"] for row in dp_rows], dtype=np.float64)
    dp_policy_returns = np.asarray([row["dp_policy_return"] for row in dp_rows], dtype=np.float64)
    dp_values = np.asarray([row["dp_value"] for row in dp_rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "params": asdict(params),
        "success_return_threshold": reliability.success_return_threshold,
        "solve_seconds": solve_seconds,
        "sac_grid_path": str(sac_grid_path) if sac_grid_path else None,
        "num_eval_cells": len(dp_rows),
        "dp_value_return_success_cell_fraction": float(np.mean(dp_value_success)),
        "dp_policy_return_success_cell_fraction": float(np.mean(dp_return_success)),
        "dp_policy_strict_success_cell_fraction": float(np.mean(dp_strict_success)),
        "dp_policy_return_mean": float(np.mean(dp_policy_returns)),
        "dp_policy_return_min": float(np.min(dp_policy_returns)),
        "dp_policy_return_p05": float(np.quantile(dp_policy_returns, 0.05)),
        "dp_policy_return_p50": float(np.quantile(dp_policy_returns, 0.50)),
        "dp_policy_return_p95": float(np.quantile(dp_policy_returns, 0.95)),
        "dp_value_minus_rollout_return_mean": float(np.mean(dp_values - dp_policy_returns)),
        "regions": region_summaries(dp_rows, comparison_rows),
    }
    if comparison_rows is not None:
        sac_return_success = np.asarray([row["sac_return_success_rate"] for row in comparison_rows], dtype=np.float64)
        sac_strict_success = np.asarray([row["sac_strict_success_rate"] for row in comparison_rows], dtype=np.float64)
        regret = np.asarray([row["sac_mean_regret_to_dp_policy"] for row in comparison_rows], dtype=np.float64)
        feasible = np.asarray([row["dp_policy_return_success"] for row in comparison_rows], dtype=bool)
        strict_feasible = np.asarray([row["dp_policy_strict_success"] for row in comparison_rows], dtype=bool)
        sac_failure_on_feasible = np.asarray(
            [row["sac_failure_rate_on_dp_feasible"] for row in comparison_rows],
            dtype=np.float64,
        )
        sac_strict_failure_on_strict_feasible = np.asarray(
            [row["sac_strict_failure_rate_on_dp_strict_feasible"] for row in comparison_rows],
            dtype=np.float64,
        )
        summary["sac_comparison"] = {
            "sac_cell_mean_return_success_rate": float(np.mean(sac_return_success)),
            "sac_cell_mean_strict_success_rate": float(np.mean(sac_strict_success)),
            "sac_mean_regret_to_dp_policy": float(np.mean(regret)),
            "sac_median_regret_to_dp_policy": float(np.median(regret)),
            "sac_p95_regret_to_dp_policy": float(np.quantile(regret, 0.95)),
            "dp_feasible_cells": int(np.sum(feasible)),
            "dp_strict_feasible_cells": int(np.sum(strict_feasible)),
            "sac_failure_rate_among_dp_feasible_cells": float(np.mean(sac_failure_on_feasible[feasible]))
            if np.any(feasible)
            else None,
            "sac_strict_failure_rate_among_dp_strict_feasible_cells": float(
                np.mean(sac_strict_failure_on_strict_feasible[strict_feasible])
            )
            if np.any(strict_feasible)
            else None,
            "sac_all_seed_failure_fraction_among_dp_feasible_cells": float(
                np.mean([row["sac_all_seed_failure_on_dp_feasible"] for row in comparison_rows if row["dp_policy_return_success"]])
            )
            if np.any(feasible)
            else None,
            "sac_all_seed_strict_failure_fraction_among_dp_strict_feasible_cells": float(
                np.mean(
                    [
                        row["sac_all_seed_strict_failure_on_dp_strict_feasible"]
                        for row in comparison_rows
                        if row["dp_policy_strict_success"]
                    ]
                )
            )
            if np.any(strict_feasible)
            else None,
        }
    return summary


def region_summaries(
    dp_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    rows = comparison_rows if comparison_rows is not None else dp_rows
    regions = {
        "all_reset_support": lambda row: True,
        "near_down_abs_theta_ge_150": lambda row: abs(float(row["theta_degrees"])) >= 150.0,
        "near_down_abs_theta_ge_150_abs_vel_le_0.5": lambda row: abs(float(row["theta_degrees"])) >= 150.0
        and abs(float(row["theta_dot"])) <= 0.5,
        "mid_angles_abs_theta_60_to_120": lambda row: 60.0 <= abs(float(row["theta_degrees"])) <= 120.0,
    }
    out: dict[str, Any] = {}
    for name, predicate in regions.items():
        selected = [row for row in rows if predicate(row)]
        if not selected:
            continue
        entry: dict[str, Any] = {
            "cells": len(selected),
            "dp_policy_return_success_rate": float(np.mean([row["dp_policy_return_success"] for row in selected])),
            "dp_policy_strict_success_rate": float(np.mean([row["dp_policy_strict_success"] for row in selected])),
            "dp_policy_return_mean": float(np.mean([row["dp_policy_return"] for row in selected])),
        }
        if comparison_rows is not None:
            feasible = [row for row in selected if row["dp_policy_return_success"]]
            strict_feasible = [row for row in selected if row["dp_policy_strict_success"]]
            entry.update(
                {
                    "sac_return_success_rate": float(np.mean([row["sac_return_success_rate"] for row in selected])),
                    "sac_strict_success_rate": float(np.mean([row["sac_strict_success_rate"] for row in selected])),
                    "sac_mean_regret_to_dp_policy": float(
                        np.mean([row["sac_mean_regret_to_dp_policy"] for row in selected])
                    ),
                    "sac_failure_rate_among_dp_feasible_cells": float(
                        np.mean([row["sac_failure_rate_on_dp_feasible"] for row in feasible])
                    )
                    if feasible
                    else None,
                    "sac_strict_failure_rate_among_dp_strict_feasible_cells": float(
                        np.mean([row["sac_strict_failure_rate_on_dp_strict_feasible"] for row in strict_feasible])
                    )
                    if strict_feasible
                    else None,
                    "dp_feasible_cells": len(feasible),
                    "dp_strict_feasible_cells": len(strict_feasible),
                }
            )
        out[name] = entry
    return out


def write_report_figures(
    output_dir: Path,
    dp_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]] | None,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    success_return_threshold: float,
) -> list[Path]:
    figures = [
        plot_heatmap(
            matrix_for(dp_rows, "dp_policy_return", theta_values, velocity_values),
            theta_values,
            velocity_values,
            "DP Greedy Policy Return",
            output_dir / "dp_policy_return_map.png",
        ),
        plot_heatmap(
            matrix_for(dp_rows, "dp_policy_return_success", theta_values, velocity_values),
            theta_values,
            velocity_values,
            "DP Return Success",
            output_dir / "dp_return_success_map.png",
            vmin=0.0,
            vmax=1.0,
        ),
        plot_heatmap(
            matrix_for(dp_rows, "dp_policy_strict_success", theta_values, velocity_values),
            theta_values,
            velocity_values,
            "DP Strict Success",
            output_dir / "dp_strict_success_map.png",
            vmin=0.0,
            vmax=1.0,
        ),
    ]
    if comparison_rows is not None:
        figures.extend(
            [
                plot_heatmap(
                    matrix_for(comparison_rows, "sac_mean_regret_to_dp_policy", theta_values, velocity_values),
                    theta_values,
                    velocity_values,
                    "SAC Mean Regret To DP Policy",
                    output_dir / "sac_regret_to_dp_map.png",
                    cmap="magma",
                ),
                plot_heatmap(
                    matrix_for(comparison_rows, "sac_failure_rate_on_dp_feasible", theta_values, velocity_values),
                    theta_values,
                    velocity_values,
                    "SAC Failure Rate On DP-Feasible Starts",
                    output_dir / "sac_failure_on_dp_feasible_map.png",
                    vmin=0.0,
                    vmax=1.0,
                ),
                plot_heatmap(
                    matrix_for(
                        comparison_rows,
                        "sac_strict_failure_rate_on_dp_strict_feasible",
                        theta_values,
                        velocity_values,
                    ),
                    theta_values,
                    velocity_values,
                    "SAC Strict Failure Rate On DP-Strict-Feasible Starts",
                    output_dir / "sac_strict_failure_on_dp_strict_feasible_map.png",
                    vmin=0.0,
                    vmax=1.0,
                ),
                plot_scatter(
                    comparison_rows,
                    output_dir / "sac_vs_dp_return_scatter.png",
                    success_return_threshold=success_return_threshold,
                ),
            ]
        )
    return figures


def matrix_for(
    rows: list[dict[str, Any]],
    key: str,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
) -> np.ndarray:
    theta_index = {round(float(value), 12): idx for idx, value in enumerate(theta_values)}
    velocity_index = {round(float(value), 12): idx for idx, value in enumerate(velocity_values)}
    matrix = np.full((len(velocity_values), len(theta_values)), np.nan, dtype=np.float64)
    for row in rows:
        matrix[velocity_index[round(float(row["theta_dot"]), 12)], theta_index[round(float(row["theta"]), 12)]] = float(
            row[key]
        )
    return matrix


def plot_heatmap(
    values: np.ndarray,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [
        float(np.degrees(theta_values[0])),
        float(np.degrees(theta_values[-1])),
        float(velocity_values[0]),
        float(velocity_values[-1]),
    ]
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("initial theta degrees")
    ax.set_ylabel("initial angular velocity")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_scatter(rows: list[dict[str, Any]], path: Path, success_return_threshold: float) -> Path:
    dp_return = np.asarray([row["dp_policy_return"] for row in rows], dtype=np.float64)
    sac_return = np.asarray([row["sac_mean_return"] for row in rows], dtype=np.float64)
    success = np.asarray([row["dp_policy_return_success"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    scatter = ax.scatter(dp_return, sac_return, c=success, s=12, alpha=0.75, cmap="coolwarm", vmin=0.0, vmax=1.0)
    lower = min(float(np.min(dp_return)), float(np.min(sac_return)))
    upper = max(float(np.max(dp_return)), float(np.max(sac_return)))
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, alpha=0.6)
    ax.axhline(success_return_threshold, color="gray", linewidth=1.0, linestyle="--")
    ax.axvline(success_return_threshold, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xlabel("DP greedy-policy return")
    ax.set_ylabel("SAC mean return across seeds")
    ax.set_title("SAC checkpoint grid vs DP calibration")
    fig.colorbar(scatter, ax=ax, label="DP return success")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_index(output_dir: Path, figures: list[Path], summary: dict[str, Any]) -> None:
    with (output_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Pendulum DP Calibration</title>\n")
        f.write("<h1>Pendulum DP Calibration</h1>\n")
        f.write("<pre>" + html.escape(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)) + "</pre>\n")
        for figure in figures:
            name = html.escape(figure.name, quote=True)
            f.write(f"<section><h2>{name}</h2><img src='{name}' style='max-width:100%;height:auto'></section>\n")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
