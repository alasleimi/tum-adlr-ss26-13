from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.checkpoints import expand_run_dirs, load_agent_from_run
from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.envs import UprightDetector
from last_nine_rl.sac import SACAgent


def main() -> None:
    args = parse_args()
    run_dirs = expand_run_dirs([Path(path) for path in args.runs], require_checkpoint=True)
    if not run_dirs:
        raise SystemExit("No run directories with config.json and checkpoints/final.pt were found.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_grid(
        run_dirs=run_dirs,
        output_dir=output_dir,
        theta_bins=args.theta_bins,
        velocity_bins=args.velocity_bins,
        velocity_limit=args.velocity_limit,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Pendulum checkpoint success over exact initial states.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parents containing checkpoints.")
    parser.add_argument("--out", required=True, help="Output directory for grid CSVs and heatmaps.")
    parser.add_argument("--theta-bins", type=int, default=61)
    parser.add_argument("--velocity-bins", type=int, default=41)
    parser.add_argument("--velocity-limit", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="final.pt")
    return parser.parse_args()


def evaluate_grid(
    run_dirs: list[Path],
    output_dir: Path,
    theta_bins: int,
    velocity_bins: int,
    velocity_limit: float,
    device: str | None,
    checkpoint: str,
) -> dict[str, Any]:
    loaded = [load_agent_from_run(run_dir, device=device, checkpoint=checkpoint) for run_dir in run_dirs]
    configs = [item[1] for item in loaded]
    agents = [item[0] for item in loaded]
    if any(not config.env.env_id.startswith("Pendulum") for config in configs):
        raise ValueError("pendulum_grid only supports Pendulum run directories.")

    reliability = configs[0].reliability
    theta_values = np.linspace(-math.pi, math.pi, theta_bins, endpoint=False, dtype=np.float64)
    velocity_values = np.linspace(-velocity_limit, velocity_limit, velocity_bins, dtype=np.float64)
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )

    rollout_rows: list[dict[str, Any]] = []
    grid_theta = np.tile(theta_values, len(velocity_values))
    grid_velocity = np.repeat(velocity_values, len(theta_values))
    horizon = pendulum_horizon(configs[0].env.max_episode_steps)
    for (agent, config, _payload), run_dir in zip(loaded, run_dirs):
        run_rollouts = rollout_pendulum_grid_vectorized(
            agent=agent,
            theta_values=grid_theta,
            theta_dot_values=grid_velocity,
            detector=detector,
            reliability=config.reliability,
            horizon=horizon,
        )
        for theta, velocity, rollout in zip(grid_theta, grid_velocity, run_rollouts):
            rollout_rows.append(
                {
                    "run_dir": str(run_dir),
                    "actual_seed": int(config.seed),
                    "theta": float(theta),
                    "theta_degrees": float(np.degrees(theta)),
                    "theta_dot": float(velocity),
                    **rollout,
                }
            )

    grid_rows = summarize_cells(rollout_rows, theta_values, velocity_values)
    write_csv(output_dir / "pendulum_grid_rollouts.csv", rollout_rows)
    write_csv(output_dir / "pendulum_grid_summary.csv", grid_rows)
    figures = write_heatmaps(output_dir, grid_rows, theta_values, velocity_values)
    summary = grid_summary(run_dirs, grid_rows, theta_bins, velocity_bins, velocity_limit)
    (output_dir / "pendulum_grid_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_index(output_dir, figures, summary)
    return {"summary": summary, "grid_rows": grid_rows, "rollout_rows": rollout_rows}


def pendulum_horizon(configured_max_steps: int | None) -> int:
    return int(configured_max_steps or 200)


def rollout_pendulum_grid_vectorized(
    agent: SACAgent,
    theta_values: np.ndarray,
    theta_dot_values: np.ndarray,
    detector: UprightDetector,
    reliability: ReliabilityConfig,
    horizon: int,
) -> list[dict[str, float]]:
    theta = np.asarray(theta_values, dtype=np.float64).copy()
    theta_dot = np.asarray(theta_dot_values, dtype=np.float64).copy()
    episode_return = np.zeros_like(theta, dtype=np.float64)
    min_reward = np.full_like(theta, np.inf, dtype=np.float64)
    near_count = np.zeros_like(theta, dtype=np.int64)
    current_not_near_streak = np.zeros_like(theta, dtype=np.int64)
    longest_not_near_streak = np.zeros_like(theta, dtype=np.int64)

    for _step in range(horizon):
        obs = pendulum_obs_batch(theta, theta_dot)
        actions = agent.act_batch(obs, deterministic=True).reshape(-1)
        theta, theta_dot, reward = pendulum_step_batch(theta, theta_dot, actions)
        episode_return += reward
        min_reward = np.minimum(min_reward, reward)

        next_obs = pendulum_obs_batch(theta, theta_dot)
        near = detector.near_upright(next_obs)
        near_count += near.astype(np.int64)
        current_not_near_streak = np.where(near, 0, current_not_near_streak + 1)
        longest_not_near_streak = np.maximum(longest_not_near_streak, current_not_near_streak)

    near_fraction = near_count / max(horizon, 1)
    return_success = episode_return >= reliability.success_return_threshold
    stability_success = near_fraction >= reliability.success_near_upright_fraction_threshold
    streak_success = longest_not_near_streak <= reliability.success_max_not_near_upright_streak
    strict_success = return_success & stability_success & streak_success
    return [
        {
            "return": float(episode_return[idx]),
            "length": float(horizon),
            "near_upright_fraction": float(near_fraction[idx]),
            "min_step_reward": float(min_reward[idx]),
            "not_near_upright_streak": float(longest_not_near_streak[idx]),
            "return_success": float(return_success[idx]),
            "stability_success": float(stability_success[idx]),
            "streak_success": float(streak_success[idx]),
            "strict_success": float(strict_success[idx]),
        }
        for idx in range(len(theta_values))
    ]


def pendulum_obs_batch(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)


def pendulum_step_batch(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    action: np.ndarray,
    g: float = 10.0,
    m: float = 1.0,
    length: float = 1.0,
    dt: float = 0.05,
    max_speed: float = 8.0,
    max_torque: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torque = np.clip(np.asarray(action, dtype=np.float64), -max_torque, max_torque)
    costs = angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * torque**2
    new_theta_dot = theta_dot + (3.0 * g / (2.0 * length) * np.sin(theta) + 3.0 / (m * length**2) * torque) * dt
    new_theta_dot = np.clip(new_theta_dot, -max_speed, max_speed)
    new_theta = theta + new_theta_dot * dt
    return new_theta, new_theta_dot, -costs


def angle_normalize(theta: np.ndarray) -> np.ndarray:
    return ((theta + np.pi) % (2.0 * np.pi)) - np.pi


def rollout_from_state(
    agent: SACAgent,
    env: gym.Env,
    theta: float,
    theta_dot: float,
    detector: UprightDetector,
    reliability: ReliabilityConfig,
) -> dict[str, float]:
    obs = reset_pendulum_to_state(env, theta, theta_dot)
    done = False
    episode_return = 0.0
    length = 0
    near_count = 0
    min_reward = math.inf
    current_not_near_streak = 0
    longest_not_near_streak = 0

    while not done:
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        episode_return += float(reward)
        length += 1
        min_reward = min(min_reward, float(reward))
        near = bool(detector.near_upright(np.asarray(obs))[0])
        near_count += int(near)
        if near:
            current_not_near_streak = 0
        else:
            current_not_near_streak += 1
            longest_not_near_streak = max(longest_not_near_streak, current_not_near_streak)
        done = bool(terminated or truncated)

    near_fraction = near_count / max(length, 1)
    return_success = episode_return >= reliability.success_return_threshold
    stability_success = near_fraction >= reliability.success_near_upright_fraction_threshold
    streak_success = longest_not_near_streak <= reliability.success_max_not_near_upright_streak
    return {
        "return": float(episode_return),
        "length": float(length),
        "near_upright_fraction": float(near_fraction),
        "min_step_reward": float(min_reward),
        "not_near_upright_streak": float(longest_not_near_streak),
        "return_success": float(return_success),
        "stability_success": float(stability_success),
        "streak_success": float(streak_success),
        "strict_success": float(return_success and stability_success and streak_success),
    }


def reset_pendulum_to_state(env: gym.Env, theta: float, theta_dot: float) -> np.ndarray:
    env.reset(seed=0)
    unwrapped = env.unwrapped
    unwrapped.state = np.asarray([theta, theta_dot], dtype=np.float64)
    unwrapped.last_u = None
    return np.asarray(unwrapped._get_obs(), dtype=np.float32)


def summarize_cells(
    rollout_rows: list[dict[str, Any]],
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rollout_rows:
        grouped.setdefault((float(row["theta"]), float(row["theta_dot"])), []).append(row)

    rows: list[dict[str, Any]] = []
    for velocity in velocity_values:
        for theta in theta_values:
            cell_rows = grouped[(float(theta), float(velocity))]
            returns = np.asarray([row["return"] for row in cell_rows], dtype=np.float64)
            rows.append(
                {
                    "theta": float(theta),
                    "theta_degrees": float(np.degrees(theta)),
                    "theta_dot": float(velocity),
                    "num_training_seeds": len(cell_rows),
                    "mean_return": float(np.mean(returns)),
                    "median_return": float(np.median(returns)),
                    "worst_return": float(np.min(returns)),
                    "best_return": float(np.max(returns)),
                    "return_success_rate": float(np.mean([row["return_success"] for row in cell_rows])),
                    "strict_success_rate": float(np.mean([row["strict_success"] for row in cell_rows])),
                    "mean_near_upright_fraction": float(
                        np.mean([row["near_upright_fraction"] for row in cell_rows])
                    ),
                    "max_not_near_upright_streak": float(
                        np.max([row["not_near_upright_streak"] for row in cell_rows])
                    ),
                }
            )
    return rows


def write_heatmaps(
    output_dir: Path,
    grid_rows: list[dict[str, Any]],
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
) -> list[Path]:
    specs = [
        ("return_success_rate", "Return Success Rate", 0.0, 1.0, "return_success_rate_map.png"),
        ("strict_success_rate", "Strict Success Rate", 0.0, 1.0, "strict_success_rate_map.png"),
        ("mean_return", "Mean Return", None, None, "mean_return_map.png"),
        ("worst_return", "Worst Return", None, None, "worst_return_map.png"),
        ("mean_near_upright_fraction", "Near-Upright Fraction", 0.0, 1.0, "near_upright_fraction_map.png"),
    ]
    return [
        plot_heatmap(
            values=matrix_for(grid_rows, key, theta_values, velocity_values),
            theta_values=theta_values,
            velocity_values=velocity_values,
            title=title,
            path=output_dir / filename,
            vmin=vmin,
            vmax=vmax,
        )
        for key, title, vmin, vmax, filename in specs
    ]


def matrix_for(
    grid_rows: list[dict[str, Any]],
    key: str,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
) -> np.ndarray:
    theta_index = {float(value): idx for idx, value in enumerate(theta_values)}
    velocity_index = {float(value): idx for idx, value in enumerate(velocity_values)}
    matrix = np.full((len(velocity_values), len(theta_values)), np.nan, dtype=np.float64)
    for row in grid_rows:
        matrix[velocity_index[float(row["theta_dot"])], theta_index[float(row["theta"])]] = float(row[key])
    return matrix


def plot_heatmap(
    values: np.ndarray,
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [
        float(np.degrees(theta_values[0])),
        float(np.degrees(theta_values[-1])),
        float(velocity_values[0]),
        float(velocity_values[-1]),
    ]
    image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest", extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("initial theta degrees")
    ax.set_ylabel("initial angular velocity")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def grid_summary(
    run_dirs: list[Path],
    grid_rows: list[dict[str, Any]],
    theta_bins: int,
    velocity_bins: int,
    velocity_limit: float,
) -> dict[str, Any]:
    strict = np.asarray([row["strict_success_rate"] for row in grid_rows], dtype=np.float64)
    ret = np.asarray([row["return_success_rate"] for row in grid_rows], dtype=np.float64)
    worst_cells = sorted(grid_rows, key=lambda row: (row["strict_success_rate"], row["mean_return"]))[:20]
    return {
        "num_training_seeds": len(run_dirs),
        "theta_bins": theta_bins,
        "velocity_bins": velocity_bins,
        "velocity_limit": velocity_limit,
        "num_initial_condition_cells": len(grid_rows),
        "cell_mean_return_success_rate": float(np.mean(ret)),
        "cell_mean_strict_success_rate": float(np.mean(strict)),
        "cell_fraction_all_training_seeds_return_success": float(np.mean(ret >= 1.0)),
        "cell_fraction_all_training_seeds_strict_success": float(np.mean(strict >= 1.0)),
        "cell_fraction_any_training_seed_strict_success": float(np.mean(strict > 0.0)),
        "worst_cells": worst_cells,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_index(output_dir: Path, figures: list[Path], summary: dict[str, Any]) -> None:
    with (output_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Pendulum Initial-Condition Grid</title>\n")
        f.write("<h1>Pendulum Initial-Condition Grid</h1>\n")
        f.write("<pre>" + html.escape(json.dumps(summary, indent=2, sort_keys=True)) + "</pre>\n")
        for figure in figures:
            name = html.escape(figure.name, quote=True)
            f.write(f"<section><h2>{name}</h2><img src='{name}' style='max-width:100%;height:auto'></section>\n")


if __name__ == "__main__":
    main()
