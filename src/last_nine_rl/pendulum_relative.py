from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.envs import UprightDetector
from last_nine_rl.pendulum_grid import reset_pendulum_to_state
from last_nine_rl.reference import PendulumEnergySwingupController


CRITERIA: tuple[tuple[str, str], ...] = (
    ("fixed_return_success", "Diagnostic: return >= threshold"),
    ("task_success", "Task-only stability"),
    ("strict_success", "Diagnostic: threshold + task"),
    ("beats_dp_return", "SAC >= DP"),
    ("near_dp_return_eps", "SAC >= DP - eps"),
    ("beats_controller_return", "SAC >= controller"),
    ("near_controller_return_eps", "SAC >= controller - eps"),
    ("beats_best_known_return", "SAC >= max(DP, controller)"),
    ("near_best_known_return_eps", "SAC >= max(DP, controller) - eps"),
)


REGIONS: tuple[tuple[str, str], ...] = (
    ("all_reset_support", "All reset support"),
    ("near_down_abs_theta_ge_150", "|theta| >= 150 deg"),
    ("near_down_abs_theta_ge_150_abs_vel_le_0.5", "|theta| >= 150 deg and |theta_dot| <= 0.5"),
    ("mid_angles_abs_theta_60_to_120", "60 deg <= |theta| <= 120 deg"),
)


def main() -> None:
    args = parse_args()
    reliability = ReliabilityConfig(success_return_threshold=args.success_return_threshold)
    result = run_relative_report(
        sac_rollouts_path=Path(args.sac_rollouts),
        dp_grid_path=Path(args.dp_grid),
        controller_grid_path=Path(args.controller_grid),
        output_dir=Path(args.out),
        condition_label=args.condition_label,
        epsilon_return=args.epsilon_return,
        reliability=reliability,
    )
    print(json.dumps(result["summary"], allow_nan=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pendulum DP/controller-relative success analysis.")
    parser.add_argument("--sac-rollouts", required=True, help="SAC pendulum_grid_rollouts.csv.")
    parser.add_argument("--dp-grid", required=True, help="DP pendulum_dp_grid.csv on the same initial-state grid.")
    parser.add_argument("--controller-grid", required=True, help="Energy-controller grid CSV; created if missing.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition-label", required=True)
    parser.add_argument("--epsilon-return", type=float, default=5.0)
    parser.add_argument("--success-return-threshold", type=float, default=-200.0)
    return parser.parse_args()


def run_relative_report(
    sac_rollouts_path: Path,
    dp_grid_path: Path,
    controller_grid_path: Path,
    output_dir: Path,
    condition_label: str,
    epsilon_return: float,
    reliability: ReliabilityConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dp_rows = read_csv(dp_grid_path)
    theta_values, velocity_values = grid_values(dp_rows)
    if not controller_grid_path.is_file():
        controller_grid_path.parent.mkdir(parents=True, exist_ok=True)
        controller_rows = evaluate_controller_grid(theta_values, velocity_values, reliability)
        write_csv(controller_grid_path, controller_rows)
    else:
        controller_rows = read_csv(controller_grid_path)

    sac_rows = read_csv(sac_rollouts_path)
    enriched_rows = enrich_rollouts(
        sac_rows=sac_rows,
        dp_rows=dp_rows,
        controller_rows=controller_rows,
        epsilon_return=epsilon_return,
        reliability=reliability,
    )
    cell_rows = summarize_cells(enriched_rows, theta_values, velocity_values)
    summary = summarize_condition(
        enriched_rows=enriched_rows,
        cell_rows=cell_rows,
        controller_rows=controller_rows,
        condition_label=condition_label,
        epsilon_return=epsilon_return,
        reliability=reliability,
        sac_rollouts_path=sac_rollouts_path,
        dp_grid_path=dp_grid_path,
        controller_grid_path=controller_grid_path,
    )

    write_csv(output_dir / "relative_rollouts.csv", enriched_rows)
    write_csv(output_dir / "relative_cell_summary.csv", cell_rows)
    write_csv(output_dir / "relative_criterion_summary.csv", criterion_summary_rows(summary))
    write_csv(output_dir / "relative_region_summary.csv", region_summary_rows(summary))
    (output_dir / "relative_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figures = write_figures(output_dir, condition_label, cell_rows, theta_values, velocity_values, summary)
    summary["figures"] = [str(path.name) for path in figures]
    (output_dir / "relative_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "rollouts": enriched_rows, "cells": cell_rows}


def evaluate_controller_grid(
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    reliability: ReliabilityConfig,
) -> list[dict[str, Any]]:
    controller = PendulumEnergySwingupController()
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    rows: list[dict[str, Any]] = []
    env = gym.make("Pendulum-v1")
    try:
        for velocity in velocity_values:
            for theta in theta_values:
                rollout = rollout_controller_from_state(
                    controller=controller,
                    env=env,
                    theta=float(theta),
                    theta_dot=float(velocity),
                    detector=detector,
                    reliability=reliability,
                )
                rows.append(
                    {
                        "theta": float(theta),
                        "theta_degrees": float(np.degrees(theta)),
                        "theta_dot": float(velocity),
                        **rollout,
                    }
                )
    finally:
        env.close()
    return rows


def rollout_controller_from_state(
    controller: PendulumEnergySwingupController,
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
    current_not_near_streak = 0
    longest_not_near_streak = 0

    while not done:
        action = controller.act(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        episode_return += float(reward)
        length += 1
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
        "controller_return": float(episode_return),
        "controller_near_upright_fraction": float(near_fraction),
        "controller_not_near_upright_streak": float(longest_not_near_streak),
        "controller_return_success": float(return_success),
        "controller_task_success": float(stability_success and streak_success),
        "controller_strict_success": float(return_success and stability_success and streak_success),
    }


def enrich_rollouts(
    sac_rows: list[dict[str, Any]],
    dp_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    epsilon_return: float,
    reliability: ReliabilityConfig,
) -> list[dict[str, Any]]:
    dp_by_state = rows_by_state(dp_rows)
    controller_by_state = rows_by_state(controller_rows)
    enriched: list[dict[str, Any]] = []

    for row in sac_rows:
        key = state_key(row)
        dp = dp_by_state[key]
        controller = controller_by_state[key]
        sac_return = float(row["return"])
        dp_return = float(dp["dp_policy_return"])
        controller_return = float(controller["controller_return"])
        best_known_return = max(dp_return, controller_return)
        task_success = float(float(row["stability_success"]) > 0.5 and float(row["streak_success"]) > 0.5)
        enriched.append(
            {
                **row,
                "return": sac_return,
                "task_success": task_success,
                "fixed_return_success": float(sac_return >= reliability.success_return_threshold),
                "strict_success": float(row["strict_success"]),
                "dp_policy_return": dp_return,
                "dp_policy_return_success": float(dp["dp_policy_return_success"]),
                "dp_policy_strict_success": float(dp["dp_policy_strict_success"]),
                "controller_return": controller_return,
                "controller_return_success": float(controller["controller_return_success"]),
                "controller_task_success": float(controller["controller_task_success"]),
                "controller_strict_success": float(controller["controller_strict_success"]),
                "best_known_return": best_known_return,
                "beats_dp_return": float(sac_return >= dp_return),
                "near_dp_return_eps": float(sac_return >= dp_return - epsilon_return),
                "beats_controller_return": float(sac_return >= controller_return),
                "near_controller_return_eps": float(sac_return >= controller_return - epsilon_return),
                "beats_best_known_return": float(sac_return >= best_known_return),
                "near_best_known_return_eps": float(sac_return >= best_known_return - epsilon_return),
                "regret_to_dp": float(dp_return - sac_return),
                "regret_to_controller": float(controller_return - sac_return),
                "regret_to_best_known": float(best_known_return - sac_return),
            }
        )
    return enriched


def summarize_cells(
    enriched_rows: list[dict[str, Any]],
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in enriched_rows:
        grouped.setdefault(state_key(row), []).append(row)

    rows: list[dict[str, Any]] = []
    for velocity in velocity_values:
        for theta in theta_values:
            cell_rows = grouped[(round(float(theta), 12), round(float(velocity), 12))]
            returns = np.asarray([float(row["return"]) for row in cell_rows], dtype=np.float64)
            first = cell_rows[0]
            out: dict[str, Any] = {
                "theta": float(theta),
                "theta_degrees": float(np.degrees(theta)),
                "theta_dot": float(velocity),
                "num_training_seeds": len(cell_rows),
                "mean_return": float(np.mean(returns)),
                "dp_policy_return": float(first["dp_policy_return"]),
                "controller_return": float(first["controller_return"]),
                "best_known_return": float(first["best_known_return"]),
                "mean_regret_to_dp": float(np.mean([float(row["regret_to_dp"]) for row in cell_rows])),
                "mean_regret_to_controller": float(np.mean([float(row["regret_to_controller"]) for row in cell_rows])),
                "mean_regret_to_best_known": float(np.mean([float(row["regret_to_best_known"]) for row in cell_rows])),
            }
            for key, _label in CRITERIA:
                out[f"{key}_rate"] = float(np.mean([float(row[key]) for row in cell_rows]))
            rows.append(out)
    return rows


def summarize_condition(
    enriched_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    condition_label: str,
    epsilon_return: float,
    reliability: ReliabilityConfig,
    sac_rollouts_path: Path,
    dp_grid_path: Path,
    controller_grid_path: Path,
) -> dict[str, Any]:
    actual_seeds = sorted({int(float(row["actual_seed"])) for row in enriched_rows})
    summary: dict[str, Any] = {
        "condition_label": condition_label,
        "epsilon_return": epsilon_return,
        "success_return_threshold": reliability.success_return_threshold,
        "num_training_seeds": len(actual_seeds),
        "actual_seeds": actual_seeds,
        "num_initial_condition_cells": len(cell_rows),
        "num_rollouts": len(enriched_rows),
        "sac_rollouts_path": str(sac_rollouts_path),
        "dp_grid_path": str(dp_grid_path),
        "controller_grid_path": str(controller_grid_path),
        "criteria": {},
        "regions": {},
        "baseline": summarize_baselines(controller_rows),
    }

    for key, label in CRITERIA:
        values = np.asarray([float(row[key]) for row in enriched_rows], dtype=np.float64)
        seed_rates = [
            float(np.mean([float(row[key]) for row in enriched_rows if int(float(row["actual_seed"])) == seed]))
            for seed in actual_seeds
        ]
        pooled = wilson_interval(int(np.sum(values)), len(values))
        seed_stats = mean_t_interval(seed_rates)
        cell_rates = np.asarray([float(row[f"{key}_rate"]) for row in cell_rows], dtype=np.float64)
        summary["criteria"][key] = {
            "label": label,
            "rate": float(np.mean(values)),
            "successes": int(np.sum(values)),
            "total": int(len(values)),
            "wilson95_low": pooled["low"],
            "wilson95_high": pooled["high"],
            "seed_mean": seed_stats["mean"],
            "seed_ci95_low": seed_stats["low"],
            "seed_ci95_high": seed_stats["high"],
            "seed_ci95_half_width": seed_stats["half_width"],
            "seed_rates": seed_rates,
            "cell_mean_rate": float(np.mean(cell_rates)),
            "cell_all_seed_success_fraction": float(np.mean(cell_rates >= 1.0)),
            "cell_any_seed_success_fraction": float(np.mean(cell_rates > 0.0)),
        }

    for region_key, region_label in REGIONS:
        selected_rollouts = [row for row in enriched_rows if in_region(row, region_key)]
        selected_cells = [row for row in cell_rows if in_region(row, region_key)]
        region: dict[str, Any] = {"label": region_label, "cells": len(selected_cells), "rollouts": len(selected_rollouts)}
        for key, _label in CRITERIA:
            values = np.asarray([float(row[key]) for row in selected_rollouts], dtype=np.float64)
            interval = wilson_interval(int(np.sum(values)), len(values))
            region[key] = {
                "rate": float(np.mean(values)) if len(values) else math.nan,
                "wilson95_low": interval["low"],
                "wilson95_high": interval["high"],
            }
        summary["regions"][region_key] = region
    return summary


def summarize_baselines(controller_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "controller_return_success_cell_fraction": float(
            np.mean([float(row["controller_return_success"]) for row in controller_rows])
        ),
        "controller_task_success_cell_fraction": float(
            np.mean([float(row["controller_task_success"]) for row in controller_rows])
        ),
        "controller_strict_success_cell_fraction": float(
            np.mean([float(row["controller_strict_success"]) for row in controller_rows])
        ),
        "controller_mean_return": float(np.mean([float(row["controller_return"]) for row in controller_rows])),
    }


def write_figures(
    output_dir: Path,
    condition_label: str,
    cell_rows: list[dict[str, Any]],
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    summary: dict[str, Any],
) -> list[Path]:
    figures: list[Path] = []
    figures.append(write_criterion_bar_plot(output_dir, condition_label, summary))
    for key, label in CRITERIA:
        figures.append(
            plot_heatmap(
                values=matrix_for(cell_rows, f"{key}_rate", theta_values, velocity_values),
                theta_values=theta_values,
                velocity_values=velocity_values,
                title=f"{condition_label}: {label}",
                path=output_dir / f"{key}_map.png",
                vmin=0.0,
                vmax=1.0,
            )
        )
    for key, title, cmap in (
        ("mean_regret_to_dp", f"{condition_label}: regret to DP", "magma"),
        ("mean_regret_to_controller", f"{condition_label}: regret to controller", "magma"),
        ("mean_regret_to_best_known", f"{condition_label}: regret to best known", "magma"),
    ):
        figures.append(
            plot_heatmap(
                values=matrix_for(cell_rows, key, theta_values, velocity_values),
                theta_values=theta_values,
                velocity_values=velocity_values,
                title=title,
                path=output_dir / f"{key}_map.png",
                cmap=cmap,
            )
        )
    return figures


def write_criterion_bar_plot(output_dir: Path, condition_label: str, summary: dict[str, Any]) -> Path:
    labels = [label for _key, label in CRITERIA]
    means = [summary["criteria"][key]["seed_mean"] for key, _label in CRITERIA]
    low = [means[i] - summary["criteria"][key]["seed_ci95_low"] for i, (key, _label) in enumerate(CRITERIA)]
    high = [summary["criteria"][key]["seed_ci95_high"] - means[i] for i, (key, _label) in enumerate(CRITERIA)]

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    x = np.arange(len(labels))
    ax.bar(x, means, color="#4c78a8", alpha=0.86)
    ax.errorbar(x, means, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate, seed mean with 95% t interval")
    ax.set_title(f"{condition_label}: task/reference rates and diagnostics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "criterion_success_rates_ci.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


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


def criterion_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, _label in CRITERIA:
        item = summary["criteria"][key]
        rows.append({"criterion": key, **{k: v for k, v in item.items() if k != "seed_rates"}})
    return rows


def region_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region_key, region in summary["regions"].items():
        base = {"region": region_key, "label": region["label"], "cells": region["cells"], "rollouts": region["rollouts"]}
        for criterion_key, _label in CRITERIA:
            item = region[criterion_key]
            rows.append({"criterion": criterion_key, **base, **item})
    return rows


def grid_values(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(sorted({float(row["theta"]) for row in rows}), dtype=np.float64)
    velocity = np.asarray(sorted({float(row["theta_dot"]) for row in rows}), dtype=np.float64)
    return theta, velocity


def rows_by_state(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {state_key(row): row for row in rows}


def state_key(row: dict[str, Any]) -> tuple[float, float]:
    return round(float(row["theta"]), 12), round(float(row["theta_dot"]), 12)


def in_region(row: dict[str, Any], region_key: str) -> bool:
    theta_deg = abs(float(row["theta_degrees"]))
    velocity = abs(float(row["theta_dot"]))
    if region_key == "all_reset_support":
        return True
    if region_key == "near_down_abs_theta_ge_150":
        return theta_deg >= 150.0
    if region_key == "near_down_abs_theta_ge_150_abs_vel_le_0.5":
        return theta_deg >= 150.0 and velocity <= 0.5
    if region_key == "mid_angles_abs_theta_60_to_120":
        return 60.0 <= theta_deg <= 120.0
    raise ValueError(f"Unknown region: {region_key}")


def mean_t_interval(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(x))
    if len(x) <= 1:
        return {"mean": mean, "low": mean, "high": mean, "half_width": 0.0}
    sem = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    half_width = t_critical_975(len(x) - 1) * sem
    return {"mean": mean, "low": mean - half_width, "high": mean + half_width, "half_width": half_width}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total <= 0:
        return {"rate": math.nan, "low": math.nan, "high": math.nan}
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return {"rate": phat, "low": max(0.0, center - half_width), "high": min(1.0, center + half_width)}


def t_critical_975(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 25:
        return table[max(k for k in table if k < df)]
    if df < 30:
        return table[25]
    return 1.96


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
