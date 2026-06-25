from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from last_nine_rl.checkpoints import expand_run_dirs, load_agent_from_run
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from last_nine_rl.sac import SACAgent


def main() -> None:
    args = parse_args()
    run_dirs = expand_run_dirs([Path(path) for path in args.runs], require_checkpoint=True)
    if not run_dirs:
        raise SystemExit("No run directories with config.json and checkpoints/final.pt were found.")

    result = analyze_reference_actions(
        run_dirs=run_dirs,
        output_dir=Path(args.out),
        policy=args.policy,
        theta_bins=args.theta_bins,
        velocity_bins=args.velocity_bins,
        velocity_limit=args.velocity_limit,
        device=args.device,
        checkpoint=args.checkpoint,
        dp_solution_path=Path(args.dp_solution) if args.dp_solution else None,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Pendulum reference actions with saved SAC/Simba critics.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parents containing checkpoints.")
    parser.add_argument("--out", required=True, help="Output directory for CSV and summary files.")
    parser.add_argument("--policy", choices=("controller", "dp", "best"), default="dp")
    parser.add_argument("--theta-bins", type=int, default=61)
    parser.add_argument("--velocity-bins", type=int, default=41)
    parser.add_argument("--velocity-limit", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--dp-solution", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def analyze_reference_actions(
    run_dirs: list[Path],
    output_dir: Path,
    policy: str,
    theta_bins: int,
    velocity_bins: int,
    velocity_limit: float,
    device: str | None,
    checkpoint: str,
    dp_solution_path: Path | None,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_values = np.linspace(-math.pi, math.pi, theta_bins, endpoint=False, dtype=np.float64)
    velocity_values = np.linspace(-velocity_limit, velocity_limit, velocity_bins, dtype=np.float64)
    grid_theta = np.tile(theta_values, len(velocity_values))
    grid_velocity = np.repeat(velocity_values, len(theta_values))
    observations = pendulum_obs_batch(grid_theta, grid_velocity)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        agent, config, _payload = load_agent_from_run(run_dir, device=device, checkpoint=checkpoint)
        if not config.env.env_id.startswith("Pendulum"):
            raise ValueError("reference_action_analysis only supports Pendulum run directories.")
        guidance = PendulumReferenceGuidance(
            policy=policy,
            dp_solution_path=dp_solution_path or config.sac.reference_guidance_dp_solution_path,
            horizon=int(config.env.max_episode_steps or 200),
        )
        actor_actions = agent.act_batch(observations, deterministic=True)
        reference_actions = guidance.act_batch(observations)
        q_actor, q_reference = score_action_values(
            agent=agent,
            observations=observations,
            actor_actions=actor_actions,
            reference_actions=reference_actions,
            batch_size=batch_size,
        )
        q_advantage = q_reference - q_actor
        action_abs_diff = np.abs(reference_actions.reshape(-1) - actor_actions.reshape(-1))
        for idx, (theta, velocity) in enumerate(zip(grid_theta, grid_velocity)):
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "actual_seed": int(config.seed),
                    "theta": float(theta),
                    "theta_degrees": float(np.degrees(theta)),
                    "theta_dot": float(velocity),
                    "policy": policy,
                    "actor_action": float(actor_actions[idx, 0]),
                    "reference_action": float(reference_actions[idx, 0]),
                    "action_abs_diff": float(action_abs_diff[idx]),
                    "q_actor": float(q_actor[idx]),
                    "q_reference": float(q_reference[idx]),
                    "q_reference_minus_actor": float(q_advantage[idx]),
                    "critic_prefers_reference": float(q_advantage[idx] > 0.0),
                }
            )

    summary = summarize(rows, theta_bins, velocity_bins, velocity_limit)
    write_csv(output_dir / "reference_action_scores.csv", rows)
    (output_dir / "reference_action_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reference_action_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def score_action_values(
    agent: SACAgent,
    observations: np.ndarray,
    actor_actions: np.ndarray,
    reference_actions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_actor_values: list[np.ndarray] = []
    q_reference_values: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        end = min(start + batch_size, len(observations))
        obs = torch.as_tensor(observations[start:end], dtype=torch.float32, device=agent.device)
        obs = agent._normalize_obs_tensor(obs)
        actor = torch.as_tensor(actor_actions[start:end], dtype=torch.float32, device=agent.device)
        reference = torch.as_tensor(reference_actions[start:end], dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            q_actor = torch.min(agent.q1(obs, actor), agent.q2(obs, actor)).view(-1)
            q_reference = torch.min(agent.q1(obs, reference), agent.q2(obs, reference)).view(-1)
        q_actor_values.append(q_actor.detach().cpu().numpy())
        q_reference_values.append(q_reference.detach().cpu().numpy())
    return np.concatenate(q_actor_values), np.concatenate(q_reference_values)


def pendulum_obs_batch(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)


def summarize(rows: list[dict[str, Any]], theta_bins: int, velocity_bins: int, velocity_limit: float) -> dict[str, Any]:
    seeds = sorted({int(row["actual_seed"]) for row in rows})
    q_advantages = np.asarray([float(row["q_reference_minus_actor"]) for row in rows], dtype=np.float64)
    action_diff = np.asarray([float(row["action_abs_diff"]) for row in rows], dtype=np.float64)
    seed_summaries = []
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["actual_seed"]) == seed]
        seed_adv = np.asarray([float(row["q_reference_minus_actor"]) for row in seed_rows], dtype=np.float64)
        seed_diff = np.asarray([float(row["action_abs_diff"]) for row in seed_rows], dtype=np.float64)
        seed_summaries.append(
            {
                "actual_seed": seed,
                "critic_prefers_reference_rate": float(np.mean(seed_adv > 0.0)),
                "q_reference_minus_actor_mean": float(np.mean(seed_adv)),
                "q_reference_minus_actor_median": float(np.median(seed_adv)),
                "action_abs_diff_mean": float(np.mean(seed_diff)),
                "action_abs_diff_median": float(np.median(seed_diff)),
            }
        )
    return {
        "theta_bins": theta_bins,
        "velocity_bins": velocity_bins,
        "velocity_limit": velocity_limit,
        "num_initial_condition_cells": theta_bins * velocity_bins,
        "num_training_seeds": len(seeds),
        "actual_seeds": seeds,
        "policy": rows[0]["policy"] if rows else None,
        "critic_prefers_reference_rate": float(np.mean(q_advantages > 0.0)) if len(q_advantages) else math.nan,
        "q_reference_minus_actor_mean": float(np.mean(q_advantages)) if len(q_advantages) else math.nan,
        "q_reference_minus_actor_median": float(np.median(q_advantages)) if len(q_advantages) else math.nan,
        "q_reference_minus_actor_p10": float(np.percentile(q_advantages, 10)) if len(q_advantages) else math.nan,
        "q_reference_minus_actor_p90": float(np.percentile(q_advantages, 90)) if len(q_advantages) else math.nan,
        "action_abs_diff_mean": float(np.mean(action_diff)) if len(action_diff) else math.nan,
        "action_abs_diff_median": float(np.median(action_diff)) if len(action_diff) else math.nan,
        "seed_summaries": seed_summaries,
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Reference Action Critic Scores",
        "",
        f"Policy: `{summary['policy']}`.",
        f"Grid: {summary['theta_bins']} x {summary['velocity_bins']} with velocity limit {summary['velocity_limit']}.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Critic prefers reference | {summary['critic_prefers_reference_rate']:.3f} |",
        f"| Mean Q(reference) - Q(actor) | {summary['q_reference_minus_actor_mean']:.4f} |",
        f"| Median Q(reference) - Q(actor) | {summary['q_reference_minus_actor_median']:.4f} |",
        f"| Mean abs action difference | {summary['action_abs_diff_mean']:.4f} |",
        "",
        "## By Seed",
        "",
        "| Seed | Critic prefers reference | Mean Q gap | Median Q gap | Mean abs action diff |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["seed_summaries"]:
        lines.append(
            f"| {row['actual_seed']} | {row['critic_prefers_reference_rate']:.3f} | "
            f"{row['q_reference_minus_actor_mean']:.4f} | {row['q_reference_minus_actor_median']:.4f} | "
            f"{row['action_abs_diff_mean']:.4f} |"
        )
    lines.append("")
    lines.append("Raw scored states: `reference_action_scores.csv`.")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
