from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.checkpoints import expand_run_dirs, load_agent_from_run
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from last_nine_rl.sac import SACAgent
from last_nine_rl.simba_v2 import SimbaCategoricalQNetwork


def main() -> None:
    args = parse_args()
    run_dirs = expand_run_dirs([Path(path) for path in args.runs], require_checkpoint=True)
    if args.seeds is not None:
        seed_names = {f"seed{seed}" for seed in args.seeds}
        run_dirs = [run_dir for run_dir in run_dirs if run_dir.name in seed_names]
    if not run_dirs:
        raise SystemExit("No run directories with config.json and checkpoints/final.pt were found.")

    result = diagnose(
        run_dirs=run_dirs,
        output_dir=Path(args.out),
        condition_label=args.condition_label,
        reference_policy=args.reference_policy,
        theta_bins=args.theta_bins,
        velocity_bins=args.velocity_bins,
        velocity_limit=args.velocity_limit,
        action_bins=args.action_bins,
        device=args.device,
        relative_rollouts_path=Path(args.relative_rollouts) if args.relative_rollouts else None,
        dp_solution_path=Path(args.dp_solution) if args.dp_solution else None,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose critic action flatness and calibration on Pendulum grids.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition-label", required=True)
    parser.add_argument("--reference-policy", choices=("controller", "dp", "best"), default="best")
    parser.add_argument("--theta-bins", type=int, default=61)
    parser.add_argument("--velocity-bins", type=int, default=41)
    parser.add_argument("--velocity-limit", type=float, default=1.0)
    parser.add_argument("--action-bins", type=int, default=41)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--relative-rollouts", default=None)
    parser.add_argument("--dp-solution", default=None)
    return parser.parse_args()


def diagnose(
    run_dirs: list[Path],
    output_dir: Path,
    condition_label: str,
    reference_policy: str,
    theta_bins: int,
    velocity_bins: int,
    velocity_limit: float,
    action_bins: int,
    device: str,
    relative_rollouts_path: Path | None,
    dp_solution_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    theta, theta_dot, obs = pendulum_grid(theta_bins, velocity_bins, velocity_limit)
    action_grid = np.linspace(-2.0, 2.0, action_bins, dtype=np.float32).reshape(-1, 1)
    relative_by_key = load_relative_rows(relative_rollouts_path)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        agent, config, _payload = load_agent_from_run(run_dir, device=device)
        if not config.env.env_id.startswith("Pendulum"):
            raise ValueError("This diagnostic currently supports only Pendulum runs.")
        horizon = int(config.env.max_episode_steps or 200)
        guidance = PendulumReferenceGuidance(
            policy=reference_policy,
            dp_solution_path=dp_solution_path or config.sac.reference_guidance_dp_solution_path,
            horizon=horizon,
        )
        seed_rows = diagnose_run(
            agent=agent,
            seed=int(config.seed),
            theta=theta,
            theta_dot=theta_dot,
            obs=obs,
            action_grid=action_grid,
            guidance=guidance,
            horizon=horizon,
            gamma=float(config.sac.gamma),
        )
        for row in seed_rows:
            joined = relative_by_key.get((row["actual_seed"], round(row["theta"], 12), round(row["theta_dot"], 12)))
            if joined is not None:
                row.update(joined)
            rows.append(row)

    summary = summarize(rows, condition_label, reference_policy, theta_bins, velocity_bins, velocity_limit, action_bins)
    write_csv(output_dir / "critic_flatness_rows.csv", rows)
    (output_dir / "critic_flatness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plots(output_dir, rows, condition_label)
    (output_dir / "critic_flatness_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def diagnose_run(
    agent: SACAgent,
    seed: int,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    obs: np.ndarray,
    action_grid: np.ndarray,
    guidance: PendulumReferenceGuidance,
    horizon: int,
    gamma: float,
) -> list[dict[str, Any]]:
    actor_action = agent.act_batch(obs, deterministic=True).reshape(-1, 1)
    reference_action = guidance.act_batch(obs).reshape(-1, 1)
    q_actor, actor_dist = score_q(agent, obs, actor_action)
    q_reference, ref_dist = score_q(agent, obs, reference_action)
    q_grid = score_action_grid(agent, obs, action_grid)
    q_argmax_index = np.argmax(q_grid, axis=1)
    q_argmax_action = action_grid[q_argmax_index, 0].reshape(-1, 1)
    q_argmax = q_grid[np.arange(len(obs)), q_argmax_index]
    q_min = np.min(q_grid, axis=1)
    q_range = q_argmax - q_min
    q_grad_actor = action_gradient(agent, obs, actor_action)
    reward_scale = reward_scale_denominator(agent)

    actor_returns = rollout_after_first_action(agent, theta, theta_dot, actor_action.reshape(-1), horizon, gamma)
    reference_returns = rollout_after_first_action(
        agent,
        theta,
        theta_dot,
        reference_action.reshape(-1),
        horizon,
        gamma,
    )
    q_argmax_returns = rollout_after_first_action(agent, theta, theta_dot, q_argmax_action.reshape(-1), horizon, gamma)

    rows: list[dict[str, Any]] = []
    for idx in range(len(obs)):
        rows.append(
            {
                "actual_seed": seed,
                "theta": float(theta[idx]),
                "theta_degrees": float(np.degrees(theta[idx])),
                "theta_dot": float(theta_dot[idx]),
                "actor_action": float(actor_action[idx, 0]),
                "reference_action": float(reference_action[idx, 0]),
                "q_argmax_action": float(q_argmax_action[idx, 0]),
                "abs_actor_reference_action_diff": float(abs(actor_action[idx, 0] - reference_action[idx, 0])),
                "abs_actor_q_argmax_action_diff": float(abs(actor_action[idx, 0] - q_argmax_action[idx, 0])),
                "abs_reference_q_argmax_action_diff": float(abs(reference_action[idx, 0] - q_argmax_action[idx, 0])),
                "q_actor": float(q_actor[idx]),
                "q_reference": float(q_reference[idx]),
                "q_argmax": float(q_argmax[idx]),
                "q_reference_minus_actor": float(q_reference[idx] - q_actor[idx]),
                "q_argmax_minus_actor": float(q_argmax[idx] - q_actor[idx]),
                "q_range": float(q_range[idx]),
                "q_range_raw_scale": float(q_range[idx] * reward_scale),
                "q_grad_actor": float(q_grad_actor[idx]),
                "abs_q_grad_actor": float(abs(q_grad_actor[idx])),
                "reward_scale_denominator": float(reward_scale),
                "actor_raw_return": float(actor_returns["raw"][idx]),
                "reference_raw_return": float(reference_returns["raw"][idx]),
                "q_argmax_raw_return": float(q_argmax_returns["raw"][idx]),
                "reference_raw_advantage": float(reference_returns["raw"][idx] - actor_returns["raw"][idx]),
                "q_argmax_raw_advantage": float(q_argmax_returns["raw"][idx] - actor_returns["raw"][idx]),
                "actor_discounted_return": float(actor_returns["discounted"][idx]),
                "reference_discounted_return": float(reference_returns["discounted"][idx]),
                "q_argmax_discounted_return": float(q_argmax_returns["discounted"][idx]),
                "reference_discounted_advantage": float(
                    reference_returns["discounted"][idx] - actor_returns["discounted"][idx]
                ),
                "q_argmax_discounted_advantage": float(
                    q_argmax_returns["discounted"][idx] - actor_returns["discounted"][idx]
                ),
                "reference_discounted_advantage_scaled": float(
                    (reference_returns["discounted"][idx] - actor_returns["discounted"][idx]) / reward_scale
                ),
                "q_argmax_discounted_advantage_scaled": float(
                    (q_argmax_returns["discounted"][idx] - actor_returns["discounted"][idx]) / reward_scale
                ),
                "q_actor_distribution_edge_prob": float(actor_dist["edge_prob"][idx]),
                "q_reference_distribution_edge_prob": float(ref_dist["edge_prob"][idx]),
                "q_actor_distribution_entropy": float(actor_dist["entropy"][idx]),
                "q_reference_distribution_entropy": float(ref_dist["entropy"][idx]),
            }
        )
    return rows


def score_q(agent: SACAgent, observations: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    obs = agent._normalize_obs_tensor(obs)
    action = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        q1 = agent.q1(obs, action).view(-1)
        q2 = agent.q2(obs, action).view(-1)
        q = torch.min(q1, q2).detach().cpu().numpy()
    return q, distribution_stats(agent, obs, action)


def score_action_grid(agent: SACAgent, observations: np.ndarray, action_grid: np.ndarray) -> np.ndarray:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    obs = agent._normalize_obs_tensor(obs)
    values = []
    for action_value in action_grid.reshape(-1):
        action = torch.full((len(observations), 1), float(action_value), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            q = torch.min(agent.q1(obs, action), agent.q2(obs, action)).view(-1).detach().cpu().numpy()
        values.append(q)
    return np.stack(values, axis=1)


def action_gradient(agent: SACAgent, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    obs = agent._normalize_obs_tensor(obs)
    action = torch.as_tensor(actions, dtype=torch.float32, device=agent.device).detach().clone().requires_grad_(True)
    q = torch.min(agent.q1(obs, action), agent.q2(obs, action)).sum()
    grad = torch.autograd.grad(q, action, retain_graph=False, create_graph=False)[0]
    return grad.detach().cpu().numpy().reshape(-1)


def distribution_stats(agent: SACAgent, obs: torch.Tensor, action: torch.Tensor) -> dict[str, np.ndarray]:
    if not isinstance(agent.q1, SimbaCategoricalQNetwork) or not isinstance(agent.q2, SimbaCategoricalQNetwork):
        nan = np.full((obs.shape[0],), np.nan, dtype=np.float64)
        return {"edge_prob": nan, "entropy": nan}
    with torch.no_grad():
        _q1, log_probs1 = agent.q1.distribution(obs, action)
        _q2, log_probs2 = agent.q2.distribution(obs, action)
        probs1 = log_probs1.exp()
        probs2 = log_probs2.exp()
        edge_prob = 0.5 * (
            probs1[:, 0] + probs1[:, -1] + probs2[:, 0] + probs2[:, -1]
        )
        entropy = -0.5 * (
            (probs1 * log_probs1).sum(dim=1) + (probs2 * log_probs2).sum(dim=1)
        )
    return {
        "edge_prob": edge_prob.detach().cpu().numpy(),
        "entropy": entropy.detach().cpu().numpy(),
    }


def reward_scale_denominator(agent: SACAgent) -> float:
    scaler = agent.reward_scaler
    if scaler is None:
        return 1.0
    var_denominator = float(np.sqrt(scaler.return_rms.var + scaler.epsilon))
    max_denominator = scaler.return_abs_max / scaler.g_max if scaler.g_max > 0.0 else 0.0
    return max(var_denominator, max_denominator, scaler.epsilon)


def rollout_after_first_action(
    agent: SACAgent,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    first_action: np.ndarray,
    horizon: int,
    gamma: float,
) -> dict[str, np.ndarray]:
    current_theta = np.asarray(theta, dtype=np.float64).copy()
    current_theta_dot = np.asarray(theta_dot, dtype=np.float64).copy()
    raw_return = np.zeros_like(current_theta, dtype=np.float64)
    discounted_return = np.zeros_like(current_theta, dtype=np.float64)
    discount = 1.0
    for step in range(horizon):
        if step == 0:
            action = np.asarray(first_action, dtype=np.float64).reshape(-1)
        else:
            obs = pendulum_obs(current_theta, current_theta_dot)
            action = agent.act_batch(obs, deterministic=True).reshape(-1).astype(np.float64)
        reward, current_theta, current_theta_dot = pendulum_step(current_theta, current_theta_dot, action)
        raw_return += reward
        discounted_return += discount * reward
        discount *= gamma
    return {"raw": raw_return, "discounted": discounted_return}


def pendulum_step(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    action: np.ndarray,
    g: float = 10.0,
    length: float = 1.0,
    mass: float = 1.0,
    dt: float = 0.05,
    max_speed: float = 8.0,
    max_torque: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torque = np.clip(np.asarray(action, dtype=np.float64), -max_torque, max_torque)
    reward = -(angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * torque**2)
    next_theta_dot = theta_dot + (
        3.0 * g / (2.0 * length) * np.sin(theta) + 3.0 / (mass * length**2) * torque
    ) * dt
    next_theta_dot = np.clip(next_theta_dot, -max_speed, max_speed)
    next_theta = theta + next_theta_dot * dt
    return reward, next_theta, next_theta_dot


def angle_normalize(theta: np.ndarray) -> np.ndarray:
    return ((theta + np.pi) % (2.0 * np.pi)) - np.pi


def pendulum_grid(theta_bins: int, velocity_bins: int, velocity_limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values = np.linspace(-math.pi, math.pi, theta_bins, endpoint=False, dtype=np.float64)
    velocity_values = np.linspace(-velocity_limit, velocity_limit, velocity_bins, dtype=np.float64)
    theta = np.tile(theta_values, len(velocity_values))
    theta_dot = np.repeat(velocity_values, len(theta_values))
    return theta, theta_dot, pendulum_obs(theta, theta_dot)


def pendulum_obs(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)


def load_relative_rows(path: Path | None) -> dict[tuple[int, float, float], dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    selected = (
        "return",
        "task_success",
        "near_best_known_return_eps",
        "near_dp_return_eps",
        "near_controller_return_eps",
        "regret_to_best_known",
        "signed_gap_to_best_known",
        "best_known_return",
    )
    out: dict[tuple[int, float, float], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(float(row["actual_seed"])), round(float(row["theta"]), 12), round(float(row["theta_dot"]), 12))
            out[key] = {f"relative_{name}": float(row[name]) for name in selected if name in row}
    return out


def summarize(
    rows: list[dict[str, Any]],
    condition_label: str,
    reference_policy: str,
    theta_bins: int,
    velocity_bins: int,
    velocity_limit: float,
    action_bins: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "condition_label": condition_label,
        "reference_policy": reference_policy,
        "theta_bins": theta_bins,
        "velocity_bins": velocity_bins,
        "velocity_limit": velocity_limit,
        "action_bins": action_bins,
        "num_rows": len(rows),
        "actual_seeds": sorted({int(row["actual_seed"]) for row in rows}),
        "overall": summarize_group(rows),
        "regions": {},
    }
    region_specs = {
        "near_best_success": lambda row: row.get("relative_near_best_known_return_eps", 1.0) > 0.5,
        "near_best_failure": lambda row: row.get("relative_near_best_known_return_eps", 1.0) < 0.5,
        "task_failure": lambda row: row.get("relative_task_success", 1.0) < 0.5,
        "near_down_abs_theta_ge_150": lambda row: abs(float(row["theta_degrees"])) >= 150.0,
        "near_down_reference_failure": lambda row: abs(float(row["theta_degrees"])) >= 150.0
        and row.get("relative_near_best_known_return_eps", 1.0) < 0.5,
    }
    for name, predicate in region_specs.items():
        group = [row for row in rows if predicate(row)]
        if group:
            summary["regions"][name] = summarize_group(group)
    return summary


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "q_reference_minus_actor",
        "q_argmax_minus_actor",
        "q_range",
        "q_range_raw_scale",
        "abs_q_grad_actor",
        "abs_actor_reference_action_diff",
        "abs_actor_q_argmax_action_diff",
        "abs_reference_q_argmax_action_diff",
        "reference_raw_advantage",
        "q_argmax_raw_advantage",
        "reference_discounted_advantage",
        "q_argmax_discounted_advantage",
        "reference_discounted_advantage_scaled",
        "q_argmax_discounted_advantage_scaled",
        "q_actor_distribution_edge_prob",
        "q_actor_distribution_entropy",
    )
    out: dict[str, float] = {"count": float(len(rows))}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[f"{key}_mean"] = math.nan
            out[f"{key}_median"] = math.nan
            continue
        out[f"{key}_mean"] = float(np.mean(finite))
        out[f"{key}_median"] = float(np.median(finite))
        out[f"{key}_p10"] = float(np.percentile(finite, 10))
        out[f"{key}_p90"] = float(np.percentile(finite, 90))
    out["critic_prefers_reference_rate"] = float(np.mean([float(row["q_reference_minus_actor"]) > 0.0 for row in rows]))
    out["true_reference_raw_better_rate"] = float(np.mean([float(row["reference_raw_advantage"]) > 0.0 for row in rows]))
    out["true_reference_discounted_better_rate"] = float(
        np.mean([float(row["reference_discounted_advantage"]) > 0.0 for row in rows])
    )
    out["q_argmax_raw_better_rate"] = float(np.mean([float(row["q_argmax_raw_advantage"]) > 0.0 for row in rows]))
    out["q_adv_vs_true_scaled_corr"] = safe_corr(
        [float(row["q_reference_minus_actor"]) for row in rows],
        [float(row["reference_discounted_advantage_scaled"]) for row in rows],
    )
    out["q_actor_vs_actor_discounted_scaled_corr"] = safe_corr(
        [float(row["q_actor"]) for row in rows],
        [float(row["actor_discounted_return"]) / float(row["reward_scale_denominator"]) for row in rows],
    )
    return out


def safe_corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return math.nan
    x = x[finite]
    y = y[finite]
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def write_plots(output_dir: Path, rows: list[dict[str, Any]], condition_label: str) -> None:
    q_adv = np.asarray([float(row["q_reference_minus_actor"]) for row in rows], dtype=np.float64)
    true_adv = np.asarray([float(row["reference_discounted_advantage_scaled"]) for row in rows], dtype=np.float64)
    q_range = np.asarray([float(row["q_range_raw_scale"]) for row in rows], dtype=np.float64)
    raw_adv = np.asarray([float(row["reference_raw_advantage"]) for row in rows], dtype=np.float64)
    action_diff = np.asarray([float(row["abs_actor_reference_action_diff"]) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(q_range, bins=60, color="#4c78a8")
    axes[0, 0].set_title("Q action range, raw-return scale")
    axes[0, 0].set_xlabel("(max_a Q - min_a Q) * reward_scale")
    axes[0, 0].set_ylabel("state-seed count")

    axes[0, 1].scatter(true_adv, q_adv, s=4, alpha=0.25, color="#f58518")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Reference first-action advantage")
    axes[0, 1].set_xlabel("true discounted advantage / reward_scale")
    axes[0, 1].set_ylabel("critic Q(ref) - Q(actor)")

    axes[1, 0].scatter(raw_adv, q_adv, s=4, alpha=0.25, color="#54a24b")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].axvline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Raw-return advantage vs critic advantage")
    axes[1, 0].set_xlabel("true raw advantage after first action")
    axes[1, 0].set_ylabel("critic Q(ref) - Q(actor)")

    axes[1, 1].scatter(action_diff, raw_adv, s=4, alpha=0.25, color="#b279a2")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Action difference vs true raw advantage")
    axes[1, 1].set_xlabel("|actor action - reference action|")
    axes[1, 1].set_ylabel("true raw advantage")

    fig.suptitle(condition_label)
    fig.tight_layout()
    fig.savefig(output_dir / "critic_flatness_diagnostics.png", dpi=180)
    plt.close(fig)


def summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        f"# Critic Flatness Diagnostic: {summary['condition_label']}",
        "",
        f"Reference policy: `{summary['reference_policy']}`.",
        f"Rows: {int(summary['num_rows'])}; seeds: {summary['actual_seeds']}.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Critic prefers reference | {overall['critic_prefers_reference_rate']:.3f} |",
        f"| True raw reference better | {overall['true_reference_raw_better_rate']:.3f} |",
        f"| True discounted reference better | {overall['true_reference_discounted_better_rate']:.3f} |",
        f"| Mean Q(ref)-Q(actor) | {overall['q_reference_minus_actor_mean']:.6f} |",
        f"| Mean true discounted ref advantage / reward scale | {overall['reference_discounted_advantage_scaled_mean']:.6f} |",
        f"| Corr Q advantage vs true scaled discounted advantage | {overall['q_adv_vs_true_scaled_corr']:.3f} |",
        f"| Mean Q action range, raw scale | {overall['q_range_raw_scale_mean']:.3f} |",
        f"| Median Q action range, raw scale | {overall['q_range_raw_scale_median']:.3f} |",
        f"| Mean |dQ/da| at actor | {overall['abs_q_grad_actor_mean']:.6f} |",
        f"| Mean distribution edge probability | {overall['q_actor_distribution_edge_prob_mean']:.6f} |",
        "",
        "## Regions",
        "",
        "| Region | Count | Critic ref | True raw ref | Mean Q adv | Mean true raw adv | Mean Q range raw | Corr Q vs true |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, region in summary["regions"].items():
        lines.append(
            f"| {name} | {int(region['count'])} | {region['critic_prefers_reference_rate']:.3f} | "
            f"{region['true_reference_raw_better_rate']:.3f} | {region['q_reference_minus_actor_mean']:.6f} | "
            f"{region['reference_raw_advantage_mean']:.3f} | {region['q_range_raw_scale_mean']:.3f} | "
            f"{region['q_adv_vs_true_scaled_corr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "- `critic_flatness_rows.csv`",
            "- `critic_flatness_summary.json`",
            "- `critic_flatness_diagnostics.png`",
        ]
    )
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
