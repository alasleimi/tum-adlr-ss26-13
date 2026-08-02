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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find when pure-RL actor trajectories first diverge from the frozen reference."
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--relative-rollouts", required=True)
    parser.add_argument("--critic-rows", default=None)
    parser.add_argument("--dp-solution", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--condition", default="policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = expand_run_dirs([Path(value) for value in args.runs], require_checkpoint=True)
    if len(run_dirs) != 5:
        raise SystemExit(f"Expected five pure-RL runs, found {len(run_dirs)}")
    relative = rows_by_seed(read_csv(Path(args.relative_rollouts)))
    critic = rows_by_key(read_csv(Path(args.critic_rows))) if args.critic_rows else None
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        agent, config, _payload = load_agent_from_run(run_dir, device=args.device)
        seed = int(config.seed)
        seed_rows, seed_validation = diagnose_seed(
            agent,
            relative[seed],
            critic,
            seed=seed,
            dp_solution=Path(args.dp_solution),
        )
        rows.extend(seed_rows)
        validation.append(seed_validation)

    rows.sort(key=lambda row: (int(row["seed"]), float(row["theta"]), float(row["theta_dot"])))
    write_csv(output_dir / "first_divergence_rows.csv", rows)
    summary = summarize(rows, validation, condition=str(args.condition))
    (output_dir / "first_divergence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "first_divergence_summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    plot(output_dir / "first_divergence_diagnostic.png", rows, summary)
    print(json.dumps(summary["headline"], indent=2, sort_keys=True, allow_nan=False))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_by_seed(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(float(row["actual_seed"])), []).append(row)
    for seed, values in grouped.items():
        values.sort(key=lambda row: (float(row["theta_dot"]), float(row["theta"])))
        if len(values) != 2501:
            raise ValueError(f"Seed {seed} has {len(values)} rows, expected 2501")
    return grouped


def rows_by_key(rows: list[dict[str, str]]) -> dict[tuple[int, float, float], dict[str, str]]:
    return {
        (
            int(float(row["actual_seed"])),
            round(float(row["theta"]), 12),
            round(float(row["theta_dot"]), 12),
        ): row
        for row in rows
    }


def diagnose_seed(
    agent: SACAgent,
    relative_rows: list[dict[str, str]],
    critic_rows: dict[tuple[int, float, float], dict[str, str]] | None,
    *,
    seed: int,
    dp_solution: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agent.actor.eval()
    theta0 = np.asarray([float(row["theta"]) for row in relative_rows], dtype=np.float64)
    velocity0 = np.asarray([float(row["theta_dot"]) for row in relative_rows], dtype=np.float64)
    use_dp = np.asarray(
        [float(row["dp_policy_return"]) >= float(row["controller_return"]) for row in relative_rows],
        dtype=bool,
    )
    dp = PendulumReferenceGuidance(policy="dp", dp_solution_path=dp_solution, horizon=200)
    controller = PendulumReferenceGuidance(policy="controller", horizon=200)

    actor_theta = theta0.copy()
    actor_velocity = velocity0.copy()
    ref_theta = theta0.copy()
    ref_velocity = velocity0.copy()
    actor_return = np.zeros_like(theta0)
    ref_return = np.zeros_like(theta0)
    first_action_gap = np.zeros_like(theta0)
    initial_actor_action = np.zeros_like(theta0)
    initial_reference_action = np.zeros_like(theta0)
    first_action_opposite = np.zeros(len(theta0), dtype=bool)
    max_action_gap_first10 = np.zeros_like(theta0)
    first_action_gap025 = np.full(len(theta0), -1, dtype=np.int64)
    first_action_gap100 = np.full(len(theta0), -1, dtype=np.int64)
    first_state_divergence = np.full(len(theta0), -1, dtype=np.int64)
    first_return_gap1 = np.full(len(theta0), -1, dtype=np.int64)
    first_return_gap5 = np.full(len(theta0), -1, dtype=np.int64)

    for step in range(200):
        actor_obs = pendulum_obs(actor_theta, actor_velocity)
        ref_obs = pendulum_obs(ref_theta, ref_velocity)
        actor_action = np.asarray(
            agent.act_batch(actor_obs, deterministic=True), dtype=np.float64
        ).reshape(-1)
        remaining = 200 - step
        dp_action = np.asarray(dp.act_batch(ref_obs, remaining_steps=remaining), dtype=np.float64).reshape(-1)
        controller_action = np.asarray(
            controller.act_batch(ref_obs, remaining_steps=remaining), dtype=np.float64
        ).reshape(-1)
        ref_action = np.where(use_dp, dp_action, controller_action)
        action_gap = np.abs(actor_action - ref_action)
        if step == 0:
            initial_actor_action = actor_action.copy()
            initial_reference_action = ref_action.copy()
            first_action_gap = action_gap.copy()
            first_action_opposite = (actor_action * ref_action < -0.25) & (
                np.abs(actor_action) > 0.5
            ) & (np.abs(ref_action) > 0.5)
        if step < 10:
            max_action_gap_first10 = np.maximum(max_action_gap_first10, action_gap)
        first_action_gap025[(first_action_gap025 < 0) & (action_gap > 0.25)] = step
        first_action_gap100[(first_action_gap100 < 0) & (action_gap > 1.0)] = step

        actor_reward, actor_theta, actor_velocity = pendulum_step(
            actor_theta, actor_velocity, actor_action
        )
        ref_reward, ref_theta, ref_velocity = pendulum_step(ref_theta, ref_velocity, ref_action)
        actor_return += actor_reward
        ref_return += ref_reward
        state_distance = normalized_state_distance(actor_theta, actor_velocity, ref_theta, ref_velocity)
        first_state_divergence[(first_state_divergence < 0) & (state_distance > 0.1)] = step + 1
        cumulative_gap = ref_return - actor_return
        first_return_gap1[(first_return_gap1 < 0) & (cumulative_gap > 1.0)] = step + 1
        first_return_gap5[(first_return_gap5 < 0) & (cumulative_gap > 5.0)] = step + 1

    counterfactual_return = first_action_then_actor_return(
        agent,
        theta0,
        velocity0,
        initial_reference_action,
        horizon=200,
    )
    reference_first_action_advantage_all = counterfactual_return - actor_return
    critic_reference_advantage_all = critic_advantage(
        agent,
        pendulum_obs(theta0, velocity0),
        initial_actor_action,
        initial_reference_action,
    )

    actor_expected = np.asarray([float(row["return"]) for row in relative_rows], dtype=np.float64)
    reference_expected = np.asarray(
        [float(row["best_known_return"]) for row in relative_rows], dtype=np.float64
    )
    output: list[dict[str, Any]] = []
    for index, relative_row in enumerate(relative_rows):
        theta = float(relative_row["theta"])
        theta_dot = float(relative_row["theta_dot"])
        key = (seed, round(theta, 12), round(theta_dot, 12))
        reference_first_action_advantage = float(reference_first_action_advantage_all[index])
        critic_reference_advantage = float(critic_reference_advantage_all[index])
        if critic_rows is not None:
            critic_row = critic_rows[key]
            stored_reference_advantage = float(critic_row["reference_raw_advantage"])
            stored_critic_advantage = float(critic_row["q_reference_minus_actor"])
            if abs(stored_reference_advantage - reference_first_action_advantage) > 1e-4:
                raise ValueError(f"Stored counterfactual mismatch at {key}")
            if abs(stored_critic_advantage - critic_reference_advantage) > 1e-5:
                raise ValueError(f"Stored critic advantage mismatch at {key}")
        failure = int(float(relative_row["near_best_known_return_eps"]) < 0.5)
        one_step_recoverable = int(failure and reference_first_action_advantage > 5.0)
        output.append(
            {
                "seed": seed,
                "theta": theta,
                "theta_degrees": float(relative_row["theta_degrees"]),
                "theta_dot": theta_dot,
                "near_reference_failure": failure,
                "task_failure": int(float(relative_row["task_success"]) < 0.5),
                "actor_return": float(actor_return[index]),
                "reference_return": float(ref_return[index]),
                "reference_minus_actor_return": float(ref_return[index] - actor_return[index]),
                "first_action_gap": float(first_action_gap[index]),
                "first_action_opposite_direction": int(first_action_opposite[index]),
                "max_action_gap_first10": float(max_action_gap_first10[index]),
                "first_action_gap_gt_0_25_step": int(first_action_gap025[index]),
                "first_action_gap_gt_1_00_step": int(first_action_gap100[index]),
                "first_state_divergence_step": int(first_state_divergence[index]),
                "first_cumulative_return_gap_gt_1_step": int(first_return_gap1[index]),
                "first_cumulative_return_gap_gt_5_step": int(first_return_gap5[index]),
                "reference_first_action_advantage": reference_first_action_advantage,
                "critic_reference_advantage": critic_reference_advantage,
                "critic_prefers_reference_first_action": int(critic_reference_advantage > 0.0),
                "one_step_recoverable_by_reference_gt_5": one_step_recoverable,
            }
        )
    validation = {
        "seed": seed,
        "actor_return_max_abs_error": float(np.max(np.abs(actor_return - actor_expected))),
        "reference_return_max_abs_error": float(np.max(np.abs(ref_return - reference_expected))),
        "actor_return_mean_abs_error": float(np.mean(np.abs(actor_return - actor_expected))),
        "reference_return_mean_abs_error": float(np.mean(np.abs(ref_return - reference_expected))),
    }
    return output, validation


def first_action_then_actor_return(
    agent: SACAgent,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    first_action: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    current_theta = np.asarray(theta, dtype=np.float64).copy()
    current_velocity = np.asarray(theta_dot, dtype=np.float64).copy()
    returns = np.zeros_like(current_theta)
    for step in range(int(horizon)):
        if step == 0:
            action = np.asarray(first_action, dtype=np.float64)
        else:
            action = np.asarray(
                agent.act_batch(pendulum_obs(current_theta, current_velocity), deterministic=True),
                dtype=np.float64,
            ).reshape(-1)
        reward, current_theta, current_velocity = pendulum_step(
            current_theta, current_velocity, action
        )
        returns += reward
    return returns


def critic_advantage(
    agent: SACAgent,
    observations: np.ndarray,
    actor_actions: np.ndarray,
    reference_actions: np.ndarray,
) -> np.ndarray:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    obs = agent._normalize_obs_tensor(obs)
    actor = torch.as_tensor(actor_actions[:, None], dtype=torch.float32, device=agent.device)
    reference = torch.as_tensor(reference_actions[:, None], dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        actor_q = torch.minimum(agent.q1(obs, actor), agent.q2(obs, actor))
        reference_q = torch.minimum(agent.q1(obs, reference), agent.q2(obs, reference))
    return (reference_q - actor_q).detach().cpu().numpy().reshape(-1).astype(np.float64)


def pendulum_obs(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)


def pendulum_step(
    theta: np.ndarray, theta_dot: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torque = np.clip(np.asarray(action, dtype=np.float64), -2.0, 2.0)
    wrapped = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
    reward = -(wrapped**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)
    next_velocity = np.clip(theta_dot + (15.0 * np.sin(theta) + 3.0 * torque) * 0.05, -8.0, 8.0)
    next_theta = theta + next_velocity * 0.05
    return reward, next_theta, next_velocity


def normalized_state_distance(
    theta_a: np.ndarray,
    velocity_a: np.ndarray,
    theta_b: np.ndarray,
    velocity_b: np.ndarray,
) -> np.ndarray:
    angle_delta = ((theta_a - theta_b + math.pi) % (2.0 * math.pi)) - math.pi
    return np.sqrt((angle_delta / math.pi) ** 2 + ((velocity_a - velocity_b) / 2.0) ** 2)


def summarize(
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    *,
    condition: str,
) -> dict[str, Any]:
    failure = [row for row in rows if int(row["near_reference_failure"]) == 1]
    success = [row for row in rows if int(row["near_reference_failure"]) == 0]
    fail_first = np.asarray([float(row["first_action_gap"]) for row in failure])
    success_first = np.asarray([float(row["first_action_gap"]) for row in success])
    fail_return_step = np.asarray(
        [int(row["first_cumulative_return_gap_gt_5_step"]) for row in failure], dtype=np.int64
    )
    fail_state_step = np.asarray([int(row["first_state_divergence_step"]) for row in failure], dtype=np.int64)
    one_step = [row for row in failure if int(row["one_step_recoverable_by_reference_gt_5"]) == 1]
    critic_catches = [row for row in one_step if int(row["critic_prefers_reference_first_action"]) == 1]
    headline = {
        "failure_trials": len(failure),
        "failure_first_action_gap_gt_0_25_rate": float(np.mean(fail_first > 0.25)),
        "success_first_action_gap_gt_0_25_rate": float(np.mean(success_first > 0.25)),
        "failure_opposite_first_action_rate": float(
            np.mean([int(row["first_action_opposite_direction"]) for row in failure])
        ),
        "one_step_recoverable_gt_5_rate": float(len(one_step) / len(failure)),
        "critic_catches_one_step_recoverable_rate": float(len(critic_catches) / max(len(one_step), 1)),
        "median_first_state_divergence_step": float(np.median(fail_state_step[fail_state_step >= 0])),
        "median_first_return_gap_gt_5_step": float(np.median(fail_return_step[fail_return_step >= 0])),
    }
    return {
        "schema_version": 1,
        "condition": condition,
        "diagnostic_scope": (
            "Frozen five-seed actor trajectories are compared with the stored best reference policy "
            "from the identical initial state. A first-action counterfactual then executes the "
            "reference action once and returns control to the actor."
        ),
        "headline": headline,
        "failure": group_summary(failure),
        "success": group_summary(success),
        "one_step_recoverable_count": len(one_step),
        "one_step_recoverable_critic_caught_count": len(critic_catches),
        "validation": validation,
        "interpretation_limits": [
            "Reference trajectories are diagnostic comparators and are never used by the evaluated actor.",
            "A successful one-step counterfactual identifies a local actor error, but it does not prove which optimizer update created that error.",
            "Trajectory divergence thresholds are descriptive and are reported alongside the underlying per-state rows.",
        ],
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    return {
        "count": len(rows),
        "first_action_gap_median": float(np.median(values("first_action_gap"))),
        "first_action_gap_p90": float(np.percentile(values("first_action_gap"), 90)),
        "first_action_gap_gt_0_25_rate": float(np.mean(values("first_action_gap") > 0.25)),
        "first_action_gap_gt_1_rate": float(np.mean(values("first_action_gap") > 1.0)),
        "opposite_first_action_rate": float(np.mean(values("first_action_opposite_direction"))),
        "max_action_gap_first10_median": float(np.median(values("max_action_gap_first10"))),
        "reference_first_action_advantage_mean": float(
            np.mean(values("reference_first_action_advantage"))
        ),
        "reference_first_action_advantage_gt_5_rate": float(
            np.mean(values("reference_first_action_advantage") > 5.0)
        ),
        "critic_prefers_reference_first_action_rate": float(
            np.mean(values("critic_prefers_reference_first_action"))
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    h = summary["headline"]
    return "\n".join(
        [
            f"# First-divergence diagnostic: {summary['condition']}",
            "",
            "This is a five-seed, fixed-checkpoint trajectory and first-action intervention.",
            "",
            f"- Near-reference failures: {h['failure_trials']}.",
            (
                "- Initial actor/reference action gap exceeds 0.25 on "
                f"{100.0 * h['failure_first_action_gap_gt_0_25_rate']:.2f}% of failures and "
                f"{100.0 * h['success_first_action_gap_gt_0_25_rate']:.2f}% of successes."
            ),
            (
                "- A single reference first action improves the actor-tail return by more than 5 on "
                f"{100.0 * h['one_step_recoverable_gt_5_rate']:.2f}% of failures."
            ),
            (
                "- The critic assigns positive advantage to that corrective action on "
                f"{100.0 * h['critic_catches_one_step_recoverable_rate']:.2f}% of one-step-recoverable failures."
            ),
            (
                f"- Median first normalized-state divergence step among failures: "
                f"{h['median_first_state_divergence_step']:.1f}."
            ),
            (
                f"- Median first cumulative return-gap-above-5 step among failures: "
                f"{h['median_first_return_gap_gt_5_step']:.1f}."
            ),
            "",
            "## Limits",
            "",
            *[f"- {item}" for item in summary["interpretation_limits"]],
            "",
        ]
    )


def plot(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    failure = [row for row in rows if int(row["near_reference_failure"]) == 1]
    success = [row for row in rows if int(row["near_reference_failure"]) == 0]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    bins = np.linspace(0.0, 4.0, 41)
    axes[0].hist(
        [float(row["first_action_gap"]) for row in success],
        bins=bins,
        density=True,
        alpha=0.55,
        label="success",
        color="#4C78A8",
    )
    axes[0].hist(
        [float(row["first_action_gap"]) for row in failure],
        bins=bins,
        density=True,
        alpha=0.65,
        label="failure",
        color="#E45756",
    )
    axes[0].set_title("Initial action mismatch")
    axes[0].set_xlabel("|actor action - reference action|")
    axes[0].set_ylabel("density")
    axes[0].legend()

    first_state = np.asarray(
        [int(row["first_state_divergence_step"]) for row in failure], dtype=np.int64
    )
    first_return = np.asarray(
        [int(row["first_cumulative_return_gap_gt_5_step"]) for row in failure], dtype=np.int64
    )
    axes[1].hist(first_state[first_state >= 0], bins=np.arange(0, 202, 5), alpha=0.7, label="state")
    axes[1].hist(first_return[first_return >= 0], bins=np.arange(0, 202, 5), alpha=0.6, label="return > 5")
    axes[1].set_title("When failure trajectories separate")
    axes[1].set_xlabel("first step")
    axes[1].set_ylabel("failed seed-state trials")
    axes[1].legend()

    categories = [
        summary["headline"]["one_step_recoverable_gt_5_rate"],
        summary["headline"]["critic_catches_one_step_recoverable_rate"],
        summary["headline"]["failure_opposite_first_action_rate"],
    ]
    labels = ["one-step\nrecoverable", "critic catches\nrecoverable", "opposite\nfirst action"]
    axes[2].bar(labels, categories, color=["#59A14F", "#F28E2B", "#E45756"])
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("fraction")
    axes[2].set_title("Local actor and critic failure modes")
    for index, value in enumerate(categories):
        axes[2].text(index, value + 0.02, f"{100.0 * value:.1f}%", ha="center")

    fig.suptitle("Pure RL failure timing and one-step counterfactuals", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
