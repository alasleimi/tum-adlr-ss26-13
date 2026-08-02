from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.envs import UprightDetector
from last_nine_rl.pendulum_grid import rollout_pendulum_grid_vectorized
from last_nine_rl.qsearch_lock import build_validation_dataset
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
try:
    from scripts.diagnose_critic_action_gradient_alignment_20260723 import (
        critic_gradients,
        load_spec,
        resolve,
    )
    from scripts.diagnose_critic_flatness import (
        pendulum_obs,
        pendulum_step,
    )
    from scripts.diagnose_first_divergence_20260723 import critic_advantage
    from scripts.train_pendulum_qregularized_dagger import (
        FiniteHorizonReferencePolicy,
    )
except ModuleNotFoundError:
    from diagnose_critic_action_gradient_alignment_20260723 import (  # type: ignore[no-redef]
        critic_gradients,
        load_spec,
        resolve,
    )
    from diagnose_critic_flatness import (  # type: ignore[no-redef]
        pendulum_obs,
        pendulum_step,
    )
    from diagnose_first_divergence_20260723 import critic_advantage  # type: ignore[no-redef]
    from train_pendulum_qregularized_dagger import (  # type: ignore[no-redef]
        FiniteHorizonReferencePolicy,
    )


ROOT = Path(__file__).resolve().parents[2]
ROW_FIELDS = [
    "condition",
    "replicate",
    "seed",
    "state_index",
    "near_reference",
    "first_action_gap_step",
    "remaining_horizon",
    "theta_at_gap",
    "velocity_at_gap",
    "actor_action",
    "reference_action",
    "action_gap",
    "reference_one_step_raw_gain",
    "reference_one_step_discounted_gain",
    "q_reference_minus_actor",
    "critic_prefers_reference",
    "critic_gradient",
    "q1_gradient",
    "q2_gradient",
    "critic_gradient_points_toward_reference",
    "twin_gradient_sign_agreement",
    "actor_at_boundary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "At each actor's first action disagreement with a fixed selected "
            "reference, test whether its own critic recognizes the reference action."
        )
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT
        / "experiments"
        / "protocols"
        / "plan2307_p0_p1_p2_action_gradient_diagnostic_20260724.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "experiments"
        / "protocols"
        / "pure_rl_offgrid_validation_protocol_20260722.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / ".build"
        / "diagnostics"
        / "divergence_state_critic",
    )
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=230723)
    parser.add_argument("--action-gap", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_reference(
    theta: np.ndarray,
    velocity: np.ndarray,
    detector: UprightDetector,
    reliability: ReliabilityConfig,
    dp_solution: Path,
) -> tuple[np.ndarray, np.ndarray]:
    returns: list[np.ndarray] = []
    for policy_name in ("dp", "controller"):
        policy = FiniteHorizonReferencePolicy(policy_name, dp_solution)
        rows = rollout_pendulum_grid_vectorized(
            policy,
            theta,
            velocity,
            detector,
            reliability,
            horizon=200,
        )
        returns.append(np.asarray([row["return"] for row in rows], dtype=np.float64))
    use_dp = returns[0] >= returns[1]
    return np.maximum(returns[0], returns[1]), use_dp


def critic_runs(condition: dict[str, Any]) -> list[str]:
    actors = list(condition["actor_runs"])
    critics = condition.get("critic_runs", "same")
    if critics == "same":
        return actors
    values = list(critics)
    return values * len(actors) if len(values) == 1 else values


def find_first_disagreement(
    actor: Any,
    theta: np.ndarray,
    velocity: np.ndarray,
    use_dp: np.ndarray,
    action_gap: float,
    dp_solution: Path,
) -> dict[str, np.ndarray]:
    current_theta = np.asarray(theta, dtype=np.float64).copy()
    current_velocity = np.asarray(velocity, dtype=np.float64).copy()
    count = len(theta)
    found = np.zeros(count, dtype=bool)
    step_found = np.full(count, -1, dtype=np.int64)
    state_theta = np.full(count, np.nan, dtype=np.float64)
    state_velocity = np.full(count, np.nan, dtype=np.float64)
    actor_at_divergence = np.full(count, np.nan, dtype=np.float64)
    reference_at_divergence = np.full(count, np.nan, dtype=np.float64)
    actor_return = np.zeros(count, dtype=np.float64)
    dp = PendulumReferenceGuidance(
        "dp", dp_solution_path=dp_solution, horizon=200
    )
    controller = PendulumReferenceGuidance("controller", horizon=200)
    for step in range(200):
        observations = pendulum_obs(current_theta, current_velocity)
        actor_action = np.asarray(
            actor.act_batch(observations, deterministic=True), dtype=np.float64
        ).reshape(-1)
        remaining = 200 - step
        dp_action = np.asarray(
            dp.act_batch(observations, remaining_steps=remaining), dtype=np.float64
        ).reshape(-1)
        controller_action = np.asarray(
            controller.act_batch(observations, remaining_steps=remaining),
            dtype=np.float64,
        ).reshape(-1)
        reference_action = np.where(use_dp, dp_action, controller_action)
        new = (~found) & (np.abs(actor_action - reference_action) > action_gap)
        step_found[new] = step
        state_theta[new] = current_theta[new]
        state_velocity[new] = current_velocity[new]
        actor_at_divergence[new] = actor_action[new]
        reference_at_divergence[new] = reference_action[new]
        found |= new
        reward, current_theta, current_velocity = pendulum_step(
            current_theta, current_velocity, actor_action
        )
        actor_return += reward
    return {
        "found": found,
        "step": step_found,
        "theta": state_theta,
        "velocity": state_velocity,
        "actor_action": actor_at_divergence,
        "reference_action": reference_at_divergence,
        "actor_return": actor_return,
    }


def rollout_after_first_action_variable_horizon(
    actor: Any,
    theta: np.ndarray,
    velocity: np.ndarray,
    first_action: np.ndarray,
    horizons: np.ndarray,
    gamma: float = 0.99,
) -> dict[str, np.ndarray]:
    current_theta = np.asarray(theta, dtype=np.float64).copy()
    current_velocity = np.asarray(velocity, dtype=np.float64).copy()
    first_action = np.asarray(first_action, dtype=np.float64).reshape(-1)
    horizons = np.asarray(horizons, dtype=np.int64).reshape(-1)
    raw = np.zeros_like(current_theta)
    discounted = np.zeros_like(current_theta)
    discount = 1.0
    for step in range(int(horizons.max())):
        active = step < horizons
        if step == 0:
            action = first_action
        else:
            action = np.asarray(
                actor.act_batch(
                    pendulum_obs(current_theta, current_velocity),
                    deterministic=True,
                ),
                dtype=np.float64,
            ).reshape(-1)
        reward, next_theta, next_velocity = pendulum_step(
            current_theta, current_velocity, action
        )
        raw[active] += reward[active]
        discounted[active] += discount * reward[active]
        current_theta = np.where(active, next_theta, current_theta)
        current_velocity = np.where(active, next_velocity, current_velocity)
        discount *= gamma
    return {"raw": raw, "discounted": discounted}


def evaluate_pair(
    condition: str,
    replicate: int,
    actor_path: Path,
    critic_path: Path,
    theta: np.ndarray,
    velocity: np.ndarray,
    reference_returns: np.ndarray,
    use_dp: np.ndarray,
    epsilon: float,
    action_gap: float,
    dp_solution: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actor, actor_config, _ = load_agent_from_run(actor_path, device=device)
    critic = (
        actor
        if actor_path.resolve() == critic_path.resolve()
        else load_agent_from_run(critic_path, device=device)[0]
    )
    located = find_first_disagreement(
        actor, theta, velocity, use_dp, action_gap, dp_solution
    )
    near = located["actor_return"] >= reference_returns - epsilon
    found_indices = np.flatnonzero(located["found"])
    rows: list[dict[str, Any]] = []
    if not len(found_indices):
        identity = {
            "condition": condition,
            "replicate": replicate,
            "seed": int(actor_config.seed),
            "sample_rows": len(theta),
            "near_reference_successes": int(near.sum()),
            "first_gap_found": 0,
            "first_gap_found_failures": 0,
            "actor_run": str(actor_path.relative_to(ROOT)).replace("\\", "/"),
            "critic_run": str(critic_path.relative_to(ROOT)).replace("\\", "/"),
            "actor_checkpoint_sha256": sha256(
                actor_path / "checkpoints" / "final.pt"
            ),
            "critic_checkpoint_sha256": sha256(
                critic_path / "checkpoints" / "final.pt"
            ),
        }
        return rows, identity
    local_theta = located["theta"][found_indices]
    local_velocity = located["velocity"][found_indices]
    actor_action = located["actor_action"][found_indices]
    reference_action = located["reference_action"][found_indices]
    remaining = 200 - located["step"][found_indices]
    actor_continuation = rollout_after_first_action_variable_horizon(
        actor,
        local_theta,
        local_velocity,
        actor_action,
        remaining,
    )
    reference_continuation = rollout_after_first_action_variable_horizon(
        actor,
        local_theta,
        local_velocity,
        reference_action,
        remaining,
    )
    observations = pendulum_obs(local_theta, local_velocity)
    q_reference_minus_actor = critic_advantage(
        critic, observations, actor_action, reference_action
    )
    q_gradient, q1_gradient, q2_gradient = critic_gradients(
        critic, observations, actor_action
    )
    toward_reference = np.sign(reference_action - actor_action)
    for local_index, source_index in enumerate(found_indices):
        raw_gain = (
            reference_continuation["raw"][local_index]
            - actor_continuation["raw"][local_index]
        )
        discounted_gain = (
            reference_continuation["discounted"][local_index]
            - actor_continuation["discounted"][local_index]
        )
        rows.append(
                {
                    "condition": condition,
                    "replicate": replicate,
                    "seed": int(actor_config.seed),
                    "state_index": int(source_index),
                    "near_reference": int(near[source_index]),
                    "first_action_gap_step": int(located["step"][source_index]),
                    "remaining_horizon": int(remaining[local_index]),
                    "theta_at_gap": float(local_theta[local_index]),
                    "velocity_at_gap": float(local_velocity[local_index]),
                    "actor_action": float(actor_action[local_index]),
                    "reference_action": float(reference_action[local_index]),
                    "action_gap": float(
                        abs(reference_action[local_index] - actor_action[local_index])
                    ),
                    "reference_one_step_raw_gain": float(raw_gain),
                    "reference_one_step_discounted_gain": float(discounted_gain),
                    "q_reference_minus_actor": float(
                        q_reference_minus_actor[local_index]
                    ),
                    "critic_prefers_reference": int(
                        q_reference_minus_actor[local_index] > 0
                    ),
                    "critic_gradient": float(q_gradient[local_index]),
                    "q1_gradient": float(q1_gradient[local_index]),
                    "q2_gradient": float(q2_gradient[local_index]),
                    "critic_gradient_points_toward_reference": int(
                        np.sign(q_gradient[local_index])
                        == toward_reference[local_index]
                    ),
                    "twin_gradient_sign_agreement": int(
                        np.sign(q1_gradient[local_index])
                        == np.sign(q2_gradient[local_index])
                    ),
                    "actor_at_boundary": int(
                        abs(actor_action[local_index]) > 1.95
                    ),
                }
        )
    identity = {
        "condition": condition,
        "replicate": replicate,
        "seed": int(actor_config.seed),
        "sample_rows": len(theta),
        "near_reference_successes": int(near.sum()),
        "first_gap_found": int(located["found"].sum()),
        "first_gap_found_failures": int((located["found"] & ~near).sum()),
        "actor_run": str(actor_path.relative_to(ROOT)).replace("\\", "/"),
        "critic_run": str(critic_path.relative_to(ROOT)).replace("\\", "/"),
        "actor_checkpoint_sha256": sha256(actor_path / "checkpoints" / "final.pt"),
        "critic_checkpoint_sha256": sha256(critic_path / "checkpoints" / "final.pt"),
    }
    return rows, identity


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "status": "undefined_no_rows",
            "median_first_action_gap_step": None,
            "critic_prefers_reference_rate": None,
            "critic_gradient_points_toward_reference_rate": None,
            "reference_one_step_beneficial_rate": None,
            "critic_prefers_helpful_reference_rate": None,
            "reference_one_step_discounted_gain_mean": None,
            "reference_one_step_discounted_gain_median": None,
            "actor_boundary_rate": None,
            "twin_gradient_sign_agreement": None,
        }
    gains = np.asarray(
        [float(row["reference_one_step_discounted_gain"]) for row in rows],
        dtype=np.float64,
    )
    helpful = gains > 1e-8
    prefers = np.asarray(
        [int(row["critic_prefers_reference"]) for row in rows], dtype=bool
    )
    gradient = np.asarray(
        [int(row["critic_gradient_points_toward_reference"]) for row in rows],
        dtype=bool,
    )
    return {
        "rows": len(rows),
        "status": "estimated",
        "median_first_action_gap_step": float(
            np.median([int(row["first_action_gap_step"]) for row in rows])
        ),
        "critic_prefers_reference_rate": float(prefers.mean()),
        "critic_gradient_points_toward_reference_rate": float(gradient.mean()),
        "reference_one_step_beneficial_rate": float(helpful.mean()),
        "critic_prefers_helpful_reference_rate": (
            float(prefers[helpful].mean()) if int(helpful.sum()) else float("nan")
        ),
        "reference_one_step_discounted_gain_mean": float(gains.mean()),
        "reference_one_step_discounted_gain_median": float(np.median(gains)),
        "actor_boundary_rate": float(
            np.mean([int(row["actor_at_boundary"]) for row in rows])
        ),
        "twin_gradient_sign_agreement": float(
            np.mean([int(row["twin_gradient_sign_agreement"]) for row in rows])
        ),
    }


def summarize(
    rows: list[dict[str, Any]], identities: list[dict[str, Any]]
) -> dict[str, Any]:
    conditions = list(dict.fromkeys(row["condition"] for row in identities))
    result: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        pooled: dict[str, Any] = {}
        for outcome, value in (("failure", 0), ("success", 1)):
            subset = [
                row for row in condition_rows if int(row["near_reference"]) == value
            ]
            pooled[outcome] = summarize_group(subset)
        per_seed = []
        for replicate in range(5):
            subset = [
                row
                for row in condition_rows
                if int(row["replicate"]) == replicate
                and int(row["near_reference"]) == 0
            ]
            per_seed.append(
                {"replicate": replicate, "failure": summarize_group(subset)}
            )
        condition_identities = [
            row for row in identities if row["condition"] == condition
        ]
        pooled["all_initial_states"] = {
            "rows": sum(int(row["sample_rows"]) for row in condition_identities),
            "first_gap_found_rate": sum(
                int(row["first_gap_found"]) for row in condition_identities
            )
            / sum(int(row["sample_rows"]) for row in condition_identities),
            "near_reference_rate": sum(
                int(row["near_reference_successes"]) for row in condition_identities
            )
            / sum(int(row["sample_rows"]) for row in condition_identities),
        }
        result[condition] = {"pooled": pooled, "per_seed": per_seed}
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ROW_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plot(output: Path, summary: dict[str, Any]) -> None:
    conditions = list(summary)
    colors = ["#147d73", "#bd5705", "#2864e8"]
    metrics = (
        (
            "critic_prefers_reference_rate",
            "A. Critic ranks reference above actor",
            "failure states (%)",
        ),
        (
            "critic_gradient_points_toward_reference_rate",
            "B. Local critic gradient points toward reference",
            "failure states (%)",
        ),
        (
            "reference_one_step_beneficial_rate",
            "C. One reference action improves continuation",
            "failure states (%)",
        ),
        (
            "critic_prefers_helpful_reference_rate",
            "D. Critic recognizes helpful reference actions",
            "helpful actions (%)",
        ),
        (
            "actor_boundary_rate",
            "E. Actor at boundary when actions first disagree",
            "failure states (%)",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3))
    x = np.arange(len(conditions))
    for axis, (metric, title, ylabel) in zip(axes.flat[:5], metrics):
        raw_values = [
            summary[c]["pooled"]["failure"][metric] for c in conditions
        ]
        values = [
            np.nan if value is None else 100 * float(value)
            for value in raw_values
        ]
        axis.bar(x, values, color=colors)
        for index, value in enumerate(raw_values):
            if value is None:
                axis.text(
                    index,
                    4,
                    "no failures",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color="#64748B",
                    fontsize=8,
                )
        axis.set_xticks(x, conditions, rotation=15, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
    for condition, color in zip(conditions, colors):
        per_seed = summary[condition]["per_seed"]
        axes[1, 2].plot(
            range(5),
            [
                (
                    np.nan
                    if row["failure"][
                        "critic_prefers_helpful_reference_rate"
                    ]
                    is None
                    else 100
                    * float(
                        row["failure"][
                            "critic_prefers_helpful_reference_rate"
                        ]
                    )
                )
                for row in per_seed
            ],
            marker="o",
            linewidth=2,
            color=color,
            label=condition,
        )
    axes[1, 2].set_title("F. Helpful-action recognition by trained seed")
    axes[1, 2].set_xlabel("actor seed")
    axes[1, 2].set_ylabel("helpful actions (%)")
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_ylim(0, 100)
    axes[1, 2].grid(axis="y", alpha=0.2)
    axes[1, 2].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Critic recognition at the first actor-reference action disagreement\n"
        "Five actor seeds per pipeline on the same locked state set",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "divergence_state_critic.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    protocol_path = resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset = build_validation_dataset(protocol["validation_dataset"])
    rng = np.random.default_rng(args.sample_seed)
    selected = np.sort(
        rng.choice(dataset["points"], size=args.sample_count, replace=False)
    )
    theta = np.asarray(dataset["theta"])[selected]
    velocity = np.asarray(dataset["velocity"])[selected]
    reference_spec = protocol["reference_protocol"]
    dp_solution = resolve(reference_spec["dp_solution"]["path"])
    if (
        dp_solution.stat().st_size
        != int(reference_spec["dp_solution"]["size_bytes"])
        or sha256(dp_solution) != str(reference_spec["dp_solution"]["sha256"])
    ):
        raise ValueError("Pinned off-grid DP solution fingerprint drift.")
    reliability = ReliabilityConfig(**protocol["evaluation_reliability"])
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    reference_returns, use_dp = selected_reference(
        theta, velocity, detector, reliability, dp_solution
    )
    conditions = load_spec(resolve(args.spec))
    all_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for condition in conditions:
        critic_values = critic_runs(condition)
        for replicate, (actor_value, critic_value) in enumerate(
            zip(condition["actor_runs"], critic_values, strict=True)
        ):
            rows, identity = evaluate_pair(
                str(condition["name"]),
                replicate,
                resolve(actor_value),
                resolve(critic_value),
                theta,
                velocity,
                reference_returns,
                use_dp,
                float(reference_spec["epsilon_return"]),
                float(args.action_gap),
                dp_solution,
                args.device,
            )
            all_rows.extend(rows)
            identities.append(identity)
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "divergence_state_critic_rows.csv", all_rows)
    condition_summary = summarize(all_rows, identities)
    summary = {
        "schema_version": 1,
        "diagnostic_scope": (
            "Frozen-policy intervention at the first actor action that differs "
            "from its initial-state-selected reference by more than the fixed gap."
        ),
        "protocol": {
            "spec": str(resolve(args.spec).relative_to(ROOT)).replace("\\", "/"),
            "protocol": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "locked_dataset_sha256": str(dataset["sha256"]),
            "selected_indices_sha256": hashlib.sha256(
                selected.astype(np.int64).tobytes()
            ).hexdigest(),
            "sample_count_per_seed": int(args.sample_count),
            "sample_seed": int(args.sample_seed),
            "action_gap": float(args.action_gap),
            "horizon": 200,
            "reference_epsilon": float(reference_spec["epsilon_return"]),
            "dp_solution": str(dp_solution.relative_to(ROOT)).replace("\\", "/"),
            "dp_solution_sha256": sha256(dp_solution),
            "dp_solution_size_bytes": dp_solution.stat().st_size,
            "selected_dp_initial_states": int(use_dp.sum()),
            "selected_controller_initial_states": int((~use_dp).sum()),
            "reference_policy_selected_from_initial_state": True,
            "reference_access_during_inference": False,
            "authority_grid_used": False,
        },
        "conditions": condition_summary,
        "checkpoint_identities": identities,
        "limitations": [
            (
                "The reference is a diagnostic comparator and never changes the "
                "deployed policy or selects a training method."
            ),
            (
                "Conditioning on the first action disagreement and final outcome "
                "does not identify the training update that created the policy."
            ),
            (
                "A one-action substitution tests a local decision inside a longer "
                "closed-loop sequence and cannot replace the sequence intervention."
            ),
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    make_plot(output, condition_summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
