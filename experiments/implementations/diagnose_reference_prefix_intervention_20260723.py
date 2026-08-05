from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
try:
    from scripts.diagnose_first_divergence_20260723 import pendulum_obs, pendulum_step
except ModuleNotFoundError:
    from diagnose_first_divergence_20260723 import pendulum_obs, pendulum_step


ROOT = Path(__file__).resolve().parents[2]
EPISODE_STEPS = 200
ZERO_PREFIX_RETURN_ATOL = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many initial reference-controlled steps are needed to "
            "repair failures of a frozen actor."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_relative(path: Path) -> dict[int, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["actual_seed"]))].append(row)
    if len(grouped) != 5:
        raise ValueError(
            f"{path} contains {len(grouped)} actor seeds, expected five"
        )
    for seed, seed_rows in grouped.items():
        seed_rows.sort(key=lambda row: (float(row["theta_dot"]), float(row["theta"])))
        if len(seed_rows) != 2501:
            raise ValueError(f"{path} seed {seed} has {len(seed_rows)} rows")
    return dict(grouped)


def rollout_prefix(
    agent: Any,
    rows: list[dict[str, str]],
    *,
    prefix_steps: int,
    dp: PendulumReferenceGuidance,
    controller: PendulumReferenceGuidance,
    prefix_mode: str = "reference",
    shuffle_seed: int = 0,
) -> dict[str, np.ndarray]:
    theta = np.asarray([float(row["theta"]) for row in rows], dtype=np.float64)
    velocity = np.asarray([float(row["theta_dot"]) for row in rows], dtype=np.float64)
    use_dp = np.asarray(
        [
            float(row["dp_policy_return"]) >= float(row["controller_return"])
            for row in rows
        ],
        dtype=bool,
    )
    returns = np.zeros(len(rows), dtype=np.float64)
    near_count = np.zeros(len(rows), dtype=np.int64)
    current_not_near = np.zeros(len(rows), dtype=np.int64)
    longest_not_near = np.zeros(len(rows), dtype=np.int64)
    rng = np.random.default_rng(int(shuffle_seed))
    for step in range(EPISODE_STEPS):
        obs = pendulum_obs(theta, velocity)
        if step < prefix_steps:
            if prefix_mode == "zero":
                action = np.zeros(len(rows), dtype=np.float64)
            else:
                remaining = EPISODE_STEPS - step
                dp_action = np.asarray(
                    dp.act_batch(obs, remaining_steps=remaining), dtype=np.float64
                ).reshape(-1)
                controller_action = np.asarray(
                    controller.act_batch(obs, remaining_steps=remaining), dtype=np.float64
                ).reshape(-1)
                action = np.where(use_dp, dp_action, controller_action)
                if prefix_mode == "shuffled_reference":
                    action = action[rng.permutation(len(action))]
                elif prefix_mode != "reference":
                    raise ValueError(f"unsupported prefix mode: {prefix_mode}")
        else:
            action = np.asarray(
                agent.act_batch(obs, deterministic=True), dtype=np.float64
            ).reshape(-1)
        reward, theta, velocity = pendulum_step(theta, velocity, action)
        returns += reward
        near = (np.cos(theta) >= 0.95) & (np.abs(velocity) <= 1.0)
        near_count += near.astype(np.int64)
        current_not_near = np.where(near, 0, current_not_near + 1)
        longest_not_near = np.maximum(longest_not_near, current_not_near)
    task = (near_count / EPISODE_STEPS >= 0.8) & (longest_not_near <= 50)
    return {
        "return": returns,
        "task_success": task,
        "near_upright_fraction": near_count / EPISODE_STEPS,
        "longest_not_near_streak": longest_not_near,
    }


def condition_rows(
    condition: dict[str, Any],
    *,
    horizons: list[int],
    dp_solution: Path,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    relative_path = ROOT / str(condition["relative_rollouts"])
    relative = read_relative(relative_path)
    run_paths = [ROOT / str(value) for value in condition["actor_runs"]]
    if len(run_paths) != 5:
        raise ValueError("each condition requires five actor runs")
    dp = PendulumReferenceGuidance(
        policy="dp", dp_solution_path=dp_solution, horizon=EPISODE_STEPS
    )
    controller = PendulumReferenceGuidance(policy="controller", horizon=EPISODE_STEPS)
    aggregate: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for run_path in run_paths:
        agent, actor_config, _payload = load_agent_from_run(run_path, device=device)
        seed = int(actor_config.seed)
        rows = relative[seed]
        best = np.asarray([float(row["best_known_return"]) for row in rows])
        original_return = np.asarray([float(row["return"]) for row in rows])
        original_near = np.asarray(
            [float(row["near_best_known_return_eps"]) >= 0.5 for row in rows],
            dtype=bool,
        )
        by_horizon: dict[int, dict[str, np.ndarray]] = {}
        for horizon in horizons:
            by_horizon[horizon] = rollout_prefix(
                agent,
                rows,
                prefix_steps=horizon,
                dp=dp,
                controller=controller,
            )
        k0_error = float(
            np.max(np.abs(by_horizon[0]["return"] - original_return))
        )
        if k0_error > ZERO_PREFIX_RETURN_ATOL:
            raise ValueError(
                f"{condition['name']} seed {seed} k=0 return mismatch {k0_error}"
            )
        failure = ~original_near
        success = original_near
        first_repair = np.full(len(rows), -1, dtype=np.int64)
        for horizon in horizons:
            result = by_horizon[horizon]
            near = result["return"] >= best - 5.0
            eligible = failure & near & (first_repair < 0)
            first_repair[eligible] = horizon
            seed_row = summarize_slice(
                str(condition["name"]),
                seed,
                horizon,
                result,
                best,
                original_return,
                failure,
                success,
            )
            per_seed.append(seed_row)
        validation.append(
            {
                "condition": condition["name"],
                "seed": seed,
                "k0_return_max_abs_error": k0_error,
                "failure_trials": int(failure.sum()),
                "first_repair_histogram": {
                    str(horizon): int(np.sum(first_repair == horizon))
                    for horizon in horizons
                    if horizon > 0
                },
                "unrepaired_after_200": int(np.sum(failure & (first_repair < 0))),
            }
        )
    for horizon in horizons:
        horizon_rows = [row for row in per_seed if int(row["prefix_steps"]) == horizon]
        aggregate.append(pool_seed_rows(str(condition["name"]), horizon, horizon_rows))
    return aggregate, per_seed, validation


def specificity_controls(
    condition: dict[str, Any],
    *,
    dp_solution: Path,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relative = read_relative(ROOT / str(condition["relative_rollouts"]))
    dp = PendulumReferenceGuidance(
        policy="dp", dp_solution_path=dp_solution, horizon=EPISODE_STEPS
    )
    controller = PendulumReferenceGuidance(policy="controller", horizon=EPISODE_STEPS)
    per_seed: list[dict[str, Any]] = []
    for run_path_value in condition["actor_runs"]:
        agent, actor_config, _payload = load_agent_from_run(
            ROOT / str(run_path_value), device=device
        )
        seed = int(actor_config.seed)
        all_rows = relative[seed]
        rows = [
            row
            for row in all_rows
            if float(row["near_best_known_return_eps"]) < 0.5
        ]
        best = np.asarray([float(row["best_known_return"]) for row in rows])
        original_return = np.asarray([float(row["return"]) for row in rows])
        for prefix_steps in (1, 8, 16, 32):
            for mode in ("reference", "shuffled_reference", "zero"):
                result = rollout_prefix(
                    agent,
                    rows,
                    prefix_steps=prefix_steps,
                    dp=dp,
                    controller=controller,
                    prefix_mode=mode,
                    shuffle_seed=73_000 + 100 * seed + prefix_steps,
                )
                repaired = result["return"] >= best - 5.0
                per_seed.append(
                    {
                        "condition": condition["name"],
                        "seed": seed,
                        "prefix_mode": mode,
                        "prefix_steps": prefix_steps,
                        "failure_trials": len(rows),
                        "repaired_failures": int(repaired.sum()),
                        "repair_rate": float(np.mean(repaired)),
                        "return_gain_mean": float(
                            np.mean(result["return"] - original_return)
                        ),
                        "return_gain_median": float(
                            np.median(result["return"] - original_return)
                        ),
                    }
                )
    pooled: list[dict[str, Any]] = []
    for prefix_steps in (1, 8, 16, 32):
        for mode in ("reference", "shuffled_reference", "zero"):
            rows = [
                row
                for row in per_seed
                if int(row["prefix_steps"]) == prefix_steps
                and row["prefix_mode"] == mode
            ]
            trials = sum(int(row["failure_trials"]) for row in rows)
            repaired = sum(int(row["repaired_failures"]) for row in rows)
            pooled.append(
                {
                    "condition": condition["name"],
                    "prefix_mode": mode,
                    "prefix_steps": prefix_steps,
                    "failure_trials": trials,
                    "repaired_failures": repaired,
                    "repair_rate": repaired / trials,
                    "seed_repair_rate_min": min(float(row["repair_rate"]) for row in rows),
                    "seed_repair_rate_max": max(float(row["repair_rate"]) for row in rows),
                }
            )
    return pooled, per_seed


def summarize_slice(
    condition: str,
    seed: int,
    prefix_steps: int,
    result: dict[str, np.ndarray],
    best: np.ndarray,
    original_return: np.ndarray,
    failure: np.ndarray,
    success: np.ndarray,
) -> dict[str, Any]:
    near = result["return"] >= best - 5.0
    strict = result["return"] > best
    gain = result["return"] - original_return
    return {
        "condition": condition,
        "seed": seed,
        "prefix_steps": prefix_steps,
        "trials": len(best),
        "original_failure_trials": int(failure.sum()),
        "original_success_trials": int(success.sum()),
        "near_successes": int(near.sum()),
        "task_successes": int(result["task_success"].sum()),
        "strict_successes": int(strict.sum()),
        "repaired_original_failures": int(np.sum(failure & near)),
        "broken_original_successes": int(np.sum(success & ~near)),
        "failure_gain_mean": float(np.mean(gain[failure])) if failure.any() else 0.0,
        "failure_gain_median": float(np.median(gain[failure])) if failure.any() else 0.0,
    }


def pool_seed_rows(
    condition: str, prefix_steps: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    keys = (
        "trials",
        "original_failure_trials",
        "original_success_trials",
        "near_successes",
        "task_successes",
        "strict_successes",
        "repaired_original_failures",
        "broken_original_successes",
    )
    pooled = {key: sum(int(row[key]) for row in rows) for key in keys}
    pooled.update(
        {
            "condition": condition,
            "prefix_steps": prefix_steps,
            "near_rate": pooled["near_successes"] / pooled["trials"],
            "task_rate": pooled["task_successes"] / pooled["trials"],
            "strict_rate": pooled["strict_successes"] / pooled["trials"],
            "repair_rate": (
                pooled["repaired_original_failures"]
                / max(pooled["original_failure_trials"], 1)
            ),
            "break_rate": (
                pooled["broken_original_successes"]
                / max(pooled["original_success_trials"], 1)
            ),
            "seed_repair_rate_min": min(
                row["repaired_original_failures"]
                / max(row["original_failure_trials"], 1)
                for row in rows
            ),
            "seed_repair_rate_max": max(
                row["repaired_original_failures"]
                / max(row["original_failure_trials"], 1)
                for row in rows
            ),
        }
    )
    return pooled


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, aggregate: list[dict[str, Any]]) -> None:
    colors = {"pure RL actor": "#2563EB", "mixed supervised actor": "#0F766E"}
    conditions = list(dict.fromkeys(str(row["condition"]) for row in aggregate))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), constrained_layout=True)
    for condition in conditions:
        rows = [row for row in aggregate if row["condition"] == condition]
        x = np.arange(len(rows))
        labels = [str(row["prefix_steps"]) for row in rows]
        color = colors.get(condition, "#334155")
        axes[0].plot(
            x, [100 * row["near_rate"] for row in rows],
            marker="o", linewidth=2.2, color=color, label=condition,
        )
        axes[1].plot(
            x, [100 * row["repair_rate"] for row in rows],
            marker="o", linewidth=2.2, color=color, label=condition,
        )
        axes[2].plot(
            x, [100 * row["task_rate"] for row in rows],
            marker="o", linewidth=2.2, color=color, label=condition,
        )
    for ax in axes:
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_xlabel("Initial steps controlled by stored reference")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("A. Near-reference success", loc="left", fontweight="bold")
    axes[0].set_ylabel("All seed-state trials (%)")
    axes[1].set_title("B. Original failures repaired", loc="left", fontweight="bold")
    axes[1].set_ylabel("Original actor failures (%)")
    axes[2].set_title("C. Task success", loc="left", fontweight="bold")
    axes[2].set_ylabel("All seed-state trials (%)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Reference-prefix intervention localizes how long correction must persist",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_specificity(path: Path, pooled: list[dict[str, Any]]) -> None:
    labels = {
        "reference": "state-matched reference",
        "shuffled_reference": "state-shuffled reference actions",
        "zero": "zero torque",
    }
    colors = {
        "reference": "#2563EB",
        "shuffled_reference": "#D97706",
        "zero": "#64748B",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for mode in ("reference", "shuffled_reference", "zero"):
        rows = [row for row in pooled if row["prefix_mode"] == mode]
        ax.plot(
            [row["prefix_steps"] for row in rows],
            [100 * row["repair_rate"] for row in rows],
            marker="o",
            linewidth=2.3,
            color=colors[mode],
            label=labels[mode],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
    ax.set_xlabel("Initial prefix steps")
    ax.set_ylabel("Original pure-actor failures repaired (%)")
    ax.set_title(
        "Does recovery require a state-matched corrective sequence?",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown(
    aggregate: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    specificity: list[dict[str, Any]],
) -> str:
    lines = [
        "# Reference-prefix intervention",
        "",
        "This frozen diagnostic gives the stored reference control only for an initial prefix, then returns control to the same actor. It is not a deployment method.",
        "",
        "| Condition | Prefix | Near reference | Task success | Original failures repaired | Original successes broken |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['condition']} | {row['prefix_steps']} | "
            f"{row['near_successes']}/{row['trials']} ({100*row['near_rate']:.2f}%) | "
            f"{row['task_successes']}/{row['trials']} ({100*row['task_rate']:.2f}%) | "
            f"{row['repaired_original_failures']}/{row['original_failure_trials']} "
            f"({100*row['repair_rate']:.2f}%) | "
            f"{row['broken_original_successes']}/{row['original_success_trials']} "
            f"({100*row['break_rate']:.3f}%) |"
        )
    lines.extend(
        [
            "",
            (
                "## Specificity control on the "
                f"{specificity[0]['failure_trials']:,} pure-actor failures"
            ),
            "",
            "| Prefix | State-matched reference | Shuffled reference actions | Zero torque |",
            "|---:|---:|---:|---:|",
        ]
    )
    for prefix_steps in (1, 8, 16, 32):
        by_mode = {
            row["prefix_mode"]: row
            for row in specificity
            if int(row["prefix_steps"]) == prefix_steps
        }
        lines.append(
            f"| {prefix_steps} | "
            f"{by_mode['reference']['repaired_failures']}/"
            f"{by_mode['reference']['failure_trials']} "
            f"({100*by_mode['reference']['repair_rate']:.2f}%) | "
            f"{by_mode['shuffled_reference']['repaired_failures']}/"
            f"{by_mode['shuffled_reference']['failure_trials']} "
            f"({100*by_mode['shuffled_reference']['repair_rate']:.2f}%) | "
            f"{by_mode['zero']['repaired_failures']}/"
            f"{by_mode['zero']['failure_trials']} "
            f"({100*by_mode['zero']['repair_rate']:.2f}%) |"
        )
    control_by_horizon = {
        int(row["prefix_steps"]): row
        for row in specificity
        if row["prefix_mode"] == "reference"
    }
    shuffled_by_horizon = {
        int(row["prefix_steps"]): row
        for row in specificity
        if row["prefix_mode"] == "shuffled_reference"
    }
    zero_by_horizon = {
        int(row["prefix_steps"]): row
        for row in specificity
        if row["prefix_mode"] == "zero"
    }
    h16 = (
        control_by_horizon[16],
        shuffled_by_horizon[16],
        zero_by_horizon[16],
    )
    h32 = (
        control_by_horizon[32],
        shuffled_by_horizon[32],
        zero_by_horizon[32],
    )
    lines.extend(
        [
            "",
            (
                "At 16 steps the state-matched sequence repairs "
                f"{100*h16[0]['repair_rate']:.2f}% of failures, compared "
                f"with {100*h16[1]['repair_rate']:.2f}% for shuffled "
                f"reference actions and {100*h16[2]['repair_rate']:.2f}% "
                "for zero torque. At 32 steps the rates are "
                f"{100*h32[0]['repair_rate']:.2f}%, "
                f"{100*h32[1]['repair_rate']:.2f}%, and "
                f"{100*h32[2]['repair_rate']:.2f}%."
            ),
            "",
            "The intervention uses the better stored DP or controller trajectory for each initial state. The same solver controls the prefix at the correct remaining horizon. Reference returns are used for scoring only after each rollout.",
            "",
            "Limitations: this is a post-selection counterfactual on the standardized authority grid. It identifies the duration of corrective control under a strong reference, but it does not identify the training update that created the actor error.",
            "",
            f"All {len(validation)} actor-seed validations reproduce the stored zero-prefix return to within {ZERO_PREFIX_RETURN_ATOL:g}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported diagnostic config schema")
    horizons = [int(value) for value in config["horizons"]]
    if horizons != [0, 1, 2, 4, 8, 16, 32, 64, 200]:
        raise ValueError("diagnostic horizons differ from the registered sequence")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    dp_solution = ROOT / str(config["dp_solution"])
    for condition in config["conditions"]:
        condition_aggregate, condition_seed, condition_validation = condition_rows(
            condition,
            horizons=horizons,
            dp_solution=dp_solution,
            device=args.device,
        )
        aggregate.extend(condition_aggregate)
        per_seed.extend(condition_seed)
        validation.extend(condition_validation)
    pure_condition = next(
        condition
        for condition in config["conditions"]
        if condition["name"] == "pure RL actor"
    )
    specificity, specificity_per_seed = specificity_controls(
        pure_condition,
        dp_solution=dp_solution,
        device=args.device,
    )
    payload = {
        "schema_version": 1,
        "status": "post_selection_diagnostic_only",
        "authority_grid_used": True,
        "selection_performed": False,
        "training_performed": False,
        "reference_available_at_deployment": False,
        "horizons": horizons,
        "aggregate": aggregate,
        "specificity_controls": specificity,
        "validation": validation,
        "limitations": [
            "The standardized authority grid is used, so this evidence cannot select a policy.",
            "A reference prefix is a diagnostic intervention and is not available at deployment.",
            "The result localizes corrective duration but does not identify a causal optimizer mechanism.",
            "Strict-win counts are retained in raw aggregates but are not interpreted because a full reference prefix reproduces the scoring reference up to small numerical rollout differences.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(output / "aggregate.csv", aggregate)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "specificity_control.csv", specificity)
    write_csv(output / "specificity_control_per_seed.csv", specificity_per_seed)
    (output / "summary.md").write_text(
        markdown(aggregate, validation, specificity), encoding="utf-8"
    )
    plot(output / "reference_prefix_intervention.png", aggregate)
    plot_specificity(output / "reference_prefix_specificity.png", specificity)
    print(json.dumps(aggregate, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
