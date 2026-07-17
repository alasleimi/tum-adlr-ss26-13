from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.pendulum_relative import run_relative_report


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DP_GRID = ROOT / "reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv"
CONTROLLER_GRID = ROOT / "reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_grid.csv"


NEW_GRID_PATHS = [
    ROOT / "reports/large_simba_corrected_dagger100k_seed0_20260718/seed0/grid/pendulum_grid_rollouts.csv",
    *[
        ROOT
        / f"reports/large_simba_corrected_dagger100k_5seed_20260718/seed{seed}/grid/pendulum_grid_rollouts.csv"
        for seed in range(1, 5)
    ],
]

COMPARATORS = {
    "Static distillation\n(1 seed)": ROOT
    / "reports/distill_best_simbav2_balanced_400k_20260701/relative_success_vellim1/relative_rollouts.csv",
    "Normal SimbaV2 100k\n(5 seeds)": ROOT
    / "reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_rollouts.csv",
    "Best pure RL router\n(5 seeds)": ROOT
    / "reports/pure_rl_initial_state_qfilter_router_5seed_20260717/relative/relative_rollouts.csv",
    "Clean small DAgger\n(5 seeds)": ROOT
    / "reports/reference_specialist_dagger_explicit_5seed_20260718/relative/relative_rollouts.csv",
    "Prior RL+supervised\ninteraction overlay": ROOT
    / "reports/reference_assisted_nonzero_rl_overlay_5seed_20260718/relative/relative_rollouts.csv",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], label: str) -> dict[str, object]:
    total = len(rows)
    near = sum(float(row["return"]) >= float(row["best_known_return"]) - 5.0 for row in rows)
    task = sum(float(row["task_success"]) for row in rows)
    strict = sum(float(row["return"]) > float(row["best_known_return"]) for row in rows)
    ties = sum(float(row["return"]) == float(row["best_known_return"]) for row in rows)
    near_down = [row for row in rows if abs(float(row["theta_degrees"])) >= 150.0]
    near_down_near = sum(
        float(row["return"]) >= float(row["best_known_return"]) - 5.0 for row in near_down
    )
    near_down_task = sum(float(row["task_success"]) for row in near_down)
    return {
        "method": label,
        "trials": total,
        "near_reference_successes": near,
        "near_reference_rate": near / total,
        "task_successes": int(task),
        "task_success_rate": task / total,
        "strict_return_wins": strict,
        "strict_return_win_rate": strict / total,
        "exact_return_ties": ties,
        "mean_return": float(np.mean([float(row["return"]) for row in rows])),
        "near_down_trials": len(near_down),
        "near_down_near_reference_rate": near_down_near / len(near_down),
        "near_down_task_success_rate": near_down_task / len(near_down),
    }


def fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build() -> None:
    new_grid_rows: list[dict[str, str]] = []
    for path in NEW_GRID_PATHS:
        new_grid_rows.extend(read_rows(path))
    combined_grid = OUT / "grid/pendulum_grid_rollouts.csv"
    write_rows(combined_grid, new_grid_rows)
    run_relative_report(
        sac_rollouts_path=combined_grid,
        dp_grid_path=DP_GRID,
        controller_grid_path=CONTROLLER_GRID,
        output_dir=OUT / "relative",
        condition_label="large_simba_corrected_dagger100k_5seed",
        epsilon_return=5.0,
        reliability=ReliabilityConfig(),
    )
    new_relative = read_rows(OUT / "relative/relative_rollouts.csv")
    summaries = [summarize(new_relative, "Large SimbaV2 + corrected 100k DAgger")]
    for label, path in COMPARATORS.items():
        summaries.append(summarize(read_rows(path), label.replace("\n", " ")))
    write_rows(OUT / "exact_metrics.csv", summaries)

    per_seed_rows: list[dict[str, object]] = []
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in new_relative:
        grouped[int(float(row["actual_seed"]))].append(row)
    for seed, rows in sorted(grouped.items()):
        item = summarize(rows, f"seed{seed}")
        item = {"seed": seed, **{key: value for key, value in item.items() if key != "method"}}
        per_seed_rows.append(item)
    write_rows(OUT / "per_seed_metrics.csv", per_seed_rows)

    order = [
        "Normal SimbaV2 100k (5 seeds)",
        "Best pure RL router (5 seeds)",
        "Clean small DAgger (5 seeds)",
        "Prior RL+supervised interaction overlay",
        "Large SimbaV2 + corrected 100k DAgger",
    ]
    by_label = {str(row["method"]): row for row in summaries}
    plot_rows = [by_label[label] for label in order]
    labels = [
        "Normal\nSimbaV2",
        "Best\npure RL",
        "Clean\nDAgger",
        "Prior\nhybrid",
        "New single\nactor",
    ]
    colors = ["#8c8c8c", "#6d8cbf", "#62a87c", "#d28b36", "#8b5fbf"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    specs = [
        ("near_reference_rate", "Near reference", (0.90, 1.002)),
        ("task_success_rate", "Task success", (0.90, 0.96)),
        ("strict_return_win_rate", "Strict return > reference", (0.05, 0.13)),
    ]
    for axis, (key, title, limits) in zip(axes, specs):
        values = [float(row[key]) for row in plot_rows]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (limits[1] - limits[0]) * 0.012,
                f"{100 * value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Authoritative 61×41 Pendulum grid", fontsize=14)
    fig.tight_layout()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "plots/comparison_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    new = summaries[0]
    hybrid = next(row for row in summaries if row["method"] == "Prior RL+supervised interaction overlay")
    clean = next(row for row in summaries if row["method"] == "Clean small DAgger (5 seeds)")
    seed_lines = "\n".join(
        f"| {row['seed']} | {int(row['near_reference_successes']):,}/2,501 "
        f"({100 * float(row['near_reference_rate']):.4f}%) | "
        f"{int(row['task_successes']):,}/2,501 ({100 * float(row['task_success_rate']):.4f}%) | "
        f"{int(row['strict_return_wins']):,}/2,501 ({100 * float(row['strict_return_win_rate']):.4f}%) |"
        for row in per_seed_rows
    )
    compare_lines = "\n".join(
        f"| {row['method']} | {int(row['trials']):,} | "
        f"{int(row['near_reference_successes']):,} ({100 * float(row['near_reference_rate']):.4f}%) | "
        f"{int(row['task_successes']):,} ({100 * float(row['task_success_rate']):.4f}%) | "
        f"{int(row['strict_return_wins']):,} ({100 * float(row['strict_return_win_rate']):.4f}%) | "
        f"{float(row['mean_return']):.6f} |"
        for row in summaries
    )
    report = f"""# Large SimbaV2 + corrected 100k DAgger: five-seed result

Date: 2026-07-18

## Outcome

The primary goal is achieved. The new method is a **single actor checkpoint at inference** and reaches **{int(new['near_reference_successes']):,}/{int(new['trials']):,} = {100 * float(new['near_reference_rate']):.4f}% near-reference success**. The previous RL+supervised interaction overlay reaches {int(hybrid['near_reference_successes']):,}/{int(hybrid['trials']):,} = {100 * float(hybrid['near_reference_rate']):.4f}%.

The new actor therefore adds **{int(new['near_reference_successes']) - int(hybrid['near_reference_successes'])}** near-reference successes. It also adds **{int(new['strict_return_wins']) - int(hybrid['strict_return_wins'])}** literal strict return wins and improves mean return by **{float(new['mean_return']) - float(hybrid['mean_return']):.6f}**. It does **not** beat the hybrid on task success: it loses **{int(hybrid['task_successes']) - int(new['task_successes'])}** task-success trials.

No result below uses `>=` for the strict metric. A strict win is computed directly as:

```text
policy_return > max(DP_return, controller_return)
```

There are {int(new['exact_return_ties'])} exact return ties in the new result.

## Exact authoritative evaluation

- Environment: `Pendulum-v1`
- Horizon: 200 deterministic steps
- Initial states per seed: 61 angles in `[-pi, pi)` × 41 velocities in `[-1, 1]` = 2,501
- Follow-up seeds: 0, 1, 2, 3, 4
- Total trials: 12,505
- Near-reference success: return at least `max(DP, controller) - 5`
- Task success: at least 80% of steps near upright and no non-upright streak longer than 50 steps

| Method | Trials | Near reference | Task success | Strict return > reference | Mean return |
|---|---:|---:|---:|---:|---:|
{compare_lines}

The static-distillation row has one training seed and is included only as a historical comparator. The other named aggregate rows have five seeds.

## New result by follow-up seed

| Seed | Near reference | Task success | Strict return > reference |
|---:|---:|---:|---:|
{seed_lines}

## What was trained

The deployed object is one SimbaV2 actor network:

- observation: `[cos(theta), sin(theta), theta_dot]`
- action: one bounded torque in `[-2, 2]`
- hidden width: 64
- residual blocks: 2
- SimbaV2 feature normalization, input shift, observation normalization, and weight projection remain part of the actor implementation
- no critic, policy gate, action router, specialist switch, or Q-search is called at inference

The initialization is `runs/reference_assisted_large_simba_reset60_dagger3_seed0_20260717/seed0/checkpoints/final.pt`. That checkpoint was trained from scratch with 400,000 static max(DP, controller) labels, 80 supervised epochs, then three learner-only DAgger rounds of 10,000 visited states each and 10 training epochs per round. Its original one-seed fine-grid result was 99.2003% near reference.

Each new follow-up seed then ran this exact additional recipe:

1. Generate a fresh 240,000-state supervised support set: 20% broad states, 50% reset-support states, 15% `120° <= |theta| <= 135°` states, and 15% near-down states.
2. Label every support state with the finite-horizon best reference, `max(DP, controller)`.
3. Use the seed-matched fixed pure-RL critics to score 41 actions in `[-2, 2]` and the reference action. Mark a state only when **every critic** estimates that the best searched action exceeds the reference action by more than `0.05` Q units.
4. Keep the supervised action target equal to the reference action (`rl_blend=0`), but give marked samples 4× weight in the supervised MSE. Thus RL changes which reference labels receive extra weight; it does not replace the reference label.
5. Run five DAgger rounds. Each round contains 100 learner episodes × 200 steps = 20,000 environment steps.
6. At every DAgger step, save the current state, execute the learned actor action in `env.step()`, and ask the best reference to label the saved state at the correct remaining horizon.
7. Aggregate all new state/reference-action pairs with the support set and apply the same critic-based weighting rule.
8. Train the full 64×2 actor with batch size 1,024, learning rate `1e-5`, and three epochs per DAgger round.
9. Select one actor iterate on the disjoint 17×11 midpoint validation grid, ordered by near-reference success, then task success, then mean return.

The experiment collects exactly 100,000 learner-visited DAgger states per seed. In all five seeds the selected policy is epoch 1 after round 1, so the returned policy was directly updated using the 240,000-state support set plus the first 20,000 DAgger states. The remaining four rounds are still part of the 100k policy sequence and validation selection, but their later actors were rejected because validation near-reference success regressed. This is policy selection from the DAgger iterate sequence, not a claim that the chosen weights consumed all 100,000 labels.

## What happened to the proposed RL cross-pollination

The best pure-RL learning recipe was tested directly before this winner was promoted:

- the corrected DAgger actor initialized the online actor;
- the pure-RL FastSACN8 critic ensemble and UTD 2 update schedule learned from real reward;
- hard-boundary replay used the `120°–135°`, `|theta_dot| <= 1` region;
- max(DP, controller) behavior cloning remained an auxiliary actor loss;
- only one actor was used at inference.

That seed-0 joint checkpoint improved held-out task success, strict wins, and mean return, but its authoritative fine-grid near-reference rate fell to 98.4006%. It was rejected. Conservative fixed-critic target blending, model-differentiated reward updates, hard-only DAgger, and critic-filtered specialist distillation were also rejected by held-out or fine-grid checks.

The winner uses the **larger SimbaV2 network backbone**, corrected DAgger, and **training-only weighting from the best pure-RL critics**. In the training command, `rl_blend=0`, so the critics never shift the reference action target. However, `selected_weight=4` means their unanimous filter gives selected reference-labeled samples four times the loss weight. The correct description is therefore **RL-weighted supervised DAgger**, not pure DAgger and not a runtime actor/critic mixture.

For the selected epoch, the number of 4×-weighted samples in the 240,000-state support set plus first 20,000-state DAgger round was: seed 0, 3,246; seed 1, 795; seed 2, 3,405; seed 3, 2,182; seed 4, 419. Their target actions remained the best-reference actions.

## Trade-offs

Compared with the previous runtime overlay:

- near reference: **+{int(new['near_reference_successes']) - int(hybrid['near_reference_successes'])} trials**
- literal strict wins: **+{int(new['strict_return_wins']) - int(hybrid['strict_return_wins'])} trials**
- mean return: **{float(new['mean_return']) - float(hybrid['mean_return']):+.6f}**
- task success: **{int(new['task_successes']) - int(hybrid['task_successes'])} trials**
- near-down near reference: **{100 * (float(new['near_down_near_reference_rate']) - float(hybrid['near_down_near_reference_rate'])):+.4f} percentage points**
- near-down task success: **{100 * (float(new['near_down_task_success_rate']) - float(hybrid['near_down_task_success_rate'])):+.4f} percentage points**

This is a clear win on the declared near-reference objective and on strict return wins, not a Pareto win over every metric.

## Plots

![Comparison metrics](plots/comparison_metrics.png)

![Near-reference success over initial states](relative/near_best_known_return_eps_map.png)

![Task success over initial states](relative/task_success_map.png)

![Strict return wins over initial states](relative/beats_best_known_return_map.png)

## Reproducibility paths

- Seed checkpoints: `runs/large_simba_corrected_dagger100k_seed0_20260718/seed0` through `runs/large_simba_corrected_dagger100k_seed4_20260718/seed4`
- Aggregate grid: `reports/large_simba_corrected_dagger100k_5seed_20260718/grid/pendulum_grid_rollouts.csv`
- Aggregate relative rollouts: `reports/large_simba_corrected_dagger100k_5seed_20260718/relative/relative_rollouts.csv`
- Exact aggregate metrics: `reports/large_simba_corrected_dagger100k_5seed_20260718/exact_metrics.csv`
- Per-seed metrics: `reports/large_simba_corrected_dagger100k_5seed_20260718/per_seed_metrics.csv`
- Training implementation used for the winner: `scripts/train_pendulum_qregularized_dagger.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "build_summary.json").write_text(
        json.dumps(
            {
                "new": new,
                "hybrid": hybrid,
                "clean": clean,
                "delta_new_minus_hybrid": {
                    "near_reference_successes": int(new["near_reference_successes"])
                    - int(hybrid["near_reference_successes"]),
                    "task_successes": int(new["task_successes"]) - int(hybrid["task_successes"]),
                    "strict_return_wins": int(new["strict_return_wins"])
                    - int(hybrid["strict_return_wins"]),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
