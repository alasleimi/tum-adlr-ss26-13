from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "week3_100k_component_ablation_20260527"
SUMMARY_CSV = REPORT_ROOT / "component_ablation_grid_summary.csv"
RELATIVE_ROOT = REPORT_ROOT / "relative_success_grid_61x41"
RUN_ROOT = ROOT / "runs" / "week3_100k_component_ablation_20260527"

OUT_DIR = REPORT_ROOT / "figures_ablation_vs_simbav2"
METRICS_PNG = OUT_DIR / "ablation_vs_simbav2_metrics.png"
MAPS_PNG = OUT_DIR / "ablation_vs_simbav2_start_state_delta_maps.png"
REPORT_MD = REPORT_ROOT / "ablation_vs_simbav2_report.md"

BASELINE = "simba_full_official_opt"
ABLATIONS = [
    "simba_full_no_feature_norm_official_opt",
    "simba_full_no_projection_official_opt",
    "simba_full_no_distributional_official_opt",
    "simba_full_no_reward_scaling_official_opt",
]

DISPLAY_NAME = {
    "simba_full_official_opt": "Full SimbaV2 (official opt, 100k)",
    "simba_full_no_feature_norm_official_opt": "Minus feature norm",
    "simba_full_no_projection_official_opt": "Minus weight projection",
    "simba_full_no_distributional_official_opt": "Minus distributional critic",
    "simba_full_no_reward_scaling_official_opt": "Minus reward scaling",
}

SHORT_NAME = {
    "simba_full_official_opt": "Full SimbaV2 (Official)",
    "simba_full_no_feature_norm_official_opt": "-Feature Norm",
    "simba_full_no_projection_official_opt": "-Projection",
    "simba_full_no_distributional_official_opt": "-Distributional",
    "simba_full_no_reward_scaling_official_opt": "-Reward Scaling",
}


@dataclass
class ConditionSummary:
    condition: str
    label: str
    seeds: str
    task_success_mean: float
    reference_success_mean: float
    near_down_reference_success: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def pp(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:+.1f} pp"


def load_summary() -> dict[str, ConditionSummary]:
    rows = read_rows(SUMMARY_CSV)
    by_condition: dict[str, ConditionSummary] = {}
    for row in rows:
        condition = row["condition"]
        by_condition[condition] = ConditionSummary(
            condition=condition,
            label=row["label"],
            seeds=row["seeds"],
            task_success_mean=parse_float(row["task_success_mean"]),
            reference_success_mean=parse_float(row["reference_success_mean"]),
            near_down_reference_success=parse_float(row["near_down_reference_success"]),
        )
    return by_condition


def load_relative_grid(condition: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = RELATIVE_ROOT / condition / "relative_cell_summary.csv"
    rows = read_rows(path)
    theta_vals = sorted({parse_float(row["theta_degrees"]) for row in rows})
    theta_dot_vals = sorted({parse_float(row["theta_dot"]) for row in rows})
    theta_to_idx = {v: i for i, v in enumerate(theta_vals)}
    theta_dot_to_idx = {v: i for i, v in enumerate(theta_dot_vals)}
    matrix = np.full((len(theta_dot_vals), len(theta_vals)), np.nan, dtype=float)
    for row in rows:
        x = parse_float(row["theta_degrees"])
        y = parse_float(row["theta_dot"])
        matrix[theta_dot_to_idx[y], theta_to_idx[x]] = parse_float(row[metric])
    return np.array(theta_vals), np.array(theta_dot_vals), matrix


def plot_metrics(summaries: dict[str, ConditionSummary]) -> None:
    conditions = [BASELINE] + ABLATIONS
    labels = [SHORT_NAME[c] for c in conditions]

    task_vals = [100.0 * summaries[c].task_success_mean for c in conditions]
    ref_vals = [100.0 * summaries[c].reference_success_mean for c in conditions]
    near_down_vals = [100.0 * summaries[c].near_down_reference_success for c in conditions]

    x = np.arange(len(conditions))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.bar(x - width, task_vals, width=width, label="Task success", color="#4f8a5b")
    ax.bar(x, ref_vals, width=width, label="Reference success", color="#376996")
    ax.bar(x + width, near_down_vals, width=width, label="Near-down reference", color="#c47a2c")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Percent")
    ax.set_title("Grid Reliability Metrics (Higher is better)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

    fig.suptitle("100k Ablations vs Full SimbaV2 Official Baseline")
    fig.savefig(METRICS_PNG, dpi=180)
    plt.close(fig)


def plot_start_state_delta_maps() -> None:
    metric_specs = [
        ("task_success_rate", "Task Success Delta"),
        ("near_best_known_return_eps_rate", "Reference Success Delta"),
    ]
    baseline_grids = {metric: load_relative_grid(BASELINE, metric) for metric, _ in metric_specs}

    rows = len(ABLATIONS)
    cols = len(metric_specs)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    # Use a symmetric color range per metric column.
    vmax_by_metric: dict[str, float] = {}
    for metric, _ in metric_specs:
        _, _, baseline = baseline_grids[metric]
        all_abs = []
        for condition in ABLATIONS:
            _, _, ablation = load_relative_grid(condition, metric)
            delta = ablation - baseline
            all_abs.append(np.nanmax(np.abs(delta)))
        vmax_by_metric[metric] = max(0.05, float(np.nanmax(all_abs)))

    for row_idx, condition in enumerate(ABLATIONS):
        for col_idx, (metric, title) in enumerate(metric_specs):
            theta, theta_dot, baseline = baseline_grids[metric]
            _, _, ablation = load_relative_grid(condition, metric)
            delta = ablation - baseline
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                delta,
                origin="lower",
                aspect="auto",
                extent=[float(theta.min()), float(theta.max()), float(theta_dot.min()), float(theta_dot.max())],
                cmap="RdBu_r",
                vmin=-vmax_by_metric[metric],
                vmax=vmax_by_metric[metric],
            )
            if row_idx == 0:
                ax.set_title(title)
            ax.set_ylabel(f"{SHORT_NAME[condition]}\nTheta dot")
            if row_idx == rows - 1:
                ax.set_xlabel("Theta (degrees)")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("Delta")

    fig.suptitle(
        "Initial-State Maps: Ablation minus Full SimbaV2 Baseline\n"
        "(positive = ablation better, negative = ablation worse)"
    )
    fig.savefig(MAPS_PNG, dpi=180)
    plt.close(fig)


def write_markdown(summaries: dict[str, ConditionSummary]) -> None:
    baseline = summaries[BASELINE]
    lines: list[str] = []
    lines.append("# 100k Ablations vs SimbaV2")
    lines.append("")
    lines.append("Weight projection doesn't seem to be important from the ablations; otherwise not interesting in the report.")
    lines.append("")
    lines.append(f"Generated on {date.today().isoformat()}.")
    lines.append("")
    lines.append("Baseline used for ablation deltas: **Full SimbaV2 official opt 100k (seeds 3+4)**.")
    lines.append("All values are from the exact 61x41 reset-support evaluation grid (2501 start states).")
    lines.append("")
    lines.append("## Aggregate Comparison")
    lines.append("")
    lines.append(
        "| Condition | Seeds | Task success | Reference success | Near-down reference | Delta task vs baseline | Delta reference | Delta near-down |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    ordered = [BASELINE] + ABLATIONS
    for condition in ordered:
        row = summaries[condition]
        delta_task = row.task_success_mean - baseline.task_success_mean
        delta_ref = row.reference_success_mean - baseline.reference_success_mean
        delta_near = row.near_down_reference_success - baseline.near_down_reference_success
        lines.append(
            f"| {DISPLAY_NAME[condition]} | {row.seeds} | {pct(row.task_success_mean)} | "
            f"{pct(row.reference_success_mean)} | {pct(row.near_down_reference_success)} | "
            f"{pp(delta_task)} | {pp(delta_ref)} | {pp(delta_near)} |"
        )

    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![Ablation metrics](figures_ablation_vs_simbav2/ablation_vs_simbav2_metrics.png)")
    lines.append("")
    lines.append(
        "![Ablation start-state delta maps](figures_ablation_vs_simbav2/ablation_vs_simbav2_start_state_delta_maps.png)"
    )
    lines.append("")
    lines.append("Map notes:")
    lines.append("- Left column: delta task-success rate per initial state.")
    lines.append("- Right column: delta reference-success rate (`near_best_known_return_eps_rate`) per initial state.")
    lines.append("- Positive values mean the ablation is better than the Full SimbaV2 official baseline at that start state.")
    lines.append("- Map baseline is `simba_full_official_opt` (seeds 3+4).")
    lines.append("")
    lines.append("## Ablation Definitions")
    lines.append("")
    lines.append("- **Minus feature norm**: keeps full official SimbaV2, but turns off feature normalization via `--simba-no-feature-norm`; features are passed unnormalized into the critic/actor stacks.")
    lines.append("- **Minus weight projection**: removes `--simba-weight-projection`; replaced by the standard Simba backbone path without projection, while keeping distributional critic and reward scaling.")
    lines.append("- **Minus distributional critic**: removes `--simba-distributional-critic` (and binning); replaced by the scalar Q critic path, while keeping backbone, projection, and reward scaling.")
    lines.append("- **Minus reward scaling**: removes `--simba-reward-scaling`; replaced by raw environment rewards (no Simba reward-scaling transform), while keeping backbone, projection, and distributional critic.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_summary()
    required = [BASELINE] + ABLATIONS
    missing = [condition for condition in required if condition not in summaries]
    if missing:
        raise KeyError(f"Missing conditions in summary CSV: {missing}")

    plot_metrics(summaries)
    plot_start_state_delta_maps()
    write_markdown(summaries)
    print(f"Wrote {METRICS_PNG}")
    print(f"Wrote {MAPS_PNG}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
