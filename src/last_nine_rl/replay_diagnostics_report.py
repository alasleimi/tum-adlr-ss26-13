from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MetricSeries = dict[tuple[str, str], dict[int, float]]

EVAL_KEYS = (
    "mean_return",
    "worst_return",
    "return_success_rate",
    "strict_success_rate",
    "near_upright_fraction",
    "max_not_near_upright_streak",
)
REPLAY_KEYS = (
    "size",
    "fill_fraction",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "near_upright_obs_fraction",
    "near_upright_next_obs_fraction",
    "near_upright_any_transition_fraction",
    "action_abs_mean",
    "action_saturation_fraction",
    "sample_count_mean",
    "sample_count_max",
    "transition_age_mean",
    "transition_age_max",
)
DIAGNOSTIC_KEYS = (
    "actor_layer0_dormant_fraction",
    "actor_layer1_dormant_fraction",
    "actor_layer0_effective_rank_fraction",
    "actor_layer1_effective_rank_fraction",
    "q1_layer0_dormant_fraction",
    "q1_layer1_dormant_fraction",
    "q1_layer0_effective_rank_fraction",
    "q1_layer1_effective_rank_fraction",
    "q2_layer0_dormant_fraction",
    "q2_layer1_dormant_fraction",
    "q2_layer0_effective_rank_fraction",
    "q2_layer1_effective_rank_fraction",
    "actor_param_norm",
    "q1_param_norm",
    "q2_param_norm",
    "alpha",
)
UPDATE_KEYS = (
    "q_loss_mean",
    "q_grad_norm_mean",
    "q_grad_norm_max",
    "q_update_norm_ratio_mean",
    "actor_loss_mean",
    "actor_grad_norm_mean",
    "actor_update_norm_ratio_mean",
    "alpha_mean",
    "policy_entropy_estimate_mean",
)
SUMMARY_KEYS = (
    "eval_mean_return",
    "eval_return_success_rate",
    "eval_strict_success_rate",
    "replay_fill_fraction",
    "replay_reward_mean",
    "replay_near_upright_any_transition_fraction",
    "replay_action_saturation_fraction",
    "replay_sample_count_mean",
    "replay_sample_count_max",
    "replay_transition_age_mean",
    "diagnostics_q1_layer1_dormant_fraction",
    "diagnostics_q2_layer1_dormant_fraction",
    "diagnostics_q1_layer1_effective_rank_fraction",
    "diagnostics_q2_layer1_effective_rank_fraction",
    "diagnostics_alpha",
    "update_q_grad_norm_mean",
    "update_q_update_norm_ratio_mean",
    "update_actor_update_norm_ratio_mean",
)
T_CRITICAL_95 = {
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
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
BOUNDED_0_1_KEY_PARTS = (
    "fraction",
    "success_rate",
    "fill_fraction",
    "saturation_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare replay, evaluation, and representation telemetry.")
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Condition label and run root. May be passed multiple times.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    conditions = parse_conditions(args.condition)
    output_dir = Path(args.out)
    result = write_replay_diagnostics_report(conditions, output_dir)
    print(json.dumps(result["summary"], allow_nan=False, indent=2, sort_keys=True))


def parse_conditions(items: list[str]) -> list[tuple[str, Path]]:
    conditions: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Condition must use LABEL=PATH format: {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Condition label cannot be empty: {item}")
        conditions.append((label, Path(path)))
    return conditions


def write_replay_diagnostics_report(conditions: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [run for label, root in conditions for run in load_condition_runs(label, root)]
    if not runs:
        raise ValueError("No runs with metrics.csv found.")

    snapshot_rows = [snapshot_row(run) for run in runs]
    time_rows = [row for run in runs for row in telemetry_time_rows(run)]
    summary = summarize_conditions(snapshot_rows)

    write_csv(output_dir / "replay_diagnostics_snapshot.csv", snapshot_rows)
    write_csv(output_dir / "replay_diagnostics_timeseries.csv", time_rows)
    (output_dir / "replay_diagnostics_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figures = write_figures(output_dir, runs)
    write_index(output_dir, figures, summary)
    return {"runs": runs, "snapshot_rows": snapshot_rows, "time_rows": time_rows, "summary": summary}


def load_condition_runs(label: str, root: Path) -> list[dict[str, Any]]:
    return [
        load_run(label, run_dir)
        for run_dir in expand_run_dirs(root)
        if (run_dir / "metrics.csv").exists()
    ]


def expand_run_dirs(root: Path) -> list[Path]:
    if (root / "metrics.csv").exists():
        return [root]
    return sorted({path.parent for path in root.glob("**/metrics.csv") if path.is_file()})


def load_run(condition: str, run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    events = read_events(run_dir / "events.jsonl")
    return {
        "condition": condition,
        "run_dir": run_dir,
        "seed": int(config.get("seed", seed_from_name(run_dir.name))),
        "complete": any(event.get("type") == "run_complete" for event in events),
        "metrics": read_metrics(run_dir / "metrics.csv"),
    }


def read_metrics(path: Path) -> MetricSeries:
    out: MetricSeries = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault((row["split"], row["name"]), {})[int(float(row["step"]))] = float(row["value"])
    return out


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit() or ch == "-")
    return int(digits) if digits not in {"", "-"} else 0


def snapshot_row(run: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition": run["condition"],
        "seed": run["seed"],
        "run_dir": str(run["run_dir"]),
        "complete": int(run["complete"]),
    }
    for split, keys in (
        ("eval", EVAL_KEYS),
        ("replay", REPLAY_KEYS),
        ("diagnostics", DIAGNOSTIC_KEYS),
        ("update", UPDATE_KEYS),
    ):
        step, values = last_split(run["metrics"], split)
        row[f"last_{split}_step"] = step if step is not None else ""
        for key in keys:
            row[f"{split}_{key}"] = values.get(key, math.nan)
    return row


def last_split(metrics: MetricSeries, split: str) -> tuple[int | None, dict[str, float]]:
    steps = sorted({step for (item_split, _name), series in metrics.items() if item_split == split for step in series})
    if not steps:
        return None, {}
    step = steps[-1]
    return step, values_at_step(metrics, split, step)


def values_at_step(metrics: MetricSeries, split: str, step: int) -> dict[str, float]:
    return {
        name: series[step]
        for (item_split, name), series in metrics.items()
        if item_split == split and step in series
    }


def telemetry_time_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    steps = sorted(
        {
            step
            for split, name in run["metrics"]
            if split in {"eval", "replay", "diagnostics", "update"}
            for step in run["metrics"][(split, name)]
        }
    )
    rows: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {
            "condition": run["condition"],
            "seed": run["seed"],
            "run_dir": str(run["run_dir"]),
            "complete": int(run["complete"]),
            "step": step,
        }
        for split, keys in (
            ("eval", EVAL_KEYS),
            ("replay", REPLAY_KEYS),
            ("diagnostics", DIAGNOSTIC_KEYS),
            ("update", UPDATE_KEYS),
        ):
            values = values_at_step(run["metrics"], split, step)
            for key in keys:
                if key in values:
                    row[f"{split}_{key}"] = values[key]
        rows.append(row)
    return rows


def summarize_conditions(snapshot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in snapshot_rows}):
        rows = [row for row in snapshot_rows if row["condition"] == condition]
        complete_rows = [row for row in rows if int(row["complete"]) == 1]
        final_rows = complete_rows or rows
        item: dict[str, Any] = {
            "num_runs": len(rows),
            "num_complete_runs": len(complete_rows),
            "complete_seeds": [int(row["seed"]) for row in complete_rows],
            "incomplete_seeds": [int(row["seed"]) for row in rows if int(row["complete"]) == 0],
        }
        for key in SUMMARY_KEYS:
            values = finite_values(final_rows, key)
            if values.size:
                add_descriptive_stats(item, key, values)
        summary[condition] = item
    return summary


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, math.nan)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            values.append(value_float)
    return np.asarray(values, dtype=np.float64)


def add_descriptive_stats(item: dict[str, Any], key: str, values: np.ndarray) -> None:
    mean, ci_low, ci_high = mean_ci95(values)
    bounds = metric_bounds(key)
    if bounds is not None:
        ci_low = max(bounds[0], ci_low)
        ci_high = min(bounds[1], ci_high)
    item[f"{key}_n"] = int(values.size)
    item[f"{key}_mean"] = mean
    item[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    item[f"{key}_sem"] = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    item[f"{key}_ci95_low"] = ci_low
    item[f"{key}_ci95_high"] = ci_high
    item[f"{key}_min"] = float(np.min(values))
    item[f"{key}_max"] = float(np.max(values))


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / math.sqrt(values.size))
    df = values.size - 1
    t_value = T_CRITICAL_95.get(df, 1.96)
    margin = t_value * sem
    return mean, mean - margin, mean + margin


def metric_bounds(key: str) -> tuple[float, float] | None:
    if any(part in key for part in BOUNDED_0_1_KEY_PARTS):
        return 0.0, 1.0
    return None


def write_figures(output_dir: Path, runs: list[dict[str, Any]]) -> list[Path]:
    return [
        plot_replay_eval_over_steps(runs, output_dir / "replay_eval_over_steps.png"),
        plot_replay_vs_eval_scatter(runs, output_dir / "replay_coverage_vs_eval_success.png"),
        plot_representation_health(runs, output_dir / "representation_health.png"),
        plot_optimizer_health(runs, output_dir / "optimizer_health.png"),
    ]


def plot_replay_eval_over_steps(runs: list[dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    plot_condition_metric(axes[0], runs, ("replay", "near_upright_any_transition_fraction"), "Replay near-upright fraction")
    plot_condition_metric(axes[1], runs, ("eval", "strict_success_rate"), "Eval strict-threshold diagnostic")
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_replay_vs_eval_scatter(runs: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.8))
    plotted = False
    for run in runs:
        if not run["complete"] and has_complete_condition_runs(runs, run["condition"]):
            continue
        replay = run["metrics"].get(("replay", "near_upright_any_transition_fraction"), {})
        strict = run["metrics"].get(("eval", "strict_success_rate"), {})
        common_steps = sorted(set(replay) & set(strict))
        if not common_steps:
            continue
        ax.plot(
            [replay[step] for step in common_steps],
            [strict[step] for step in common_steps],
            marker="o",
            linewidth=1,
            alpha=0.65,
            label=f"{run['condition']} seed {run['seed']}",
        )
        plotted = True
    ax.set_title("Replay Coverage vs Evaluation Reliability")
    ax.set_xlabel("replay near-upright transition fraction")
    ax.set_ylabel("eval strict-threshold diagnostic")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no common replay/eval steps", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_representation_health(runs: list[dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specs = [
        (("diagnostics", "q1_layer1_dormant_fraction"), "Q1 layer1 dormant fraction"),
        (("diagnostics", "q2_layer1_dormant_fraction"), "Q2 layer1 dormant fraction"),
        (("diagnostics", "q1_layer1_effective_rank_fraction"), "Q1 layer1 effective-rank fraction"),
        (("diagnostics", "q2_layer1_effective_rank_fraction"), "Q2 layer1 effective-rank fraction"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), specs):
        plot_condition_metric(ax, runs, metric, title)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_optimizer_health(runs: list[dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specs = [
        (("update", "q_grad_norm_mean"), "Q gradient norm mean"),
        (("update", "actor_update_norm_ratio_mean"), "Actor update norm ratio mean"),
        (("diagnostics", "alpha"), "Entropy temperature alpha"),
        (("update", "policy_entropy_estimate_mean"), "Policy entropy estimate mean"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), specs):
        plot_condition_metric(ax, runs, metric, title)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_condition_metric(ax: Any, runs: list[dict[str, Any]], metric: tuple[str, str], title: str) -> None:
    for condition in sorted({run["condition"] for run in runs}):
        condition_runs = [run for run in runs if run["condition"] == condition]
        plot_runs = [run for run in condition_runs if run["complete"]] or condition_runs
        steps = sorted({step for run in plot_runs for step in run["metrics"].get(metric, {})})
        if not steps:
            continue
        means = []
        lows = []
        highs = []
        used_steps = []
        for step in steps:
            values = np.asarray(
                [run["metrics"][metric][step] for run in plot_runs if step in run["metrics"].get(metric, {})],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            used_steps.append(step)
            mean, ci_low, ci_high = mean_ci95(values)
            bounds = metric_bounds(f"{metric[0]}_{metric[1]}")
            if bounds is not None:
                ci_low = max(bounds[0], ci_low)
                ci_high = min(bounds[1], ci_high)
            means.append(mean)
            lows.append(ci_low)
            highs.append(ci_high)
        label = condition_plot_label(condition_runs)
        ax.plot(used_steps, means, marker="o", linewidth=2, label=label)
        ax.fill_between(used_steps, lows, highs, alpha=0.12)
    ax.set_title(title)


def has_complete_condition_runs(runs: list[dict[str, Any]], condition: str) -> bool:
    return any(run["condition"] == condition and run["complete"] for run in runs)


def condition_plot_label(condition_runs: list[dict[str, Any]]) -> str:
    condition = str(condition_runs[0]["condition"])
    complete = sum(1 for run in condition_runs if run["complete"])
    total = len(condition_runs)
    if complete == total:
        return f"{condition} (n={total})"
    return f"{condition} ({complete}/{total} complete)"


def write_index(output_dir: Path, figures: list[Path], summary: dict[str, Any]) -> None:
    with (output_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Replay Diagnostics Report</title><h1>Replay Diagnostics Report</h1>\n")
        f.write(
            "<p>Terminal summaries and plots use complete runs when a condition has any complete runs. "
            "Incomplete runs remain listed in the CSV and JSON summaries.</p>\n"
        )
        f.write("<p>Shaded bands are seed-level 95% t intervals where at least two seeds are available.</p>\n")
        f.write("<h2>Summary</h2><pre>" + html.escape(json.dumps(summary, indent=2, sort_keys=True)) + "</pre>\n")
        for figure in figures:
            name = html.escape(figure.name, quote=True)
            f.write(f"<section><h2>{name}</h2><img src='{name}' style='max-width:100%;height:auto'></section>\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
