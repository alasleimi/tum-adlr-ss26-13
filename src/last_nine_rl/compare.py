from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MetricSeries = dict[tuple[str, str], dict[int, float]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cross-seed reliability plots for Project 15 reports.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parents containing metrics.csv.")
    parser.add_argument("--out", required=True, help="Output directory for aggregate figures.")
    args = parser.parse_args()

    run_dirs = expand_run_dirs([Path(p) for p in args.runs])
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = compare_runs(run_dirs, output_dir)
    write_index(output_dir, figures)
    print(f"Wrote comparison report: {output_dir / 'index.html'}")


def expand_run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if (path / "metrics.csv").exists():
            out.append(path)
        else:
            out.extend(child.parent for child in path.glob("**/metrics.csv") if child.is_file())
    return sorted({path.resolve() for path in out})


def compare_runs(run_dirs: list[Path], output_dir: Path) -> list[Path]:
    runs = [load_run(run_dir) for run_dir in run_dirs]
    figures = [
        plot_learning_curves(runs, output_dir / "learning_curves.png"),
        plot_reliability_nines(runs, output_dir / "reliability_nines.png"),
        plot_threshold_ladder(runs, output_dir / "threshold_ladder.png"),
        plot_replay_vs_eval(runs, output_dir / "replay_vs_eval.png"),
    ]
    heatmaps = plot_final_eval_heatmaps(runs, output_dir)
    figures.extend(heatmaps)
    pendulum_map = plot_pendulum_initial_state_map(runs, output_dir / "pendulum_initial_state_map.png")
    if pendulum_map is not None:
        figures.append(pendulum_map)
    return [figure for figure in figures if figure.exists()]


def load_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {
        "run_dir": run_dir,
        "label": run_dir.name,
        "seed": int(config.get("seed", seed_from_name(run_dir.name))),
        "env_id": config.get("env", {}).get("env_id", ""),
        "metrics": read_metrics(run_dir / "metrics.csv"),
        "eval_rows": read_eval_rows(run_dir / "eval_episodes.csv"),
    }


def seed_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit() or ch == "-")
    return int(digits) if digits not in {"", "-"} else 0


def read_metrics(path: Path) -> MetricSeries:
    metrics: MetricSeries = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["split"], row["name"])
            metrics.setdefault(key, {})[int(row["step"])] = float(row["value"])
    return metrics


def read_eval_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def plot_learning_curves(runs: list[dict[str, Any]], path: Path) -> Path:
    specs = [
        ("eval", "mean_return", "Mean Return"),
        ("eval", "worst_return", "Worst Episode Return"),
        ("eval", "success_rate", "Return Success"),
        ("eval", "strict_success_rate", "Strict Success"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, spec in zip(axes.ravel(), specs):
        plot_metric_with_seed_lines(ax, runs, spec)
        ax.set_title(spec[2])
        ax.set_xlabel("environment steps")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_reliability_nines(runs: list[dict[str, Any]], path: Path) -> Path:
    specs = [
        ("eval", "return_reliability_nines_empirical", "Return nines empirical"),
        ("eval", "return_reliability_nines_wilson95_low", "Return nines Wilson lower"),
        ("eval", "strict_reliability_nines_empirical", "Strict nines empirical"),
        ("eval", "strict_reliability_nines_wilson95_low", "Strict nines Wilson lower"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for split, name, label in specs:
        plot_metric_mean(ax, runs, (split, name), label=label)
    ax.set_title("Reliability Nines Over Training")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("-log10(failure rate)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_threshold_ladder(runs: list[dict[str, Any]], path: Path) -> Path:
    names = sorted(
        {
            name
            for run in runs
            for split, name in run["metrics"]
            if split == "eval" and name.startswith("fraction_return_ge_")
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        plot_metric_mean(ax, runs, ("eval", name), label=name.removeprefix("fraction_return_ge_"))
    ax.set_title("Return Threshold Ladder")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("fraction of eval episodes")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if names:
        ax.legend(title="threshold", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no threshold ladder metrics", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_replay_vs_eval(runs: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for run in runs:
        replay = run["metrics"].get(("replay", "near_upright_any_transition_fraction"), {})
        success = run["metrics"].get(("eval", "strict_success_rate"), {})
        common_steps = sorted(set(replay) & set(success))
        if not common_steps:
            continue
        ax.plot(
            [replay[step] for step in common_steps],
            [success[step] for step in common_steps],
            marker="o",
            linewidth=1,
            alpha=0.7,
            label=f"seed {run['seed']}",
        )
        plotted = True
    ax.set_title("Replay Coverage vs Strict Evaluation Success")
    ax.set_xlabel("replay near-upright transition fraction")
    ax.set_ylabel("strict success rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no common replay/eval steps", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_final_eval_heatmaps(runs: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    final_rows_by_run = [final_eval_rows(run["eval_rows"]) for run in runs]
    eval_seeds = sorted({int(row["seed"]) for rows in final_rows_by_run for row in rows})
    if not eval_seeds:
        return []

    return_matrix = np.full((len(runs), len(eval_seeds)), np.nan)
    strict_matrix = np.full((len(runs), len(eval_seeds)), np.nan)
    for row_idx, rows in enumerate(final_rows_by_run):
        by_seed = {int(row["seed"]): row for row in rows}
        for col_idx, eval_seed in enumerate(eval_seeds):
            row = by_seed.get(eval_seed)
            if row is None:
                continue
            return_matrix[row_idx, col_idx] = row["return"]
            strict_matrix[row_idx, col_idx] = row.get("strict_success", row.get("success", 0.0))

    order = np.argsort(np.nanmean(return_matrix, axis=0))
    eval_seeds = [eval_seeds[i] for i in order]
    return_matrix = return_matrix[:, order]
    strict_matrix = strict_matrix[:, order]

    paths = []
    paths.append(plot_heatmap(return_matrix, runs, eval_seeds, output_dir / "final_eval_return_heatmap.png", "Final Eval Return"))
    paths.append(
        plot_heatmap(
            strict_matrix,
            runs,
            eval_seeds,
            output_dir / "final_eval_strict_success_heatmap.png",
            "Final Eval Strict Success",
            vmin=0.0,
            vmax=1.0,
        )
    )
    return paths


def plot_heatmap(
    values: np.ndarray,
    runs: list[dict[str, Any]],
    eval_seeds: list[int],
    path: Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(eval_seeds)), max(3, 0.45 * len(runs))))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("eval seed, sorted by mean return")
    ax.set_ylabel("training seed")
    ax.set_yticks(range(len(runs)), [str(run["seed"]) for run in runs])
    step = max(1, len(eval_seeds) // 12)
    tick_positions = list(range(0, len(eval_seeds), step))
    ax.set_xticks(tick_positions, [str(eval_seeds[i]) for i in tick_positions], rotation=45, ha="right")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_pendulum_initial_state_map(runs: list[dict[str, Any]], path: Path) -> Path | None:
    if not runs or not runs[0]["env_id"].startswith("Pendulum"):
        return None
    final_rows = [row for run in runs for row in final_eval_rows(run["eval_rows"])]
    if not final_rows:
        return None
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in final_rows:
        grouped.setdefault(int(row["seed"]), []).append(row)

    xs = []
    ys = []
    mean_returns = []
    strict_rates = []
    env = gym.make("Pendulum-v1")
    try:
        for eval_seed, rows in sorted(grouped.items()):
            obs, _ = env.reset(seed=eval_seed)
            theta = float(np.arctan2(obs[1], obs[0]))
            xs.append(theta)
            ys.append(float(obs[2]))
            mean_returns.append(float(np.mean([row["return"] for row in rows])))
            strict_rates.append(float(np.mean([row.get("strict_success", row.get("success", 0.0)) for row in rows])))
    finally:
        env.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    plots = [
        (axes[0], mean_returns, "Mean Final Return"),
        (axes[1], strict_rates, "Strict Success Rate"),
    ]
    for ax, values, title in plots:
        scatter = ax.scatter(xs, ys, c=values, s=60, cmap="viridis", edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel("initial theta radians")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax)
    axes[0].set_ylabel("initial angular velocity")
    fig.suptitle("Pendulum Final Eval by Initial State")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def final_eval_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if not rows:
        return []
    final_step = max(int(row["step"]) for row in rows)
    return [row for row in rows if int(row["step"]) == final_step]


def plot_metric_with_seed_lines(ax, runs: list[dict[str, Any]], metric: tuple[str, str, str]) -> None:
    split, name, _ = metric
    for run in runs:
        series = run["metrics"].get((split, name))
        if not series:
            continue
        steps = sorted(series)
        ax.plot(steps, [series[step] for step in steps], color="tab:blue", alpha=0.25, linewidth=1)
    plot_metric_mean(ax, runs, (split, name), label="mean", color="tab:blue")


def plot_metric_mean(
    ax,
    runs: list[dict[str, Any]],
    metric: tuple[str, str],
    label: str,
    color: str | None = None,
) -> None:
    steps = sorted({step for run in runs for step in run["metrics"].get(metric, {})})
    if not steps:
        return
    means = []
    lows = []
    highs = []
    for step in steps:
        values = np.asarray(
            [run["metrics"][metric][step] for run in runs if step in run["metrics"].get(metric, {})],
            dtype=np.float64,
        )
        means.append(float(np.mean(values)))
        lows.append(float(np.min(values)))
        highs.append(float(np.max(values)))
    ax.plot(steps, means, marker="o", linewidth=2, label=label, color=color)
    if len(runs) > 1:
        ax.fill_between(steps, lows, highs, alpha=0.12, color=color)


def write_index(output_dir: Path, figures: list[Path]) -> None:
    with (output_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Project 15 Comparison Report</title><h1>Project 15 Comparison Report</h1>\n")
        for figure in figures:
            name = html.escape(figure.name, quote=True)
            f.write(f"<section><h2>{name}</h2><img src='{name}' style='max-width:100%;height:auto'></section>\n")


if __name__ == "__main__":
    main()
