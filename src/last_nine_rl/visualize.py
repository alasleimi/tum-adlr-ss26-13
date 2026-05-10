from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Series:
    steps: list[int]
    values: list[float]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual debug reports from Week 1 run telemetry.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parents containing events.jsonl.")
    parser.add_argument("--out", default="reports", help="Output report directory.")
    args = parser.parse_args()

    run_dirs = expand_run_dirs([Path(p) for p in args.runs])
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    report_paths = [visualize_run(run_dir, output_root / sanitize(run_dir.name)) for run_dir in run_dirs]
    write_index(output_root, report_paths)
    print(f"Wrote report index: {output_root / 'index.html'}")


def expand_run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if (path / "metrics.csv").exists():
            out.append(path)
            continue
        out.extend(sorted(child.parent for child in path.glob("**/metrics.csv") if child.is_file()))
    return sorted({path.resolve() for path in out})


def visualize_run(run_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(run_dir / "metrics.csv")
    figures: list[Path] = []

    figures.append(
        plot_named_metrics(
            metrics,
            out_dir / "eval_reliability.png",
            "Evaluation Reliability",
            [
                ("eval", "mean_return"),
                ("eval", "worst_return"),
                ("eval", "return_p10"),
                ("eval", "return_p90"),
            ],
        )
    )
    figures.append(
        plot_named_metrics(
            metrics,
            out_dir / "eval_success_collapse.png",
            "Evaluation Success and Collapse",
            [
                ("eval", "success_rate"),
                ("eval", "strict_success_rate"),
                ("eval", "stability_success_rate"),
                ("eval", "failure_rate"),
                ("eval", "collapse_rate"),
                ("eval", "success_rate_wilson95_low"),
                ("eval", "success_rate_wilson95_high"),
            ],
        )
    )
    figures.append(
        plot_named_metrics(
            metrics,
            out_dir / "replay_coverage.png",
            "Replay Coverage",
            [
                ("replay", "near_upright_any_transition_fraction"),
                ("replay", "near_upright_obs_fraction"),
                ("replay", "reward_mean"),
                ("replay", "reward_max"),
                ("replay", "action_saturation_fraction"),
            ],
        )
    )
    figures.append(
        plot_named_metrics(
            metrics,
            out_dir / "update_health.png",
            "SAC Update Health",
            [
                ("update", "q_loss_cleanrl_logged"),
                ("update", "actor_loss"),
                ("update", "alpha"),
                ("update", "q_update_norm_ratio"),
                ("update", "actor_update_norm_ratio"),
            ],
        )
    )
    figures.extend(plot_matching_metrics(metrics, out_dir, "diagnostics", "dormant_fraction", "dormant_fractions.png"))
    figures.extend(
        plot_matching_metrics(metrics, out_dir, "diagnostics", "effective_rank_fraction", "effective_rank_fractions.png")
    )

    eval_csv = run_dir / "eval_episodes.csv"
    if eval_csv.exists():
        figures.append(plot_eval_episodes(eval_csv, out_dir / "eval_episode_returns.png"))

    return write_run_index(run_dir, out_dir, figures)


def read_metrics(path: Path) -> dict[tuple[str, str], Series]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grouped[(row["split"], row["name"])].append((int(row["step"]), float(row["value"])))
    return {
        key: Series(
            steps=[step for step, _ in values],
            values=[value for _, value in values],
        )
        for key, values in grouped.items()
    }


def plot_named_metrics(
    metrics: dict[tuple[str, str], Series],
    path: Path,
    title: str,
    names: list[tuple[str, str]],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for split, name in names:
        series = metrics.get((split, name))
        if series is None:
            continue
        ax.plot(series.steps, series.values, marker="o", linewidth=1.5, label=f"{split}/{name}")
        plotted = True
    finish_plot(fig, ax, path, title, plotted)
    return path


def plot_matching_metrics(
    metrics: dict[tuple[str, str], Series],
    out_dir: Path,
    split: str,
    pattern: str,
    filename: str,
) -> list[Path]:
    names = sorted((metric_split, name) for metric_split, name in metrics if metric_split == split and pattern in name)
    if not names:
        return []
    path = out_dir / filename
    return [plot_named_metrics(metrics, path, pattern.replace("_", " ").title(), names)]


def plot_eval_episodes(path: Path, out_path: Path) -> Path:
    by_step: dict[int, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_step[int(row["step"])].append(float(row["return"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for step, returns in sorted(by_step.items()):
        ax.scatter([step] * len(returns), returns, s=18, alpha=0.65, color="#1f77b4")
        plotted = True
    finish_plot(fig, ax, out_path, "Per-Episode Evaluation Returns", plotted)
    return out_path


def finish_plot(fig, ax, path: Path, title: str, plotted: bool) -> None:
    ax.set_title(title)
    ax.set_xlabel("environment steps")
    ax.grid(True, alpha=0.25)
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    else:
        ax.text(0.5, 0.5, "metric not available", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_run_index(run_dir: Path, out_dir: Path, figures: list[Path]) -> Path:
    index_path = out_dir / "index.html"
    rel_figures = [figure.name for figure in figures if figure.exists()]
    with index_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write(f"<title>{html.escape(run_dir.name)}</title><h1>{html.escape(str(run_dir))}</h1>\n")
        for figure in rel_figures:
            escaped = html.escape(figure, quote=True)
            f.write(f"<section><h2>{escaped}</h2><img src='{escaped}' style='max-width:100%;height:auto'></section>\n")
    return index_path


def write_index(output_root: Path, run_indexes: list[Path]) -> None:
    with (output_root / "index.html").open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Last-Nine Reports</title><h1>Last-Nine Reports</h1>\n")
        for index_path in run_indexes:
            rel = index_path.relative_to(output_root)
            href = html.escape(rel.as_posix(), quote=True)
            label = html.escape(index_path.parent.name)
            f.write(f"<p><a href='{href}'>{label}</a></p>\n")


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


if __name__ == "__main__":
    main()
