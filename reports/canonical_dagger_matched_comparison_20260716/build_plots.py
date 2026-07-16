from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATHS = {
    "Clean DAgger\n100k labels": REPO_ROOT
    / "reports/canonical_reference_dagger_100k_n5_20260716/relative/relative_summary.json",
    "DAgger 100k +\nSimbaV2 100k": REPO_ROOT
    / "reports/canonical_reference_dagger_then_simbav2_100k_n5_20260716/relative/relative_summary.json",
    "Standard SimbaV2\n100k": REPO_ROOT
    / "reports/week3_simbav2_scale_100k_n5_20260527/relative_success/"
    "simba_full_official_opt/relative_summary.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summaries = {label: read_json(path) for label, path in SUMMARY_PATHS.items()}
    older = read_json(REPO_ROOT / "reports/dagger_exact_comparison_20260716/computed_metrics.json")
    q_filtered = older["Q-filtered FastSACN, 5 seeds"]
    write_primary_metrics(summaries, q_filtered)
    write_per_seed_metrics(summaries)


def write_primary_metrics(summaries: dict[str, dict], q_filtered: dict) -> None:
    labels = [*summaries, "FastSACN8 UTD2\n50k + Q-filter"]
    near = [
        summary["criteria"]["near_best_known_return_eps"]["rate"]
        for summary in summaries.values()
    ] + [q_filtered["near_best_known_return_eps"]]
    task = [summary["criteria"]["task_success"]["rate"] for summary in summaries.values()] + [
        q_filtered["task_success"]
    ]
    x = np.arange(len(labels))
    width = 0.35
    figure, axis = plt.subplots(figsize=(11, 5.8))
    near_bars = axis.bar(
        x - width / 2,
        near,
        width,
        label="Near max(DP, controller) - 5",
        color="#315b7d",
    )
    task_bars = axis.bar(x + width / 2, task, width, label="Task success", color="#d47a35")
    axis.set_ylim(0.75, 0.96)
    axis.set_ylabel("Five-seed success rate")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for bars in (near_bars, task_bars):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height() * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "primary_metrics.png", dpi=180)
    plt.close(figure)


def write_per_seed_metrics(summaries: dict[str, dict]) -> None:
    colors = ["#315b7d", "#4f8f73", "#8b5d9b"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, metric, title in zip(
        axes,
        ("near_best_known_return_eps", "task_success"),
        ("Near-reference success", "Task success"),
    ):
        for color, (name, summary) in zip(colors, summaries.items()):
            rates = summary["criteria"][metric]["seed_rates"]
            axis.plot(
                range(5),
                rates,
                marker="o",
                linewidth=2,
                color=color,
                label=name.replace("\n", " "),
            )
        axis.set_title(title)
        axis.set_xlabel("Training seed")
        axis.set_xticks(range(5))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Grid success rate")
    axes[0].set_ylim(0.75, 0.98)
    axes[1].legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "per_seed_primary_metrics.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
