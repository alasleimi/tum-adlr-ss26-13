from __future__ import annotations

import csv
import html
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "week3_workshop_presentation_20260527"
FIG = OUT / "figures"

RELATIVE_CSV = ROOT / "reports" / "week3_relative_frontier_20260526" / "relative_frontier.csv"
RELIABILITY_CSV = ROOT / "reports" / "week3_reliability_frontier_20260526" / "reliability_frontier.csv"
POSTHOC_CSV = ROOT / "reports" / "week3_followup_20260526" / "key_posthoc_results.csv"
DIAGNOSTIC_CSV = ROOT / "reports" / "week3_followup_20260526" / "key_diagnostic_results.csv"
SAC_RELATIVE_ROLLOUTS_CSV = (
    ROOT / "reports" / "week3_simbav2_scale_100k_20260526" / "relative_success" / "sac" / "relative_rollouts.csv"
)
SIMBA_RELATIVE_ROLLOUTS_CSV = (
    ROOT
    / "reports"
    / "week3_simbav2_scale_100k_20260526"
    / "relative_success"
    / "simba_full_official_opt"
    / "relative_rollouts.csv"
)
SAC_CELL_SUMMARY_CSV = (
    ROOT / "reports" / "week3_simbav2_scale_100k_20260526" / "relative_success" / "sac" / "relative_cell_summary.csv"
)
SIMBA_CELL_SUMMARY_CSV = (
    ROOT
    / "reports"
    / "week3_simbav2_scale_100k_20260526"
    / "relative_success"
    / "simba_full_official_opt"
    / "relative_cell_summary.csv"
)
DP_GRID_CSV = (
    ROOT
    / "reports"
    / "pendulum_investigation_20260509"
    / "pendulum_dp_100k_reset_support_241x161x81"
    / "pendulum_dp_grid.csv"
)
CONTROLLER_GRID_CSV = (
    ROOT
    / "reports"
    / "pendulum_investigation_20260509"
    / "pendulum_controller_reset_support_61x41"
    / "controller_grid.csv"
)


PALETTE = {
    "blue": "#376996",
    "teal": "#2e7d78",
    "green": "#4f8a5b",
    "orange": "#c47a2c",
    "red": "#b9564c",
    "purple": "#6e5e9a",
    "gray": "#5f6b76",
    "light": "#eef2ec",
    "line": "#d8ddd6",
    "paper": "#fbfaf5",
    "panel": "#fffdf8",
    "dark": "#17202a",
}

plt.rcParams.update(
    {
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["line"],
        "axes.labelcolor": PALETTE["dark"],
        "xtick.color": PALETTE["gray"],
        "ytick.color": PALETTE["gray"],
        "grid.color": PALETTE["line"],
        "font.family": "DejaVu Sans",
    }
)


@dataclass(frozen=True)
class Asset:
    name: str
    source: Path
    dest: Path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    relative = read_rows(RELATIVE_CSV)
    reliability = read_rows(RELIABILITY_CSV)
    posthoc = read_rows(POSTHOC_CSV)
    diagnostics = read_rows(DIAGNOSTIC_CSV)
    analysis = build_analysis(relative, diagnostics)

    generated = {
        "raw_maps": plot_raw_maps(FIG / "raw_maps_task_return_regret.png"),
        "main_result": plot_main_result(analysis, FIG / "main_result_seed_intervals.png"),
        "exploration": plot_exploration_optimization(analysis, diagnostics, FIG / "exploration_vs_optimization.png"),
        "compute_negative": plot_compute_negative(relative, diagnostics, FIG / "compute_negative_result.png"),
        "hard_interventions": plot_hard_interventions(relative, FIG / "hard_interventions_result.png"),
    }

    copied: dict[str, Path] = {}

    gif_path = FIG / "pendulum_sac_vs_simba.gif"
    gif_status = build_pendulum_gif(gif_path)

    summary = build_scientific_summary_rows(analysis, relative, posthoc, diagnostics)
    write_summary_csv(OUT / "presentation_numbers.csv", summary)
    notes = build_notes(analysis, gif_status)
    script = build_script(analysis, gif_status)
    (OUT / "speaker_notes.md").write_text(notes, encoding="utf-8")
    (OUT / "script.md").write_text(script, encoding="utf-8")
    (OUT / "index.html").write_text(build_deck(generated, copied, gif_path, analysis), encoding="utf-8")

    print(f"Wrote {OUT / 'index.html'}")
    print(f"Wrote {OUT / 'speaker_notes.md'}")
    print(f"Wrote {OUT / 'script.md'}")
    print(f"Wrote {OUT / 'presentation_numbers.csv'}")
    print(gif_status)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def find_row(rows: Iterable[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"Missing row matching {criteria}")


def first_row(rows: Iterable[dict[str, str]], predicate) -> dict[str, str]:
    for row in rows:
        if predicate(row):
            return row
    raise KeyError("Missing row for predicate")


def pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def num(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def setup_ax(ax, title: str, ylabel: str | None = None, ylim: tuple[float, float] | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_facecolor(PALETTE["panel"])
    ax.grid(axis="y", alpha=0.65, linewidth=0.8)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax, bars, fmt=pct) -> None:
    for bar in bars:
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.012,
            fmt(h),
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["dark"],
        )


def plot_matched_100k(
    posthoc: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    path: Path,
) -> Path:
    sac = find_row(posthoc, budget="100k", condition="SAC")
    simba = find_row(posthoc, budget="100k", condition="Full SimbaV2 official opt")
    sac_diag = first_row(diagnostics, lambda r: r["condition"] == "100k SAC")
    simba_diag = first_row(diagnostics, lambda r: r["condition"] == "100k Full SimbaV2 official opt")

    success_labels = ["Task", "Stable", "Long streak"]
    sac_values = [
        as_float(sac, "task_success"),
        as_float(sac, "stability_success"),
        as_float(sac, "streak_success"),
    ]
    simba_values = [
        as_float(simba, "task_success"),
        as_float(simba, "stability_success"),
        as_float(simba, "streak_success"),
    ]
    mechanism_labels = ["Replay near", "Q1 rank", "Q1 dormant"]
    sac_mechanism = [
        as_float(sac_diag, "replay_near_any"),
        as_float(sac_diag, "q1_rank"),
        as_float(sac_diag, "q1_dormant"),
    ]
    simba_mechanism = [
        as_float(simba_diag, "replay_near_any"),
        as_float(simba_diag, "q1_rank"),
        as_float(simba_diag, "q1_dormant"),
    ]

    x = np.arange(len(success_labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1), gridspec_kw={"width_ratios": [1.05, 1.0]})
    bars_a = axes[0].bar(x - width / 2, sac_values, width, label="SAC 100k", color=PALETTE["gray"])
    bars_b = axes[0].bar(x + width / 2, simba_values, width, label="Full SimbaV2 100k", color=PALETTE["teal"])
    annotate_bars(axes[0], bars_a)
    annotate_bars(axes[0], bars_b)
    setup_ax(axes[0], "Task-specific reliability", "rate", (0.68, 1.02))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(success_labels)
    axes[0].legend(frameon=False, loc="upper left")

    x2 = np.arange(len(mechanism_labels))
    axes[1].bar(x2 - width / 2, sac_mechanism, width, label="SAC 100k", color=PALETTE["gray"])
    axes[1].bar(x2 + width / 2, simba_mechanism, width, label="Full SimbaV2 100k", color=PALETTE["teal"])
    setup_ax(axes[1], "Exploration proxy vs optimization health", "fraction", (0.0, 0.9))
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(mechanism_labels)
    fig.suptitle(
        "Matched 100k: the reliability gap is not explained by seeing fewer good states",
        x=0.05,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_relative_frontier(relative: list[dict[str, str]], path: Path) -> Path:
    wanted = [
        ("SAC50k", "SAC\n50k"),
        ("SAC100k", "SAC\n100k"),
        ("Legacy500kUTD1", "SAC\n500k"),
        ("FullSimba50k", "Full\n50k"),
        ("FullSimbaNoDistUTD2_50k", "No dist.\nUTD2"),
        ("FullSimba100k", "Full\n100k"),
        ("FullSimbaHardReset02_50k", "Hard reset\n50k"),
        ("FullSimbaHardReplay02_50k", "Hard replay\n50k"),
    ]
    rows = [(short, find_row(relative, label=label)) for label, short in wanted]
    task = [as_float(row, "task_rate") for _, row in rows]
    near = [as_float(row, "near_best_known_rate") for _, row in rows]
    labels = [short for short, _ in rows]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    bars_a = ax.bar(x - width / 2, task, width, label="Task success", color=PALETTE["green"])
    bars_b = ax.bar(
        x + width / 2,
        near,
        width,
        label="Within epsilon of max(DP, controller)",
        color=PALETTE["blue"],
    )
    setup_ax(ax, "Exact initial-state grid: task success and DP/controller-relative success", "success rate", (0.55, 1.02))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    annotate_bars(ax, bars_a)
    annotate_bars(ax, bars_b)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_ablation_summary(
    posthoc: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    relative: list[dict[str, str]],
    path: Path,
) -> Path:
    selections = [
        ("50k", "SAC", "SAC"),
        ("50k", "Backbone", "Backbone"),
        ("50k", "Full SimbaV2 no distributional", "No dist."),
        ("50k", "Full SimbaV2 official opt", "Full"),
        ("50k UTD2", "Full SimbaV2 no distributional", "No dist.\nUTD2"),
    ]
    rows = []
    for budget, condition, label in selections:
        try:
            rows.append((label, find_row(posthoc, budget=budget, condition=condition)))
        except KeyError:
            pass
    labels = [label for label, _ in rows]
    task = [as_float(row, "task_success") for _, row in rows]
    near_lookup = {
        "SAC": ("SAC50k", "near_best_known_rate"),
        "No dist.": ("FullSimbaNoDist50k", "near_best_known_rate"),
        "Full": ("FullSimba50k", "near_best_known_rate"),
        "No dist.\nUTD2": ("FullSimbaNoDistUTD2_50k", "near_best_known_rate"),
    }
    near = []
    for label in labels:
        if label not in near_lookup:
            near.append(math.nan)
            continue
        rel_label, col = near_lookup[label]
        near.append(as_float(find_row(relative, label=rel_label), col))
    qrank = []
    dormant = []
    diagnostic_label = {
        "SAC": "50k SAC",
        "Backbone": "50k Backbone",
        "No dist.": "50k Full SimbaV2 no distributional",
        "No dist.\nUTD2": "50k UTD2 Full SimbaV2 no distributional",
        "Full": "50k Full SimbaV2 official opt",
    }
    for label, _ in rows:
        exact_condition = diagnostic_label.get(label, "")
        diag = next((d for d in diagnostics if d["condition"] == exact_condition), {})
        qrank.append(as_float(diag, "q1_rank", 0.0))
        dormant.append(as_float(diag, "q1_dormant", 0.0))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axes[0].bar(x - 0.18, task, 0.36, color=PALETTE["green"], label="Task posthoc")
    axes[0].bar(x + 0.18, near, 0.36, color=PALETTE["blue"], label="Near max(DP, controller)")
    setup_ax(axes[0], "Which SimbaV2 pieces help most?", "success rate", (0.65, 0.94))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].bar(x - 0.18, qrank, 0.36, color=PALETTE["blue"], label="Q1 effective rank")
    axes[1].bar(x + 0.18, dormant, 0.36, color=PALETTE["red"], label="Q1 dormant")
    setup_ax(axes[1], "Representation tracker", "fraction", (0.0, 0.48))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_component_evidence(
    component_posthoc: list[dict[str, str]],
    component_fixed: list[dict[str, str]],
    relative: list[dict[str, str]],
    path: Path,
) -> Path:
    short_budget = [
        ("SAC", "SAC"),
        ("Backbone", "Backbone"),
        ("Full official", "Full"),
        ("Full no feature norm", "No feat."),
        ("Full no projection", "No proj."),
        ("Full no distributional", "No dist."),
        ("Full no reward scaling", "No reward"),
        ("Backbone + distributional + official opt", "Dist only"),
        ("Projection + official opt", "Proj only"),
    ]
    short_rows_raw = [(label, find_row(component_posthoc, condition=condition)) for condition, label in short_budget]
    short_rows = sorted(
        [
            (label, row, as_float(row, "stability"), as_float(row, "collapse"))
            for label, row in short_rows_raw
        ],
        key=lambda item: item[2],
        reverse=True,
    )
    short_labels = [label for label, _, _, _ in short_rows]
    stability = [value for _, _, value, _ in short_rows]
    collapse = [value for _, _, _, value in short_rows]
    short_colors = [PALETTE["red"] if value > 0.1 else PALETTE["green"] for value in collapse]

    fixed_rows = [
        ("SAC", find_row(component_fixed, condition="SAC")),
        ("Backbone", find_row(component_fixed, condition="Backbone")),
        ("Full", find_row(component_fixed, condition="Full official")),
        ("No dist.", find_row(component_fixed, condition="Full no distributional")),
    ]
    fixed_labels = [label for label, _ in fixed_rows]
    qrank = [as_float(row, "q1_rank") for _, row in fixed_rows]
    dormant = [as_float(row, "q1_dormant") for _, row in fixed_rows]

    exact_grid = [
        ("Full", find_row(relative, label="FullSimba50k")),
        ("No dist.", find_row(relative, label="FullSimbaNoDist50k")),
        ("No dist.\nUTD2", find_row(relative, label="FullSimbaNoDistUTD2_50k")),
    ]
    grid_labels = [label for label, _ in exact_grid]
    grid_task = [as_float(row, "task_rate") for _, row in exact_grid]
    grid_near = [as_float(row, "near_best_known_rate") for _, row in exact_grid]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.05]},
    )

    y = np.arange(len(short_labels))
    bars = axes[0].barh(y, stability, color=short_colors)
    setup_ax(axes[0], "10k diagnostic screen", "task-stability proxy")
    axes[0].set_xlim(0.0, 0.90)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.20)
    axes[0].grid(axis="y", alpha=0.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(short_labels)
    for bar, value, collapse_value in zip(bars, stability, collapse):
        label = f"{100.0 * value:.0f}%"
        if collapse_value > 0.1:
            label += f"  collapse {100.0 * collapse_value:.0f}%"
        axes[0].text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            label,
            va="center",
            fontsize=8,
            color=PALETTE["dark"],
        )
    axes[0].text(
        0.02,
        0.04,
        "Short budget: use this to detect unstable component combinations.",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color=PALETTE["dark"],
        bbox={"facecolor": PALETTE["paper"], "edgecolor": "none", "alpha": 0.90, "pad": 2.0},
    )

    x2 = np.arange(len(fixed_labels))
    width = 0.36
    axes[1].bar(x2 - width / 2, qrank, width, color=PALETTE["blue"], label="Q1 rank")
    axes[1].bar(x2 + width / 2, dormant, width, color=PALETTE["red"], label="Q1 dormant")
    setup_ax(axes[1], "Critic representation", "fraction", (0.0, 0.48))
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(fixed_labels)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    axes[1].text(
        0.02,
        0.88,
        "Dormant: units nearly unused (lower better)\nRank: independent feature directions (higher better)",
        transform=axes[1].transAxes,
        fontsize=8.5,
        color=PALETTE["dark"],
        bbox={"facecolor": PALETTE["paper"], "edgecolor": "none", "alpha": 0.90, "pad": 2.0},
    )

    x3 = np.arange(len(grid_labels))
    bars_task = axes[2].bar(x3 - width / 2, grid_task, width, color=PALETTE["green"], label="Task grid")
    bars_near = axes[2].bar(x3 + width / 2, grid_near, width, color=PALETTE["blue"], label="Near max(DP, controller)")
    setup_ax(axes[2], "50k exact-grid subset", "success rate", (0.70, 0.92))
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(grid_labels)
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=8)
    for bar in list(bars_task) + list(bars_near):
        value = float(bar.get_height())
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.004,
            pct(value),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=PALETTE["dark"],
        )

    fig.suptitle(
        "SimbaV2 components: backbone stabilizes critics; full categorical still leads task-grid reliability",
        x=0.03,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_scale_summary(relative: list[dict[str, str]], path: Path) -> Path:
    wanted = [
        ("Legacy100k", "SAC\n100k"),
        ("Legacy500kUTD1", "SAC\n500k"),
        ("FullSimba50k", "Full\n50k"),
        ("FullSimba100k", "Full\n100k"),
        ("FullSimbaHardReset02_50k", "Hard reset\np=0.2"),
        ("FullSimbaHardReplay02_50k", "Hard replay\np=0.2"),
    ]
    rows = [(short, find_row(relative, label=label)) for label, short in wanted]
    labels = [short for short, _ in rows]
    task = [as_float(row, "task_rate") for _, row in rows]
    near = [as_float(row, "near_best_known_rate") for _, row in rows]
    hard = [as_float(row, "near_down_task_rate") for _, row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axes[0].plot(x, task, marker="o", linewidth=2.4, color=PALETTE["green"], label="Task")
    axes[0].plot(x, near, marker="o", linewidth=2.4, color=PALETTE["blue"], label="Near max(DP, controller)")
    setup_ax(axes[0], "Compute and data-distribution interventions", "grid success rate", (0.62, 0.98))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].bar(x, hard, color=PALETTE["orange"])
    setup_ax(axes[1], "Hard-start task success", "near-down region rate", (0.0, 0.82))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_method_pipeline(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    ax.axis("off")
    steps = [
        ("Train", "CleanRL SAC\nSimbaV2 subsets\nmultiple seeds"),
        ("Track", "replay coverage\nfeature norms\nrank/dormancy"),
        ("Evaluate", "1000 posthoc eps\nexact grid\nDP/controller refs"),
        ("Diagnose", "exploration?\noptimization?\ncalibration?"),
    ]
    xs = np.linspace(0.12, 0.88, len(steps))
    for idx, ((title, body), x) in enumerate(zip(steps, xs)):
        ax.add_patch(
            plt.Rectangle(
                (x - 0.105, 0.36),
                0.21,
                0.34,
                facecolor=PALETTE["light"],
                edgecolor=PALETTE["dark"],
                linewidth=1.2,
            )
        )
        ax.text(x, 0.61, title, ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["dark"])
        ax.text(x, 0.47, body, ha="center", va="center", fontsize=10, color=PALETTE["dark"])
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.13, 0.53),
                xytext=(x + 0.13, 0.53),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": PALETTE["dark"]},
            )
    ax.text(
        0.5,
        0.17,
        "Key design choice: define success by the task or by DP/controller-relative return.",
        ha="center",
        fontsize=14,
        color=PALETTE["teal"],
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_metric_ladder(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.axis("off")
    labels = [
        ("Task success", "swing up and stay\nnear upright", PALETTE["green"]),
        ("DP relative", "within epsilon of\nsame-start DP", PALETTE["blue"]),
        ("Return match", "within epsilon of\nmax(DP, controller)", PALETTE["teal"]),
        ("Nines", "-log10(failure)\non real criteria", PALETTE["red"]),
        ("Trackers", "replay coverage,\nrank, dormancy", PALETTE["gray"]),
    ]
    for idx, (title, body, color) in enumerate(labels):
        y = 0.82 - idx * 0.17
        ax.add_patch(plt.Rectangle((0.08, y - 0.055), 0.84, 0.1, facecolor=PALETTE["panel"], edgecolor=color, linewidth=2.0))
        ax.text(0.12, y, title, ha="left", va="center", fontsize=13, fontweight="bold", color=color)
        ax.text(0.38, y, body, ha="left", va="center", fontsize=10, color=PALETTE["dark"])
    ax.text(
        0.5,
        0.03,
        "A fixed return bar is only a diagnostic; it is not our success definition.",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["orange"],
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_dp_reference(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    boxes = [
        (0.05, 0.58, 0.22, 0.28, "Initial state", "theta, theta_dot\none grid cell"),
        (0.38, 0.58, 0.22, 0.28, "References", "finite-horizon DP\nhand controller"),
        (0.71, 0.58, 0.22, 0.28, "Policy rollout", "SAC or SimbaV2\nfrom same cell"),
        (0.19, 0.12, 0.62, 0.28, "Success test", "Task: swing up and stabilize\nRelative: R(policy) >= max(DP, controller) - eps"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=PALETTE["panel"], edgecolor=PALETTE["dark"], linewidth=1.4))
        ax.text(x + 0.02, y + h - 0.08, title, fontsize=13, fontweight="bold", color=PALETTE["dark"])
        ax.text(x + 0.02, y + h - 0.18, body, fontsize=9.5, color=PALETTE["gray"], va="top")

    arrow = {"arrowstyle": "->", "lw": 1.8, "color": PALETTE["dark"]}
    ax.annotate("", xy=(0.38, 0.72), xytext=(0.27, 0.72), arrowprops=arrow)
    ax.annotate("", xy=(0.71, 0.72), xytext=(0.60, 0.72), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.38), xytext=(0.50, 0.58), arrowprops=arrow)
    ax.annotate("", xy=(0.64, 0.38), xytext=(0.79, 0.58), arrowprops=arrow)

    ax.text(
        0.5,
        0.94,
        "DP reference: compare each rollout to same-start DP/controller",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(
        0.5,
        0.02,
        "Use max(DP, controller) because grid DP is approximate and the hand controller is strong in some regions.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["teal"],
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.08)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def grid_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("theta", ""), row.get("theta_dot", "")


def seed_ci95(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan, math.nan
    mean = float(np.mean(clean))
    if len(clean) < 2:
        return mean, math.nan
    # Small-n seed intervals are intentionally conservative. The unit is a
    # training seed, not a rollout cell.
    tcrit = {
        2: 12.706204736432095,
        3: 4.302652729911275,
        4: 3.182446305284263,
        5: 2.7764451051977987,
        6: 2.5705818366147395,
    }.get(len(clean), 1.959963984540054)
    half = tcrit * float(np.std(clean, ddof=1)) / math.sqrt(len(clean))
    return mean, half


def metric_from_flags(
    rows: list[dict[str, str]],
    flag_key: str,
    feasible_keys: set[tuple[str, str]] | None = None,
) -> dict[str, object]:
    by_seed: dict[str, list[float]] = {}
    values: list[float] = []
    for row in rows:
        if feasible_keys is not None and grid_key(row) not in feasible_keys:
            continue
        value = 1.0 if flag(row, flag_key) else 0.0
        values.append(value)
        by_seed.setdefault(row.get("actual_seed", "0"), []).append(value)

    seed_rates = [float(np.mean(by_seed[seed])) for seed in sorted(by_seed, key=lambda item: int(float(item)))]
    mean, half = seed_ci95(seed_rates)
    return {
        "mean": mean,
        "ci95": half,
        "seed_rates": seed_rates,
        "count": int(sum(values)),
        "total": len(values),
    }


def known_feasible_keys() -> set[tuple[str, str]]:
    dp_rows = read_rows(DP_GRID_CSV)
    controller_rows = read_rows(CONTROLLER_GRID_CSV)
    controller_by_key = {grid_key(row): row for row in controller_rows}
    feasible: set[tuple[str, str]] = set()
    for row in dp_rows:
        dp_task = as_float(row, "dp_near_upright_fraction", 0.0) >= 0.8 and as_float(
            row, "dp_not_near_upright_streak", 999.0
        ) <= 50.0
        controller = controller_by_key.get(grid_key(row), {})
        controller_task = flag(controller, "controller_task_success") if controller else False
        if dp_task or controller_task:
            feasible.add(grid_key(row))
    return feasible


def summarize_policy_rollouts(path: Path, feasible_keys: set[tuple[str, str]]) -> dict[str, dict[str, object]]:
    rows = read_rows(path)
    return {
        "task_all": metric_from_flags(rows, "task_success"),
        "task_feasible": metric_from_flags(rows, "task_success", feasible_keys),
        "return_match_all": metric_from_flags(rows, "near_best_known_return_eps"),
        "return_match_feasible": metric_from_flags(rows, "near_best_known_return_eps", feasible_keys),
    }


def build_analysis(
    relative: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
) -> dict[str, object]:
    feasible = known_feasible_keys()
    total_cells = len(read_rows(DP_GRID_CSV))
    policies = {
        "SAC 100k": summarize_policy_rollouts(SAC_RELATIVE_ROLLOUTS_CSV, feasible),
        "Full SimbaV2 100k": summarize_policy_rollouts(SIMBA_RELATIVE_ROLLOUTS_CSV, feasible),
    }
    return {
        "epsilon_return": 5.0,
        "total_cells": total_cells,
        "feasible_cells": len(feasible),
        "uncertified_cells": total_cells - len(feasible),
        "policies": policies,
        "relative": {row["label"]: row for row in relative},
        "diagnostics": {row["condition"]: row for row in diagnostics},
    }


def fmt_pp(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f} pp"


def fmt_metric(metric: dict[str, object]) -> str:
    mean = float(metric["mean"])
    half = float(metric["ci95"])
    if math.isfinite(half):
        return f"{pct(mean)} +/- {fmt_pp(half)}"
    return pct(mean)


def plot_cell_heatmap(
    ax,
    rows: list[dict[str, str]],
    value_key: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
) -> None:
    theta_values = sorted({as_float(row, "theta_degrees") for row in rows})
    velocity_values = sorted({as_float(row, "theta_dot") for row in rows})
    theta_index = {value: idx for idx, value in enumerate(theta_values)}
    velocity_index = {value: idx for idx, value in enumerate(velocity_values)}
    matrix = np.full((len(velocity_values), len(theta_values)), np.nan, dtype=np.float64)
    for row in rows:
        matrix[velocity_index[as_float(row, "theta_dot")], theta_index[as_float(row, "theta_degrees")]] = as_float(
            row, value_key
        )

    theta_step = theta_values[1] - theta_values[0] if len(theta_values) > 1 else 1.0
    velocity_step = velocity_values[1] - velocity_values[0] if len(velocity_values) > 1 else 1.0
    extent = [
        theta_values[0] - theta_step / 2.0,
        theta_values[-1] + theta_step / 2.0,
        velocity_values[0] - velocity_step / 2.0,
        velocity_values[-1] + velocity_step / 2.0,
    ]
    im = ax.imshow(matrix, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    ax.set_xlabel("initial angle theta (degrees)")
    ax.set_ylabel("initial angular velocity")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=9)


def plot_raw_maps(path: Path) -> Path:
    sac_rows = read_rows(SAC_CELL_SUMMARY_CSV)
    simba_rows = read_rows(SIMBA_CELL_SUMMARY_CSV)
    # A few extreme shortfalls exceed 100 return points; a full-range colorbar
    # makes almost every nonzero cell look identical. Cap the color scale at 20
    # so the map shows ordinary differences near the success boundary.
    regret_vmax = 20.0

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.0))
    specs = [
        (sac_rows, "SAC 100k"),
        (simba_rows, "Full SimbaV2 100k"),
    ]
    for row_idx, (rows, label) in enumerate(specs):
        plot_cell_heatmap(
            axes[row_idx, 0],
            rows,
            "task_success_rate",
            f"{label}: task-stability",
            "Greens",
            0.0,
            1.0,
            "fraction of seeds",
        )
        plot_cell_heatmap(
            axes[row_idx, 1],
            rows,
            "near_best_known_return_eps_rate",
            f"{label}: reference success",
            "Blues",
            0.0,
            1.0,
            "fraction of seeds",
        )
        plot_cell_heatmap(
            axes[row_idx, 2],
            rows,
            "mean_regret_to_best_known",
            f"{label}: return shortfall",
            "YlOrRd",
            0.0,
            regret_vmax,
            "return points below reference, color cap 20",
        )
    fig.suptitle(
        "Same 61 x 41 reset-support grid: task stability, reference success, and return shortfall",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_main_result(analysis: dict[str, object], path: Path) -> Path:
    policies = analysis["policies"]  # type: ignore[index]
    categories = [
        ("Reference success", "return_match_all"),
        ("All-grid task stability", "task_all"),
        ("Known-feasible task stability", "task_feasible"),
    ]
    methods = [("SAC 100k", PALETTE["blue"]), ("Full SimbaV2 100k", PALETTE["teal"])]

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    width = 0.28
    for cat_idx, (_, metric_key) in enumerate(categories):
        for method_idx, (method, color) in enumerate(methods):
            metric = policies[method][metric_key]  # type: ignore[index]
            x = cat_idx + (method_idx - 0.5) * width
            seed_rates = [float(v) for v in metric["seed_rates"]]  # type: ignore[index]
            jitter = np.linspace(-0.035, 0.035, len(seed_rates)) if seed_rates else []
            ax.scatter(
                np.asarray(jitter) + x,
                seed_rates,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=44,
                zorder=4,
                alpha=0.88,
            )
            ax.errorbar(
                x,
                float(metric["mean"]),
                yerr=float(metric["ci95"]),
                fmt="D",
                color=color,
                ecolor=color,
                markersize=7,
                capsize=5,
                linewidth=1.5,
                zorder=5,
                label=method if cat_idx == 0 else None,
            )
            ax.text(
                x,
                min(1.035, float(metric["mean"]) + 0.04),
                f"{100.0 * float(metric['mean']):.1f}\n+/-{100.0 * float(metric['ci95']):.1f}pp",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["dark"],
            )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([label for label, _ in categories], fontsize=10)
    setup_ax(ax, "100k exact-grid reliability, seed is the statistical unit", "rate", (0.0, 1.04))
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.0,
        -0.22,
        "Dots are training seeds; diamonds are seed means; intervals are 95% t-intervals over seeds (n=3).",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["gray"],
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_component_screen(component_posthoc: list[dict[str, str]], path: Path) -> Path:
    stable_items = [
        ("SAC", "SAC"),
        ("SAC + ReDo", "SAC + ReDo"),
        ("Backbone", "Backbone"),
        ("Full official", "Full official"),
        ("Full no feature norm", "No feature norm"),
        ("Full no projection", "No projection"),
        ("Full no distributional", "No distributional"),
    ]
    broken_items = [
        ("Full no reward scaling", "No reward scaling"),
        ("Backbone + distributional + official opt", "Distributional only"),
        ("Projection + official opt", "Projection only"),
    ]
    stable_rows = [(label, find_row(component_posthoc, condition=condition)) for condition, label in stable_items]
    broken_rows = [(label, find_row(component_posthoc, condition=condition)) for condition, label in broken_items]
    sac_strict = as_float(find_row(component_posthoc, condition="SAC"), "strict")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.6),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.34},
    )

    y = np.arange(len(stable_rows))[::-1]
    for ypos, (label, row) in zip(y, stable_rows):
        strict = as_float(row, "strict")
        low = as_float(row, "strict_wilson_low")
        high = as_float(row, "strict_wilson_high")
        color = PALETTE["blue"] if label == "SAC" else PALETTE["teal"]
        axes[0].errorbar(
            strict,
            ypos,
            xerr=[[max(0.0, strict - low)], [max(0.0, high - strict)]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            linewidth=1.8,
            zorder=3,
        )
        axes[0].text(strict + 0.0035, ypos, pct(strict), va="center", fontsize=8.5, color=PALETTE["dark"])
    axes[0].axvline(sac_strict, color=PALETTE["blue"], linestyle="--", linewidth=1.4)
    axes[0].axvspan(sac_strict - 0.02, sac_strict + 0.02, color=PALETTE["blue"], alpha=0.08, linewidth=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label for label, _ in stable_rows], fontsize=9)
    setup_ax(axes[0], "Near-tied at 10k")
    axes[0].set_xlabel("strict success")
    axes[0].set_xlim(0.635, 0.725)
    axes[0].grid(axis="x", alpha=0.5)
    axes[0].grid(axis="y", visible=False)
    axes[0].text(
        0.01,
        -0.18,
        "Dashed line is SAC. Full official is 67.6% vs SAC 69.3%; the Wilson intervals overlap.",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color=PALETTE["gray"],
    )

    y2 = np.arange(len(broken_rows))[::-1]
    h = 0.24
    for ypos, (label, row) in zip(y2, broken_rows):
        strict = as_float(row, "strict")
        collapse = as_float(row, "collapse", 0.0)
        axes[1].barh(ypos + h / 2, strict, height=h, color=PALETTE["teal"], label="strict success" if ypos == y2[0] else None)
        axes[1].barh(ypos - h / 2, collapse, height=h, color=PALETTE["red"], label="collapse" if ypos == y2[0] else None)
        axes[1].text(strict + 0.012, ypos + h / 2, pct(strict), va="center", fontsize=8.5, color=PALETTE["dark"])
        axes[1].text(collapse + 0.012, ypos - h / 2, pct(collapse), va="center", fontsize=8.5, color=PALETTE["red"])
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels([label for label, _ in broken_rows], fontsize=9)
    setup_ax(axes[1], "Scale-control failures")
    axes[1].set_xlabel("rate")
    axes[1].set_xlim(0.0, 0.65)
    axes[1].grid(axis="x", alpha=0.5)
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(
        "10k component screen: official recipe is slower, missing scale controls break",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    fig.text(
        0.02,
        0.01,
        "Posthoc screen: 3 training seeds x 500 evaluation episodes. It is a debugging screen, not the exact-grid reliability frontier.",
        fontsize=8.8,
        color=PALETTE["gray"],
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_component_tradeoff(relative: list[dict[str, str]], path: Path) -> Path:
    wanted = [
        ("SAC50k", "SAC 50k"),
        ("FullSimba50k", "Full categorical"),
        ("FullSimbaNoDist50k", "Scalar critic"),
        ("FullSimbaNoDistUTD2_50k", "Scalar UTD2"),
    ]
    rows = [(name, find_row(relative, label=label)) for label, name in wanted]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for ypos, (name, row) in zip(y, rows):
        task = as_float(row, "task_rate")
        task_ci = as_float(row, "task_seed_ci95_half_width")
        match = as_float(row, "near_best_known_rate")
        match_ci = as_float(row, "near_best_known_seed_ci95_half_width")
        ax.plot([task, match], [ypos, ypos], color=PALETTE["line"], linewidth=3, zorder=1)
        ax.errorbar(
            task,
            ypos + 0.07,
            xerr=task_ci,
            fmt="o",
            color=PALETTE["green"],
            ecolor=PALETTE["green"],
            capsize=3,
            label="task" if ypos == y[0] else None,
            zorder=3,
        )
        ax.errorbar(
            match,
            ypos - 0.07,
            xerr=match_ci,
            fmt="s",
            color=PALETTE["blue"],
            ecolor=PALETTE["blue"],
            capsize=3,
            label="reference success" if ypos == y[0] else None,
            zorder=3,
        )
        ax.text(task + 0.008, ypos + 0.18, pct(task), fontsize=8, color=PALETTE["green"])
        ax.text(match + 0.008, ypos - 0.24, pct(match), fontsize=8, color=PALETTE["blue"])
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in rows], fontsize=10)
    setup_ax(ax, "50k exact-grid component tradeoff", None, (None if False else ()))
    ax.set_xlabel("rate, with seed-level 95% intervals")
    ax.set_xlim(0.55, 1.02)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_exploration_optimization(
    analysis: dict[str, object],
    diagnostics: list[dict[str, str]],
    path: Path,
) -> Path:
    policies = analysis["policies"]  # type: ignore[index]
    sac_diag = first_row(diagnostics, lambda row: row.get("condition") == "100k SAC")
    simba_diag = first_row(diagnostics, lambda row: row.get("condition") == "100k Full SimbaV2 official opt")
    names = ["SAC 100k", "Full SimbaV2 100k"]
    colors = [PALETTE["blue"], PALETTE["teal"]]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    replay = [as_float(sac_diag, "replay_near_any"), as_float(simba_diag, "replay_near_any")]
    axes[0].bar(names, replay, color=colors, width=0.58)
    setup_ax(axes[0], "Replay coverage is tied", "near-upright replay", (0.0, 1.0))
    annotate_bars(axes[0], axes[0].containers[0])
    axes[0].tick_params(axis="x", labelrotation=12)

    feasible_task = [
        float(policies["SAC 100k"]["task_feasible"]["mean"]),  # type: ignore[index]
        float(policies["Full SimbaV2 100k"]["task_feasible"]["mean"]),  # type: ignore[index]
    ]
    feasible_ci = [
        float(policies["SAC 100k"]["task_feasible"]["ci95"]),  # type: ignore[index]
        float(policies["Full SimbaV2 100k"]["task_feasible"]["ci95"]),  # type: ignore[index]
    ]
    axes[1].bar(names, feasible_task, color=colors, width=0.58)
    axes[1].errorbar(names, feasible_task, yerr=feasible_ci, fmt="none", ecolor=PALETTE["dark"], capsize=4)
    setup_ax(axes[1], "Reliability separates", "known-feasible task", (0.0, 1.04))
    axes[1].tick_params(axis="x", labelrotation=12)
    for idx, value in enumerate(feasible_task):
        axes[1].text(idx, min(1.03, value + 0.03), pct(value), ha="center", fontsize=9)

    x = np.arange(2)
    dormant = [as_float(sac_diag, "q1_dormant"), as_float(simba_diag, "q1_dormant")]
    rank = [as_float(sac_diag, "q1_rank"), as_float(simba_diag, "q1_rank")]
    axes[2].bar(x - 0.18, dormant, width=0.35, color=PALETTE["red"], label="dormant fraction")
    axes[2].bar(x + 0.18, rank, width=0.35, color=PALETTE["purple"], label="effective rank")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=12)
    setup_ax(axes[2], "Critic health separates", "fraction / normalized rank", (0.0, 0.60))
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_compute_negative(
    relative: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    path: Path,
) -> Path:
    rows = [
        ("SAC 100k", find_row(relative, label="SAC100k"), first_row(diagnostics, lambda r: r["condition"] == "100k SAC")),
        (
            "SAC 500k",
            find_row(relative, label="Legacy500kUTD1"),
            first_row(diagnostics, lambda r: r["condition"] == "Legacy 500k UTD1 CleanRL SAC"),
        ),
        (
            "Full SimbaV2 100k",
            find_row(relative, label="FullSimba100k"),
            first_row(diagnostics, lambda r: r["condition"] == "100k Full SimbaV2 official opt"),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.35, 0.75]})

    sac500 = rows[1][1]
    simba100 = rows[2][1]
    comparisons = [
        ("reference success", "near_best_known_rate", "near_best_known_seed_ci95_half_width"),
        ("task-stability", "task_rate", "task_seed_ci95_half_width"),
        ("near-down task", "near_down_task_rate", ""),
    ]
    y = np.arange(len(comparisons))[::-1]
    for ypos, (label, key, ci_key) in zip(y, comparisons):
        sac_value = as_float(sac500, key)
        simba_value = as_float(simba100, key)
        sac_ci = as_float(sac500, ci_key, 0.0) if ci_key else 0.0
        simba_ci = as_float(simba100, ci_key, 0.0) if ci_key else 0.0
        axes[0].plot([sac_value, simba_value], [ypos, ypos], color=PALETTE["line"], linewidth=3, zorder=1)
        axes[0].errorbar(
            sac_value,
            ypos + 0.045,
            xerr=sac_ci,
            fmt="s",
            color=PALETTE["blue"],
            ecolor=PALETTE["blue"],
            capsize=3 if ci_key else 0,
            markersize=7,
            zorder=3,
            label="SAC 500k" if ypos == y[0] else None,
        )
        axes[0].errorbar(
            simba_value,
            ypos - 0.045,
            xerr=simba_ci,
            fmt="o",
            color=PALETTE["teal"],
            ecolor=PALETTE["teal"],
            capsize=3 if ci_key else 0,
            markersize=7,
            zorder=3,
            label="Full SimbaV2 100k" if ypos == y[0] else None,
        )
        delta_pp = 100.0 * (sac_value - simba_value)
        delta_color = PALETTE["green"] if delta_pp >= 0.0 else PALETTE["red"]
        delta_text = f"{delta_pp:+.1f} pp"
        axes[0].text(0.505, ypos, delta_text, va="center", ha="left", fontsize=9, color=delta_color, fontweight="bold")
        axes[0].text(sac_value + 0.008, ypos + 0.045, pct(sac_value), va="center", fontsize=8, color=PALETTE["blue"])
        axes[0].text(simba_value + 0.008, ypos - 0.045, pct(simba_value), va="center", fontsize=8, color=PALETTE["teal"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label for label, _, _ in comparisons], fontsize=10)
    setup_ax(axes[0], "SAC 500k minus Full SimbaV2 100k")
    axes[0].set_xlabel("exact-grid rate")
    axes[0].set_xlim(0.50, 1.02)
    axes[0].grid(axis="x", alpha=0.55)
    axes[0].grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)

    x = np.arange(len(rows))
    dormant = [as_float(diag, "q1_dormant") for _, _, diag in rows]
    axes[1].bar(x, dormant, color=[PALETTE["blue"], PALETTE["blue"], PALETTE["teal"]], width=0.58)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name for name, _, _ in rows], rotation=10)
    setup_ax(axes[1], "Critic dormancy stays bad", "Q1 dormant fraction", (0.0, 0.85))
    annotate_bars(axes[1], axes[1].containers[0])
    fig.suptitle(
        "More SAC compute helps return matching, not the reliability failure mode",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hard_interventions(relative: list[dict[str, str]], path: Path) -> Path:
    wanted = [
        ("FullSimba50k", "ordinary\n50k"),
        ("FullSimbaHardReset02_50k", "hard reset\np=0.2"),
        ("FullSimbaHardReplay02_50k", "hard replay\np=0.2"),
    ]
    rows = [(name, find_row(relative, label=label)) for label, name in wanted]
    metrics = [
        ("task_rate", "task-stability", PALETTE["green"]),
        ("near_best_known_rate", "reference success", PALETTE["blue"]),
        ("near_down_task_rate", "near-down task", PALETTE["orange"]),
    ]
    x = np.arange(len(rows))
    width = 0.22
    offsets = [-width, 0.0, width]
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    for offset, (key, label, color) in zip(offsets, metrics):
        values = [as_float(row, key) for _, row in rows]
        bars = ax.bar(x + offset, values, width=width, color=color, label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(1.01, value + 0.012),
                pct(value),
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["dark"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in rows], fontsize=10)
    setup_ax(ax, "Hard-start interventions: categorical variants, not a progression", "exact-grid rate", (0.55, 1.02))
    ax.grid(axis="y", alpha=0.55)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    ax.text(
        0.0,
        -0.32,
        "Near-down task is task-stability restricted to |theta| >= 150 deg starts.",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["gray"],
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_assets(assets: list[Asset]) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for asset in assets:
        if asset.source.exists():
            shutil.copyfile(asset.source, asset.dest)
            copied[asset.name] = asset.dest
    return copied


def build_pendulum_gif(path: Path) -> str:
    try:
        import imageio.v2 as imageio

        from last_nine_rl.checkpoints import load_agent_from_run
        from last_nine_rl.envs import make_env
    except Exception as exc:  # pragma: no cover - presentation fallback.
        fallback_pendulum_gif(path)
        return f"Built fallback illustrative pendulum gif because imports failed: {exc}"

    try:
        sac_run = ROOT / "runs" / "week3_simbav2_scale_100k_20260526" / "sac" / "seed0"
        simba_run = ROOT / "runs" / "week3_simbav2_scale_100k_20260526" / "simba_full_official_opt" / "seed0"
        sac_agent, sac_cfg, _ = load_agent_from_run(sac_run, device="cpu")
        simba_agent, simba_cfg, _ = load_agent_from_run(simba_run, device="cpu")

        contrast = choose_rollout_contrast_state()
        if contrast is None:
            candidates = candidate_initial_states()
            theta, theta_dot = choose_interesting_state(sac_agent, sac_cfg, simba_agent, simba_cfg, candidates)
            state_note = (
                "fallback DP-feasible hard-boundary band "
                f"at |theta|={abs(theta) * 180.0 / math.pi:.1f} degrees, theta_dot={theta_dot:.2f}"
            )
        else:
            theta, theta_dot, state_note = contrast
        sac_trace = rollout_trace(sac_agent, sac_cfg, theta, theta_dot, seed=701, steps=200)
        simba_trace = rollout_trace(simba_agent, simba_cfg, theta, theta_dot, seed=701, steps=200)
        frames = []
        for idx in range(0, len(sac_trace), 3):
            left = draw_pendulum_panel(sac_trace[idx], "SAC 100k", width=390, height=300, outcome="task fail")
            right = draw_pendulum_panel(
                simba_trace[idx],
                "Full SimbaV2 100k",
                width=390,
                height=300,
                outcome="task stable",
            )
            canvas = Image.new("RGB", (780, 300), PALETTE["paper"])
            canvas.paste(left, (0, 0))
            canvas.paste(right, (390, 0))
            frames.append(np.asarray(canvas))
        imageio.mimsave(path, frames, duration=0.08, loop=0)
        return f"Built policy GIF from {state_note}: {path}"
    except Exception as exc:  # pragma: no cover - presentation fallback.
        fallback_pendulum_gif(path)
        return f"Built fallback illustrative pendulum gif because checkpoint rollout failed: {exc}"


def choose_rollout_contrast_state() -> tuple[float, float, str] | None:
    if not SAC_RELATIVE_ROLLOUTS_CSV.exists() or not SIMBA_RELATIVE_ROLLOUTS_CSV.exists():
        return None

    sac_rows = read_rows(SAC_RELATIVE_ROLLOUTS_CSV)
    simba_rows = read_rows(SIMBA_RELATIVE_ROLLOUTS_CSV)
    simba_by_key = {(row.get("actual_seed"), row.get("theta"), row.get("theta_dot")): row for row in simba_rows}
    candidates: list[tuple[float, float, dict[str, str], dict[str, str]]] = []
    for sac in sac_rows:
        if sac.get("actual_seed") != "0":
            continue
        simba = simba_by_key.get((sac.get("actual_seed"), sac.get("theta"), sac.get("theta_dot")))
        if simba is None:
            continue
        if flag(sac, "task_success") or not flag(simba, "task_success"):
            continue
        if flag(sac, "near_best_known_return_eps") or not flag(simba, "near_best_known_return_eps"):
            continue

        near_gap = as_float(simba, "near_upright_fraction", 0.0) - as_float(sac, "near_upright_fraction", 0.0)
        return_gap = as_float(simba, "return", 0.0) - as_float(sac, "return", 0.0)
        regret_gap = as_float(sac, "regret_to_best_known", 0.0) - as_float(simba, "regret_to_best_known", 0.0)
        score = 120.0 * near_gap + 0.05 * return_gap + 0.20 * regret_gap
        theta_deg = abs(as_float(sac, "theta_degrees", 0.0))
        candidates.append((score, theta_deg, sac, simba))

    if not candidates:
        return None

    hard_candidates = [item for item in candidates if item[1] >= 150.0]
    _, _, sac, simba = max(hard_candidates or candidates, key=lambda item: (item[1], item[0]))
    theta = as_float(sac, "theta")
    theta_dot = as_float(sac, "theta_dot")
    theta_deg = as_float(sac, "theta_degrees", theta * 180.0 / math.pi)
    return_gap = as_float(simba, "return") - as_float(sac, "return")
    note = (
        "exact-grid contrast where SAC seed0 fails and full SimbaV2 seed0 succeeds "
        f"(theta={theta_deg:.1f} deg, theta_dot={theta_dot:.2f}, return gap={return_gap:+.1f})"
    )
    return theta, theta_dot, note


def flag(row: dict[str, str], key: str) -> bool:
    return as_float(row, key, 0.0) >= 0.5


def candidate_initial_states() -> list[tuple[float, float]]:
    states: list[tuple[float, float]] = []
    # Use the same DP-feasible hard-boundary band as the curriculum checks:
    # 120-135 degrees from upright, low velocity. This links the GIF to the
    # failure region in the initial-state maps instead of showing a random start.
    for abs_theta in np.linspace(2.0943951023931953, 2.356194490192345, 5):
        for sign in (-1.0, 1.0):
            for theta_dot in (-1.0, 0.0, 1.0):
                states.append((float(sign * abs_theta), float(theta_dot)))
    return states


def choose_interesting_state(
    sac_agent,
    sac_cfg,
    simba_agent,
    simba_cfg,
    states: list[tuple[float, float]],
) -> tuple[float, float]:
    best_state = states[0]
    best_gap = -1e9
    for theta, theta_dot in states:
        sac_score = rollout_score(sac_agent, sac_cfg, theta, theta_dot, seed=621)
        simba_score = rollout_score(simba_agent, simba_cfg, theta, theta_dot, seed=621)
        gap = simba_score - sac_score
        if gap > best_gap:
            best_gap = gap
            best_state = (theta, theta_dot)
    return best_state


def rollout_score(agent, cfg, theta: float, theta_dot: float, seed: int) -> float:
    trace = rollout_trace(agent, cfg, theta, theta_dot, seed=seed, steps=200)
    near = [1.0 if obs[0] >= 0.95 and abs(obs[2]) <= 1.0 else 0.0 for obs, _, _, _ in trace[-60:]]
    ret = trace[-1][2]
    return 100.0 * float(np.mean(near)) + 0.01 * ret


def rollout_trace(agent, cfg, theta: float, theta_dot: float, seed: int, steps: int):
    from last_nine_rl.envs import make_env

    env = make_env(cfg.env.env_id, seed=seed, max_episode_steps=cfg.env.max_episode_steps)
    try:
        env.reset(seed=seed)
        env.unwrapped.state = np.asarray([theta, theta_dot], dtype=np.float64)
        env.unwrapped.last_u = None
        obs = np.asarray(env.unwrapped._get_obs(), dtype=np.float32)
        total = 0.0
        trace = []
        for step in range(steps):
            action = np.asarray(agent.act(obs, deterministic=True), dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            trace.append((np.asarray(obs, dtype=np.float32), float(action.reshape(-1)[0]), total, step))
            if terminated or truncated:
                break
        return trace
    finally:
        env.close()


def draw_pendulum_panel(entry, title: str, width: int, height: int, outcome: str | None = None) -> Image.Image:
    obs, action, total, step = entry
    theta = math.atan2(float(obs[1]), float(obs[0]))
    img = Image.new("RGB", (width, height), PALETTE["paper"])
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    pivot = (width // 2, 136)
    length = 100
    bob = (pivot[0] + int(length * math.sin(theta)), pivot[1] - int(length * math.cos(theta)))

    draw.rectangle((0, 0, width - 1, height - 1), outline=PALETTE["line"])
    draw.text((18, 16), title, fill=PALETTE["dark"], font=font)
    draw.text((18, 34), f"return {total:7.1f}   torque {action:+.2f}", fill=PALETTE["gray"], font=font)
    draw.line((pivot[0] - 120, pivot[1] + 118, pivot[0] + 120, pivot[1] + 118), fill=PALETTE["line"], width=2)
    draw.ellipse((pivot[0] - 7, pivot[1] - 7, pivot[0] + 7, pivot[1] + 7), fill=PALETTE["dark"])
    draw.line((pivot[0], pivot[1], bob[0], bob[1]), fill=PALETTE["teal"], width=8)
    draw.ellipse((bob[0] - 15, bob[1] - 15, bob[0] + 15, bob[1] + 15), fill=PALETTE["orange"], outline="#7a461f")
    near = bool(obs[0] >= 0.95 and abs(obs[2]) <= 1.0)
    badge = outcome if outcome is not None else ("near upright" if near else "recovering")
    fill = PALETTE["green"] if ("success" in badge or "stable" in badge or (outcome is None and near)) else PALETTE["red"]
    draw.rounded_rectangle((18, height - 42, 134, height - 18), radius=5, fill=fill)
    draw.text((27, height - 35), badge, fill="white", font=font)
    draw.text((width - 72, height - 35), f"t={step:03d}", fill=PALETTE["gray"], font=font)
    return img


def fallback_pendulum_gif(path: Path) -> None:
    import imageio.v2 as imageio

    frames = []
    total_steps = 72
    for step in range(total_steps):
        angle = 2.8 * math.cos(step / 10.0) * math.exp(-step / 90.0)
        obs = np.asarray([math.cos(angle), math.sin(angle), -math.sin(step / 10.0)], dtype=np.float32)
        entry = (obs, 0.0, -180.0 + step, step)
        panel = draw_pendulum_panel(entry, "Pendulum rollout", 390, 300)
        frames.append(np.asarray(panel))
    imageio.mimsave(path, frames, duration=0.08, loop=0)


def build_summary_rows(
    relative: list[dict[str, str]],
    posthoc: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    reliability: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in ["SAC100k", "FullSimba100k", "FullSimba50k", "FullSimbaHardReplay02_50k"]:
        rel = find_row(relative, label=label)
        rows.append(
            {
                "source": "relative_frontier",
                "condition": rel["condition_label"],
                "task_success": pct(as_float(rel, "task_rate")),
                "task_nines": num(as_float(rel, "task_nines"), 3),
                "near_best_known": pct(as_float(rel, "near_best_known_rate")),
                "strict_success": pct(as_float(rel, "strict_rate")),
                "near_down_task": pct(as_float(rel, "near_down_task_rate")),
            }
        )
    for budget, condition in [
        ("100k", "SAC"),
        ("100k", "Full SimbaV2 official opt"),
        ("50k hard replay p=0.2", "Full SimbaV2 official opt"),
    ]:
        row = find_row(posthoc, budget=budget, condition=condition)
        rows.append(
            {
                "source": "posthoc",
                "condition": f"{budget} {condition}",
                "task_success": pct(as_float(row, "task_success")),
                "task_nines": num(as_float(row, "task_nines"), 3),
                "near_best_known": "",
                "strict_success": pct(as_float(row, "strict_success")),
                "near_down_task": "",
            }
        )
    for condition in ["100k SAC", "100k Full SimbaV2 official opt"]:
        row = first_row(diagnostics, lambda r, c=condition: r["condition"] == c)
        rows.append(
            {
                "source": "diagnostics",
                "condition": condition,
                "task_success": "",
                "task_nines": "",
                "near_best_known": f"replay near {pct(as_float(row, 'replay_near_any'))}",
                "strict_success": f"q1 dormant {pct(as_float(row, 'q1_dormant'))}",
                "near_down_task": f"q1 rank {pct(as_float(row, 'q1_rank'))}",
            }
        )
    kept = 0
    for row in reliability:
        condition_name = f"{row.get('budget', '')} {row.get('condition', '')}"
        if "ReDo" in condition_name:
            continue
        rows.append(
            {
                "source": "reliability_frontier",
                "condition": condition_name.strip(),
                "task_success": pct(as_float(row, "task_success")),
                "task_nines": num(as_float(row, "task_nines"), 3),
                "near_best_known": f"replay near {pct(as_float(row, 'replay_near'))}",
                "strict_success": pct(as_float(row, "strict_success")),
                "near_down_task": "",
            }
        )
        kept += 1
        if kept >= 6:
            break
    return rows


def build_scientific_summary_rows(
    analysis: dict[str, object],
    relative: list[dict[str, str]],
    posthoc: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    policies = analysis["policies"]  # type: ignore[index]
    for policy_name in ["SAC 100k", "Full SimbaV2 100k"]:
        for metric_key, criterion in [
            ("task_all", "task-stability success on all 2501 grid cells"),
            ("task_feasible", "task-stability success on known-feasible cells"),
            ("return_match_all", "reference success on all grid cells"),
        ]:
            metric = policies[policy_name][metric_key]  # type: ignore[index]
            rows.append(
                {
                    "source": "computed_from_relative_rollouts",
                    "condition": policy_name,
                    "criterion": criterion,
                    "rate": pct(float(metric["mean"])),
                    "seed_ci95_half_width": fmt_pp(float(metric["ci95"])),
                    "count": f"{metric['count']}/{metric['total']}",
                    "seed_rates": ";".join(f"{100.0 * value:.1f}%" for value in metric["seed_rates"]),  # type: ignore[index]
                }
            )
    rows.append(
        {
            "source": "computed_from_dp_and_controller",
            "condition": "known-feasible mask",
            "criterion": "DP task-stability success or hand-controller task-stability success",
            "rate": pct(float(analysis["feasible_cells"]) / float(analysis["total_cells"])),
            "count": f"{analysis['feasible_cells']}/{analysis['total_cells']}",
            "note": f"{analysis['uncertified_cells']} cells are uncertified, not proven impossible",
        }
    )
    for label in [
        "SAC100k",
        "FullSimba100k",
        "Legacy500kUTD1",
        "FullSimba50k",
        "FullSimbaHardReset02_50k",
        "FullSimbaHardReplay02_50k",
    ]:
        row = find_row(relative, label=label)
        rows.append(
            {
                "source": "relative_frontier",
                "condition": row["condition_label"],
                "criterion": "exact-grid all-cell summary",
                "task_rate": pct(as_float(row, "task_rate")),
                "task_seed_ci95_half_width": fmt_pp(as_float(row, "task_seed_ci95_half_width")),
                "return_match_rate": pct(as_float(row, "near_best_known_rate")),
                "return_match_seed_ci95_half_width": fmt_pp(as_float(row, "near_best_known_seed_ci95_half_width")),
                "near_down_task_rate": pct(as_float(row, "near_down_task_rate")),
            }
        )
    for condition in ["100k SAC", "100k Full SimbaV2 official opt", "Legacy 500k UTD1 CleanRL SAC"]:
        row = first_row(diagnostics, lambda r, condition=condition: r["condition"] == condition)
        rows.append(
            {
                "source": "diagnostics",
                "condition": condition,
                "criterion": "replay and critic health",
                "replay_near_any": pct(as_float(row, "replay_near_any")),
                "q1_dormant": pct(as_float(row, "q1_dormant")),
                "q1_rank": pct(as_float(row, "q1_rank")),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    return html.escape(path.relative_to(OUT).as_posix())


def deck_img(path: Path, alt: str, cls: str = "") -> str:
    return f'<img class="{cls}" src="{rel(path)}" alt="{html.escape(alt)}">'


def build_deck(generated: dict[str, Path], copied: dict[str, Path], gif_path: Path, analysis: dict[str, object]) -> str:
    policies = analysis["policies"]  # type: ignore[index]
    sac_all = policies["SAC 100k"]["task_all"]  # type: ignore[index]
    simba_all = policies["Full SimbaV2 100k"]["task_all"]  # type: ignore[index]
    sac_feasible = policies["SAC 100k"]["task_feasible"]  # type: ignore[index]
    simba_feasible = policies["Full SimbaV2 100k"]["task_feasible"]  # type: ignore[index]
    sac_return = policies["SAC 100k"]["return_match_all"]  # type: ignore[index]
    simba_return = policies["Full SimbaV2 100k"]["return_match_all"]  # type: ignore[index]
    feasible_rate = float(analysis["feasible_cells"]) / float(analysis["total_cells"])
    representation_backup = copied.get("representation_health", generated["exploration"])

    slides = [
        slide(
            "More Nines in Deep RL Reliability",
            "Workshop update: Project 15, Week 3",
            f"""
            <div class="hero-grid">
              <div>
                <div class="kicker">Goal</div>
                <h1>Make easy control tasks reliable, not just high return on average.</h1>
                <p class="lead">Pendulum should be close to solved. The interesting failures are the last starts where a neural SAC policy still does not swing up and stabilize.</p>
                <div class="metric-row">
                  <div class="metric"><b>{pct(float(sac_return["mean"]))}</b><span>SAC 100k: within 5 of max(DP, controller)</span></div>
                  <div class="metric"><b>{pct(float(simba_return["mean"]))}</b><span>SimbaV2 100k: within 5 of max(DP, controller)</span></div>
                  <div class="metric"><b>{pct(float(sac_all["mean"]))}</b><span>SAC 100k: task-stability predicate</span></div>
                  <div class="metric"><b>{pct(float(simba_all["mean"]))}</b><span>SimbaV2 100k: task-stability predicate</span></div>
                </div>
              </div>
              <div>
                {deck_img(gif_path, "Exact-grid fail/success rollout comparison", "hero-gif")}
                <p class="caption">GIF: exact grid start initial angle=-174.1 deg, angular velocity=-1.00; SAC seed0 fails, full SimbaV2 seed0 succeeds.</p>
              </div>
            </div>
            """,
            speaker="A",
            time="0:00-0:45",
        ),
        slide(
            "What Is Success?",
            "Why the headline metric uses max(DP, controller)",
            f"""
            <div class="definition-grid">
              <div>
                <h2>First attempt: intuitive task success</h2>
                <p>We tried a behavioral criterion: swing up and stay near upright.</p>
                <ul>
                  <li>Near upright if <code>cos(theta) &gt;= 0.95</code> and <code>|theta_dot| &lt;= 1.0</code>.</li>
                  <li>Stable if near-upright fraction is at least 80% during the 200-step rollout.</li>
                  <li>No long loss if the longest not-near-upright streak is at most 50 steps.</li>
                </ul>
              </div>
              <div>
                <h2>Problem: not feasible everywhere</h2>
                <p>Our DP/controller references satisfy that task-stability predicate on <b>{analysis["feasible_cells"]}/{analysis["total_cells"]}</b> cells ({pct(feasible_rate)}). The other <b>{analysis["uncertified_cells"]}</b> cells are not certified by either reference, so task-stability alone is not a fair universal success definition.</p>
              </div>
              <div class="wide-card">
                <h2>Final headline success: match the best same-start reference</h2>
                <p><code>R(policy, s0) &gt;= max(R_DP(s0), R_controller(s0)) - 5</code>.</p>
                <p>DP is the stronger return reference on <b>2403/2501</b> cells (96.1%); the hand controller is stronger on <b>98/2501</b> cells (3.9%). We use the max because DP is approximate and the controller fixes some regions.</p>
              </div>
            </div>
            """,
            speaker="A",
            time="0:40-1:35",
        ),
        slide(
            "Evaluation Protocol",
            "What was run and how each plotted number is counted",
            f"""
            <div class="two-col">
              <div>
                <h2>Exact initial-state grid</h2>
                <ul>
                  <li>61 angle bins x 41 velocity bins = <b>2501</b> reset-support states.</li>
                  <li>Three training seeds per current 100k comparison, so <b>7503</b> deterministic grid rollouts.</li>
                  <li>Map color means fraction of seeds satisfying the plotted criterion in that exact cell: 0%, 33%, 67%, or 100%.</li>
                </ul>
              </div>
              <div>
                <h2>Reference and diagnostics</h2>
                <ul>
                  <li>DP: finite-horizon value iteration on a 241 x 161 state grid and 81 torque actions, horizon 200.</li>
                  <li>Controller: energy swing-up plus local PD stabilization.</li>
                  <li>Trackers: replay near-upright coverage, critic dormancy, and effective rank.</li>
                </ul>
              </div>
            </div>
            """,
            speaker="A",
            time="1:35-2:15",
        ),
        slide(
            "What Did SimbaV2 Change?",
            "Four concrete changes relative to the CleanRL SAC baseline",
            """
            <div class="change-grid">
              <div><h2>1. Hyperspherical feature normalization</h2><p>Replace layer norm-style scale learning with L2 normalization, so hidden feature vectors keep a fixed length instead of drifting to larger or smaller magnitudes.</p></div>
              <div><h2>2. Hyperspherical weight normalization</h2><p>Remove weight decay and project selected weights back to unit norm after each gradient update. The optimizer mainly changes directions, not raw scale.</p></div>
              <div><h2>3. Distributional critic + reward scaling</h2><p>Predict a distribution over return bins instead of one Q value. Scale rewards so the critic target stays near unit variance and gradient norms are less sensitive to outliers.</p></div>
              <div><h2>4. Official SimbaV2 SAC recipe</h2><p>Use the paper-style optimizer and entropy settings rather than the default CleanRL SAC hyperparameters. This makes early learning slower but is meant to be more stable.</p></div>
            </div>
            <table class="compact-table recipe-table">
              <thead><tr><th>Parameter</th><th>CleanRL SAC baseline</th><th>Full SimbaV2 run</th></tr></thead>
              <tbody>
                <tr><td>Q / actor learning rate</td><td>1e-3 / 3e-4</td><td>1e-4 -> 5e-5 / 1e-4 -> 5e-5</td></tr>
                <tr><td>Initial entropy weight alpha</td><td>1.0</td><td>0.01</td></tr>
                <tr><td>Target entropy scale</td><td>-1.0</td><td>-0.5</td></tr>
                <tr><td>Network size</td><td>actor 128, critic 512</td><td>Simba actor 32, critic 64</td></tr>
                <tr><td>Critic target</td><td>scalar Q value</td><td>51-bin distribution + reward scaling</td></tr>
              </tbody>
            </table>
            <p class="takeaway">We should not claim yet which of these four changes drives the 100k improvement. The component ablations are now a next experiment at representative budget.</p>
            """,
            speaker="A",
            time="2:15-3:05",
        ),
        slide(
            "Raw Plots",
            "The same starts under the two success criteria",
            f"""
            <div class="figure-full">{deck_img(generated["raw_maps"], "Raw exact-grid maps for task-stability, reference success, and regret")}</div>
            <p class="takeaway">Middle column is the headline success metric. Right column shows return points below max(DP, controller); colors saturate at 20 because rare outliers exceed 100 and would wash out the map.</p>
            """,
            speaker="A",
            time="3:05-3:50",
        ),
        slide(
            "First Result",
            "Full SimbaV2 is the current 100k reference-success frontier",
            f"""
            <div class="figure-full">{deck_img(generated["main_result"], "100k seed-level reliability comparison")}</div>
            <p class="takeaway">Reference success: {fmt_metric(sac_return)} for SAC vs {fmt_metric(simba_return)} for SimbaV2. Task-stability all-grid: {fmt_metric(sac_all)} vs {fmt_metric(simba_all)}.</p>
            """,
            speaker="B",
            time="3:50-4:35",
        ),
        slide(
            "Exploration vs Optimization",
            "Replay coverage is necessary, but not sufficient",
            f"""
            <div class="two-col wide-left">
              <div>{deck_img(generated["exploration"], "Exploration versus optimization diagnostics")}</div>
              <div>
                <h2>Current diagnosis</h2>
                <ul>
                  <li>Replay near-upright coverage: SAC {pct(as_float(analysis["diagnostics"]["100k SAC"], "replay_near_any"))}, SimbaV2 {pct(as_float(analysis["diagnostics"]["100k Full SimbaV2 official opt"], "replay_near_any"))}.</li>
                  <li>Dormant critic units are units that barely activate; lower is better.</li>
                  <li>Effective rank is a feature-diversity proxy; higher is better.</li>
                  <li>Same replay coverage but different critic health points to optimization and plasticity, not pure exploration.</li>
                </ul>
              </div>
            </div>
            """,
            speaker="B",
            time="4:35-5:25",
        ),
        slide(
            "What Did Not Solve It?",
            "More SAC compute improves return matching, but still misses task reliability",
            f"""
            <div class="two-col wide-left">
              <div>{deck_img(generated["compute_negative"], "Compute negative result")}</div>
              <div>
                <h2>Why this is negative</h2>
                <ul>
                  <li>Reference success is not the problem: SAC 500k is 96.3% vs Full SimbaV2 100k at 92.5% (+3.8 pp).</li>
                  <li>Task-stability is still worse: 88.6% vs 91.4% (-2.8 pp), and near-down task success is 58.4% vs 70.6% (-12.2 pp).</li>
                  <li>Critic dormancy worsens with compute: SAC 500k has 77.3% dormant Q1 units vs 0.0% for Full SimbaV2 100k.</li>
                </ul>
              </div>
            </div>
            """,
            speaker="B",
            time="5:25-6:05",
        ),
        slide(
            "What Else Did Not Solve It?",
            "Hard reset and hard replay are categorical data interventions",
            f"""
            <div class="two-col wide-left">
              <div>{deck_img(generated["hard_interventions"], "Hard-state intervention result")}</div>
              <div>
                <h2>Definitions</h2>
                <ul>
                  <li>Hard reset p=0.2: 20% of training resets are forced to |theta|=120-135 deg, |theta_dot|<=1.</li>
                  <li>Hard replay p=0.2: 20% of each replay update batch is sampled from transitions in that same hard-start band.</li>
                  <li>Near-down task: task-stability on evaluation starts with |theta|>=150 deg.</li>
                </ul>
                <h2>Result</h2>
                <ul>
                  <li>Hard reset p=0.2 helps 50k task-stability slightly, but reference success falls sharply.</li>
                  <li>Hard replay p=0.2 is worse on both task-stability and reference success than ordinary full 50k.</li>
                </ul>
              </div>
            </div>
            """,
            speaker="B",
            time="6:05-6:45",
        ),
        slide(
            "Next Experiments",
            "Move the ablations to representative budget",
            """
            <div class="next-grid">
              <div>
                <h2>1. Ablate SimbaV2 at 100k</h2>
                <p><b>Experiment:</b> remove one component at a time: feature normalization, weight projection, distributional critic, reward scaling, and official SAC recipe.</p>
                <p><b>Decision:</b> identify which change actually drives the reliability gain, using seed-level intervals and exact-grid maps.</p>
              </div>
              <div>
                <h2>2. Push Pendulum toward 0.99</h2>
                <p><b>Experiment:</b> try plasticity and reliability fixes: ReDo, Sample Weight Decay, Fisher-guided selective forgetting, and regret-weighted auxiliary losses.</p>
                <p><b>Decision:</b> keep only changes that improve reference success, task-stability, and near-down starts without damaging critic health.</p>
              </div>
              <div>
                <h2>3. Move to CartPole-Swingup</h2>
                <p><b>Experiment:</b> reuse the exact-grid/replay/critic-health protocol on CartPole-Swingup after the Pendulum frontier is stable.</p>
                <p><b>Decision:</b> test whether the Pendulum diagnosis transfers when exploration is genuinely harder.</p>
              </div>
              <div>
                <h2>Why not present 50k ablations?</h2>
                <p>Short-budget component screens were too noisy for a scientific claim about which SimbaV2 piece helps. We keep them as debugging evidence, not as a talk result.</p>
              </div>
            </div>
            """,
            speaker="B",
            time="6:45-8:00",
        ),
        slide(
            "Backup: DP Reference",
            "What DP means in this project",
            """
            <div class="backup-grid">
              <div>
                <h2>Dynamic programming setup</h2>
                <ul>
                  <li>Uses the known Gymnasium Pendulum-v1 transition equation and reward.</li>
                  <li>No neural model is learned; Bellman backups are computed on a grid.</li>
                  <li>Horizon 200, torque limit 2.0, velocity limit 8.0.</li>
                  <li>State grid: 241 theta x 161 theta_dot; action grid: 81 torques.</li>
                </ul>
              </div>
              <div>
                <h2>Why it is only an approximate reference</h2>
                <ul>
                  <li>Continuous state and action are discretized; next-state values use bilinear interpolation.</li>
                  <li>It is finite-horizon and optimizes Gym return, not the task-stability predicate directly.</li>
                  <li>DP is better on 2403/2501 cells; the controller is better on 98/2501 cells.</li>
                  <li>That is why reference success uses max(DP, controller), and task-stability feasibility is reported separately.</li>
                </ul>
              </div>
            </div>
            """,
            speaker="Q",
            time="Q&A",
        ),
        slide(
            "Backup: Hand Controller",
            "Energy shaping plus a local PD switch",
            """
            <div class="backup-grid">
              <div>
                <h2>Control law used</h2>
                <div class="equation">
                  E = 0.5 dot(theta)^2 + E_u cos(theta), &nbsp; E_u = 3g/(2l)
                </div>
                <div class="equation">
                  u = clip[-2,2](-k_p theta - k_d dot(theta))<br>
                  if |theta| <= 0.4 and |dot(theta)| <= 3
                </div>
                <div class="equation">
                  u = clip[-2,2](-k_E(E - E_u)dot(theta) - 0.5 sign(theta) 1[|dot(theta)| &lt; 0.1])<br>
                  otherwise
                </div>
                <p class="caption">Gains: k_E=2, k_p=9, k_d=3. Implemented in <code>reference.py</code>.</p>
              </div>
              <div>
                <h2>Why it is in the reference</h2>
                <ul>
                  <li>Energy-shaping swing-up follows the classic Astrom-Furuta energy-control idea.</li>
                  <li>The PD branch is our local stabilizer near the upright equilibrium.</li>
                  <li>Reference success uses <code>max(DP, controller)</code> from the same initial state.</li>
                  <li>The controller is better than DP on 98/2501 grid cells.</li>
                  <li>It does not certify impossibility when it fails.</li>
                </ul>
                <p class="caption">Citation: K. J. Astrom and K. Furuta, "Swinging up a pendulum by energy control," Automatica, 36(2), 287-295, 2000.</p>
              </div>
            </div>
            """,
            speaker="Q",
            time="Q&A",
        ),
        slide(
            "Backup: Bad SAC Seed",
            "Why the SAC 100k interval is so wide",
            """
            <div class="backup-grid">
              <div>
                <h2>What happened</h2>
                <table class="data-table">
                  <tr><th>SAC 100k seed</th><th>reference</th><th>task</th><th>replay near</th></tr>
                  <tr><td>seed 0</td><td>52.4%</td><td>45.8%</td><td>82.1%</td></tr>
                  <tr><td>seed 1</td><td>93.0%</td><td>92.2%</td><td>82.0%</td></tr>
                  <tr><td>seed 2</td><td>92.2%</td><td>89.3%</td><td>82.5%</td></tr>
                </table>
                <ul>
                  <li>Seed 0 creates the 57.6 pp SAC seed interval.</li>
                  <li>Replay near-upright coverage is tied, so this is not simply "did not see upright states."</li>
                </ul>
              </div>
              <div>
                <h2>Why we kept it</h2>
                <ul>
                  <li>Configs differ only by seed; the run completed 100k steps and 99k updates.</li>
                  <li>Checkpoint has no NaNs/Infs; deterministic grid evaluation matched scalar Gym rollout on checked states.</li>
                  <li>Failures are broad and mostly moderate: many cells are 10-20 return points below max(DP, controller)-5.</li>
                  <li>Interpretation: a real optimization/reliability failure, not an obvious logging or evaluator bug.</li>
                </ul>
              </div>
            </div>
            """,
            speaker="Q",
            time="Q&A",
        ),
    ]

    css = """
    :root {
      --dark: #17202a;
      --muted: #5f6b76;
      --line: #d8ddd6;
      --paper: #fbfaf5;
      --panel: #fffdf8;
      --teal: #2e7d78;
      --green: #4f8a5b;
      --orange: #c47a2c;
      --red: #b9564c;
      --blue: #376996;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #1f262d;
      color: var(--dark);
      font-family: Arial, Helvetica, sans-serif;
    }
    .deck {
      width: 100vw;
      height: 100vh;
      overflow: hidden;
    }
    .slide {
      display: none;
      width: min(100vw, 177.777vh);
      height: min(56.25vw, 100vh);
      margin: auto;
      padding: 42px 54px 34px;
      background: var(--paper);
      position: relative;
      overflow: hidden;
    }
    .slide::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 10px;
      background: var(--teal);
    }
    .slide.active { display: block; }
    .topline {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 20px;
      border-bottom: 2px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 24px;
    }
    .title h1 {
      margin: 0;
      font-size: 36px;
      line-height: 1.05;
      letter-spacing: 0;
      color: var(--dark);
    }
    .subtitle {
      margin-top: 7px;
      font-size: 17px;
      color: var(--muted);
    }
    .badge {
      display: flex;
      gap: 8px;
      align-items: center;
      color: #ffffff;
      background: var(--dark);
      padding: 8px 11px;
      font-size: 13px;
      white-space: nowrap;
    }
    h1 { font-size: 43px; line-height: 1.04; margin: 12px 0 18px; letter-spacing: 0; }
    h2 { font-size: 21px; margin: 0 0 10px; letter-spacing: 0; }
    .subhead { margin-top: 24px; }
    p, li { font-size: 20px; line-height: 1.34; }
    ul, ol { padding-left: 24px; margin: 12px 0 0; }
    li { margin: 8px 0; }
    code {
      font-family: Consolas, "Liberation Mono", monospace;
      font-size: 0.86em;
      background: #f0f3ef;
      padding: 1px 4px;
      border: 1px solid var(--line);
    }
    img { max-width: 100%; height: auto; display: block; }
    .hero-grid {
      display: grid;
      grid-template-columns: 1.02fr 0.98fr;
      gap: 32px;
      align-items: center;
    }
    .lead { font-size: 23px; color: var(--muted); max-width: 820px; }
    .kicker { color: var(--teal); font-weight: 700; text-transform: uppercase; font-size: 15px; }
    .hero-gif { width: 100%; border: 1px solid var(--line); background: var(--panel); }
    .caption { font-size: 14px; color: var(--muted); margin: 8px 0 0; }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 30px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 16px;
      min-height: 98px;
    }
    .metric b { display: block; font-size: 29px; color: var(--teal); }
    .metric span { display: block; color: var(--muted); font-size: 13px; margin-top: 8px; }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 32px;
      align-items: center;
    }
    .wide-left { grid-template-columns: 1.28fr 0.82fr; }
    .figure-full img { max-height: 560px; margin: 0 auto; }
    .definition-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: stretch;
    }
    .definition-grid > div,
    .next-grid > div,
    .backup-grid > div {
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 5px solid var(--teal);
      padding: 18px 20px;
    }
    .definition-grid p,
    .definition-grid li,
    .next-grid p,
    .backup-grid li {
      font-size: 18px;
    }
    .wide-card { grid-column: 1 / -1; }
    .next-grid,
    .backup-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: stretch;
    }
    .next-grid h2,
    .backup-grid h2 {
      color: var(--teal);
    }
    .equation {
      font-family: Consolas, "Liberation Mono", monospace;
      font-size: 16px;
      line-height: 1.35;
      background: #f0f3ef;
      border: 1px solid var(--line);
      padding: 10px 12px;
      margin: 9px 0;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      margin: 6px 0 14px;
      font-size: 17px;
    }
    .data-table th,
    .data-table td {
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      text-align: left;
    }
    .data-table th {
      color: var(--teal);
      font-weight: 700;
      background: #f0f3ef;
    }
    .map-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 22px;
      align-items: start;
    }
    .map-grid img { border: 1px solid var(--line); background: var(--panel); max-height: 505px; object-fit: contain; }
    .change-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .change-grid div {
      background: var(--panel);
      border-left: 6px solid var(--teal);
      padding: 12px 14px;
      min-height: 118px;
    }
    .change-grid h2 { font-size: 17px; margin-bottom: 7px; }
    .change-grid p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.30; }
    .change-grid .tested {
      margin-top: 12px;
      color: var(--dark);
      font-size: 15px;
      font-weight: 700;
    }
    .compact-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 13px;
      font-size: 13px;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    .compact-table th,
    .compact-table td {
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }
    .compact-table th {
      color: var(--teal);
      background: #f0f3ef;
      font-weight: 700;
    }
    .recipe-table td:first-child { width: 24%; font-weight: 700; color: var(--dark); }
    .takeaway {
      position: absolute;
      left: 54px;
      right: 54px;
      bottom: 34px;
      margin: 0;
      padding-top: 14px;
      border-top: 2px solid var(--line);
      color: var(--dark);
      font-size: 19px;
      font-weight: 700;
    }
    .footer {
      position: absolute;
      right: 54px;
      bottom: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    @media print {
      body { background: #ffffff; }
      .deck { height: auto; overflow: visible; }
      .slide { display: block; width: 13.333in; height: 7.5in; page-break-after: always; }
    }
    """

    script = """
    const slides = Array.from(document.querySelectorAll('.slide'));
    let idx = 0;
    function show(i) {
      idx = Math.max(0, Math.min(slides.length - 1, i));
      slides.forEach((s, n) => s.classList.toggle('active', n === idx));
      history.replaceState(null, '', '#' + (idx + 1));
    }
    const start = parseInt(location.hash.replace('#', ''), 10);
    if (!Number.isNaN(start)) idx = start - 1;
    show(idx);
    window.addEventListener('keydown', (e) => {
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) show(idx + 1);
      if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) show(idx - 1);
    });
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project 15 Week 3 Workshop Presentation</title>
  <style>{css}</style>
</head>
<body>
  <main class="deck">
    {''.join(slides)}
  </main>
  <script>{script}</script>
</body>
</html>
"""


def find_summary(summary: list[dict[str, str]], contains: str) -> dict[str, str]:
    for row in summary:
        if contains in row["condition"]:
            return row
    raise KeyError(contains)


def slide(title: str, subtitle: str, body: str, speaker: str, time: str) -> str:
    return f"""
    <section class="slide">
      <div class="topline">
        <div class="title">
          <h1>{html.escape(title)}</h1>
          <div class="subtitle">{html.escape(subtitle)}</div>
        </div>
        <div class="badge">Speaker {html.escape(speaker)} | {html.escape(time)}</div>
      </div>
      {body}
      <div class="footer">Project 15 | Week 3 workshop</div>
    </section>
    """


def build_notes(analysis: dict[str, object], gif_status: str) -> str:
    policies = analysis["policies"]  # type: ignore[index]
    sac_all = policies["SAC 100k"]["task_all"]  # type: ignore[index]
    simba_all = policies["Full SimbaV2 100k"]["task_all"]  # type: ignore[index]
    sac_feasible = policies["SAC 100k"]["task_feasible"]  # type: ignore[index]
    simba_feasible = policies["Full SimbaV2 100k"]["task_feasible"]  # type: ignore[index]
    sac_return = policies["SAC 100k"]["return_match_all"]  # type: ignore[index]
    simba_return = policies["Full SimbaV2 100k"]["return_match_all"]  # type: ignore[index]
    feasible_rate = float(analysis["feasible_cells"]) / float(analysis["total_cells"])
    return f"""# Project 15 Week 3 Workshop Speaker Notes

Format: 8 minutes talk plus 5 minutes questions. Speaker A covers slides 1-5. Speaker B covers slides 6-10. Slides 11-13 are backup.

## Story Arc
1. Pendulum is the controlled testbed: high average return is not enough; we care about the last failing starts.
2. Define success carefully. We first tried an intuitive task-stability definition, found that it was not feasible/certified everywhere, then switched the headline metric to same-start max(DP, controller) within epsilon.
3. Explain SimbaV2 in plain language through the four paper changes, without claiming yet which component drives the gain.
4. Show raw maps first, then seed-level statistics.
5. Diagnose exploration versus optimization using replay and critic-health trackers.
6. Close with negative results and concrete next steps.

## Key Numbers To Say Correctly
- Known-feasible cells: {analysis['feasible_cells']}/{analysis['total_cells']} = {pct(feasible_rate)}. The remaining {analysis['uncertified_cells']} cells are uncertified, not proven impossible.
- DP has the higher reference return on 2403/2501 cells; the hand controller is higher on 98/2501 cells.
- SAC 100k all-grid task: {fmt_metric(sac_all)}.
- Full SimbaV2 100k all-grid task: {fmt_metric(simba_all)}.
- SAC 100k known-feasible task: {fmt_metric(sac_feasible)}.
- Full SimbaV2 100k known-feasible task: {fmt_metric(simba_feasible)}.
- SAC 100k reference success: {fmt_metric(sac_return)}.
- Full SimbaV2 100k reference success: {fmt_metric(simba_return)}.

## Slide Timing
- Slide 1: 0:00-0:40. Goal and GIF.
- Slide 2: 0:40-1:35. Definitions and feasibility caveat.
- Slide 3: 1:35-2:15. Evaluation protocol.
- Slide 4: 2:15-3:05. SimbaV2 changes and official-recipe comparison.
- Slide 5: 3:05-3:50. Raw maps.
- Slide 6: 3:50-4:35. Main seed-level result.
- Slide 7: 4:35-5:25. Exploration versus optimization.
- Slide 8: 5:25-6:05. More compute negative result.
- Slide 9: 6:05-6:45. Hard reset/replay negative result.
- Slide 10: 6:45-8:00. Next steps.

## Q&A Backup
- DP uses the known Pendulum dynamics and reward on a discretized grid; it is an approximate reference, not a proof of optimality.
- The hand controller is energy shaping plus local PD, following Astrom-Furuta style swing-up; it is a witness and return reference, not an oracle.
- SAC 100k seed0 is a real bad seed: 52.4% reference success versus 93.0%/92.2% for seeds 1/2, with replay coverage tied.
- Seed is the statistical unit. Cell-level pooling is for maps; seed-level intervals are used for claims.

GIF status: {gif_status}
"""


def build_script(analysis: dict[str, object], gif_status: str) -> str:
    policies = analysis["policies"]  # type: ignore[index]
    sac_all = policies["SAC 100k"]["task_all"]  # type: ignore[index]
    simba_all = policies["Full SimbaV2 100k"]["task_all"]  # type: ignore[index]
    sac_feasible = policies["SAC 100k"]["task_feasible"]  # type: ignore[index]
    simba_feasible = policies["Full SimbaV2 100k"]["task_feasible"]  # type: ignore[index]
    sac_return = policies["SAC 100k"]["return_match_all"]  # type: ignore[index]
    simba_return = policies["Full SimbaV2 100k"]["return_match_all"]  # type: ignore[index]
    feasible_rate = float(analysis["feasible_cells"]) / float(analysis["total_cells"])
    return f"""# Project 15 Week 3 Workshop Script

This script is written for an 8 minute talk. Speaker A covers slides 1-5. Speaker B covers slides 6-10. Backup slides are for questions.

## Slide 1, Speaker A, 0:00-0:40
Our project is about reliability in deep reinforcement learning. Pendulum is not supposed to be a hard benchmark, so average return is not the interesting part. The interesting part is the last set of initial states where a neural SAC policy still fails to swing up and stabilize.

The GIF is not random. It is the same initial state from the exact evaluation grid: initial angle is minus 174.1 degrees and angular velocity is minus 1.0. SAC seed 0 fails there, while full SimbaV2 seed 0 succeeds. The headline success metric is within 5 return points of max(DP, controller): SAC 100k is at {pct(float(sac_return["mean"]))}, while full SimbaV2 100k is at {pct(float(simba_return["mean"]))}. The stricter all-grid task-stability check is {pct(float(sac_all["mean"]))} versus {pct(float(simba_all["mean"]))}. Seed-level intervals are on slide 6, not on the opening slide.

## Slide 2, Speaker A, 0:40-1:35
The most important thing is the success definition. We first tried an intuitive task-stability definition: the policy should swing up, stay near upright for at least 80 percent of the episode, and not lose the upright region for more than 50 consecutive steps.

That is useful, but it is not a fair universal success definition because our references do not satisfy it everywhere. DP or the hand controller satisfies that task-stability predicate on {analysis['feasible_cells']} out of {analysis['total_cells']} cells, which is {pct(feasible_rate)}. The remaining cells are not certified by either reference under that behavioral predicate.

So the headline success metric is state-conditioned return matching: the policy return must be within 5 of max(DP, controller) from the same start. Most of the time DP is the better reference: 2403 out of 2501 cells. The controller is better on 98 cells, so using the max matters.

## Slide 3, Speaker A, 1:35-2:15
The evaluation grid has 61 angle bins and 41 angular-velocity bins, so 2501 reset-support states. For the 100k comparison we have three training seeds, so 7503 deterministic grid rollouts.

In the maps, each cell is the fraction of seeds that satisfy the plotted criterion from that exact state. With three seeds, that means zero, one third, two thirds, or all seeds. The references are finite-horizon DP and an energy-swing-up plus PD hand controller. The diagnostics track replay coverage and critic representation health.

## Slide 4, Speaker A, 2:15-3:05
For listeners who have not read SimbaV2, the paper is not just "a bigger SAC network." It makes four concrete changes to SAC.

First, hyperspherical feature normalization means L2-normalizing hidden feature vectors so their scale does not drift. Second, hyperspherical weight normalization means removing weight decay and projecting selected weights back to unit norm after each update. Third, the critic becomes distributional: it predicts binned return probabilities, and reward scaling keeps the target variance in a stable range. Fourth, the paper uses a different SAC recipe from CleanRL: smaller learning rates, lower initial entropy weight, different target entropy scale, smaller Simba networks, and the distributional critic.

The table gives the exact side-by-side settings from our 100k runs. The important scientific point is that we should not claim which component drives the improvement yet. The short-budget ablations were useful for debugging, but not representative enough for the talk.

## Slide 5, Speaker A, 3:05-3:50
These are the raw maps. The left column is task-stability success. The middle column is the headline metric: within 5 return points of max(DP, controller). The right column is the continuous version of that same reference comparison: how many return points the policy is below max(DP, controller). The first row is SAC 100k and the second row is full SimbaV2 100k.

The right column uses a color cap at 20 return points. That is not changing the numbers; it only saturates the color scale. We do it because a few cells are more than 100 return points below the reference, and a full-range color scale would make all ordinary near-boundary differences look the same.

## Slide 6, Speaker B, 3:50-4:35
Here is the seed-level main result. The dots are training seeds and the diamonds are means. The intervals are 95 percent t-intervals over seeds, so they are deliberately conservative with only three seeds.

Reference success improves from {fmt_metric(sac_return)} for SAC to {fmt_metric(simba_return)} for full SimbaV2. The stricter all-grid task-stability check improves from {fmt_metric(sac_all)} to {fmt_metric(simba_all)}. On known-feasible cells, task-stability improves from {fmt_metric(sac_feasible)} to {fmt_metric(simba_feasible)}. SAC has one very bad seed, which is why its interval is huge. We should present that uncertainty explicitly.

## Slide 7, Speaker B, 4:35-5:25
This is the exploration versus optimization diagnosis. If SAC failed only because it never saw useful states, replay near-upright coverage should separate. It does not: SAC and full SimbaV2 are both about 82.6 to 82.7 percent.

The separation is critic health. Dormant units are critic units that barely activate, so lower is better. Effective rank is a proxy for how diverse the critic features are, so higher is better. SAC has high dormancy and low rank, while SimbaV2 has zero measured Q1 dormancy and much higher rank. That points to optimization, plasticity, and value estimation, not pure exploration.

## Slide 8, Speaker B, 5:25-6:05
More SAC compute is a useful negative result, but we need to state the metric carefully. SAC 500k is not worse on reference success: it reaches 96.3 percent, compared with 92.5 percent for full SimbaV2 100k.

It is still worse on the behavioral reliability metrics. Exact-grid task-stability is 88.6 percent for SAC 500k versus 91.4 percent for full SimbaV2 100k, so SAC is 2.8 percentage points behind. On near-down starts the gap is much larger: 58.4 percent versus 70.6 percent, so SAC is 12.2 percentage points behind. The critic-health signal also worsens: SAC 500k has 77.3 percent dormant Q1 units, while full SimbaV2 has 0.0 percent in this diagnostic.

## Slide 9, Speaker B, 6:05-6:45
Hard reset and hard replay test the data-distribution hypothesis more directly. Hard reset p=0.2 means 20 percent of episode resets are forced into a large-angle band: absolute theta between 120 and 135 degrees, with absolute angular velocity at most 1. Hard replay p=0.2 means 20 percent of each replay minibatch is sampled from transitions in that same hard-start band.

The graph is now a grouped bar chart because these are categorical interventions, not a progression. Ordinary full SimbaV2 50k has task-stability 89.8 percent and reference success 81.6 percent. Hard reset moves task-stability slightly to 90.3 percent, but reference success drops to 73.6 percent. Hard replay is worse on both headline metrics, at 89.0 percent task-stability and 72.0 percent reference success. Near-down task success is also worse for hard replay.

So just showing hard states more often is too blunt. The next intervention has to preserve value accuracy on the rest of the state space.

## Slide 10, Speaker B, 6:45-8:00
The next step is to move the component claims to representative budget. We should ablate away SimbaV2 components one at a time at 100k: feature normalization, weight projection, distributional critic, reward scaling, and the official SAC recipe. That tells us what actually drives the reliability improvement.

In parallel, we should push Pendulum toward at least 0.99 reference success using other reliability ideas from the proposal: ReDo, Sample Weight Decay, Fisher-guided selective forgetting, and regret-weighted auxiliary losses. After the Pendulum frontier is stable, we move the same protocol to CartPole-Swingup, where exploration is a more serious part of the problem.

## Backup / Q&A
Use the DP slide if asked whether DP is optimal. It is approximate finite-horizon DP, not a proof. Use the controller slide if asked what max(DP, controller) means. Use the bad-seed slide if asked why SAC has a huge confidence interval.

GIF status: {gif_status}
"""


if __name__ == "__main__":
    main()
