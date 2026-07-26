from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "deliverables" / "chasing_nines_20260723" / "figures"
LEDGER = (
    ROOT
    / "reports"
    / "two_repo_forensic_inventory_20260723"
    / "unique_standardized_evaluations.csv"
)

INK = "#102a35"
MUTED = "#526b76"
MIXED = "#0d9f90"
MIXED_PALE = "#ddf4ee"
PURE = "#3478ef"
PURE_PALE = "#e6efff"
ORANGE = "#ef6a3a"
GRAY = "#8193a1"
GRID = "#d9e2e5"
WHITE = "#fffefa"

N = 12_505


MIXED_NODES = [
    {
        "id": "no_refit_q",
        "label": "No learner-state\nrefit + Q-search",
        "near": 12_467,
        "strict": 1_128,
        "x": 0.4,
    },
    {
        "id": "uniform_q",
        "label": "Uniform starts\n+ Q-search",
        "near": 12_486,
        "strict": 1_459,
        "x": 1.45,
    },
    {
        "id": "priority_raw",
        "label": "Priority refit\nwithout Q-search",
        "near": 12_470,
        "strict": 1_813,
        "x": 2.5,
    },
    {
        "id": "mixed_final",
        "label": "Priority refit\n+ local Q-search",
        "near": 12_496,
        "strict": 1_570,
        "x": 3.65,
    },
]

MIXED_EDGES = [
    ("no_refit_q", "mixed_final", "+29 near", "learner-state refit"),
    ("uniform_q", "mixed_final", "+10 near", "priority sampling"),
    ("priority_raw", "mixed_final", "+26 near", "local Q-search"),
]

PURE_NODES = [
    {
        "id": "pure_raw",
        "label": "FastSACN8\nactor",
        "near": 11_238,
        "strict": 2_145,
        "x": 0.35,
    },
    {
        "id": "pure_reflect",
        "label": "+ reflection",
        "near": 11_550,
        "strict": 2_521,
        "x": 2.0,
    },
    {
        "id": "pure_final",
        "label": "+ 41-action\nQ-search",
        "near": 12_008,
        "strict": 2_854,
        "x": 3.65,
    },
]

PURE_EDGES = [
    ("pure_raw", "pure_reflect", "+312 near", "reflection"),
    ("pure_reflect", "pure_final", "+458 near", "Q-search"),
]


def failures(node: dict) -> int:
    return N - int(node["near"])


def rate(count: int) -> float:
    return 100.0 * count / N


def node_map(nodes: list[dict]) -> dict[str, dict]:
    return {str(node["id"]): node for node in nodes}


def save(fig: plt.Figure, name: str) -> None:
    path = FIG / name
    fig.savefig(path, dpi=240, facecolor=WHITE, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(path)


def rounded_node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    title: str,
    failure_count: int,
    strict_count: int,
    final: bool = False,
) -> None:
    face = color if final else WHITE
    text_color = WHITE if final else INK
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.055",
        linewidth=3.0 if final else 2.2,
        edgecolor=color,
        facecolor=face,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + 0.25 * height,
        title,
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color=text_color,
        linespacing=0.95,
        zorder=5,
    )
    ax.text(
        x,
        y - 0.04 * height,
        f"{failure_count:,} failures",
        ha="center",
        va="center",
        fontsize=29,
        fontweight="bold",
        color=ORANGE if not final else WHITE,
        zorder=5,
    )
    ax.text(
        x,
        y - 0.30 * height,
        f"{rate(N - failure_count):.3f}% near  ·  {rate(strict_count):.1f}% strict",
        ha="center",
        va="center",
        fontsize=20,
        color=text_color,
        zorder=5,
    )


def build_recipe_graph() -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15.8, 8.8),
        gridspec_kw={"height_ratios": [1.05, 0.95]},
        facecolor=WHITE,
    )

    ax = axes[0]
    ax.set_xlim(-0.35, 4.25)
    ax.set_ylim(-0.08, 1.22)
    ax.axis("off")
    ax.text(
        -0.30,
        1.15,
        "RL + supervised: three matched controls converge on nine failures",
        fontsize=30,
        fontweight="bold",
        color=MIXED,
        va="top",
    )
    positions = {
        "no_refit_q": (0.35, 0.72),
        "uniform_q": (0.35, 0.32),
        "priority_raw": (1.90, 0.52),
        "mixed_final": (3.55, 0.52),
    }
    mixed_by_id = node_map(MIXED_NODES)
    for node_id, (x, y) in positions.items():
        node = mixed_by_id[node_id]
        rounded_node(
            ax,
            x,
            y,
            1.20 if node_id != "mixed_final" else 1.28,
            0.31,
            MIXED,
            str(node["label"]),
            failures(node),
            int(node["strict"]),
            final=node_id == "mixed_final",
        )
    for source, target, delta, component in MIXED_EDGES:
        sx, sy = positions[source]
        tx, ty = positions[target]
        start = (sx + (0.62 if source != "priority_raw" else 0.62), sy)
        end = (tx - 0.68, ty)
        arrow = FancyArrowPatch(
            start,
            end,
            connectionstyle="arc3,rad=0.02",
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=3,
            color=MIXED,
            zorder=2,
        )
        ax.add_patch(arrow)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.07,
            delta,
            fontsize=22,
            fontweight="bold",
            color=MIXED,
            ha="center",
        )
        ax.text(mx, my - 0.035, component, fontsize=18, color=MUTED, ha="center")
    ax.text(
        0.35,
        0.02,
        "Each arrow is a separate paired five-seed comparison. The three controls are not a sequential chain.",
        fontsize=18,
        color=MUTED,
        ha="left",
    )

    ax = axes[1]
    ax.set_xlim(-0.35, 4.25)
    ax.set_ylim(-0.05, 1.0)
    ax.axis("off")
    ax.text(
        -0.30,
        0.95,
        "Pure RL: deployment components remove 770 of 1,267 actor failures",
        fontsize=30,
        fontweight="bold",
        color=PURE,
        va="top",
    )
    positions_pure = {
        "pure_raw": (0.35, 0.45),
        "pure_reflect": (2.0, 0.45),
        "pure_final": (3.65, 0.45),
    }
    pure_by_id = node_map(PURE_NODES)
    for node_id, (x, y) in positions_pure.items():
        node = pure_by_id[node_id]
        rounded_node(
            ax,
            x,
            y,
            1.23,
            0.38,
            PURE,
            str(node["label"]),
            failures(node),
            int(node["strict"]),
            final=node_id == "pure_final",
        )
    for source, target, delta, component in PURE_EDGES:
        sx, sy = positions_pure[source]
        tx, ty = positions_pure[target]
        start, end = (sx + 0.64, sy), (tx - 0.64, ty)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=22,
                linewidth=3,
                color=PURE,
                zorder=2,
            )
        )
        mx = (start[0] + end[0]) / 2
        ax.text(
            mx,
            sy + 0.13,
            delta,
            fontsize=22,
            fontweight="bold",
            color=PURE,
            ha="center",
        )
        ax.text(mx, sy - 0.13, component, fontsize=18, color=MUTED, ha="center")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.985, bottom=0.02, hspace=0.02)
    save(fig, "59_two_track_recipe_graph.png")


def build_failure_waterfall() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.8), facecolor=WHITE)
    panels = [
        (
            axes[0],
            "RL + supervised matched controls",
            MIXED,
            [
                ("No refit + Q", 38, 9, "learner-state refit", 29),
                ("Uniform + Q", 19, 9, "priority sampling", 10),
                ("Priority refit", 35, 9, "local Q-search", 26),
            ],
        ),
        (
            axes[1],
            "Pure RL sequential deployment",
            PURE,
            [
                ("Raw actor", 1267, 955, "reflection", 312),
                ("Reflection", 955, 497, "Q-search", 458),
            ],
        ),
    ]
    for ax, title, color, rows in panels:
        ax.set_title(title, fontsize=29, fontweight="bold", color=color, loc="left", pad=18)
        y = np.arange(len(rows))[::-1]
        before = np.array([row[1] for row in rows])
        after = np.array([row[2] for row in rows])
        ax.barh(y, before, height=0.52, color="#dfe7ea", edgecolor="none")
        ax.barh(y, after, height=0.52, color=color, edgecolor="none")
        for yi, row in zip(y, rows):
            label, before_n, after_n, component, fixed = row
            ax.text(0, yi + 0.31, label, fontsize=23, fontweight="bold", color=INK)
            ax.text(
                before_n,
                yi,
                f" {before_n:,}",
                fontsize=24,
                fontweight="bold",
                color=GRAY,
                va="center",
            )
            ax.text(
                after_n,
                yi,
                f" {after_n:,}",
                fontsize=24,
                fontweight="bold",
                color=color,
                va="center",
            )
            ax.text(
                max(before_n, after_n) * 0.53,
                yi - 0.30,
                f"{component}: −{fixed} failures",
                fontsize=20,
                color=MUTED,
                ha="center",
            )
        ax.set_xlabel("Near-reference failures remaining, log scale", fontsize=22, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(6, 2000)
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, linewidth=1.3)
        ax.tick_params(axis="x", labelsize=19, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.5,
        0.015,
        "All comparisons use five trained actors × 2,501 starts. Mixed rows are three separate paired controls; pure rows form one chain.",
        ha="center",
        fontsize=19,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.87, bottom=0.14, wspace=0.16)
    save(fig, "60_failure_waterfall_controls.png")


def load_experiment_map() -> pd.DataFrame:
    data = pd.read_csv(LEDGER)
    steps = data["selected_checkpoint_learning_transition_steps"].fillna(
        data["executed_learning_transition_steps"]
    )
    keep = (
        data["standard_grid_protocol"].eq(True)
        & data["cheating_status"].eq("clean")
        & data["seed_count"].ge(5)
        & steps.le(100_000)
    )
    data = data.loc[
        keep,
        [
            "method",
            "category",
            "near_rate",
            "strict_rate",
            "task_rate",
            "seed_count",
        ],
    ].copy()
    data["source"] = "forensic ledger"
    overlays = []
    for node in MIXED_NODES + PURE_NODES:
        overlays.append(
            {
                "method": node["label"].replace("\n", " "),
                "category": "RL + supervised"
                if str(node["id"]).startswith(("no_", "uniform", "priority", "mixed"))
                else "pure RL",
                "near_rate": int(node["near"]) / N,
                "strict_rate": int(node["strict"]) / N,
                "task_rate": np.nan,
                "seed_count": 5,
                "source": "matched final-lineage ablation",
            }
        )
    overlays.extend(
        [
            {
                "method": "Pure supervised",
                "category": "supervised only",
                "near_rate": 0.9972,
                "strict_rate": 0.1450,
                "task_rate": 0.9360,
                "seed_count": 1,
                "source": "selected poster comparator",
            },
            {
                "method": "Uniform mixed control",
                "category": "RL + supervised",
                "near_rate": 0.9985,
                "strict_rate": 0.1167,
                "task_rate": 0.9384,
                "seed_count": 5,
                "source": "selected poster comparator",
            },
        ]
    )
    return pd.concat([data, pd.DataFrame(overlays)], ignore_index=True)


def build_experiment_map() -> None:
    data = load_experiment_map()
    data.to_csv(FIG / "61_experiment_map_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(15.8, 7.2), facecolor=WHITE)
    colors = {
        "pure RL": PURE,
        "RL + supervised": MIXED,
        "supervised only": ORANGE,
    }
    for category, group in data.groupby("category"):
        ax.scatter(
            100 * group["near_rate"],
            100 * group["strict_rate"],
            s=np.where(group["source"].eq("forensic ledger"), 85, 210),
            c=colors.get(category, GRAY),
            alpha=np.where(group["source"].eq("forensic ledger"), 0.25, 0.95),
            edgecolors=np.where(group["source"].eq("forensic ledger"), "none", WHITE),
            linewidths=2,
            label=category,
            zorder=3,
        )
    mixed = node_map(MIXED_NODES)
    pure = node_map(PURE_NODES)
    for source, target, *_ in MIXED_EDGES:
        s, t = mixed[source], mixed[target]
        ax.annotate(
            "",
            xy=(rate(int(t["near"])), rate(int(t["strict"]))),
            xytext=(rate(int(s["near"])), rate(int(s["strict"]))),
            arrowprops=dict(arrowstyle="-|>", color=MIXED, lw=2.4, alpha=0.9),
            zorder=2,
        )
    for source, target, *_ in PURE_EDGES:
        s, t = pure[source], pure[target]
        ax.annotate(
            "",
            xy=(rate(int(t["near"])), rate(int(t["strict"]))),
            xytext=(rate(int(s["near"])), rate(int(s["strict"]))),
            arrowprops=dict(arrowstyle="-|>", color=PURE, lw=2.6, alpha=0.9),
            zorder=2,
        )
    labels = [
        ("mixed_final", mixed["mixed_final"], (-120, 24), MIXED),
        ("priority_raw", mixed["priority_raw"], (-205, 24), MIXED),
        ("pure_raw", pure["pure_raw"], (-55, 28), PURE),
        ("pure_reflect", pure["pure_reflect"], (-35, 26), PURE),
        ("pure_final", pure["pure_final"], (-30, 28), PURE),
    ]
    for _, node, offset, color in labels:
        ax.annotate(
            str(node["label"]).replace("\n", " "),
            (rate(int(node["near"])), rate(int(node["strict"]))),
            xytext=offset,
            textcoords="offset points",
            fontsize=20,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
        )
    ax.axvline(99, color=GRID, linewidth=1.5, linestyle="--")
    ax.text(99.05, 2.0, "99% reliability", fontsize=18, color=MUTED, rotation=90, va="bottom")
    ax.set_xlim(67, 100.25)
    ax.set_ylim(0, 25.5)
    ax.set_xlabel("Near-reference success (%)", fontsize=25, color=INK)
    ax.set_ylabel("Strictly beats reference (%)", fontsize=25, color=INK)
    ax.set_title(
        "The search space: reliability and return dominance are different objectives",
        fontsize=30,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=16,
    )
    ax.text(
        67.1,
        24.2,
        "Faint points: all clean five-seed ≤100k standardized ledger results. Arrows: exact matched final-lineage ablations.",
        fontsize=19,
        color=MUTED,
    )
    ax.grid(color=GRID, linewidth=1.0, alpha=0.8)
    ax.tick_params(labelsize=20, colors=MUTED)
    ax.legend(frameon=False, fontsize=19, loc="upper left", ncol=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.15)
    save(fig, "61_experiment_landscape.png")


def build_experiment_map_poster() -> None:
    data = load_experiment_map()
    fig, ax = plt.subplots(figsize=(11.8, 6.4), facecolor=WHITE)
    colors = {"pure RL": PURE, "RL + supervised": MIXED, "supervised only": ORANGE}
    ledger = data[data["source"].eq("forensic ledger")]
    for category, group in ledger.groupby("category"):
        ax.scatter(
            100 * group["near_rate"],
            100 * group["strict_rate"],
            s=80,
            c=colors.get(category, GRAY),
            alpha=0.18,
            edgecolors="none",
            zorder=1,
        )
    mixed = node_map(MIXED_NODES)
    pure = node_map(PURE_NODES)
    for source, target, *_ in MIXED_EDGES:
        s, t = mixed[source], mixed[target]
        ax.annotate(
            "",
            xy=(rate(int(t["near"])), rate(int(t["strict"]))),
            xytext=(rate(int(s["near"])), rate(int(s["strict"]))),
            arrowprops=dict(arrowstyle="-|>", color=MIXED, lw=4),
            zorder=2,
        )
    for source, target, *_ in PURE_EDGES:
        s, t = pure[source], pure[target]
        ax.annotate(
            "",
            xy=(rate(int(t["near"])), rate(int(t["strict"]))),
            xytext=(rate(int(s["near"])), rate(int(s["strict"]))),
            arrowprops=dict(arrowstyle="-|>", color=PURE, lw=4),
            zorder=2,
        )
    selected = [
        (pure["pure_raw"], "FastSACN8 actor", (-10, -52), PURE),
        (pure["pure_reflect"], "+ reflection", (-65, 25), PURE),
        (pure["pure_final"], "+ Q-search", (-65, 25), PURE),
        (mixed["priority_raw"], "mixed actor", (-175, 35), MIXED),
        (mixed["mixed_final"], "final mixed", (-165, -55), MIXED),
    ]
    for node, label, offset, color in selected:
        x, y = rate(int(node["near"])), rate(int(node["strict"]))
        ax.scatter([x], [y], s=320, c=color, edgecolors=WHITE, linewidths=3, zorder=4)
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=32,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=2),
            zorder=5,
        )
    ax.text(68.0, 24.2, "faint: every clean five-seed ≤100k result", fontsize=30, color=MUTED)
    ax.text(68.0, 22.6, "blue: pure RL   green: RL + supervised", fontsize=30, color=INK)
    ax.set_xlim(67, 100.6)
    ax.set_ylim(0, 25.5)
    ax.set_xlabel("Near-reference success (%)", fontsize=34, color=INK)
    ax.set_ylabel("Strict wins (%)", fontsize=34, color=INK)
    ax.tick_params(labelsize=30, colors=MUTED)
    ax.grid(color=GRID, linewidth=1.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.18)
    save(fig, "63_experiment_landscape_poster.png")


def build_experiment_ribbon() -> None:
    """Wide experiment map sized for a full-width A0 band."""
    data = load_experiment_map()
    fig, ax = plt.subplots(figsize=(22.0, 5.5), facecolor=WHITE)
    colors = {"pure RL": PURE, "RL + supervised": MIXED, "supervised only": ORANGE}
    ledger = data[data["source"].eq("forensic ledger")]
    for category, group in ledger.groupby("category"):
        ax.scatter(
            100 * group["near_rate"],
            100 * group["strict_rate"],
            s=150,
            c=colors.get(category, GRAY),
            alpha=0.16,
            edgecolors="none",
            zorder=1,
        )

    mixed = node_map(MIXED_NODES)
    pure = node_map(PURE_NODES)
    for source, target, *_ in MIXED_EDGES:
        s, t = mixed[source], mixed[target]
        ax.annotate(
            "",
            xy=(rate(int(t["near"])), rate(int(t["strict"]))),
            xytext=(rate(int(s["near"])), rate(int(s["strict"]))),
            arrowprops=dict(arrowstyle="-|>", color=MIXED, lw=5),
            zorder=2,
        )
    for source, target, *_ in PURE_EDGES:
        s, t = pure[source], pure[target]
        ax.annotate(
            "",
            xy=(rate(int(t["near"])), rate(int(t["strict"]))),
            xytext=(rate(int(s["near"])), rate(int(s["strict"]))),
            arrowprops=dict(arrowstyle="-|>", color=PURE, lw=5),
            zorder=2,
        )

    for node in list(mixed.values()) + list(pure.values()):
        color = MIXED if node in mixed.values() else PURE
        ax.scatter(
            [rate(int(node["near"]))],
            [rate(int(node["strict"]))],
            s=430,
            c=color,
            edgecolors=WHITE,
            linewidths=4,
            zorder=4,
        )

    callouts = [
        (pure["pure_raw"], "FastSACN8 actor\n1,267 failures", (82.0, 13.0), PURE),
        (pure["pure_reflect"], "+ reflection\n955 failures", (87.2, 24.5), PURE),
        (pure["pure_final"], "+ Q-search\n497 failures", (93.0, 25.8), PURE),
        (mixed["priority_raw"], "priority refit\nwithout Q-search\n35 failures", (88.3, 7.8), MIXED),
        (mixed["mixed_final"], "priority refit\n+ Q-search\n9 failures", (95.7, 3.8), MIXED),
    ]
    for node, label, text_xy, color in callouts:
        x, y = rate(int(node["near"])), rate(int(node["strict"]))
        ax.annotate(
            label,
            (x, y),
            xytext=text_xy,
            textcoords="data",
            fontsize=16,
            linespacing=0.92,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=3),
            bbox=dict(boxstyle="round,pad=0.18", facecolor=WHITE, edgecolor="none", alpha=0.92),
            zorder=5,
        )

    ax.text(
        67.4,
        27.2,
        "faint points: all standardized five-seed results at 100k steps or less",
        fontsize=16,
        color=MUTED,
        va="top",
    )
    ax.text(
        67.4,
        23.7,
        "blue path: pure-RL additions     green arrows: separate matched mixed controls",
        fontsize=16,
        color=INK,
        va="top",
    )
    ax.set_xlim(67, 101.0)
    ax.set_ylim(0, 28)
    ax.set_xlabel("Near-reference success (%)", fontsize=18, color=INK, labelpad=8)
    ax.set_ylabel("Strict wins (%)", fontsize=18, color=INK, labelpad=8)
    ax.tick_params(labelsize=16, colors=MUTED)
    ax.grid(color=GRID, linewidth=1.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.98, bottom=0.24)
    save(fig, "65_experiment_ribbon_final.png")


def build_compact_hybrid() -> None:
    fig = plt.figure(figsize=(15.8, 8.2), facecolor=WHITE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1, 1], wspace=0.08, hspace=0.10)
    ax_m = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[:, 1])

    for ax in (ax_m, ax_p):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax_m.text(0, 0.95, "RL + supervised", fontsize=29, fontweight="bold", color=MIXED, va="top")
    mixed_rows = [
        ("No learner-state refit + Q", 38, "+29"),
        ("Uniform starts + Q", 19, "+10"),
        ("Priority refit without Q", 35, "+26"),
    ]
    yvals = [0.72, 0.47, 0.22]
    for (label, fail, delta), y in zip(mixed_rows, yvals):
        ax_m.text(0.01, y, label, fontsize=20, color=INK, va="center")
        ax_m.plot([0.47, 0.69], [y, y], color=MIXED, linewidth=4)
        ax_m.text(0.58, y + 0.045, delta, fontsize=19, fontweight="bold", color=MIXED, ha="center")
        ax_m.text(0.45, y, f"{fail}", fontsize=24, fontweight="bold", color=ORANGE, ha="right", va="center")
    rounded_node(ax_m, 0.86, 0.47, 0.25, 0.56, MIXED, "Final mixed", 9, 1570, final=True)
    ax_m.text(0.01, 0.03, "Separate paired controls; no false sequential chain.", fontsize=17, color=MUTED)

    ax_p.text(0, 0.95, "Pure RL", fontsize=29, fontweight="bold", color=PURE, va="top")
    xvals = [0.14, 0.50, 0.86]
    for node, x in zip(PURE_NODES, xvals):
        rounded_node(
            ax_p,
            x,
            0.50,
            0.24,
            0.52,
            PURE,
            str(node["label"]),
            failures(node),
            int(node["strict"]),
            final=node["id"] == "pure_final",
        )
    for (source, target, delta, _), x0, x1 in zip(PURE_EDGES, xvals[:-1], xvals[1:]):
        ax_p.add_patch(
            FancyArrowPatch(
                (x0 + 0.13, 0.50),
                (x1 - 0.13, 0.50),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=3,
                color=PURE,
            )
        )
        ax_p.text((x0 + x1) / 2, 0.68, delta, fontsize=18, fontweight="bold", color=PURE, ha="center")

    data = load_experiment_map()
    colors = {"pure RL": PURE, "RL + supervised": MIXED, "supervised only": ORANGE}
    for category, group in data.groupby("category"):
        ax_s.scatter(
            100 * group["near_rate"],
            100 * group["strict_rate"],
            s=np.where(group["source"].eq("forensic ledger"), 35, 100),
            c=colors.get(category, GRAY),
            alpha=np.where(group["source"].eq("forensic ledger"), 0.20, 0.9),
            edgecolors="none",
        )
    ax_s.set_xlim(67, 100.3)
    ax_s.set_ylim(0, 25.5)
    ax_s.set_xlabel("Near-reference success (%)", fontsize=19)
    ax_s.set_ylabel("Strict wins (%)", fontsize=19)
    ax_s.set_title("All clean five-seed ≤100k results", fontsize=25, fontweight="bold", loc="left")
    ax_s.tick_params(labelsize=17, colors=MUTED)
    ax_s.grid(color=GRID)
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Two improvement paths inside the broader experiment search",
        x=0.03,
        ha="left",
        fontsize=32,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.08)
    save(fig, "62_compact_recipe_and_search_map.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    build_recipe_graph()
    build_failure_waterfall()
    build_experiment_map()
    build_experiment_map_poster()
    build_experiment_ribbon()
    build_compact_hybrid()


if __name__ == "__main__":
    main()
