from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "deliverables" / "chasing_nines_20260723" / "figures"
MECH = ROOT / "reports" / "plan2307_mechanistic_20260723"
TARGET = (
    ROOT
    / "reports"
    / "plan2307_pure_target_architecture_20260723"
    / "p0_p1_p2_action_gradient_alignment"
    / "summary.json"
)
GEOMETRY = MECH / "completed_target_actor_geometry" / "summary.json"
PREFIX = (
    ROOT
    / "reports"
    / "plan2507_p7_reference_prefix_diagnostic_20260725"
)
PURE = (
    ROOT
    / "reports"
    / "plan2507_p7_authority_20260725"
    / "relative"
    / "relative_rollouts.csv"
)
MIXED = (
    ROOT
    / "reports"
    / "systematic_100k_budget_best_20260722"
    / "ablation_no_rl_shift_qsearch"
    / "relative"
    / "relative_rollouts.csv"
)

INK = "#102a35"
MUTED = "#526b76"
PURE_BLUE = "#3478ef"
MIXED_TEAL = "#0f9f91"
ORANGE = "#e85c38"
GOLD = "#f2a43a"
GRID = "#dce5e8"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.spines[["left", "bottom"]].set_linewidth(1.8)
    ax.tick_params(labelsize=14, colors=INK, width=1.5, length=5)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)


def build_actor_critic_transfer() -> None:
    target = load_json(TARGET)["conditions"]
    geometry = load_json(GEOMETRY)["arms"]

    labels = ["One-step\nSAC", "FastSACN8\nλ = 1", "SACN8\nλ = 1"]
    keys = ["P0 one-step", "P1 FastSACN8", "P2 SACn8"]
    helpful = [
        100 * target[key]["pooled"]["failure_critic_step_beneficial_rate"]
        for key in keys
    ]
    diagnostic_near = [
        100 * target[key]["pooled"]["near_reference_rate"] for key in keys
    ]
    saturation = [
        100
        * target[key]["pooled"]["actor_saturation_rate"]
        for key in keys
    ]

    p0_sat = (
        100
        * geometry["p0_simba_onestep_utd1_100k"]["metrics"][
            "deterministic_action_saturation_fraction_abs_ge_0p995"
        ]["median"]
    )
    p7_sat = (
        100
        * geometry[
            "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k"
        ]["metrics"]["deterministic_action_saturation_fraction_abs_ge_0p995"][
            "median"
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1), facecolor="white")
    x = np.arange(3)
    colors = ["#738596", PURE_BLUE, "#8a5be8"]

    ax = axes[0]
    bars = ax.bar(x, helpful, color=colors, width=0.66)
    ax.set_title("A critic-guided change helps more often", fontsize=16, fontweight="bold", color=INK)
    ax.set_ylabel("Changed action improves return (%)", fontsize=14, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    for bar, value in zip(bars, helpful):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
    style_axis(ax)

    ax = axes[1]
    bars = ax.bar(x, saturation, color=colors, width=0.66)
    ax.set_title("Actors saturate more", fontsize=17, fontweight="bold", color=INK)
    ax.set_ylabel("Actions at torque bound (%)", fontsize=14, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    for bar, sat, near in zip(bars, saturation, diagnostic_near):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            sat + 2,
            f"{sat:.1f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            4,
            f"near ref.\n{near:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11.5,
            color="white",
            fontweight="bold",
        )
    style_axis(ax)

    ax = axes[2]
    stages = ["Raw actor", "Reflection", "Twin-Q search"]
    authority = [89.868, 92.363, 96.026]
    ax.plot(
        np.arange(3),
        authority,
        "-o",
        color=PURE_BLUE,
        linewidth=5,
        markersize=13,
    )
    ax.fill_between(np.arange(3), authority, 88, color=PURE_BLUE, alpha=0.10)
    ax.set_title("Search repairs actor failures", fontsize=16, fontweight="bold", color=INK)
    ax.set_ylabel("Near-reference success (%)", fontsize=14, color=INK)
    ax.set_xticks(np.arange(3), stages)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(88, 97)
    ax.set_yticks([88, 90, 92, 94, 96])
    for i, value in enumerate(authority):
        ax.text(
            i,
            value + 0.35,
            f"{value:.3f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=PURE_BLUE,
        )
    style_axis(ax)

    fig.suptitle(
        "Multi-step losses help critic-guided changes more than raw actors",
        fontsize=24,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Panels A–B: 512 locked off-grid states per seed, five seeds. "
            "Panel C: 12,505 authority-grid rollouts from the selected five-seed lineage. "
            f"Matched median saturation rises from {p0_sat:.1f}% to {p7_sat:.1f}%."
        ),
        ha="center",
        fontsize=11.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.21, wspace=0.28)
    fig.savefig(FIG / "47_actor_critic_transfer.png", dpi=220, facecolor="white")
    plt.close(fig)


def build_sequence_intervention() -> None:
    summary = load_json(PREFIX / "summary.json")
    aggregate = pd.DataFrame(summary["aggregate"])
    controls = pd.DataFrame(summary["specificity_controls"])
    steps = [1, 8, 16, 32]
    matched = (
        aggregate[
            (aggregate["condition"] == "pure RL actor")
            & aggregate["prefix_steps"].isin(steps)
        ]
        .sort_values("prefix_steps")
        .set_index("prefix_steps")
    )
    shuffled = (
        controls[
            (controls["prefix_mode"] == "shuffled_reference")
            & controls["prefix_steps"].isin(steps)
        ]
        .sort_values("prefix_steps")
        .set_index("prefix_steps")
    )

    fig = plt.figure(figsize=(16.5, 5.0), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.55], wspace=0.20)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])

    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.text(
        0.0,
        0.94,
        "What the intervention asks",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    boxes = [
        (0.02, 0.52, 0.25, 0.22, "Pure-RL\nfailure", "#e6efff", PURE_BLUE),
        (0.37, 0.52, 0.25, 0.22, "Correct for\nK steps", "#fff0e5", ORANGE),
        (0.72, 0.52, 0.25, 0.22, "Return control\nto same actor", "#edf7f5", MIXED_TEAL),
    ]
    for x, y, w, h, text, face, edge in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=3)
        ax0.add_patch(rect)
        ax0.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=INK,
        )
    for start in (0.28, 0.63):
        ax0.annotate(
            "",
            xy=(start + 0.075, 0.63),
            xytext=(start, 0.63),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": MUTED},
        )
    ax0.text(
        0.02,
        0.34,
        "If only the first move is wrong,\none correction should rescue most failures.",
        fontsize=16,
        color=INK,
        linespacing=1.25,
    )
    ax0.text(
        0.02,
        0.14,
        "Observed: 1 step rescues 19.1%.\nA state-matched 16-step segment rescues 91.3%.",
        fontsize=16,
        color=PURE_BLUE,
        fontweight="bold",
        linespacing=1.25,
    )

    x = np.arange(len(steps))
    y_matched = 100 * matched.loc[steps, "repair_rate"].to_numpy()
    y_shuffled = 100 * shuffled.loc[steps, "repair_rate"].to_numpy()
    ax1.plot(
        x,
        y_matched,
        "-o",
        color=PURE_BLUE,
        linewidth=5,
        markersize=12,
        label="State-matched actions",
    )
    ax1.plot(
        x,
        y_shuffled,
        "-o",
        color=GOLD,
        linewidth=4,
        markersize=11,
        label="Same actions, wrong order/state",
    )
    for xi, yi in zip(x, y_matched):
        ax1.text(
            xi,
            yi + 3.0,
            f"{yi:.1f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=PURE_BLUE,
        )
    ax1.set_title("Recovery needs a state-matched control segment", fontsize=20, fontweight="bold", color=INK)
    ax1.set_ylabel("Original failures repaired (%)", fontsize=15, color=INK)
    ax1.set_xlabel("Corrective steps before returning control", fontsize=15, color=INK)
    ax1.set_xticks(x, steps)
    ax1.set_ylim(0, 106)
    ax1.legend(frameon=False, fontsize=13, loc="upper left")
    style_axis(ax1)

    fig.suptitle(
        "Pure-RL failures persist across a short critical segment",
        fontsize=25,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.83, bottom=0.13)
    fig.savefig(FIG / "48_sequence_intervention_explained.png", dpi=220, facecolor="white")
    plt.close(fig)


def build_sequence_chart_compact() -> None:
    summary = load_json(PREFIX / "summary.json")
    aggregate = pd.DataFrame(summary["aggregate"])
    controls = pd.DataFrame(summary["specificity_controls"])
    steps = [1, 8, 16, 32]
    matched = (
        aggregate[
            (aggregate["condition"] == "pure RL actor")
            & aggregate["prefix_steps"].isin(steps)
        ]
        .sort_values("prefix_steps")
        .set_index("prefix_steps")
    )
    shuffled = (
        controls[
            (controls["prefix_mode"] == "shuffled_reference")
            & controls["prefix_steps"].isin(steps)
        ]
        .sort_values("prefix_steps")
        .set_index("prefix_steps")
    )
    x = np.arange(len(steps))
    y_matched = 100 * matched.loc[steps, "repair_rate"].to_numpy()
    y_shuffled = 100 * shuffled.loc[steps, "repair_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 3.2), facecolor="white")
    ax.plot(x, y_matched, "-o", color=PURE_BLUE, linewidth=5, markersize=12, label="State-matched")
    ax.plot(x, y_shuffled, "-o", color=GOLD, linewidth=4, markersize=11, label="Wrong order or state")
    for xi, yi in zip(x, y_matched):
        ax.text(xi, yi + 3.2, f"{yi:.1f}%", ha="center", fontsize=14, fontweight="bold", color=PURE_BLUE)
    ax.set_ylabel("Failures repaired (%)", fontsize=15, color=INK)
    ax.set_xlabel("Corrective steps before returning control", fontsize=15, color=INK)
    ax.set_xticks(x, steps)
    ax.set_ylim(0, 106)
    ax.legend(frameon=False, fontsize=13, loc="upper left", ncol=2)
    style_axis(ax)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.23)
    fig.savefig(FIG / "50_sequence_repair_compact.png", dpi=220, facecolor="white")
    plt.close(fig)


def failure_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    frame = pd.read_csv(path)
    frame["failure"] = 1 - frame["near_best_known_return_eps"].astype(int)
    grouped = frame.groupby(["theta_dot", "theta_degrees"], as_index=False)["failure"].sum()
    angles = np.sort(grouped["theta_degrees"].unique())
    velocities = np.sort(grouped["theta_dot"].unique())
    pivot = grouped.pivot(index="theta_dot", columns="theta_degrees", values="failure")
    grid = pivot.loc[velocities, angles].to_numpy()
    return angles, velocities, grid, int(frame["failure"].sum()), int((grid > 0).sum())


def build_failure_footprints_clean() -> None:
    mixed = failure_grid(MIXED)
    pure = failure_grid(PURE)
    colors = ["#fffdf8", "#9de4dc", "#f5df83", "#f3a43b", "#e85c38", "#8f1d24"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.4), facecolor="white")
    for ax, data, name, accent in (
        (axes[0], mixed, "RL + supervision", MIXED_TEAL),
        (axes[1], pure, "Pure RL", PURE_BLUE),
    ):
        angles, velocities, grid, failures, starts = data
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[angles.min(), angles.max(), velocities.min(), velocities.max()],
        )
        ax.set_title(
            f"{name}\n{failures} failures across {starts} starts",
            fontsize=21,
            fontweight="bold",
            color=INK,
            pad=9,
        )
        ax.set_xlabel("Initial angle (degrees)", fontsize=15, fontweight="bold", color=INK)
        ax.tick_params(labelsize=13, width=1.5, length=5, colors=INK)
        for spine in ax.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(3)
        ax.text(
            0.50,
            0.47,
            f"{2501 - starts:,} / 2,501 starts\nall five actors succeed",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=accent,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": accent,
                "linewidth": 2,
                "alpha": 0.94,
            },
        )

    axes[0].set_ylabel("Initial angular velocity", fontsize=15, fontweight="bold", color=INK, labelpad=10)
    axes[1].set_yticklabels([])
    colorbar = fig.colorbar(
        image,
        ax=axes,
        ticks=np.arange(1, 6),
        orientation="horizontal",
        fraction=0.075,
        pad=0.24,
        aspect=60,
    )
    colorbar.set_label("Number of the five actors that fail", fontsize=14, fontweight="bold", color=INK)
    colorbar.ax.tick_params(labelsize=13, colors=INK)
    fig.suptitle(
        "Supervision compresses a seed-dependent failure tail",
        fontsize=26,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.75, bottom=0.34, wspace=0.10)
    fig.savefig(FIG / "49_failure_footprints_clean.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    FIG.mkdir(parents=True, exist_ok=True)
    build_actor_critic_transfer()
    build_sequence_intervention()
    build_sequence_chart_compact()
    build_failure_footprints_clean()
    for name in (
        "47_actor_critic_transfer.png",
        "48_sequence_intervention_explained.png",
        "49_failure_footprints_clean.png",
        "50_sequence_repair_compact.png",
    ):
        print(FIG / name)


if __name__ == "__main__":
    main()
