from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.ticker import PercentFormatter

from .manifest import load_json
from .metrics import load_claims, summarize_method, tolerance_curve


INK = "#102a35"
MUTED = "#526b76"
MIXED = "#0d9f90"
PURE = "#3478ef"
ORANGE = "#ef6a3a"
GOLD = "#f2a43a"
GRAY = "#8193a1"
GRID = "#dce5e8"

FIGURE_NAMES = (
    "37_core_scorecard_large.png",
    "77_failure_footprints_stacked_axes.png",
    "58_poster_target_family_internals_wide.png",
    "36_fastsacn_objective_share.png",
    "111_tolerance_curve_wide.png",
    "79_worst_seed_return_margin.png",
    "16_completed_target_actor_geometry.png",
    "40_p7_reference_prefix_intervention.png",
    "41_p7_reference_prefix_specificity.png",
)


def _save(fig: plt.Figure, path: Path, *, dpi: int = 220) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)


def render_scorecard(frames: Mapping[str, pd.DataFrame], output: Path) -> Path:
    order = [
        "mixed_selected",
        "mixed_uniform",
        "mixed_actor_only",
        "pure_selected_q41",
        "simba_onestep",
        "canonical_dagger",
    ]
    labels = [
        "Mixed, selected",
        "Mixed, uniform starts",
        "Mixed actor only",
        "Pure RL, selected",
        "SimbaV2 one-step",
        "Canonical DAgger",
    ]
    summaries = [summarize_method(frames[key]) for key in order]
    series = [
        ("Near reference", MIXED, np.asarray([row.near for row in summaries]) / 12505),
        ("Task success", PURE, np.asarray([row.task for row in summaries]) / 12505),
        ("Strict win", ORANGE, np.asarray([row.strict for row in summaries]) / 12505),
    ]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor="white")
    for offset, (name, color, values) in zip((-0.22, 0.0, 0.22), series):
        bars = ax.barh(y + offset, 100 * values, height=0.19, color=color, label=name)
        for bar, value in zip(bars, 100 * values):
            ax.text(
                min(float(value) + 0.6, 101.2),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=10,
                color=INK,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel("Share of 12,505 seed-state trials")
    ax.set_title("Common-grid report scorecard", loc="left", fontweight="bold", color=INK)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    _style_axis(ax, "x")
    fig.tight_layout()
    return _save(fig, output / "37_core_scorecard_large.png")


def _failure_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame.assign(failure=1 - frame["near_best_known_return_eps"].astype(int))
    grouped = work.groupby(["theta_dot", "theta_degrees"], as_index=False)["failure"].sum()
    angles = np.sort(grouped["theta_degrees"].unique())
    velocities = np.sort(grouped["theta_dot"].unique())
    grid = grouped.pivot(index="theta_dot", columns="theta_degrees", values="failure")
    return angles, velocities, grid.loc[velocities, angles].to_numpy()


def render_failure_footprints(frames: Mapping[str, pd.DataFrame], output: Path) -> Path:
    colors = ["#fffdf8", "#9de4dc", "#f5df83", "#f3a43b", "#e85c38", "#8f1d24"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), facecolor="white")
    for ax, key, title, accent in (
        (axes[0], "mixed_selected", "RL + supervised", MIXED),
        (axes[1], "pure_selected_q41", "Pure RL", PURE),
    ):
        angles, velocities, grid = _failure_grid(frames[key])
        ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[angles.min(), angles.max(), velocities.min(), velocities.max()],
        )
        summary = summarize_method(frames[key])
        ax.set_title(
            f"{title}: {summary.failures} failures on {summary.failure_cells} starts",
            loc="left",
            color=accent,
            fontweight="bold",
        )
        ax.set_ylabel(r"$\dot{\theta}_0$ (rad/s)")
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([-180, -90, 0, 90, 174])
        for spine in ax.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(2.5)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel(r"Initial angle $\theta_0$ (degrees)")
    fig.tight_layout(h_pad=1.2)
    return _save(fig, output / "77_failure_footprints_stacked_axes.png", dpi=240)


def _condition_values(payload: dict, conditions: list[str], field: str) -> np.ndarray:
    return np.asarray([payload["conditions"][key]["pooled"][field] for key in conditions], dtype=float)


def render_target_family_internals(data_dir: Path, output: Path) -> Path:
    geometry = load_json(data_dir / "diagnostics" / "actor_geometry" / "summary.json")["arms"]
    c32 = load_json(data_dir / "diagnostics" / "critic_direction" / "summary.json")
    c33 = load_json(data_dir / "diagnostics" / "action_projection" / "summary.json")
    c34 = load_json(data_dir / "diagnostics" / "reference_recognition" / "summary.json")
    claims = load_claims(data_dir)
    semantic = claims["diagnostic_semantics"]["c32"]
    conditions = list(semantic["conditions"])
    arm_keys = [
        "p0_simba_onestep_utd1_100k",
        "p1_simba_fastsacn8_lambda1_utd1_100k",
        "p2_simba_sacn8_lambda1_utd1_100k",
    ]
    labels = ["SAC", "FastSACN8", "SACN8"]
    colors = [GRAY, PURE, MIXED]
    saturation = [
        100
        * np.asarray(
            geometry[key]["metrics"]["deterministic_action_saturation_fraction_abs_ge_0p995"][
                "seed_values"
            ],
            dtype=float,
        )
        for key in arm_keys
    ]
    slopes = [
        np.asarray(geometry[key]["metrics"]["mean_tanh_derivative"]["seed_values"], dtype=float)
        for key in arm_keys
    ]
    direction = 100 * _condition_values(c32, conditions, str(semantic["figure_field"]))
    harmful = 100 * _condition_values(c32, conditions, "critic_step_harmful_rate")
    helpful = 100 * np.asarray(
        [c34["conditions"][key]["pooled"]["failure"]["critic_prefers_helpful_reference_rate"] for key in conditions]
    )
    projection = {row["condition"]: row for row in c33["pooled"] if row["outcome"] == "failure"}
    boundary = 100 * np.asarray([projection[key]["boundary_rate"] for key in conditions])
    outward = 100 * np.asarray([projection[key]["outward_among_boundary_rate"] for key in conditions])
    effective = 100 * np.asarray([projection[key]["mean_effective_step_fraction"] for key in conditions])

    fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.2), facecolor="white")
    x = np.arange(3)
    for seed in range(5):
        axes[0].plot(x, [row[seed] for row in saturation], color="#b8c7ce", marker="o")
        axes[1].plot(x, [row[seed] for row in slopes], color="#b8c7ce", marker="o")
    axes[0].scatter(x, [np.median(row) for row in saturation], s=110, color=colors, zorder=3)
    axes[1].scatter(x, [np.median(row) for row in slopes], s=110, color=colors, zorder=3)
    width = 0.24
    for offset, values, color, label in (
        (-width, direction, PURE, "Gradient agrees"),
        (0.0, helpful, MIXED, "Helpful action ranked"),
        (width, harmful, ORANGE, "Move harmful"),
    ):
        axes[2].bar(x + offset, values, width, color=color, label=label)
    for offset, values, color, label in (
        (-width, boundary, GRAY, "At bound"),
        (0.0, outward, PURE, "Points outward"),
        (width, effective, ORANGE, "Step preserved"),
    ):
        axes[3].bar(x + offset, values, width, color=color, label=label)
    for ax, title in zip(
        axes,
        ("A  Actor bound", "B  Actor slope", "C  Critic test", "D  Clip effect"),
    ):
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left", fontweight="bold")
        _style_axis(ax)
    axes[0].set_ylim(40, 100)
    axes[1].set_ylim(0, 0.17)
    axes[2].set_ylim(0, 110)
    axes[3].set_ylim(0, 115)
    axes[2].legend(frameon=False, fontsize=8, loc="upper center")
    axes[3].legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout(w_pad=1.5)
    return _save(fig, output / "58_poster_target_family_internals_wide.png", dpi=240)


def render_objective_share(data_dir: Path, output: Path) -> Path:
    payload = load_json(data_dir / "diagnostics" / "objective_share" / "summary.json")
    values = payload["eight_step_objective_share_percent"]
    density = np.asarray(values["density_empirical_seed_values"], dtype=float)
    labels = ["Nominal", "No importance", "Density mean"]
    heights = [values["nominal"], values["no_importance_empirical"], values["density_empirical_mean"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.8), facecolor="white")
    x = np.arange(3)
    bars = ax.bar(x, heights, color=[GRAY, PURE, MIXED], width=0.62)
    ax.scatter(np.full(len(density), 2), density, color=INK, s=35, zorder=3)
    for bar, value in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}%", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Eight-step share of critic objective (%)")
    ax.set_title("FastSACN8 is dominated by its one-step endpoint", loc="left", fontweight="bold")
    ax.set_ylim(0, max(heights) * 1.25)
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, output / "36_fastsacn_objective_share.png")


def render_tolerance(frames: Mapping[str, pd.DataFrame], output: Path) -> Path:
    epsilons = np.linspace(0, 20, 81)
    fig, ax = plt.subplots(figsize=(11.2, 4.1), facecolor="white")
    for key, label, color in (
        ("mixed_selected", "RL + supervised", MIXED),
        ("pure_selected_q41", "Pure RL", PURE),
    ):
        seed_curves, mean_curve = tolerance_curve(frames[key], epsilons)
        for curve in seed_curves:
            ax.plot(epsilons, curve, color=color, alpha=0.16, linewidth=1.4)
        ax.plot(epsilons, mean_curve, color=color, linewidth=4, label=label)
        at_five = mean_curve[np.argmin(np.abs(epsilons - 5.0))]
        ax.scatter([5], [at_five], color=color, edgecolor="white", s=85, zorder=5)
        ax.annotate(f"{at_five:.3%}", (5, at_five), xytext=(12, -20), textcoords="offset points", color=color)
    ax.axvline(5, color="#64748b", linestyle="--")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.1, 1.01)
    ax.set_xlabel("Return tolerance")
    ax.set_ylabel("Fraction near reference")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, output / "111_tolerance_curve_wide.png")


def _worst_seed_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame.assign(return_margin=frame["return"] - frame["best_known_return"])
    grouped = work.groupby(["theta_dot", "theta_degrees"], as_index=False)["return_margin"].min()
    angles = np.sort(grouped["theta_degrees"].unique())
    velocities = np.sort(grouped["theta_dot"].unique())
    grid = grouped.pivot(index="theta_dot", columns="theta_degrees", values="return_margin")
    return angles, velocities, grid.loc[velocities, angles].to_numpy()


def render_worst_seed_margin(frames: Mapping[str, pd.DataFrame], output: Path) -> Path:
    cmap = LinearSegmentedColormap.from_list(
        "return_margin", ["#8f1d24", "#ef6a3a", "#fffdf8", "#8ddbd0", "#2367d8"]
    )
    norm = Normalize(vmin=-20, vmax=20, clip=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), facecolor="white")
    image = None
    for ax, key, title, accent in (
        (axes[0], "mixed_selected", "RL + supervised", MIXED),
        (axes[1], "pure_selected_q41", "Pure RL", PURE),
    ):
        angles, velocities, grid = _worst_seed_grid(frames[key])
        image = ax.imshow(
            np.clip(grid, -20, 20),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[angles.min(), angles.max(), velocities.min(), velocities.max()],
        )
        ax.contour(angles, velocities, grid, levels=[-5], colors=["#111827"], linewidths=1.3)
        ax.set_title(title, loc="left", color=accent, fontweight="bold")
        ax.set_xlabel("Initial angle (degrees)")
        ax.set_yticks([-1, 0, 1])
    axes[0].set_ylabel("Initial angular velocity")
    assert image is not None
    fig.colorbar(image, ax=axes, label="Worst-seed return margin", fraction=0.035, pad=0.03)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.16, wspace=0.12)
    return _save(fig, output / "79_worst_seed_return_margin.png", dpi=230)


def render_actor_geometry(data_dir: Path, output: Path) -> Path:
    arms = load_json(data_dir / "diagnostics" / "actor_geometry" / "summary.json")["arms"]
    keys = list(arms)
    labels = ["SAC", "Fast N8", "SACN8", "Fast N8\nselected"]
    metrics = (
        ("deterministic_action_saturation_fraction_abs_ge_0p995", "Actions at bound", 100.0),
        ("mean_tanh_derivative", "Remaining tanh slope", 1.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    x = np.arange(len(keys))
    for ax, (metric, title, scale) in zip(axes, metrics):
        values = [np.asarray(arms[key]["metrics"][metric]["seed_values"]) * scale for key in keys]
        for seed in range(5):
            ax.plot(x, [row[seed] for row in values], color="#b8c7ce", marker="o")
        ax.scatter(x, [np.median(row) for row in values], color=[GRAY, PURE, MIXED, GOLD], s=100, zorder=3)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left", fontweight="bold")
        _style_axis(ax)
    axes[0].set_ylabel("Percent")
    fig.suptitle("Completed target-family actor geometry", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, output / "16_completed_target_actor_geometry.png", dpi=240)


def render_prefix_intervention(data_dir: Path, output: Path) -> Path:
    aggregate = pd.read_csv(data_dir / "diagnostics" / "prefix_intervention" / "aggregate.csv")
    selected = aggregate[
        (aggregate["condition"] == "pure RL actor")
        & aggregate["prefix_steps"].isin([1, 8, 16, 32])
    ].sort_values("prefix_steps")
    fig, ax = plt.subplots(figsize=(9.8, 4.8), facecolor="white")
    ax.plot(selected["prefix_steps"], 100 * selected["repair_rate"], "-o", color=PURE, linewidth=4)
    ax.fill_between(
        selected["prefix_steps"],
        100 * selected["seed_repair_rate_min"],
        100 * selected["seed_repair_rate_max"],
        color=PURE,
        alpha=0.14,
    )
    ax.set_xticks([1, 8, 16, 32])
    ax.set_ylim(0, 105)
    ax.set_xlabel("State-matched corrective prefix length")
    ax.set_ylabel("Raw-actor failures repaired (%)")
    ax.set_title("Recovery often needs a coordinated sequence", loc="left", fontweight="bold")
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, output / "40_p7_reference_prefix_intervention.png")


def render_prefix_specificity(data_dir: Path, output: Path) -> Path:
    controls = pd.read_csv(
        data_dir / "diagnostics" / "prefix_intervention" / "specificity_control.csv"
    )
    controls = controls[controls["prefix_steps"].isin([8, 16, 32])]
    steps = [8, 16, 32]
    modes = [
        ("reference", "State-matched", PURE),
        ("shuffled_reference", "Shuffled", GOLD),
        ("zero", "Zero torque", GRAY),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor="white")
    x = np.arange(len(steps))
    width = 0.24
    for offset, (mode, label, color) in zip((-width, 0.0, width), modes):
        values = (
            controls[controls["prefix_mode"] == mode]
            .set_index("prefix_steps")
            .loc[steps, "repair_rate"]
            .to_numpy()
            * 100
        )
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, steps)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Raw-actor failures repaired (%)")
    ax.set_title("The corrective sequence must match the state", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, output / "41_p7_reference_prefix_specificity.png")


def render_all(
    data_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    output.mkdir(parents=True, exist_ok=True)
    return [
        render_scorecard(frames, output),
        render_failure_footprints(frames, output),
        render_target_family_internals(data_dir, output),
        render_objective_share(data_dir, output),
        render_tolerance(frames, output),
        render_worst_seed_margin(frames, output),
        render_actor_geometry(data_dir, output),
        render_prefix_intervention(data_dir, output),
        render_prefix_specificity(data_dir, output),
    ]
