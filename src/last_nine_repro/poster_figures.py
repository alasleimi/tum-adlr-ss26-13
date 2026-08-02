from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Arc, Circle, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from .metrics import summarize_method, tolerance_curve


INK = "#102a35"
MUTED = "#526b76"
MIXED = "#0d9f90"
PURE = "#3478ef"
ORANGE = "#f0623a"
PAPER = "#f7f4ec"

POSTER_FIGURE_NAMES = (
    "101_sequence_repair_wide.png",
    "102_baseline_scorecard_panel.png",
    "111_tolerance_curve_wide.png",
    "116_recovery_atlas_3starts.png",
    "118_failure_footprints_clean.png",
)

RECOVERY_KEYS = (
    "starts_theta",
    "starts_theta_degrees",
    "starts_theta_dot",
    "mixed_theta",
    "mixed_velocity",
    "pure_theta",
    "pure_velocity",
)


def _save(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int,
    tight: bool = False,
    facecolor: str = "white",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {"dpi": dpi, "facecolor": facecolor}
    if tight:
        kwargs.update({"bbox_inches": "tight", "pad_inches": 0.08})
    figure.savefig(path, **kwargs)
    plt.close(figure)
    return path


def render_scorecard(
    frames: Mapping[str, pd.DataFrame], output: Path
) -> Path:
    order = (
        "mixed_selected",
        "mixed_actor_only",
        "pure_selected_q41",
        "simba_onestep",
        "canonical_dagger",
    )
    labels = (
        "RL + supervision",
        "Supervised actor",
        "Pure RL",
        "Plain SimbaV2",
        "Clean DAgger",
    )
    summaries = [summarize_method(frames[key]) for key in order]
    metrics = (
        (
            "Near reference",
            np.asarray([100 * row.near / row.trials for row in summaries]),
            "#0d7f73",
        ),
        (
            "Task success",
            np.asarray([100 * row.task / row.trials for row in summaries]),
            "#2f68e8",
        ),
        (
            "Strict win",
            np.asarray([100 * row.strict / row.trials for row in summaries]),
            "#ef5a0a",
        ),
    )
    y = np.arange(len(labels))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
        }
    )
    figure, axis = plt.subplots(figsize=(14, 10), facecolor="white")
    for (name, values, color), offset in zip(metrics, (-0.24, 0.0, 0.24)):
        bars = axis.barh(y + offset, values, height=0.2, color=color, label=name)
        for bar, value in zip(bars, values):
            axis.text(
                min(float(value) + 1.0, 101.0),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=16,
                fontweight="bold",
                color=INK,
            )
    axis.set_xlim(0, 105)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_yticks(y, labels, fontsize=20)
    axis.invert_yaxis()
    trials = summaries[0].trials
    axis.set_xlabel(
        f"Share of {trials:,} seed-state trials (%)",
        fontsize=18,
        fontweight="bold",
        labelpad=12,
    )
    axis.grid(axis="x", color="#d7e0e4", linewidth=1.4)
    axis.set_axisbelow(True)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        fontsize=17,
        handlelength=1.8,
        columnspacing=1.2,
    )
    axis.tick_params(axis="x", labelsize=16)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    figure.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.14)
    return _save(figure, output / "102_baseline_scorecard_panel.png", dpi=180)


def render_tolerance(
    frames: Mapping[str, pd.DataFrame], output: Path
) -> Path:
    epsilons = np.linspace(0, 20, 81)
    figure, axis = plt.subplots(figsize=(11.2, 4.1), facecolor="white")
    for key, label, color in (
        ("mixed_selected", "RL + supervised", MIXED),
        ("pure_selected_q41", "Pure RL", PURE),
    ):
        seed_curves, mean_curve = tolerance_curve(frames[key], epsilons)
        for curve in seed_curves:
            axis.plot(epsilons, curve, color=color, alpha=0.16, linewidth=1.6)
        axis.plot(epsilons, mean_curve, color=color, linewidth=4.5, label=label)
        at_five = float(mean_curve[np.argmin(np.abs(epsilons - 5.0))])
        axis.scatter(
            [5],
            [at_five],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=2,
            zorder=5,
        )
        axis.annotate(
            f"{at_five:.3%}",
            (5, at_five),
            xytext=(18, -27 if key == "mixed_selected" else 15),
            textcoords="offset points",
            fontsize=15,
            color=color,
            fontweight="bold",
        )
    axis.axvline(5, color="#64748b", linestyle="--", linewidth=2)
    axis.text(
        5.18,
        0.18,
        "reported tolerance = 5",
        rotation=90,
        va="bottom",
        color="#64748b",
        fontsize=12,
    )
    axis.set_xlim(0, 20)
    axis.set_ylim(0.1, 1.01)
    axis.set_xlabel("Return tolerance", fontsize=15)
    axis.set_ylabel("Fraction near reference", fontsize=15)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.tick_params(labelsize=13, colors=MUTED)
    axis.grid(alpha=0.18)
    axis.legend(loc="lower right", frameon=False, fontsize=15)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.8)
    return _save(
        figure,
        output / "111_tolerance_curve_wide.png",
        dpi=220,
        tight=True,
    )


def render_sequence_repair(data_dir: Path, output: Path) -> Path:
    frame = pd.read_csv(
        data_dir / "diagnostics" / "prefix_intervention" / "specificity_control.csv"
    )
    modes = (
        ("reference", "State-matched actions", PURE),
        ("shuffled_reference", "Same actions, shuffled", "#ef8b26"),
        ("zero", "Zero torque", "#758797"),
    )
    x_order = [1, 8, 16, 32]
    figure, axis = plt.subplots(figsize=(18, 6.5), facecolor="white")
    reference_values: np.ndarray | None = None
    for mode, label, color in modes:
        rows = frame.loc[frame["prefix_mode"].eq(mode)].set_index("prefix_steps")
        values = 100 * rows.loc[x_order, "repair_rate"].to_numpy()
        axis.plot(
            range(len(x_order)),
            values,
            marker="o",
            markersize=12,
            linewidth=5,
            color=color,
            label=label,
            zorder=3,
        )
        if mode == "reference":
            reference_values = values
            for index, value in enumerate(values):
                axis.text(
                    index,
                    value + (4.2 if index < 2 else 3.0),
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=19,
                    fontweight="bold",
                    color=color,
                )
    if reference_values is None:
        raise ValueError("Prefix evidence contains no state-matched reference rows")
    failures = int(
        frame.loc[
            (frame["prefix_mode"] == "reference") & (frame["prefix_steps"] == 32),
            "failure_trials",
        ].iloc[0]
    )
    repaired = round(failures * float(reference_values[-1]) / 100)
    axis.annotate(
        f"32 matched steps repair {repaired:,} of {failures:,} failures",
        xy=(3, float(reference_values[-1])),
        xytext=(1.55, 78),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 2.2},
        fontsize=18,
        fontweight="bold",
        color=INK,
        ha="center",
    )
    shuffled = frame.loc[
        (frame["prefix_mode"] == "shuffled_reference")
        & (frame["prefix_steps"] == 32),
        "repair_rate",
    ].iloc[0]
    zero = frame.loc[
        (frame["prefix_mode"] == "zero") & (frame["prefix_steps"] == 32),
        "repair_rate",
    ].iloc[0]
    axis.text(
        2.98,
        5.0,
        f"shuffled {100 * shuffled:.1f}%  ·  zero torque {100 * zero:.1f}%",
        ha="right",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color="#ef8b26",
    )
    axis.set_xticks(range(len(x_order)), [str(value) for value in x_order])
    axis.set_xlabel(
        "Initial steps replaced before the same frozen actor resumes",
        fontsize=18,
        fontweight="bold",
        labelpad=12,
    )
    axis.set_ylabel(
        "Original actor failures repaired (%)",
        fontsize=18,
        fontweight="bold",
        labelpad=12,
    )
    axis.set_ylim(-4, 108)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(axis="y", color="#d7e0e4", linewidth=1.4)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=17,
        handlelength=2.3,
        columnspacing=1.4,
    )
    axis.tick_params(labelsize=17)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    figure.subplots_adjust(left=0.095, right=0.985, top=0.94, bottom=0.22)
    return _save(figure, output / "101_sequence_repair_wide.png", dpi=180)


def _failure_grid(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    work = frame.assign(
        failure=1 - frame["near_best_known_return_eps"].astype(int)
    )
    grouped = (
        work.groupby(["theta_dot", "theta_degrees"], as_index=False)["failure"]
        .sum()
        .sort_values(["theta_dot", "theta_degrees"])
    )
    angles = np.sort(grouped["theta_degrees"].unique())
    velocities = np.sort(grouped["theta_dot"].unique())
    pivot = grouped.pivot(
        index="theta_dot", columns="theta_degrees", values="failure"
    )
    grid = pivot.loc[velocities, angles].to_numpy()
    return angles, velocities, grid, int(work["failure"].sum()), int((grid > 0).sum())


def render_failure_footprints(
    frames: Mapping[str, pd.DataFrame], output: Path
) -> Path:
    colors = ["#fffdf8", "#9de4dc", "#f5df83", "#f3a43b", "#e85c38", "#8f1d24"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), facecolor="white")
    image = None
    for axis, data, name, accent in (
        (axes[0], _failure_grid(frames["pure_selected_q41"]), "Pure RL", PURE),
        (axes[1], _failure_grid(frames["mixed_selected"]), "RL + supervision", "#0f9f91"),
    ):
        angles, velocities, grid, failed_trials, failed_starts = data
        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[angles.min(), angles.max(), velocities.min(), velocities.max()],
        )
        axis.set_title(
            f"{name}\n{failed_trials} failures across {failed_starts} starts",
            fontsize=16,
            fontweight="bold",
            color=INK,
            pad=10,
        )
        axis.set_xlabel("Initial angle (degrees)", fontsize=13)
        axis.tick_params(labelsize=11)
        for spine in axis.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(2.2)
    axes[0].set_ylabel("Initial angular velocity", fontsize=13)
    axes[1].set_yticklabels([])
    if image is None:
        raise RuntimeError("No failure footprint image was produced")
    colorbar = figure.colorbar(
        image, ax=axes, ticks=np.arange(0, 6), fraction=0.035, pad=0.025
    )
    colorbar.set_label("Number of the five actors that fail", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    figure.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.13, wspace=0.12)
    path = output / "118_failure_footprints_clean.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _load_recovery_evidence(data_dir: Path) -> dict[str, np.ndarray]:
    path = data_dir / "diagnostics" / "recovery_atlas" / "trajectories.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Recovery-atlas evidence is missing: {path}. "
            "Run scripts/extract_recovery_atlas_evidence.py once from preserved checkpoints."
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in RECOVERY_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Recovery-atlas evidence lacks: {', '.join(missing)}")
        payload = {key: np.asarray(archive[key]) for key in RECOVERY_KEYS}
    for key in ("mixed_theta", "mixed_velocity", "pure_theta", "pure_velocity"):
        if payload[key].shape != (3, 5, 201):
            raise ValueError(f"{key} has shape {payload[key].shape}, expected (3, 5, 201)")
        if not np.isfinite(payload[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in ("starts_theta", "starts_theta_degrees", "starts_theta_dot"):
        if payload[key].shape != (3,):
            raise ValueError(f"{key} has shape {payload[key].shape}, expected (3,)")
    provenance_path = path.with_name("provenance.json")
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("schema_version") != 1:
        raise ValueError(f"Unsupported recovery-atlas provenance: {provenance_path}")
    return payload


def _draw_fan(
    axis: plt.Axes,
    theta_history: np.ndarray,
    velocity_history: np.ndarray,
    *,
    step: int,
    color: str,
) -> None:
    thetas = theta_history[:, step]
    velocities = velocity_history[:, step]
    upright = (np.cos(thetas) >= 0.95) & (np.abs(velocities) <= 1.0)
    fraction = float(np.mean(upright))
    warm = np.array([1.0, 0.94, 0.90])
    cool = np.array([0.88, 0.96, 0.94])
    background = warm * (1.0 - fraction) + cool * fraction
    axis.set_facecolor(background)
    axis.add_patch(
        FancyBboxPatch(
            (-1.22, -1.22),
            2.44,
            2.44,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=background,
            edgecolor="#cbd8dc",
            linewidth=1.1,
            zorder=-3,
        )
    )
    axis.add_patch(
        Arc(
            (0, 0),
            1.92,
            1.92,
            theta1=72,
            theta2=108,
            color=MIXED,
            linewidth=10,
            alpha=0.16,
            zorder=-1,
        )
    )
    for seed, (theta, velocity) in enumerate(zip(thetas, velocities)):
        x, y = float(np.sin(theta)), float(np.cos(theta))
        alpha = 0.34 + 0.13 * seed
        axis.plot(
            [0, x],
            [0, y],
            color=color,
            linewidth=7.2,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2 + seed,
        )
        axis.add_patch(
            Circle(
                (x, y),
                0.094,
                facecolor=color,
                edgecolor="white",
                linewidth=1.2,
                alpha=alpha,
                zorder=4 + seed,
            )
        )
        tangent = np.array([np.cos(theta), -np.sin(theta)])
        velocity_mark = np.sign(velocity) * min(abs(float(velocity)) / 8.0, 0.20)
        endpoint = np.array([x, y]) + velocity_mark * tangent
        axis.plot(
            [x, endpoint[0]],
            [y, endpoint[1]],
            color=ORANGE,
            linewidth=2.0,
            alpha=alpha,
            zorder=8,
        )
    axis.add_patch(
        Circle((0, 0), 0.062, facecolor=INK, edgecolor="white", linewidth=1, zorder=12)
    )
    axis.axhline(0, color="#cdd9dd", linewidth=0.8, zorder=-2)
    axis.text(
        0.05,
        0.07,
        f"upright now  {int(np.sum(upright))}/5",
        transform=axis.transAxes,
        color=INK,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_row_label(
    axis: plt.Axes,
    *,
    start_number: int,
    theta_degrees: float,
    theta_dot: float,
) -> None:
    axis.set_facecolor(INK)
    axis.text(
        0.08,
        0.76,
        f"Difficult start {start_number}",
        transform=axis.transAxes,
        color="white",
        fontsize=18.5,
        fontweight="bold",
        va="center",
    )
    axis.text(
        0.08,
        0.48,
        rf"$\theta_0={theta_degrees:.0f}^\circ,\ \dot{{\theta}}_0={theta_dot:.2f}$",
        transform=axis.transAxes,
        color="#c9d9df",
        fontsize=15.5,
        va="center",
    )
    axis.text(
        0.08,
        0.22,
        "final return:  mixed 5/5  ·  pure 0/5",
        transform=axis.transAxes,
        color=MIXED,
        fontsize=11.5,
        fontweight="bold",
        va="center",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def render_recovery_atlas(data_dir: Path, output: Path) -> Path:
    evidence = _load_recovery_evidence(data_dir)
    figure = plt.figure(figsize=(16.8, 6.8), facecolor=PAPER)
    layout = figure.add_gridspec(
        3,
        5,
        width_ratios=[1.22, 1, 1, 1, 1],
        hspace=0.065,
        wspace=0.045,
    )
    columns = (
        ("mixed · step 24", "mixed", MIXED, 24),
        ("mixed · step 64", "mixed", MIXED, 64),
        ("pure RL · step 24", "pure", PURE, 24),
        ("pure RL · step 64", "pure", PURE, 64),
    )
    for row in range(3):
        label_axis = figure.add_subplot(layout[row, 0])
        _draw_row_label(
            label_axis,
            start_number=row + 1,
            theta_degrees=float(evidence["starts_theta_degrees"][row]),
            theta_dot=float(evidence["starts_theta_dot"][row]),
        )
        for column, (title, prefix, color, step) in enumerate(columns, start=1):
            axis = figure.add_subplot(layout[row, column])
            _draw_fan(
                axis,
                evidence[f"{prefix}_theta"][row],
                evidence[f"{prefix}_velocity"][row],
                step=step,
                color=color,
            )
            if row == 0:
                axis.set_title(
                    title,
                    fontsize=16,
                    color=MIXED if column <= 2 else PURE,
                    fontweight="bold",
                    pad=6,
                )
    figure.text(
        0.995,
        0.006,
        "Five translucent rods = five retraining seeds  ·  orange tick = angular velocity",
        color="#58707a",
        fontsize=12,
        ha="right",
        va="bottom",
    )
    figure.subplots_adjust(left=0.008, right=0.995, top=0.92, bottom=0.035)
    return _save(
        figure,
        output / "116_recovery_atlas_3starts.png",
        dpi=300,
        facecolor=PAPER,
    )


def render_all_poster(
    data_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    return [
        render_sequence_repair(data_dir, output),
        render_scorecard(frames, output),
        render_tolerance(frames, output),
        render_recovery_atlas(data_dir, output),
        render_failure_footprints(frames, output),
    ]
