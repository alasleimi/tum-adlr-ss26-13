from __future__ import annotations

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


def _save(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int = 220,
    bbox_inches: str | None = "tight",
    pad_inches: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {"dpi": dpi, "facecolor": "white"}
    if bbox_inches is not None:
        save_options["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_options["pad_inches"] = pad_inches
    fig.savefig(path, **save_options)
    plt.close(fig)
    return path


def _style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.spines[["left", "bottom"]].set_linewidth(1.5)
    ax.tick_params(labelsize=32, colors=INK, width=1.8, length=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


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
        "Mixed, RL + supervised",
        "Uniform-start mixed",
        "Supervised actor",
        "Selected pure RL",
        "Plain SimbaV2",
        "Clean DAgger",
    ]
    summaries = [summarize_method(frames[key]) for key in order]
    trials = summaries[0].trials
    series = [
        ("Near reference", "#0f766e", np.asarray([row.near for row in summaries]) / trials),
        ("Task success", "#2563eb", np.asarray([row.task for row in summaries]) / trials),
        ("Strict win", "#ea580c", np.asarray([row.strict for row in summaries]) / trials),
    ]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12.8, 5.8), facecolor="white")
    for offset, (name, color, values) in zip((-0.22, 0.0, 0.22), series):
        bars = ax.barh(y + offset, 100 * values, height=0.19, color=color, label=name)
        for bar, value in zip(bars, 100 * values):
            ax.text(
                min(float(value) + 0.7, 101.5),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#0f172a",
            )
    ax.set_yticks(y, labels, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xlabel(f"Share of {trials:,} seed-state trials", fontsize=13)
    ax.set_title(
        "Reliability rises in layers, while strict wins follow a different ordering",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        fontsize=12,
    )
    ax.grid(axis="x", color="#dbe4ea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
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
        ax.set_xlim(angles.min(), angles.max())
        ax.set_ylim(velocities.min(), velocities.max())
        summary = summarize_method(frames[key])
        ax.set_title(
            f"{title}: {summary.failures} failures on {summary.failure_cells} starts",
            loc="left",
            color=accent,
            fontsize=20,
            fontweight="bold",
            pad=8,
        )
        ax.set_ylabel(r"$\dot{\theta}_0$ (rad/s)", fontsize=18, fontweight="bold")
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([-180, -90, 0, 90, 174])
        ax.tick_params(axis="both", labelsize=17, width=1.6, length=5)
        for spine in ax.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(3)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel(
        r"Initial angle $\theta_0$ (degrees)", fontsize=19, fontweight="bold"
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.11, hspace=0.25)
    return _save(
        fig,
        output / "77_failure_footprints_stacked_axes.png",
        dpi=220,
        bbox_inches=None,
    )


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
    labels = ["SAC", "Fast", "N8"]
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
    sensitivity = [
        np.asarray(geometry[key]["metrics"]["mean_tanh_derivative"]["seed_values"], dtype=float)
        for key in arm_keys
    ]
    critic_direction = 100 * _condition_values(
        c32, conditions, str(semantic["figure_field"])
    )
    harmful_step = 100 * _condition_values(c32, conditions, "critic_step_harmful_rate")
    helpful_recognition = 100 * np.asarray(
        [
            c34["conditions"][key]["pooled"]["failure"][
                "critic_prefers_helpful_reference_rate"
            ]
            for key in conditions
        ]
    )
    projection = {row["condition"]: row for row in c33["pooled"] if row["outcome"] == "failure"}
    failure_boundary = 100 * np.asarray(
        [projection[key]["boundary_rate"] for key in conditions]
    )
    outward_at_boundary = 100 * np.asarray(
        [projection[key]["outward_among_boundary_rate"] for key in conditions]
    )
    effective_fraction = 100 * np.asarray(
        [projection[key]["mean_effective_step_fraction"] for key in conditions]
    )

    fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.2), facecolor="white")
    x = np.arange(3)

    ax = axes[0]
    for seed in range(5):
        ax.plot(
            x,
            [saturation[family][seed] for family in range(3)],
            color="#b8c7ce",
            linewidth=2.0,
            marker="o",
            markersize=5,
            zorder=1,
        )
    medians = [np.median(values) for values in saturation]
    ax.scatter(
        x,
        medians,
        s=120,
        color=colors,
        edgecolor="white",
        linewidth=1.8,
        zorder=3,
    )
    for index, value in enumerate(medians):
        ax.text(
            index,
            value + 3.0,
            f"{value:.1f}",
            ha="center",
            fontsize=28,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.35},
        )
    ax.set_ylim(40, 100)
    ax.set_ylabel("")

    ax = axes[1]
    for seed in range(5):
        ax.plot(
            x,
            [sensitivity[family][seed] for family in range(3)],
            color="#b8c7ce",
            linewidth=2.0,
            marker="o",
            markersize=5,
            zorder=1,
        )
    medians = [np.median(values) for values in sensitivity]
    ax.scatter(
        x,
        medians,
        s=120,
        color=colors,
        edgecolor="white",
        linewidth=1.8,
        zorder=3,
    )
    for index, value in enumerate(medians):
        ax.text(
            index,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
            fontsize=28,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.35},
        )
    ax.set_ylim(0, 0.17)
    ax.set_ylabel("")

    width = 0.24
    ax = axes[2]
    for offset, values, color in (
        (-width, critic_direction, PURE),
        (0.0, helpful_recognition, MIXED),
        (width, harmful_step, ORANGE),
    ):
        ax.bar(x + offset, values, width, color=color)
    ax.set_ylim(0, 110)
    ax.set_ylabel("")

    ax = axes[3]
    for offset, values, color in (
        (-width, failure_boundary, GRAY),
        (0.0, outward_at_boundary, PURE),
        (width, effective_fraction, ORANGE),
    ):
        ax.bar(x + offset, values, width, color=color)
    ax.set_ylim(0, 115)
    ax.set_ylabel("")

    for ax, title in zip(
        axes,
        ("A  Actor bound", "B  Actor slope", "C  Critic test", "D  Clip effect"),
        strict=True,
    ):
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=32, fontweight="bold", loc="left")
        _style_axis(ax)
        ax.tick_params(labelsize=30)
    fig.subplots_adjust(left=0.052, right=0.982, top=0.80, bottom=0.25, wspace=0.32)
    return _save(
        fig,
        output / "58_poster_target_family_internals_wide.png",
        dpi=240,
        bbox_inches=None,
    )


def render_objective_share(data_dir: Path, output: Path) -> Path:
    payload = load_json(data_dir / "diagnostics" / "objective_share" / "summary.json")
    values = payload["eight_step_objective_share_percent"]
    diagnostics = payload["density_eight_step_diagnostics_percent"]
    nominal = float(values["nominal"])
    empirical_none = float(values["no_importance_empirical"])
    empirical_density_values = np.asarray(
        values["density_empirical_seed_values"], dtype=float
    )
    empirical_density = float(values["density_empirical_mean"])
    density_mean_importance_values = np.asarray(
        diagnostics["mean_importance_weight_seed_values"], dtype=float
    )
    density_ess_values = np.asarray(
        diagnostics["effective_weight_ess_fraction_seed_values"], dtype=float
    )
    density_collapsed_values = np.asarray(
        diagnostics["effective_weight_at_most_1e-3_seed_values"], dtype=float
    )
    density_mean_importance = float(diagnostics["mean_importance_weight_mean"])
    density_ess = float(diagnostics["effective_weight_ess_fraction_mean"])
    density_collapsed = float(diagnostics["effective_weight_at_most_1e-3_mean"])

    rc = {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#334155",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            r"FastSACN8 with $\lambda=0.5$: the eight-step endpoint is a small auxiliary",
            fontsize=18,
            fontweight="bold",
            y=0.985,
        )

        left_labels = ["Nominal", "No importance\nempirical", "Density\nempirical"]
        left_values = [nominal, empirical_none, empirical_density]
        left_colors = ["#64748B", "#2563EB", "#0F766E"]
        bars = axes[0].bar(left_labels, left_values, color=left_colors, width=0.62)
        axes[0].scatter(
            [2] * len(empirical_density_values),
            empirical_density_values,
            s=46,
            facecolors="white",
            edgecolors="#0F172A",
            linewidths=1.2,
            zorder=4,
            label="density actor seeds",
        )
        axes[0].set_title("A. Share of critic objective assigned to step 8")
        axes[0].set_ylabel("Objective share (%)")
        axes[0].set_ylim(0.0, 0.9)
        axes[0].grid(axis="y", alpha=0.23)
        axes[0].set_axisbelow(True)
        for bar, value in zip(bars, left_values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        axes[0].legend(frameon=False, loc="upper right", fontsize=9)

        right_labels = [
            "Mean importance\nweight",
            "Effective-weight\nESS",
            r"Weight $\leq 10^{-3}$",
        ]
        right_values = [density_mean_importance, density_ess, density_collapsed]
        right_colors = ["#0EA5A4", "#F59E0B", "#DC2626"]
        bars = axes[1].bar(right_labels, right_values, color=right_colors, width=0.62)
        for index, seed_values in enumerate(
            (
                density_mean_importance_values,
                density_ess_values,
                density_collapsed_values,
            )
        ):
            axes[1].scatter(
                [index] * len(seed_values),
                seed_values,
                s=42,
                facecolors="white",
                edgecolors="#0F172A",
                linewidths=1.1,
                zorder=4,
            )
        axes[1].set_title("B. Density weighting at the eight-step endpoint")
        axes[1].set_ylabel("Replay sequences or relative weight (%)")
        axes[1].set_ylim(0.0, 86.0)
        axes[1].grid(axis="y", alpha=0.23)
        axes[1].set_axisbelow(True)
        for bar, value in zip(bars, right_values, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        sequences_per_seed = int(payload["replay_sequences_per_seed"])
        density_total = int(payload["density_replay_sequences_total"])
        density_seed_count = len(payload["actor_seeds"]["density"])
        density_seed_text = "five" if density_seed_count == 5 else str(density_seed_count)
        no_importance_seed_values = [
            int(seed) for seed in payload["actor_seeds"]["no_importance"]
        ]
        no_importance_seed_text = (
            "seed zero"
            if no_importance_seed_values == [0]
            else f"{len(no_importance_seed_values)} actor seeds"
        )
        fig.text(
            0.5,
            0.012,
            (
                f"Density: {density_seed_text} actor seeds and "
                f"{density_total:,} replay sequences. No-importance control: "
                f"{no_importance_seed_text} and {sequences_per_seed:,} sequences."
            ),
            ha="center",
            fontsize=10.5,
            color="#475569",
        )
        fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.93), w_pad=4.0)
        return _save(
            fig,
            output / "36_fastsacn_objective_share.png",
            dpi=220,
            bbox_inches=None,
        )


def render_tolerance(frames: Mapping[str, pd.DataFrame], output: Path) -> Path:
    epsilons = np.linspace(0, 20, 81)
    fig, ax = plt.subplots(figsize=(11.2, 4.1), facecolor="white")
    for key, label, color in (
        ("mixed_selected", "RL + supervised", MIXED),
        ("pure_selected_q41", "Pure RL", PURE),
    ):
        seed_curves, mean_curve = tolerance_curve(frames[key], epsilons)
        for curve in seed_curves:
            ax.plot(epsilons, curve, color=color, alpha=0.16, linewidth=1.6)
        ax.plot(epsilons, mean_curve, color=color, linewidth=4.5, label=label)
        at_five = mean_curve[np.argmin(np.abs(epsilons - 5.0))]
        ax.scatter(
            [5],
            [at_five],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=2,
            zorder=5,
        )
        ax.annotate(
            f"{at_five:.3%}",
            (5, at_five),
            xytext=(18, -27 if label == "RL + supervised" else 15),
            textcoords="offset points",
            fontsize=15,
            color=color,
            fontweight="bold",
        )
    ax.axvline(5, color="#64748b", linestyle="--", linewidth=2)
    ax.text(
        5.18,
        0.18,
        "reported tolerance = 5",
        rotation=90,
        va="bottom",
        color="#64748b",
        fontsize=12,
    )
    ax.set_xlim(0, 20)
    ax.set_ylim(0.1, 1.01)
    ax.set_xlabel("Return tolerance", fontsize=15)
    ax.set_ylabel("Fraction near reference", fontsize=15)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(labelsize=13, colors=MUTED)
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False, fontsize=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.8)
    return _save(
        fig,
        output / "111_tolerance_curve_wide.png",
        dpi=220,
        pad_inches=0.08,
    )


def _worst_seed_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame.assign(return_margin=frame["return"] - frame["best_known_return"])
    grouped = work.groupby(["theta_dot", "theta_degrees"], as_index=False)["return_margin"].min()
    angles = np.sort(grouped["theta_degrees"].unique())
    velocities = np.sort(grouped["theta_dot"].unique())
    grid = grouped.pivot(index="theta_dot", columns="theta_degrees", values="return_margin")
    return angles, velocities, grid.loc[velocities, angles].to_numpy()


def render_worst_seed_margin(frames: Mapping[str, pd.DataFrame], output: Path) -> Path:
    cmap = LinearSegmentedColormap.from_list(
        "return_margin",
        [
            (0.00, "#8f1d24"),
            (0.25, "#ef6a3a"),
            (0.375, "#f4b860"),
            (0.50, "#fffdf8"),
            (0.70, "#8ddbd0"),
            (1.00, "#2367d8"),
        ],
    )
    norm = Normalize(vmin=-20.0, vmax=20.0, clip=True)
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 5.0), facecolor="white")
    image = None
    for ax, key, title, accent in (
        (axes[0], "mixed_selected", "RL + supervised", MIXED),
        (axes[1], "pure_selected_q41", "Pure RL", PURE),
    ):
        angles, velocities, grid = _worst_seed_grid(frames[key])
        image = ax.imshow(
            np.clip(grid, -20.0, 20.0),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[angles.min(), angles.max(), velocities.min(), velocities.max()],
        )
        ax.contour(
            angles,
            velocities,
            grid,
            levels=[-5.0],
            colors=[INK],
            linewidths=2.2,
            linestyles=["solid"],
        )
        ax.set_title(
            title,
            loc="left",
            color=accent,
            fontsize=28,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlim(angles.min(), angles.max())
        ax.set_ylim(velocities.min(), velocities.max())
        ax.set_xticks([-180, -90, 0, 90, 174])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel(
            r"Initial angle $\theta_0$ (degrees)", fontsize=22, fontweight="bold"
        )
        ax.tick_params(axis="both", labelsize=20, width=1.8, length=5)
        for spine in ax.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(3.2)
    axes[0].set_ylabel(
        r"Initial velocity $\dot{\theta}_0$ (rad/s)",
        fontsize=19,
        fontweight="bold",
    )
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.070,
        pad=0.18,
        aspect=45,
        ticks=[-20, -5, 0, 20],
    )
    colorbar.ax.tick_params(labelsize=18, width=1.5)
    colorbar.set_label(
        r"Worst-seed return margin $R_\pi - R^*$ (clipped); dark contour = near-reference boundary at $-5$",
        fontsize=17,
        fontweight="bold",
        labelpad=7,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.32, wspace=0.13)
    return _save(
        fig,
        output / "79_worst_seed_return_margin.png",
        dpi=230,
        pad_inches=0.08,
    )


def render_actor_geometry(data_dir: Path, output: Path) -> Path:
    payload = load_json(data_dir / "diagnostics" / "actor_geometry" / "summary.json")
    arms = payload["arms"]
    keys = [
        "p0_simba_onestep_utd1_100k",
        "p1_simba_fastsacn8_lambda1_utd1_100k",
        "p2_simba_sacn8_lambda1_utd1_100k",
        "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k",
    ]
    labels = [str(arms[key]["label"]) for key in keys]
    metrics = (
        (
            "deterministic_action_saturation_fraction_abs_ge_0p995",
            "Actor saturation",
            "fraction at ≥99.5% of torque bound",
        ),
        (
            "mean_tanh_derivative",
            "Actor action sensitivity",
            r"mean $1-\tanh^2(z)$",
        ),
        (
            "reflection_action_abs_error_mean",
            "Reflection error",
            r"mean $|a(s)+a(Ms)|$",
        ),
    )
    colors = ["#334155", "#2563eb", "#0f766e", "#b91c1c"]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
    x = np.arange(len(keys))
    for axis, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        for index, (key, color) in enumerate(zip(keys, colors, strict=True)):
            values = np.asarray(
                arms[key]["metrics"][metric]["seed_values"], dtype=float
            )
            jitter = np.linspace(-0.11, 0.11, len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=44,
                color=color,
                alpha=0.9,
                zorder=3,
            )
            axis.plot(
                [index - 0.22, index + 0.22],
                [np.median(values), np.median(values)],
                color="#0f172a",
                linewidth=2.5,
                zorder=4,
            )
        axis.set_xticks(x, labels, rotation=16)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in {
            "deterministic_action_saturation_fraction_abs_ge_0p995",
            "mean_tanh_derivative",
        }:
            axis.set_ylim(bottom=0)
    fig.suptitle(
        "Temporal-target recipes change the learned SimbaV2 actor",
        fontsize=18,
        fontweight="bold",
    )
    points = int(payload["points_per_seed"])
    fig.text(
        0.5,
        -0.035,
        (
            "Five independently trained actors per method on the same "
            f"{points:,} locked off-grid initial states. No rollouts or reference "
            "returns are used."
        ),
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    return _save(
        fig,
        output / "16_completed_target_actor_geometry.png",
        dpi=260,
        bbox_inches=None,
    )


def render_prefix_intervention(data_dir: Path, output: Path) -> Path:
    aggregate = pd.read_csv(data_dir / "diagnostics" / "prefix_intervention" / "aggregate.csv")
    colors = {"pure RL actor": "#2563EB", "mixed supervised actor": "#0F766E"}
    conditions = list(dict.fromkeys(aggregate["condition"].astype(str)))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), constrained_layout=True)
    labels: list[str] = []
    for condition in conditions:
        rows = aggregate[aggregate["condition"] == condition].sort_values("prefix_steps")
        x = np.arange(len(rows))
        labels = [str(value) for value in rows["prefix_steps"]]
        color = colors.get(condition, "#334155")
        axes[0].plot(
            x,
            100 * rows["near_rate"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=condition,
        )
        axes[1].plot(
            x,
            100 * rows["repair_rate"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=condition,
        )
        axes[2].plot(
            x,
            100 * rows["task_rate"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=condition,
        )
    for axis in axes:
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_xlabel("Initial steps controlled by stored reference")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("A. Near-reference success", loc="left", fontweight="bold")
    axes[0].set_ylabel("All seed-state trials (%)")
    axes[1].set_title("B. Original failures repaired", loc="left", fontweight="bold")
    axes[1].set_ylabel("Original actor failures (%)")
    axes[2].set_title("C. Task success", loc="left", fontweight="bold")
    axes[2].set_ylabel("All seed-state trials (%)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Reference-prefix intervention localizes how long correction must persist",
        fontsize=13,
        fontweight="bold",
    )
    return _save(fig, output / "40_p7_reference_prefix_intervention.png")


def render_prefix_specificity(data_dir: Path, output: Path) -> Path:
    controls = pd.read_csv(
        data_dir / "diagnostics" / "prefix_intervention" / "specificity_control.csv"
    )
    labels = {
        "reference": "state-matched reference",
        "shuffled_reference": "state-shuffled reference actions",
        "zero": "zero torque",
    }
    colors = {
        "reference": "#2563EB",
        "shuffled_reference": "#D97706",
        "zero": "#64748B",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for mode in ("reference", "shuffled_reference", "zero"):
        rows = controls[controls["prefix_mode"] == mode].sort_values("prefix_steps")
        ax.plot(
            rows["prefix_steps"],
            100 * rows["repair_rate"],
            marker="o",
            linewidth=2.3,
            color=colors[mode],
            label=labels[mode],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
    ax.set_xlabel("Initial prefix steps")
    ax.set_ylabel("Original pure-actor failures repaired (%)")
    ax.set_title(
        "Does recovery require a state-matched corrective sequence?",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
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
