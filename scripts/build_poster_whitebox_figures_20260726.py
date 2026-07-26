from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "deliverables" / "chasing_nines_20260723"
FIG = DELIVERY / "figures"
SCORECARD = DELIVERY / "verified_scorecard.csv"
GEOMETRY = (
    ROOT
    / "reports"
    / "plan2307_mechanistic_20260723"
    / "completed_target_actor_geometry"
    / "summary.json"
)

INK = "#102a35"
MUTED = "#526b76"
PURE = "#3478ef"
MIXED = "#0d9f90"
ORANGE = "#ef6a3a"
GRAY = "#8193a1"
GRID = "#dce5e8"


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.spines[["left", "bottom"]].set_linewidth(1.5)
    ax.tick_params(labelsize=32, colors=INK, width=1.8, length=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def build_scorecard() -> None:
    df = pd.read_csv(SCORECARD).set_index("method_key")
    order = ["mixed", "mixed_uniform", "supervised", "pure", "simba", "dagger"]
    labels = [
        "RL + supervised",
        "Uniform-start mixed",
        "Supervised actor",
        "Selected pure RL",
        "Plain SimbaV2",
        "Clean DAgger",
    ]
    near = 100 * df.loc[order, "near_best_known_return_eps_rate"].to_numpy()
    task = 100 * df.loc[order, "task_success_rate"].to_numpy()
    strict = 100 * df.loc[order, "beats_best_known_return_rate"].to_numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.5, 4.2),
        gridspec_kw={"width_ratios": [1.65, 0.75]},
        facecolor="white",
    )
    y = np.arange(len(order))

    ax = axes[0]
    for row in range(len(order)):
        ax.plot(
            [task[row], near[row]],
            [row, row],
            color="#c4d0d5",
            linewidth=5,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(task, y, s=180, color=PURE, label="Task success", zorder=3)
    ax.scatter(near, y, s=180, color=MIXED, label="Near reference", zorder=3)
    for row, (near_value, task_value) in enumerate(zip(near, task, strict=True)):
        ax.text(
            task_value - 0.22,
            row - 0.29,
            f"{task_value:.1f}",
            ha="right",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=PURE,
        )
        ax.text(
            near_value + 0.22,
            row + 0.29,
            f"{near_value:.1f}",
            ha="left",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=MIXED,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(82, 101.5)
    ax.set_xticks([85, 90, 95, 100])
    ax.set_xlabel("Success rate (%)", fontsize=27, color=INK)
    ax.set_title(
        "Reliability and task success",
        loc="left",
        fontsize=29,
        fontweight="bold",
        color=INK,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.12),
        ncol=2,
        frameon=False,
        fontsize=24,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    style_axis(ax, "x")

    ax = axes[1]
    bars = ax.barh(y, strict, height=0.50, color=ORANGE)
    for bar, value in zip(bars, strict, strict=True):
        ax.text(
            value + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=25,
            fontweight="bold",
            color=INK,
        )
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlim(0, 28.5)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel("Rate (%)", fontsize=27, color=INK)
    ax.set_title(
        "Strict wins",
        loc="left",
        fontsize=27,
        fontweight="bold",
        color=INK,
    )
    style_axis(ax, "x")

    fig.subplots_adjust(left=0.30, right=0.985, top=0.61, bottom=0.22, wspace=0.17)
    fig.savefig(
        FIG / "54_poster_scorecard_wide.png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def build_selected_pure_internal_diagnostics() -> None:
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))["arms"]
    p0 = geometry["p0_simba_onestep_utd1_100k"]["metrics"]
    p7 = geometry[
        "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k"
    ]["metrics"]
    sat0 = 100 * np.asarray(
        p0["deterministic_action_saturation_fraction_abs_ge_0p995"]["seed_values"]
    )
    sat7 = 100 * np.asarray(
        p7["deterministic_action_saturation_fraction_abs_ge_0p995"]["seed_values"]
    )
    sens0 = np.asarray(p0["mean_tanh_derivative"]["seed_values"])
    sens7 = np.asarray(p7["mean_tanh_derivative"]["seed_values"])

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 5.1), facecolor="white")
    labels = ["1-step SAC", "FastSACN8"]
    colors = [GRAY, PURE]

    ax = axes[0]
    for i, (vals, color) in enumerate(((sat0, GRAY), (sat7, PURE))):
        ax.scatter(np.full(5, i) + np.linspace(-0.08, 0.08, 5), vals, s=62, color=color)
        ax.hlines(np.median(vals), i - 0.24, i + 0.24, color=INK, linewidth=3)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Torque-limit actions (%)", fontsize=32, color=INK)
    ax.set_title("Actor\nsaturation", fontsize=32, fontweight="bold", color=INK)
    ax.set_ylim(45, 85)
    ax.text(0, 47.5, "59.1", ha="center", fontsize=32, color=INK)
    ax.text(1, 47.5, "70.3", ha="center", fontsize=32, color=INK)
    style_axis(ax, "y")

    ax = axes[1]
    for i, (vals, color) in enumerate(((sens0, GRAY), (sens7, PURE))):
        ax.scatter(np.full(5, i) + np.linspace(-0.08, 0.08, 5), vals, s=62, color=color)
        ax.hlines(np.median(vals), i - 0.24, i + 0.24, color=INK, linewidth=3)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(r"Mean $1-\tanh^2(u)$", fontsize=32, color=INK)
    ax.set_title("Output\nsensitivity", fontsize=32, fontweight="bold", color=INK)
    ax.set_ylim(0.035, 0.16)
    ax.text(0, 0.039, ".120", ha="center", fontsize=32, color=INK)
    ax.text(1, 0.039, ".089", ha="center", fontsize=32, color=INK)
    style_axis(ax, "y")

    ax = axes[2]
    raw = [91.864, 89.937]
    deployed = [94.489, 95.909]
    x = np.arange(2)
    ax.plot(x, raw, "-o", color=GRAY, linewidth=3.5, markersize=8, label="Raw actor")
    ax.plot(
        x,
        deployed,
        "-o",
        color=PURE,
        linewidth=3.5,
        markersize=8,
        label="Reflection + Q-search",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Near reference (%)", fontsize=32, color=INK)
    ax.set_title("Frozen\ndeployment", fontsize=32, fontweight="bold", color=INK)
    ax.set_ylim(88.5, 97)
    for i, value in enumerate(raw):
        ax.text(i, value - 0.65, f"{value:.2f}", ha="center", fontsize=32, color=GRAY)
    for i, value in enumerate(deployed):
        ax.text(i, value + 0.35, f"{value:.2f}", ha="center", fontsize=32, color=PURE)
    ax.legend(frameon=False, fontsize=32, loc="lower right")
    style_axis(ax, "y")

    ax = axes[3]
    names = ["Precision", "Recall", "Balanced\nacc.", "ROC AUC"]
    values = [0.782, 0.703, 0.623, 0.655]
    bars = ax.bar(np.arange(4), values, color=[PURE, PURE, "#8a5be8", "#8a5be8"])
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.8)
    ax.set_xticks(np.arange(4), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate", fontsize=32, color=INK)
    ax.set_title("Critic ranking\non failures", fontsize=32, fontweight="bold", color=INK)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.3f}",
            ha="center",
            fontsize=32,
            fontweight="bold",
            color=INK,
        )
    style_axis(ax, "y")

    fig.text(
        0.5,
        0.015,
        (
            "Panels 1–3: matched 100k-step SimbaV2 families, five paired seeds; "
            "FastSACN8 changes the target and critic update dose together. "
            "Panel 4: 2,259 selected pure-RL failure rows at the first actionable disagreement."
        ),
        ha="center",
        fontsize=10.8,
        color=MUTED,
        visible=False,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.28, wspace=0.62)
    fig.savefig(FIG / "55_poster_selected_pure_internal.png", dpi=220, facecolor="white")
    plt.close(fig)


def build_joint_gradient_compact() -> None:
    run_dir = (
        ROOT
        / "runs"
        / "systematic_joint_staging_vs_mixing_20260722"
        / "sm2_joint_bc_plus_sac_same_update_after6k"
        / "seed0"
    )
    metrics = pd.read_csv(run_dir / "metrics.csv")

    def series(name: str) -> pd.Series:
        rows = metrics.loc[metrics["name"].eq(name), ["step", "value"]].copy()
        rows["step"] = pd.to_numeric(rows["step"], errors="raise").astype(int)
        rows["value"] = pd.to_numeric(rows["value"], errors="raise")
        return (
            rows.drop_duplicates("step", keep="last")
            .sort_values("step")
            .set_index("step")["value"]
        )

    bc = series("actor_weighted_bc_gradient_norm_mean")
    sac = series("actor_weighted_sac_gradient_norm_mean")
    cosine = series("actor_sac_bc_gradient_cosine_mean")
    cosine_min = series("actor_sac_bc_gradient_cosine_min")
    cosine_max = series("actor_sac_bc_gradient_cosine_max")
    common = bc.index.intersection(sac.index).intersection(cosine.index)
    common = common[common >= 6000]
    bc = bc.loc[common]
    sac = sac.loc[common]
    cosine = cosine.loc[common]
    cosine_min = cosine_min.reindex(common)
    cosine_max = cosine_max.reindex(common)
    ratio = bc / sac

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.4), facecolor="white")
    ax = axes[0]
    ax.plot(common / 1000, ratio, color=MIXED, linewidth=4, marker="o", markersize=7)
    ax.axhline(np.median(ratio), color=INK, linewidth=2, linestyle="--")
    ax.set_xlim(5.5, 25.5)
    ax.set_ylim(25, 47)
    ax.set_ylabel("")
    ax.set_xlabel("Steps (k)", fontsize=52, color=INK)
    ax.set_title("A  Norm ratio", fontsize=52, fontweight="bold", loc="left", color=INK)
    style_axis(ax, "y")

    ax = axes[1]
    ax.fill_between(
        common / 1000,
        cosine_min.to_numpy(float),
        cosine_max.to_numpy(float),
        color=ORANGE,
        alpha=0.20,
    )
    ax.plot(common / 1000, cosine, color=ORANGE, linewidth=4, marker="o", markersize=7)
    ax.axhline(0, color=INK, linewidth=2, linestyle="--")
    ax.set_xlim(5.5, 25.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("")
    ax.set_xlabel("Steps (k)", fontsize=52, color=INK)
    ax.set_title("B  Cosine", fontsize=52, fontweight="bold", loc="left", color=INK)
    style_axis(ax, "y")
    for ax in axes:
        ax.tick_params(labelsize=50)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.72, bottom=0.43, wspace=0.34)
    fig.savefig(FIG / "56_poster_joint_gradient_compact.png", dpi=240, facecolor="white")
    plt.close(fig)


def build_selected_pure_internal_diagnostics() -> None:
    """Poster-scale 2x2 white-box diagnostic with no sub-minimum plot text."""
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))["arms"]
    p0 = geometry["p0_simba_onestep_utd1_100k"]["metrics"]
    p7 = geometry[
        "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k"
    ]["metrics"]
    sat0 = 100 * np.asarray(
        p0["deterministic_action_saturation_fraction_abs_ge_0p995"]["seed_values"]
    )
    sat7 = 100 * np.asarray(
        p7["deterministic_action_saturation_fraction_abs_ge_0p995"]["seed_values"]
    )
    sens0 = np.asarray(p0["mean_tanh_derivative"]["seed_values"])
    sens7 = np.asarray(p7["mean_tanh_derivative"]["seed_values"])

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 5.1), facecolor="white")
    for ax in axes:
        ax.set_facecolor("#f7fafb")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(2)

    ax = axes[0]
    ax.set_title("Saturation", fontsize=32, fontweight="bold", color=INK)
    ax.barh([0.67, 0.30], [0.591, 0.703], height=0.20, color=[GRAY, PURE])
    ax.text(0.04, 0.67, "1-step 59.1%", va="center", fontsize=32, fontweight="bold", color=INK)
    ax.text(0.04, 0.30, "Fast 70.3%", va="center", fontsize=32, fontweight="bold", color=INK)

    ax = axes[1]
    ax.set_title(
        "Sensitivity",
        fontsize=32,
        fontweight="bold",
        color=INK,
    )
    ax.barh([0.67, 0.30], [0.120 / 0.15, 0.089 / 0.15], height=0.20, color=[GRAY, PURE])
    ax.text(0.04, 0.67, "1-step .120", va="center", fontsize=32, fontweight="bold", color=INK)
    ax.text(0.04, 0.30, "Fast .089", va="center", fontsize=32, fontweight="bold", color=INK)

    ax = axes[2]
    ax.set_title("Deployment", fontsize=32, fontweight="bold", color=INK)
    ax.text(0.05, 0.72, "27,765 trials", fontsize=32, color=INK)
    ax.text(0.05, 0.53, "Raw 91.9 → 89.9", fontsize=32, fontweight="bold", color=GRAY)
    ax.text(0.05, 0.30, "Reflection + Q", fontsize=32, color=INK)
    ax.text(0.05, 0.11, "Q 94.5 → 95.9", fontsize=32, fontweight="bold", color=PURE)

    ax = axes[3]
    ax.set_title("Critic ranking", fontsize=32, fontweight="bold", color=INK)
    ax.text(0.05, 0.80, "2,259 failures", fontsize=32, color=INK)
    ax.text(0.05, 0.62, "Precision .782", fontsize=32, fontweight="bold", color=PURE)
    ax.text(0.05, 0.44, "Recall .703", fontsize=32, fontweight="bold", color=PURE)
    ax.text(0.05, 0.26, "Balanced .623", fontsize=32, fontweight="bold", color="#8a5be8")
    ax.text(0.05, 0.08, "ROC AUC .655", fontsize=32, fontweight="bold", color="#8a5be8")

    fig.subplots_adjust(
        left=0.012,
        right=0.988,
        top=0.78,
        bottom=0.08,
        wspace=0.08,
    )
    fig.savefig(FIG / "55_poster_selected_pure_internal.png", dpi=220, facecolor="white")
    plt.close(fig)


def build_target_family_internal_diagnostics() -> None:
    """Matched five-seed actor and critic internals for the target-family study."""
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))["arms"]
    keys = [
        "p0_simba_onestep_utd1_100k",
        "p1_simba_fastsacn8_lambda1_utd1_100k",
        "p2_simba_sacn8_lambda1_utd1_100k",
    ]
    labels = ["SAC", "Fast", "N8"]
    colors = [GRAY, PURE, MIXED]
    saturation = [
        100
        * np.asarray(
            geometry[key]["metrics"][
                "deterministic_action_saturation_fraction_abs_ge_0p995"
            ]["seed_values"]
        )
        for key in keys
    ]
    sensitivity = [
        np.asarray(
            geometry[key]["metrics"]["mean_tanh_derivative"]["seed_values"]
        )
        for key in keys
    ]

    critic_direction = np.asarray([73.09, 74.84, 76.21])
    harmful_step = np.asarray([27.77, 26.64, 24.61])
    helpful_recognition = np.asarray([86.39, 54.47, 80.38])
    failure_boundary = np.asarray([91.50, 98.51, 98.92])
    outward_at_boundary = np.asarray([81.97, 94.64, 95.06])
    effective_fraction = np.asarray([28.17, 7.13, 6.94])

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.1), facecolor="white")
    x = np.arange(3)

    ax = axes[0, 0]
    for seed in range(5):
        ax.plot(
            x,
            [saturation[family][seed] for family in range(3)],
            color="#b8c7ce",
            linewidth=2.2,
            marker="o",
            markersize=6,
            zorder=1,
        )
    medians = [np.median(values) for values in saturation]
    ax.scatter(x, medians, s=170, color=colors, edgecolor="white", linewidth=2, zorder=3)
    for i, value in enumerate(medians):
        ax.text(i, value + 3.1, f"{value:.1f}", ha="center", fontsize=21, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Actions at |a| ≥ 1.99 (%)", fontsize=20, color=INK)
    ax.set_title("A  Actor at torque bound", fontsize=24, fontweight="bold", loc="left")
    style_axis(ax, "y")

    ax = axes[0, 1]
    for seed in range(5):
        ax.plot(
            x,
            [sensitivity[family][seed] for family in range(3)],
            color="#b8c7ce",
            linewidth=2.2,
            marker="o",
            markersize=6,
            zorder=1,
        )
    medians = [np.median(values) for values in sensitivity]
    ax.scatter(x, medians, s=170, color=colors, edgecolor="white", linewidth=2, zorder=3)
    for i, value in enumerate(medians):
        ax.text(i, value + 0.010, f"{value:.3f}", ha="center", fontsize=21, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.17)
    ax.set_ylabel(r"Mean $1-\tanh^2(u)$", fontsize=20, color=INK)
    ax.set_title("B  Actor sensitivity", fontsize=24, fontweight="bold", loc="left")
    style_axis(ax, "y")

    ax = axes[1, 0]
    width = 0.24
    ax.bar(x - width, critic_direction, width, color=PURE, label=r"$\partial Q/\partial a$ toward helpful action")
    ax.bar(x, helpful_recognition, width, color=MIXED, label="Q ranks helpful action higher")
    ax.bar(x + width, harmful_step, width, color=ORANGE, label="Projected critic step harmful")
    for offset, values in (
        (-width, critic_direction),
        (0, helpful_recognition),
        (width, harmful_step),
    ):
        for i, value in enumerate(values):
            ax.text(i + offset, value + 2.0, f"{value:.0f}", ha="center", fontsize=17, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Failure-state trials (%)", fontsize=20, color=INK)
    ax.set_title("C  Critic tests on failures", fontsize=24, fontweight="bold", loc="left")
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=15,
        handlelength=1.1,
        labelspacing=0.25,
    )
    style_axis(ax, "y")

    ax = axes[1, 1]
    ax.bar(x - width, failure_boundary, width, color=GRAY, label="Actor at torque bound")
    ax.bar(x, outward_at_boundary, width, color=PURE, label=r"$\partial Q/\partial a$ points outward")
    ax.bar(x + width, effective_fraction, width, color=ORANGE, label="Critic step retained")
    for offset, values in (
        (-width, failure_boundary),
        (0, outward_at_boundary),
        (width, effective_fraction),
    ):
        for i, value in enumerate(values):
            ax.text(i + offset, value + 2.0, f"{value:.0f}", ha="center", fontsize=17, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent", fontsize=20, color=INK)
    ax.set_title("D  Clipping the critic step", fontsize=24, fontweight="bold", loc="left")
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=15,
        handlelength=1.1,
        labelspacing=0.25,
    )
    style_axis(ax, "y")

    for ax in axes.flat:
        ax.tick_params(labelsize=18)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.10, hspace=0.42, wspace=0.28)
    fig.savefig(FIG / "57_poster_target_family_internals.png", dpi=240, facecolor="white")
    plt.close(fig)


def build_target_family_internal_diagnostics_wide() -> None:
    """Four poster-scale internal diagnostics in a single horizontal band."""
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))["arms"]
    keys = [
        "p0_simba_onestep_utd1_100k",
        "p1_simba_fastsacn8_lambda1_utd1_100k",
        "p2_simba_sacn8_lambda1_utd1_100k",
    ]
    labels = ["SAC", "Fast", "N8"]
    colors = [GRAY, PURE, MIXED]
    saturation = [
        100
        * np.asarray(
            geometry[key]["metrics"][
                "deterministic_action_saturation_fraction_abs_ge_0p995"
            ]["seed_values"]
        )
        for key in keys
    ]
    sensitivity = [
        np.asarray(
            geometry[key]["metrics"]["mean_tanh_derivative"]["seed_values"]
        )
        for key in keys
    ]
    critic_direction = np.asarray([73.09, 74.84, 76.21])
    harmful_step = np.asarray([27.77, 26.64, 24.61])
    helpful_recognition = np.asarray([86.39, 54.47, 80.38])
    failure_boundary = np.asarray([91.50, 98.51, 98.92])
    outward_at_boundary = np.asarray([81.97, 94.64, 95.06])
    effective_fraction = np.asarray([28.17, 7.13, 6.94])

    fig, axes = plt.subplots(1, 4, figsize=(19.2, 5.2), facecolor="white")
    x = np.arange(3)
    titles = [
        "A  Actor bound",
        "B  Actor slope",
        "C  Critic test",
        "D  Clip effect",
    ]

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
    ax.scatter(x, medians, s=120, color=colors, edgecolor="white", linewidth=1.8, zorder=3)
    for i, value in enumerate(medians):
        ax.text(i, value + 3.0, f"{value:.1f}", ha="center", fontsize=36, fontweight="bold")
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
    ax.scatter(x, medians, s=120, color=colors, edgecolor="white", linewidth=1.8, zorder=3)
    for i, value in enumerate(medians):
        ax.text(i, value + 0.009, f"{value:.3f}", ha="center", fontsize=36, fontweight="bold")
    ax.set_ylim(0, 0.17)
    ax.set_ylabel("")

    width = 0.24
    ax = axes[2]
    for offset, values, color in (
        (-width, critic_direction, PURE),
        (0, helpful_recognition, MIXED),
        (width, harmful_step, ORANGE),
    ):
        ax.bar(x + offset, values, width, color=color)
    ax.set_ylim(0, 110)
    ax.set_ylabel("")

    ax = axes[3]
    for offset, values, color in (
        (-width, failure_boundary, GRAY),
        (0, outward_at_boundary, PURE),
        (width, effective_fraction, ORANGE),
    ):
        ax.bar(x + offset, values, width, color=color)
    ax.set_ylim(0, 115)
    ax.set_ylabel("")

    for ax, title in zip(axes, titles):
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=36, fontweight="bold", loc="left")
        style_axis(ax, "y")
        ax.tick_params(labelsize=36)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.80, bottom=0.25, wspace=0.32)
    fig.savefig(FIG / "58_poster_target_family_internals_wide.png", dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    build_scorecard()
    build_selected_pure_internal_diagnostics()
    build_joint_gradient_compact()
    build_target_family_internal_diagnostics()
    build_target_family_internal_diagnostics_wide()


if __name__ == "__main__":
    main()
