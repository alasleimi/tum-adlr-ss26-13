from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "chasing_nines_20260723"
FIG = OUT / "figures"

SOURCES = {
    "mixed": {
        "label": "RL + supervised winner",
        "category": "RL + supervised",
        "color": "#0F766E",
        "path": ROOT
        / "reports/systematic_100k_budget_best_20260722/ablation_no_rl_shift_qsearch/relative/relative_rollouts.csv",
    },
    "mixed_uniform": {
        "label": "Uniform-start mixed control",
        "category": "RL + supervised",
        "color": "#5AA89C",
        "path": ROOT
        / "reports/systematic_100k_budget_best_20260722/ablation_uniform_dagger_qsearch/relative/relative_rollouts.csv",
    },
    "supervised": {
        "label": "Supervised actor",
        "category": "Supervised",
        "color": "#7C3AED",
        "path": ROOT
        / "reports/current_best_noncheating_20260723/pure_supervised_no_rl_shift_actor_relative/relative_rollouts.csv",
    },
    "pure": {
        "label": "Pure-RL winner",
        "category": "Pure RL",
        "color": "#2563EB",
        "path": ROOT
        / "reports/pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/relative_rollouts.csv",
    },
    "fastsacn": {
        "label": "FastSACN8 50k",
        "category": "Pure RL",
        "color": "#F59E0B",
        "path": ROOT
        / "reports/pure_rl_plus1pp_20260719/authority_clean_fastsacn8_utd2_q41m005_unanimous_relative/relative_rollouts.csv",
    },
    "simba": {
        "label": "Plain SimbaV2 100k",
        "category": "Pure RL",
        "color": "#60A5FA",
        "path": ROOT
        / "reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_rollouts.csv",
    },
    "dagger": {
        "label": "Clean DAgger 100k",
        "category": "Supervised",
        "color": "#A78BFA",
        "path": ROOT
        / "reports/canonical_reference_dagger_100k_5seed_20260716/relative/relative_rollouts.csv",
    },
}

METRICS = {
    "near_best_known_return_eps": "Near reference",
    "task_success": "Task success",
    "beats_best_known_return": "Strictly beats reference",
}


def load_and_validate() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    required = {
        "actual_seed",
        "theta",
        "theta_degrees",
        "theta_dot",
        "return",
        *METRICS,
    }
    for key, spec in SOURCES.items():
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{key}: missing columns {missing}")
        if len(frame) != 12_505:
            raise ValueError(f"{key}: expected 12,505 rows, found {len(frame)}")
        if frame["actual_seed"].nunique() != 5:
            raise ValueError(f"{key}: expected five actor seeds")
        cells = frame[["theta", "theta_dot"]].drop_duplicates()
        if len(cells) != 2_501:
            raise ValueError(f"{key}: expected 2,501 grid cells, found {len(cells)}")
        counts = frame.groupby(["theta", "theta_dot"]).size()
        if not (counts == 5).all():
            raise ValueError(f"{key}: every cell must have one row from each actor seed")
        for metric in METRICS:
            values = set(frame[metric].dropna().astype(float).unique())
            if not values.issubset({0.0, 1.0}):
                raise ValueError(f"{key}: {metric} is not binary")
        frames[key] = frame
    return frames


def build_scorecard(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, frame in frames.items():
        spec = SOURCES[key]
        row = {
            "method_key": key,
            "method": spec["label"],
            "category": spec["category"],
            "trials": len(frame),
            "mean_return": frame["return"].mean(),
        }
        for metric, label in METRICS.items():
            count = int(frame[metric].sum())
            row[f"{metric}_count"] = count
            row[f"{metric}_rate"] = count / len(frame)
            row[f"{metric}_label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def build_consistency(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key in ["mixed", "supervised", "pure", "simba", "dagger"]:
        frame = frames[key]
        for metric in ["near_best_known_return_eps", "task_success"]:
            by_cell = (
                frame.groupby(["theta_degrees", "theta_dot"], sort=True)[metric]
                .agg(["sum", "count"])
                .reset_index()
            )
            all_five = int((by_cell["sum"] == 5).sum())
            any_seed = int((by_cell["sum"] > 0).sum())
            all_fail = int((by_cell["sum"] == 0).sum())
            variable = int(((by_cell["sum"] > 0) & (by_cell["sum"] < 5)).sum())
            rows.append(
                {
                    "method_key": key,
                    "method": SOURCES[key]["label"],
                    "criterion": METRICS[metric],
                    "all_five_success_cells": all_five,
                    "all_five_success_rate": all_five / 2_501,
                    "any_seed_success_cells": any_seed,
                    "any_seed_success_rate": any_seed / 2_501,
                    "all_five_failure_cells": all_fail,
                    "variable_cells": variable,
                }
            )
    return pd.DataFrame(rows)


def plot_scorecard(scorecard: pd.DataFrame) -> None:
    order = ["mixed", "mixed_uniform", "supervised", "pure", "fastsacn", "simba", "dagger"]
    table = scorecard.set_index("method_key").loc[order]
    y = np.arange(len(table))
    height = 0.22
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    metric_styles = [
        ("near_best_known_return_eps_rate", "Near reference", "#0F766E"),
        ("task_success_rate", "Task success", "#2563EB"),
        ("beats_best_known_return_rate", "Strictly beats reference", "#EA580C"),
    ]
    for offset, (column, label, color) in zip([-height, 0, height], metric_styles):
        values = table[column].to_numpy()
        bars = ax.barh(y + offset, values, height=height * 0.84, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{100 * value:.1f}%",
                va="center",
                fontsize=10,
                fontweight="semibold",
            )
    ax.set_yticks(y, table["method"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of 12,505 seed-state trials")
    ax.set_title("Reliability improves in layers, while strict wins follow a different ordering", loc="left", pad=18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        frameon=False,
    )
    ax.grid(axis="x", alpha=0.18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "01_verified_scorecard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _cell_matrix(frame: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot_table(
        index="theta_dot",
        columns="theta_degrees",
        values=metric,
        aggfunc="mean",
    ).sort_index(ascending=True)
    return pivot.to_numpy(), pivot.columns.to_numpy(), pivot.index.to_numpy()


def plot_reliability_maps(frames: dict[str, pd.DataFrame]) -> None:
    colors = ["#7F1D1D", "#D94801", "#F59E0B", "#FDE68A", "#14B8A6", "#064E3B"]
    cmap = ListedColormap(colors, name="five_seed_reliability")
    boundaries = np.arange(-0.1, 1.21, 0.2)
    norm = BoundaryNorm(boundaries, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), sharex=True, sharey=True)
    for ax, key, title in zip(
        axes,
        ["pure", "mixed"],
        ["Pure RL: seed-dependent failure tail", "RL + supervision: nine sparse failures"],
    ):
        matrix, theta, velocity = _cell_matrix(frames[key], "near_best_known_return_eps")
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[theta.min(), theta.max(), velocity.min(), velocity.max()],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        failures = int((1 - frames[key]["near_best_known_return_eps"]).sum())
        ax.set_title(f"{title}\n{failures} failed seed-state trials")
        ax.set_xlabel(r"Initial angle $\theta_0$ (degrees)")
        ax.grid(False)
    axes[0].set_ylabel(r"Initial angular velocity $\dot{\theta}_0$")
    cbar = fig.colorbar(
        image,
        ax=axes,
        boundaries=boundaries,
        ticks=np.linspace(0, 1, 6),
        fraction=0.03,
        pad=0.045,
    )
    cbar.set_label("Fraction of five actor seeds that succeed")
    fig.suptitle("The mixed method closes the difficult-state reliability tail", x=0.08, ha="left", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.15, top=0.8, wspace=0.11)
    fig.savefig(FIG / "02_reliability_maps.png", dpi=240, bbox_inches="tight")
    fig._suptitle.set_visible(False)
    for ax in axes:
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=11)
    axes[0].yaxis.label.set_fontsize(12)
    cbar.ax.yaxis.label.set_fontsize(12)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.15, top=0.92, wspace=0.11)
    fig.savefig(FIG / "02_reliability_maps_poster.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_consistency(consistency: pd.DataFrame) -> None:
    keys = ["mixed", "supervised", "pure", "simba", "dagger"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    for ax, criterion in zip(axes, ["Near reference", "Task success"]):
        data = consistency[consistency["criterion"] == criterion].set_index("method_key").loc[keys]
        y = np.arange(len(keys))
        all_rates = data["all_five_success_rate"].to_numpy()
        any_rates = data["any_seed_success_rate"].to_numpy()
        for index, (all_rate, any_rate) in enumerate(zip(all_rates, any_rates)):
            ax.plot([all_rate, any_rate], [index, index], color="#CBD5E1", linewidth=6, solid_capstyle="round")
            ax.scatter(all_rate, index, s=80, color="#0F766E", label="All five succeed" if index == 0 else None)
            ax.scatter(any_rate, index, s=80, color="#F59E0B", label="At least one succeeds" if index == 0 else None)
        ax.set_xlim(0.7, 1.01)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(criterion)
        ax.set_xlabel("Fraction of 2,501 grid cells")
        ax.grid(axis="x", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(np.arange(len(keys)), [SOURCES[key]["label"] for key in keys])
    axes[0].invert_yaxis()
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("A policy can solve a state in one seed without solving it reliably", x=0.08, ha="left", fontsize=16)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(FIG / "03_seed_consistency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # The poster card is nearly square. Stacking the two criteria uses that
    # area and keeps every label readable from an A0 viewing distance.
    poster_fig, poster_axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    for ax, criterion in zip(poster_axes, ["Near reference", "Task success"]):
        data = consistency[consistency["criterion"] == criterion].set_index("method_key").loc[keys]
        y = np.arange(len(keys))
        all_rates = data["all_five_success_rate"].to_numpy()
        any_rates = data["any_seed_success_rate"].to_numpy()
        for index, (all_rate, any_rate) in enumerate(zip(all_rates, any_rates)):
            ax.plot(
                [all_rate, any_rate],
                [index, index],
                color="#CBD5E1",
                linewidth=6,
                solid_capstyle="round",
            )
            ax.scatter(
                all_rate,
                index,
                s=82,
                color="#0F766E",
                label="All five succeed" if index == 0 else None,
            )
            ax.scatter(
                any_rate,
                index,
                s=82,
                color="#F59E0B",
                label="At least one succeeds" if index == 0 else None,
            )
        ax.set_xlim(0.7, 1.01)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_yticks(y, [SOURCES[key]["label"] for key in keys])
        ax.invert_yaxis()
        ax.set_title(criterion, loc="left", fontsize=13, fontweight="semibold")
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    poster_axes[-1].set_xlabel("Fraction of 2,501 grid cells", fontsize=11)
    poster_axes[-1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    poster_fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.0)
    poster_fig.savefig(FIG / "03_seed_consistency_poster.png", dpi=260, bbox_inches="tight")
    plt.close(poster_fig)


def selected_local_qsearch_counts(
    frames: dict[str, pd.DataFrame],
) -> dict[str, int | float | str]:
    keys = ["actual_seed", "theta", "theta_dot"]
    baseline = frames["supervised"][keys + ["return", *METRICS]].copy()
    corrected = frames["mixed"][keys + ["return", *METRICS]].copy()
    paired = baseline.merge(
        corrected,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_corrected"),
    )
    if len(paired) != 12505:
        raise ValueError(f"Expected 12,505 selected local-Q-search pairs, got {len(paired)}")
    result: dict[str, int | float | str] = {
        "comparison": "local FastSACN Q-search at inference",
        "trials": len(paired),
        "return_delta_mean": float(
            (paired["return_corrected"] - paired["return_baseline"]).mean()
        ),
    }
    prefixes = {
        "near_best_known_return_eps": "near",
        "task_success": "task",
        "beats_best_known_return": "strict",
    }
    for metric, prefix in prefixes.items():
        base = paired[f"{metric}_baseline"].astype(bool)
        new = paired[f"{metric}_corrected"].astype(bool)
        fixed = int((~base & new).sum())
        broken = int((base & ~new).sum())
        result[f"{prefix}_fixed"] = fixed
        result[f"{prefix}_broken"] = broken
        result[f"{prefix}_net"] = fixed - broken
    return result


def plot_ablation(frames: dict[str, pd.DataFrame]) -> dict[str, int | float | str]:
    source = ROOT / "reports/current_best_noncheating_20260723/paired_ablation_diagnostics.csv"
    data = pd.read_csv(source)
    local_counts = selected_local_qsearch_counts(frames)
    local_mask = data["comparison"].eq(local_counts["comparison"])
    if int(local_mask.sum()) != 1:
        raise ValueError("Expected exactly one historical local-Q-search ablation row.")
    for column, value in local_counts.items():
        if column in data.columns:
            data.loc[local_mask, column] = value
    wanted = {
        "local FastSACN Q-search at inference": "Mixed: local Q-search",
        "automatic priority versus uniform DAgger starts": "Mixed: priority starts",
        "tiny RL target shifts versus no shift": "Mixed: tiny label shift",
        "pure-RL reflection fallback added to global Q-search": "Pure RL: reflection",
        "pure-RL global unanimous Q-search added to actor": "Pure RL: Q-search",
    }
    data = data[data["comparison"].isin(wanted)].copy()
    data["label"] = data["comparison"].map(wanted)
    order = [
        "Pure RL: Q-search",
        "Pure RL: reflection",
        "Mixed: priority starts",
        "Mixed: local Q-search",
        "Mixed: tiny label shift",
    ]
    data = data.set_index("label").loc[order].reset_index()
    y = np.arange(len(data))
    height = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for offset, column, label, color in [
        (-height, "near_net", "Near reference", "#0F766E"),
        (0, "task_net", "Task success", "#2563EB"),
        (height, "strict_net", "Strictly beats reference", "#EA580C"),
    ]:
        values = data[column].to_numpy()
        bars = ax.barh(y + offset, values, height=height * 0.82, color=color, label=label)
        for bar, value in zip(bars, values):
            horizontal = 12 if value >= 0 else -12
            ax.annotate(
                f"{value:+d}",
                (value, bar.get_y() + bar.get_height() / 2),
                xytext=(horizontal, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=10,
                fontweight="semibold",
            )
    ax.axvline(0, color="#0F172A", linewidth=1)
    ax.set_yticks(y, data["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Net fixed minus broken classifications from the paired baseline")
    ax.set_title("Matched component additions show different metric trade-offs", loc="left", pad=14)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "04_component_ablation.png", dpi=220, bbox_inches="tight")
    ax.set_title("")
    ax.xaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=11)
    legend = ax.get_legend()
    if legend is not None:
        for text_item in legend.get_texts():
            text_item.set_fontsize(11)
    fig.tight_layout()
    fig.savefig(FIG / "04_component_ablation_poster.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    report_wanted = {
        "pure-RL global unanimous Q-search added to actor": "Pure RL: Q-search",
        "pure-RL reflection fallback added to global Q-search": "Pure RL: reflection",
        "20k automatic-priority training stage": "Mixed: learner follow-up",
        "automatic priority versus uniform DAgger starts": "Mixed: priority starts",
        "local FastSACN Q-search at inference": "Mixed: local Q-search",
        "tiny RL target shifts versus no shift": "Mixed: tiny label shift",
    }
    report_data = pd.read_csv(source)
    report_local_mask = report_data["comparison"].eq(local_counts["comparison"])
    for column, value in local_counts.items():
        if column in report_data.columns:
            report_data.loc[report_local_mask, column] = value
    report_data = report_data[report_data["comparison"].isin(report_wanted)].copy()
    report_data["label"] = report_data["comparison"].map(report_wanted)
    report_order = list(report_wanted.values())
    report_data = report_data.set_index("label").loc[report_order].reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.3), sharey=True)
    panels = [
        ("near_net", "Near reference", "#0F766E"),
        ("task_net", "Task success", "#2563EB"),
        ("strict_net", "Strictly beats reference", "#EA580C"),
    ]
    y = np.arange(len(report_data))
    for ax, (column, title, color) in zip(axes, panels):
        values = report_data[column].to_numpy()
        bars = ax.barh(y, values, color=color, height=0.58)
        ax.axvline(0, color="#0F172A", linewidth=0.9)
        for bar, value in zip(bars, values):
            offset = 7
            ax.annotate(
                f"{value:+d}",
                (value, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="semibold",
            )
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.grid(axis="x", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(y, report_data["label"].tolist())
    axes[0].invert_yaxis()
    axes[-1].set_xlabel("Net corrected seed-state classifications")
    fig.suptitle(
        "What changes reliability, task success, and strict wins?",
        x=0.02,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "04_component_ablation_report.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return local_counts


def plot_tradeoff(scorecard: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for _, row in scorecard.iterrows():
        key = row["method_key"]
        ax.scatter(
            row["near_best_known_return_eps_rate"],
            row["beats_best_known_return_rate"],
            s=150,
            color=SOURCES[key]["color"],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            SOURCES[key]["label"],
            (row["near_best_known_return_eps_rate"], row["beats_best_known_return_rate"]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=12,
        )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.82, 1.008)
    ax.set_ylim(0.04, 0.2)
    ax.set_xlabel("Near-reference reliability", fontsize=12)
    ax.set_ylabel("Strictly beats reference", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_title("Reliability and strict improvement are different operating points", loc="left", pad=14)
    ax.grid(alpha=0.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "05_reliability_strict_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tolerance_curve(frames: dict[str, pd.DataFrame]) -> None:
    epsilons = np.linspace(0.0, 20.0, 81)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for key, color, label in [
        ("mixed", "#0F766E", "RL + supervised"),
        ("pure", "#2563EB", "Pure RL"),
    ]:
        frame = frames[key]
        seed_curves = []
        for _seed, seed_frame in frame.groupby("actual_seed", sort=True):
            gaps = seed_frame["signed_gap_to_best_known"].to_numpy(dtype=float)
            curve = np.asarray([(gaps <= epsilon).mean() for epsilon in epsilons])
            seed_curves.append(curve)
            ax.plot(epsilons, curve, color=color, alpha=0.20, linewidth=1.2)
        mean_curve = np.mean(np.stack(seed_curves, axis=0), axis=0)
        ax.plot(epsilons, mean_curve, color=color, linewidth=3.2, label=label)
        at_five = float(mean_curve[np.argmin(np.abs(epsilons - 5.0))])
        ax.scatter([5.0], [at_five], color=color, s=45, zorder=5)
        ax.annotate(
            f"{at_five:.3%}",
            (5.0, at_five),
            xytext=(12, -34 if key == "mixed" else 14),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="semibold",
        )
    ax.axvline(5.0, color="#64748B", linestyle="--", linewidth=1.2)
    ax.text(5.25, 0.13, "reported ε = 5", color="#64748B", fontsize=9, rotation=90, va="bottom")
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.1, 1.01)
    ax.set_xlabel("Tolerance ε (raw return units)", fontsize=11)
    ax.set_ylabel(r"Fraction with $R_\pi \geq R^*-\epsilon$", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "06_tolerance_curve_poster.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_verified_metrics(
    scorecard: pd.DataFrame,
    consistency: pd.DataFrame,
    local_qsearch: dict[str, int | float | str],
) -> None:
    mixed = scorecard.set_index("method_key").loc["mixed"]
    pure = scorecard.set_index("method_key").loc["pure"]
    pure_near = consistency[
        (consistency["method_key"] == "pure") & (consistency["criterion"] == "Near reference")
    ].iloc[0]
    mixed_near = consistency[
        (consistency["method_key"] == "mixed") & (consistency["criterion"] == "Near reference")
    ].iloc[0]
    payload = {
        "protocol": {
            "actor_seeds": 5,
            "cells_per_seed": 2501,
            "trials": 12505,
        },
        "mixed": {
            "near_count": int(mixed["near_best_known_return_eps_count"]),
            "near_rate": float(mixed["near_best_known_return_eps_rate"]),
            "task_count": int(mixed["task_success_count"]),
            "task_rate": float(mixed["task_success_rate"]),
            "strict_count": int(mixed["beats_best_known_return_count"]),
            "strict_rate": float(mixed["beats_best_known_return_rate"]),
            "variable_near_cells": int(mixed_near["variable_cells"]),
            "all_five_failure_near_cells": int(mixed_near["all_five_failure_cells"]),
        },
        "pure_rl": {
            "near_count": int(pure["near_best_known_return_eps_count"]),
            "near_rate": float(pure["near_best_known_return_eps_rate"]),
            "task_count": int(pure["task_success_count"]),
            "task_rate": float(pure["task_success_rate"]),
            "strict_count": int(pure["beats_best_known_return_count"]),
            "strict_rate": float(pure["beats_best_known_return_rate"]),
            "variable_near_cells": int(pure_near["variable_cells"]),
            "all_five_failure_near_cells": int(pure_near["all_five_failure_cells"]),
        },
        "pure_minus_mixed": {
            "near_count": int(
                pure["near_best_known_return_eps_count"] - mixed["near_best_known_return_eps_count"]
            ),
            "task_count": int(pure["task_success_count"] - mixed["task_success_count"]),
            "strict_count": int(
                pure["beats_best_known_return_count"] - mixed["beats_best_known_return_count"]
            ),
        },
        "selected_mixed_local_qsearch_ablation": local_qsearch,
        "source_rollouts": {key: str(spec["path"].relative_to(ROOT)) for key, spec in SOURCES.items()},
    }
    (OUT / "verified_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    frames = load_and_validate()
    scorecard = build_scorecard(frames)
    consistency = build_consistency(frames)
    scorecard.to_csv(OUT / "verified_scorecard.csv", index=False)
    consistency.to_csv(OUT / "verified_cell_consistency.csv", index=False)
    plot_scorecard(scorecard)
    plot_reliability_maps(frames)
    plot_consistency(consistency)
    local_qsearch = plot_ablation(frames)
    plot_tradeoff(scorecard)
    plot_tolerance_curve(frames)
    write_verified_metrics(scorecard, consistency, local_qsearch)
    print(
        json.dumps(
            {
                "verified_methods": len(scorecard),
                "verified_trials_per_method": 12505,
                "figures": sorted(path.name for path in FIG.glob("*.png")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
