from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "current_best_noncheating_20260723"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METHODS = {
    "hybrid": {
        "label": "RL + supervised winner",
        "short": "Mixed winner",
        "category": "RL + supervised",
        "color": "#0B6E99",
        "path": ROOT
        / "reports/systematic_100k_budget_best_20260722/ablation_no_rl_shift_qsearch/relative/relative_rollouts.csv",
    },
    "hybrid_uniform": {
        "label": "Uniform-start mixed control",
        "short": "Uniform mixed",
        "category": "RL + supervised",
        "color": "#59A7C6",
        "path": ROOT
        / "reports/systematic_100k_budget_best_20260722/ablation_uniform_dagger_qsearch/relative/relative_rollouts.csv",
    },
    "supervised": {
        "label": "Supervised actor deployment",
        "short": "Supervised actor",
        "category": "Supervised",
        "color": "#D89000",
        "path": ROOT
        / "reports/current_best_noncheating_20260723/pure_supervised_no_rl_shift_actor_relative/relative_rollouts.csv",
    },
    "pure_rl": {
        "label": "Pure RL winner",
        "short": "Pure RL winner",
        "category": "Pure RL",
        "color": "#7B2CBF",
        "path": ROOT
        / "reports/pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/relative_rollouts.csv",
    },
    "fastsacn": {
        "label": "FastSACN 50k + Q-search",
        "short": "FastSACN 50k",
        "category": "Pure RL",
        "color": "#C05BC7",
        "path": ROOT
        / "reports/pure_rl_plus1pp_20260719/authority_clean_fastsacn8_utd2_q41m005_unanimous_relative/relative_rollouts.csv",
    },
    "simba": {
        "label": "Plain SimbaV2 100k",
        "short": "SimbaV2 100k",
        "category": "Pure RL",
        "color": "#B089D3",
        "path": ROOT
        / "reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_rollouts.csv",
    },
    "dagger": {
        "label": "Clean DAgger 100k",
        "short": "Clean DAgger",
        "category": "Supervised",
        "color": "#5B6573",
        "path": ROOT
        / "reports/canonical_reference_dagger_100k_5seed_20260716/relative/relative_rollouts.csv",
    },
}

METRICS = {
    "near_best_known_return_eps": "Near reference",
    "task_success": "Task success",
    "beats_best_known_return": "Strictly beats reference",
}


def read_methods() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, spec in METHODS.items():
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        expected = {
            "actual_seed",
            "theta",
            "theta_degrees",
            "theta_dot",
            "return",
            "near_best_known_return_eps",
            "task_success",
            "beats_best_known_return",
            "signed_gap_to_best_known",
        }
        missing = expected.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if len(frame) != 12_505:
            raise ValueError(f"{path} has {len(frame)} rows, expected 12,505")
        frames[key] = frame
    return frames


def percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = PercentFormatter(xmax=100, decimals=0)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_scorecard(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, frame in frames.items():
        row = {
            "method_key": key,
            "method": METHODS[key]["label"],
            "category": METHODS[key]["category"],
            "trials": len(frame),
            "mean_return": frame["return"].mean(),
            "median_return": frame["return"].median(),
            "worst_return": frame["return"].min(),
        }
        for col, label in METRICS.items():
            row[f"{col}_count"] = int(frame[col].sum())
            row[f"{col}_percent"] = 100.0 * frame[col].mean()
        rows.append(row)
    score = pd.DataFrame(rows)
    score.to_csv(OUT / "main_scorecard.csv", index=False)

    order = ["hybrid", "hybrid_uniform", "supervised", "pure_rl", "fastsacn", "simba", "dagger"]
    plot = score.set_index("method_key").loc[order].reset_index()
    x = np.arange(len(plot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    for i, (col, label) in enumerate(METRICS.items()):
        vals = plot[f"{col}_percent"].to_numpy()
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=label,
            color=["#1B9E77", "#377EB8", "#E6550D"][i],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.45,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax.set_xticks(x, plot["method"].tolist(), rotation=18, ha="right")
    ax.set_ylabel("Success rate over 12,505 seed-state trials")
    ax.set_ylim(0, 106)
    percent_axis(ax)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("Authoritative five-seed scorecard")
    fig.subplots_adjust(top=0.82, bottom=0.24)
    finish(fig, "01_main_scorecard.png")
    return score


def make_seedwise(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["hybrid", "supervised", "pure_rl", "simba", "dagger"]
    rows = []
    for key in keys:
        grouped = frames[key].groupby("actual_seed", sort=True)
        for seed, group in grouped:
            for col, label in METRICS.items():
                rows.append(
                    {
                        "method_key": key,
                        "method": METHODS[key]["label"],
                        "seed": int(seed),
                        "metric": label,
                        "count": int(group[col].sum()),
                        "percent": 100.0 * group[col].mean(),
                    }
                )
    seedwise = pd.DataFrame(rows)
    seedwise.to_csv(OUT / "seedwise_rates.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True)
    for ax, metric in zip(axes, METRICS.values()):
        panel = seedwise[seedwise["metric"] == metric]
        for key in keys:
            p = panel[panel["method_key"] == key].sort_values("seed")
            ax.plot(
                p["seed"],
                p["percent"],
                marker="o",
                linewidth=2,
                markersize=5,
                label=METHODS[key]["short"],
                color=METHODS[key]["color"],
            )
        ax.set_title(metric)
        ax.set_xlabel("Training seed")
        ax.set_xticks(range(5))
        ax.grid(alpha=0.22)
        percent_axis(ax)
    axes[0].set_ylabel("Success rate on 2,501 states")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Seedwise robustness, not just pooled counts", y=1.12)
    fig.tight_layout()
    finish(fig, "02_seedwise_success.png")
    return seedwise


def add_angle_region(frame: pd.DataFrame) -> pd.Series:
    x = frame["theta_degrees"].abs()
    return pd.cut(
        x,
        bins=[-np.inf, 60, 120, 150, np.inf],
        right=False,
        labels=["|theta| < 60", "60 <= |theta| < 120", "120 <= |theta| < 150", "|theta| >= 150"],
    )


def make_regions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["hybrid", "supervised", "pure_rl", "simba", "dagger"]
    rows = []
    for key in keys:
        frame = frames[key].copy()
        frame["region"] = add_angle_region(frame)
        for region, group in frame.groupby("region", observed=False):
            for col, label in list(METRICS.items())[:2]:
                rows.append(
                    {
                        "method_key": key,
                        "method": METHODS[key]["label"],
                        "criterion": label,
                        "region": str(region),
                        "trials": len(group),
                        "failures": int((1 - group[col]).sum()),
                        "failure_rate_percent": 100.0 * (1.0 - group[col].mean()),
                    }
                )
    regions = pd.DataFrame(rows)
    regions.to_csv(OUT / "regional_failure_rates.csv", index=False)

    region_order = ["|theta| < 60", "60 <= |theta| < 120", "120 <= |theta| < 150", "|theta| >= 150"]
    x = np.arange(len(region_order))
    width = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharey=False)
    for ax, criterion in zip(axes, ["Near reference", "Task success"]):
        panel = regions[regions["criterion"] == criterion]
        for i, key in enumerate(keys):
            p = panel[panel["method_key"] == key].set_index("region").loc[region_order]
            ax.bar(
                x + (i - 2) * width,
                p["failure_rate_percent"],
                width=width,
                label=METHODS[key]["short"],
                color=METHODS[key]["color"],
            )
        ax.set_xticks(x, region_order, rotation=18, ha="right")
        ax.set_ylabel("Failure rate within angle region")
        ax.set_title(f"{criterion} failures")
        ax.grid(axis="y", alpha=0.2)
        percent_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Where the remaining failures live", y=1.12)
    fig.tight_layout()
    finish(fig, "03_failure_by_angle_region.png")
    return regions


def make_ecdf(frames: dict[str, pd.DataFrame]) -> None:
    keys = ["hybrid", "supervised", "pure_rl", "simba", "dagger"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.7))
    for ax, limits, title in [
        (axes[0], None, "Full signed-gap distribution"),
        (axes[1], (-12, 12), "Central region around the near-reference threshold"),
    ]:
        for key in keys:
            values = np.sort(frames[key]["signed_gap_to_best_known"].to_numpy())
            y = 100.0 * np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, linewidth=2, color=METHODS[key]["color"], label=METHODS[key]["short"])
        ax.axvline(-5, color="#222222", linestyle="--", linewidth=1.2, label="Near threshold" if ax is axes[0] else None)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=1.2, label="Strict threshold" if ax is axes[0] else None)
        if limits is not None:
            ax.set_xlim(*limits)
        ax.set_xlabel("Policy return minus best reference return")
        ax.set_ylabel("Cumulative percentage of trials")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        percent_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Return-gap distributions expose tail risk and threshold tradeoffs", y=1.12)
    fig.tight_layout()
    finish(fig, "04_return_gap_ecdf.png")


def make_paired_ablation_plot() -> pd.DataFrame:
    path = ROOT / "reports/systematic_100k_budget_best_20260722/paired_ablation_diagnostics.csv"
    paired = pd.read_csv(path)
    paired.to_csv(OUT / "paired_ablation_diagnostics.csv", index=False)
    labels = {
        "20k automatic-priority training stage": "20k priority DAgger stage",
        "local FastSACN Q-search at inference": "Local FastSACN Q-search",
        "automatic priority versus uniform DAgger starts": "Priority vs uniform starts",
        "tiny RL target shifts versus no shift": "Tiny RL label shifts",
        "pure-RL reflection fallback added to global Q-search": "Pure reflection fallback",
        "pure-RL global unanimous Q-search added to actor": "Pure global Q-search",
    }
    plot = paired.copy()
    plot["short"] = plot["comparison"].map(labels)
    plot = plot.iloc[::-1]
    y = np.arange(len(plot))
    height = 0.22
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for i, (col, label, color) in enumerate(
        [
            ("near_net", "Near net", "#1B9E77"),
            ("task_net", "Task net", "#377EB8"),
            ("strict_net", "Strict net", "#E6550D"),
        ]
    ):
        values = plot[col].to_numpy()
        bars = ax.barh(y + (i - 1) * height, values, height=height, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                value + (12 if value >= 0 else -12),
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):+d}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=9,
            )
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, plot["short"])
    ax.set_xlabel("Net fixed minus broken seed-state classifications")
    ax.set_title("Matched component ablations on identical grid cells")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=3, loc="upper right")
    finish(fig, "05_paired_component_ablation.png")
    return paired


def paired_delta(
    improved: pd.DataFrame, baseline: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = ["actual_seed", "theta", "theta_dot"]
    merged = improved.merge(
        baseline,
        on=keys,
        suffixes=("_improved", "_baseline"),
        validate="one_to_one",
    )
    merged["return_delta"] = merged["return_improved"] - merged["return_baseline"]
    stats = {
        "comparison": name,
        "trials": int(len(merged)),
        "mean": float(merged["return_delta"].mean()),
        "median": float(merged["return_delta"].median()),
        "p05": float(merged["return_delta"].quantile(0.05)),
        "p95": float(merged["return_delta"].quantile(0.95)),
        "improved_fraction": float((merged["return_delta"] > 0).mean()),
        "harmed_fraction": float((merged["return_delta"] < 0).mean()),
    }
    return merged[keys + ["return_delta"]], stats


def make_paired_return_deltas(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mixed, mixed_stats = paired_delta(
        frames["hybrid"],
        frames["supervised"],
        "Mixed local Q-search minus its actor",
    )
    pure, pure_stats = paired_delta(
        frames["pure_rl"],
        frames["simba"],
        "Pure reflection plus Q-search minus plain actor",
    )
    stats = pd.DataFrame([mixed_stats, pure_stats])
    stats.to_csv(OUT / "paired_return_delta_summary.csv", index=False)
    mixed.to_csv(OUT / "mixed_qsearch_paired_return_deltas.csv", index=False)
    pure.to_csv(OUT / "pure_deployment_paired_return_deltas.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))
    for ax, data, item, color in [
        (axes[0], mixed["return_delta"], mixed_stats, METHODS["hybrid"]["color"]),
        (axes[1], pure["return_delta"], pure_stats, METHODS["pure_rl"]["color"]),
    ]:
        clipped = data.clip(data.quantile(0.005), data.quantile(0.995))
        ax.hist(clipped, bins=90, color=color, alpha=0.86)
        ax.axvline(0, color="#222222", linewidth=1)
        ax.axvline(item["median"], color="#F0C808", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Paired return change")
        ax.set_ylabel("Seed-state trials")
        ax.set_title(
            f"{item['comparison']}\n"
            f"mean {item['mean']:+.3f}, median {item['median']:+.3g}, "
            f"improves {100*item['improved_fraction']:.1f}%"
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Inference operators have very different gain distributions", y=1.03)
    fig.tight_layout()
    finish(fig, "06_paired_return_delta_distributions.png")
    return stats


def make_tradeoff(score: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for _, row in score.iterrows():
        key = row["method_key"]
        x = row["beats_best_known_return_percent"]
        y = row["near_best_known_return_eps_percent"]
        size = 60 + 4.0 * (row["task_success_percent"] - 80)
        ax.scatter(
            x,
            y,
            s=size,
            color=METHODS[key]["color"],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        dx = 0.18
        dy = 0.12 if key not in {"hybrid_uniform", "supervised"} else -0.42
        ax.text(x + dx, y + dy, METHODS[key]["short"], fontsize=9.5)
    ax.set_xlabel("Strictly beats reference")
    ax.set_ylabel("Near reference")
    ax.set_xlim(4, 20)
    ax.set_ylim(82, 101)
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    ax.grid(alpha=0.22)
    ax.set_title("Near-reference reliability and strict wins are different objectives\nMarker size tracks task success")
    finish(fig, "07_reliability_vs_strict_tradeoff.png")


def make_consistency(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["hybrid", "supervised", "pure_rl", "simba", "dagger"]
    rows = []
    for key in keys:
        frame = frames[key]
        for col, label in list(METRICS.items())[:2]:
            cell = frame.groupby(["theta", "theta_dot"])[col].sum()
            rows.append(
                {
                    "method_key": key,
                    "method": METHODS[key]["label"],
                    "criterion": label,
                    "all_five_success_percent": 100.0 * (cell == 5).mean(),
                    "any_seed_success_percent": 100.0 * (cell >= 1).mean(),
                    "all_five_failure_cells": int((cell == 0).sum()),
                    "variable_cells": int(((cell > 0) & (cell < 5)).sum()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "cross_seed_cell_consistency.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7))
    x = np.arange(len(keys))
    width = 0.36
    for ax, criterion in zip(axes, ["Near reference", "Task success"]):
        panel = result[result["criterion"] == criterion].set_index("method_key").loc[keys]
        ax.bar(
            x - width / 2,
            panel["all_five_success_percent"],
            width=width,
            color="#176B87",
            label="All five seeds succeed",
        )
        ax.bar(
            x + width / 2,
            panel["any_seed_success_percent"],
            width=width,
            color="#A7D8E8",
            label="At least one seed succeeds",
        )
        ax.set_xticks(x, [METHODS[k]["short"] for k in keys], rotation=18, ha="right")
        ax.set_ylim(0, 103)
        ax.set_ylabel("Fraction of 2,501 grid cells")
        ax.set_title(criterion)
        ax.grid(axis="y", alpha=0.2)
        percent_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Across-seed coverage distinguishes uniform solutions from lucky seeds", y=1.11)
    fig.tight_layout()
    finish(fig, "08_cross_seed_cell_consistency.png")
    return result


def make_gap_decomposition(score: pd.DataFrame) -> pd.DataFrame:
    indexed = score.set_index("method_key")
    rows = []
    for col, label in METRICS.items():
        mixed = indexed.loc["hybrid", f"{col}_count"]
        pure = indexed.loc["pure_rl", f"{col}_count"]
        rows.append(
            {
                "metric": label,
                "mixed_count": int(mixed),
                "pure_count": int(pure),
                "mixed_minus_pure_count": int(mixed - pure),
                "mixed_minus_pure_percentage_points": 100.0 * (mixed - pure) / 12_505,
            }
        )
    mixed_return = indexed.loc["hybrid", "mean_return"]
    pure_return = indexed.loc["pure_rl", "mean_return"]
    gap = pd.DataFrame(rows)
    gap["mixed_minus_pure_mean_return"] = mixed_return - pure_return
    gap.to_csv(OUT / "mixed_vs_pure_gap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = ["#1B9E77", "#377EB8", "#E6550D"]
    bars = axes[0].bar(gap["metric"], gap["mixed_minus_pure_count"], color=colors)
    axes[0].axhline(0, color="#222222", linewidth=1)
    axes[0].set_ylabel("Mixed minus pure successful trials")
    axes[0].set_title("Classification gap")
    axes[0].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, gap["mixed_minus_pure_count"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + (22 if value >= 0 else -22),
            f"{value:+d}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    values = [indexed.loc["hybrid", "mean_return"], indexed.loc["pure_rl", "mean_return"]]
    bars = axes[1].bar(["Mixed winner", "Pure RL winner"], values, color=[METHODS["hybrid"]["color"], METHODS["pure_rl"]["color"]])
    axes[1].set_ylim(min(values) - 1.2, max(values) + 0.8)
    axes[1].set_ylabel("Mean return")
    axes[1].set_title(f"Mean-return gap: mixed {mixed_return - pure_return:+.3f}")
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.10, f"{value:.3f}", ha="center")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("The mixed method is more reliable, while pure RL wins strictly more often", y=1.03)
    fig.tight_layout()
    finish(fig, "09_mixed_vs_pure_gap.png")
    return gap


def make_pure_diagnostics() -> pd.DataFrame:
    rows = [
        {"diagnostic": "Critic rank correlation", "value": 0.6931, "unit": "Spearman rho", "direction": "higher is better"},
        {"diagnostic": "Pairwise action ordering", "value": 0.8437, "unit": "fraction", "direction": "higher is better"},
        {"diagnostic": "Q-search harms actor", "value": 0.1899, "unit": "fraction", "direction": "lower is better"},
        {"diagnostic": "Actor near torque bound", "value": 0.9313, "unit": "fraction", "direction": "lower is better"},
        {"diagnostic": "Critic disagreement / local signal", "value": 2.4264, "unit": "ratio", "direction": "lower is better"},
        {"diagnostic": "Critic-loss / return-loss gradient cosine", "value": -0.8157, "unit": "cosine", "direction": "positive agreement is better"},
        {"diagnostic": "Actor reflection error", "value": 0.2432, "unit": "action units", "direction": "lower is better"},
        {"diagnostic": "Critic reflection error / local Q range", "value": 6.91, "unit": "ratio", "direction": "lower is better"},
        {"diagnostic": "Dormant-unit evidence", "value": 0.0, "unit": "detected", "direction": "zero means ruled out"},
    ]
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT / "pure_rl_diagnostic_key_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
    panels = [
        ("Critic ordering", 69.31, 0, 100, "Spearman x 100"),
        ("Pairwise ranking", 84.37, 0, 100, "percent correct"),
        ("Q-search harm rate", 18.99, 0, 50, "percent of cells"),
        ("Actor saturation", 93.13, 0, 100, "percent near bound"),
        ("Disagreement / signal", 2.4264, 0, 4, "ratio"),
        ("Gradient cosine", -0.8157, -1, 1, "critic loss vs return loss"),
    ]
    for ax, (title, value, lo, hi, subtitle) in zip(axes.flat, panels):
        color = "#2A9D8F" if title in {"Critic ordering", "Pairwise ranking"} else "#C8553D"
        if title == "Gradient cosine":
            ax.barh([0], [value], left=0, color=color, height=0.45)
            ax.axvline(0, color="#222222", linewidth=1)
        else:
            ax.barh([0], [value - lo], left=lo, color=color, height=0.45)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_title(title)
        ax.text(value, 0, f" {value:.3g}", va="center", ha="left" if value >= 0 else "right", fontsize=11)
        ax.set_xlabel(subtitle)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Pure-RL diagnostics: useful critics, saturated actors, and misaligned local geometry", y=1.02)
    fig.tight_layout()
    finish(fig, "10_pure_rl_diagnostic_flags.png")
    return diag


def make_joint_loss_screen() -> pd.DataFrame:
    arm_specs = [
        ("h0_bc_only_fast8", "BC only", "#5B6573"),
        ("h3k_detmean_gate_m005_bcfinal4_pcgrad", "Joint gated + PCGrad", "#3E8E7E"),
        ("h14_detmean_gate_m005_pcgrad_bcfinal4_all_lambda1", "Joint all-horizon lambda 1", "#0B6E99"),
        ("h15_bc_global41_unanimous_logprob_distill_delayed6k", "Unbounded Q log-prob distill", "#D1495B"),
        ("h19_bc_global41_unanimous_logprob_distill_delayed6k_lr2e6", "Same distill, lower LR", "#9B2226"),
        ("h20_qfiltered_replay_bc_only_m005_delayed6k", "Q-filtered replay BC", "#D89000"),
    ]
    rows = []
    for arm, label, color in arm_specs:
        path = ROOT / "runs/systematic_joint_loss_screen_20260722" / arm / "seed0/eval_episodes.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        for step, group in frame.groupby("step", sort=True):
            rows.append(
                {
                    "arm": arm,
                    "label": label,
                    "color": color,
                    "step": int(step),
                    "episodes": len(group),
                    "mean_return": group["return"].mean(),
                    "worst_return": group["return"].min(),
                    "near_upright_fraction": group["near_upright_fraction"].mean(),
                    "task_success_count": int(group["task_success"].sum()),
                    "task_success_rate": group["task_success"].mean(),
                }
            )
    screen = pd.DataFrame(rows)
    screen.to_csv(OUT / "joint_loss_short_screen_curves.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2), sharex=True)
    fields = [
        ("mean_return", "Mean return", None),
        ("worst_return", "Worst return", None),
        ("near_upright_fraction", "Near-upright fraction", (0.65, 0.9)),
        ("task_success_rate", "Task success on ten fixed episodes", (0.3, 0.9)),
    ]
    for ax, (field, title, ylim) in zip(axes.flat, fields):
        for arm, label, color in arm_specs:
            p = screen[screen["arm"] == arm].sort_values("step")
            ax.plot(p["step"], p[field], marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Additional environment steps")
        ax.grid(alpha=0.22)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if field == "task_success_rate":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Small seed-0 joint-loss screen: useful mechanism evidence, not an authority ranking", y=1.07)
    fig.tight_layout()
    finish(fig, "11_joint_loss_short_screen.png")
    return screen


def main() -> None:
    frames = read_methods()
    score = make_scorecard(frames)
    seedwise = make_seedwise(frames)
    regions = make_regions(frames)
    make_ecdf(frames)
    paired = make_paired_ablation_plot()
    delta_stats = make_paired_return_deltas(frames)
    make_tradeoff(score)
    consistency = make_consistency(frames)
    gap = make_gap_decomposition(score)
    diagnostics = make_pure_diagnostics()
    joint_screen = make_joint_loss_screen()

    manifest = {
        "date": "2026-07-23",
        "grid": {"angles": 61, "velocities": 41, "states_per_seed": 2501, "seeds": 5, "trials": 12505},
        "source_rollouts": {key: str(spec["path"].relative_to(ROOT)) for key, spec in METHODS.items()},
        "outputs": {
            "scorecard_rows": len(score),
            "seedwise_rows": len(seedwise),
            "region_rows": len(regions),
            "paired_ablation_rows": len(paired),
            "paired_delta_rows": len(delta_stats),
            "consistency_rows": len(consistency),
            "gap_rows": len(gap),
            "diagnostic_rows": len(diagnostics),
            "joint_screen_rows": len(joint_screen),
            "figures": sorted(path.name for path in FIG.glob("*.png")),
        },
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
