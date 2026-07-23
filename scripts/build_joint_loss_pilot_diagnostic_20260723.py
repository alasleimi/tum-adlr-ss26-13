from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    PROJECT_ROOT
    / "runs"
    / "systematic_joint_staging_vs_mixing_20260722"
    / "sm2_joint_bc_plus_sac_same_update_after6k"
    / "seed0"
)
DEFAULT_OUT = PROJECT_ROOT / "deliverables" / "chasing_nines_20260723"
EVALUATION_ROOT = (
    PROJECT_ROOT / "reports" / "joint_staging_vs_mixing_pilot_20260723"
)


def evaluation_summary(path: Path) -> dict[str, float | int]:
    rows = pd.read_csv(path)
    required = {
        "return",
        "near_best_known_return_eps",
        "task_success",
        "beats_best_known_return",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return {
        "trials": int(len(rows)),
        "near_reference_successes": int(rows["near_best_known_return_eps"].sum()),
        "near_reference_rate": float(rows["near_best_known_return_eps"].mean()),
        "task_successes": int(rows["task_success"].sum()),
        "task_success_rate": float(rows["task_success"].mean()),
        "strict_wins": int(rows["beats_best_known_return"].sum()),
        "strict_win_rate": float(rows["beats_best_known_return"].mean()),
        "mean_return": float(rows["return"].mean()),
    }


def metric_series(metrics: pd.DataFrame, name: str) -> pd.Series:
    rows = metrics.loc[metrics["name"].eq(name), ["step", "value"]].copy()
    rows["step"] = pd.to_numeric(rows["step"], errors="raise").astype(int)
    rows["value"] = pd.to_numeric(rows["value"], errors="raise")
    rows = rows.drop_duplicates("step", keep="last").sort_values("step")
    return rows.set_index("step")["value"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the seed-0 simultaneous BC and SAC gradient diagnostic."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing pilot artifacts under {run_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    total_steps = int(config["sac"]["total_steps"])
    metrics = pd.read_csv(metrics_path)

    bc = metric_series(metrics, "actor_weighted_bc_gradient_norm_mean")
    sac = metric_series(metrics, "actor_weighted_sac_gradient_norm_mean")
    cosine = metric_series(metrics, "actor_sac_bc_gradient_cosine_mean")
    cosine_min = metric_series(metrics, "actor_sac_bc_gradient_cosine_min")
    cosine_max = metric_series(metrics, "actor_sac_bc_gradient_cosine_max")

    common = bc.index.intersection(sac.index).intersection(cosine.index)
    common = common[common >= int(config["sac"]["sac_actor_loss_start_step"])]
    if len(common) < 2:
        raise RuntimeError("The pilot has too few joint-loss diagnostic checkpoints.")

    bc = bc.loc[common]
    sac = sac.loc[common]
    cosine = cosine.loc[common]
    cosine_min = cosine_min.reindex(common)
    cosine_max = cosine_max.reindex(common)
    ratio = bc / sac

    final_checkpoint = run_dir / "checkpoints" / "final.pt"
    max_logged_step = int(pd.to_numeric(metrics["step"], errors="coerce").max())
    completed = final_checkpoint.is_file() and max_logged_step >= total_steps

    dormancy_rows = metrics.loc[
        metrics["name"].str.endswith("_dormant_fraction", na=False),
        ["step", "name", "value"],
    ].copy()
    dormancy_rows["value"] = pd.to_numeric(dormancy_rows["value"], errors="raise")

    control_eval_path = (
        EVALUATION_ROOT
        / "sm0_pure_sac_after6k_seed0"
        / "relative"
        / "relative_rollouts.csv"
    )
    joint_eval_path = (
        EVALUATION_ROOT
        / "sm2_joint_bc_plus_sac_same_update_after6k_seed0"
        / "relative"
        / "relative_rollouts.csv"
    )
    evaluation = None
    if control_eval_path.is_file() and joint_eval_path.is_file():
        control_eval = evaluation_summary(control_eval_path)
        joint_eval = evaluation_summary(joint_eval_path)
        assert control_eval["trials"] == joint_eval["trials"] == 2501
        evaluation = {
            "protocol": "one seed on the fixed 61 by 41 grid; actor action only",
            "reward_only_control": control_eval,
            "simultaneous_bc_sac": joint_eval,
            "joint_minus_control": {
                key: joint_eval[key] - control_eval[key]
                for key in (
                    "near_reference_successes",
                    "near_reference_rate",
                    "task_successes",
                    "task_success_rate",
                    "strict_wins",
                    "strict_win_rate",
                    "mean_return",
                )
            },
            "limitation": (
                "This seed-0 comparison does not include the BC-only or staged "
                "controls and does not estimate seed variability."
            ),
        }

    summary = {
        "scope": "seed-0 25k diagnostic pilot; not a five-seed performance result",
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "configured_total_steps": total_steps,
        "max_logged_step": max_logged_step,
        "completed": completed,
        "joint_diagnostic_steps": [int(step) for step in common],
        "weighted_bc_to_sac_gradient_norm_ratio": {
            "median": float(np.median(ratio)),
            "minimum": float(np.min(ratio)),
            "maximum": float(np.max(ratio)),
            "final_logged": float(ratio.iloc[-1]),
        },
        "bc_sac_gradient_cosine": {
            "median_interval_mean": float(np.median(cosine)),
            "minimum_interval_mean": float(np.min(cosine)),
            "maximum_interval_mean": float(np.max(cosine)),
            "minimum_observed_in_interval": float(np.nanmin(cosine_min)),
            "maximum_observed_in_interval": float(np.nanmax(cosine_max)),
        },
        "dormancy": {
            "registered_relative_threshold": float(
                config["reliability"]["dormant_relative_threshold"]
            ),
            "maximum_logged_layer_fraction": (
                float(dormancy_rows["value"].max()) if not dormancy_rows.empty else None
            ),
            "diagnostic_rows": int(len(dormancy_rows)),
        },
        "standard_grid_evaluation": evaluation,
        "interpretation_boundary": (
            "The simultaneous objective is numerically dominated by the weighted BC "
            "gradient in this pilot. The cosine range includes both aligned and "
            "conflicting minibatches. This does not establish a performance effect."
        ),
    }

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "joint_loss_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    axes[0].plot(common, bc, marker="o", color="#13857b", label="weighted BC gradient")
    axes[0].plot(common, sac, marker="o", color="#2463eb", label="weighted SAC gradient")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean actor-gradient norm")
    axes[0].set_title(
        "Simultaneous BC + SAC is scale-imbalanced in the seed-0 pilot",
        loc="left",
        fontweight="bold",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].text(
        0.01,
        0.06,
        f"Median weighted norm ratio: {np.median(ratio):.1f}×",
        transform=axes[0].transAxes,
        fontweight="bold",
    )

    axes[1].fill_between(
        common,
        cosine_min.to_numpy(dtype=float),
        cosine_max.to_numpy(dtype=float),
        color="#ff7a18",
        alpha=0.18,
        label="min to max minibatch cosine",
    )
    axes[1].plot(
        common,
        cosine,
        marker="o",
        color="#ff7a18",
        label="interval mean cosine",
    )
    axes[1].axhline(0.0, color="#172033", linewidth=1, linestyle="--")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Environment step")
    axes[1].set_ylabel("BC–SAC gradient cosine")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=2, loc="lower right")

    status = "complete" if completed else f"partial through step {max_logged_step:,}"
    fig.text(
        0.01,
        0.005,
        (
            f"Seed 0, 25k-step diagnostic ({status}). Shading is the logged "
            "within-interval range. Tier C evidence: mechanism hypothesis only."
        ),
        fontsize=8.5,
        color="#485266",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    figure_path = figures_dir / "07_joint_loss_gradient_diagnostic.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"summary": str(summary_path), "figure": str(figure_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
