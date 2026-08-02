from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
NO_IMPORTANCE_SOURCE = (
    ROOT
    / "artifacts"
    / "report_reproduction"
    / "replay"
    / "objective_share"
    / "none"
    / "seed0"
    / "sacn_horizon_summary.json"
)
DENSITY_SOURCES = [
    *[
        ROOT
        / "artifacts"
        / "report_reproduction"
        / "replay"
        / "objective_share"
        / "density"
        / f"seed{seed}"
        / "sacn_horizon_summary.json"
        for seed in range(5)
    ],
]
OUT = (
    ROOT
    / ".build"
    / "diagnostics"
    / "fastsacn_objective_share"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def horizon_eight(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        int(data["samples"]) != 4096
        or data["sacn_target_mode"] != "fast_last"
        or float(data["sacn_horizon_lambda"]) != 0.5
    ):
        raise ValueError(f"unexpected FastSACN diagnostic protocol: {path}")
    row = next(item for item in data["summary"] if int(item["horizon"]) == 8)
    return data, dict(row["all"])


def main() -> None:
    none_protocol, none = horizon_eight(NO_IMPORTANCE_SOURCE)
    density_loaded = [horizon_eight(path) for path in DENSITY_SOURCES]
    density_rows = [row for _protocol, row in density_loaded]

    nominal = 100.0 * float(none["nominal_last_horizon_objective_share_mean"])
    empirical_none = 100.0 * float(none["empirical_last_horizon_objective_share"])
    empirical_density_values = [
        100.0 * float(row["empirical_last_horizon_objective_share"])
        for row in density_rows
    ]
    density_mean_importance_values = [
        100.0 * float(row["importance_weight_last_mean"])
        for row in density_rows
    ]
    density_ess_values = [
        100.0 * float(row["effective_weight_ess_last_fraction"])
        for row in density_rows
    ]
    density_collapsed_values = [
        100.0 * float(row["effective_weight_last_le_1e_3_rate"])
        for row in density_rows
    ]
    empirical_density = sum(empirical_density_values) / len(
        empirical_density_values
    )
    density_mean_importance = sum(density_mean_importance_values) / len(
        density_mean_importance_values
    )
    density_ess = sum(density_ess_values) / len(density_ess_values)
    density_collapsed = sum(density_collapsed_values) / len(
        density_collapsed_values
    )

    if abs(nominal - 0.7751937955617905) > 1e-9:
        raise ValueError("nominal eight-step objective share drifted")
    if abs(empirical_none - nominal) > 1e-9:
        raise ValueError("unweighted empirical share differs from the objective")
    if empirical_density >= empirical_none:
        raise ValueError("density weighting no longer reduces the long endpoint")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "selection_use": False,
        "diagnostic_only": True,
        "actor_seeds": {
            "no_importance": [int(none_protocol["seed"])],
            "density": list(range(len(density_loaded))),
        },
        "replay_sequences_per_seed": 4096,
        "density_replay_sequences_total": 4096 * len(density_rows),
        "target_mode": "fast_last",
        "horizon_lambda": 0.5,
        "eight_step_objective_share_percent": {
            "nominal": nominal,
            "no_importance_empirical": empirical_none,
            "density_empirical_seed_values": empirical_density_values,
            "density_empirical_mean": empirical_density,
        },
        "density_eight_step_diagnostics_percent": {
            "mean_importance_weight_seed_values": density_mean_importance_values,
            "mean_importance_weight_mean": density_mean_importance,
            "effective_weight_ess_fraction_seed_values": density_ess_values,
            "effective_weight_ess_fraction_mean": density_ess,
            "effective_weight_at_most_1e-3_seed_values": (
                density_collapsed_values
            ),
            "effective_weight_at_most_1e-3_mean": density_collapsed,
        },
        "interpretation": (
            "Lambda-0.5 FastSACN8 is dominated by its one-step endpoint. "
            "Density weighting reduces the already small eight-step objective "
            "share further across five independently trained replay buffers."
        ),
        "limitations": [
            (
                "The no-importance replay diagnostic uses one actor seed; the "
                "density diagnostic uses five independently trained actors."
            ),
            "Objective share does not determine reliability by itself.",
            "The result does not identify which training update caused a policy failure.",
        ],
        "sources": {
            "no_importance_seed0": {
                "path": str(NO_IMPORTANCE_SOURCE.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "sha256": sha256(NO_IMPORTANCE_SOURCE),
            },
            "density": [
                {
                    "actor_seed": actor_seed,
                    "diagnostic_sample_seed": int(protocol["seed"]),
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(path),
                }
                for actor_seed, (path, (protocol, _row)) in enumerate(
                    zip(DENSITY_SOURCES, density_loaded, strict=True)
                )
            ],
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )
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
    right_values = [
        density_mean_importance,
        density_ess,
        density_collapsed,
    ]
    right_colors = ["#0EA5A4", "#F59E0B", "#DC2626"]
    bars = axes[1].bar(right_labels, right_values, color=right_colors, width=0.62)
    for index, values in enumerate(
        (
            density_mean_importance_values,
            density_ess_values,
            density_collapsed_values,
        )
    ):
        axes[1].scatter(
            [index] * len(values),
            values,
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

    fig.text(
        0.5,
        0.012,
        (
            "Density: five actor seeds and 20,480 replay sequences. "
            "No-importance control: seed zero and 4,096 sequences."
        ),
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.93), w_pad=4.0)
    fig.savefig(OUT / "fastsacn_objective_share.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
