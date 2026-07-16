from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

DAGGER = ROOT / "reports/distill_best_simbav2_dagger_2iter_static240k_20260702/relative_success_vellim1_3seed/relative_cell_summary.csv"
STATIC = ROOT / "reports/distill_best_simbav2_balanced_400k_20260701/relative_success_vellim1/relative_cell_summary.csv"
SIMBA = ROOT / "reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_cell_summary.csv"
QFILTERED = [
    ROOT / "reports/simbav2_compact_fastsacn8_lam05_utd2_hard02to001_s10k_d20k_seed0_50k_20260707/final_critic_grid41_m005_eval/relative/relative_cell_summary.csv",
    ROOT / "reports/simbav2_compact_fastsacn8_lam05_utd2_hard02to001_s10k_d20k_seed1_50k_20260708/final_critic_grid41_m005_eval/relative/relative_cell_summary.csv",
    ROOT / "reports/simbav2_compact_fastsacn8_lam05_utd2_hard02to001_s10k_d20k_seed2_50k_20260708/final_critic_grid41_m005_eval/relative/relative_cell_summary.csv",
    ROOT / "reports/simbav2_compact_fastsacn8_lam05_utd2_hard02to001_s10k_d20k_seed3_50k_20260708/final_critic_grid41_m005_eval/relative/relative_cell_summary.csv",
    ROOT / "reports/simbav2_compact_fastsacn8_lam05_utd2_hard02to001_s10k_d20k_seed4_50k_20260708/final_critic_grid41_m005_eval/relative/relative_cell_summary.csv",
]


def save_opaque_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", transparent=False)
    with Image.open(path) as image:
        image.convert("RGB").save(path)


def read_cell_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != 61 * 41:
        raise ValueError(f"Expected 2,501 rows in {path}, found {len(rows)}")
    return rows


def aggregate(rows_by_seed: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    keys = ("task_success_rate", "near_best_known_return_eps_rate", "beats_best_known_return_rate")
    output: list[dict[str, float]] = []
    for cell_rows in zip(*rows_by_seed, strict=True):
        first = cell_rows[0]
        for row in cell_rows[1:]:
            if row["theta"] != first["theta"] or row["theta_dot"] != first["theta_dot"]:
                raise ValueError("Per-seed cell tables are not aligned")
        item = {
            "theta": first["theta"],
            "theta_degrees": first["theta_degrees"],
            "theta_dot": first["theta_dot"],
            "num_training_seeds": float(len(cell_rows)),
        }
        for key in keys:
            item[key] = float(np.mean([row[key] for row in cell_rows]))
        output.append(item)
    return output


def matrix(rows: list[dict[str, float]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.array(sorted({row["theta_degrees"] for row in rows}))
    velocities = np.array(sorted({row["theta_dot"] for row in rows}))
    lookup = {(row["theta_degrees"], row["theta_dot"]): row[metric] for row in rows}
    values = np.array([[lookup[(angle, velocity)] for angle in angles] for velocity in velocities])
    return angles, velocities, values


def plot_heatmaps(methods: list[dict[str, object]], metric: str, title: str, output_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.08, top=0.78, wspace=0.08, hspace=0.42)
    image = None
    for ax, method in zip(axes.flat, methods, strict=True):
        rows = method["rows"]
        assert isinstance(rows, list)
        angles, velocities, values = matrix(rows, metric)
        image = ax.imshow(
            values,
            origin="lower",
            extent=(angles.min(), angles.max(), velocities.min(), velocities.max()),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        rate = float(np.mean(values))
        ax.set_title(f'{method["label"]}\nmean={rate:.4f}; seeds={method["seeds"]}', fontsize=11)
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Initial angle (degrees)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Initial angular velocity")
    assert image is not None
    colorbar_axis = fig.add_axes([0.90, 0.16, 0.018, 0.56])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Fraction of training seeds succeeding from this initial state")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    save_opaque_png(fig, OUT / output_name)
    plt.close(fig)


def plot_headline_metrics(methods: list[dict[str, object]]) -> None:
    metrics = [
        ("near_best_known_return_eps_rate", "Near reference"),
        ("task_success_rate", "Task success"),
        ("beats_best_known_return_rate", "Beats reference"),
    ]
    labels = [str(method["short_label"]) for method in methods]
    x = np.arange(len(labels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11.8, 5.6), constrained_layout=True)
    for index, (key, label) in enumerate(metrics):
        values = [float(np.mean([row[key] for row in method["rows"]])) for method in methods]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=8, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Mean success fraction over 2,501 initial states")
    ax.set_title("Matched Pendulum reset-grid comparison")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    save_opaque_png(fig, OUT / "headline_metrics.png")
    plt.close(fig)


def main() -> None:
    methods = [
        {
            "label": "DAgger distillation\n2 aggregation rounds",
            "short_label": "DAgger\n3 seeds",
            "seeds": 3,
            "rows": read_cell_csv(DAGGER),
            "sources": [str(DAGGER.relative_to(ROOT))],
        },
        {
            "label": "Static balanced distillation\nno policy rollouts",
            "short_label": "Static distill\n1 seed",
            "seeds": 1,
            "rows": read_cell_csv(STATIC),
            "sources": [str(STATIC.relative_to(ROOT))],
        },
        {
            "label": "Standard SimbaV2\n100k environment steps",
            "short_label": "SimbaV2\n5 seeds",
            "seeds": 5,
            "rows": read_cell_csv(SIMBA),
            "sources": [str(SIMBA.relative_to(ROOT))],
        },
        {
            "label": "FastSACN8 + Q-filtered search\n50k environment steps",
            "short_label": "Q-filtered\nFastSACN, 5 seeds",
            "seeds": 5,
            "rows": aggregate([read_cell_csv(path) for path in QFILTERED]),
            "sources": [str(path.relative_to(ROOT)) for path in QFILTERED],
        },
    ]

    plot_heatmaps(
        methods,
        "near_best_known_return_eps_rate",
        "Near-reference success by exact initial state\nreturn >= max(DP, controller) - 5",
        "near_reference_success_heatmaps.png",
    )
    plot_heatmaps(
        methods,
        "task_success_rate",
        "Task success by exact initial state\n>=80% near-upright steps and no >50-step not-near-upright streak",
        "task_success_heatmaps.png",
    )
    plot_headline_metrics(methods)

    metrics = {}
    for method in methods:
        rows = method["rows"]
        assert isinstance(rows, list)
        metrics[str(method["short_label"]).replace("\n", " ")] = {
            "seeds": method["seeds"],
            "near_best_known_return_eps": float(np.mean([row["near_best_known_return_eps_rate"] for row in rows])),
            "task_success": float(np.mean([row["task_success_rate"] for row in rows])),
            "beats_best_known_return": float(np.mean([row["beats_best_known_return_rate"] for row in rows])),
            "sources": method["sources"],
        }
    (OUT / "computed_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
