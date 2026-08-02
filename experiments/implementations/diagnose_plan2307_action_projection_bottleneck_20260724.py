from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "report"
    / "diagnostics"
    / "critic_direction"
    / "rows.csv"
)
DEFAULT_SOURCE_SUMMARY = DEFAULT_INPUT.with_name("summary.json")
DEFAULT_OUTPUT = (
    ROOT
    / ".build"
    / "diagnostics"
    / "action_projection_bottleneck"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how action clipping mediates a frozen critic-gradient "
            "intervention for one-step, FastSACN8, and SACn8 policies."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--boundary-threshold", type=float, default=1.95)
    parser.add_argument("--nominal-step", type=float, default=0.05)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "replicate",
            "near_reference",
            "actor_action",
            "critic_gradient",
            "critic_step_action",
            "critic_step_discounted_gain",
        ):
            row[key] = float(row[key])
    return rows


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def aggregate(
    rows: list[dict[str, Any]],
    boundary_threshold: float,
    nominal_step: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        outcome = "success" if int(row["near_reference"]) == 1 else "failure"
        groups[(str(row["condition"]), int(row["replicate"]), outcome)].append(row)

    output: list[dict[str, Any]] = []
    for (condition, replicate, outcome), group in sorted(groups.items()):
        boundary_rows = [
            row for row in group if abs(float(row["actor_action"])) > boundary_threshold
        ]
        outward_boundary = [
            row
            for row in boundary_rows
            if float(row["actor_action"]) * float(row["critic_gradient"]) > 0
        ]
        effective_steps = [
            abs(float(row["critic_step_action"]) - float(row["actor_action"]))
            for row in group
        ]
        gains = [float(row["critic_step_discounted_gain"]) for row in group]
        output.append(
            {
                "condition": condition,
                "replicate": replicate,
                "outcome": outcome,
                "rows": len(group),
                "boundary_rows": len(boundary_rows),
                "boundary_rate": len(boundary_rows) / len(group),
                "outward_boundary_rows": len(outward_boundary),
                "outward_among_boundary_rate": (
                    len(outward_boundary) / len(boundary_rows)
                    if boundary_rows
                    else float("nan")
                ),
                "mean_effective_step": safe_mean(effective_steps),
                "mean_effective_step_fraction": safe_mean(effective_steps)
                / nominal_step,
                "mean_discounted_gain": safe_mean(gains),
                "gain_q01": float(np.quantile(gains, 0.01)),
                "gain_q05": float(np.quantile(gains, 0.05)),
                "gain_median": float(np.median(gains)),
            }
        )
    return output


def pooled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["outcome"]))].append(row)
    result: list[dict[str, Any]] = []
    for (condition, outcome), group in sorted(groups.items()):
        total = sum(int(row["rows"]) for row in group)
        boundary = sum(int(row["boundary_rows"]) for row in group)
        outward = sum(int(row["outward_boundary_rows"]) for row in group)
        result.append(
            {
                "condition": condition,
                "outcome": outcome,
                "rows": total,
                "boundary_rate": boundary / total,
                "outward_among_boundary_rate": outward / boundary,
                "mean_effective_step": sum(
                    float(row["mean_effective_step"]) * int(row["rows"])
                    for row in group
                )
                / total,
                "mean_effective_step_fraction": sum(
                    float(row["mean_effective_step_fraction"]) * int(row["rows"])
                    for row in group
                )
                / total,
                "mean_discounted_gain": sum(
                    float(row["mean_discounted_gain"]) * int(row["rows"])
                    for row in group
                )
                / total,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    output: Path,
    seed_rows: list[dict[str, Any]],
    pooled_rows: list[dict[str, Any]],
) -> None:
    conditions = list(dict.fromkeys(row["condition"] for row in pooled_rows))
    outcomes = ["success", "failure"]
    colors = {
        "P0 one-step": "#147d73",
        "P1 FastSACN8": "#bd5705",
        "P2 SACn8": "#2864e8",
    }
    x = np.arange(len(conditions))
    width = 0.34
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3))

    def values(metric: str, outcome: str) -> list[float]:
        lookup = {
            (row["condition"], row["outcome"]): float(row[metric])
            for row in pooled_rows
        }
        return [lookup[(condition, outcome)] for condition in conditions]

    for offset, outcome in zip((-width / 2, width / 2), outcomes):
        label = "near-reference success" if outcome == "success" else "failure"
        axes[0, 0].bar(
            x + offset,
            np.asarray(values("boundary_rate", outcome)) * 100,
            width,
            label=label,
            alpha=0.85,
        )
        axes[0, 1].bar(
            x + offset,
            np.asarray(values("outward_among_boundary_rate", outcome)) * 100,
            width,
            label=label,
            alpha=0.85,
        )
        axes[0, 2].bar(
            x + offset,
            values("mean_effective_step", outcome),
            width,
            label=label,
            alpha=0.85,
        )
        axes[1, 0].bar(
            x + offset,
            values("mean_discounted_gain", outcome),
            width,
            label=label,
            alpha=0.85,
        )

    titles = (
        "A. Actor action near the torque boundary",
        "B. Critic points farther outward at boundary",
        "C. Effective size of the nominal 0.05 step",
        "D. Mean discounted return change",
    )
    ylabels = (
        "states (%)",
        "boundary states (%)",
        "absolute torque",
        "discounted return change",
    )
    for axis, title, ylabel in zip(axes.flat[:4], titles, ylabels):
        axis.set_title(title)
        axis.set_xticks(x, conditions, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)

    failure_rows = [row for row in seed_rows if row["outcome"] == "failure"]
    for condition in conditions:
        current = [row for row in failure_rows if row["condition"] == condition]
        current.sort(key=lambda row: int(row["replicate"]))
        seeds = [int(row["replicate"]) for row in current]
        color = colors.get(condition)
        axes[1, 1].plot(
            seeds,
            [100 * float(row["boundary_rate"]) for row in current],
            marker="o",
            linewidth=2,
            color=color,
            label=condition,
        )
        axes[1, 2].plot(
            seeds,
            [100 * float(row["outward_among_boundary_rate"]) for row in current],
            marker="o",
            linewidth=2,
            color=color,
            label=condition,
        )
    axes[1, 1].set_title("E. Boundary failures by trained seed")
    axes[1, 1].set_ylabel("failure states (%)")
    axes[1, 2].set_title("F. Outward critic direction on boundary failures")
    axes[1, 2].set_ylabel("boundary failure states (%)")
    for axis in axes[1, 1:]:
        axis.set_xlabel("actor seed")
        axis.set_xticks(range(5))
        axis.set_ylim(0, 102)
        axis.grid(axis="y", alpha=0.2)
    axes[1, 2].legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle(
        "Action limits mediate local critic corrections\n"
        "Five actor-critic pairs per target family on the same locked sample",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "action_projection_bottleneck.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    source_summary_path = resolve(args.source_summary)
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_rows = read_rows(input_path)
    seed_rows = aggregate(
        source_rows,
        boundary_threshold=float(args.boundary_threshold),
        nominal_step=float(args.nominal_step),
    )
    pooled_rows = pooled(seed_rows)
    write_csv(output / "action_projection_by_seed.csv", seed_rows)
    write_csv(output / "action_projection_pooled.csv", pooled_rows)
    plot(output, seed_rows, pooled_rows)

    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "diagnostic_scope": (
            "Read-only decomposition of the registered C32 frozen first-action "
            "critic-gradient intervention. It measures how the bounded action "
            "space changes the realized size of a nominal critic-directed step."
        ),
        "protocol": {
            "input_rows": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "input_rows_sha256": sha256(input_path),
            "source_summary": str(source_summary_path.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "source_summary_sha256": sha256(source_summary_path),
            "locked_dataset_sha256": source_summary["protocol"][
                "locked_dataset_sha256"
            ],
            "selected_indices_sha256": source_summary["protocol"][
                "selected_indices_sha256"
            ],
            "sample_count_per_seed": source_summary["protocol"]["sample_count"],
            "boundary_threshold": float(args.boundary_threshold),
            "nominal_step": float(args.nominal_step),
            "reference_access_during_inference": False,
            "authority_grid_used": False,
        },
        "pooled": pooled_rows,
        "interpretation": (
            "The multi-step failure trials are concentrated at the action bound, "
            "where the critic usually points farther outward and projection makes "
            "the nominal local step very small. This explains why favorable local "
            "direction frequency need not produce an available corrective action."
        ),
        "limitations": [
            (
                "The diagnostic is conditioned on post-training outcomes and does "
                "not establish that saturation caused a failure."
            ),
            (
                "Many correct pendulum actions are saturated. The result concerns "
                "the interaction between the action bound and a local intervention, "
                "not whether saturated control is generally undesirable."
            ),
            (
                "The intervention changes the first action only. The registered "
                "prefix experiment shows that the larger failure tail is sequential."
            ),
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
