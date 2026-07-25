from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "reports"
    / "plan2307_pure_target_architecture_20260723"
    / "terminal_offgrid_20260725"
)
SELECTED_VARIANT = (
    "global_n41_m0.005_b1_symmetric_actor_unanimous_advantage"
)
REPORTED_VARIANTS = ("actor", "reflection_actor", SELECTED_VARIANT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", type=Path, default=DEFAULT_ROOT / "p0.json")
    parser.add_argument("--p7", type=Path, default=DEFAULT_ROOT / "p7.json")
    parser.add_argument("--p8", type=Path, default=DEFAULT_ROOT / "p8_seed0.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], field: str) -> float:
    value = metrics[field]
    if isinstance(value, dict):
        return float(value["rate"])
    return float(value)


def aggregate(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["results"]:
        grouped[str(row["variant"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for variant, rows in grouped.items():
        points = [int(float(row["metrics"]["points"])) for row in rows]
        total = sum(points)

        def pooled_rate(field: str) -> tuple[int, float]:
            successes = 0
            for row, count in zip(rows, points, strict=True):
                value = row["metrics"][field]
                successes += (
                    int(value["successes"])
                    if isinstance(value, dict)
                    else int(round(float(value) * count))
                )
            return successes, successes / total

        near_count, near_rate = pooled_rate("near_reference_eps")
        task_count, task_rate = pooled_rate("task_success_rate")
        strict_count, strict_rate = pooled_rate("strict_beats_reference")
        result[variant] = {
            "seeds": len(rows),
            "points_per_seed": sorted(set(points)),
            "trials": total,
            "near_count": near_count,
            "near_rate": near_rate,
            "task_count": task_count,
            "task_rate": task_rate,
            "strict_count": strict_count,
            "strict_rate": strict_rate,
            "mean_return": fmean(float(row["metrics"]["mean_return"]) for row in rows),
            "bottom10_mean_return": fmean(
                float(row["metrics"]["bottom10_conditional_mean_return"])
                for row in rows
            ),
            "near_seed_rates": [
                metric_value(row["metrics"], "near_reference_eps") for row in rows
            ],
            "task_seed_rates": [
                metric_value(row["metrics"], "task_success_rate") for row in rows
            ],
            "strict_seed_rates": [
                metric_value(row["metrics"], "strict_beats_reference") for row in rows
            ],
        }
    return result


def seed_row(payload: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in payload["results"]
        if row["variant"] == variant and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {variant} row for seed {seed}")
    return matches[0]


def plot(summary: dict[str, Any], out: Path) -> None:
    methods = ["P0 one-step", "P7 FastSACN8"]
    keys = ["p0", "p7"]
    metrics = [
        ("near_rate", "Near reference", "#0f766e"),
        ("task_rate", "Task success", "#2563eb"),
        ("strict_rate", "Strict win", "#ea580c"),
    ]
    x = np.arange(len(methods))
    width = 0.23
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for index, (field, label, color) in enumerate(metrics):
        values = [
            100 * float(summary[key]["selected_variant"][field]) for key in keys
        ]
        positions = x + (index - 1) * width
        bars = axis.bar(positions, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    axis.set_xticks(x, methods)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Locked off-grid success rate (%)")
    axis.set_title(
        "100k FastSACN8 combination test: same architecture and deployment",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncols=3, loc="upper center")
    fig.savefig(out / "terminal_fastsacn_gate.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p0_payload = load(args.p0)
    p7_payload = load(args.p7)
    p8_payload = load(args.p8)
    p0 = aggregate(p0_payload)
    p7 = aggregate(p7_payload)
    p8 = aggregate(p8_payload)
    for label, data, expected_seeds in (
        ("P0", p0, 5),
        ("P7", p7, 5),
        ("P8", p8, 1),
    ):
        missing = [variant for variant in REPORTED_VARIANTS if variant not in data]
        if missing:
            raise ValueError(f"{label} lacks variants: {missing}")
        if int(data[SELECTED_VARIANT]["seeds"]) != expected_seeds:
            raise ValueError(f"{label} has the wrong seed count")

    p0_selected = p0[SELECTED_VARIANT]
    p7_selected = p7[SELECTED_VARIANT]
    p7_seed0 = seed_row(p7_payload, SELECTED_VARIANT, 0)["metrics"]
    p8_seed0 = seed_row(p8_payload, SELECTED_VARIANT, 0)["metrics"]
    p7_promote = (
        float(p7_selected["near_rate"]) > float(p0_selected["near_rate"])
        and float(p7_selected["task_rate"]) >= float(p0_selected["task_rate"])
    )
    p8_supporting_improvement = any(
        metric_value(p8_seed0, field) > metric_value(p7_seed0, field)
        for field in (
            "strict_beats_reference",
            "mean_return",
            "bottom10_conditional_mean_return",
        )
    )
    p8_gate = (
        metric_value(p8_seed0, "near_reference_eps")
        > metric_value(p7_seed0, "near_reference_eps")
        and metric_value(p8_seed0, "task_success_rate")
        >= metric_value(p7_seed0, "task_success_rate")
        and p8_supporting_improvement
    )
    summary = {
        "protocol": {
            "p0_state_sha256": p0_payload["protocol"]["state_sha256"],
            "p7_state_sha256": p7_payload["protocol"]["state_sha256"],
            "p8_state_sha256": p8_payload["protocol"]["state_sha256"],
            "selected_variant": SELECTED_VARIANT,
            "authority_grid_queried": False,
        },
        "p0": {
            "selected_variant": p0_selected,
            "variants": {variant: p0[variant] for variant in REPORTED_VARIANTS},
        },
        "p7": {
            "selected_variant": p7_selected,
            "variants": {variant: p7[variant] for variant in REPORTED_VARIANTS},
        },
        "p8_seed0": {
            "selected_variant": p8[SELECTED_VARIANT],
            "p7_seed0_metrics": p7_seed0,
            "p8_seed0_metrics": p8_seed0,
        },
        "decision": {
            "promote_p7_to_authority_evaluation": p7_promote,
            "promote_p8_to_four_more_seeds": p8_gate,
            "p7_rule": "higher near-reference rate and no task-success regression",
            "p8_supporting_improvement": p8_supporting_improvement,
            "p8_rule": (
                "seed-0 near-reference improvement, at least one supporting "
                "improvement, and no task-success regression"
            ),
        },
    }
    require_hashes = {
        p0_payload["protocol"]["state_sha256"],
        p7_payload["protocol"]["state_sha256"],
        p8_payload["protocol"]["state_sha256"],
    }
    if len(require_hashes) != 1:
        raise ValueError("P0, P7, and P8 used different locked state sets")
    (args.out / "terminal_fastsacn_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(summary, args.out)
    lines = [
        "# Terminal FastSACN8 promotion decision",
        "",
        f"- P7 authority promotion: **{p7_promote}**",
        f"- P8 five-seed promotion: **{p8_gate}**",
        f"- Locked state hash: `{next(iter(require_hashes))}`",
        "",
        "| Method | Variant | Seeds | Near | Task | Strict | Mean return |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, family in (("P0 one-step", p0), ("P7 FastSACN8", p7)):
        for variant in ("actor", SELECTED_VARIANT):
            row = family[variant]
            label = "ordinary actor" if variant == "actor" else "reflection + Q-search"
            lines.append(
                f"| {name} | {label} | {row['seeds']} "
                f"| {100*row['near_rate']:.3f}% | {100*row['task_rate']:.3f}% "
                f"| {100*row['strict_rate']:.3f}% | {row['mean_return']:.3f} |"
            )
    lines.extend(
        [
            "",
            "P8 is a seed-zero gate only. It cannot enter the mainline.",
            "The authority grid was not queried by this decision.",
        ]
    )
    (args.out / "terminal_fastsacn_gate_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["decision"]))


if __name__ == "__main__":
    main()
