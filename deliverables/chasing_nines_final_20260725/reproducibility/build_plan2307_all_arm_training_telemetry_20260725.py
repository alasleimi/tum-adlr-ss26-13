from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "runs"
    / "plan2307_completion_20260723"
    / "pure_target_architecture_matrix"
)
MANIFESTS = (
    RUN_ROOT / "completion_manifest.json",
    RUN_ROOT / "extension_completion_manifest.json",
)
OUT = (
    ROOT
    / "reports"
    / "plan2307_pure_target_architecture_20260723"
    / "training_telemetry_screen"
)
LABELS = {
    "p0_simba_onestep_utd1_100k": "P0 Simba one-step",
    "p1_simba_fastsacn8_lambda1_utd1_100k": "P1 FastSACN8",
    "p2_simba_sacn8_lambda1_utd1_100k": "P2 SACn8",
    "p3_simba_sacn16_lambda1_utd1_100k": "P3 SACn16",
    "p4_simba_fastsacn8_lambda1_utd2_actorutd1_100k": "P4 FastSACN8 critic UTD2",
    "p5_simba_fastsacn8_lambda0p5_utd1_100k": "P5 FastSACN8 lambda 0.5",
    "p6_simba_fastsacn8_density_lambda0p5_utd1_100k": "P6 P5 plus density",
    "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k": "P7 P5 plus critic UTD2",
    "p8_simba_fastsacn8_lambda0p5_utd2_actorutd2_100k": "P8 P7 plus actor UTD2",
    "p9_simba_sacn16_density_lambda1_utd1_100k": "P9 SACn16 plus density",
    "a0_compact_onestep_utd1_100k": "A0 compact one-step",
    "a1_compact_fastsacn8_lambda1_utd1_100k": "A1 compact FastSACN8",
    "a2_compact_sacn8_lambda1_utd1_100k": "A2 compact SACn8",
}
PLOT_METRICS = (
    ("online_task_success", "20-start task success", "#2563eb", False),
    ("online_mean_return", "20-start mean return", "#7c3aed", False),
    ("actor_loss", "actor loss", "#ea580c", False),
    ("q_loss", "critic loss", "#0f766e", True),
    ("max_dormant_fraction", "maximum dormant fraction", "#dc2626", False),
    ("min_effective_rank_fraction", "minimum effective-rank fraction", "#0891b2", False),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def control_records() -> list[dict[str, Any]]:
    records = []
    for seed in range(5):
        family = (
            "runs/week3_simbav2_scale_100k_20260526/simba_full_official_opt"
            if seed <= 2
            else "runs/week3_100k_component_ablation_20260527/"
            "simba_full_official_opt"
        )
        records.append(
            {
                "arm": "p0_simba_onestep_utd1_100k",
                "seed": seed,
                "run_dir": f"{family}/seed{seed}",
            }
        )
    return records


def manifest_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in MANIFESTS:
        if not path.is_file():
            continue
        payload = read_json(path)
        records.extend(dict(row) for row in payload.get("records", []))
    return records


def discovered_records() -> list[dict[str, Any]]:
    """Recover terminal runs even when a pruned queue writer did not update a manifest."""
    records: list[dict[str, Any]] = []
    for arm in LABELS:
        arm_root = RUN_ROOT / arm
        if not arm_root.is_dir():
            continue
        for seed in range(5):
            run = arm_root / f"seed{seed}"
            if terminal_complete(run):
                records.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "run_dir": str(run.relative_to(ROOT)).replace("\\", "/"),
                    }
                )
    return records


def terminal_complete(run: Path) -> bool:
    checkpoint = run / "checkpoints" / "final.pt"
    events = run / "events.jsonl"
    if not checkpoint.is_file() or not events.is_file():
        return False
    return any(
        '"type": "run_complete"' in line
        for line in events.read_text(encoding="utf-8").splitlines()[-100:]
    )


def metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest(
    rows: list[dict[str, str]],
    *,
    name: str,
    split: str | None = None,
) -> tuple[int, float] | None:
    matches = [
        row
        for row in rows
        if row["name"] == name and (split is None or row["split"] == split)
    ]
    if not matches:
        return None
    row = max(matches, key=lambda item: int(float(item["step"])))
    return int(float(row["step"])), float(row["value"])


def extrema(
    rows: list[dict[str, str]],
    *,
    contains: str,
    reduction: str,
) -> tuple[int, float] | None:
    matches = [row for row in rows if contains in row["name"]]
    if not matches:
        return None
    terminal_step = max(int(float(row["step"])) for row in matches)
    values = [
        float(row["value"])
        for row in matches
        if int(float(row["step"])) == terminal_step
    ]
    value = max(values) if reduction == "max" else min(values)
    return terminal_step, value


def extract(record: dict[str, Any]) -> dict[str, Any]:
    run = ROOT / str(record["run_dir"])
    rows = metric_rows(run / "metrics.csv")
    task = latest(rows, name="task_success_rate", split="eval")
    task_metric = "task_success_rate"
    if task is None:
        task = latest(rows, name="success_rate", split="eval")
        task_metric = "success_rate"
    mean_return = latest(rows, name="mean_return", split="eval")
    actor_loss = latest(rows, name="actor_loss_mean")
    q_loss = latest(rows, name="q_loss_mean")
    dormant = extrema(rows, contains="dormant_fraction", reduction="max")
    rank = extrema(rows, contains="effective_rank_fraction", reduction="min")
    if None in (task, mean_return, actor_loss, q_loss, dormant, rank):
        missing = [
            name
            for name, value in (
                ("task_success_rate", task),
                ("mean_return", mean_return),
                ("actor_loss_mean", actor_loss),
                ("q_loss_mean", q_loss),
                ("dormant_fraction", dormant),
                ("effective_rank_fraction", rank),
            )
            if value is None
        ]
        raise ValueError(f"{run} is missing telemetry: {missing}")
    assert task and mean_return and actor_loss and q_loss and dormant and rank
    return {
        "arm": str(record["arm"]),
        "label": LABELS.get(str(record["arm"]), str(record["arm"])),
        "seed": int(record["seed"]),
        "run_dir": str(record["run_dir"]).replace("\\", "/"),
        "online_eval_step": task[0],
        "online_task_metric": task_metric,
        "online_task_success": task[1],
        "online_mean_return": mean_return[1],
        "actor_loss_step": actor_loss[0],
        "actor_loss": actor_loss[1],
        "q_loss_step": q_loss[0],
        "q_loss": q_loss[1],
        "representation_step": dormant[0],
        "max_dormant_fraction": dormant[1],
        "min_effective_rank_fraction": rank[1],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], arms: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.5), constrained_layout=True)
    y = np.arange(len(arms))
    for axis, (field, title, color, log_scale) in zip(
        axes.flat, PLOT_METRICS, strict=True
    ):
        for index, arm in enumerate(arms):
            values = np.array(
                [float(row[field]) for row in rows if row["arm"] == arm],
                dtype=float,
            )
            axis.plot(
                [values.min(), values.max()],
                [index, index],
                color="#cbd5e1",
                linewidth=5,
                solid_capstyle="round",
                zorder=1,
            )
            axis.scatter(
                values,
                np.full(len(values), index),
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                s=42,
                zorder=3,
            )
            axis.scatter(
                [fmean(values)],
                [index],
                color=color,
                marker="D",
                s=36,
                zorder=4,
            )
        axis.set_yticks(y)
        axis.set_yticklabels([LABELS.get(arm, arm) for arm in arms], fontsize=9)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=14, fontweight="bold", loc="left")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        if log_scale and all(float(row[field]) > 0 for row in rows):
            axis.set_xscale("log")
        if field in {
            "online_task_success",
            "max_dormant_fraction",
            "min_effective_rank_fraction",
        }:
            axis.set_xlim(-0.02, 1.02)
    fig.suptitle(
        "Training scalars are health checks, not fixed-grid reliability tests",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Circles are independent seeds; diamonds are five-seed means. "
        "Only complete 100,000-step arms are shown. Final selection uses the "
        "locked off-grid protocol, not these 20-start online evaluations.",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.savefig(OUT / "all_arm_training_telemetry.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for record in [
        *control_records(),
        *manifest_records(),
        *discovered_records(),
    ]:
        candidates_by_identity[(str(record["arm"]), int(record["seed"]))] = record
    candidates = list(candidates_by_identity.values())
    complete = [
        record
        for record in candidates
        if terminal_complete(ROOT / str(record["run_dir"]))
    ]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in complete:
        by_arm[str(record["arm"])].append(record)
    selected_arms = [
        arm
        for arm in LABELS
        if sorted(int(row["seed"]) for row in by_arm.get(arm, []))
        == [0, 1, 2, 3, 4]
    ]
    if not selected_arms:
        raise ValueError("no complete five-seed arms found")
    rows = [
        extract(record)
        for arm in selected_arms
        for record in sorted(by_arm[arm], key=lambda item: int(item["seed"]))
    ]
    write_csv(OUT / "training_telemetry_rows.csv", rows)
    plot(rows, selected_arms)
    summary = {
        "schema_version": 1,
        "evidence_scope": "terminal telemetry for complete five-seed 100k arms",
        "complete_five_seed_arms": selected_arms,
        "arm_count": len(selected_arms),
        "policy_count": len(rows),
        "selection_use": False,
        "interpretation_limit": (
            "The online evaluation uses twenty starts. Scalar losses, dormancy, "
            "and effective rank are descriptive health checks. They cannot "
            "replace locked off-grid evaluation or identify a causal mechanism."
        ),
        "terminal_means": {
            arm: {
                field: fmean(
                    float(row[field]) for row in rows if row["arm"] == arm
                )
                for field, _title, _color, _log in PLOT_METRICS
            }
            for arm in selected_arms
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"arms": len(selected_arms), "policies": len(rows)}))


if __name__ == "__main__":
    main()
