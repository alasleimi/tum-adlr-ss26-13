from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".build" / "diagnostics" / "mixed_q41_validation"
BASELINE = (
    ROOT
    / ".build"
    / "diagnostics"
    / "mixed_reflection_validation"
    / "summary.json"
)
MODELS = ROOT / "artifacts" / "report_reproduction" / "models"
CRITIC = str(MODELS / "mixed_shared_critic" / "seed1")
ACTORS = [
    str(MODELS / "mixed_selected" / f"seed{seed}")
    for seed in range(5)
]
MARGINS = (0.0, 0.005)


def weighted_aggregate(seed_variants: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, float]] = {}
    for split in ("midpoint", "continuous_uniform_reset"):
        points = sum(int(variant[split]["points"]) for variant in seed_variants)
        row: dict[str, float] = {"points": float(points)}
        for key in (
            "near_reference_eps",
            "task_success",
            "strict_beats_reference",
            "mean_return",
        ):
            row[key] = sum(
                float(variant[split][key]) * int(variant[split]["points"])
                for variant in seed_variants
            ) / points
        aggregate[split] = row
    return aggregate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "experiments" / "implementations")]
    )
    for seed, actor in enumerate(ACTORS):
        target = OUT / f"global_q41_seed{seed}.json"
        command = [
            sys.executable,
            str(
                ROOT
                / "experiments"
                / "implementations"
                / "evaluate_general_hybrid_qsearch_validation.py"
            ),
            "--actor-run",
            actor,
            "--critic-run",
            CRITIC,
            "--out",
            str(target),
            "--device",
            "cuda",
            "--seed",
            "0",
            "--global-search",
            "--num-actions",
            "41",
            "--search-radius",
            "0.1",
            "--max-action-delta",
            "4.0",
            "--margins",
            *(str(margin) for margin in MARGINS),
            "--theta-bins",
            "47",
            "--velocity-bins",
            "31",
            "--random-points",
            "4096",
            "--reset-velocity-limit",
            "1.0",
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    conditions: dict[str, Any] = {
        "local_q5_margin0": baseline["conditions"]["local_qsearch"]
    }
    for index, margin in enumerate(MARGINS):
        seed_variants = []
        for seed in range(5):
            payload = json.loads(
                (OUT / f"global_q41_seed{seed}.json").read_text(encoding="utf-8")
            )
            seed_variants.append(
                {"seed": seed, **payload["variants"][index]}
            )
        conditions[f"global_q41_margin{margin:g}"] = {
            "aggregate": weighted_aggregate(seed_variants),
            "seeds": seed_variants,
        }

    summary = {
        "protocol": {
            "actor_seeds": 5,
            "same_offgrid_states_for_every_condition": True,
            "evaluation_seed": 0,
            "midpoint_grid": [47, 31],
            "continuous_uniform_reset_points": 4096,
            "authority_grid_queried": False,
            "reference_used_at_inference": False,
            "actor_runs": ACTORS,
            "critic_run": CRITIC,
            "global_candidate_actions": 41,
            "global_action_support": [-2.0, 2.0],
            "global_max_action_delta": 4.0,
        },
        "conditions": conditions,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(conditions, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
