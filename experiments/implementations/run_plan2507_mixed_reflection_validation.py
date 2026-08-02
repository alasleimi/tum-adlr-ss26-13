from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".build" / "diagnostics" / "mixed_reflection_validation"
MODELS = ROOT / "artifacts" / "report_reproduction" / "models"
CRITIC = str(MODELS / "mixed_shared_critic" / "seed1")
ACTORS = [
    str(MODELS / "mixed_selected" / f"seed{seed}")
    for seed in range(5)
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "experiments" / "implementations")]
    )
    for symmetric in (False, True):
        mode = "reflection_local_qsearch" if symmetric else "local_qsearch"
        for seed, actor in enumerate(ACTORS):
            target = OUT / f"{mode}_seed{seed}.json"
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
                "--num-actions",
                "5",
                "--search-radius",
                "0.1",
                "--margins",
                "0.0",
                "--theta-bins",
                "47",
                "--velocity-bins",
                "31",
                "--random-points",
                "4096",
                "--reset-velocity-limit",
                "1.0",
            ]
            if symmetric:
                command.append("--symmetric-actor-fallback")
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    summary: dict[str, object] = {
        "protocol": {
            "actor_seeds": 5,
            "same_offgrid_states_for_every_actor": True,
            "evaluation_seed": 0,
            "midpoint_grid": [47, 31],
            "continuous_uniform_reset_points": 4096,
            "authority_grid_queried": False,
            "reference_used_at_inference": False,
            "critic_run": CRITIC,
        },
        "conditions": {},
    }
    for mode in ("local_qsearch", "reflection_local_qsearch"):
        aggregate = {
            "midpoint": {
                "points": 0,
                "near_reference_eps": 0.0,
                "task_success": 0.0,
                "strict_beats_reference": 0.0,
                "mean_return": 0.0,
            },
            "continuous_uniform_reset": {
                "points": 0,
                "near_reference_eps": 0.0,
                "task_success": 0.0,
                "strict_beats_reference": 0.0,
                "mean_return": 0.0,
            },
        }
        seed_rows = []
        for seed in range(5):
            payload = json.loads((OUT / f"{mode}_seed{seed}.json").read_text())
            variant = payload["variants"][0]
            seed_rows.append({"seed": seed, **variant})
            for split in aggregate:
                points = int(variant[split]["points"])
                aggregate[split]["points"] += points
                for key in (
                    "near_reference_eps",
                    "task_success",
                    "strict_beats_reference",
                    "mean_return",
                ):
                    aggregate[split][key] += float(variant[split][key]) * points
        for split in aggregate:
            points = int(aggregate[split]["points"])
            for key in (
                "near_reference_eps",
                "task_success",
                "strict_beats_reference",
                "mean_return",
            ):
                aggregate[split][key] /= points
        summary["conditions"][mode] = {
            "aggregate": aggregate,
            "seeds": seed_rows,
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["conditions"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
