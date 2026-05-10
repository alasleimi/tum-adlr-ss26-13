from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
from typing import Any

from last_nine_rl.config import ExperimentConfig
from last_nine_rl.train import train


def main() -> None:
    args = parse_args()
    base_config = ExperimentConfig.from_json(args.config)
    base_config.validate()
    run_root = Path(args.run_root or base_config.telemetry.run_root)

    entries = build_sweep_entries(base_config, args, run_root)
    manifest = {
        "config": args.config,
        "num_runs": len(entries),
        "runs": entries,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    for entry in entries:
        config = ExperimentConfig.from_dict(entry["config"])
        train(config, Path(entry["run_dir"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated Week 1 SAC experiments over simple scale grids.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--same-seed-repeats",
        type=int,
        default=1,
        help="Number of repeated runs for each actual seed. The actual seed is unchanged across repeats.",
    )
    parser.add_argument(
        "--repeat-offsets",
        nargs="*",
        type=int,
        default=[0],
        help=(
            "Offsets added to each base seed. Use 0 for exact seed runs; use nonzero offsets to create "
            "intentional additional random seeds while preserving the base-seed grouping."
        ),
    )
    parser.add_argument("--total-steps", nargs="*", type=int, default=None)
    parser.add_argument("--updates-per-step", nargs="*", type=int, default=None)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--buffer-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow each generated run to replace known outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_sweep_entries(base_config: ExperimentConfig, args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    if args.same_seed_repeats <= 0:
        raise ValueError("--same-seed-repeats must be positive.")
    if not args.repeat_offsets:
        raise ValueError("--repeat-offsets must contain at least one offset.")

    total_steps = args.total_steps or [base_config.sac.total_steps]
    updates_per_step = args.updates_per_step or [base_config.sac.updates_per_step]
    batch_sizes = args.batch_sizes or [base_config.sac.batch_size]
    buffer_sizes = args.buffer_sizes or [base_config.sac.buffer_size]

    entries: list[dict[str, Any]] = []
    for base_seed, repeat_offset, repeat_index, steps, utd, batch_size, buffer_size in itertools.product(
        args.seeds,
        args.repeat_offsets,
        range(args.same_seed_repeats),
        total_steps,
        updates_per_step,
        batch_sizes,
        buffer_sizes,
    ):
        config = deepcopy(base_config)
        actual_seed = int(base_seed + repeat_offset)
        config.seed = actual_seed
        config.sac.total_steps = int(steps)
        config.sac.updates_per_step = int(utd)
        config.sac.batch_size = int(batch_size)
        config.sac.buffer_size = int(buffer_size)
        config.telemetry.run_root = str(run_root)
        config.telemetry.overwrite = bool(getattr(args, "overwrite", False))
        config.validate()

        scale_tag = f"steps{steps}_utd{utd}_batch{batch_size}_buffer{buffer_size}"
        seed_tag = f"base{base_seed}_offset{repeat_offset}_repeat{repeat_index}_seed{actual_seed}"
        run_dir = run_root / config.name / scale_tag / seed_tag
        entries.append(
            {
                "run_dir": str(run_dir),
                "base_seed": int(base_seed),
                "repeat_offset": int(repeat_offset),
                "repeat_index": int(repeat_index),
                "actual_seed": actual_seed,
                "total_steps": int(steps),
                "updates_per_step": int(utd),
                "batch_size": int(batch_size),
                "buffer_size": int(buffer_size),
                "config": config.to_dict(),
            }
        )
    return entries


if __name__ == "__main__":
    main()
