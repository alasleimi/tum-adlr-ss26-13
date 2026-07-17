from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route complete Pendulum episodes by initial angle.")
    parser.add_argument("--ordinary-rollouts", nargs="+", required=True)
    parser.add_argument("--hard-rollouts", nargs="+", required=True)
    parser.add_argument("--threshold-degrees", type=float, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[int, float, float]:
    return int(float(row["actual_seed"])), float(row["theta"]), float(row["theta_dot"])


def main() -> None:
    args = parse_args()
    if len(args.ordinary_rollouts) != len(args.hard_rollouts):
        raise ValueError("ordinary and hard rollout lists must have matching seeds")
    output_rows: list[dict[str, Any]] = []
    hard_count = 0
    for ordinary_path, hard_path in zip(args.ordinary_rollouts, args.hard_rollouts):
        ordinary = read_rows(Path(ordinary_path))
        hard = {key(row): row for row in read_rows(Path(hard_path))}
        if set(map(key, ordinary)) != set(hard):
            raise ValueError(f"rollout grids differ: {ordinary_path} vs {hard_path}")
        for ordinary_row in ordinary:
            use_hard = abs(float(ordinary_row["theta_degrees"])) >= float(args.threshold_degrees)
            selected = dict(hard[key(ordinary_row)] if use_hard else ordinary_row)
            selected["run_dir"] = "pure_rl_initial_state_qfilter_router"
            selected["router_component"] = "unanimous_advantage" if use_hard else "clipped_value"
            output_rows.append(selected)
            hard_count += int(use_hard)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "threshold_degrees": float(args.threshold_degrees),
        "num_rollouts": len(output_rows),
        "unanimous_rollouts": hard_count,
        "clipped_rollouts": len(output_rows) - hard_count,
        "unanimous_fraction": hard_count / len(output_rows),
        "output": str(output_path),
    }
    (output_path.parent / "router_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
