"""Re-run the report's mixed initializer or one 20k follow-up variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "report_reproduction" / "models"
IMPLEMENTATION = ROOT / "experiments" / "implementations" / "train_pendulum_qregularized_dagger.py"
DP_SOLUTION = ROOT / "data" / "reference" / "pendulum_dp_solution.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("initializer", "selected", "uniform"))
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--initializer-run",
        type=Path,
        help="defaults to the retained mixed_base checkpoint for the same seed",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def initializer_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "last_nine_rl.distill_reference",
        "--run-dir",
        str(output),
        "--policy",
        "best",
        "--dp-solution-path",
        str(DP_SOLUTION),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--actor-hidden-dim",
        "64",
        "--actor-blocks",
        "2",
        "--dataset-size",
        "400000",
        "--eval-dataset-size",
        "50000",
        "--batch-size",
        "1024",
        "--epochs",
        "80",
        "--lr",
        "0.0003",
        "--velocity-limit",
        "8",
        "--reset-support-fraction",
        "0.6",
        "--reset-support-velocity-limit",
        "1",
        "--near-down-fraction",
        "0",
        "--near-upright-fraction",
        "0.2",
        "--near-upright-abs-theta-high-deg",
        "35",
        "--selection-metric",
        "eval_action_mae",
        "--dagger-iterations",
        "3",
        "--dagger-episodes-per-iteration",
        "50",
        "--dagger-train-epochs-per-iteration",
        "10",
        "--dagger-rollout-mode",
        "deterministic",
        "--rollout-backend",
        "vectorized_pendulum",
    ]


def followup_command(args: argparse.Namespace, output: Path) -> list[str]:
    initializer = args.initializer_run or MODELS / "mixed_base" / f"seed{args.seed}"
    critic = MODELS / "mixed_shared_critic" / "seed1"
    if not (initializer / "checkpoints" / "final.pt").is_file():
        raise FileNotFoundError(f"missing initializer checkpoint: {initializer}")
    if not (critic / "checkpoints" / "final.pt").is_file():
        raise FileNotFoundError(f"missing shared critic checkpoint: {critic}")
    priority = args.stage == "selected"
    return [
        sys.executable,
        str(IMPLEMENTATION),
        "--dagger-run",
        str(initializer),
        "--rl-run",
        str(critic),
        "--run-dir",
        str(output),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--dp-solution",
        str(DP_SOLUTION),
        "--static-size",
        "240000",
        "--static-target-policy",
        "reference",
        "--broad-fraction",
        "0",
        "--reset-support-fraction",
        "1",
        "--hard120-fraction",
        "0",
        "--near-down-fraction",
        "0",
        "--dagger-rounds",
        "1",
        "--dagger-episodes",
        "100",
        "--dagger-initial-mode",
        "priority_uniform" if priority else "standard_reset",
        "--priority-candidate-multiplier",
        "40",
        "--priority-fraction",
        "0.9",
        "--epochs-per-round",
        "3",
        "--lr",
        "0.00001",
        "--rl-margin",
        "0.01",
        "--rl-blend",
        "0" if priority else "0.005",
        "--max-target-shift",
        "0.02",
        "--selected-weight",
        "1",
        "--trainable-actor",
        "all",
        "--targeted-validation-size",
        "5001",
        "--targeted-validation-mode",
        "reset_uniform",
        "--validation-every-epochs",
        "3",
        "--validation-theta-bins",
        "47",
        "--validation-velocity-bins",
        "31",
        "--validation-qsearch-radius",
        "0.1",
        "--validation-qsearch-num-actions",
        "5",
        "--validation-qsearch-margin",
        "0",
    ]


def main() -> int:
    args = parse_args()
    default = ROOT / "runs" / "report_reproduction" / f"mixed_{args.stage}" / f"seed{args.seed}"
    output = (args.run_dir or default).resolve()
    cmd = initializer_command(args, output) if args.stage == "initializer" else followup_command(args, output)
    if args.overwrite:
        cmd.append("--overwrite")
    record = {"stage": args.stage, "seed": args.seed, "output": str(output), "command": cmd}
    print(json.dumps(record, indent=2))
    if args.dry_run:
        return 0
    final = output / "checkpoints" / "final.pt"
    if final.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace completed run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not final.is_file():
        raise RuntimeError(f"pipeline returned without final checkpoint: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
