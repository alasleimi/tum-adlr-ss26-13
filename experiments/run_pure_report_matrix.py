"""Run only the reward-only training families used in the final report."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "report_reproduction" / "models"
DEFAULT_OUTPUT = ROOT / "runs" / "report_reproduction"
REPORT_FAMILIES = (
    "pure_selected",
    "pure_onestep",
    "pure_fastsacn8",
    "pure_sacn8",
    "mlp_sac",
    "mlp_fastsacn8",
    "mlp_sacn8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=REPORT_FAMILIES,
        help="repeat to run several report families; default: pure_selected",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=range(5))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(family: str, seed: int, args: argparse.Namespace) -> tuple[list[str], Path]:
    config = MODELS / family / f"seed{seed}" / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing retained config: {config}")
    output = args.output_root / family / f"seed{seed}"
    cmd = [
        sys.executable,
        "-m",
        "last_nine_rl.train",
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--run-dir",
        str(output),
        "--device",
        args.device,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd, output


def run_one(family: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    cmd, output = command(family, seed, args)
    final = output / "checkpoints" / "final.pt"
    if final.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace completed run: {output}")
    record = {"family": family, "seed": seed, "output": str(output), "command": cmd}
    if args.dry_run:
        return {**record, "status": "dry-run"}
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not final.is_file():
        raise RuntimeError(f"training returned without final checkpoint: {final}")
    return {**record, "status": "complete"}


def main() -> int:
    args = parse_args()
    families = args.family or ["pure_selected"]
    if any(seed not in range(5) for seed in args.seeds):
        raise SystemExit("report seeds are exactly 0, 1, 2, 3, and 4")
    jobs = [(family, seed) for family in families for seed in args.seeds]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, family, seed, args): (family, seed)
            for family, seed in jobs
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
