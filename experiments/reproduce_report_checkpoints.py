"""Rebuild every checkpoint family used by the report from its recorded seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
RETAINED_MODELS = ROOT / "artifacts" / "report_reproduction" / "models"
DEFAULT_MODELS_ROOT = ROOT / "runs" / "report_reproduction" / "models"
MIXED_RUNNER = ROOT / "experiments" / "run_mixed_report_pipeline.py"
DP_SOLUTION = ROOT / "data" / "reference" / "pendulum_dp_solution.npz"

CONFIG_FIVE_SEED_FAMILIES = (
    "pure_selected",
    "pure_onestep",
    "pure_fastsacn8",
    "pure_sacn8",
    "mlp_sac",
    "mlp_fastsacn8",
    "mlp_sacn8",
    "objective_density",
)
MIXED_FOLLOWUPS = (
    ("selected", "mixed_selected"),
    ("uniform", "mixed_uniform"),
    ("priority_shifted", "mixed_priority_shifted"),
)


@dataclass(frozen=True)
class Task:
    key: str
    phase: int
    family: str
    seed: int
    output: Path
    command: tuple[str, ...]
    dependencies: tuple[str, ...] = ()

    def record(self, status: str) -> dict[str, object]:
        return {
            "task": self.key,
            "phase": self.phase,
            "family": self.family,
            "seed": self.seed,
            "output": str(self.output),
            "dependencies": list(self.dependencies),
            "command": list(self.command),
            "status": status,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="output model tree (default: runs/report_reproduction/models)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="actor seeds to rebuild (default: 0 1 2 3 4)",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every resolved command in dependency order without running it",
    )
    return parser.parse_args(argv)


def _task_key(family: str, seed: int) -> str:
    return f"{family}/seed{seed}"


def _with_overwrite(command: list[str], overwrite: bool) -> tuple[str, ...]:
    if overwrite:
        command.append("--overwrite")
    return tuple(command)


def config_task(
    family: str,
    seed: int,
    *,
    models_root: Path,
    device: str,
    overwrite: bool,
) -> Task:
    config = RETAINED_MODELS / family / f"seed{seed}" / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing retained config: {config}")
    output = models_root / family / f"seed{seed}"
    command = [
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
        device,
    ]
    return Task(
        key=_task_key(family, seed),
        phase=0,
        family=family,
        seed=seed,
        output=output,
        command=_with_overwrite(command, overwrite),
    )


def canonical_dagger_task(
    seed: int,
    *,
    models_root: Path,
    device: str,
    overwrite: bool,
) -> Task:
    output = models_root / "canonical_dagger" / f"seed{seed}"
    command = [
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
        str(seed),
        "--device",
        device,
        "--actor-hidden-dim",
        "32",
        "--actor-blocks",
        "1",
        "--dataset-size",
        "20000",
        "--eval-dataset-size",
        "30000",
        "--batch-size",
        "1024",
        "--epochs",
        "40",
        "--lr",
        "0.0003",
        "--velocity-limit",
        "8",
        "--initial-dataset-source",
        "expert_rollout",
        "--initial-expert-episodes",
        "100",
        "--selection-metric",
        "last",
        "--dagger-iterations",
        "4",
        "--dagger-episodes-per-iteration",
        "100",
        "--dagger-train-epochs-per-iteration",
        "20",
        "--dagger-max-dataset-size",
        "100000",
        "--dagger-rollout-mode",
        "deterministic",
        "--dagger-expert-beta-start",
        "0.75",
        "--dagger-expert-beta-final",
        "0",
        "--rollout-backend",
        "vectorized_pendulum",
    ]
    return Task(
        key=_task_key("canonical_dagger", seed),
        phase=0,
        family="canonical_dagger",
        seed=seed,
        output=output,
        command=_with_overwrite(command, overwrite),
    )


def mixed_initializer_task(
    seed: int,
    *,
    models_root: Path,
    device: str,
    overwrite: bool,
) -> Task:
    output = models_root / "mixed_base" / f"seed{seed}"
    command = [
        sys.executable,
        str(MIXED_RUNNER),
        "initializer",
        "--seed",
        str(seed),
        "--device",
        device,
        "--run-dir",
        str(output),
    ]
    return Task(
        key=_task_key("mixed_base", seed),
        phase=0,
        family="mixed_base",
        seed=seed,
        output=output,
        command=_with_overwrite(command, overwrite),
    )


def mixed_followup_task(
    stage: str,
    family: str,
    seed: int,
    *,
    models_root: Path,
    device: str,
    overwrite: bool,
) -> Task:
    initializer = models_root / "mixed_base" / f"seed{seed}"
    critic = models_root / "mixed_shared_critic" / "seed1"
    output = models_root / family / f"seed{seed}"
    dependencies = (
        _task_key("mixed_shared_critic", 1),
        _task_key("mixed_base", seed),
    )
    command = [
        sys.executable,
        str(MIXED_RUNNER),
        stage,
        "--seed",
        str(seed),
        "--device",
        device,
        "--run-dir",
        str(output),
        "--initializer-run",
        str(initializer),
        "--critic-run",
        str(critic),
    ]
    return Task(
        key=_task_key(family, seed),
        phase=1,
        family=family,
        seed=seed,
        output=output,
        command=_with_overwrite(command, overwrite),
        dependencies=dependencies,
    )


def build_plan(
    *,
    models_root: Path,
    device: str,
    overwrite: bool,
    seeds: Sequence[int],
) -> list[Task]:
    requested_seeds = sorted(set(int(seed) for seed in seeds))
    if not requested_seeds or any(seed not in range(5) for seed in requested_seeds):
        raise ValueError("report actor seeds are exactly 0, 1, 2, 3, and 4")

    tasks = [
        config_task(
            "mixed_shared_critic",
            1,
            models_root=models_root,
            device=device,
            overwrite=overwrite,
        )
    ]
    for family in CONFIG_FIVE_SEED_FAMILIES:
        for seed in requested_seeds:
            tasks.append(
                config_task(
                    family,
                    seed,
                    models_root=models_root,
                    device=device,
                    overwrite=overwrite,
                )
            )
    if 0 in requested_seeds:
        tasks.append(
            config_task(
                "objective_none",
                0,
                models_root=models_root,
                device=device,
                overwrite=overwrite,
            )
        )
    for seed in requested_seeds:
        tasks.append(
            canonical_dagger_task(
                seed,
                models_root=models_root,
                device=device,
                overwrite=overwrite,
            )
        )
        tasks.append(
            mixed_initializer_task(
                seed,
                models_root=models_root,
                device=device,
                overwrite=overwrite,
            )
        )
    for stage, family in MIXED_FOLLOWUPS:
        for seed in requested_seeds:
            tasks.append(
                mixed_followup_task(
                    stage,
                    family,
                    seed,
                    models_root=models_root,
                    device=device,
                    overwrite=overwrite,
                )
            )
    return tasks


def run_task(task: Task, *, overwrite: bool) -> dict[str, object]:
    final = task.output / "checkpoints" / "final.pt"
    if final.is_file() and not overwrite:
        return task.record("already-complete")
    task.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(task.command, cwd=ROOT, check=True)
    if not final.is_file():
        raise RuntimeError(f"training returned without final checkpoint: {final}")
    return task.record("complete")


def run_plan(
    tasks: Sequence[Task],
    *,
    workers: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for phase in sorted({task.phase for task in tasks}):
        phase_tasks = [task for task in tasks if task.phase == phase]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_task, task, overwrite=overwrite): task
                for task in phase_tasks
            }
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    return records


def _resolve_models_root(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    artifacts = (ROOT / "artifacts").resolve()
    if path == artifacts or path.is_relative_to(artifacts):
        raise ValueError("checkpoint rebuild output cannot be inside tracked artifacts/")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        models_root = _resolve_models_root(args.models_root)
        tasks = build_plan(
            models_root=models_root,
            device=args.device,
            overwrite=bool(args.overwrite),
            seeds=args.seeds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        for task in tasks:
            print(json.dumps(task.record("dry-run"), sort_keys=True))
        return 0

    records = run_plan(
        tasks,
        workers=int(args.workers),
        overwrite=bool(args.overwrite),
    )
    complete = sum(record["status"] == "complete" for record in records)
    existing = sum(record["status"] == "already-complete" for record in records)
    print(
        json.dumps(
            {
                "models_root": str(models_root),
                "tasks": len(records),
                "complete": complete,
                "already_complete": existing,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
