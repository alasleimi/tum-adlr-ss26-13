from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproduce_report_checkpoints",
    ROOT / "experiments" / "reproduce_report_checkpoints.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _plan(models_root: Path):
    return MODULE.build_plan(
        models_root=models_root,
        device="cpu",
        overwrite=False,
        seeds=range(5),
    )


def test_complete_checkpoint_plan_has_every_report_family(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    plan = _plan(models_root)
    by_family: dict[str, set[int]] = {}
    for task in plan:
        by_family.setdefault(task.family, set()).add(task.seed)

    five_seed_families = {
        "pure_selected",
        "pure_onestep",
        "pure_fastsacn8",
        "pure_sacn8",
        "mlp_sac",
        "mlp_fastsacn8",
        "mlp_sacn8",
        "objective_density",
        "canonical_dagger",
        "mixed_base",
        "mixed_selected",
        "mixed_uniform",
        "mixed_priority_shifted",
    }
    assert all(
        by_family[family] == set(range(5)) for family in five_seed_families
    )
    assert by_family["mixed_shared_critic"] == {1}
    assert by_family["objective_none"] == {0}
    assert len(plan) == 67


def test_mixed_followups_depend_on_rebuilt_initializer_and_critic(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "models"
    plan = _plan(models_root)
    positions = {task.key: index for index, task in enumerate(plan)}
    followups = [task for task in plan if task.phase == 1]
    assert len(followups) == 15
    for task in followups:
        assert task.dependencies == (
            "mixed_shared_critic/seed1",
            f"mixed_base/seed{task.seed}",
        )
        assert all(
            positions[dependency] < positions[task.key]
            for dependency in task.dependencies
        )
        command = list(task.command)
        assert command[command.index("--initializer-run") + 1] == str(
            models_root / "mixed_base" / f"seed{task.seed}"
        )
        assert command[command.index("--critic-run") + 1] == str(
            models_root / "mixed_shared_critic" / "seed1"
        )


def test_canonical_dagger_plan_matches_the_frozen_recipe(tmp_path: Path) -> None:
    task = next(
        task
        for task in _plan(tmp_path / "models")
        if task.key == "canonical_dagger/seed0"
    )
    command = list(task.command)
    expected = {
        "--epochs": "40",
        "--initial-expert-episodes": "100",
        "--dagger-iterations": "4",
        "--dagger-episodes-per-iteration": "100",
        "--dagger-train-epochs-per-iteration": "20",
        "--dagger-max-dataset-size": "100000",
        "--dagger-expert-beta-start": "0.75",
        "--dagger-expert-beta-final": "0",
        "--selection-metric": "last",
        "--rollout-backend": "vectorized_pendulum",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value


def test_dry_run_prints_plan_without_creating_output(
    tmp_path: Path, capsys
) -> None:
    models_root = tmp_path / "models"
    assert MODULE.main(
        ["--models-root", str(models_root), "--device", "cpu", "--dry-run"]
    ) == 0
    output = capsys.readouterr().out
    assert '"task": "mixed_shared_critic/seed1"' in output
    assert '"task": "mixed_selected/seed4"' in output
    assert '"task": "mixed_priority_shifted/seed4"' in output
    assert not models_root.exists()
