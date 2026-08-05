from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


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


def test_smoke_plan_executes_every_unique_training_path_with_tiny_budgets(
    tmp_path: Path,
) -> None:
    plan = MODULE.build_plan(
        models_root=tmp_path / "models",
        device="cpu",
        overwrite=True,
        seeds=[0],
        smoke=True,
    )
    config = next(task for task in plan if task.key == "pure_sacn8/seed0")
    assert config.command[config.command.index("--total-steps") + 1] == "10"
    assert config.command[config.command.index("--learning-starts") + 1] == "8"

    dagger = next(task for task in plan if task.key == "canonical_dagger/seed0")
    dataset_option = max(
        index for index, value in enumerate(dagger.command) if value == "--dataset-size"
    )
    assert dagger.command[dataset_option + 1] == "8"
    assert dagger.command[dagger.command.index("--dagger-max-episode-steps") + 1] == "4"

    mixed = [task for task in plan if task.family.startswith("mixed_")]
    assert all("--smoke" in task.command for task in mixed if task.family != "mixed_shared_critic")


def test_smoke_defaults_to_ignored_build_tree(capsys) -> None:
    assert MODULE.main(["--smoke", "--seeds", "0", "--dry-run"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(
        Path(record["output"]).is_relative_to(MODULE.SMOKE_MODELS_ROOT)
        for record in records
    )
    assert any("--total-steps" in record["command"] for record in records)


def test_smoke_refuses_output_outside_build_tree(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="must be inside .build"):
        MODULE.main(
            [
                "--smoke",
                "--models-root",
                str(tmp_path / "models"),
                "--dry-run",
            ]
        )
