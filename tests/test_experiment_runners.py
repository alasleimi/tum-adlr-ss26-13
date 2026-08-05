from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_mixed_report_pipeline",
    ROOT / "experiments" / "run_mixed_report_pipeline.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
followup_command = MODULE.followup_command
initializer_command = MODULE.initializer_command


def _args(
    initializer: Path,
    *,
    dry_run: bool,
    critic: Path | None = None,
    stage: str = "selected",
) -> argparse.Namespace:
    return argparse.Namespace(
        initializer_run=initializer,
        critic_run=critic,
        seed=0,
        dry_run=dry_run,
        stage=stage,
        device="cpu",
    )


def test_mixed_followup_dry_run_can_describe_a_not_yet_built_initializer(
    tmp_path: Path,
) -> None:
    initializer = tmp_path / "planned-initializer"
    command = followup_command(
        _args(initializer, dry_run=True), tmp_path / "selected"
    )
    assert str(initializer) in command


def test_mixed_followup_execution_requires_initializer_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing initializer checkpoint"):
        followup_command(
            _args(tmp_path / "missing-initializer", dry_run=False),
            tmp_path / "selected",
        )


def test_mixed_initializer_preserves_the_recorded_seed_family_split(
    tmp_path: Path,
) -> None:
    seed0 = argparse.Namespace(seed=0, device="cpu")
    seed1 = argparse.Namespace(seed=1, device="cpu")
    command0 = initializer_command(seed0, tmp_path / "seed0")
    command1 = initializer_command(seed1, tmp_path / "seed1")
    option = "--near-upright-fraction"
    assert command0[command0.index(option) + 1] == "0.2"
    assert command1[command1.index(option) + 1] == "0"


def test_mixed_followup_uses_explicit_rebuilt_critic(tmp_path: Path) -> None:
    initializer = tmp_path / "initializer"
    critic = tmp_path / "critic"
    command = followup_command(
        _args(initializer, critic=critic, dry_run=True),
        tmp_path / "selected",
    )
    assert command[command.index("--dagger-run") + 1] == str(initializer)
    assert command[command.index("--rl-run") + 1] == str(critic)


def test_priority_shifted_keeps_priority_starts_and_tiny_target_blend(
    tmp_path: Path,
) -> None:
    command = followup_command(
        _args(
            tmp_path / "initializer",
            critic=tmp_path / "critic",
            dry_run=True,
            stage="priority_shifted",
        ),
        tmp_path / "priority-shifted",
    )
    assert command[command.index("--dagger-initial-mode") + 1] == "priority_uniform"
    assert command[command.index("--rl-blend") + 1] == "0.005"


def test_mixed_smoke_commands_keep_recipe_but_shrink_runtime(tmp_path: Path) -> None:
    initializer_args = argparse.Namespace(seed=0, device="cpu", smoke=True)
    initializer = initializer_command(initializer_args, tmp_path / "initializer")
    dataset_option = max(
        index for index, value in enumerate(initializer) if value == "--dataset-size"
    )
    assert initializer[dataset_option + 1] == "8"
    assert initializer[initializer.index("--near-upright-fraction") + 1] == "0.2"

    followup_args = _args(
        tmp_path / "initializer",
        critic=tmp_path / "critic",
        dry_run=True,
    )
    followup_args.smoke = True
    followup = followup_command(followup_args, tmp_path / "selected")
    static_option = max(
        index for index, value in enumerate(followup) if value == "--static-size"
    )
    assert followup[static_option + 1] == "8"
    assert followup[followup.index("--validation-qsearch-num-actions") + 1] == "5"
