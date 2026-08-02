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


def _args(initializer: Path, *, dry_run: bool) -> argparse.Namespace:
    return argparse.Namespace(
        initializer_run=initializer,
        seed=0,
        dry_run=dry_run,
        stage="selected",
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
