from __future__ import annotations

from pathlib import Path

import pytest

from last_nine_repro.cli import (
    _exit_code,
    build_parser,
    safe_output_path,
    submission_evidence_paths,
)
from last_nine_repro.manifest import Finding


def test_cli_exposes_evaluator_and_compatible_commands() -> None:
    parser = build_parser()
    for command in ("verify", "figures", "reproduce", "evaluate"):
        args = parser.parse_args([command])
        assert args.command == command
    assert parser.parse_args(["reproduce"]).target == "all"
    assert parser.parse_args(["reproduce", "--target", "poster"]).target == "poster"


def test_output_guard_keeps_repository_writes_under_build(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    assert safe_output_path(root, root / ".build" / "reproduction") == (
        root / ".build" / "reproduction"
    ).resolve()
    external = tmp_path / "external"
    assert safe_output_path(root, external) == external.resolve()
    for unsafe in (root, root / "report", root / "poster", root / "data" / "report"):
        with pytest.raises(ValueError, match="cannot|must be below"):
            safe_output_path(root, unsafe)


def test_submission_redraws_are_bound_to_manifest_protected_evidence(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "project").resolve()
    expected_data = root / "data" / "report"
    expected_manifest = expected_data / "manifest.json"
    assert submission_evidence_paths(root, None, None) == (expected_data, None)
    assert submission_evidence_paths(
        root, expected_data, expected_manifest
    ) == (expected_data, expected_manifest)
    with pytest.raises(ValueError, match="manifest-protected"):
        submission_evidence_paths(root, tmp_path / "other-data", None)
    with pytest.raises(ValueError, match="require ROOT/data/report/manifest.json"):
        submission_evidence_paths(root, None, tmp_path / "other-manifest.json")


def test_warnings_do_not_fail_verification() -> None:
    warnings = [Finding("warning", "KNOWN_CAVEAT", "documented")]
    assert _exit_code(warnings) == 0
    errors = [Finding("error", "BROKEN", "invalid")]
    assert _exit_code(errors) == 1
