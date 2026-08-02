from __future__ import annotations

from last_nine_repro.cli import _exit_code, build_parser
from last_nine_repro.manifest import Finding


def test_cli_exposes_three_report_scoped_commands() -> None:
    parser = build_parser()
    for command in ("verify", "figures", "reproduce"):
        args = parser.parse_args([command])
        assert args.command == command


def test_strict_provenance_promotes_warnings_to_nonzero() -> None:
    warnings = [Finding("warning", "KNOWN_CAVEAT", "documented")]
    assert _exit_code(warnings, strict_provenance=False) == 0
    assert _exit_code(warnings, strict_provenance=True) == 2
    errors = [Finding("error", "BROKEN", "invalid")]
    assert _exit_code(errors, strict_provenance=False) == 1
