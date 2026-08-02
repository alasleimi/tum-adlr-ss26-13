from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .figures import render_all
from .manifest import Finding, verify_manifest
from .metrics import (
    derived_report_payload,
    diagnostic_semantic_findings,
    load_claims,
    verify_claims,
)
from .provenance import audit_mixed_provenance
from .validation import (
    EvidenceValidationError,
    read_and_validate_rollout,
    validate_cross_method_grid,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_frames(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[Finding]]:
    frames: dict[str, pd.DataFrame] = {}
    findings: list[Finding] = []
    rollout_dir = data_dir / "rollouts"
    paths = sorted(rollout_dir.glob("*.csv")) if rollout_dir.is_dir() else []
    if not paths:
        return frames, [
            Finding("error", "ROLLOUTS_MISSING", f"No rollout CSV files under {rollout_dir}")
        ]
    for path in paths:
        try:
            frames[path.stem] = read_and_validate_rollout(path, path.stem)
        except EvidenceValidationError as exc:
            findings.append(Finding("error", "ROLLOUT_INVALID", str(exc), path.stem))
    if len(frames) == len(paths):
        try:
            validate_cross_method_grid(frames)
        except EvidenceValidationError as exc:
            findings.append(Finding("error", "CROSS_METHOD_GRID_INVALID", str(exc)))
    return frames, findings


def run_verification(
    root: Path,
    data_dir: Path,
    *,
    manifest: Path | None = None,
    require_manifest: bool = False,
) -> tuple[list[Finding], dict[str, pd.DataFrame]]:
    findings = verify_manifest(root, manifest, require=require_manifest)
    frames, rollout_findings = load_frames(data_dir)
    findings.extend(rollout_findings)
    try:
        claims = load_claims(data_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", "CLAIMS_INVALID", str(exc), "claims.json"))
        claims = {}
    if claims:
        findings.extend(verify_claims(frames, claims))
        findings.extend(diagnostic_semantic_findings(data_dir, claims))
    findings.extend(audit_mixed_provenance(data_dir, frames))
    return findings, frames


def _print_findings(findings: Sequence[Finding], *, as_json: bool) -> None:
    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    if as_json:
        print(
            json.dumps(
                {
                    "ok": errors == 0,
                    "errors": errors,
                    "warnings": warnings,
                    "findings": [item.to_dict() for item in findings],
                },
                indent=2,
            )
        )
        return
    for item in findings:
        suffix = f" [{item.artifact}]" if item.artifact else ""
        print(f"{item.severity.upper():7s} {item.code}: {item.message}{suffix}")
    print(f"Verification summary: {errors} error(s), {warnings} warning(s)")


def _exit_code(findings: Sequence[Finding], strict_provenance: bool) -> int:
    if any(item.severity == "error" for item in findings):
        return 1
    if strict_provenance and any(item.severity == "warning" for item in findings):
        return 2
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=default_root(), help="Project root")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Curated report data (default: ROOT/data/report)",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional hash manifest")
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Treat a missing hash manifest as an error",
    )
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Return nonzero for documented provenance/semantic warnings",
    )
    parser.add_argument("--json", action="store_true", help="Emit verification JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="last-nine",
        description="Verify and reproduce only the evidence used by the final report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Verify hashes, schema, counts, and provenance")
    _add_common_arguments(verify)
    for name, help_text in (
        ("figures", "Regenerate the nine final report figures"),
        ("reproduce", "Verify evidence, regenerate figures, and write derived metrics"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(command)
        command.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Figure output directory (default: ROOT/report/figures/generated)",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    data_dir = (args.data_dir or root / "data" / "report").resolve()
    manifest = args.manifest.resolve() if args.manifest else None
    findings, frames = run_verification(
        root,
        data_dir,
        manifest=manifest,
        require_manifest=args.require_manifest,
    )
    _print_findings(findings, as_json=args.json)
    status = _exit_code(findings, args.strict_provenance)
    if args.command == "verify" or status != 0:
        return status

    output = (args.output or root / "report" / "figures" / "generated").resolve()
    try:
        paths = render_all(data_dir, frames, output)
    except Exception as exc:
        print(f"Figure generation failed: {exc}", file=sys.stderr)
        return 1
    if not args.json:
        print(f"Generated {len(paths)} figures in {output}")
    if args.command == "reproduce":
        metrics_path = output / "derived_metrics.json"
        metrics_path.write_text(
            json.dumps(derived_report_payload(frames), indent=2),
            encoding="utf-8",
        )
        if not args.json:
            print(f"Wrote {metrics_path}")
    return 0
