from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .comparison import compare_images, write_comparison
from .figures import render_all
from .manifest import Finding, verify_manifest
from .metrics import (
    derived_report_payload,
    load_claims,
    verify_claims,
)
from .poster_figures import render_all_poster
from .validation import (
    EvidenceValidationError,
    read_and_validate_rollout,
    validate_cross_method_grid,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output(root: Path) -> Path:
    return root / ".build" / "reproduction"


def safe_output_path(root: Path, requested: Path | None) -> Path:
    """Resolve output and reject any in-repository canonical destination."""

    root = root.resolve()
    output = (requested or default_output(root)).expanduser().resolve()
    if output == root:
        raise ValueError("The project root cannot be used as reproduction output")
    if output.is_relative_to(root):
        relative = output.relative_to(root)
        if not relative.parts or relative.parts[0] != ".build":
            raise ValueError(
                "In-repository reproduction output must be below ROOT/.build; "
                f"refusing canonical or tracked path: {output}"
            )
    return output


def submission_evidence_paths(
    root: Path,
    requested_data_dir: Path | None,
    requested_manifest: Path | None,
) -> tuple[Path, Path | None]:
    """Bind grading redraws to the manifest-protected submission evidence."""

    root = root.resolve()
    expected_data_dir = (root / "data" / "report").resolve()
    data_dir = (
        requested_data_dir.expanduser().resolve()
        if requested_data_dir is not None
        else expected_data_dir
    )
    expected_manifest = (expected_data_dir / "manifest.json").resolve()
    manifest = (
        requested_manifest.expanduser().resolve()
        if requested_manifest is not None
        else None
    )
    if data_dir != expected_data_dir:
        raise ValueError(
            "evaluate/reproduce require the manifest-protected ROOT/data/report "
            "tree; use --root to evaluate another checkout"
        )
    if manifest is not None and manifest != expected_manifest:
        raise ValueError(
            "evaluate/reproduce require ROOT/data/report/manifest.json; use --root "
            "to evaluate another checkout"
        )
    return data_dir, manifest


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
    return findings, frames


def _finding_counts(findings: Sequence[Finding]) -> tuple[int, int]:
    return (
        sum(item.severity == "error" for item in findings),
        sum(item.severity == "warning" for item in findings),
    )


def _verification_payload(findings: Sequence[Finding]) -> dict[str, object]:
    errors, warnings = _finding_counts(findings)
    return {
        "ok": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "findings": [item.to_dict() for item in findings],
    }


def _print_findings(findings: Sequence[Finding]) -> None:
    errors, warnings = _finding_counts(findings)
    for item in findings:
        suffix = f" [{item.artifact}]" if item.artifact else ""
        print(f"{item.severity.upper():7s} {item.code}: {item.message}{suffix}")
    state = "PASS" if errors == 0 else "FAIL"
    detail = f"{errors} errors"
    if warnings:
        detail += f", {warnings} warnings"
    print(f"VERIFY {state} ({detail})")


def _exit_code(findings: Sequence[Finding]) -> int:
    if any(item.severity == "error" for item in findings):
        return 1
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=default_root(), help="Project root")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Curated report data (default: ROOT/data/report; evaluate/reproduce "
            "are bound to that manifest-protected path)"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional hash manifest (evaluate/reproduce are bound to "
            "ROOT/data/report/manifest.json)"
        ),
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Treat a missing hash manifest as an error",
    )
    parser.add_argument("--json", action="store_true", help="Emit exactly one JSON result")


def _add_reproduction_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_arguments(parser)
    parser.add_argument(
        "--target",
        choices=("report", "poster", "all"),
        default="all",
        help="Deliverable figures to rebuild (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Disposable output root (default: ROOT/.build/reproduction)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="last-nine",
        description="Verify and reproduce the evidence used by the final deliverables.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify", help="Verify hashes, schemas, counts, and report claims"
    )
    _add_common_arguments(verify)
    reproduce = subparsers.add_parser(
        "reproduce", help="Verify evidence and independently redraw report/poster figures"
    )
    _add_reproduction_arguments(reproduce)
    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluator shortcut for `reproduce --target all`"
    )
    _add_reproduction_arguments(evaluate)
    figures = subparsers.add_parser(
        "figures", help="Backward-compatible report-figure redraw command"
    )
    _add_common_arguments(figures)
    figures.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report figure directory (default: ROOT/.build/reproduction/report/figures)",
    )
    return parser


def _stage_poster(root: Path, poster_output: Path) -> Path:
    """Create a disposable poster tree before replacing its derived assets."""

    source = root / "poster"
    assets = poster_output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for item in (source / "assets").iterdir():
        if item.is_file():
            shutil.copy2(item, assets / item.name)
    for name in ("poster_visual_v115.html", "tum_logo_template.png"):
        shutil.copy2(source / name, poster_output / name)
    return assets


def _render_report(
    root: Path,
    data_dir: Path,
    frames: dict[str, pd.DataFrame],
    output: Path,
) -> tuple[list[Path], Path, Path]:
    figures = render_all(data_dir, frames, output / "figures")
    metrics_path = output / "derived_metrics.json"
    metrics_path.write_text(
        json.dumps(derived_report_payload(frames), indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = compare_images(figures, root / "report" / "source" / "figures")
    comparison_path = write_comparison(comparison, output / "comparison.json")
    return figures, metrics_path, comparison_path


def _render_poster(
    root: Path,
    data_dir: Path,
    frames: dict[str, pd.DataFrame],
    output: Path,
) -> tuple[list[Path], Path, Path]:
    assets = _stage_poster(root, output)
    figures = render_all_poster(data_dir, frames, assets)
    comparison = compare_images(figures, root / "poster" / "assets")
    comparison_path = write_comparison(comparison, output / "comparison.json")
    return figures, output / "poster_visual_v115.html", comparison_path


def _plain_success(result: dict[str, object]) -> None:
    generated = result.get("generated", {})
    if isinstance(generated, dict) and "report" in generated:
        print(f"REPORT PASS ({len(generated['report'])}/9 independent redraws)")
    if isinstance(generated, dict) and "poster" in generated:
        print(f"POSTER PASS ({len(generated['poster'])}/5 evidence-derived panels)")
    print(f"OUTPUT {result['output']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    reproduction_command = args.command in {"reproduce", "evaluate"}
    if reproduction_command:
        try:
            data_dir, manifest = submission_evidence_paths(
                root, args.data_dir, args.manifest
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        data_dir = (args.data_dir or root / "data" / "report").expanduser().resolve()
        manifest = args.manifest.expanduser().resolve() if args.manifest else None
    require_manifest = bool(args.require_manifest or reproduction_command)
    findings, frames = run_verification(
        root,
        data_dir,
        manifest=manifest,
        require_manifest=require_manifest,
    )
    status = _exit_code(findings)
    result: dict[str, object] = {
        "ok": status == 0,
        "command": args.command,
        "verification": _verification_payload(findings),
    }
    if args.command == "verify" or status != 0:
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_findings(findings)
        return status

    try:
        if args.command == "figures":
            requested = args.output
            if requested is None:
                requested = default_output(root) / "report" / "figures"
            output = safe_output_path(root, requested)
            paths = render_all(data_dir, frames, output)
            result.update(
                {
                    "output": str(output),
                    "target": "report",
                    "generated": {"report": [str(path) for path in paths]},
                }
            )
        else:
            output = safe_output_path(root, args.output)
            target = args.target
            generated: dict[str, list[str]] = {}
            supporting: dict[str, str] = {}
            comparisons: dict[str, str] = {}
            if target in {"report", "all"}:
                figures, metrics, comparison = _render_report(
                    root, data_dir, frames, output / "report"
                )
                generated["report"] = [str(path) for path in figures]
                supporting["derived_metrics"] = str(metrics)
                comparisons["report"] = str(comparison)
            if target in {"poster", "all"}:
                figures, html, comparison = _render_poster(
                    root, data_dir, frames, output / "poster"
                )
                generated["poster"] = [str(path) for path in figures]
                supporting["poster_html"] = str(html)
                comparisons["poster"] = str(comparison)
            result.update(
                {
                    "output": str(output),
                    "target": target,
                    "generated": generated,
                    "supporting": supporting,
                    "comparisons": comparisons,
                }
            )
            summary_path = output / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            result["summary"] = str(summary_path)
            summary_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_findings(findings)
            print(f"REPRODUCE FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_findings(findings)
        _plain_success(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
