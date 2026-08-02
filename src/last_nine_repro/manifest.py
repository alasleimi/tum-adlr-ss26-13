from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Finding:
    """One machine-readable verification result."""

    severity: str
    code: str
    message: str
    artifact: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_rows(payload: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        for artifact_id, spec in artifacts.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Manifest artifact {artifact_id!r} is not an object")
            yield str(artifact_id), spec
        return
    if isinstance(artifacts, list):
        for index, spec in enumerate(artifacts):
            if not isinstance(spec, dict) or "id" not in spec:
                raise ValueError(f"Manifest artifact row {index} lacks an id")
            yield str(spec["id"]), spec
        return
    raise ValueError("Manifest 'artifacts' must be an object or array")


def _safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(relative)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Manifest path escapes the project root: {relative}")
    return candidate


def verify_manifest(
    root: Path,
    manifest_path: Path | None = None,
    *,
    require: bool = False,
) -> list[Finding]:
    """Verify an optional hash manifest.

    The future manifest may use either a mapping or a list under ``artifacts``.
    Artifact paths are project-root-relative and may declare ``sha256`` and
    ``bytes``. A missing manifest is conspicuous but non-fatal unless requested.
    """

    path = manifest_path or root / "data" / "report" / "manifest.json"
    if not path.is_file():
        severity = "error" if require else "warning"
        return [
            Finding(
                severity,
                "MANIFEST_MISSING",
                f"Hash manifest is not present yet: {path}",
                str(path),
            )
        ]

    try:
        payload = load_json(path)
        rows = list(_artifact_rows(payload))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [Finding("error", "MANIFEST_INVALID", str(exc), str(path))]

    findings: list[Finding] = []
    for artifact_id, spec in rows:
        relative = spec.get("path")
        if not isinstance(relative, str) or not relative:
            findings.append(
                Finding("error", "MANIFEST_PATH_MISSING", "Artifact has no path", artifact_id)
            )
            continue
        try:
            artifact = _safe_path(root, relative)
        except ValueError as exc:
            findings.append(Finding("error", "MANIFEST_PATH_UNSAFE", str(exc), artifact_id))
            continue
        if not artifact.is_file():
            findings.append(
                Finding("error", "ARTIFACT_MISSING", f"Missing file: {relative}", artifact_id)
            )
            continue
        expected_size = spec.get("bytes")
        if expected_size is not None and artifact.stat().st_size != int(expected_size):
            findings.append(
                Finding(
                    "error",
                    "ARTIFACT_SIZE_MISMATCH",
                    f"Expected {expected_size} bytes, found {artifact.stat().st_size}",
                    artifact_id,
                )
            )
        expected_hash = spec.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            findings.append(
                Finding("error", "ARTIFACT_HASH_MISSING", "Missing valid SHA-256", artifact_id)
            )
        else:
            actual_hash = sha256_file(artifact)
            if actual_hash.lower() != expected_hash.lower():
                findings.append(
                    Finding(
                        "error",
                        "ARTIFACT_HASH_MISMATCH",
                        f"Expected {expected_hash}, found {actual_hash}",
                        artifact_id,
                    )
                )
    if not rows:
        findings.append(Finding("error", "MANIFEST_EMPTY", "Manifest contains no artifacts", str(path)))
    return findings
