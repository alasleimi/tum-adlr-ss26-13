from __future__ import annotations

import json
from pathlib import Path

from last_nine_repro.manifest import sha256_file, verify_manifest


def test_missing_manifest_is_visible_but_optional(tmp_path: Path) -> None:
    findings = verify_manifest(tmp_path)
    assert [(item.severity, item.code) for item in findings] == [
        ("warning", "MANIFEST_MISSING")
    ]
    required = verify_manifest(tmp_path, require=True)
    assert required[0].severity == "error"


def test_manifest_binds_size_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "data" / "report" / "sample.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("preserved evidence\n", encoding="utf-8")
    manifest = tmp_path / "data" / "report" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "sample": {
                        "path": "data/report/sample.txt",
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert verify_manifest(tmp_path) == []
    artifact.write_text("changed evidence\n", encoding="utf-8")
    codes = {item.code for item in verify_manifest(tmp_path)}
    assert "ARTIFACT_HASH_MISMATCH" in codes
    assert "ARTIFACT_SIZE_MISMATCH" in codes
