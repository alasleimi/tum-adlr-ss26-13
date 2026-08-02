"""Create or verify the SHA-256 manifest for retained report artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "report" / "manifest.json"
SCOPES = (
    ROOT / "artifacts" / "report_reproduction",
    ROOT / "data" / "reference",
    ROOT / "data" / "report",
    ROOT / "report" / "source",
    ROOT / "poster",
    ROOT / "presentation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory() -> list[dict[str, object]]:
    files: set[Path] = set()
    for scope in SCOPES:
        if scope.exists():
            files.update(
                path
                for path in scope.rglob("*")
                if path.is_file() and path.resolve() != OUTPUT.resolve()
            )
    return [
        {
            "id": path.relative_to(ROOT).as_posix(),
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    ]


def payload() -> dict[str, object]:
    entries = inventory()
    return {
        "schema": "last-nine-artifact-manifest/v1",
        "hash": "sha256",
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "artifacts": entries,
    }


def check() -> int:
    if not OUTPUT.exists():
        print(f"ERROR missing manifest: {OUTPUT.relative_to(ROOT)}")
        return 1
    expected = json.loads(OUTPUT.read_text(encoding="utf-8"))
    actual = payload()
    if expected != actual:
        expected_rows = {item["path"]: item for item in expected.get("artifacts", [])}
        actual_rows = {item["path"]: item for item in actual["artifacts"]}
        changed = sorted(
            path for path in expected_rows.keys() | actual_rows.keys()
            if expected_rows.get(path) != actual_rows.get(path)
        )
        print(f"ERROR artifact manifest differs in {len(changed)} path(s)")
        for path in changed[:20]:
            print(f"  {path}")
        return 1
    print(
        f"OK {actual['file_count']} files, {actual['total_bytes']} bytes, "
        "all SHA-256 values match"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()
    if args.check:
        return check()
    document = payload()
    OUTPUT.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)} with {document['file_count']} files "
        f"and {document['total_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
