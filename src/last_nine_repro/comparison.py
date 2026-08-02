from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image

from .manifest import sha256_file


def compare_images(
    generated: Iterable[Path], canonical_dir: Path
) -> dict[str, object]:
    """Describe redraws without treating pixel identity as a correctness gate."""

    records: list[dict[str, object]] = []
    for generated_path in generated:
        canonical_path = canonical_dir / generated_path.name
        if not canonical_path.is_file():
            raise FileNotFoundError(
                f"Canonical comparison image is missing: {canonical_path}"
            )
        with Image.open(generated_path) as image:
            image.load()
            generated_size = list(image.size)
            generated_mode = image.mode
        with Image.open(canonical_path) as image:
            image.load()
            canonical_size = list(image.size)
            canonical_mode = image.mode
        generated_hash = sha256_file(generated_path)
        canonical_hash = sha256_file(canonical_path)
        records.append(
            {
                "name": generated_path.name,
                "generated": {
                    "bytes": generated_path.stat().st_size,
                    "sha256": generated_hash,
                    "pixels": generated_size,
                    "mode": generated_mode,
                },
                "canonical": {
                    "bytes": canonical_path.stat().st_size,
                    "sha256": canonical_hash,
                    "pixels": canonical_size,
                    "mode": canonical_mode,
                },
                "exact_bytes": generated_hash == canonical_hash,
                "same_dimensions": generated_size == canonical_size,
            }
        )
    return {
        "role": "informational_independent_redraw_comparison",
        "gating": False,
        "qualification": (
            "Evidence and claim validation are gating. Pixel identity is not: "
            "font rasterization and plotting-library versions can change image bytes."
        ),
        "images": records,
    }


def write_comparison(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
