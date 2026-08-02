from __future__ import annotations

from pathlib import Path

from PIL import Image

from last_nine_repro.comparison import compare_images


def test_comparison_reports_pixel_identity_without_gating_redraws(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    generated = tmp_path / "generated"
    canonical.mkdir()
    generated.mkdir()
    Image.new("RGB", (12, 8), "white").save(canonical / "same.png")
    (generated / "same.png").write_bytes((canonical / "same.png").read_bytes())
    payload = compare_images([generated / "same.png"], canonical)
    assert payload["gating"] is False
    assert payload["images"][0]["exact_bytes"] is True
    Image.new("RGB", (12, 8), "navy").save(generated / "same.png")
    payload = compare_images([generated / "same.png"], canonical)
    assert payload["images"][0]["exact_bytes"] is False
    assert payload["images"][0]["same_dimensions"] is True
