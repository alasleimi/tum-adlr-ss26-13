from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "deliverables" / "chasing_nines_20260723"
REPORT_PDF = DELIVERY / "report" / "report.pdf"
REPORT_SOURCE = DELIVERY / "report" / "report.tex"
POSTER_PDF = DELIVERY / "poster" / "poster.pdf"
POSTER_SOURCE = DELIVERY / "poster" / "poster.html"
OUT = DELIVERY / "final_validation.json"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def embedded_fonts(pdf: Path) -> dict[str, object]:
    result = subprocess.run(
        ["pdffonts", str(pdf)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [line for line in result.stdout.splitlines()[2:] if line.strip()]
    unembedded = []
    for line in lines:
        flags = re.search(
            r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$",
            line,
            flags=re.I,
        )
        if flags is None or flags.group(1).lower() != "yes":
            unembedded.append(line)
    return {"font_rows": len(lines), "unembedded_rows": unembedded}


def render_rgb(pdf: Path, scale: float = 0.35) -> np.ndarray:
    document = fitz.open(pdf)
    assert_true(document.page_count == 1, f"Expected one page in {pdf}")
    pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    return array.reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]


def main() -> None:
    report = fitz.open(REPORT_PDF)
    poster = fitz.open(POSTER_PDF)
    report_pages = report.page_count
    poster_pages = poster.page_count
    poster_rect = poster[0].rect

    assert_true(report_pages >= 4, "Report is shorter than four pages.")
    assert_true(poster_pages == 1, "Poster must have exactly one page.")
    assert_true(
        abs(poster_rect.width - 3371.04) < 2.0
        and abs(poster_rect.height - 2385.12) < 2.0,
        f"Unexpected poster dimensions: {poster_rect.width} x {poster_rect.height}",
    )

    report_text = REPORT_SOURCE.read_text(encoding="utf-8")
    poster_text = POSTER_SOURCE.read_text(encoding="utf-8")
    style = {}
    for label, text in (("report", report_text), ("poster", poster_text)):
        style[label] = {
            "em_dash_count": text.count("\N{EM DASH}"),
            "remains_count": len(re.findall(r"\bremains\b", text, flags=re.I)),
            "contrast_template_count": len(
                re.findall(
                    r"(?:it(?:'|’)s not|it is not)\b[^.]{0,120}\bit is\b",
                    text,
                    flags=re.I,
                )
            ),
        }
        assert_true(
            style[label]["em_dash_count"] == 0,
            f"{label} source contains an em dash.",
        )
        assert_true(
            style[label]["remains_count"] == 0,
            f"{label} source contains the discouraged word 'remains'.",
        )
        assert_true(
            style[label]["contrast_template_count"] == 0,
            f"{label} source contains a banned contrast template.",
        )

    report_reviews = sorted((DELIVERY / "reviews").glob("report_review_*.md"))
    poster_reviews = sorted((DELIVERY / "reviews").glob("poster_review_*.md"))
    assert_true(len(report_reviews) >= 3, "Fewer than three report reviews.")
    assert_true(len(poster_reviews) >= 3, "Fewer than three poster reviews.")

    blind_expectations = {
        "geometry_round6": ("B", DELIVERY / "blind_tests" / "geometry_round6" / "B.pdf"),
        "need_round5": ("B", DELIVERY / "blind_tests" / "need_round5" / "B.pdf"),
    }
    current_poster = render_rgb(POSTER_PDF)
    blind = {}
    for name, (expected_winner, tested_pdf) in blind_expectations.items():
        verdict = (
            DELIVERY / "blind_tests" / name / "verdict.md"
        ).read_text(encoding="utf-8")
        match = re.search(r"Winner:\s*\**([AB])", verdict, flags=re.I)
        assert_true(match is not None, f"No winner found in {name} verdict.")
        winner = match.group(1).upper()
        assert_true(winner == expected_winner, f"Unexpected winner in {name}: {winner}")
        tested_poster = render_rgb(tested_pdf)
        pixel_identical = (
            current_poster.shape == tested_poster.shape
            and np.array_equal(current_poster, tested_poster)
        )
        assert_true(
            pixel_identical,
            f"Current poster differs visually from the winning {name} poster.",
        )
        blind[name] = {
            "winner": winner,
            "current_is_pixel_identical_to_tested_poster": pixel_identical,
        }

    with (DELIVERY / "evidence_registry.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        claims = list(csv.DictReader(handle))
    assert_true(len(claims) >= 13, "Claim registry has fewer than 13 claims.")
    assert_true(
        all(row["claim_id"] and row["evidence_tier"] for row in claims),
        "Claim registry has a missing claim ID or tier.",
    )

    with (DELIVERY / "verified_scorecard.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        scorecard = list(csv.DictReader(handle))
    assert_true(len(scorecard) == 7, "Verified scorecard must contain seven methods.")

    report_fonts = embedded_fonts(REPORT_PDF)
    poster_fonts = embedded_fonts(POSTER_PDF)
    assert_true(
        not report_fonts["unembedded_rows"], "Report has unembedded fonts."
    )
    assert_true(
        not poster_fonts["unembedded_rows"], "Poster has unembedded fonts."
    )

    result = {
        "status": "PASS",
        "report": {
            "pages": report_pages,
            "page_size_points": [
                float(report[0].rect.width),
                float(report[0].rect.height),
            ],
            **report_fonts,
        },
        "poster": {
            "pages": poster_pages,
            "page_size_points": [
                float(poster_rect.width),
                float(poster_rect.height),
            ],
            **poster_fonts,
        },
        "style": style,
        "reviews": {
            "report_count": len(report_reviews),
            "poster_count": len(poster_reviews),
        },
        "blind_stopping_test": blind,
        "evidence_registry": {
            "claim_count": len(claims),
            "tier_counts": {
                tier: sum(row["evidence_tier"] == tier for row in claims)
                for tier in ("A", "B", "C")
            },
        },
        "verified_scorecard_method_count": len(scorecard),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
