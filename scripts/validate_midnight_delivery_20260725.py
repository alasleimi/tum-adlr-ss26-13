from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "deliverables" / "chasing_nines_20260723"
REPORT = DELIVERY / "report" / "report.pdf"
POSTER = DELIVERY / "poster" / "poster.pdf"
REPORT_SOURCE = DELIVERY / "report" / "report_compact_body.tex"
ANNEX_SOURCE = DELIVERY / "report" / "report_annex_current.tex"
POSTER_SOURCE = DELIVERY / "poster" / "poster.html"
REPORT_LOG = DELIVERY / "report" / "report.log"
SCORECARD = DELIVERY / "verified_scorecard.csv"
EVIDENCE_REGISTRY = DELIVERY / "evidence_registry.csv"
MIXED_ROWS = (
    ROOT
    / "reports"
    / "systematic_100k_budget_best_20260722"
    / "ablation_no_rl_shift_qsearch"
    / "relative"
    / "relative_rollouts.csv"
)
PURE_ROWS = (
    ROOT
    / "reports"
    / "pure_rl_plus1pp_20260719"
    / "authority_simba100k_symmetric_actor_q41m005_unanimous_relative"
    / "relative_rollouts.csv"
)
TERMINAL_FASTSACN = (
    ROOT
    / "reports"
    / "plan2307_pure_target_architecture_20260723"
    / "terminal_offgrid_20260725"
    / "terminal_fastsacn_gate_summary.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(row: dict[str, str], candidates: tuple[str, ...]) -> bool:
    for key in candidates:
        if key in row:
            value = row[key].strip().lower()
            if value in {"true", "yes"}:
                return True
            if value in {"false", "no"}:
                return False
            return float(value) != 0.0
    raise KeyError(f"none of {candidates} exists")


def summarize_rollouts(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 12_505, f"{path} does not contain 12,505 rows")
    seed_key = "seed" if "seed" in rows[0] else "actual_seed"
    seeds = sorted({int(row[seed_key]) for row in rows})
    require(seeds == [0, 1, 2, 3, 4], f"{path} does not contain seeds 0 through 4")
    return {
        "trials": len(rows),
        "near": sum(
            bool_value(
                row,
                (
                    "near_best_known_return_eps",
                    "near_reference",
                    "near_best_known_return_eps_success",
                ),
            )
            for row in rows
        ),
        "task": sum(
            bool_value(row, ("task_success", "task_success_success")) for row in rows
        ),
        "strict": sum(
            bool_value(
                row,
                (
                    "beats_best_known_return",
                    "strictly_beats_reference",
                    "beats_best_known_return_success",
                ),
            )
            for row in rows
        ),
    }


def check_fonts(path: Path) -> int:
    completed = subprocess.run(
        ["pdffonts", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    rows = [line for line in completed.stdout.splitlines()[2:] if line.strip()]
    require(rows, f"pdffonts returned no font rows for {path}")
    require(
        not any(re.search(r"\sno\s", line) for line in rows),
        f"{path} contains an unembedded font",
    )
    return len(rows)


def main() -> None:
    require(REPORT.is_file(), "final report is missing")
    require(POSTER.is_file(), "final poster is missing")

    report = fitz.open(REPORT)
    poster = fitz.open(POSTER)
    require(report.page_count >= 6, "report is missing references or annex")
    first_four = "\n".join(report[index].get_text() for index in range(4))
    page_five = report[4].get_text().strip()
    page_six = report[5].get_text().strip()
    require("References" not in first_four, "references enter the four-page body")
    require(
        "Evidence and diagnostics annex" not in first_four,
        "annex enters the four-page body",
    )
    require(page_five.startswith("References"), "references do not begin on page 5")
    require(
        "Evidence and diagnostics annex" in page_six,
        "diagnostics annex does not begin on page 6",
    )

    require(poster.page_count == 1, "poster must have one page")
    rect = poster[0].rect
    require(
        abs(rect.width - 3371.04) < 2 and abs(rect.height - 2385.12) < 2,
        f"poster is not landscape A0: {rect.width} by {rect.height}",
    )

    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (REPORT_SOURCE, ANNEX_SOURCE)
    )
    poster_text = POSTER_SOURCE.read_text(encoding="utf-8")
    forbidden = {
        "em dash": "—",
        "remains": r"\bremains\b",
        "contrast template": r"\bit(?:'s| is) not\b[^.\n]{0,100}\bit(?:'s| is)\b",
        "false 88% mixed story": r"\b88(?:\.0+)?%",
        "obsolete ablation value": r"(?<!\d)-239(?!\d)",
    }
    for label, pattern in forbidden.items():
        flags = 0 if label == "em dash" else re.IGNORECASE
        require(
            re.search(pattern, source_text + "\n" + poster_text, flags) is None,
            f"forbidden {label} appears in an active source",
        )

    require(
        "Overfull \\hbox" not in REPORT_LOG.read_text(encoding="utf-8", errors="replace")
        and "Overfull \\vbox"
        not in REPORT_LOG.read_text(encoding="utf-8", errors="replace"),
        "report build contains an overfull box",
    )

    mixed = summarize_rollouts(MIXED_ROWS)
    pure = summarize_rollouts(PURE_ROWS)
    require(
        mixed == {"trials": 12_505, "near": 12_496, "task": 11_737, "strict": 1_570},
        f"mixed evidence changed: {mixed}",
    )
    require(
        pure == {"trials": 12_505, "near": 11_832, "task": 11_567, "strict": 2_303},
        f"pure evidence changed: {pure}",
    )

    terminal = json.loads(TERMINAL_FASTSACN.read_text(encoding="utf-8"))
    protocol = terminal["protocol"]
    require(
        protocol["authority_grid_queried"] is False,
        "terminal promotion gate queried the authority grid",
    )
    require(
        len(
            {
                protocol["p0_state_sha256"],
                protocol["p7_state_sha256"],
                protocol["p8_state_sha256"],
            }
        )
        == 1,
        "terminal arms used different locked state sets",
    )
    require(
        terminal["p0"]["selected_variant"]["seeds"] == 5
        and terminal["p7"]["selected_variant"]["seeds"] == 5
        and terminal["p8_seed0"]["selected_variant"]["seeds"] == 1,
        "terminal FastSACN gate has the wrong seed counts",
    )

    with SCORECARD.open(newline="", encoding="utf-8-sig") as handle:
        scorecard = list(csv.DictReader(handle))
    require(len(scorecard) == 7, "verified scorecard must contain seven methods")
    require(
        all(int(row["trials"]) == 12_505 for row in scorecard),
        "every scorecard row must contain 12,505 trials",
    )

    with EVIDENCE_REGISTRY.open(newline="", encoding="utf-8-sig") as handle:
        evidence = list(csv.DictReader(handle))
    require(evidence, "evidence registry is empty")
    for row in evidence:
        for field in ("raw_artifact", "generation_code"):
            for value in row[field].split(";"):
                path = ROOT / value.strip()
                require(path.exists(), f"missing evidence path for {row['claim_id']}: {path}")
    tier_counts = {
        tier: sum(row["evidence_tier"] == tier for row in evidence)
        for tier in ("A", "B", "C")
    }

    report_reviews = sorted((DELIVERY / "reviews").glob("midnight_report_review_*.md"))
    poster_reviews = sorted((DELIVERY / "reviews").glob("midnight_poster_review_*.md"))
    require(len(report_reviews) == 3, "three midnight report reviews are required")
    require(len(poster_reviews) == 3, "three midnight poster reviews are required")

    report_verdict = (
        DELIVERY
        / "blind_tests"
        / "report_terminal_final_vs_27of30"
        / "verdict.md"
    )
    need_verdict = (
        DELIVERY / "blind_tests" / "poster_terminal_final_vs_need" / "verdict.md"
    )
    geometry_verdict = (
        DELIVERY / "blind_tests" / "poster_terminal_final_vs_3d" / "verdict.md"
    )
    require("Preferred report: **A**" in report_verdict.read_text(encoding="utf-8"), "report did not win blind test")
    require("Preferred poster: **A**" in need_verdict.read_text(encoding="utf-8"), "poster did not beat NEED")
    require("Preferred poster: **B**" in geometry_verdict.read_text(encoding="utf-8"), "poster did not beat 3DPRAC")

    payload = {
        "status": "PASS",
        "report": {
            "pages": report.page_count,
            "main_pages": 4,
            "references_page": 5,
            "annex_start_page": 6,
            "sha256": sha256(REPORT),
            "font_rows": check_fonts(REPORT),
        },
        "poster": {
            "pages": poster.page_count,
            "page_size_points": [rect.width, rect.height],
            "sha256": sha256(POSTER),
            "font_rows": check_fonts(POSTER),
        },
        "mainline": {"mixed": mixed, "pure": pure},
        "terminal_fastsacn_gate": {
            "authority_grid_queried": False,
            "p7_promoted": terminal["decision"][
                "promote_p7_to_authority_evaluation"
            ],
            "p8_promoted": terminal["decision"]["promote_p8_to_four_more_seeds"],
            "state_sha256": protocol["p0_state_sha256"],
        },
        "evidence_registry": {
            "claims": len(evidence),
            "tier_counts": tier_counts,
            "all_paths_exist": True,
        },
        "reviews": {"report": len(report_reviews), "poster": len(poster_reviews)},
        "blind_tests": {
            "report_vs_frozen": "win",
            "poster_vs_need": "win",
            "poster_vs_3dprac": "win",
        },
        "style_checks": {label: "absent" for label in forbidden},
    }
    output = DELIVERY / "final_validation_20260725.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
