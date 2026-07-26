from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "deliverables" / "chasing_nines_20260723"
REPORT = WORK / "report" / "report.pdf"
POSTER = WORK / "poster" / "poster.pdf"
REPORT_SOURCES = (
    WORK / "report" / "report_compact_body.tex",
    WORK / "report" / "report_annex_current.tex",
)
POSTER_SOURCE = WORK / "poster" / "poster.html"
POSTER_AUDIT = ROOT / "reports" / "plan2507_template_poster_final_audit.json"
VIDEO_DIR = WORK / "videos"
VIDEO_MANIFEST = VIDEO_DIR / "manifest.json"
VIDEOS = (
    VIDEO_DIR / "01_same_hard_start_learning_gap.mp4",
    VIDEO_DIR / "02_pure_rl_qsearch_repairs_failure.mp4",
)
REPORT_LOG = WORK / "report" / "report.log"
EVIDENCE = WORK / "evidence_registry.csv"
MIXED_ROLLOUTS = (
    ROOT
    / "reports"
    / "systematic_100k_budget_best_20260722"
    / "ablation_no_rl_shift_qsearch"
    / "relative"
    / "relative_rollouts.csv"
)
PURE_ROLLOUTS = (
    ROOT
    / "reports"
    / "plan2507_p7_authority_20260725"
    / "relative"
    / "relative_rollouts.csv"
)
BLIND_GATE = ROOT / "reports" / "plan2507_final_blind_gate.json"
POSTER_RUBRIC = ROOT / "reports" / "plan2507_poster_rubric_final" / "verdict.md"
REPORT_RUBRIC = ROOT / "reports" / "plan2507_report_rubric_final" / "verdict.md"
REPORT_BLIND = ROOT / "reports" / "plan2507_report_blind_final" / "verdict.md"
OUTPUT = ROOT / "reports" / "plan2507_final_validation.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(row: dict[str, str], names: tuple[str, ...]) -> bool:
    for name in names:
        if name not in row:
            continue
        value = row[name].strip().lower()
        if value in {"true", "yes"}:
            return True
        if value in {"false", "no"}:
            return False
        return float(value) != 0.0
    raise KeyError(f"none of the columns {names} is present")


def summarize_rollouts(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 12_505, f"{path} has {len(rows)} rows instead of 12,505")
    seed_key = "seed" if "seed" in rows[0] else "actual_seed"
    seeds = sorted({int(row[seed_key]) for row in rows})
    require(seeds == [0, 1, 2, 3, 4], f"{path} has actor seeds {seeds}")
    return {
        "trials": len(rows),
        "near": sum(
            as_bool(
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
            as_bool(row, ("task_success", "task_success_success")) for row in rows
        ),
        "strict": sum(
            as_bool(
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
    require(rows, f"no fonts found in {path}")
    require(
        not any(re.search(r"\sno\s", line) for line in rows),
        f"{path} contains an unembedded font",
    )
    return len(rows)


def main() -> None:
    for path in (
        REPORT,
        POSTER,
        *REPORT_SOURCES,
        POSTER_SOURCE,
        POSTER_AUDIT,
        VIDEO_MANIFEST,
        *VIDEOS,
        REPORT_LOG,
        EVIDENCE,
        MIXED_ROLLOUTS,
        PURE_ROLLOUTS,
        BLIND_GATE,
        POSTER_RUBRIC,
        REPORT_RUBRIC,
        REPORT_BLIND,
    ):
        require(path.is_file(), f"required delivery artifact is missing: {path}")

    report = fitz.open(REPORT)
    poster = fitz.open(POSTER)
    require(report.page_count >= 6, "report must include references and annex")
    main_text = "\n".join(report[index].get_text() for index in range(4))
    page_five = report[4].get_text().strip()
    page_six = report[5].get_text().strip()
    require("References" not in main_text, "references enter the four-page body")
    require(
        "Evidence and diagnostics annex" not in main_text,
        "annex enters the four-page body",
    )
    require(page_five.startswith("References"), "references do not begin on page 5")
    require(
        "Evidence and diagnostics annex" in page_six,
        "annex does not begin on page 6",
    )

    require(poster.page_count == 1, "poster is not one page")
    poster_rect = poster[0].rect
    require(
        abs(poster_rect.width - 3371.04) < 2
        and abs(poster_rect.height - 2385.12) < 2,
        f"poster is not A0 landscape: {poster_rect.width} by {poster_rect.height}",
    )
    poster_audit = json.loads(POSTER_AUDIT.read_text(encoding="utf-8"))
    require(poster_audit["pass"], "template poster audit did not pass")
    require(
        poster_audit["orientation"] == "landscape"
        and not poster_audit["undersized_text"],
        "poster violates orientation or minimum-type template requirements",
    )

    video_manifest = json.loads(VIDEO_MANIFEST.read_text(encoding="utf-8"))
    for video in VIDEOS:
        metadata = video_manifest["videos"][video.name]
        require(metadata["sha256"] == sha256(video), f"video hash changed: {video.name}")
        require(metadata["codec"] == "h264", f"video codec is not H.264: {video.name}")
        require(
            metadata["width"] == 1920
            and metadata["height"] == 1080
            and metadata["duration_seconds"] > 0,
            f"video dimensions or duration are invalid: {video.name}",
        )

    active_source = "\n".join(
        [path.read_text(encoding="utf-8") for path in REPORT_SOURCES]
        + [POSTER_SOURCE.read_text(encoding="utf-8")]
    )
    forbidden = {
        "em dash": "—",
        "vague remains": r"\bremains\b",
        "contrast template": r"\bit(?:'s| is) not\b[^.\n]{0,120}\bit(?:'s| is)\b",
        "false mixed 88 percent": r"\b88(?:\.0+)?%",
    }
    for label, pattern in forbidden.items():
        require(
            re.search(pattern, active_source, flags=re.IGNORECASE) is None,
            f"forbidden construction appears in active source: {label}",
        )

    report_text = "\n".join(page.get_text() for page in report)
    poster_text = poster[0].get_text()
    combined_pdf_text = report_text + "\n" + poster_text
    for token in ("12,496", "99.928%", "12,008", "96.026%", "11,614", "2,854", "497"):
        require(token in combined_pdf_text, f"verified final token is missing: {token}")
    for token in ("11,832", "94.618%", "673 failures"):
        require(token not in combined_pdf_text, f"obsolete pure-RL token appears: {token}")

    build_log = REPORT_LOG.read_text(encoding="utf-8", errors="replace")
    require("Overfull \\hbox" not in build_log, "report has an overfull hbox")
    require("Overfull \\vbox" not in build_log, "report has an overfull vbox")

    mixed = summarize_rollouts(MIXED_ROLLOUTS)
    pure = summarize_rollouts(PURE_ROLLOUTS)
    require(
        mixed == {"trials": 12_505, "near": 12_496, "task": 11_737, "strict": 1_570},
        f"mixed evidence changed: {mixed}",
    )
    require(
        pure == {"trials": 12_505, "near": 12_008, "task": 11_614, "strict": 2_854},
        f"pure evidence changed: {pure}",
    )

    with EVIDENCE.open(newline="", encoding="utf-8-sig") as handle:
        evidence_rows = list(csv.DictReader(handle))
    evidence_ids = {row["claim_id"] for row in evidence_rows}
    require(
        {"C39", "C40", "C41", "C42", "C43"}.issubset(evidence_ids),
        "plan2507 evidence claims C39 through C43 are incomplete",
    )
    for row in evidence_rows:
        for field in ("raw_artifact", "generation_code"):
            for value in row[field].split(";"):
                target = ROOT / value.strip()
                require(target.exists(), f"{row['claim_id']} points to missing {target}")

    blind_gate = json.loads(BLIND_GATE.read_text(encoding="utf-8"))
    require(blind_gate["poster_sha256"] == sha256(POSTER), "blind gate used another poster")
    comparisons = blind_gate["comparisons"]
    require(len(comparisons) == 6, "blind gate must contain six comparisons")
    require(
        sum(item["reference"] == "3DPRAC" for item in comparisons) == 3
        and sum(item["reference"] == "NEED" for item in comparisons) == 3,
        "blind gate does not contain three judgments per reference",
    )
    blind_votes = {
        reference: sum(
            item["reference"] == reference and item["project_poster_preferred"]
            for item in comparisons
        )
        for reference in ("3DPRAC", "NEED")
    }
    blind_preference_pass = all(value >= 2 for value in blind_votes.values())

    rubric_text = POSTER_RUBRIC.read_text(encoding="utf-8")
    report_rubric_text = REPORT_RUBRIC.read_text(encoding="utf-8")
    report_blind_text = REPORT_BLIND.read_text(encoding="utf-8")
    require("PASS" in rubric_text.upper(), "poster rubric review did not pass")
    require("PASS" in report_rubric_text.upper(), "report rubric review did not pass")
    require(
        "Preferred report: A" in report_blind_text,
        "updated report did not beat the frozen report",
    )

    payload = {
        "status": (
            "PASS"
            if blind_preference_pass
            else "TECHNICAL_PASS_BLIND_PREFERENCE_UNMET"
        ),
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
            "page_size_points": [poster_rect.width, poster_rect.height],
            "sha256": sha256(POSTER),
            "font_rows": check_fonts(POSTER),
            "template_audit": poster_audit,
        },
        "videos": video_manifest["videos"],
        "mainline": {"mixed": mixed, "pure": pure},
        "blind_gate": blind_gate,
        "blind_preference_pass": blind_preference_pass,
        "blind_preference_votes": blind_votes,
        "evidence_claims": sorted(evidence_ids),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
