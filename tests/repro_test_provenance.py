from __future__ import annotations

from pathlib import Path

from last_nine_repro.provenance import audit_mixed_provenance, normalize_run_path
from last_nine_repro.validation import read_and_validate_rollout


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "report"


def test_windows_absolute_and_relative_run_paths_normalize_identically() -> None:
    absolute = r"C:\Users\researcher\Project\runs\family\seed0"
    relative = r"runs\family\seed0"
    assert normalize_run_path(absolute) == normalize_run_path(relative)


def test_known_mixed_lineage_and_contrast_caveats_are_detected() -> None:
    frame = read_and_validate_rollout(DATA / "rollouts" / "mixed_base_q5.csv")
    findings = audit_mixed_provenance(DATA, {"mixed_base_q5": frame})
    codes = {item.code for item in findings}
    assert "PROV_MIXED_SEED_LINEAGE_SPLIT" in codes
    assert "PROV_MIXED_ANGLE_TARGETED_INITIALIZER" in codes
    assert "PROV_MIXED_CONTRAST_CAVEAT" in codes
    assert not {item.code for item in findings if item.severity == "error"}
