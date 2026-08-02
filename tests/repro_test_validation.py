from __future__ import annotations

from pathlib import Path

import pytest

from last_nine_repro.metrics import diagnostic_semantic_findings, load_claims, summarize_method
from last_nine_repro.validation import (
    EvidenceValidationError,
    read_rollouts,
    validate_rollout_schema,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "report"


@pytest.fixture(scope="module")
def mixed_frame():
    return read_rollouts(DATA / "rollouts" / "mixed_selected.csv")


def test_selected_mixed_claims_are_derived_from_12505_rows(mixed_frame) -> None:
    spec = validate_rollout_schema(mixed_frame, "mixed_selected")
    summary = summarize_method(mixed_frame)
    assert spec.rows == 12_505
    assert summary.near == 12_496
    assert summary.task == 11_737
    assert summary.strict == 1_570
    assert summary.failures == 9
    assert summary.failure_cells == 8


def test_duplicate_and_missing_grid_cell_fail_even_with_same_row_count(mixed_frame) -> None:
    broken = mixed_frame.copy()
    coordinate_columns = ["actual_seed", "theta", "theta_degrees", "theta_dot"]
    broken.loc[broken.index[-1], coordinate_columns] = broken.loc[
        broken.index[0], coordinate_columns
    ].to_numpy()
    with pytest.raises(EvidenceValidationError, match="duplicate seed-state"):
        validate_rollout_schema(broken, "broken")


def test_metric_semantics_are_checked(mixed_frame) -> None:
    broken = mixed_frame.copy()
    broken.loc[broken.index[0], "near_best_known_return_eps"] = (
        1 - int(broken.loc[broken.index[0], "near_best_known_return_eps"])
    )
    with pytest.raises(EvidenceValidationError, match="near-reference flag"):
        validate_rollout_schema(broken, "broken")


def test_c32_selector_mismatch_is_computed_not_hardcoded() -> None:
    findings = diagnostic_semantic_findings(DATA, load_claims(DATA))
    assert [(item.severity, item.code) for item in findings] == [
        ("warning", "C32_SEMANTIC_MISMATCH")
    ]
