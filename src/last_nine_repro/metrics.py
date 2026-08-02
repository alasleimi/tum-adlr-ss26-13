from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .manifest import Finding, load_json
from .validation import TRIALS, canonical_grid


@dataclass(frozen=True)
class MethodMetrics:
    trials: int
    near: int
    task: int
    strict: int
    failures: int
    failure_cells: int
    mean_return: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def summarize_method(frame: pd.DataFrame) -> MethodMetrics:
    near = int(frame["near_best_known_return_eps"].sum())
    failures = len(frame) - near
    canonical = canonical_grid(frame)
    per_cell = (
        1 - canonical["near_best_known_return_eps"].astype(int)
    ).groupby([canonical["_theta_index"], canonical["_velocity_index"]]).sum()
    return MethodMetrics(
        trials=len(frame),
        near=near,
        task=int(frame["task_success"].sum()),
        strict=int(frame["beats_best_known_return"].sum()),
        failures=failures,
        failure_cells=int((per_cell > 0).sum()),
        mean_return=float(frame["return"].mean()),
    )


def summarize_seed_ranges(frame: pd.DataFrame) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for name, column in (
        ("near", "near_best_known_return_eps"),
        ("task", "task_success"),
        ("strict", "beats_best_known_return"),
    ):
        counts = frame.groupby("actual_seed", sort=True)[column].sum().astype(int)
        result[name] = (int(counts.min()), int(counts.max()))
    return result


def pair_methods(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["actual_seed", "_theta_index", "_velocity_index"]
    metrics = [
        "return",
        "near_best_known_return_eps",
        "task_success",
        "beats_best_known_return",
    ]
    left_canonical = canonical_grid(left)[keys + metrics]
    right_canonical = canonical_grid(right)[keys + metrics]
    paired = left_canonical.merge(
        right_canonical,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if len(paired) != TRIALS:
        raise ValueError(f"Expected {TRIALS:,} paired rows, found {len(paired):,}")
    return paired


def paired_transition_counts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
) -> dict[str, int]:
    paired = pair_methods(left, right)
    baseline = paired[f"{metric}_left"].astype(bool)
    changed = paired[f"{metric}_right"].astype(bool)
    fixed = int((~baseline & changed).sum())
    broken = int((baseline & ~changed).sum())
    return {"fixed": fixed, "broken": broken, "net": fixed - broken}


def tolerance_curve(frame: pd.DataFrame, epsilons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for _seed, seed_frame in frame.groupby("actual_seed", sort=True):
        gaps = seed_frame["signed_gap_to_best_known"].to_numpy(dtype=float)
        curves.append(np.asarray([(gaps <= epsilon).mean() for epsilon in epsilons]))
    seed_curves = np.stack(curves)
    return seed_curves, seed_curves.mean(axis=0)


def load_claims(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "claims.json"
    if not path.is_file():
        raise FileNotFoundError(f"Report claim ledger is missing: {path}")
    return load_json(path)


def verify_claims(
    frames: Mapping[str, pd.DataFrame],
    claims: Mapping[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    methods = claims.get("methods", {})
    if not isinstance(methods, dict):
        return [Finding("error", "CLAIMS_INVALID", "'methods' must be an object", "claims.json")]
    for method, expected in methods.items():
        if method not in frames:
            findings.append(
                Finding("error", "CLAIM_SOURCE_MISSING", f"No rollout source for {method}", method)
            )
            continue
        actual = summarize_method(frames[method]).to_dict()
        for field, expected_value in expected.items():
            if field in {"label", "role"}:
                continue
            if field not in actual:
                findings.append(
                    Finding("error", "CLAIM_FIELD_UNKNOWN", f"Unknown claim field {field}", method)
                )
                continue
            if isinstance(expected_value, bool) or not isinstance(expected_value, (int, float)):
                findings.append(
                    Finding("error", "CLAIM_VALUE_INVALID", f"Invalid expected value for {field}", method)
                )
                continue
            actual_value = actual[field]
            equal = (
                int(actual_value) == int(expected_value)
                if field != "mean_return"
                else np.isclose(actual_value, expected_value, rtol=0.0, atol=1e-12)
            )
            if not equal:
                findings.append(
                    Finding(
                        "error",
                        "REPORT_CLAIM_MISMATCH",
                        f"{field}: expected {expected_value}, derived {actual_value}",
                        method,
                    )
                )
    return findings


def _condition_series(payload: Mapping[str, Any], conditions: list[str], field: str) -> np.ndarray:
    return np.asarray(
        [float(payload["conditions"][condition]["pooled"][field]) for condition in conditions],
        dtype=float,
    )


def diagnostic_semantic_findings(data_dir: Path, claims: Mapping[str, Any]) -> list[Finding]:
    """Detect documented label/selector disagreements without embedding values."""

    specification = claims.get("diagnostic_semantics", {}).get("c32")
    if not isinstance(specification, dict):
        return [
            Finding(
                "error",
                "C32_SEMANTICS_MISSING",
                "claims.json does not define the C32 semantic check",
                "claims.json",
            )
        ]
    source = data_dir / str(specification["source"])
    try:
        payload = load_json(source)
        conditions = [str(item) for item in specification["conditions"]]
        figure = _condition_series(payload, conditions, str(specification["figure_field"]))
        prose = _condition_series(payload, conditions, str(specification["prose_field"]))
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return [Finding("error", "C32_SEMANTICS_INVALID", str(exc), str(source))]
    if np.allclose(figure, prose, rtol=0.0, atol=1e-12):
        return [
            Finding(
                "error",
                "C32_MISMATCH_NOT_REPRODUCED",
                "The two documented C32 selectors unexpectedly agree",
                str(source),
            )
        ]
    return [
        Finding(
            "warning",
            "C32_SEMANTIC_MISMATCH",
            (
                f"Figure uses {specification['figure_field']!r}, while report prose describes "
                f"{specification['prose_field']!r}; values are recomputed from the diagnostic JSON."
            ),
            str(source),
        )
    ]


def derived_report_payload(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        "protocol": {"seeds": 5, "angles": 61, "velocities": 41, "trials": TRIALS},
        "methods": {name: summarize_method(frame).to_dict() for name, frame in frames.items()},
    }
