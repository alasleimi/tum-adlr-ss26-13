from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


TRIALS = 12_505
SEEDS = frozenset(range(5))
ANGLES = 61
VELOCITIES = 41
CELLS = ANGLES * VELOCITIES
RETURN_EPSILON = 5.0

REQUIRED_COLUMNS = frozenset(
    {
        "run_dir",
        "actual_seed",
        "theta",
        "theta_degrees",
        "theta_dot",
        "return",
        "length",
        "stability_success",
        "streak_success",
        "task_success",
        "best_known_return",
        "beats_best_known_return",
        "near_best_known_return_eps",
        "signed_gap_to_best_known",
    }
)
BINARY_COLUMNS = (
    "stability_success",
    "streak_success",
    "task_success",
    "beats_best_known_return",
    "near_best_known_return_eps",
)
NUMERIC_COLUMNS = (
    "actual_seed",
    "theta",
    "theta_degrees",
    "theta_dot",
    "return",
    "length",
    "best_known_return",
    "signed_gap_to_best_known",
    *BINARY_COLUMNS,
)


class EvidenceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GridSpec:
    rows: int
    seeds: tuple[int, ...]
    angles: tuple[float, ...]
    velocities: tuple[float, ...]


def read_rollouts(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise EvidenceValidationError(f"Rollout file is missing: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pandas supplies useful parser context
        raise EvidenceValidationError(f"Could not parse {path}: {exc}") from exc


def canonical_grid(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_theta_key"] = result["theta"].astype(float).round(12)
    result["_velocity_key"] = result["theta_dot"].astype(float).round(12)
    theta_rank = {value: index for index, value in enumerate(sorted(result["_theta_key"].unique()))}
    velocity_rank = {
        value: index for index, value in enumerate(sorted(result["_velocity_key"].unique()))
    }
    result["_theta_index"] = result["_theta_key"].map(theta_rank).astype(int)
    result["_velocity_index"] = result["_velocity_key"].map(velocity_rank).astype(int)
    return result


def _require_close(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-9, equal_nan=False):
        difference = float(np.max(np.abs(actual - expected)))
        raise EvidenceValidationError(f"{message}; maximum absolute difference is {difference:g}")


def validate_rollout_schema(frame: pd.DataFrame, artifact_id: str = "rollouts") -> GridSpec:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise EvidenceValidationError(f"{artifact_id}: missing columns {missing}")
    if len(frame) != TRIALS:
        raise EvidenceValidationError(f"{artifact_id}: expected {TRIALS:,} rows, found {len(frame):,}")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise EvidenceValidationError(f"{artifact_id}: {column} contains non-finite values")

    seeds_numeric = pd.to_numeric(frame["actual_seed"], errors="raise").to_numpy(dtype=float)
    if not np.equal(seeds_numeric, np.floor(seeds_numeric)).all():
        raise EvidenceValidationError(f"{artifact_id}: actual_seed is not integral")
    seeds = frozenset(seeds_numeric.astype(int).tolist())
    if seeds != SEEDS:
        raise EvidenceValidationError(f"{artifact_id}: expected seeds {sorted(SEEDS)}, found {sorted(seeds)}")

    for column in BINARY_COLUMNS:
        values = frozenset(pd.to_numeric(frame[column], errors="raise").astype(float).unique())
        if not values.issubset({0.0, 1.0}):
            raise EvidenceValidationError(f"{artifact_id}: {column} is not binary: {sorted(values)}")

    if not (pd.to_numeric(frame["length"]) == 200).all():
        raise EvidenceValidationError(f"{artifact_id}: every report rollout must have length 200")

    canonical = canonical_grid(frame)
    angles = tuple(sorted(canonical["_theta_key"].unique().tolist()))
    velocities = tuple(sorted(canonical["_velocity_key"].unique().tolist()))
    if len(angles) != ANGLES or len(velocities) != VELOCITIES:
        raise EvidenceValidationError(
            f"{artifact_id}: expected {ANGLES}x{VELOCITIES} grid, found "
            f"{len(angles)}x{len(velocities)}"
        )
    key_columns = ["actual_seed", "_theta_index", "_velocity_index"]
    if canonical.duplicated(key_columns).any():
        raise EvidenceValidationError(f"{artifact_id}: duplicate seed-state grid rows")
    per_seed = canonical.groupby("actual_seed", sort=True).size()
    if not (per_seed == CELLS).all():
        raise EvidenceValidationError(
            f"{artifact_id}: rows per seed are {per_seed.astype(int).to_dict()}, expected {CELLS}"
        )

    theta = pd.to_numeric(frame["theta"], errors="raise").to_numpy(dtype=float)
    theta_degrees = pd.to_numeric(frame["theta_degrees"], errors="raise").to_numpy(dtype=float)
    _require_close(theta_degrees, np.degrees(theta), f"{artifact_id}: theta degrees disagree")

    gap = pd.to_numeric(frame["signed_gap_to_best_known"], errors="raise").to_numpy(dtype=float)
    best = pd.to_numeric(frame["best_known_return"], errors="raise").to_numpy(dtype=float)
    returns = pd.to_numeric(frame["return"], errors="raise").to_numpy(dtype=float)
    _require_close(gap, best - returns, f"{artifact_id}: signed-gap semantics disagree")

    near = pd.to_numeric(frame["near_best_known_return_eps"], errors="raise").to_numpy(dtype=int)
    strict = pd.to_numeric(frame["beats_best_known_return"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(near, (gap <= RETURN_EPSILON).astype(int)):
        raise EvidenceValidationError(f"{artifact_id}: near-reference flag is not gap <= 5")
    if not np.array_equal(strict, (gap < 0.0).astype(int)):
        raise EvidenceValidationError(f"{artifact_id}: strict-win flag is not gap < 0")

    stability = pd.to_numeric(frame["stability_success"], errors="raise").to_numpy(dtype=int)
    streak = pd.to_numeric(frame["streak_success"], errors="raise").to_numpy(dtype=int)
    task = pd.to_numeric(frame["task_success"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(task, stability & streak):
        raise EvidenceValidationError(f"{artifact_id}: task_success is not stability AND streak")

    for optional, expected in (
        ("regret_to_best_known", np.maximum(gap, 0.0)),
        ("advantage_over_best_known", np.maximum(-gap, 0.0)),
    ):
        if optional in frame:
            actual = pd.to_numeric(frame[optional], errors="raise").to_numpy(dtype=float)
            _require_close(actual, expected, f"{artifact_id}: {optional} semantics disagree")

    comparator_counts = canonical.groupby(["_theta_index", "_velocity_index"])[
        "best_known_return"
    ].nunique(dropna=False)
    if not (comparator_counts == 1).all():
        raise EvidenceValidationError(f"{artifact_id}: comparator changes across actor seeds")

    return GridSpec(len(frame), tuple(sorted(seeds)), angles, velocities)


def read_and_validate_rollout(path: Path, artifact_id: str | None = None) -> pd.DataFrame:
    frame = read_rollouts(path)
    validate_rollout_schema(frame, artifact_id or path.stem)
    return frame


def validate_cross_method_grid(frames: Mapping[str, pd.DataFrame]) -> None:
    if not frames:
        raise EvidenceValidationError("No rollout frames were supplied")
    reference_name, reference_frame = next(iter(frames.items()))
    reference = canonical_grid(reference_frame).sort_values(
        ["actual_seed", "_theta_index", "_velocity_index"]
    )
    reference_keys = reference[["actual_seed", "_theta_index", "_velocity_index"]].to_numpy()
    reference_best = reference["best_known_return"].to_numpy(dtype=float)
    for name, frame in list(frames.items())[1:]:
        candidate = canonical_grid(frame).sort_values(
            ["actual_seed", "_theta_index", "_velocity_index"]
        )
        keys = candidate[["actual_seed", "_theta_index", "_velocity_index"]].to_numpy()
        if not np.array_equal(keys, reference_keys):
            raise EvidenceValidationError(f"{name}: grid differs from {reference_name}")
        _require_close(
            candidate["best_known_return"].to_numpy(dtype=float),
            reference_best,
            f"{name}: comparator differs from {reference_name}",
        )
