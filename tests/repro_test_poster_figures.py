from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from last_nine_repro.poster_figures import POSTER_FIGURE_NAMES, render_all_poster
from last_nine_repro.validation import read_and_validate_rollout


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "report"


def test_recovery_trajectory_returns_match_retained_rollouts() -> None:
    archive_path = DATA / "diagnostics" / "recovery_atlas" / "trajectories.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        starts = list(zip(archive["starts_theta_degrees"], archive["starts_theta_dot"]))
        for method, filename in (
            ("mixed", "mixed_selected.csv"),
            ("pure", "pure_selected_q41.csv"),
        ):
            frame = pd.read_csv(DATA / "rollouts" / filename)
            for index, (angle, velocity) in enumerate(starts):
                selected = frame.loc[
                    np.isclose(frame["theta_degrees"], angle)
                    & np.isclose(frame["theta_dot"], velocity)
                ].sort_values("actual_seed")
                assert selected["actual_seed"].tolist() == list(range(5))
                np.testing.assert_allclose(
                    archive[f"{method}_return"][index],
                    selected["return"].to_numpy(),
                    atol=1e-3,
                    rtol=0,
                )


def test_all_five_numerical_poster_panels_render_from_preserved_evidence(
    tmp_path: Path,
) -> None:
    required = {
        "mixed_selected",
        "mixed_actor_only",
        "pure_selected_q41",
        "simba_onestep",
        "canonical_dagger",
    }
    frames = {
        name: read_and_validate_rollout(DATA / "rollouts" / f"{name}.csv")
        for name in required
    }
    paths = render_all_poster(DATA, frames, tmp_path)
    assert {path.name for path in paths} == set(POSTER_FIGURE_NAMES)
    for path in paths:
        assert path.stat().st_size > 10_000
        with Image.open(path) as image:
            generated_size = image.size
            image.verify()
        with Image.open(ROOT / "poster" / "assets" / path.name) as image:
            canonical_size = image.size
        assert all(
            abs(generated - canonical) <= 12
            for generated, canonical in zip(generated_size, canonical_size, strict=True)
        )
