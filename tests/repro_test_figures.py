from __future__ import annotations

from pathlib import Path

from last_nine_repro.figures import FIGURE_NAMES, render_all
from last_nine_repro.validation import read_and_validate_rollout


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "report"


def test_all_nine_report_figures_render_from_curated_evidence(tmp_path: Path) -> None:
    required = {
        "mixed_selected",
        "mixed_uniform",
        "mixed_actor_only",
        "pure_selected_q41",
        "simba_onestep",
        "canonical_dagger",
    }
    frames = {
        name: read_and_validate_rollout(DATA / "rollouts" / f"{name}.csv")
        for name in required
    }
    paths = render_all(DATA, frames, tmp_path)
    assert {path.name for path in paths} == set(FIGURE_NAMES)
    assert all(path.stat().st_size > 10_000 for path in paths)
