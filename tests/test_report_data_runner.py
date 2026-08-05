from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproduce_report_data",
    ROOT / "experiments" / "reproduce_report_data.py",
)
assert SPEC is not None and SPEC.loader is not None
report_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report_data
SPEC.loader.exec_module(report_data)


def test_report_data_plan_covers_every_figure_rollout_recipe() -> None:
    assert set(report_data.FIGURE_METHODS) == {
        "mixed_selected",
        "mixed_uniform",
        "mixed_actor_only",
        "pure_selected_q41",
        "simba_onestep",
        "canonical_dagger",
    }
    assert report_data.ROLLOUT_RECIPES["mixed_selected"].policy == "mixed_local_q5"
    assert report_data.ROLLOUT_RECIPES["mixed_uniform"].policy == "mixed_local_q5"
    assert report_data.ROLLOUT_RECIPES["pure_selected_q41"].policy == "pure_global_q41"
    assert report_data.ROLLOUT_RECIPES["pure_actor"].family == "pure_selected"


def test_report_data_smoke_grid_exercises_all_policies_cheaply() -> None:
    theta, velocity, horizon, seeds = report_data.state_grid(smoke=True)
    assert theta.shape == (2,)
    assert velocity.shape == (2,)
    assert horizon == 2
    assert list(seeds) == [0]


def test_report_data_prefers_rebuilt_replay_then_retained_fallback(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    replay = tmp_path / "replay"
    local = models / "objective_density" / "seed2" / "replay_final.npz"
    retained = replay / "objective_share" / "density" / "seed2" / "replay_final.npz"
    retained.parent.mkdir(parents=True)
    retained.touch()
    assert report_data.replay_path(
        models, replay, "objective_density", 2, "density"
    ) == retained
    local.parent.mkdir(parents=True)
    local.touch()
    assert report_data.replay_path(
        models, replay, "objective_density", 2, "density"
    ) == local
