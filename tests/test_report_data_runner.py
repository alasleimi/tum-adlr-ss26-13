from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


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


def test_smoke_dry_run_reports_the_smoke_output(capsys) -> None:
    assert report_data.main(["--smoke", "--dry-run", "--device", "cpu"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert Path(plan["output"]) == (
        report_data.ROOT / ".build" / "report-data-smoke"
    )


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


def _rollout_row(
    *, return_value: float, gap: float, strict: int, near: int = 1
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "actual_seed": 0,
                "theta": 0.0,
                "theta_dot": 0.0,
                "return": return_value,
                "task_success": 1,
                "near_best_known_return_eps": near,
                "beats_best_known_return": strict,
                "signed_gap_to_best_known": gap,
            }
        ]
    )


def test_rollout_fidelity_accepts_only_threshold_adjacent_float_flip() -> None:
    retained = pd.concat(
        [_rollout_row(return_value=-10.0, gap=0.01, strict=1) for _ in range(25)],
        ignore_index=True,
    )
    retained["theta"] = range(25)
    generated = retained.copy()
    generated.loc[0, ["return", "signed_gap_to_best_known", "beats_best_known_return"]] = [
        -10.02,
        -0.01,
        0,
    ]
    result = report_data.compare_rollout_frame("example", generated, retained)
    assert result["strict_boundary_flips"] == 1
    assert result["return_max_abs_error"] == pytest.approx(0.02)

    generated.loc[0, "signed_gap_to_best_known"] = -1.0
    with pytest.raises(ValueError, match="away from its numeric boundary"):
        report_data.compare_rollout_frame("example", generated, retained)


def test_rollout_fidelity_rejects_material_return_change() -> None:
    retained = _rollout_row(return_value=-10.0, gap=-1.0, strict=0)
    generated = _rollout_row(return_value=-10.5, gap=-1.5, strict=0)
    with pytest.raises(ValueError, match="maximum return drift"):
        report_data.compare_rollout_frame("example", generated, retained)


def test_rollout_fidelity_uses_positive_five_as_near_boundary() -> None:
    retained = pd.concat(
        [
            _rollout_row(return_value=-10.0, gap=4.99, strict=0, near=1)
            for _ in range(25)
        ],
        ignore_index=True,
    )
    retained["theta"] = range(25)
    generated = retained.copy()
    generated.loc[
        0, ["return", "signed_gap_to_best_known", "near_best_known_return_eps"]
    ] = [-10.02, 5.01, 0]
    result = report_data.compare_rollout_frame("example", generated, retained)
    assert result["near_boundary_flips"] == 1


def test_objective_share_uses_each_replay_seed(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        report_data,
        "family_runs",
        lambda models_root, family, seeds=range(5): [
            tmp_path / family / f"seed{seed}" for seed in seeds
        ],
    )
    monkeypatch.setattr(
        report_data,
        "replay_path",
        lambda models_root, replay_root, family, seed, importance: tmp_path
        / f"{importance}-{seed}.npz",
    )
    monkeypatch.setattr(report_data, "run", lambda command: commands.append(command))
    monkeypatch.setattr(report_data, "copy_as", lambda source, destination: None)

    report_data.rebuild_objective_share(
        models_root=tmp_path / "models",
        replay_root=tmp_path / "replay",
        output=tmp_path / "output",
        device="cpu",
    )
    diagnostic_commands = commands[:-1]
    assert [command[command.index("--seed") + 1] for command in diagnostic_commands] == [
        "0",
        "0",
        "1",
        "2",
        "3",
        "4",
    ]


def test_diagnostic_fidelity_checks_all_figure_inputs(tmp_path: Path) -> None:
    required = (
        "actor_geometry/summary.json",
        "critic_direction/summary.json",
        "action_projection/summary.json",
        "reference_recognition/summary.json",
        "objective_share/summary.json",
        "prefix_intervention/aggregate.csv",
        "prefix_intervention/specificity_control.csv",
    )
    for relative in required:
        source = report_data.RETAINED_DATA / "diagnostics" / relative
        destination = tmp_path / "diagnostics" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result = report_data.check_retained_diagnostic_fidelity(tmp_path)
    assert all(value == 0.0 for value in result.values())

    objective_path = tmp_path / "diagnostics" / "objective_share" / "summary.json"
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    objective["density_eight_step_diagnostics_percent"][
        "mean_importance_weight_mean"
    ] += 1.0
    objective_path.write_text(json.dumps(objective), encoding="utf-8")
    with pytest.raises(ValueError, match="objective-share mean diagnostics"):
        report_data.check_retained_diagnostic_fidelity(tmp_path)
