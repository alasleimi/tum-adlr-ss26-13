"""Rebuild every input used by the nine report figures from retained models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from last_nine_repro.comparison import compare_images, write_comparison
from last_nine_repro.figures import render_all
from last_nine_repro.metrics import derived_report_payload
from last_nine_repro.validation import (
    read_and_validate_rollout,
    validate_cross_method_grid,
)
from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.envs import UprightDetector
from last_nine_rl.hybrid_qsearch import FixedLocalCriticQSearchPolicy
from last_nine_rl.pendulum_grid import (
    CriticSearchPolicy,
    rollout_pendulum_grid_vectorized,
)
from last_nine_rl.pendulum_relative import enrich_rollouts, read_csv, write_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "artifacts" / "report_reproduction" / "models"
DEFAULT_REPLAY = ROOT / "artifacts" / "report_reproduction" / "replay"
DEFAULT_OUTPUT = ROOT / ".build" / "report-data"
RETAINED_DATA = ROOT / "data" / "report"
IMPLEMENTATIONS = ROOT / "experiments" / "implementations"
PROTOCOLS = ROOT / "experiments" / "protocols"
DP_GRID = ROOT / "data" / "reference" / "pendulum_dp_grid.csv"
CONTROLLER_GRID = ROOT / "data" / "reference" / "controller_grid.csv"
DP_SOLUTION = ROOT / "data" / "reference" / "pendulum_dp_solution.npz"

# A 200-step rollout can amplify device-level rounding, particularly when a
# Q-search action lies close to an acceptance boundary. These limits are still
# much smaller than the report's return tolerance of 5.0, while avoiding brittle
# bit-for-bit comparisons across CPU/GPU and library versions.
ROLLOUT_RETURN_MAX_ABS_ERROR = 0.25
ROLLOUT_RETURN_MEAN_ABS_ERROR = 0.001
ROLLOUT_CLASSIFICATION_BOUNDARY = 0.25
DIAGNOSTIC_RATE_ABS_ERROR = 0.005


@dataclass(frozen=True)
class RolloutRecipe:
    family: str
    policy: str


# These seven tables are the complete rollout dependency set for the nine
# figures. pure_actor is consumed by the reference-prefix diagnostic.
ROLLOUT_RECIPES: dict[str, RolloutRecipe] = {
    "mixed_selected": RolloutRecipe("mixed_selected", "mixed_local_q5"),
    "mixed_uniform": RolloutRecipe("mixed_uniform", "mixed_local_q5"),
    "mixed_actor_only": RolloutRecipe("mixed_selected", "actor"),
    "pure_selected_q41": RolloutRecipe("pure_selected", "pure_global_q41"),
    "simba_onestep": RolloutRecipe("pure_onestep", "actor"),
    "canonical_dagger": RolloutRecipe("canonical_dagger", "actor"),
    "pure_actor": RolloutRecipe("pure_selected", "actor"),
}
FIGURE_METHODS = (
    "mixed_selected",
    "mixed_uniform",
    "mixed_actor_only",
    "pure_selected_q41",
    "simba_onestep",
    "canonical_dagger",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Rebuild the data without drawing the nine figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the complete plan without evaluating models.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run every deployment policy for two steps on four states, then stop.",
    )
    return parser.parse_args(argv)


def family_runs(models_root: Path, family: str, seeds: range = range(5)) -> list[Path]:
    runs = [models_root / family / f"seed{seed}" for seed in seeds]
    for run in runs:
        for relative in (Path("config.json"), Path("checkpoints") / "final.pt"):
            path = run / relative
            if not path.is_file():
                raise FileNotFoundError(f"missing report model input: {path}")
    return runs


def replay_path(
    models_root: Path,
    replay_root: Path,
    family: str,
    seed: int,
    importance: str,
) -> Path:
    local = models_root / family / f"seed{seed}" / "replay_final.npz"
    retained = replay_root / "objective_share" / importance / f"seed{seed}" / "replay_final.npz"
    path = local if local.is_file() else retained
    if not path.is_file():
        raise FileNotFoundError(f"missing objective-share replay: {path}")
    return path


def validate_inputs(models_root: Path, replay_root: Path) -> None:
    families = {
        recipe.family for recipe in ROLLOUT_RECIPES.values()
    } | {"pure_fastsacn8", "pure_sacn8", "objective_density"}
    for family in sorted(families):
        family_runs(models_root, family)
    family_runs(models_root, "mixed_shared_critic", range(1, 2))
    family_runs(models_root, "objective_none", range(1))
    replay_path(models_root, replay_root, "objective_none", 0, "none")
    for seed in range(5):
        replay_path(models_root, replay_root, "objective_density", seed, "density")
    for path in (DP_GRID, CONTROLLER_GRID, DP_SOLUTION):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen reference input: {path}")


def state_grid(smoke: bool) -> tuple[np.ndarray, np.ndarray, int, range]:
    theta = np.linspace(-math.pi, math.pi, 61, endpoint=False, dtype=np.float64)
    velocity = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    if smoke:
        theta = theta[:2]
        velocity = velocity[:2]
        return theta, velocity, 2, range(1)
    return theta, velocity, 200, range(5)


def make_policy(
    recipe: RolloutRecipe,
    actor: Any,
    shared_critic: Any | None,
) -> Any:
    if recipe.policy == "actor":
        return actor
    if recipe.policy == "mixed_local_q5":
        if shared_critic is None:
            raise ValueError("mixed local Q-search requires the retained shared critic")
        return FixedLocalCriticQSearchPolicy(
            actor_agent=actor,
            critic_agent=shared_critic,
            num_actions=5,
            margin=0.0,
            search_radius=0.1,
        )
    if recipe.policy == "pure_global_q41":
        return CriticSearchPolicy(
            actor,
            num_actions=41,
            margin=0.005,
            filter_mode="symmetric_actor_unanimous_advantage",
            blend_fraction=1.0,
            max_action_delta=4.0,
        )
    raise ValueError(f"unknown rollout policy: {recipe.policy}")


def rebuild_rollout(
    name: str,
    recipe: RolloutRecipe,
    *,
    models_root: Path,
    output: Path,
    device: str,
    smoke: bool,
) -> Path:
    theta_values, velocity_values, horizon, seeds = state_grid(smoke)
    grid_theta = np.tile(theta_values, len(velocity_values))
    grid_velocity = np.repeat(velocity_values, len(theta_values))
    critic = None
    if recipe.policy == "mixed_local_q5":
        critic_run = family_runs(models_root, "mixed_shared_critic", range(1, 2))[0]
        critic, _critic_config, _payload = load_agent_from_run(
            critic_run, device=device
        )

    rows: list[dict[str, Any]] = []
    for run in family_runs(models_root, recipe.family, seeds):
        actor, config, _payload = load_agent_from_run(run, device=device)
        detector = UprightDetector(
            "Pendulum-v1",
            cos_threshold=config.reliability.near_upright_cos_threshold,
            abs_velocity_threshold=(
                config.reliability.near_upright_abs_velocity_threshold
            ),
        )
        policy = make_policy(recipe, actor, critic)
        evaluated = rollout_pendulum_grid_vectorized(
            policy,
            grid_theta,
            grid_velocity,
            detector,
            config.reliability,
            horizon=horizon,
        )
        for theta, velocity, result in zip(
            grid_theta, grid_velocity, evaluated, strict=True
        ):
            rows.append(
                {
                    "run_dir": str(run),
                    "actual_seed": int(config.seed),
                    "theta": float(theta),
                    "theta_degrees": float(np.degrees(theta)),
                    "theta_dot": float(velocity),
                    **result,
                }
            )

    enriched = enrich_rollouts(
        rows,
        read_csv(DP_GRID),
        read_csv(CONTROLLER_GRID),
        epsilon_return=5.0,
    )
    path = output / "rollouts" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, enriched)
    print(f"ROLLOUT {name}: {len(enriched):,} rows", flush=True)
    return path


def command_environment() -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(ROOT / "src"), str(IMPLEMENTATIONS)]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=command_environment(), check=True)


def runtime_protocols(models_root: Path, output: Path) -> tuple[Path, Path]:
    work = output / ".work"
    work.mkdir(parents=True, exist_ok=True)
    critic_spec = json.loads(
        (PROTOCOLS / "plan2307_p0_p1_p2_action_gradient_diagnostic_20260724.json").read_text(
            encoding="utf-8"
        )
    )
    condition_families = ("pure_onestep", "pure_fastsacn8", "pure_sacn8")
    for condition, family in zip(
        critic_spec["conditions"], condition_families, strict=True
    ):
        condition["actor_runs"] = [str(path) for path in family_runs(models_root, family)]
        condition["critic_runs"] = "same"
    critic_path = work / "critic_protocol.json"
    critic_path.write_text(json.dumps(critic_spec, indent=2) + "\n", encoding="utf-8")

    prefix_spec = json.loads(
        (PROTOCOLS / "plan2507_p7_reference_prefix_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    prefix_spec["dp_solution"] = str(DP_SOLUTION)
    condition = prefix_spec["conditions"][0]
    condition["relative_rollouts"] = str(output / "rollouts" / "pure_actor.csv")
    condition["actor_runs"] = [
        str(path) for path in family_runs(models_root, "pure_selected")
    ]
    prefix_path = work / "prefix_protocol.json"
    prefix_path.write_text(json.dumps(prefix_spec, indent=2) + "\n", encoding="utf-8")
    return critic_path, prefix_path


def copy_as(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"diagnostic did not produce {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def rebuild_objective_share(
    *,
    models_root: Path,
    replay_root: Path,
    output: Path,
    device: str,
) -> None:
    target = output / "diagnostics" / "objective_share"
    work = output / ".work" / "objective_share"
    target.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    jobs = [("objective_none", "none", 0, [1, 2, 4, 8])]
    jobs.extend(
        ("objective_density", "density", seed, [1, 2, 4, 8] if seed == 0 else [8])
        for seed in range(5)
    )
    for family, importance, seed, horizons in jobs:
        run_dir = family_runs(models_root, family, range(seed, seed + 1))[0]
        out_dir = work / importance / f"seed{seed}"
        run(
            [
                sys.executable,
                str(IMPLEMENTATIONS / "diagnose_sacn_horizon_weights.py"),
                "--run",
                str(run_dir),
                "--replay",
                str(replay_path(models_root, replay_root, family, seed, importance)),
                "--out",
                str(out_dir),
                "--horizons",
                *(str(value) for value in horizons),
                "--samples",
                "4096",
                "--batch-size",
                "256",
                "--seed",
                str(seed),
                "--device",
                device,
            ]
        )
        named = target / (
            "none_seed0.json" if importance == "none" else f"density_seed{seed}.json"
        )
        copy_as(out_dir / "sacn_horizon_summary.json", named)
        generated.append(named)

    run(
        [
            sys.executable,
            str(IMPLEMENTATIONS / "build_plan2307_fastsacn_objective_share_20260725.py"),
            "--no-importance-source",
            str(generated[0]),
            "--density-sources",
            *(str(path) for path in generated[1:]),
            "--output-dir",
            str(target),
        ]
    )


def rebuild_diagnostics(
    *,
    models_root: Path,
    replay_root: Path,
    output: Path,
    device: str,
) -> None:
    diagnostic = output / "diagnostics"
    critic_protocol, prefix_protocol = runtime_protocols(models_root, output)

    actor_geometry = diagnostic / "actor_geometry"
    run(
        [
            sys.executable,
            str(IMPLEMENTATIONS / "diagnose_plan2307_completed_target_actor_geometry_20260724.py"),
            "--models-root",
            str(models_root),
            "--output-dir",
            str(actor_geometry),
            "--device",
            device,
        ]
    )

    critic_direction = diagnostic / "critic_direction"
    run(
        [
            sys.executable,
            str(IMPLEMENTATIONS / "diagnose_critic_action_gradient_alignment_20260723.py"),
            "--spec",
            str(critic_protocol),
            "--output-dir",
            str(critic_direction),
            "--device",
            device,
        ]
    )
    copy_as(critic_direction / "critic_action_gradient_rows.csv", critic_direction / "rows.csv")

    action_projection = diagnostic / "action_projection"
    run(
        [
            sys.executable,
            str(IMPLEMENTATIONS / "diagnose_plan2307_action_projection_bottleneck_20260724.py"),
            "--input",
            str(critic_direction / "critic_action_gradient_rows.csv"),
            "--source-summary",
            str(critic_direction / "summary.json"),
            "--output-dir",
            str(action_projection),
        ]
    )
    copy_as(action_projection / "action_projection_by_seed.csv", action_projection / "by_seed.csv")
    copy_as(action_projection / "action_projection_pooled.csv", action_projection / "pooled.csv")

    recognition = diagnostic / "reference_recognition"
    run(
        [
            sys.executable,
            str(IMPLEMENTATIONS / "diagnose_plan2307_divergence_state_critic_20260724.py"),
            "--spec",
            str(critic_protocol),
            "--protocol",
            str(PROTOCOLS / "pure_rl_offgrid_validation_protocol_20260722.json"),
            "--output-dir",
            str(recognition),
            "--device",
            device,
        ]
    )
    copy_as(recognition / "divergence_state_critic_rows.csv", recognition / "rows.csv")

    rebuild_objective_share(
        models_root=models_root,
        replay_root=replay_root,
        output=output,
        device=device,
    )

    prefix = diagnostic / "prefix_intervention"
    run(
        [
            sys.executable,
            str(IMPLEMENTATIONS / "diagnose_reference_prefix_intervention_20260723.py"),
            "--config",
            str(prefix_protocol),
            "--out",
            str(prefix),
            "--device",
            device,
        ]
    )


def validate_generated_rollouts(output: Path) -> dict[str, Any]:
    frames = {
        name: read_and_validate_rollout(output / "rollouts" / f"{name}.csv", name)
        for name in ROLLOUT_RECIPES
    }
    validate_cross_method_grid(frames)
    return frames


def compare_rollout_frame(
    name: str,
    generated: pd.DataFrame,
    retained: pd.DataFrame,
) -> dict[str, float | int]:
    """Compare behavior while allowing small accumulated floating-point drift."""

    keys = ("actual_seed", "theta", "theta_dot")
    required = {
        *keys,
        "return",
        "task_success",
        "near_best_known_return_eps",
        "beats_best_known_return",
        "signed_gap_to_best_known",
    }
    for label, frame in (("generated", generated), ("retained", retained)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} {label} rollout is missing columns: {sorted(missing)}")
    if len(generated) != len(retained):
        raise ValueError(
            f"{name} rollout row count differs: {len(generated)} != {len(retained)}"
        )

    actual = generated.sort_values(list(keys)).reset_index(drop=True)
    expected = retained.sort_values(list(keys)).reset_index(drop=True)
    for key in keys:
        if not np.array_equal(actual[key].to_numpy(), expected[key].to_numpy()):
            raise ValueError(f"{name} rollout grid differs in {key}")

    return_error = np.abs(
        actual["return"].to_numpy(dtype=float)
        - expected["return"].to_numpy(dtype=float)
    )
    if not np.isfinite(return_error).all():
        raise ValueError(f"{name} rollout contains non-finite return differences")
    max_error = float(return_error.max(initial=0.0))
    mean_error = float(return_error.mean()) if len(return_error) else 0.0
    if max_error > ROLLOUT_RETURN_MAX_ABS_ERROR:
        raise ValueError(
            f"{name} maximum return drift {max_error:.6g} exceeds "
            f"{ROLLOUT_RETURN_MAX_ABS_ERROR}"
        )
    if mean_error > ROLLOUT_RETURN_MEAN_ABS_ERROR:
        raise ValueError(
            f"{name} mean absolute return drift {mean_error:.6g} exceeds "
            f"{ROLLOUT_RETURN_MEAN_ABS_ERROR}"
        )

    task_mismatch = (
        actual["task_success"].to_numpy(dtype=int)
        != expected["task_success"].to_numpy(dtype=int)
    )
    if task_mismatch.any():
        raise ValueError(f"{name} changes {int(task_mismatch.sum())} task outcomes")

    gap_actual = actual["signed_gap_to_best_known"].to_numpy(dtype=float)
    gap_expected = expected["signed_gap_to_best_known"].to_numpy(dtype=float)
    boundary_flips: dict[str, int] = {}
    for field, threshold in (
        ("near_best_known_return_eps", 5.0),
        ("beats_best_known_return", 0.0),
    ):
        mismatch = (
            actual[field].to_numpy(dtype=int)
            != expected[field].to_numpy(dtype=int)
        )
        count = int(mismatch.sum())
        if count:
            close_to_boundary = (
                np.abs(gap_actual[mismatch] - threshold)
                <= ROLLOUT_CLASSIFICATION_BOUNDARY
            ) & (
                np.abs(gap_expected[mismatch] - threshold)
                <= ROLLOUT_CLASSIFICATION_BOUNDARY
            )
            if not close_to_boundary.all():
                raise ValueError(
                    f"{name} changes {field} away from its numeric boundary"
                )
        boundary_flips[field] = count

    return {
        "rows": int(len(actual)),
        "return_max_abs_error": max_error,
        "return_mean_abs_error": mean_error,
        "near_boundary_flips": boundary_flips["near_best_known_return_eps"],
        "strict_boundary_flips": boundary_flips["beats_best_known_return"],
        "task_flips": 0,
    }


def check_retained_rollout_fidelity(
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for name in ROLLOUT_RECIPES:
        retained = read_and_validate_rollout(
            RETAINED_DATA / "rollouts" / f"{name}.csv",
            name,
        )
        results[name] = compare_rollout_frame(name, frames[name], retained)
    return results


def _require_close(
    label: str,
    generated: Any,
    retained: Any,
    *,
    atol: float,
) -> float:
    actual = np.asarray(generated, dtype=float)
    expected = np.asarray(retained, dtype=float)
    if actual.shape != expected.shape:
        raise ValueError(f"{label} shape differs: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{label} contains non-finite values")
    difference = np.abs(actual - expected)
    maximum = float(difference.max(initial=0.0))
    if maximum > atol:
        raise ValueError(f"{label} maximum drift {maximum:.6g} exceeds {atol}")
    return maximum


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_prefix_table(
    generated_path: Path,
    retained_path: Path,
    *,
    keys: tuple[str, ...],
    fields: tuple[str, ...],
) -> float:
    generated = pd.read_csv(generated_path).sort_values(list(keys)).reset_index(drop=True)
    retained = pd.read_csv(retained_path).sort_values(list(keys)).reset_index(drop=True)
    if len(generated) != len(retained):
        raise ValueError(
            f"{generated_path.name} row count differs: {len(generated)} != {len(retained)}"
        )
    for key in keys:
        if generated[key].astype(str).tolist() != retained[key].astype(str).tolist():
            raise ValueError(f"{generated_path.name} keys differ in {key}")
    return _require_close(
        generated_path.name,
        generated[list(fields)].to_numpy(dtype=float),
        retained[list(fields)].to_numpy(dtype=float),
        atol=DIAGNOSTIC_RATE_ABS_ERROR,
    )


def check_retained_diagnostic_fidelity(output: Path) -> dict[str, float]:
    generated_root = output / "diagnostics"
    retained_root = RETAINED_DATA / "diagnostics"
    results: dict[str, float] = {}

    generated = _load_json(generated_root / "actor_geometry" / "summary.json")
    retained = _load_json(retained_root / "actor_geometry" / "summary.json")
    geometry_values: list[float] = []
    geometry_expected: list[float] = []
    for arm in (
        "p0_simba_onestep_utd1_100k",
        "p1_simba_fastsacn8_lambda1_utd1_100k",
        "p2_simba_sacn8_lambda1_utd1_100k",
        "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k",
    ):
        for metric in (
            "deterministic_action_saturation_fraction_abs_ge_0p995",
            "mean_tanh_derivative",
            "reflection_action_abs_error_mean",
        ):
            geometry_values.extend(generated["arms"][arm]["metrics"][metric]["seed_values"])
            geometry_expected.extend(retained["arms"][arm]["metrics"][metric]["seed_values"])
    results["actor_geometry_max_abs_error"] = _require_close(
        "actor geometry", geometry_values, geometry_expected, atol=1e-5
    )

    claims = _load_json(RETAINED_DATA / "claims.json")
    semantic = claims["diagnostic_semantics"]["c32"]
    conditions = [str(value) for value in semantic["conditions"]]
    generated = _load_json(generated_root / "critic_direction" / "summary.json")
    retained = _load_json(retained_root / "critic_direction" / "summary.json")
    critic_fields = (str(semantic["figure_field"]), "critic_step_harmful_rate")
    critic_values = [
        generated["conditions"][condition]["pooled"][field]
        for condition in conditions
        for field in critic_fields
    ]
    critic_expected = [
        retained["conditions"][condition]["pooled"][field]
        for condition in conditions
        for field in critic_fields
    ]
    results["critic_direction_max_abs_error"] = _require_close(
        "critic direction", critic_values, critic_expected, atol=DIAGNOSTIC_RATE_ABS_ERROR
    )

    generated = _load_json(generated_root / "reference_recognition" / "summary.json")
    retained = _load_json(retained_root / "reference_recognition" / "summary.json")
    recognition_values = [
        generated["conditions"][condition]["pooled"]["failure"][
            "critic_prefers_helpful_reference_rate"
        ]
        for condition in conditions
    ]
    recognition_expected = [
        retained["conditions"][condition]["pooled"]["failure"][
            "critic_prefers_helpful_reference_rate"
        ]
        for condition in conditions
    ]
    results["reference_recognition_max_abs_error"] = _require_close(
        "reference recognition",
        recognition_values,
        recognition_expected,
        atol=DIAGNOSTIC_RATE_ABS_ERROR,
    )

    generated = _load_json(generated_root / "action_projection" / "summary.json")
    retained = _load_json(retained_root / "action_projection" / "summary.json")
    generated_rows = {
        (row["condition"], row["outcome"]): row for row in generated["pooled"]
    }
    retained_rows = {
        (row["condition"], row["outcome"]): row for row in retained["pooled"]
    }
    projection_fields = (
        "boundary_rate",
        "outward_among_boundary_rate",
        "mean_effective_step_fraction",
    )
    projection_values = [
        generated_rows[(condition, "failure")][field]
        for condition in conditions
        for field in projection_fields
    ]
    projection_expected = [
        retained_rows[(condition, "failure")][field]
        for condition in conditions
        for field in projection_fields
    ]
    results["action_projection_max_abs_error"] = _require_close(
        "action projection",
        projection_values,
        projection_expected,
        atol=DIAGNOSTIC_RATE_ABS_ERROR,
    )

    generated = _load_json(generated_root / "objective_share" / "summary.json")
    retained = _load_json(retained_root / "objective_share" / "summary.json")
    for field in ("actor_seeds", "replay_sequences_per_seed", "density_replay_sequences_total"):
        if generated[field] != retained[field]:
            raise ValueError(f"objective share differs in {field}")
    share_fields = generated["eight_step_objective_share_percent"]
    share_expected = retained["eight_step_objective_share_percent"]
    share_values = [
        value
        for field in share_fields.values()
        for value in (field if isinstance(field, list) else [field])
    ]
    share_retained = [
        value
        for field in share_expected.values()
        for value in (field if isinstance(field, list) else [field])
    ]
    results["objective_share_max_abs_error_percent"] = _require_close(
        "objective share", share_values, share_retained, atol=0.05
    )
    diagnostic_values = generated["density_eight_step_diagnostics_percent"]
    diagnostic_expected = retained["density_eight_step_diagnostics_percent"]
    seed_fields = [field for field in diagnostic_values if field.endswith("_seed_values")]
    results["objective_seed_diagnostic_max_abs_error_percent"] = _require_close(
        "objective-share seed diagnostics",
        [value for field in seed_fields for value in diagnostic_values[field]],
        [value for field in seed_fields for value in diagnostic_expected[field]],
        atol=2.0,
    )
    mean_fields = [field for field in diagnostic_values if field.endswith("_mean")]
    results["objective_mean_diagnostic_max_abs_error_percent"] = _require_close(
        "objective-share mean diagnostics",
        [diagnostic_values[field] for field in mean_fields],
        [diagnostic_expected[field] for field in mean_fields],
        atol=0.5,
    )

    results["prefix_aggregate_max_abs_error"] = _compare_prefix_table(
        generated_root / "prefix_intervention" / "aggregate.csv",
        retained_root / "prefix_intervention" / "aggregate.csv",
        keys=("condition", "prefix_steps"),
        fields=("near_rate", "repair_rate", "task_rate"),
    )
    results["prefix_specificity_max_abs_error"] = _compare_prefix_table(
        generated_root / "prefix_intervention" / "specificity_control.csv",
        retained_root / "prefix_intervention" / "specificity_control.csv",
        keys=("prefix_mode", "prefix_steps"),
        fields=("repair_rate",),
    )
    return results


def plan_payload(models_root: Path, replay_root: Path, output: Path) -> dict[str, Any]:
    return {
        "models_root": str(models_root),
        "replay_root": str(replay_root),
        "output": str(output),
        "rollouts": {
            name: {"family": recipe.family, "policy": recipe.policy}
            for name, recipe in ROLLOUT_RECIPES.items()
        },
        "diagnostics": [
            "actor_geometry",
            "critic_direction",
            "action_projection",
            "reference_recognition",
            "objective_share",
            "prefix_intervention",
        ],
        "figures": 9,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models_root = args.models_root.expanduser().resolve()
    replay_root = args.replay_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.smoke and output == DEFAULT_OUTPUT.resolve():
        output = ROOT / ".build" / "report-data-smoke"
    validate_inputs(models_root, replay_root)
    plan = plan_payload(models_root, replay_root, output)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    for name, recipe in ROLLOUT_RECIPES.items():
        rebuild_rollout(
            name,
            recipe,
            models_root=models_root,
            output=output,
            device=args.device,
            smoke=bool(args.smoke),
        )
    if args.smoke:
        print(f"SMOKE PASS ({len(ROLLOUT_RECIPES)} deployment recipes): {output}")
        return 0

    frames = validate_generated_rollouts(output)
    retained_models = models_root == DEFAULT_MODELS.resolve()
    fidelity: dict[str, Any] = {"checked_against_retained_evidence": retained_models}
    if retained_models:
        fidelity["rollouts"] = check_retained_rollout_fidelity(frames)
        print("ROLLOUT FIDELITY PASS", flush=True)

    rebuild_diagnostics(
        models_root=models_root,
        replay_root=replay_root,
        output=output,
        device=args.device,
    )
    if retained_models:
        fidelity["diagnostics"] = check_retained_diagnostic_fidelity(output)
        print("DIAGNOSTIC FIDELITY PASS", flush=True)
    (output / "fidelity.json").write_text(
        json.dumps(fidelity, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(RETAINED_DATA / "claims.json", output / "claims.json")
    (output / "derived_metrics.json").write_text(
        json.dumps(
            derived_report_payload({name: frames[name] for name in FIGURE_METHODS}),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figures: list[Path] = []
    comparison_path: Path | None = None
    if not args.no_figures:
        figures = render_all(
            output,
            {name: frames[name] for name in FIGURE_METHODS},
            output / "figures",
        )
        comparison_path = write_comparison(
            compare_images(figures, ROOT / "report" / "source" / "figures"),
            output / "comparison.json",
        )
    result = {
        **plan,
        "status": "complete",
        "rollout_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "fidelity": str(output / "fidelity.json"),
        "derived_metrics": str(output / "derived_metrics.json"),
        "generated_figures": [str(path) for path in figures],
        "comparison": str(comparison_path) if comparison_path else None,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REPORT DATA PASS: {output}")
    if figures:
        print(f"REPORT FIGURES PASS ({len(figures)}/9): {output / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
