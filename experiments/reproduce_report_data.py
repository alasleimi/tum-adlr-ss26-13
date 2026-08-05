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
from typing import Any

import numpy as np

from last_nine_repro.figures import render_all
from last_nine_repro.metrics import summarize_method
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
IMPLEMENTATIONS = ROOT / "experiments" / "implementations"
PROTOCOLS = ROOT / "experiments" / "protocols"
DP_GRID = ROOT / "data" / "reference" / "pendulum_dp_grid.csv"
CONTROLLER_GRID = ROOT / "data" / "reference" / "controller_grid.csv"
DP_SOLUTION = ROOT / "data" / "reference" / "pendulum_dp_solution.npz"


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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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
                "0",
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


def check_retained_claims(frames: dict[str, Any]) -> None:
    claims = json.loads((ROOT / "data" / "report" / "claims.json").read_text(encoding="utf-8"))
    for name in FIGURE_METHODS:
        expected = claims["methods"][name]
        actual = summarize_method(frames[name]).to_dict()
        for field in ("trials", "near", "task", "strict", "failures", "failure_cells"):
            if int(actual[field]) != int(expected[field]):
                raise ValueError(
                    f"{name} {field} differs from retained evidence: "
                    f"{actual[field]} != {expected[field]}"
                )


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


def main() -> int:
    args = parse_args()
    models_root = args.models_root.expanduser().resolve()
    replay_root = args.replay_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_inputs(models_root, replay_root)
    plan = plan_payload(models_root, replay_root, output)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if args.smoke and args.output == DEFAULT_OUTPUT:
        output = ROOT / ".build" / "report-data-smoke"
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

    rebuild_diagnostics(
        models_root=models_root,
        replay_root=replay_root,
        output=output,
        device=args.device,
    )
    frames = validate_generated_rollouts(output)
    if models_root == DEFAULT_MODELS.resolve():
        check_retained_claims(frames)
    shutil.copy2(ROOT / "data" / "report" / "claims.json", output / "claims.json")

    figures: list[Path] = []
    if not args.no_figures:
        figures = render_all(
            output,
            {name: frames[name] for name in FIGURE_METHODS},
            output / "figures",
        )
    result = {
        **plan,
        "status": "complete",
        "rollout_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "generated_figures": [str(path) for path in figures],
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
