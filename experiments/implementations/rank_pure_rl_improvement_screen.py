from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.envs import UprightDetector
from last_nine_rl.hybrid_qsearch import reflection_averaged_actor_actions
from last_nine_rl.pendulum_grid import rollout_pendulum_grid_vectorized
from last_nine_rl.qsearch_lock import (
    GLOBAL_CANDIDATE_SUPPORT,
    UNANIMOUS_ACCEPTANCE,
    authoritative_grid_mask,
    build_validation_dataset,
    file_fingerprint,
    sha256_file,
    sha256_json,
    validate_artifact_manifest,
)

try:
    from scripts.train_pendulum_qregularized_dagger import (
        validation_reference_returns,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from train_pendulum_qregularized_dagger import (  # type: ignore[no-redef]
        validation_reference_returns,
    )


PROTOCOL_VERSION = 2
WORKFLOW_VERSION = "pure-rl-offgrid-ranking-v2"
DEFAULT_PROTOCOL_CONFIG = "experiments/protocols/pure_rl_offgrid_validation_protocol_20260722.json"
DEFAULT_POINTS = 4096
# Frozen before any arm in this screen was evaluated. This seed is distinct from
# every pure-RL development set catalogued in reports/pure_rl_plus1pp_20260719.
DEFAULT_RNG_SEED = 22072601
DEFAULT_HORIZON = 200
DEFAULT_VELOCITY_LIMIT = 1.0
MAX_ENVIRONMENT_STEPS = 100_000
MAX_STRICT_TRAINING_STEPS = 100_000
PRIMARY_VARIANT = "reflection_actor_global41_unanimous_m005"
BASELINE_LABEL = "official_pure_simba_seed0_baseline"
RANKING_RULE = [
    "near_reference_eps_successes_desc",
    "task_successes_desc",
    "strict_beats_reference_successes_desc",
    "bottom10_conditional_mean_return_desc",
    "mean_return_desc",
    "condition_asc_deterministic_tiebreak",
]
LOCKED_PRIMARY_POLICY_SPEC = {
    "kind": "conservative_global",
    "reflection": True,
    "num_actions": 41,
    "margin": 0.005,
    # Pendulum actions lie in [-2, 2], so four spans the entire support and is
    # exactly equivalent to the preregistered unrestricted global search.
    "max_action_delta": 4.0,
    "acceptance": UNANIMOUS_ACCEPTANCE,
    "candidate_support": GLOBAL_CANDIDATE_SUPPORT,
}
FIXED_STATE_GEOMETRY_SPEC = {
    "state_source": "frozen_validation_dataset_initial_states_before_rollout",
    "mean_logit_abs_excess_threshold": 4.15,
    "deterministic_tanh_saturation_threshold": 0.995,
    "log_std_thresholds": [-1.0, -1.5, -2.0, -3.0],
    "log_std_reporting": "effective_clamped_and_pre_floor_unclamped",
    "critic_source": "mean_of_all_online_checkpoint_critics",
    "critic_reflection_action": "negative_of_original_deterministic_actor_action",
}
FIXED_STATE_ACTOR_GEOMETRY_FIELDS = {
    "points",
    "state_sha256",
    "mean_logit_abs_mean",
    "mean_logit_abs_max",
    "mean_logit_abs_gt_4p15_fraction",
    "deterministic_action_saturation_fraction_abs_ge_0p995",
    "mean_tanh_derivative",
    "log_std_mean",
    "log_std_min",
    "log_std_below_minus_1p0_fraction",
    "log_std_below_minus_1p5_fraction",
    "log_std_below_minus_2p0_fraction",
    "log_std_below_minus_3p0_fraction",
    "unclamped_log_std_mean",
    "unclamped_log_std_min",
    "unclamped_log_std_below_minus_1p0_fraction",
    "unclamped_log_std_below_minus_1p5_fraction",
    "unclamped_log_std_below_minus_2p0_fraction",
    "unclamped_log_std_below_minus_3p0_fraction",
    "log_std_floor_active_fraction",
    "log_std_floor_lift_mean",
    "reflection_action_abs_error_mean",
    "reflection_action_abs_error_max",
}
FIXED_STATE_CRITIC_GEOMETRY_FIELDS = {
    "points",
    "state_sha256",
    "critics",
    "reflection_q_abs_error_mean",
    "reflection_q_abs_error_max",
    "ensemble_q_std_mean",
    "ensemble_q_std_max",
}
REPORT_FILENAMES = (
    "pure_rl_seed0_offgrid_ranking.json",
    "pure_rl_seed0_offgrid_ranking.csv",
    "pure_rl_seed0_offgrid_variants.csv",
    "REPORT.md",
)

VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "ordinary_actor",
        "kind": "actor",
        "reflection_averaged": False,
        "critic_q_search": False,
    },
    {
        "name": "reflection_actor",
        "kind": "reflection_actor",
        "reflection_averaged": True,
        "critic_q_search": False,
    },
    {
        "name": PRIMARY_VARIANT,
        "kind": "global_q_search",
        "reflection_averaged": True,
        "critic_q_search": True,
        "num_actions": 41,
        "margin": 0.005,
        "filter_mode": "symmetric_actor_unanimous_advantage",
        "blend_fraction": 1.0,
        "max_action_delta": 4.0,
    },
)


@dataclass(frozen=True)
class Condition:
    label: str
    run_dir: Path
    completed_steps: int
    kind: str = "arm"


class ReflectionActorPolicy:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def act_batch(
        self, observations: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        del deterministic
        return reflection_averaged_actor_actions(self.agent, observations)


class TrackedReflectionGlobalQSearchPolicy:
    """The frozen pure-RL deployment rule used by the preceding best result.

    The actor and both critics come from one checkpoint. The rule has no state
    router, reference call, coordinate range, second model, or learned mixture.
    """

    def __init__(
        self,
        agent: Any,
        *,
        num_actions: int = 41,
        margin: float = 0.005,
        batch_size: int = 512,
    ) -> None:
        self.agent = agent
        self.num_actions = int(num_actions)
        self.margin = float(margin)
        self.batch_size = int(batch_size)
        self.selected_count = 0
        self.total_count = 0
        self.selected_abs_delta_sum = 0.0
        self.selected_abs_delta_max = 0.0

    def act_batch(
        self, observations: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        del deterministic
        fallback = reflection_averaged_actor_actions(self.agent, observations)
        selected = np.asarray(
            self.agent.act_batch_critic_search(
                observations,
                num_actions=self.num_actions,
                margin=self.margin,
                batch_size=self.batch_size,
                filter_mode="symmetric_actor_unanimous_advantage",
                blend_fraction=1.0,
            ),
            dtype=np.float32,
        ).reshape(-1, 1)
        delta = np.abs(selected - fallback).reshape(-1)
        changed = delta > 1e-7
        count = int(changed.sum())
        self.selected_count += count
        self.total_count += int(len(delta))
        if count:
            self.selected_abs_delta_sum += float(delta[changed].sum())
            self.selected_abs_delta_max = max(
                self.selected_abs_delta_max, float(delta[changed].max())
            )
        return selected

    def selection_metrics(self) -> dict[str, float]:
        return {
            "selected_count": float(self.selected_count),
            "total_decisions": float(self.total_count),
            "switch_fraction_vs_reflection_actor": float(
                self.selected_count / max(self.total_count, 1)
            ),
            "selected_abs_action_delta_mean": float(
                self.selected_abs_delta_sum / max(self.selected_count, 1)
            ),
            "selected_abs_action_delta_max": float(self.selected_abs_delta_max),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally evaluate completed seed-0 pure-RL screen arms on one "
            "frozen continuous/off-grid set, then rank the preregistered "
            "reflection-actor + global-41 unanimous-Q policy."
        )
    )
    parser.add_argument(
        "--screen-root",
        default="runs/report_reproduction/offgrid_screen",
        help="Directory containing <arm>/seed0 run directories.",
    )
    parser.add_argument(
        "--evaluation-root",
        required=True,
        help="Persistent per-arm evaluations; valid entries are reused exactly.",
    )
    parser.add_argument(
        "--baseline-run",
        default=None,
        help=(
            "Existing official pure-RL seed-0 run to evaluate under the identical "
            "frozen protocol and include in paired recipe deltas. The baseline is "
            "loaded independently; no checkpoint copying or model mixing occurs."
        ),
    )
    parser.add_argument(
        "--additional-run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Independently load and rank one completed pure-RL run outside "
            "--screen-root. Repeat for multiple runs. Each LABEL must be unique; "
            "the checkpoint is evaluated in place and is never copied or mixed."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="New report directory. Existing report files are never overwritten.",
    )
    parser.add_argument(
        "--evaluate-missing",
        action="store_true",
        help="Evaluate newly completed arms; otherwise only inventory them as pending.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--critic-search-batch-size", type=int, default=512)
    parser.add_argument("--protocol-config", default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="Compatibility assertion only; must equal the locked continuous-point count.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Compatibility assertion only; must equal the locked validation RNG seed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.torch_threads) <= 0:
        raise SystemExit("--torch-threads must be positive")
    if int(args.critic_search_batch_size) <= 0:
        raise SystemExit("--critic-search-batch-size must be positive")

    project_root = Path(__file__).resolve().parents[1]
    screen_root = resolve_from(project_root, Path(args.screen_root))
    evaluation_root = resolve_from(project_root, Path(args.evaluation_root))
    output_dir = resolve_from(project_root, Path(args.out))
    assert_report_outputs_absent(output_dir)

    screen_conditions, skipped = discover_completed_arms(screen_root)
    baseline_condition: Condition | None = None
    if args.baseline_run:
        baseline_condition = baseline_condition_from_run(
            resolve_from(project_root, Path(args.baseline_run))
        )
    additional_conditions = additional_conditions_from_specs(
        list(args.additional_run), project_root
    )
    conditions = combine_unique_conditions(
        [baseline_condition] if baseline_condition is not None else [],
        screen_conditions,
        additional_conditions,
    )
    protocol_path = resolve_from(project_root, Path(args.protocol_config))
    protocol = evaluation_protocol(
        protocol_config=protocol_path,
        points=args.points,
        rng_seed=args.rng_seed,
        critic_search_batch_size=int(args.critic_search_batch_size),
    )
    if args.evaluate_missing:
        evaluation_root.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(int(args.torch_threads))

    available: list[tuple[Condition, Path]] = []
    pending: list[dict[str, Any]] = []
    for condition in conditions:
        evaluation_dir = evaluation_root / safe_label(condition.label)
        if evaluation_dir.exists():
            validate_evaluation(condition, evaluation_dir, protocol)
        elif args.evaluate_missing:
            evaluate_condition_new(
                condition=condition,
                evaluation_dir=evaluation_dir,
                evaluation_root=evaluation_root,
                protocol=protocol,
                device=str(args.device),
            )
        else:
            pending.append(
                {
                    "condition": condition.label,
                    "run_dir": str(condition.run_dir),
                    "reason": "frozen off-grid evaluation is missing",
                    "training_budget": strict_training_budget_from_run(
                        condition.run_dir
                    ),
                }
            )
            continue
        available.append((condition, evaluation_dir))

    # Leakage firewall: every candidate policy rollout and exact-cache check is
    # complete before the off-grid reference is computed or read. The 61x41
    # authority CSVs are neither opened nor parsed anywhere in this program.
    reference_cache: dict[str, Any] | None = None
    if available:
        reference_cache = offgrid_reference_returns(
            evaluation_root=evaluation_root,
            protocol=protocol,
            create_if_missing=bool(args.evaluate_missing),
        )
    if available and reference_cache is None:
        for condition, _evaluation_dir in available:
            pending.append(
                {
                    "condition": condition.label,
                    "run_dir": str(condition.run_dir),
                    "reason": "hash-bound off-grid reference cache is missing",
                    "training_budget": strict_training_budget_from_run(
                        condition.run_dir
                    ),
                }
            )
        available = []

    evaluated: list[dict[str, Any]] = []
    for condition, evaluation_dir in available:
        evaluation = read_json(evaluation_dir / "evaluation.json")
        assert reference_cache is not None
        variants = add_reference_metrics_to_variants(
            evaluation["variants"],
            evaluation_dir / "rollouts.npz",
            np.asarray(reference_cache["returns"], dtype=np.float64),
            epsilon_return=float(protocol["reference_protocol"]["epsilon_return"]),
        )
        training = summarize_training_diagnostics(condition.run_dir)
        evaluated.append(
            {
                "condition": condition.label,
                "kind": condition.kind,
                "run_dir": str(condition.run_dir),
                "completed_environment_steps": condition.completed_steps,
                "training_budget": strict_training_budget_from_run(
                    condition.run_dir
                ),
                "checkpoint": checkpoint_identity(condition.run_dir),
                "evaluation_dir": str(evaluation_dir),
                "fixed_state_actor_geometry": evaluation[
                    "fixed_state_actor_geometry"
                ],
                "fixed_state_critic_geometry": evaluation[
                    "fixed_state_critic_geometry"
                ],
                "variants": variants,
                "training": training,
            }
        )

    ranked_arms = rank_primary_variants(evaluated)
    ranked_variants = rank_all_variants(evaluated)
    add_baseline_deltas(ranked_arms, ranked_variants)
    report = {
        "protocol": {
            **protocol,
            "screen_root": str(screen_root),
            "evaluation_root": str(evaluation_root),
            "baseline_run": (
                str(baseline_condition.run_dir)
                if baseline_condition is not None
                else None
            ),
            "baseline_condition": (
                baseline_condition.label if baseline_condition is not None else None
            ),
            "additional_runs": [
                {
                    "condition": condition.label,
                    "run_dir": str(condition.run_dir),
                    "completed_environment_steps": condition.completed_steps,
                    "training_budget": strict_training_budget_from_run(
                        condition.run_dir
                    ),
                    "independently_loaded": True,
                }
                for condition in additional_conditions
            ],
            "primary_recipe_variant": PRIMARY_VARIANT,
            "strict_training_step_cap": MAX_STRICT_TRAINING_STEPS,
            "strict_training_budget_definition": (
                "learning environment transitions + analytically generated model "
                "transitions entering replay + failure-curriculum discovery "
                "environment steps; eligibility also requires the independently "
                "computed failure-discovery planned upper bound to fit the cap"
            ),
            "primary_arm_ranking_order": list(RANKING_RULE),
            "selection_scope": "seed0_development_only",
            "authoritative_grid_queried": False,
            "authoritative_reference_queried": False,
            "reference_policy_queried": bool(reference_cache is not None),
            "reference_used_during_candidate_rollout": False,
            "reference_used_only_after_all_candidate_rollouts": True,
            "offgrid_reference_cache": (
                reference_cache["identity"] if reference_cache is not None else None
            ),
            "inference_router": False,
            "model_mixture": False,
            "selection_warning": (
                "This fixed off-grid seed-0 screen may select candidates for a later "
                "authoritative reference-relative grid; it is not final evidence. "
                "A fresh multi-seed confirmation is required after recipe selection."
            ),
        },
        "ranked_arms": ranked_arms,
        "ranked_variants": ranked_variants,
        "pending_conditions": pending,
        "skipped_screen_arms": skipped,
    }
    write_report_new(output_dir, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def resolve_from(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def discover_completed_arms(
    screen_root: Path,
) -> tuple[list[Condition], list[dict[str, Any]]]:
    if not screen_root.is_dir():
        raise SystemExit(f"Pure-RL screen root does not exist: {screen_root}")
    conditions: list[Condition] = []
    skipped: list[dict[str, Any]] = []
    for arm_dir in sorted(
        (path for path in screen_root.iterdir() if path.is_dir()), key=lambda p: p.name
    ):
        run_dir = arm_dir / "seed0"
        reason = incomplete_or_ineligible_reason(run_dir)
        if reason is not None:
            skipped_row: dict[str, Any] = {
                "condition": arm_dir.name,
                "run_dir": str(run_dir),
                "reason": reason,
            }
            budget = try_strict_training_budget_from_run(run_dir)
            if budget is not None:
                skipped_row["training_budget"] = budget
            skipped.append(skipped_row)
            continue
        completed_steps = completed_run_step(run_dir / "events.jsonl")
        assert completed_steps is not None
        conditions.append(
            Condition(arm_dir.name, run_dir.resolve(), int(completed_steps), "arm")
        )
    return conditions, skipped


def baseline_condition_from_run(run_dir: Path) -> Condition:
    """Validate and identify one independently loaded official control run."""

    reason = incomplete_or_ineligible_reason(run_dir)
    if reason is not None:
        raise SystemExit(f"Baseline run is incomplete or ineligible: {reason} ({run_dir})")
    complete_step = completed_run_step(run_dir / "events.jsonl")
    assert complete_step is not None
    return Condition(BASELINE_LABEL, run_dir.resolve(), int(complete_step), "baseline")


def additional_conditions_from_specs(
    specs: list[str], project_root: Path
) -> list[Condition]:
    """Parse and validate independently loaded LABEL=PATH conditions."""

    conditions: list[Condition] = []
    labels: set[str] = set()
    for spec in specs:
        label_value, separator, path_value = str(spec).partition("=")
        label = label_value.strip()
        run_value = path_value.strip()
        if not separator or not label or not run_value:
            raise SystemExit(
                "--additional-run must use non-empty LABEL=PATH syntax; "
                f"received {spec!r}"
            )
        if label in labels:
            raise SystemExit(f"Duplicate --additional-run label: {label!r}")
        # Fail early for labels that cannot own a persistent evaluation directory.
        safe_label(label)
        run_dir = resolve_from(project_root, Path(run_value))
        reason = incomplete_or_ineligible_reason(run_dir)
        if reason is not None:
            raise SystemExit(
                f"Additional run {label!r} is incomplete or ineligible: "
                f"{reason} ({run_dir})"
            )
        complete_step = completed_run_step(run_dir / "events.jsonl")
        assert complete_step is not None
        labels.add(label)
        conditions.append(
            Condition(label, run_dir.resolve(), int(complete_step), "additional")
        )
    return conditions


def combine_unique_conditions(*groups: list[Condition]) -> list[Condition]:
    """Combine condition sources without ambiguous labels or evaluation keys."""

    combined: list[Condition] = []
    by_label: dict[str, Condition] = {}
    by_safe_label: dict[str, Condition] = {}
    by_run: dict[str, Condition] = {}
    for condition in (item for group in groups for item in group):
        prior = by_label.get(condition.label)
        if prior is not None:
            raise SystemExit(
                f"Duplicate condition label {condition.label!r}: "
                f"{prior.run_dir} and {condition.run_dir}"
            )
        evaluation_key = safe_label(condition.label)
        prior_safe = by_safe_label.get(evaluation_key)
        if prior_safe is not None:
            raise SystemExit(
                "Condition labels collide after filesystem sanitization: "
                f"{prior_safe.label!r} and {condition.label!r} -> {evaluation_key!r}"
            )
        run_key = os.path.normcase(str(condition.run_dir.resolve()))
        prior_run = by_run.get(run_key)
        if prior_run is not None:
            raise SystemExit(
                "The same run was supplied more than once under different conditions: "
                f"{prior_run.label!r} and {condition.label!r} ({condition.run_dir})"
            )
        by_label[condition.label] = condition
        by_safe_label[evaluation_key] = condition
        by_run[run_key] = condition
        combined.append(condition)
    return combined


def incomplete_or_ineligible_reason(run_dir: Path) -> str | None:
    if not run_dir.is_dir():
        return "seed0 directory is missing"
    required = (
        run_dir / "config.json",
        run_dir / "events.jsonl",
        run_dir / "checkpoints" / "final.pt",
    )
    missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
    if missing:
        return "missing " + ", ".join(missing)
    complete_step = completed_run_step(run_dir / "events.jsonl")
    if complete_step is None:
        return "run_complete event is absent"
    config = read_json(run_dir / "config.json")
    if int(config.get("seed", -1)) != 0:
        return f"config seed is {config.get('seed')}, expected 0"
    configured_steps = int(config.get("sac", {}).get("total_steps", -1))
    if configured_steps <= 0:
        return "sac.total_steps is absent or nonpositive"
    if configured_steps > MAX_ENVIRONMENT_STEPS or complete_step > MAX_ENVIRONMENT_STEPS:
        return (
            f"environment-step budget exceeds {MAX_ENVIRONMENT_STEPS}: "
            f"configured={configured_steps}, complete={complete_step}"
        )
    if complete_step != configured_steps:
        return (
            "run_complete step does not equal configured total_steps: "
            f"{complete_step} != {configured_steps}"
        )
    try:
        budget = strict_training_budget_from_run(run_dir)
    except (OSError, ValueError) as exc:
        return f"strict training-budget accounting failed: {exc}"
    if not budget["strict_budget_eligible"]:
        return (
            f"strict training budget exceeds {MAX_STRICT_TRAINING_STEPS}: "
            f"learning={budget['learning_environment_steps']}, "
            f"model={budget['model_generated_transitions']}, "
            "failure_discovery_counted="
            f"{budget['failure_discovery_environment_steps_counted']}, "
            f"strict={budget['strict_total_steps_counted']}, "
            "planned_upper_bound="
            f"{budget['strict_total_steps_planned_upper_bound']}"
        )
    purity_errors = pure_rl_config_errors(config)
    if purity_errors:
        return "not pure RL: " + "; ".join(purity_errors)
    replay_error = replay_reference_label_error(run_dir / "replay_final.npz")
    if replay_error is not None:
        return "not pure RL: " + replay_error
    return None


def try_strict_training_budget_from_run(run_dir: Path) -> dict[str, Any] | None:
    """Return an auditable budget when enough run metadata exists.

    This helper is only for enriching rejected-arm rows. Eligibility itself uses
    :func:`strict_training_budget_from_run` and fails closed on every malformed
    active schedule.
    """

    try:
        if not (run_dir / "config.json").is_file() or not (
            run_dir / "events.jsonl"
        ).is_file():
            return None
        return strict_training_budget_from_run(run_dir)
    except (OSError, ValueError):
        return None


def strict_training_budget_from_run(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    complete_event = completed_run_event(run_dir / "events.jsonl")
    if complete_event is None:
        raise ValueError("run_complete event is absent")
    try:
        learning_steps = int(complete_event["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run_complete step is absent or non-integral") from exc
    return strict_training_budget(config, complete_event, learning_steps=learning_steps)


def strict_training_budget(
    config: dict[str, Any],
    complete_event: dict[str, Any],
    *,
    learning_steps: int,
) -> dict[str, Any]:
    """Account for every transition-producing training path exactly.

    ``learning_steps`` counts real learner/environment transitions. Model replay
    and model rollouts are generated inside the inclusive ``1..total_steps``
    loop and enter replay buffers sampled by learned components. Failure-start
    discovery does not enter replay, but it consumes auxiliary simulator steps,
    so it is included in the user's strict budget as a separate component.
    """

    if learning_steps <= 0:
        raise ValueError("learning environment steps must be positive")
    sac = _mapping(config.get("sac"), "sac")
    env = _mapping(config.get("env"), "env")
    configured_steps = _integer_field(sac, "total_steps", required=True)
    if configured_steps != learning_steps:
        raise ValueError(
            "run_complete step does not equal configured sac.total_steps: "
            f"{learning_steps} != {configured_steps}"
        )

    learning_starts = _integer_field(sac, "learning_starts", default=5_000)
    if learning_starts < 0:
        raise ValueError("sac.learning_starts must be nonnegative")

    model_replay_ratio = _float_field(
        sac, "pendulum_model_replay_ratio", default=0.0
    )
    if not (0.0 <= model_replay_ratio < 1.0):
        raise ValueError("sac.pendulum_model_replay_ratio must be in [0, 1)")
    model_replay = _model_replay_budget(
        sac,
        total_steps=learning_steps,
        learning_starts=learning_starts,
        active=model_replay_ratio > 0.0,
    )
    model_rollout_ratio = _float_field(
        sac, "pendulum_model_rollout_ratio", default=0.0
    )
    if not (0.0 <= model_rollout_ratio < 1.0):
        raise ValueError("sac.pendulum_model_rollout_ratio must be in [0, 1)")
    model_rollout = _model_rollout_budget(
        sac,
        total_steps=learning_steps,
        learning_starts=learning_starts,
        active=model_rollout_ratio > 0.0,
    )
    model_generated = int(
        model_replay["generated_transitions"]
        + model_rollout["generated_transitions"]
    )

    failure = _failure_discovery_budget(
        env,
        complete_event,
        learning_steps=learning_steps,
    )
    discovery_actual = failure["actual_environment_steps"]
    discovery_counted = int(failure["counted_environment_steps"])
    discovery_planned = int(failure["planned_environment_steps_upper_bound"])
    strict_actual = (
        learning_steps + model_generated + int(discovery_actual)
        if discovery_actual is not None
        else None
    )
    strict_counted = learning_steps + model_generated + discovery_counted
    strict_planned = learning_steps + model_generated + discovery_planned
    strict_for_eligibility = max(strict_counted, strict_planned)

    return {
        # Preserve the conventional environmental-interaction interpretation as
        # an explicit field while exposing the stricter all-training-cost view.
        "learning_environment_steps": int(learning_steps),
        "model_generated_transitions": int(model_generated),
        "model_replay_generated_transitions": int(
            model_replay["generated_transitions"]
        ),
        "model_rollout_generated_transitions": int(
            model_rollout["generated_transitions"]
        ),
        "failure_discovery_environment_steps_actual": discovery_actual,
        "failure_discovery_environment_steps_counted": discovery_counted,
        "failure_discovery_environment_steps_planned_upper_bound": discovery_planned,
        "failure_discovery_accounting_source": failure["accounting_source"],
        "strict_total_steps_actual": strict_actual,
        "strict_total_steps_counted": int(strict_counted),
        "strict_total_steps_planned_upper_bound": int(strict_planned),
        "strict_total_steps_for_eligibility": int(strict_for_eligibility),
        "strict_training_step_cap": MAX_STRICT_TRAINING_STEPS,
        "strict_budget_eligible": bool(
            strict_for_eligibility <= MAX_STRICT_TRAINING_STEPS
        ),
        "model_replay_schedule": model_replay,
        "model_rollout_schedule": model_rollout,
        "failure_discovery_schedule": failure,
    }


def _model_replay_budget(
    sac: dict[str, Any],
    *,
    total_steps: int,
    learning_starts: int,
    active: bool,
) -> dict[str, Any]:
    if not active:
        return {
            "active": False,
            "first_generation_step": None,
            "last_generation_step": None,
            "generation_step_count": 0,
            "transitions_per_generation_step": 0,
            "generated_transitions": 0,
            "inclusive_schedule": True,
        }
    transitions_per_step = _integer_field(
        sac, "pendulum_model_replay_steps_per_step", default=1
    )
    configured_start = _integer_field(
        sac, "pendulum_model_replay_start_step", default=0
    )
    if transitions_per_step <= 0:
        raise ValueError(
            "active sac.pendulum_model_replay_steps_per_step must be positive"
        )
    if configured_start < 0:
        raise ValueError("sac.pendulum_model_replay_start_step must be nonnegative")
    first = max(1, learning_starts + 1, configured_start)
    count = max(0, total_steps - first + 1)
    return {
        "active": True,
        "configured_start_step": configured_start,
        "strictly_after_learning_starts": learning_starts,
        "first_generation_step": first if count else None,
        "last_generation_step": total_steps if count else None,
        "generation_step_count": int(count),
        "transitions_per_generation_step": int(transitions_per_step),
        "generated_transitions": int(count * transitions_per_step),
        "inclusive_schedule": True,
    }


def _model_rollout_budget(
    sac: dict[str, Any],
    *,
    total_steps: int,
    learning_starts: int,
    active: bool,
) -> dict[str, Any]:
    if not active:
        return {
            "active": False,
            "first_generation_step": None,
            "last_generation_step": None,
            "generation_event_count": 0,
            "transitions_per_generation_event": 0,
            "generated_transitions": 0,
            "inclusive_schedule": True,
        }
    starts = _integer_field(
        sac, "pendulum_model_rollout_starts_per_step", default=1
    )
    horizon = _integer_field(sac, "pendulum_model_rollout_horizon", default=8)
    interval = _integer_field(
        sac, "pendulum_model_rollout_interval_steps", default=1
    )
    configured_start = _integer_field(
        sac, "pendulum_model_rollout_start_step", default=0
    )
    if starts <= 0 or horizon <= 0 or interval <= 0:
        raise ValueError(
            "active model-rollout starts, horizon, and interval must be positive"
        )
    if configured_start < 0:
        raise ValueError("sac.pendulum_model_rollout_start_step must be nonnegative")
    lower = max(1, learning_starts + 1, configured_start)
    first = lower + (-lower % interval)
    count = (
        ((total_steps - first) // interval) + 1 if first <= total_steps else 0
    )
    last = first + (count - 1) * interval if count else None
    per_event = starts * horizon
    return {
        "active": True,
        "configured_start_step": configured_start,
        "strictly_after_learning_starts": learning_starts,
        "generation_interval_steps": interval,
        "first_generation_step": first if count else None,
        "last_generation_step": last,
        "generation_event_count": int(count),
        "rollout_starts_per_generation_event": int(starts),
        "rollout_horizon": int(horizon),
        "transitions_per_generation_event": int(per_event),
        "generated_transitions": int(count * per_event),
        "inclusive_schedule": True,
    }


def _failure_discovery_budget(
    env: dict[str, Any],
    complete_event: dict[str, Any],
    *,
    learning_steps: int,
) -> dict[str, Any]:
    probability = _float_field(env, "pendulum_failure_reset_prob", default=0.0)
    if not (0.0 <= probability <= 1.0):
        raise ValueError("env.pendulum_failure_reset_prob must be in [0, 1]")
    active = probability > 0.0
    if active:
        start = _integer_field(
            env, "pendulum_failure_curriculum_start_step", default=20_000
        )
        interval = _integer_field(
            env,
            "pendulum_failure_curriculum_refresh_interval_steps",
            default=20_000,
        )
        candidates = _integer_field(
            env, "pendulum_failure_curriculum_candidate_count", default=32
        )
        rollouts = _integer_field(
            env,
            "pendulum_failure_curriculum_rollouts_per_candidate",
            default=1,
        )
        horizon = _integer_field(
            env, "pendulum_failure_curriculum_rollout_horizon", default=200
        )
        if start < 0 or interval <= 0 or candidates <= 0 or rollouts <= 0 or horizon <= 0:
            raise ValueError(
                "active failure-curriculum start must be nonnegative and interval, "
                "candidate count, rollout count, and horizon must be positive"
            )
        first = start
        if first < 1:
            first += math.ceil((1 - first) / interval) * interval
        refresh_steps = list(range(first, learning_steps, interval))
        configured_episode_horizon = env.get("max_episode_steps")
        episode_horizon = (
            DEFAULT_HORIZON
            if configured_episode_horizon is None
            else _coerce_integer(
                configured_episode_horizon, "env.max_episode_steps"
            )
        )
        if episode_horizon <= 0:
            raise ValueError("env.max_episode_steps must be positive")
        effective_horizon = min(horizon, episode_horizon)
        per_refresh_upper_bound = candidates * rollouts * effective_horizon
        planned_upper_bound = len(refresh_steps) * per_refresh_upper_bound
    else:
        start = None
        interval = None
        candidates = None
        rollouts = None
        horizon = None
        effective_horizon = None
        per_refresh_upper_bound = 0
        refresh_steps = []
        planned_upper_bound = 0

    payload_value = complete_event.get("payload", {})
    if payload_value is None:
        payload_value = {}
    payload = _mapping(payload_value, "run_complete.payload")
    if "learning_environment_steps" in payload:
        recorded_learning = _coerce_nonnegative_integer(
            payload["learning_environment_steps"],
            "run_complete.payload.learning_environment_steps",
        )
        if recorded_learning != learning_steps:
            raise ValueError(
                "run_complete learning_environment_steps disagrees with step: "
                f"{recorded_learning} != {learning_steps}"
            )

    actual: int | None = None
    accounting_source: str
    if "failure_discovery_environment_steps" in payload:
        actual = _coerce_nonnegative_integer(
            payload["failure_discovery_environment_steps"],
            "run_complete.payload.failure_discovery_environment_steps",
        )
        accounting_source = "run_complete_actual"
    elif "learning_plus_failure_discovery_environment_steps" in payload:
        combined = _coerce_nonnegative_integer(
            payload["learning_plus_failure_discovery_environment_steps"],
            "run_complete.payload.learning_plus_failure_discovery_environment_steps",
        )
        actual = combined - learning_steps
        if actual < 0:
            raise ValueError(
                "run_complete learning+failure total is smaller than learning steps"
            )
        accounting_source = "run_complete_combined_total"
    elif active:
        accounting_source = "planned_upper_bound_fallback"
    else:
        actual = 0
        accounting_source = "disabled"

    if actual is not None and actual > planned_upper_bound:
        raise ValueError(
            "actual failure-discovery steps exceed the independently computed "
            f"planned upper bound: {actual} > {planned_upper_bound}"
        )
    if not active and actual not in (None, 0):
        raise ValueError(
            "run_complete records failure discovery while the curriculum is disabled"
        )

    if "failure_discovery_environment_steps_planned_upper_bound" in payload:
        recorded_planned = _coerce_nonnegative_integer(
            payload["failure_discovery_environment_steps_planned_upper_bound"],
            "run_complete.payload.failure_discovery_environment_steps_planned_upper_bound",
        )
        if recorded_planned != planned_upper_bound:
            raise ValueError(
                "recorded failure-discovery planned upper bound disagrees with the "
                f"resolved-config schedule: {recorded_planned} != {planned_upper_bound}"
            )
    if "learning_plus_failure_discovery_steps_planned_upper_bound" in payload:
        recorded_combined_planned = _coerce_nonnegative_integer(
            payload["learning_plus_failure_discovery_steps_planned_upper_bound"],
            "run_complete.payload.learning_plus_failure_discovery_steps_planned_upper_bound",
        )
        if recorded_combined_planned != learning_steps + planned_upper_bound:
            raise ValueError(
                "recorded learning+failure planned upper bound is inconsistent "
                "with the resolved-config schedule"
            )
    if actual is not None and "learning_plus_failure_discovery_environment_steps" in payload:
        combined = _coerce_nonnegative_integer(
            payload["learning_plus_failure_discovery_environment_steps"],
            "run_complete.payload.learning_plus_failure_discovery_environment_steps",
        )
        if combined != learning_steps + actual:
            raise ValueError(
                "recorded learning+failure total is inconsistent with its components"
            )
    counted = actual if actual is not None else planned_upper_bound
    return {
        "active": active,
        "reset_probability": probability,
        "configured_start_step": start,
        "refresh_interval_steps": interval,
        "planned_refresh_steps": refresh_steps,
        "planned_refresh_count": len(refresh_steps),
        "candidate_count": candidates,
        "rollouts_per_candidate": rollouts,
        "configured_rollout_horizon": horizon,
        "effective_rollout_horizon": effective_horizon,
        "environment_steps_per_refresh_upper_bound": per_refresh_upper_bound,
        "actual_environment_steps": actual,
        "counted_environment_steps": int(counted),
        "planned_environment_steps_upper_bound": int(planned_upper_bound),
        "accounting_source": accounting_source,
        "fail_closed_against_planned_upper_bound": True,
    }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _integer_field(
    mapping: dict[str, Any],
    name: str,
    *,
    default: int | None = None,
    required: bool = False,
) -> int:
    if name not in mapping:
        if required:
            raise ValueError(f"{name} is absent")
        assert default is not None
        return int(default)
    return _coerce_integer(mapping[name], name)


def _coerce_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != integer:
        raise ValueError(f"{name} must be an integer")
    return integer


def _coerce_nonnegative_integer(value: Any, name: str) -> int:
    integer = _coerce_integer(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative")
    return integer


def _float_field(
    mapping: dict[str, Any], name: str, *, default: float
) -> float:
    value = mapping.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def completed_run_event(path: Path) -> dict[str, Any] | None:
    complete: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "run_complete":
                try:
                    _coerce_integer(event.get("step"), "run_complete.step")
                except ValueError:
                    continue
                complete = event
    return complete


def completed_run_step(path: Path) -> int | None:
    event = completed_run_event(path)
    return (
        _coerce_integer(event["step"], "run_complete.step")
        if event is not None
        else None
    )


def pure_rl_config_errors(config: dict[str, Any]) -> list[str]:
    sac = config.get("sac", {})
    errors: list[str] = []
    inactive_modes = (
        "reference_guidance_mode",
        "reference_auxiliary_mode",
        "reference_critic_mode",
        "reference_prior_mode",
        "sac_actor_filter_mode",
    )
    for name in inactive_modes:
        if str(sac.get(name, "none")) != "none":
            errors.append(f"sac.{name}={sac.get(name)!r}")
    zero_fields = (
        "reference_guidance_probability",
        "reference_auxiliary_weight",
        "reference_anchor_ratio",
        "reference_anchor_size",
        "reference_critic_weight",
        "reference_prior_ratio",
        "reference_prior_dataset_steps",
        "pendulum_potential_shaping_weight",
    )
    for name in zero_fields:
        try:
            value = float(sac.get(name, 0.0))
        except (TypeError, ValueError):
            errors.append(f"sac.{name} is not numeric")
            continue
        if value != 0.0:
            errors.append(f"sac.{name}={value:g}")
    auxiliary_final = sac.get("reference_auxiliary_weight_final")
    if auxiliary_final is not None:
        try:
            final_value = float(auxiliary_final)
        except (TypeError, ValueError):
            errors.append("sac.reference_auxiliary_weight_final is not numeric")
        else:
            if final_value != 0.0:
                errors.append(
                    f"sac.reference_auxiliary_weight_final={final_value:g}"
                )
    if sac.get("actor_init_checkpoint_path") not in (None, ""):
        errors.append("sac.actor_init_checkpoint_path is set")
    env_id = str(config.get("env", {}).get("env_id", ""))
    if not env_id.startswith("Pendulum"):
        errors.append(f"env.env_id={env_id!r}")
    horizon = config.get("env", {}).get("max_episode_steps")
    if int(horizon or DEFAULT_HORIZON) != DEFAULT_HORIZON:
        errors.append(f"env.max_episode_steps={horizon!r}")
    return errors


def replay_reference_label_error(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as replay:
            for field in ("reference_actions", "reference_critic_actions"):
                if field in replay.files and np.isfinite(replay[field]).any():
                    return f"{path.name} contains finite {field}"
    except (OSError, ValueError) as exc:
        return f"cannot audit {path.name}: {exc}"
    return None


def evaluation_protocol(
    *,
    critic_search_batch_size: int,
    protocol_config: Path | None = None,
    points: int | None = None,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        protocol_config.resolve()
        if protocol_config is not None
        else (project_root / DEFAULT_PROTOCOL_CONFIG).resolve()
    )
    config = read_json(config_path)
    expected = {
        "schema_version",
        "workflow_version",
        "validation_dataset",
        "fixed_state_geometry",
        "ranking_rule",
        "selection_policy_variant",
        "reported_policy_variants",
        "evaluation_reliability",
        "reference_protocol",
    }
    if set(config) != expected:
        raise ValueError("pure off-grid protocol has missing or unexpected fields")
    if int(config["schema_version"]) != 1 or config["workflow_version"] != WORKFLOW_VERSION:
        raise ValueError("pure off-grid protocol version drift")
    if config["ranking_rule"] != RANKING_RULE:
        raise ValueError("pure off-grid ranking rule differs from preregistration")
    variant_names = [spec["name"] for spec in VARIANT_SPECS]
    if (
        config["selection_policy_variant"] != PRIMARY_VARIANT
        or config["reported_policy_variants"] != variant_names
    ):
        raise ValueError("pure off-grid policy variants differ from preregistration")
    reliability = ReliabilityConfig(**config["evaluation_reliability"])
    if dict(reliability.__dict__) != config["evaluation_reliability"]:
        raise ValueError("pure off-grid reliability protocol is not canonical")
    dataset = build_validation_dataset(config["validation_dataset"])
    if points is not None and int(points) != int(dataset["continuous_count"]):
        raise ValueError("--points cannot change the locked validation dataset")
    if rng_seed is not None and int(rng_seed) != int(
        dataset["spec"]["continuous_rng_seed"]
    ):
        raise ValueError("--rng-seed cannot change the locked validation dataset")
    if authoritative_grid_mask(dataset["theta"], dataset["velocity"]).any():
        raise ValueError("pure selection dataset intersects the 61x41 authority grid")
    if config["fixed_state_geometry"] != FIXED_STATE_GEOMETRY_SPEC:
        raise ValueError("pure fixed-state geometry protocol differs from preregistration")
    reference = config["reference_protocol"]
    if set(reference) != {
        "epsilon_return",
        "dp_solution",
        "authority_dp_grid",
        "authority_controller_grid",
    }:
        raise ValueError("pure reference protocol fields differ")
    if float(reference["epsilon_return"]) != 5.0:
        raise ValueError("pure off-grid epsilon_return must remain 5.0")
    dp_solution = _verify_pinned_file(project_root, reference["dp_solution"])
    for name in ("authority_dp_grid", "authority_controller_grid"):
        _validate_pinned_file_shape(reference[name], label=name)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "protocol_config": file_fingerprint(config_path),
        "state_distribution": "fixed_offgrid_midpoint_plus_continuous_reset_support",
        "dataset": {
            key: value
            for key, value in dataset.items()
            if key not in {"theta", "velocity"}
        },
        "points": int(dataset["points"]),
        "continuous_points": int(dataset["continuous_count"]),
        "rng_seed": int(dataset["spec"]["continuous_rng_seed"]),
        "theta_range": [-math.pi, math.pi],
        "velocity_range": [-DEFAULT_VELOCITY_LIMIT, DEFAULT_VELOCITY_LIMIT],
        "horizon": DEFAULT_HORIZON,
        "state_sha256": str(dataset["sha256"]),
        "authoritative_grid_intersection_count": 0,
        "authority_grid_selection_forbidden": True,
        "ranking_rule": list(RANKING_RULE),
        "variants": [dict(spec) for spec in VARIANT_SPECS],
        "locked_primary_policy": dict(LOCKED_PRIMARY_POLICY_SPEC),
        "fixed_state_geometry": dict(FIXED_STATE_GEOMETRY_SPEC),
        "evaluation_reliability": dict(reliability.__dict__),
        "critic_search_batch_size": int(critic_search_batch_size),
        "reference_protocol": {
            "epsilon_return": 5.0,
            "dp_solution": dp_solution,
            # These are carried as preregistered values but deliberately not
            # opened or verified until the post-fresh-seed authority handoff.
            "authority_dp_grid": dict(reference["authority_dp_grid"]),
            "authority_controller_grid": dict(
                reference["authority_controller_grid"]
            ),
        },
    }


def _validate_pinned_file_shape(value: Mapping[str, Any], *, label: str) -> None:
    if set(value) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"{label} fingerprint fields differ")
    if int(value["size_bytes"]) <= 0 or len(str(value["sha256"])) != 64:
        raise ValueError(f"{label} fingerprint is malformed")


def _verify_pinned_file(
    project_root: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_pinned_file_shape(value, label="off-grid DP solution")
    path = resolve_from(project_root, Path(str(value["path"])))
    actual = file_fingerprint(path)
    if actual["size_bytes"] != int(value["size_bytes"]) or actual["sha256"] != str(
        value["sha256"]
    ):
        raise ValueError(f"off-grid DP solution fingerprint drift: {path}")
    return actual


def frozen_validation_states(
    *,
    protocol: Mapping[str, Any] | None = None,
    points: int | None = None,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if protocol is None:
        locked = evaluation_protocol(
            critic_search_batch_size=512,
            points=points,
            rng_seed=rng_seed,
        )
    else:
        locked = dict(protocol)
    dataset = build_validation_dataset(locked["dataset"]["spec"])
    if dataset["sha256"] != locked["state_sha256"]:
        raise ValueError("locked off-grid state hash drift")
    if authoritative_grid_mask(dataset["theta"], dataset["velocity"]).any():
        raise ValueError("locked selection states intersect the authority grid")
    return np.asarray(dataset["theta"]), np.asarray(dataset["velocity"])


def state_sha256(theta: np.ndarray, velocity: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(theta, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(velocity, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def fixed_state_policy_geometry(
    agent: Any,
    theta: np.ndarray,
    velocity: np.ndarray,
    *,
    state_hash: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure actor and critic geometry on frozen initial states only.

    This function intentionally runs before any rollout or reference scoring.  It
    uses the checkpoint actor's raw normalized-observation outputs, rather than
    replay minibatches whose state distribution changes between treatments.
    """

    if dict(spec) != FIXED_STATE_GEOMETRY_SPEC:
        raise ValueError("fixed-state geometry spec drift")
    theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
    velocity_values = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if (
        theta_values.shape != velocity_values.shape
        or theta_values.size == 0
        or not np.isfinite(theta_values).all()
        or not np.isfinite(velocity_values).all()
    ):
        raise ValueError("fixed-state geometry inputs are malformed")
    if len(str(state_hash)) != 64:
        raise ValueError("fixed-state geometry state hash is malformed")

    raw_observations = np.stack(
        [np.cos(theta_values), np.sin(theta_values), velocity_values], axis=1
    ).astype(np.float32)
    mirrored_observations = raw_observations.copy()
    mirrored_observations[:, 1:] *= -1.0
    raw_tensor = torch.as_tensor(
        raw_observations, dtype=torch.float32, device=agent.device
    )
    mirrored_raw_tensor = torch.as_tensor(
        mirrored_observations, dtype=torch.float32, device=agent.device
    )

    with torch.no_grad():
        observations = agent._normalize_obs_tensor(raw_tensor)
        mirrored = agent._normalize_obs_tensor(mirrored_raw_tensor)
        if hasattr(agent.actor, "forward_with_unclamped_log_std"):
            actor_outputs = agent.actor.forward_with_unclamped_log_std(observations)
            if not isinstance(actor_outputs, tuple) or len(actor_outputs) != 3:
                raise ValueError(
                    "forward_with_unclamped_log_std must return "
                    "(mean, effective_log_std, unclamped_log_std)"
                )
            mean_logits, log_std, unclamped_log_std = actor_outputs
        else:
            mean_logits, log_std = agent.actor(observations)
            agent_config = getattr(agent, "cfg", None)
            configured_floor = getattr(
                agent_config,
                "simba_actor_log_std_floor",
                getattr(agent_config, "simba_actor_log_std_min", None),
            )
            if configured_floor is not None:
                raise ValueError(
                    "configured Simba log-std floor requires unclamped actor output"
                )
            # A historical checkpoint has no post-mapping floor, so its
            # effective and pre-floor values are identical by definition.
            unclamped_log_std = log_std
        mirrored_mean_logits, _mirrored_log_std = agent.actor(mirrored)
        for name, value in (
            ("mean logits", mean_logits),
            ("log std", log_std),
            ("unclamped log std", unclamped_log_std),
            ("mirrored mean logits", mirrored_mean_logits),
        ):
            if value.ndim != 2 or value.shape[0] != theta_values.size:
                raise ValueError(f"checkpoint actor {name} shape is malformed")
            if not torch.isfinite(value).all():
                raise ValueError(f"checkpoint actor {name} contains non-finite values")
        if (
            mean_logits.shape != log_std.shape
            or mean_logits.shape != unclamped_log_std.shape
            or mean_logits.shape != mirrored_mean_logits.shape
        ):
            raise ValueError("checkpoint actor output shapes differ")
        if torch.any(log_std + 1e-7 < unclamped_log_std):
            raise ValueError("effective actor log std lies below its pre-floor value")

        normalized_actions = torch.tanh(mean_logits)
        mirrored_normalized_actions = torch.tanh(mirrored_mean_logits)
        action_scale = torch.as_tensor(
            agent.actor.action_scale,
            dtype=mean_logits.dtype,
            device=mean_logits.device,
        ).reshape(1, -1)
        action_bias = torch.as_tensor(
            agent.actor.action_bias,
            dtype=mean_logits.dtype,
            device=mean_logits.device,
        ).reshape(1, -1)
        if action_scale.shape[1] != mean_logits.shape[1]:
            raise ValueError("checkpoint actor action scaling shape differs")
        deterministic_actions = normalized_actions * action_scale + action_bias
        mirrored_deterministic_actions = (
            mirrored_normalized_actions * action_scale + action_bias
        )

        absolute_logits = mean_logits.abs()
        logit_threshold = float(spec["mean_logit_abs_excess_threshold"])
        saturation_threshold = float(
            spec["deterministic_tanh_saturation_threshold"]
        )
        reflection_action_error = (
            deterministic_actions + mirrored_deterministic_actions
        ).abs()
        actor_geometry: dict[str, Any] = {
            "points": int(theta_values.size),
            "state_sha256": str(state_hash),
            "mean_logit_abs_mean": float(absolute_logits.mean().cpu()),
            "mean_logit_abs_max": float(absolute_logits.max().cpu()),
            "mean_logit_abs_gt_4p15_fraction": float(
                (absolute_logits > logit_threshold).float().mean().cpu()
            ),
            "deterministic_action_saturation_fraction_abs_ge_0p995": float(
                (normalized_actions.abs() >= saturation_threshold)
                .float()
                .mean()
                .cpu()
            ),
            "mean_tanh_derivative": float(
                (1.0 - normalized_actions.square()).mean().cpu()
            ),
            "log_std_mean": float(log_std.mean().cpu()),
            "log_std_min": float(log_std.min().cpu()),
            "log_std_below_minus_1p0_fraction": float(
                (log_std < -1.0).float().mean().cpu()
            ),
            "log_std_below_minus_1p5_fraction": float(
                (log_std < -1.5).float().mean().cpu()
            ),
            "log_std_below_minus_2p0_fraction": float(
                (log_std < -2.0).float().mean().cpu()
            ),
            "log_std_below_minus_3p0_fraction": float(
                (log_std < -3.0).float().mean().cpu()
            ),
            "unclamped_log_std_mean": float(unclamped_log_std.mean().cpu()),
            "unclamped_log_std_min": float(unclamped_log_std.min().cpu()),
            "unclamped_log_std_below_minus_1p0_fraction": float(
                (unclamped_log_std < -1.0).float().mean().cpu()
            ),
            "unclamped_log_std_below_minus_1p5_fraction": float(
                (unclamped_log_std < -1.5).float().mean().cpu()
            ),
            "unclamped_log_std_below_minus_2p0_fraction": float(
                (unclamped_log_std < -2.0).float().mean().cpu()
            ),
            "unclamped_log_std_below_minus_3p0_fraction": float(
                (unclamped_log_std < -3.0).float().mean().cpu()
            ),
            "log_std_floor_active_fraction": float(
                (log_std > unclamped_log_std + 1e-7).float().mean().cpu()
            ),
            "log_std_floor_lift_mean": float(
                (log_std - unclamped_log_std).mean().cpu()
            ),
            "reflection_action_abs_error_mean": float(
                reflection_action_error.mean().cpu()
            ),
            "reflection_action_abs_error_max": float(
                reflection_action_error.max().cpu()
            ),
        }

        q_values = torch.stack(
            [
                critic(observations, deterministic_actions).reshape(-1)
                for critic in agent.q_networks
            ],
            dim=0,
        )
        # Negating the original deterministic action isolates critic reflection
        # invariance from the actor's separately reported antisymmetry error.
        mirrored_q_values = torch.stack(
            [
                critic(mirrored, -deterministic_actions).reshape(-1)
                for critic in agent.q_networks
            ],
            dim=0,
        )
        if (
            q_values.ndim != 2
            or q_values.shape[1] != theta_values.size
            or q_values.shape != mirrored_q_values.shape
            or q_values.shape[0] < 2
            or not torch.isfinite(q_values).all()
            or not torch.isfinite(mirrored_q_values).all()
        ):
            raise ValueError("checkpoint critic geometry outputs are malformed")
        reflection_q_error = (
            q_values.mean(dim=0) - mirrored_q_values.mean(dim=0)
        ).abs()
        ensemble_q_std = q_values.std(dim=0, correction=0)
        critic_geometry: dict[str, Any] = {
            "points": int(theta_values.size),
            "state_sha256": str(state_hash),
            "critics": int(q_values.shape[0]),
            "reflection_q_abs_error_mean": float(reflection_q_error.mean().cpu()),
            "reflection_q_abs_error_max": float(reflection_q_error.max().cpu()),
            "ensemble_q_std_mean": float(ensemble_q_std.mean().cpu()),
            "ensemble_q_std_max": float(ensemble_q_std.max().cpu()),
        }
    validate_fixed_state_geometry(
        actor_geometry,
        critic_geometry,
        expected_points=int(theta_values.size),
        expected_state_hash=str(state_hash),
    )
    return actor_geometry, critic_geometry


def validate_fixed_state_geometry(
    actor: Mapping[str, Any],
    critic: Mapping[str, Any],
    *,
    expected_points: int,
    expected_state_hash: str,
) -> None:
    if set(actor) != FIXED_STATE_ACTOR_GEOMETRY_FIELDS:
        raise ValueError("fixed-state actor geometry fields differ")
    if set(critic) != FIXED_STATE_CRITIC_GEOMETRY_FIELDS:
        raise ValueError("fixed-state critic geometry fields differ")
    for label, payload in (("actor", actor), ("critic", critic)):
        if int(payload["points"]) != int(expected_points):
            raise ValueError(f"fixed-state {label} geometry point count differs")
        if str(payload["state_sha256"]) != str(expected_state_hash):
            raise ValueError(f"fixed-state {label} geometry state hash differs")
        numeric = [
            float(value)
            for key, value in payload.items()
            if key not in {"points", "state_sha256", "critics"}
        ]
        if not numeric or not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
            raise ValueError(f"fixed-state {label} geometry is non-finite")
    if int(critic["critics"]) < 2:
        raise ValueError("fixed-state critic geometry requires at least two critics")
    fraction_fields = {
        "mean_logit_abs_gt_4p15_fraction",
        "deterministic_action_saturation_fraction_abs_ge_0p995",
        "mean_tanh_derivative",
        "log_std_below_minus_1p0_fraction",
        "log_std_below_minus_1p5_fraction",
        "log_std_below_minus_2p0_fraction",
        "log_std_below_minus_3p0_fraction",
        "unclamped_log_std_below_minus_1p0_fraction",
        "unclamped_log_std_below_minus_1p5_fraction",
        "unclamped_log_std_below_minus_2p0_fraction",
        "unclamped_log_std_below_minus_3p0_fraction",
        "log_std_floor_active_fraction",
    }
    if any(not 0.0 <= float(actor[field]) <= 1.0 for field in fraction_fields):
        raise ValueError("fixed-state actor fraction lies outside [0, 1]")
    if float(actor["mean_logit_abs_mean"]) < 0.0 or float(
        actor["mean_logit_abs_max"]
    ) < float(actor["mean_logit_abs_mean"]):
        raise ValueError("fixed-state actor logit summary is inconsistent")
    if float(actor["log_std_min"]) > float(actor["log_std_mean"]):
        raise ValueError("fixed-state actor log-std summary is inconsistent")
    if float(actor["unclamped_log_std_min"]) > float(
        actor["unclamped_log_std_mean"]
    ):
        raise ValueError("fixed-state actor unclamped log-std summary is inconsistent")
    if float(actor["log_std_mean"]) + 1e-7 < float(
        actor["unclamped_log_std_mean"]
    ) or float(actor["log_std_min"]) + 1e-7 < float(
        actor["unclamped_log_std_min"]
    ):
        raise ValueError("fixed-state actor effective log std is below unclamped")
    if float(actor["log_std_floor_lift_mean"]) < -1e-7:
        raise ValueError("fixed-state actor log-std floor lift is negative")
    for prefix in ("log_std", "unclamped_log_std"):
        threshold_fractions = [
            float(actor[f"{prefix}_below_minus_1p0_fraction"]),
            float(actor[f"{prefix}_below_minus_1p5_fraction"]),
            float(actor[f"{prefix}_below_minus_2p0_fraction"]),
            float(actor[f"{prefix}_below_minus_3p0_fraction"]),
        ]
        if any(
            lower + 1e-12 < upper
            for lower, upper in zip(
                threshold_fractions, threshold_fractions[1:]
            )
        ):
            raise ValueError(
                f"fixed-state actor {prefix} threshold fractions are inconsistent"
            )
    for suffix in (
        "minus_1p0_fraction",
        "minus_1p5_fraction",
        "minus_2p0_fraction",
        "minus_3p0_fraction",
    ):
        if float(actor[f"log_std_below_{suffix}"]) > float(
            actor[f"unclamped_log_std_below_{suffix}"]
        ) + 1e-12:
            raise ValueError(
                "fixed-state actor effective lower-tail fraction exceeds unclamped"
            )
    if (
        float(actor["log_std_floor_active_fraction"]) == 0.0
        and float(actor["log_std_floor_lift_mean"]) > 1e-7
    ):
        raise ValueError("fixed-state actor floor lift has no active states")
    for mean_field, max_field in (
        ("reflection_action_abs_error_mean", "reflection_action_abs_error_max"),
        ("reflection_q_abs_error_mean", "reflection_q_abs_error_max"),
        ("ensemble_q_std_mean", "ensemble_q_std_max"),
    ):
        payload = actor if mean_field.startswith("reflection_action") else critic
        if float(payload[mean_field]) < 0.0 or float(payload[max_field]) < float(
            payload[mean_field]
        ):
            raise ValueError(f"fixed-state geometry {mean_field} is inconsistent")


def checkpoint_identity(run_dir: Path) -> dict[str, Any]:
    checkpoint = (run_dir / "checkpoints" / "final.pt").resolve()
    config_path = (run_dir / "config.json").resolve()
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest(
    condition: Condition, protocol: dict[str, Any]
) -> dict[str, Any]:
    return {
        "condition": condition.label,
        "kind": condition.kind,
        "run_dir": str(condition.run_dir.resolve()),
        "completed_environment_steps": condition.completed_steps,
        "checkpoint": checkpoint_identity(condition.run_dir),
        "protocol": protocol,
        "authority_grid_queried": False,
        "reference_policy_queried": False,
    }


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_evaluation(
    condition: Condition,
    evaluation_dir: Path,
    protocol: dict[str, Any],
) -> None:
    manifest_path = evaluation_dir / "manifest.json"
    result_path = evaluation_dir / "evaluation.json"
    rollout_path = evaluation_dir / "rollouts.npz"
    missing = [
        path.name
        for path in (manifest_path, result_path, rollout_path)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "Refusing partial/unrecognized evaluation directory "
            f"{evaluation_dir}; missing {', '.join(missing)}"
        )
    expected = expected_manifest(condition, protocol)
    actual = read_json(manifest_path)
    try:
        if set(actual.get("artifacts", {})) != {"evaluation.json", "rollouts.npz"}:
            raise ValueError("candidate cache artifact set differs")
        validate_artifact_manifest(
            actual,
            expected_guard=expected,
            root=evaluation_dir,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(
            "Refusing stale/tampered evaluation (checkpoint, config, frozen "
            f"protocol, or artifact changed): {evaluation_dir}: {exc}"
        ) from exc
    result = read_json(result_path)
    names = [row.get("variant") for row in result.get("variants", [])]
    expected_names = [spec["name"] for spec in VARIANT_SPECS]
    if result.get("guard") != expected or names != expected_names:
        raise SystemExit(f"Refusing malformed/stale evaluation result: {result_path}")
    try:
        validate_fixed_state_geometry(
            result["fixed_state_actor_geometry"],
            result["fixed_state_critic_geometry"],
            expected_points=int(protocol["points"]),
            expected_state_hash=str(protocol["state_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Refusing malformed fixed-state geometry: {result_path}: {exc}"
        ) from exc


def offgrid_reference_returns(
    *,
    evaluation_root: Path,
    protocol: Mapping[str, Any],
    create_if_missing: bool,
) -> dict[str, Any] | None:
    """Read or create the fixed off-grid reference cache after policy rollout.

    This function deliberately has no authority-grid argument.  It uses only the
    pinned finite-horizon DP solution on the fixed off-grid states and carries an
    explicit false authority-query bit in its cache guard.
    """

    cache_dir = evaluation_root / "_offgrid_reference_v2"
    manifest_path = cache_dir / "manifest.json"
    returns_path = cache_dir / "reference_returns.npz"
    summary_path = cache_dir / "reference_summary.json"
    guard = {
        "workflow_version": WORKFLOW_VERSION,
        "cache_kind": "fixed_offgrid_best_of_dp_and_controller_reference",
        "protocol_config": dict(protocol["protocol_config"]),
        "dataset_sha256": str(protocol["state_sha256"]),
        "dataset_points": int(protocol["points"]),
        "horizon": int(protocol["horizon"]),
        "evaluation_reliability": dict(protocol["evaluation_reliability"]),
        "epsilon_return": float(protocol["reference_protocol"]["epsilon_return"]),
        "dp_solution": dict(protocol["reference_protocol"]["dp_solution"]),
        "authority_grid_queried": False,
    }
    if cache_dir.exists():
        required = (manifest_path, returns_path, summary_path)
        if any(not path.is_file() for path in required):
            raise SystemExit(f"Refusing partial off-grid reference cache: {cache_dir}")
        manifest = read_json(manifest_path)
        try:
            if set(manifest.get("artifacts", {})) != {
                "reference_returns.npz",
                "reference_summary.json",
            }:
                raise ValueError("reference cache artifact set differs")
            validate_artifact_manifest(
                manifest,
                expected_guard=guard,
                root=cache_dir,
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"Refusing stale/tampered off-grid reference cache {cache_dir}: {exc}"
            ) from exc
        summary = read_json(summary_path)
        if summary.get("guard") != guard:
            raise SystemExit(f"Refusing malformed off-grid reference cache: {summary_path}")
        with np.load(returns_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"theta", "theta_dot", "reference_return"}:
                raise SystemExit(
                    f"Refusing malformed off-grid reference arrays: {returns_path}"
                )
            theta = np.asarray(arrays["theta"], dtype=np.float64)
            velocity = np.asarray(arrays["theta_dot"], dtype=np.float64)
            returns = np.asarray(arrays["reference_return"], dtype=np.float64)
        expected_theta, expected_velocity = frozen_validation_states(protocol=protocol)
        if (
            theta.shape != expected_theta.shape
            or velocity.shape != expected_velocity.shape
            or returns.shape != expected_theta.shape
            or not np.array_equal(theta, expected_theta)
            or not np.array_equal(velocity, expected_velocity)
            or not np.isfinite(returns).all()
        ):
            raise SystemExit(f"Refusing drifted off-grid reference arrays: {returns_path}")
        return {
            "returns": returns,
            "identity": {
                "manifest": file_fingerprint(manifest_path),
                "guard_sha256": sha256_json(guard),
                "reference_returns_artifact": dict(
                    manifest["artifacts"]["reference_returns.npz"]
                ),
                "authority_grid_queried": False,
            },
        }

    if not create_if_missing:
        return None
    evaluation_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="._offgrid_reference_v2.tmp-", dir=evaluation_root)
    )
    try:
        theta, velocity = frozen_validation_states(protocol=protocol)
        reliability = ReliabilityConfig(**protocol["evaluation_reliability"])
        detector = UprightDetector(
            "Pendulum-v1",
            cos_threshold=reliability.near_upright_cos_threshold,
            abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
        )
        dp_solution = Path(
            str(protocol["reference_protocol"]["dp_solution"]["path"])
        ).resolve()
        returns = validation_reference_returns(
            theta,
            velocity,
            detector,
            reliability,
            dp_solution,
        )
        returns = np.asarray(returns, dtype=np.float64)
        if returns.shape != theta.shape or not np.isfinite(returns).all():
            raise ValueError("computed off-grid reference returns are malformed")
        np.savez_compressed(
            temporary / "reference_returns.npz",
            theta=theta,
            theta_dot=velocity,
            reference_return=returns,
        )
        write_json_exclusive(
            temporary / "reference_summary.json",
            {
                "guard": guard,
                "points": int(len(returns)),
                "mean_return": float(returns.mean()),
                "bottom10_conditional_mean_return": conditional_mean(returns, 0.10),
                "authority_grid_queried": False,
            },
        )
        manifest = {
            "guard": guard,
            "artifacts": {
                name: _artifact_entry(temporary / name)
                for name in ("reference_returns.npz", "reference_summary.json")
            },
        }
        write_json_exclusive(temporary / "manifest.json", manifest)
        os.replace(temporary, cache_dir)
    except Exception:
        if temporary.exists() and temporary.parent.resolve() == evaluation_root.resolve():
            shutil.rmtree(temporary)
        raise
    return offgrid_reference_returns(
        evaluation_root=evaluation_root,
        protocol=protocol,
        create_if_missing=False,
    )


def evaluate_condition_new(
    *,
    condition: Condition,
    evaluation_dir: Path,
    evaluation_root: Path,
    protocol: dict[str, Any],
    device: str,
) -> None:
    if evaluation_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation_dir}")
    evaluation_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{safe_label(condition.label)}.tmp-", dir=evaluation_root
        )
    )
    try:
        guard = expected_manifest(condition, protocol)
        evaluation, rollout_arrays = run_frozen_evaluation(
            condition=condition,
            protocol=protocol,
            device=device,
        )
        evaluation["guard"] = guard
        write_json_exclusive(temporary / "evaluation.json", evaluation)
        np.savez_compressed(temporary / "rollouts.npz", **rollout_arrays)
        manifest = {
            "guard": guard,
            "artifacts": {
                name: _artifact_entry(temporary / name)
                for name in ("evaluation.json", "rollouts.npz")
            },
        }
        # The manifest is the commit record and is written only after every
        # payload exists and has been hashed.
        write_json_exclusive(temporary / "manifest.json", manifest)
        os.replace(temporary, evaluation_dir)
    except Exception:
        if temporary.exists() and temporary.parent.resolve() == evaluation_root.resolve():
            shutil.rmtree(temporary)
        raise


def run_frozen_evaluation(
    *, condition: Condition, protocol: dict[str, Any], device: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    agent, config, _payload = load_agent_from_run(condition.run_dir, device=device)
    reliability = ReliabilityConfig(**protocol["evaluation_reliability"])
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    theta, velocity = frozen_validation_states(protocol=protocol)
    actor_geometry, critic_geometry = fixed_state_policy_geometry(
        agent,
        theta,
        velocity,
        state_hash=str(protocol["state_sha256"]),
        spec=protocol["fixed_state_geometry"],
    )
    policies: list[tuple[dict[str, Any], Any]] = [
        (VARIANT_SPECS[0], agent),
        (VARIANT_SPECS[1], ReflectionActorPolicy(agent)),
        (
            VARIANT_SPECS[2],
            TrackedReflectionGlobalQSearchPolicy(
                agent,
                num_actions=41,
                margin=0.005,
                batch_size=int(protocol["critic_search_batch_size"]),
            ),
        ),
    ]
    variants: list[dict[str, Any]] = []
    rollout_arrays: dict[str, np.ndarray] = {
        "theta": theta,
        "theta_dot": velocity,
    }
    baseline_returns: dict[str, np.ndarray] = {}
    for spec, policy in policies:
        rows = rollout_pendulum_grid_vectorized(
            policy,
            theta,
            velocity,
            detector,
            reliability,
            horizon=DEFAULT_HORIZON,
        )
        name = str(spec["name"])
        returns = np.asarray([float(row["return"]) for row in rows], dtype=np.float64)
        task = np.asarray([bool(row["task_success"]) for row in rows], dtype=np.bool_)
        near = np.asarray(
            [float(row["near_upright_fraction"]) for row in rows], dtype=np.float64
        )
        rollout_arrays[f"{name}_return"] = returns
        rollout_arrays[f"{name}_task_success"] = task
        rollout_arrays[f"{name}_near_upright_fraction"] = near
        baseline_returns[name] = returns
        selection = (
            policy.selection_metrics()
            if hasattr(policy, "selection_metrics")
            else {}
        )
        variants.append(
            {
                "variant": name,
                "spec": dict(spec),
                "metrics": offgrid_metrics(returns, task, near),
                "selection": selection,
            }
        )
    ordinary = baseline_returns["ordinary_actor"]
    reflection = baseline_returns["reflection_actor"]
    for row in variants:
        returns = baseline_returns[row["variant"]]
        row["paired_delta_vs_ordinary_actor"] = paired_return_deltas(
            returns, ordinary
        )
        row["paired_delta_vs_reflection_actor"] = paired_return_deltas(
            returns, reflection
        )
    return {
        "fixed_state_actor_geometry": actor_geometry,
        "fixed_state_critic_geometry": critic_geometry,
        "variants": variants,
    }, rollout_arrays


def offgrid_metrics(
    returns: np.ndarray,
    task: np.ndarray,
    near_fraction: np.ndarray,
    *,
    reference_returns: np.ndarray | None = None,
    epsilon_return: float = 5.0,
) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=np.float64)
    metrics: dict[str, float | int] = {
        "points": int(len(returns)),
        "mean_return": float(np.mean(values)),
        "median_return": float(np.median(values)),
        "return_p01": float(np.quantile(values, 0.01)),
        "return_p05": float(np.quantile(values, 0.05)),
        "return_p10": float(np.quantile(values, 0.10)),
        "bottom10_conditional_mean_return": conditional_mean(values, 0.10),
        "task_successes": int(np.sum(task)),
        "task_success": float(np.mean(task)),
        "mean_near_upright_fraction": float(np.mean(near_fraction)),
    }
    if reference_returns is not None:
        reference = np.asarray(reference_returns, dtype=np.float64)
        if reference.shape != values.shape or not np.isfinite(reference).all():
            raise ValueError("off-grid reference returns are malformed")
        near_reference = values >= reference - float(epsilon_return)
        strict = values > reference
        regret = reference - values
        metrics.update(
            {
                "near_reference_epsilon_return": float(epsilon_return),
                "near_reference_successes": int(near_reference.sum()),
                "near_reference_success": float(near_reference.mean()),
                "strict_beats_reference_successes": int(strict.sum()),
                "strict_beats_reference": float(strict.mean()),
                "mean_regret_to_reference": float(regret.mean()),
                "regret_to_reference_p95": float(np.quantile(regret, 0.95)),
            }
        )
    return metrics


def add_reference_metrics_to_variants(
    variants: list[dict[str, Any]],
    rollout_path: Path,
    reference_returns: np.ndarray,
    *,
    epsilon_return: float,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    with np.load(rollout_path, allow_pickle=False) as rollouts:
        expected_points = int(reference_returns.shape[0])
        for row in variants:
            name = str(row["variant"])
            keys = (
                f"{name}_return",
                f"{name}_task_success",
                f"{name}_near_upright_fraction",
            )
            if any(key not in rollouts.files for key in keys):
                raise ValueError(f"candidate rollout cache is missing {name!r} arrays")
            returns = np.asarray(rollouts[keys[0]], dtype=np.float64)
            task = np.asarray(rollouts[keys[1]], dtype=np.bool_)
            near = np.asarray(rollouts[keys[2]], dtype=np.float64)
            if any(array.shape != (expected_points,) for array in (returns, task, near)):
                raise ValueError(f"candidate rollout cache has wrong {name!r} shape")
            updated = dict(row)
            updated["metrics"] = offgrid_metrics(
                returns,
                task,
                near,
                reference_returns=reference_returns,
                epsilon_return=epsilon_return,
            )
            enriched.append(updated)
    return enriched


def conditional_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    return float(np.sort(values)[:count].mean())


def paired_return_deltas(
    returns: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    delta = np.asarray(returns, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "improved_fraction": float(np.mean(delta > 0.0)),
        "degraded_fraction": float(np.mean(delta < 0.0)),
        "degraded_by_1_fraction": float(np.mean(delta < -1.0)),
        "degraded_by_5_fraction": float(np.mean(delta < -5.0)),
    }


def summarize_training_diagnostics(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    sac = config.get("sac", {})
    categorized = categorized_metric_summaries(run_dir / "metrics.csv")
    representation = representation_aggregate(categorized["representation"])
    replay = replay_snapshot_summary(run_dir / "replay_final.npz", config)
    if replay.get("finite_reference_action_fraction", 0.0) > 0.0 or replay.get(
        "finite_reference_critic_action_fraction", 0.0
    ) > 0.0:
        raise ValueError(f"Pure-RL replay contains teacher labels: {run_dir}")
    return {
        # Bind every read-only diagnostic source into the ranking report.  A
        # later promotion analysis can therefore refuse post-ranking edits
        # without loading the checkpoint or replay arrays.
        "artifact_fingerprints": {
            name: file_fingerprint(run_dir / name)
            for name in (
                "config.json",
                "events.jsonl",
                "metrics.csv",
                "eval_episodes.csv",
                "replay_final.npz",
            )
            if (run_dir / name).is_file()
        },
        "config": {
            "optimization": select_fields(
                sac,
                (
                    "total_steps",
                    "learning_starts",
                    "batch_size",
                    "gamma",
                    "tau",
                    "policy_lr",
                    "policy_lr_final",
                    "q_lr",
                    "q_lr_final",
                    "updates_per_step",
                    "policy_frequency",
                    "actor_updates_per_trigger",
                    "sacn_n_step",
                    "sacn_target_mode",
                    "sacn_horizon_lambda",
                    "sacn_stop_after_steps",
                ),
            ),
            "architecture": select_fields(
                sac,
                (
                    "simba_backbone",
                    "simba_actor_hidden_dim",
                    "simba_actor_blocks",
                    "simba_critic_hidden_dim",
                    "simba_critic_blocks",
                    "simba_distributional_critic",
                    "simba_reward_scaling",
                    "simba_weight_projection",
                    "simba_feature_norm",
                    "simba_observation_norm",
                ),
            ),
            "prioritized_replay": select_fields(
                sac,
                (
                    "replay_priority_mode",
                    "replay_priority_alpha",
                    "replay_priority_beta_initial",
                    "replay_priority_beta_final",
                    "replay_priority_beta_anneal_steps",
                    "replay_priority_uniform_fraction",
                    "replay_priority_epsilon",
                    "replay_priority_clip",
                ),
            ),
            "q_distillation": select_fields(
                sac,
                (
                    "critic_search_actor_weight",
                    "critic_search_num_actions",
                    "critic_search_margin",
                    "critic_search_start_update",
                    "critic_search_filter_mode",
                ),
            ),
            "self_imitation": select_fields(
                sac,
                (
                    "self_imitation_weight",
                    "self_imitation_loss_type",
                    "self_imitation_start_step",
                    "self_imitation_temperature",
                    "self_imitation_margin",
                    "self_imitation_max_weight",
                ),
            ),
            "symmetry": select_fields(
                sac,
                (
                    "pendulum_symmetry_augmentation",
                    "pendulum_actor_symmetry_weight",
                    "pendulum_critic_symmetry_weight",
                ),
            ),
            "replay_and_representation": select_fields(
                sac,
                (
                    "buffer_size",
                    "swd_linear_decay_steps",
                    "swd_min_weight",
                    "redo_interval_updates",
                    "redo_dormant_threshold",
                    "pendulum_hard_replay_fraction",
                    "pendulum_model_replay_ratio",
                    "pendulum_model_rollout_ratio",
                ),
            ),
        },
        "metric_series": categorized,
        "representation_aggregate": representation,
        "replay_snapshot": replay,
    }


def select_fields(mapping: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    return {name: mapping.get(name) for name in names}


def categorized_metric_summaries(path: Path) -> dict[str, dict[str, Any]]:
    categories: dict[str, dict[str, Any]] = {
        "optimization_losses": {},
        "prioritized_replay": {},
        "q_distillation": {},
        "self_imitation": {},
        "symmetry": {},
        "representation": {},
        "replay": {},
    }
    if not path.is_file():
        return categories
    rows: list[tuple[int, str, str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                step = int(float(row["step"]))
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                rows.append((step, str(row.get("split", "")), str(row.get("name", "")), value))
    all_names = {name for _step, _split, name, _value in rows}
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for step, split, name, value in rows:
        if is_window_derivative(name, all_names):
            continue
        category = metric_category(name, split)
        if category is None:
            continue
        grouped.setdefault((category, name), []).append((step, value))
    for (category, name), values in grouped.items():
        values.sort(key=lambda item: item[0])
        categories[category][name] = series_summary(values)
    return categories


def is_window_derivative(name: str, all_names: set[str]) -> bool:
    for suffix in ("_mean", "_min", "_max"):
        if name.endswith(suffix) and name[: -len(suffix)] in all_names:
            return True
    return False


def metric_category(name: str, split: str) -> str | None:
    if name.startswith("replay_priority_") or name.startswith("priority_"):
        return "prioritized_replay"
    if name.startswith("critic_search_"):
        return "q_distillation"
    if name.startswith("self_imitation_"):
        return "self_imitation"
    if name.startswith("pendulum_symmetry_") or name.startswith(
        "pendulum_actor_symmetry_"
    ) or name.startswith("pendulum_critic_symmetry_"):
        return "symmetry"
    if any(
        token in name
        for token in (
            "dormant_fraction",
            "effective_rank_fraction",
            "activation_abs_",
            "param_norm",
            "redo_",
        )
    ):
        return "representation"
    if split == "replay" or name.startswith(
        ("pendulum_hard_", "pendulum_model_", "reference_prior_")
    ) or name in {
        "size",
        "capacity",
        "fill_fraction",
        "sample_count_mean",
        "sample_count_max",
        "transition_age_mean",
        "transition_age_max",
        "action_abs_mean",
        "action_abs_max",
        "action_saturation_fraction",
        "reward_mean",
        "reward_std",
        "near_upright_obs_fraction",
    }:
        return "replay"
    if name in {
        "actor_loss",
        "sac_actor_loss",
        "q_loss",
        "q1_loss",
        "q2_loss",
        "alpha",
        "alpha_loss",
        "target_q_mean",
        "q1_mean",
        "q2_mean",
        "policy_entropy_estimate",
        "policy_log_prob_mean",
    }:
        return "optimization_losses"
    return None


def series_summary(values: list[tuple[int, float]]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "first_step": None,
            "last_step": None,
            "first": None,
            "last": None,
            "mean": None,
            "tail_mean": None,
            "min": None,
            "max": None,
            "least_squares_slope_per_10k_steps": None,
        }
    steps = np.asarray([step for step, _value in values], dtype=np.float64)
    samples = np.asarray([value for _step, value in values], dtype=np.float64)
    tail_count = max(1, int(math.ceil(len(samples) * 0.2)))
    if len(samples) >= 2 and np.ptp(steps) > 0.0:
        slope = float(np.polyfit(steps, samples, 1)[0] * 10_000.0)
    else:
        slope = None
    return {
        "count": int(len(samples)),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first": float(samples[0]),
        "last": float(samples[-1]),
        "mean": float(samples.mean()),
        "tail_mean": float(samples[-tail_count:].mean()),
        "min": float(samples.min()),
        "max": float(samples.max()),
        "least_squares_slope_per_10k_steps": slope,
    }


def representation_aggregate(series: dict[str, Any]) -> dict[str, Any]:
    dormant = [value for name, value in series.items() if "dormant_fraction" in name]
    ranks = [value for name, value in series.items() if "effective_rank_fraction" in name]
    return {
        "num_dormancy_series": len(dormant),
        "latest_max_dormant_fraction": max_or_none(item["last"] for item in dormant),
        "ever_max_dormant_fraction": max_or_none(item["max"] for item in dormant),
        "num_effective_rank_series": len(ranks),
        "latest_min_effective_rank_fraction": min_or_none(item["last"] for item in ranks),
        "ever_min_effective_rank_fraction": min_or_none(item["min"] for item in ranks),
    }


def replay_snapshot_summary(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    with np.load(path, allow_pickle=False) as replay:
        files = set(replay.files)
        observations = np.asarray(replay["observations"], dtype=np.float64)
        actions = np.asarray(replay["actions"], dtype=np.float64).reshape(-1)
        rewards = np.asarray(replay["rewards"], dtype=np.float64).reshape(-1)
        sample_counts = (
            np.asarray(replay["sample_counts"], dtype=np.float64).reshape(-1)
            if "sample_counts" in files
            else np.zeros(len(actions), dtype=np.float64)
        )
        theta = np.arctan2(observations[:, 1], observations[:, 0])
        velocity = observations[:, 2]
        hard = (np.abs(theta) >= 2.0943951023931953) & (np.abs(velocity) <= 1.0)
        boundary = np.abs(theta) >= 2.6179938779914944
        hist, _edges = np.histogramdd(
            np.stack([theta, velocity], axis=1),
            bins=(24, 8),
            range=((-math.pi, math.pi), (-1.0, 1.0)),
        )
        output: dict[str, Any] = {
            "present": True,
            "path": str(path),
            "transitions": int(len(actions)),
            "hard_reset_support_fraction_abs_theta_ge_120": float(np.mean(hard)),
            "boundary_fraction_abs_theta_ge_150": float(np.mean(boundary)),
            "reset_support_24x8_occupied_fraction": float(np.mean(hist > 0.0)),
            "action_abs_mean": float(np.mean(np.abs(actions))),
            "action_saturation_fraction_ge_1p9": float(np.mean(np.abs(actions) >= 1.9)),
            "reward_mean": float(np.mean(rewards)),
            "reward_p05": float(np.quantile(rewards, 0.05)),
            "sample_count_mean": float(np.mean(sample_counts)),
            "sample_count_p95": float(np.quantile(sample_counts, 0.95)),
            "sample_count_top10_share": top_fraction_share(sample_counts, 0.10),
            "sample_count_hard_mean": float(np.mean(sample_counts[hard]))
            if hard.any()
            else None,
            "sample_count_nonhard_mean": float(np.mean(sample_counts[~hard]))
            if (~hard).any()
            else None,
            "sample_count_hard_to_nonhard_ratio": (
                float(np.mean(sample_counts[hard]))
                / max(float(np.mean(sample_counts[~hard])), 1e-12)
                if hard.any() and (~hard).any()
                else None
            ),
            "finite_reference_action_fraction": finite_fraction(
                replay["reference_actions"] if "reference_actions" in files else None
            ),
            "finite_reference_critic_action_fraction": finite_fraction(
                replay["reference_critic_actions"]
                if "reference_critic_actions" in files
                else None
            ),
        }
        if "priorities" in files:
            priorities = np.asarray(replay["priorities"], dtype=np.float64).reshape(-1)
            output.update(
                {
                    "priority_mean": float(np.mean(priorities)),
                    "priority_p95": float(np.quantile(priorities, 0.95)),
                    "priority_max": float(np.max(priorities)),
                    "priority_top10_share": top_fraction_share(priorities, 0.10),
                    "priority_hard_mean": float(np.mean(priorities[hard]))
                    if hard.any()
                    else None,
                    "priority_nonhard_mean": float(np.mean(priorities[~hard]))
                    if (~hard).any()
                    else None,
                }
            )
    output["configured_buffer_size"] = config.get("sac", {}).get("buffer_size")
    return output


def finite_fraction(values: Any | None) -> float:
    if values is None:
        return 0.0
    array = np.asarray(values)
    return float(np.mean(np.isfinite(array))) if array.size else 0.0


def top_fraction_share(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(values))
    if len(values) == 0 or total <= 0.0:
        return 0.0
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.sort(values)[-count:].sum() / total)


def max_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def min_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return min(finite) if finite else None


def primary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["primary_metrics"]
    return (
        -int(metrics["near_reference_successes"]),
        -int(metrics["task_successes"]),
        -int(metrics["strict_beats_reference_successes"]),
        -float(metrics["bottom10_conditional_mean_return"]),
        -float(metrics["mean_return"]),
        row["condition"],
    )


def rank_primary_variants(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        by_name = {row["variant"]: row for row in condition["variants"]}
        primary = by_name[PRIMARY_VARIANT]
        rows.append(
            {
                **condition,
                "primary_variant": PRIMARY_VARIANT,
                "primary_metrics": primary["metrics"],
                "primary_selection": primary["selection"],
                "q_delta_vs_reflection_actor": primary[
                    "paired_delta_vs_reflection_actor"
                ],
                "reflection_delta_vs_ordinary_actor": by_name["reflection_actor"][
                    "paired_delta_vs_ordinary_actor"
                ],
            }
        )
    ranked = sorted(rows, key=primary_sort_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def rank_all_variants(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for variant in condition["variants"]:
            rows.append(
                {
                    "condition": condition["condition"],
                    "kind": condition["kind"],
                    "run_dir": condition["run_dir"],
                    "training_budget": condition.get("training_budget"),
                    "fixed_state_actor_geometry": condition.get(
                        "fixed_state_actor_geometry"
                    ),
                    "fixed_state_critic_geometry": condition.get(
                        "fixed_state_critic_geometry"
                    ),
                    **variant,
                }
            )
    rows.sort(
        key=lambda row: (
            -int(row["metrics"]["near_reference_successes"]),
            -int(row["metrics"]["task_successes"]),
            -int(row["metrics"]["strict_beats_reference_successes"]),
            -float(row["metrics"]["bottom10_conditional_mean_return"]),
            -float(row["metrics"]["mean_return"]),
            row["condition"],
            row["variant"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def add_baseline_deltas(
    ranked_arms: list[dict[str, Any]], ranked_variants: list[dict[str, Any]]
) -> None:
    baseline_arm = next(
        (row for row in ranked_arms if row.get("kind") == "baseline"), None
    )
    if baseline_arm is not None:
        baseline = baseline_arm["primary_metrics"]
        for row in ranked_arms:
            metrics = row["primary_metrics"]
            row["delta_vs_baseline_primary"] = metric_delta(metrics, baseline)

    baseline_variants = {
        row["variant"]: row
        for row in ranked_variants
        if row.get("kind") == "baseline"
    }
    for row in ranked_variants:
        baseline_row = baseline_variants.get(row["variant"])
        if baseline_row is not None:
            row["delta_vs_baseline_same_variant"] = metric_delta(
                row["metrics"], baseline_row["metrics"]
            )


def metric_delta(
    metrics: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float | int]:
    return {
        "near_reference_successes": int(metrics["near_reference_successes"])
        - int(baseline["near_reference_successes"]),
        "near_reference_success": float(metrics["near_reference_success"])
        - float(baseline["near_reference_success"]),
        "strict_beats_reference_successes": int(
            metrics["strict_beats_reference_successes"]
        )
        - int(baseline["strict_beats_reference_successes"]),
        "strict_beats_reference": float(metrics["strict_beats_reference"])
        - float(baseline["strict_beats_reference"]),
        "mean_return": float(metrics["mean_return"])
        - float(baseline["mean_return"]),
        "bottom10_conditional_mean_return": float(
            metrics["bottom10_conditional_mean_return"]
        )
        - float(baseline["bottom10_conditional_mean_return"]),
        "task_successes": int(metrics["task_successes"])
        - int(baseline["task_successes"]),
        "task_success": float(metrics["task_success"])
        - float(baseline["task_success"]),
        "mean_near_upright_fraction": float(metrics["mean_near_upright_fraction"])
        - float(baseline["mean_near_upright_fraction"]),
    }


def safe_label(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError(f"Condition label has no safe filename characters: {value!r}")
    return cleaned


def assert_report_outputs_absent(output_dir: Path) -> None:
    existing = [
        output_dir / name for name in REPORT_FILENAMES if (output_dir / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite report output(s): "
            + ", ".join(str(path) for path in existing)
        )


def write_report_new(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_report_outputs_absent(output_dir)
    write_json_exclusive(output_dir / REPORT_FILENAMES[0], report)
    write_ranking_csv(output_dir / REPORT_FILENAMES[1], report["ranked_arms"])
    write_variant_csv(output_dir / REPORT_FILENAMES[2], report["ranked_variants"])
    write_text_exclusive(output_dir / REPORT_FILENAMES[3], markdown_report(report))


def write_json_exclusive(path: Path, value: Any) -> None:
    write_text_exclusive(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_text_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "rank",
        "condition",
        "kind",
        "completed_environment_steps",
        "learning_environment_steps",
        "model_generated_transitions",
        "model_replay_generated_transitions",
        "model_rollout_generated_transitions",
        "failure_discovery_environment_steps_actual",
        "failure_discovery_environment_steps_counted",
        "failure_discovery_environment_steps_planned_upper_bound",
        "strict_total_steps_actual",
        "strict_total_steps_counted",
        "strict_total_steps_planned_upper_bound",
        "strict_total_steps_for_eligibility",
        "near_reference_successes",
        "near_reference_success",
        "mean_return",
        "bottom10_conditional_mean_return",
        "task_successes",
        "points",
        "task_success",
        "strict_beats_reference_successes",
        "strict_beats_reference",
        "mean_near_upright_fraction",
        "q_mean_return_delta_vs_reflection_actor",
        "reflection_mean_return_delta_vs_ordinary_actor",
        "q_switch_fraction_vs_reflection_actor",
        "delta_near_reference_successes_vs_baseline",
        "delta_strict_beats_reference_successes_vs_baseline",
        "delta_mean_return_vs_baseline",
        "delta_bottom10_return_vs_baseline",
        "delta_task_successes_vs_baseline",
        "delta_task_rate_vs_baseline",
        "replay_priority_mode",
        "critic_search_actor_weight",
        "self_imitation_weight",
        "pendulum_actor_symmetry_weight",
        "pendulum_critic_symmetry_weight",
        "latest_max_dormant_fraction",
        "latest_min_effective_rank_fraction",
        "replay_hard_fraction",
        "run_dir",
        "evaluation_dir",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metric = row["primary_metrics"]
            config = row["training"]["config"]
            representation = row["training"]["representation_aggregate"]
            replay = row["training"]["replay_snapshot"]
            budget = row["training_budget"]
            writer.writerow(
                {
                    "rank": row["rank"],
                    "condition": row["condition"],
                    "kind": row["kind"],
                    "completed_environment_steps": row[
                        "completed_environment_steps"
                    ],
                    **{
                        name: budget.get(name)
                        for name in fields
                        if name in budget
                    },
                    **{name: metric.get(name) for name in fields if name in metric},
                    "q_mean_return_delta_vs_reflection_actor": row[
                        "q_delta_vs_reflection_actor"
                    ]["mean"],
                    "reflection_mean_return_delta_vs_ordinary_actor": row[
                        "reflection_delta_vs_ordinary_actor"
                    ]["mean"],
                    "q_switch_fraction_vs_reflection_actor": row[
                        "primary_selection"
                    ].get("switch_fraction_vs_reflection_actor"),
                    "delta_near_reference_successes_vs_baseline": row.get(
                        "delta_vs_baseline_primary", {}
                    ).get("near_reference_successes"),
                    "delta_strict_beats_reference_successes_vs_baseline": row.get(
                        "delta_vs_baseline_primary", {}
                    ).get("strict_beats_reference_successes"),
                    "delta_mean_return_vs_baseline": row.get(
                        "delta_vs_baseline_primary", {}
                    ).get("mean_return"),
                    "delta_bottom10_return_vs_baseline": row.get(
                        "delta_vs_baseline_primary", {}
                    ).get("bottom10_conditional_mean_return"),
                    "delta_task_successes_vs_baseline": row.get(
                        "delta_vs_baseline_primary", {}
                    ).get("task_successes"),
                    "delta_task_rate_vs_baseline": row.get(
                        "delta_vs_baseline_primary", {}
                    ).get("task_success"),
                    "replay_priority_mode": config["prioritized_replay"].get(
                        "replay_priority_mode"
                    ),
                    "critic_search_actor_weight": config["q_distillation"].get(
                        "critic_search_actor_weight"
                    ),
                    "self_imitation_weight": config["self_imitation"].get(
                        "self_imitation_weight"
                    ),
                    "pendulum_actor_symmetry_weight": config["symmetry"].get(
                        "pendulum_actor_symmetry_weight"
                    ),
                    "pendulum_critic_symmetry_weight": config["symmetry"].get(
                        "pendulum_critic_symmetry_weight"
                    ),
                    "latest_max_dormant_fraction": representation.get(
                        "latest_max_dormant_fraction"
                    ),
                    "latest_min_effective_rank_fraction": representation.get(
                        "latest_min_effective_rank_fraction"
                    ),
                    "replay_hard_fraction": replay.get(
                        "hard_reset_support_fraction_abs_theta_ge_120"
                    ),
                    "run_dir": row["run_dir"],
                    "evaluation_dir": row["evaluation_dir"],
                }
            )


def write_variant_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "rank",
        "condition",
        "kind",
        "variant",
        "strict_total_steps_for_eligibility",
        "near_reference_successes",
        "near_reference_success",
        "mean_return",
        "bottom10_conditional_mean_return",
        "task_successes",
        "points",
        "task_success",
        "strict_beats_reference_successes",
        "strict_beats_reference",
        "mean_near_upright_fraction",
        "paired_mean_return_delta_vs_ordinary_actor",
        "paired_mean_return_delta_vs_reflection_actor",
        "mean_return_delta_vs_same_baseline_variant",
        "bottom10_delta_vs_same_baseline_variant",
        "task_successes_delta_vs_same_baseline_variant",
        "near_reference_successes_delta_vs_same_baseline_variant",
        "strict_beats_reference_successes_delta_vs_same_baseline_variant",
        "run_dir",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            baseline = row.get("delta_vs_baseline_same_variant", {})
            writer.writerow(
                {
                    "rank": row["rank"],
                    "condition": row["condition"],
                    "kind": row["kind"],
                    "variant": row["variant"],
                    "strict_total_steps_for_eligibility": (
                        row.get("training_budget") or {}
                    ).get("strict_total_steps_for_eligibility"),
                    **{name: metrics.get(name) for name in fields if name in metrics},
                    "paired_mean_return_delta_vs_ordinary_actor": row[
                        "paired_delta_vs_ordinary_actor"
                    ]["mean"],
                    "paired_mean_return_delta_vs_reflection_actor": row[
                        "paired_delta_vs_reflection_actor"
                    ]["mean"],
                    "mean_return_delta_vs_same_baseline_variant": baseline.get(
                        "mean_return"
                    ),
                    "bottom10_delta_vs_same_baseline_variant": baseline.get(
                        "bottom10_conditional_mean_return"
                    ),
                    "task_successes_delta_vs_same_baseline_variant": baseline.get(
                        "task_successes"
                    ),
                    "near_reference_successes_delta_vs_same_baseline_variant": baseline.get(
                        "near_reference_successes"
                    ),
                    "strict_beats_reference_successes_delta_vs_same_baseline_variant": baseline.get(
                        "strict_beats_reference_successes"
                    ),
                    "run_dir": row["run_dir"],
                }
            )


def markdown_report(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "# Pure-RL Seed-0 Off-Grid Screen",
        "",
        (
            f"Each completed strict-<=100k-training-step arm is evaluated on the same {protocol['points']:,} "
            "fixed midpoint-plus-continuous off-grid reset-support states "
            f"(continuous RNG {protocol['rng_seed']}, horizon 200). The validation set "
            "has zero intersections with the 61x41 authority heat-map grid. Candidate "
            "rollouts are completed before the hash-bound off-grid DP/controller reference "
            "cache is read; the authority DP and controller CSVs are never opened here."
        ),
        "",
        (
            "Arms are ranked using the frozen deployment rule: reflection-averaged "
            "actor fallback plus one global 41-action search, accepted only when both "
            "online critics report advantage > 0.005. All actor and critic parameters "
            "come from one checkpoint; there is no router or model mixture."
        ),
        "",
        "Strict budget is learning environment transitions + model-generated replay "
        "transitions + failure-discovery simulator steps. The budget cell is "
        "`learning/model/discovery/strict`; discovery uses the run-complete actual "
        "when present, while eligibility also checks the planned upper bound.",
        "",
        "Promotion is preregistered lexicographically: near-reference successes, task "
        "successes, strict beats-reference successes, bottom-10% conditional mean, "
        "then mean return; the condition label is only a deterministic final tie-break.",
        "",
        "| Rank | Arm | Kind | Budget L/M/D/strict | Near ref | Task | Strict > ref | Mean return | Bottom 10% | Delta near/task/strict vs baseline | Q delta vs reflection | Reflection delta vs actor | Q switch | PER | Q distill | Self imitation | Symmetry A/C | Dormant max | Rank min | Hard replay states |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in report["ranked_arms"]:
        metrics = row["primary_metrics"]
        config = row["training"]["config"]
        representation = row["training"]["representation_aggregate"]
        replay = row["training"]["replay_snapshot"]
        budget = row["training_budget"]
        baseline_delta = row.get("delta_vs_baseline_primary", {})
        lines.append(
            "| {rank} | {condition} | {kind} | {learning:,}/{model:,}/{discovery:,}/{strict:,} | "
            "{near_ref}/{points} ({near_ref_rate:.3%}) | {task}/{points} ({task_rate:.3%}) | "
            "{strict_ref}/{points} ({strict_ref_rate:.3%}) | {mean:.6f} | {tail:.6f} | "
            "{baseline_near}/{baseline_task}/{baseline_strict} | "
            "{q_delta:+.6f} | {reflection_delta:+.6f} | "
            "{switch} | {per} | {q_weight} | {si_weight} | {actor_sym}/{critic_sym} | "
            "{dormant} | {effective_rank} | {hard} |".format(
                rank=row["rank"],
                condition=row["condition"],
                kind=row["kind"],
                learning=budget["learning_environment_steps"],
                model=budget["model_generated_transitions"],
                discovery=budget[
                    "failure_discovery_environment_steps_counted"
                ],
                strict=budget["strict_total_steps_for_eligibility"],
                near_ref=metrics["near_reference_successes"],
                near_ref_rate=metrics["near_reference_success"],
                strict_ref=metrics["strict_beats_reference_successes"],
                strict_ref_rate=metrics["strict_beats_reference"],
                mean=metrics["mean_return"],
                tail=metrics["bottom10_conditional_mean_return"],
                task=metrics["task_successes"],
                points=metrics["points"],
                task_rate=metrics["task_success"],
                baseline_near=format_signed_integer_optional(
                    baseline_delta.get("near_reference_successes")
                ),
                baseline_task=format_signed_integer_optional(
                    baseline_delta.get("task_successes")
                ),
                baseline_strict=format_signed_integer_optional(
                    baseline_delta.get("strict_beats_reference_successes")
                ),
                q_delta=row["q_delta_vs_reflection_actor"]["mean"],
                reflection_delta=row["reflection_delta_vs_ordinary_actor"]["mean"],
                switch=format_optional(
                    row["primary_selection"].get(
                        "switch_fraction_vs_reflection_actor"
                    )
                ),
                per=config["prioritized_replay"].get("replay_priority_mode"),
                q_weight=format_optional(
                    config["q_distillation"].get("critic_search_actor_weight")
                ),
                si_weight=format_optional(
                    config["self_imitation"].get("self_imitation_weight")
                ),
                actor_sym=format_optional(
                    config["symmetry"].get("pendulum_actor_symmetry_weight")
                ),
                critic_sym=format_optional(
                    config["symmetry"].get("pendulum_critic_symmetry_weight")
                ),
                dormant=format_optional(
                    representation.get("latest_max_dormant_fraction")
                ),
                effective_rank=format_optional(
                    representation.get("latest_min_effective_rank_fraction")
                ),
                hard=format_optional(
                    replay.get("hard_reset_support_fraction_abs_theta_ge_120")
                ),
            )
        )
    if not report["ranked_arms"]:
        lines.append("| -- | No evaluated completed arms | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    lines.extend(
        [
            "",
            "## Ordinary, reflection, and Q-search variants",
            "",
            (
                "Every evaluated checkpoint is reported under all three fixed inference "
                "rules. These are independent evaluations of one checkpoint at a time; "
                "no actor, critic, or parameter is copied between conditions. Only the "
                f"`{PRIMARY_VARIANT}` row is used for arm promotion."
            ),
            "",
            "| Overall rank | Condition | Kind | Variant | Near ref | Task | Strict > ref | Mean return | Bottom 10% | Mean delta vs ordinary | Mean delta vs reflection | Near/task/strict delta vs matching baseline |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["ranked_variants"]:
        metrics = row["metrics"]
        baseline_delta = row.get("delta_vs_baseline_same_variant", {})
        lines.append(
            "| {rank} | {condition} | {kind} | {variant} | {near_ref}/{points} "
            "({near_ref_rate:.3%}) | {task}/{points} ({task_rate:.3%}) | "
            "{strict_ref}/{points} ({strict_ref_rate:.3%}) | {mean:.6f} | "
            "{tail:.6f} | {ordinary:+.6f} | {reflection:+.6f} | "
            "{baseline_near}/{baseline_task}/{baseline_strict} |".format(
                rank=row["rank"],
                condition=row["condition"],
                kind=row["kind"],
                variant=row["variant"],
                near_ref=metrics["near_reference_successes"],
                near_ref_rate=metrics["near_reference_success"],
                strict_ref=metrics["strict_beats_reference_successes"],
                strict_ref_rate=metrics["strict_beats_reference"],
                mean=metrics["mean_return"],
                tail=metrics["bottom10_conditional_mean_return"],
                task=metrics["task_successes"],
                points=metrics["points"],
                task_rate=metrics["task_success"],
                ordinary=row["paired_delta_vs_ordinary_actor"]["mean"],
                reflection=row["paired_delta_vs_reflection_actor"]["mean"],
                baseline_near=format_signed_integer_optional(
                    baseline_delta.get("near_reference_successes")
                ),
                baseline_task=format_signed_integer_optional(
                    baseline_delta.get("task_successes")
                ),
                baseline_strict=format_signed_integer_optional(
                    baseline_delta.get("strict_beats_reference_successes")
                ),
            )
        )
    if not report["ranked_variants"]:
        lines.append("| -- | No evaluated variants | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    lines.extend(
        [
            "",
            "## Diagnostics retained",
            "",
            (
                "The JSON report contains complete loss-curve summaries and separate "
                "sections for prioritized replay, Q-search distillation, self-imitation, "
                "symmetry losses, layer dormancy/effective rank, and replay coverage. "
                "Per-arm paired rollout arrays are retained under the evaluation root."
            ),
            "",
            "## Pending and skipped",
            "",
        ]
    )
    pending = report["pending_conditions"] + report["skipped_screen_arms"]
    if pending:
        for row in pending:
            lines.append(
                f"- `{row['condition']}`: {row['reason']} (`{row['run_dir']}`)."
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            protocol["selection_warning"],
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def format_signed_optional(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.6f}"


def format_signed_integer_optional(value: Any) -> str:
    if value is None:
        return "--"
    return f"{int(value):+d}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
