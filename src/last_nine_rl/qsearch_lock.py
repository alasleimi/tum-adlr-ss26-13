from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from last_nine_rl.qsearch_candidate_adapters import (
    validate_inherited_sac_verification,
)

from last_nine_rl.hybrid_qsearch import (
    FixedGlobalCriticQSearchPolicy,
    FixedLocalCriticQSearchPolicy,
    ReflectionAveragedActorPolicy,
)


WORKFLOW_VERSION = "joint-qsearch-lock-v1"
VALIDATION_EVALUATOR_VERSION = "joint-qsearch-offgrid-selector-v2-manifest"
AUTHORITY_EVALUATOR_VERSION = "joint-qsearch-authority-v2-external-config"
LOCK_SCHEMA_VERSION = 1
RAW_ROLLOUT_KEYS = (
    "return",
    "near_upright_fraction",
    "min_step_reward",
    "not_near_upright_streak",
    "stability_success",
    "streak_success",
    "task_success",
)
UNANIMOUS_ACCEPTANCE = "all_online_critics_unanimous_advantage"
GLOBAL_CANDIDATE_SUPPORT = "checkpoint_action_bounds_uniform"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def verify_file_fingerprint(fingerprint: Mapping[str, Any]) -> Path:
    if set(fingerprint) != {"path", "size_bytes", "sha256"}:
        raise ValueError("file fingerprint has missing or unexpected fields")
    path = Path(str(fingerprint["path"])).resolve()
    actual = file_fingerprint(path)
    if actual != dict(fingerprint):
        raise ValueError(f"file fingerprint drift: {path}")
    return path


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], description: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{description} fields differ; missing={missing!r}, extra={extra!r}"
        )


def validate_policy_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, strictly allow-listed inference policy spec."""
    kind = spec.get("kind")
    if kind == "actor":
        _require_exact_keys(spec, {"kind", "reflection"}, "actor policy")
        if not isinstance(spec["reflection"], bool):
            raise ValueError("actor reflection must be boolean")
        canonical = {"kind": "actor", "reflection": bool(spec["reflection"])}
    elif kind == "local":
        _require_exact_keys(
            spec,
            {
                "kind",
                "reflection",
                "num_actions",
                "search_radius",
                "margin",
                "acceptance",
            },
            "local policy",
        )
        if spec["acceptance"] != UNANIMOUS_ACCEPTANCE:
            raise ValueError("local Q-search must use online-critic unanimity")
        if not isinstance(spec["reflection"], bool):
            raise ValueError("local reflection must be boolean")
        num_actions = int(spec["num_actions"])
        radius = float(spec["search_radius"])
        margin = float(spec["margin"])
        if num_actions < 2 or radius <= 0.0 or margin < 0.0:
            raise ValueError("invalid local Q-search count, radius, or margin")
        canonical = {
            "kind": "local",
            "reflection": bool(spec["reflection"]),
            "num_actions": num_actions,
            "search_radius": radius,
            "margin": margin,
            "acceptance": UNANIMOUS_ACCEPTANCE,
        }
    elif kind == "conservative_global":
        _require_exact_keys(
            spec,
            {
                "kind",
                "reflection",
                "num_actions",
                "margin",
                "max_action_delta",
                "acceptance",
                "candidate_support",
            },
            "conservative global policy",
        )
        if spec["acceptance"] != UNANIMOUS_ACCEPTANCE:
            raise ValueError("global Q-search must use online-critic unanimity")
        if spec["candidate_support"] != GLOBAL_CANDIDATE_SUPPORT:
            raise ValueError("global candidates must uniformly span checkpoint bounds")
        if not isinstance(spec["reflection"], bool):
            raise ValueError("global reflection must be boolean")
        num_actions = int(spec["num_actions"])
        margin = float(spec["margin"])
        max_delta = float(spec["max_action_delta"])
        if num_actions < 2 or margin < 0.0 or max_delta <= 0.0:
            raise ValueError("invalid global Q-search count, margin, or trust region")
        canonical = {
            "kind": "conservative_global",
            "reflection": bool(spec["reflection"]),
            "num_actions": num_actions,
            "margin": margin,
            "max_action_delta": max_delta,
            "acceptance": UNANIMOUS_ACCEPTANCE,
            "candidate_support": GLOBAL_CANDIDATE_SUPPORT,
        }
    else:
        raise ValueError(f"unsupported locked policy kind: {kind!r}")
    # Canonicalization must never silently coerce a non-canonical user object.
    if not all(math.isfinite(value) for value in _policy_float_values(canonical)):
        raise ValueError("policy numeric values must be finite")
    return canonical


def _policy_float_values(spec: Mapping[str, Any]) -> list[float]:
    return [
        float(spec[key])
        for key in ("search_radius", "margin", "max_action_delta")
        if key in spec
    ]


def policy_id(spec: Mapping[str, Any]) -> str:
    canonical = validate_policy_spec(spec)
    reflection = "reflection" if canonical["reflection"] else "plain"
    if canonical["kind"] == "actor":
        return f"actor_{reflection}"
    if canonical["kind"] == "local":
        return (
            f"local_{reflection}_n{canonical['num_actions']}_"
            f"r{canonical['search_radius']:g}_m{canonical['margin']:g}"
        )
    return (
        f"global_{reflection}_n{canonical['num_actions']}_"
        f"m{canonical['margin']:g}_d{canonical['max_action_delta']:g}"
    )


def expand_policy_grid(candidate_grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require_exact_keys(
        candidate_grid,
        {"reflection", "include_actor", "local", "conservative_global"},
        "candidate grid",
    )
    reflections = list(candidate_grid["reflection"])
    if not reflections or any(not isinstance(value, bool) for value in reflections):
        raise ValueError("candidate reflections must be a nonempty boolean list")
    if len(set(reflections)) != len(reflections):
        raise ValueError("candidate reflections contain duplicates")
    specs: list[dict[str, Any]] = []
    if bool(candidate_grid["include_actor"]):
        specs.extend({"kind": "actor", "reflection": value} for value in reflections)

    local = candidate_grid["local"]
    _require_exact_keys(
        local, {"num_actions", "search_radii", "margins"}, "local grid"
    )
    for reflection in reflections:
        for count in local["num_actions"]:
            for radius in local["search_radii"]:
                for margin in local["margins"]:
                    specs.append(
                        {
                            "kind": "local",
                            "reflection": reflection,
                            "num_actions": count,
                            "search_radius": radius,
                            "margin": margin,
                            "acceptance": UNANIMOUS_ACCEPTANCE,
                        }
                    )

    global_grid = candidate_grid["conservative_global"]
    _require_exact_keys(
        global_grid,
        {"num_actions", "margins", "max_action_deltas"},
        "conservative global grid",
    )
    for reflection in reflections:
        for count in global_grid["num_actions"]:
            for margin in global_grid["margins"]:
                for max_delta in global_grid["max_action_deltas"]:
                    specs.append(
                        {
                            "kind": "conservative_global",
                            "reflection": reflection,
                            "num_actions": count,
                            "margin": margin,
                            "max_action_delta": max_delta,
                            "acceptance": UNANIMOUS_ACCEPTANCE,
                            "candidate_support": GLOBAL_CANDIDATE_SUPPORT,
                        }
                    )
    canonical = [validate_policy_spec(spec) for spec in specs]
    hashes = [sha256_json(spec) for spec in canonical]
    if len(hashes) != len(set(hashes)):
        raise ValueError("candidate grid expands to duplicate policies")
    return canonical


def make_policy(actor_agent: Any, critic_agent: Any, spec: Mapping[str, Any]) -> Any:
    canonical = validate_policy_spec(spec)
    if canonical["kind"] == "actor":
        return (
            ReflectionAveragedActorPolicy(actor_agent)
            if canonical["reflection"]
            else actor_agent
        )
    if canonical["kind"] == "local":
        return FixedLocalCriticQSearchPolicy(
            actor_agent=actor_agent,
            critic_agent=critic_agent,
            num_actions=canonical["num_actions"],
            margin=canonical["margin"],
            search_radius=canonical["search_radius"],
            symmetric_actor_fallback=canonical["reflection"],
        )
    return FixedGlobalCriticQSearchPolicy(
        actor_agent=actor_agent,
        critic_agent=critic_agent,
        num_actions=canonical["num_actions"],
        margin=canonical["margin"],
        max_action_delta=canonical["max_action_delta"],
        symmetric_actor_fallback=canonical["reflection"],
    )


def validate_dataset_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        spec,
        {
            "midpoint_theta_bins",
            "midpoint_velocity_bins",
            "continuous_points",
            "continuous_rng_seed",
            "velocity_limit",
        },
        "validation dataset",
    )
    canonical = {
        "midpoint_theta_bins": int(spec["midpoint_theta_bins"]),
        "midpoint_velocity_bins": int(spec["midpoint_velocity_bins"]),
        "continuous_points": int(spec["continuous_points"]),
        "continuous_rng_seed": int(spec["continuous_rng_seed"]),
        "velocity_limit": float(spec["velocity_limit"]),
    }
    if (
        canonical["midpoint_theta_bins"] < 2
        or canonical["midpoint_velocity_bins"] < 2
        or canonical["continuous_points"] < 1
        or canonical["velocity_limit"] <= 0.0
        or not math.isfinite(canonical["velocity_limit"])
    ):
        raise ValueError("invalid off-grid validation dataset specification")
    if (
        canonical["midpoint_theta_bins"] == 61
        and canonical["midpoint_velocity_bins"] == 41
    ):
        raise ValueError("the authoritative 61x41 lattice is forbidden for selection")
    return canonical


def _midpoint_grid(theta_bins: int, velocity_bins: int, velocity_limit: float) -> tuple[np.ndarray, np.ndarray]:
    theta = -math.pi + (np.arange(theta_bins, dtype=np.float64) + 0.5) * (
        2.0 * math.pi / theta_bins
    )
    velocity = -velocity_limit + (
        np.arange(velocity_bins, dtype=np.float64) + 0.5
    ) * (2.0 * velocity_limit / velocity_bins)
    return np.tile(theta, velocity_bins), np.repeat(velocity, theta_bins)


def authoritative_grid_mask(theta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    authority_theta = np.linspace(-math.pi, math.pi, 61, endpoint=False, dtype=np.float64)
    authority_velocity = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    theta_match = np.isclose(theta[:, None], authority_theta[None, :], rtol=0.0, atol=1e-13).any(axis=1)
    velocity_match = np.isclose(
        velocity[:, None], authority_velocity[None, :], rtol=0.0, atol=1e-13
    ).any(axis=1)
    return theta_match & velocity_match


def build_validation_dataset(spec: Mapping[str, Any]) -> dict[str, Any]:
    canonical = validate_dataset_spec(spec)
    midpoint_theta, midpoint_velocity = _midpoint_grid(
        canonical["midpoint_theta_bins"],
        canonical["midpoint_velocity_bins"],
        canonical["velocity_limit"],
    )
    rng = np.random.default_rng(canonical["continuous_rng_seed"])
    continuous_theta: list[float] = []
    continuous_velocity: list[float] = []
    remaining = canonical["continuous_points"]
    while remaining:
        batch = max(remaining, 64)
        candidate_theta = rng.uniform(-math.pi, math.pi, batch)
        candidate_velocity = rng.uniform(
            -canonical["velocity_limit"], canonical["velocity_limit"], batch
        )
        keep = ~authoritative_grid_mask(candidate_theta, candidate_velocity)
        take = min(remaining, int(keep.sum()))
        continuous_theta.extend(candidate_theta[keep][:take].tolist())
        continuous_velocity.extend(candidate_velocity[keep][:take].tolist())
        remaining -= take
    theta = np.concatenate(
        [midpoint_theta, np.asarray(continuous_theta, dtype=np.float64)]
    )
    velocity = np.concatenate(
        [midpoint_velocity, np.asarray(continuous_velocity, dtype=np.float64)]
    )
    if authoritative_grid_mask(theta, velocity).any():
        raise ValueError("selection dataset intersects the authoritative grid")
    split = int(len(midpoint_theta))
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(canonical))
    digest.update(np.asarray(theta, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(velocity, dtype="<f8").tobytes(order="C"))
    digest.update(int(split).to_bytes(8, "little", signed=False))
    return {
        "spec": canonical,
        "theta": theta,
        "velocity": velocity,
        "midpoint_count": split,
        "continuous_count": int(len(theta) - split),
        "points": int(len(theta)),
        "sha256": digest.hexdigest(),
        "authoritative_grid_intersection_count": 0,
    }


def rows_to_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in RAW_ROLLOUT_KEYS:
        arrays[key] = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return arrays


def conditional_mean(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    return float(np.sort(values)[:count].mean())


def _rate(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=bool)
    return {
        "successes": int(values.sum()),
        "trials": int(len(values)),
        "rate": float(values.mean()),
    }


def rollout_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    reference_returns: np.ndarray | None = None,
    fallback_returns: np.ndarray | None = None,
    epsilon_return: float = 5.0,
) -> dict[str, Any]:
    returns = np.asarray(arrays["return"], dtype=np.float64)
    task = np.asarray(arrays["task_success"], dtype=np.float64)
    near = np.asarray(arrays["near_upright_fraction"], dtype=np.float64)
    result: dict[str, Any] = {
        "points": int(len(returns)),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "worst_return": float(returns.min()),
        "return_p01": float(np.quantile(returns, 0.01)),
        "return_p05": float(np.quantile(returns, 0.05)),
        "return_p10": float(np.quantile(returns, 0.10)),
        "bottom10_conditional_mean_return": conditional_mean(returns, 0.10),
        "task_success": _rate(task >= 0.5),
        "mean_near_upright_fraction": float(near.mean()),
    }
    if reference_returns is not None:
        reference = np.asarray(reference_returns, dtype=np.float64)
        if reference.shape != returns.shape:
            raise ValueError("reference and candidate return shapes differ")
        result.update(
            {
                "near_reference_eps": _rate(returns >= reference - epsilon_return),
                "strict_beats_reference": _rate(returns > reference),
                "reference_regret_mean": float((reference - returns).mean()),
                "reference_regret_p95": float(np.quantile(reference - returns, 0.95)),
                "reference_regret_p99": float(np.quantile(reference - returns, 0.99)),
            }
        )
    if fallback_returns is not None:
        fallback = np.asarray(fallback_returns, dtype=np.float64)
        if fallback.shape != returns.shape:
            raise ValueError("fallback and candidate return shapes differ")
        delta = returns - fallback
        fallback_tail_count = max(1, int(math.ceil(len(fallback) * 0.10)))
        fallback_tail = np.argsort(fallback)[:fallback_tail_count]
        result["harm"] = {
            "mean_return_delta": float(delta.mean()),
            "median_return_delta": float(np.median(delta)),
            "delta_p01": float(np.quantile(delta, 0.01)),
            "delta_p05": float(np.quantile(delta, 0.05)),
            "worst_delta": float(delta.min()),
            "improved_fraction": float((delta > 0.0).mean()),
            "degraded_fraction": float((delta < 0.0).mean()),
            "degraded_by_1_fraction": float((delta < -1.0).mean()),
            "degraded_by_5_fraction": float((delta < -5.0).mean()),
            "bottom10_conditional_mean_return_delta": float(
                conditional_mean(returns, 0.10) - conditional_mean(fallback, 0.10)
            ),
            "mean_delta_on_fallback_bottom10": float(delta[fallback_tail].mean()),
        }
    return result


def selection_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    """Pre-registered lexicographic rule; larger tuples are better."""
    metrics = result["metrics"]["all"]
    harm = metrics["harm"]
    selection = result.get("selection", {})
    return (
        int(metrics["near_reference_eps"]["successes"]),
        int(metrics["task_success"]["successes"]),
        int(metrics["strict_beats_reference"]["successes"]),
        -float(harm["degraded_by_5_fraction"]),
        float(metrics["bottom10_conditional_mean_return"]),
        float(metrics["mean_return"]),
        -float(selection.get("switch_fraction", 0.0)),
        str(result["checkpoint"]["sha256"]),
        str(result["policy_sha256"]),
    )


def fixed_authority_spec(epsilon_return: float = 5.0) -> dict[str, Any]:
    return {
        "theta_bins": 61,
        "theta_endpoint": False,
        "theta_min": -math.pi,
        "theta_max": math.pi,
        "velocity_bins": 41,
        "velocity_min": -1.0,
        "velocity_max": 1.0,
        "horizon": 200,
        "initial_condition_cells_per_checkpoint": 2501,
        "epsilon_return": float(epsilon_return),
        "selection_allowed": False,
    }


def attach_lock_integrity(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "lock_sha256" in payload:
        raise ValueError("payload already has lock_sha256")
    result = dict(payload)
    result["lock_sha256"] = sha256_json(payload)
    return result


def _verify_completion_provenance(completion: Mapping[str, Any]) -> None:
    kind = completion.get("kind")
    if kind == "standard_events":
        verify_file_fingerprint(completion["events"])
        return
    if kind == "specialized_run_manifest":
        verify_file_fingerprint(completion["run_manifest"])
        verify_file_fingerprint(completion["events"])
        artifacts = completion.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ValueError("specialized completion has no pinned artifacts")
        for fingerprint in artifacts.values():
            verify_file_fingerprint(fingerprint)
        source = completion.get("source")
        if isinstance(source, Mapping):
            for field in ("config", "checkpoint", "events"):
                verify_file_fingerprint(source[field])
        return
    if kind == "distillation_summary":
        verify_file_fingerprint(completion["summary"])
        verify_file_fingerprint(completion["events"])
        if completion.get("critics_eligible_for_qsearch") is not False:
            raise ValueError("distillation completion incorrectly enables critic Q-search")
        return
    raise ValueError(f"unsupported selected completion provenance: {kind!r}")


def _verify_budget_evidence(evidence: Mapping[str, Any], strict_budget: int) -> None:
    if int(evidence.get("strict_budget_steps", -1)) != strict_budget:
        raise ValueError("selected budget evidence differs from the selected strict budget")
    kind = evidence.get("kind")
    if kind == "standard_lineage_v1":
        initializer = evidence.get("initializer")
        upstream_summary = evidence.get("upstream_summary")
        if (initializer is None) != (upstream_summary is None):
            raise ValueError("standard budget evidence has partial upstream provenance")
        if initializer is not None:
            verify_file_fingerprint(initializer)
            verify_file_fingerprint(upstream_summary)
        expected = (
            int(evidence["upstream_environment_steps"])
            + int(evidence["reference_prior_environment_steps"])
            + int(evidence["local_environment_steps"])
        )
        if expected != strict_budget:
            raise ValueError("standard budget evidence arithmetic mismatch")
        return
    if kind == "standard_recursive_lineage_v1":
        verify_file_fingerprint(evidence["initializer"])
        expected = (
            int(evidence["upstream_environment_steps"])
            + int(evidence["reference_prior_environment_steps"])
            + int(evidence["local_environment_steps"])
        )
        if expected != strict_budget:
            raise ValueError("recursive standard budget evidence arithmetic mismatch")
        upstream = evidence.get("upstream_provenance")
        if not isinstance(upstream, Mapping):
            raise ValueError("recursive standard budget evidence has no upstream provenance")
        upstream_kind = upstream.get("kind")
        if upstream_kind == "specialized_run":
            verify_file_fingerprint(upstream["run_manifest"])
        elif upstream_kind == "standard_run":
            verify_file_fingerprint(upstream["config"])
            verify_file_fingerprint(upstream["events"])
            nested = upstream.get("budget_evidence")
            if not isinstance(nested, Mapping):
                raise ValueError("recursive standard source has no nested budget evidence")
            _verify_budget_evidence(
                nested, int(evidence["upstream_environment_steps"])
            )
        else:
            raise ValueError(
                f"unsupported recursive standard upstream provenance: {upstream_kind!r}"
            )
        return
    if kind == "specialized_native_budget_v1":
        verify_file_fingerprint(evidence["run_manifest"])
        return
    if kind == "distillation_lineage_v1":
        verify_file_fingerprint(evidence["summary"])
        if int(evidence.get("training_environment_steps", -1)) != strict_budget:
            raise ValueError("distillation budget arithmetic mismatch")
        return
    raise ValueError(f"unsupported selected budget evidence: {kind!r}")


def verify_lock(lock: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    _require_exact_keys(
        lock,
        {
            "schema_version",
            "workflow_version",
            "selection",
            "selected",
            "authority",
            "lock_sha256",
        },
        "protocol lock",
    )
    body = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock["lock_sha256"] != sha256_json(body):
        raise ValueError("protocol lock integrity hash mismatch")
    if int(lock["schema_version"]) != LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported protocol lock schema")
    if lock["workflow_version"] != WORKFLOW_VERSION:
        raise ValueError("unsupported Q-search lock workflow")
    selection = lock["selection"]
    evaluator_version = selection.get("evaluator_version")
    if evaluator_version not in {None, VALIDATION_EVALUATOR_VERSION}:
        raise ValueError("selection evaluator version drift")
    if selection.get("authoritative_grid_queried") is not False:
        raise ValueError("selection lock does not certify an off-grid-only screen")
    if selection.get("reference_used_only_after_all_candidate_rollouts") is not True:
        raise ValueError("selection lock does not certify post-rollout reference use")
    selected = lock["selected"]
    policy = validate_policy_spec(selected["policy"])
    if selected["policy_sha256"] != sha256_json(policy):
        raise ValueError("selected policy hash mismatch")
    model_kind = str(selected.get("model_kind", "standard_sac_agent"))
    if model_kind not in {
        "standard_sac_agent",
        "actor_only_sac_agent",
        "direct_actor_sac_agent",
        "time_conditioned_actor",
    }:
        raise ValueError(f"unsupported selected model kind: {model_kind!r}")
    if model_kind == "time_conditioned_actor" and policy["kind"] != "actor":
        raise ValueError("time-conditioned checkpoints cannot lock critic Q-search policies")
    if model_kind == "direct_actor_sac_agent" and policy["kind"] != "actor":
        raise ValueError("direct-actor SAC checkpoints cannot lock critic Q-search policies")
    if model_kind == "time_conditioned_actor" and selected.get("completion", {}).get(
        "kind"
    ) != "specialized_run_manifest":
        raise ValueError("time-conditioned selection lacks specialized completion provenance")
    if model_kind == "actor_only_sac_agent":
        completion = selected.get("completion")
        if not isinstance(completion, Mapping) or completion.get("kind") != "specialized_run_manifest":
            raise ValueError("inherited-critic SAC selection lacks specialized provenance")
        source = completion.get("source")
        if not isinstance(source, Mapping) or not isinstance(source.get("checkpoint"), Mapping):
            raise ValueError("inherited-critic SAC selection lacks its pinned H0 checkpoint")
        evidence = selected.get("inherited_critic_verification")
        if not isinstance(evidence, Mapping):
            raise ValueError("inherited-critic SAC selection lacks runtime state verification")
        validate_inherited_sac_verification(evidence)
        if evidence.get("derived_checkpoint") != selected.get("checkpoint"):
            raise ValueError("inherited-critic verification is bound to a different checkpoint")
        if evidence.get("source_checkpoint") != source.get("checkpoint"):
            raise ValueError("inherited-critic verification is bound to a different H0 source")
    if model_kind == "direct_actor_sac_agent":
        completion = selected.get("completion")
        if not isinstance(completion, Mapping) or completion.get("kind") != "distillation_summary":
            raise ValueError("direct-actor SAC selection lacks distillation provenance")
        if completion.get("critics_eligible_for_qsearch") is not False:
            raise ValueError("direct-actor SAC selection incorrectly enables critic Q-search")
    if evaluator_version == VALIDATION_EVALUATOR_VERSION:
        for field in (
            "candidate_source",
            "evaluation_contract",
        ):
            if field not in selection:
                raise ValueError(f"manifest selector lock is missing selection.{field}")
        for field in (
            "candidate_id",
            "source_family",
            "checkpoint_training_step",
            "checkpoint_step_unit",
            "strict_budget_steps",
            "cumulative_environment_steps",
            "budget_evidence",
            "completion",
        ):
            if field not in selected:
                raise ValueError(f"manifest selector lock is missing selected.{field}")
        if selection.get("authority_grid_values_read_for_selection") is not False:
            raise ValueError(
                "selection lock does not certify that authority-grid values were unread"
            )
    strict_budget: int | None = None
    if "strict_budget_steps" in selected:
        strict_budget = int(selected["strict_budget_steps"])
        if not 1 <= strict_budget <= 100_000:
            raise ValueError("selected checkpoint violates the strict <=100k budget")
        if int(selected.get("cumulative_environment_steps", strict_budget)) != strict_budget:
            raise ValueError("selected cumulative environment budget is internally inconsistent")
    authority = lock["authority"]
    expected_spec = fixed_authority_spec(authority["spec"]["epsilon_return"])
    if authority["spec"] != expected_spec:
        raise ValueError("authoritative grid/protocol drift")
    if authority["evaluator_version"] != AUTHORITY_EVALUATOR_VERSION:
        raise ValueError("authoritative evaluator version drift")
    if verify_files:
        verify_file_fingerprint(selected["checkpoint"])
        verify_file_fingerprint(selected["config"])
        if "completion" in selected:
            _verify_completion_provenance(selected["completion"])
        if "budget_evidence" in selected:
            if strict_budget is None:
                raise ValueError("selected budget evidence has no strict budget")
            _verify_budget_evidence(selected["budget_evidence"], strict_budget)
        candidate_source = selection.get("candidate_source")
        if isinstance(candidate_source, Mapping):
            if candidate_source.get("models_copied") is not False:
                raise ValueError("candidate source does not certify zero model copies")
            if candidate_source.get("kind") == "exact_candidate_manifest":
                verify_file_fingerprint(candidate_source["candidate_manifest"])
        verify_file_fingerprint(selection["protocol_config"])
        verify_file_fingerprint(selection["dp_solution"])
        verify_file_fingerprint(selection["selection_report"])
        verify_file_fingerprint(authority["dp_grid"])
        verify_file_fingerprint(authority["controller_grid"])
    return {
        **dict(lock),
        "selected": {**dict(selected), "model_kind": model_kind, "policy": policy},
    }


def validate_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_guard: Mapping[str, Any],
    root: Path,
) -> None:
    _require_exact_keys(manifest, {"guard", "artifacts"}, "artifact manifest")
    if manifest["guard"] != dict(expected_guard):
        raise ValueError("cache guard mismatch; refusing stale or drifted cache")
    artifacts = manifest["artifacts"]
    if not artifacts:
        raise ValueError("cache manifest contains no artifacts")
    for relative, expected in artifacts.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("cache artifact path escapes cache root")
        actual_path = (root / relative).resolve()
        actual = {
            "size_bytes": int(actual_path.stat().st_size),
            "sha256": sha256_file(actual_path),
        }
        if actual != expected:
            raise ValueError(f"cache artifact drift: {actual_path}")
