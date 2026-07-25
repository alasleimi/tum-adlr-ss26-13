from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.envs import UprightDetector
from last_nine_rl.hybrid_qsearch import (
    FixedLocalCriticQSearchPolicy,
    ReflectionAveragedActorPolicy,
)
from last_nine_rl.pendulum_grid import rollout_pendulum_grid_vectorized
from last_nine_rl.qsearch_lock import (
    authoritative_grid_mask,
    build_validation_dataset,
)

try:
    from scripts.train_pendulum_qregularized_dagger import (
        validation_reference_returns,
    )
except ModuleNotFoundError:
    from train_pendulum_qregularized_dagger import (  # type: ignore[no-redef]
        validation_reference_returns,
    )


class TrackedGlobalQSearchPolicy:
    def __init__(
        self,
        agent: Any,
        num_actions: int,
        margin: float,
        filter_mode: str,
        blend_fraction: float = 1.0,
        critic_search_batch_size: int = 512,
        max_action_delta: float | None = None,
    ) -> None:
        self.agent = agent
        self.num_actions = int(num_actions)
        self.margin = float(margin)
        self.filter_mode = str(filter_mode)
        self.blend_fraction = float(blend_fraction)
        self.critic_search_batch_size = int(critic_search_batch_size)
        self.max_action_delta = (
            None if max_action_delta is None else float(max_action_delta)
        )
        self.selected_count = 0
        self.total_count = 0
        self.trust_region_rejected_count = 0
        self.selected_abs_delta_sum = 0.0
        self.selected_abs_delta_max = 0.0

    def act_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        actor = np.asarray(
            self.agent.act_batch(observations, deterministic=True), dtype=np.float32
        ).reshape(-1, 1)
        proposed = np.asarray(
            self.agent.act_batch_critic_search(
                observations,
                num_actions=self.num_actions,
                margin=self.margin,
                batch_size=self.critic_search_batch_size,
                filter_mode=self.filter_mode,
                blend_fraction=self.blend_fraction,
            ),
            dtype=np.float32,
        ).reshape(-1, 1)
        proposal_delta = np.abs(proposed - actor).reshape(-1)
        if self.max_action_delta is None:
            selected = proposed
        else:
            rejected = proposal_delta > self.max_action_delta
            self.trust_region_rejected_count += int(rejected.sum())
            selected = np.where(rejected.reshape(-1, 1), actor, proposed)
        delta = np.abs(selected - actor).reshape(-1)
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
            "switch_fraction": float(self.selected_count / max(self.total_count, 1)),
            "trust_region_rejected_count": float(self.trust_region_rejected_count),
            "trust_region_rejected_fraction": float(
                self.trust_region_rejected_count / max(self.total_count, 1)
            ),
            "selected_abs_action_delta_mean": float(
                self.selected_abs_delta_sum / max(self.selected_count, 1)
            ),
            "selected_abs_action_delta_max": float(self.selected_abs_delta_max),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-free continuous-state screen of fixed pure-RL critic "
            "Q-search rules."
        )
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points", type=int)
    parser.add_argument("--rng-seed", type=int)
    parser.add_argument(
        "--frozen-protocol-config",
        help=(
            "Optional locked off-grid protocol. When supplied, candidate policies "
            "are scored on its midpoint-plus-continuous states against its pinned "
            "finite-horizon reference. The authority grid is never read."
        ),
    )
    parser.add_argument("--global-num-actions", type=int, nargs="*", default=[])
    parser.add_argument("--global-margins", type=float, nargs="*", default=[])
    parser.add_argument("--global-blend-fractions", type=float, nargs="*", default=[1.0])
    parser.add_argument("--global-max-action-deltas", type=float, nargs="*", default=[])
    parser.add_argument("--global-include-uncapped", action="store_true")
    parser.add_argument("--critic-search-batch-size", type=int, default=512)
    parser.add_argument(
        "--global-filter-modes",
        nargs="*",
        choices=(
            "always",
            "clipped_value",
            "unanimous_advantage",
            "online_target_unanimous_advantage",
            "online_target_joint_unanimous_advantage",
            "mean_proposal_unanimous_advantage",
            "mid0125_proposal_unanimous_advantage",
            "mid025_proposal_unanimous_advantage",
            "mid0375_proposal_unanimous_advantage",
            "unc025_increase_unanimous_advantage",
            "unc05_increase_unanimous_advantage",
            "unc1_increase_unanimous_advantage",
            "unc2_increase_unanimous_advantage",
            "target_unanimous_advantage",
            "target_proposal_online_unanimous_advantage",
            "target_proposal_online_target_unanimous_advantage",
            "lcb025_proposal_unanimous_advantage",
            "lcb05_proposal_unanimous_advantage",
            "lcb1_proposal_unanimous_advantage",
            "symmetric_actor_unanimous_advantage",
            "symmetric_critic_unanimous_advantage",
            "symmetric_actor_critic_unanimous_advantage",
        ),
        default=[],
    )
    parser.add_argument("--local-radii", type=float, nargs="*", default=[])
    parser.add_argument("--local-num-actions", type=int, nargs="*", default=[])
    parser.add_argument("--local-margins", type=float, nargs="*", default=[])
    parser.add_argument(
        "--include-reflection",
        action="store_true",
        help="Also evaluate the globally applied reflection-projected actor.",
    )
    return parser.parse_args()


def conditional_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    return float(np.sort(values)[:count].mean())


def metrics(
    rows: list[dict[str, Any]],
    baseline_returns: np.ndarray | None = None,
    reference_returns: np.ndarray | None = None,
    epsilon_return: float = 5.0,
) -> dict[str, Any]:
    returns = np.asarray([float(row["return"]) for row in rows], dtype=np.float64)
    task = np.asarray([float(row["task_success"]) for row in rows], dtype=np.float64)
    near_fraction = np.asarray(
        [float(row["near_upright_fraction"]) for row in rows], dtype=np.float64
    )
    result = {
        "points": float(len(rows)),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "return_p01": float(np.quantile(returns, 0.01)),
        "return_p05": float(np.quantile(returns, 0.05)),
        "return_p10": float(np.quantile(returns, 0.10)),
        "bottom10_conditional_mean_return": conditional_mean(returns, 0.10),
        "task_success": float(task.mean()),
        "mean_near_upright_fraction": float(near_fraction.mean()),
    }
    if reference_returns is not None:
        reference = np.asarray(reference_returns, dtype=np.float64)
        if reference.shape != returns.shape:
            raise ValueError("reference and candidate return shapes differ")
        near_reference = returns >= reference - float(epsilon_return)
        strict = returns > reference
        task_bool = task >= 0.5
        result.update(
            {
                "near_reference_eps": rate(near_reference),
                "task_success_rate": rate(task_bool),
                "strict_beats_reference": rate(strict),
                "reference_regret_mean": float((reference - returns).mean()),
                "reference_regret_p95": float(
                    np.quantile(reference - returns, 0.95)
                ),
                "reference_regret_p99": float(
                    np.quantile(reference - returns, 0.99)
                ),
            }
        )
    if baseline_returns is not None:
        delta = returns - baseline_returns
        baseline_bottom = conditional_mean(baseline_returns, 0.10)
        result.update(
            {
                "mean_return_delta": float(delta.mean()),
                "median_return_delta": float(np.median(delta)),
                "return_improved_fraction": float((delta > 0.0).mean()),
                "return_degraded_fraction": float((delta < 0.0).mean()),
                "return_degraded_by_1_fraction": float((delta < -1.0).mean()),
                "return_degraded_by_5_fraction": float((delta < -5.0).mean()),
                "bottom10_conditional_mean_return_delta": float(
                    result["bottom10_conditional_mean_return"] - baseline_bottom
                ),
            }
        )
    return result


def rate(values: np.ndarray) -> dict[str, Any]:
    success = np.asarray(values, dtype=bool)
    return {
        "successes": int(success.sum()),
        "trials": int(len(success)),
        "rate": float(success.mean()),
    }


def variant_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"kind": "actor", "name": "actor"}]
    if args.include_reflection:
        specs.append({"kind": "reflection", "name": "reflection_actor"})
    max_action_deltas: list[float | None]
    if args.global_max_action_deltas:
        max_action_deltas = [float(value) for value in args.global_max_action_deltas]
        if args.global_include_uncapped:
            max_action_deltas.insert(0, None)
    else:
        max_action_deltas = [None]
    for num_actions in args.global_num_actions:
        for margin in args.global_margins:
            for filter_mode in args.global_filter_modes:
                for blend_fraction in args.global_blend_fractions:
                    for max_action_delta in max_action_deltas:
                        delta_suffix = (
                            "" if max_action_delta is None else f"_d{max_action_delta:g}"
                        )
                        specs.append(
                            {
                                "kind": "global",
                                "name": (
                                    f"global_n{num_actions}_m{margin:g}_b{blend_fraction:g}_"
                                    f"{filter_mode}{delta_suffix}"
                                ),
                                "num_actions": int(num_actions),
                                "margin": float(margin),
                                "filter_mode": str(filter_mode),
                                "blend_fraction": float(blend_fraction),
                                "max_action_delta": max_action_delta,
                            }
                        )
    for radius in args.local_radii:
        for num_actions in args.local_num_actions:
            for margin in args.local_margins:
                specs.append(
                    {
                        "kind": "local",
                        "name": f"local_r{radius:g}_n{num_actions}_m{margin:g}_unanimous",
                        "radius": float(radius),
                        "num_actions": int(num_actions),
                        "margin": float(margin),
                        "filter_mode": "unanimous_advantage",
                    }
                )
    return specs


def make_policy(
    agent: Any, spec: dict[str, Any], critic_search_batch_size: int = 512
) -> Any:
    if spec["kind"] == "actor":
        return agent
    if spec["kind"] == "reflection":
        return ReflectionAveragedActorPolicy(agent)
    if spec["kind"] == "global":
        return TrackedGlobalQSearchPolicy(
            agent,
            num_actions=int(spec["num_actions"]),
            margin=float(spec["margin"]),
            filter_mode=str(spec["filter_mode"]),
            blend_fraction=float(spec.get("blend_fraction", 1.0)),
            critic_search_batch_size=int(critic_search_batch_size),
            max_action_delta=spec.get("max_action_delta"),
        )
    return FixedLocalCriticQSearchPolicy(
        actor_agent=agent,
        critic_agent=agent,
        num_actions=int(spec["num_actions"]),
        margin=float(spec["margin"]),
        search_radius=float(spec["radius"]),
    )


def main() -> None:
    args = parse_args()
    if int(args.critic_search_batch_size) <= 0:
        raise SystemExit("--critic-search-batch-size must be positive")
    if any(float(value) <= 0.0 for value in args.global_max_action_deltas):
        raise SystemExit("--global-max-action-deltas values must be positive")

    project_root = Path(__file__).resolve().parents[1]
    reference_returns: np.ndarray | None = None
    frozen_protocol: dict[str, Any] | None = None
    frozen_dataset_sha256: str | None = None
    if args.frozen_protocol_config:
        protocol_path = Path(args.frozen_protocol_config)
        if not protocol_path.is_absolute():
            protocol_path = project_root / protocol_path
        frozen_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        dataset = build_validation_dataset(frozen_protocol["validation_dataset"])
        frozen_dataset_sha256 = str(dataset["sha256"])
        theta = np.asarray(dataset["theta"], dtype=np.float64)
        velocity = np.asarray(dataset["velocity"], dtype=np.float64)
        if authoritative_grid_mask(theta, velocity).any():
            raise ValueError("frozen off-grid dataset intersects the authority grid")
        if args.points is not None and int(args.points) != int(
            dataset["continuous_count"]
        ):
            raise ValueError("--points differs from the locked continuous count")
        if args.rng_seed is not None and int(args.rng_seed) != int(
            dataset["spec"]["continuous_rng_seed"]
        ):
            raise ValueError("--rng-seed differs from the locked protocol")
        reference_spec = frozen_protocol["reference_protocol"]
        dp_spec = reference_spec["dp_solution"]
        dp_solution = Path(str(dp_spec["path"]))
        if not dp_solution.is_absolute():
            dp_solution = project_root / dp_solution
        if (
            dp_solution.stat().st_size != int(dp_spec["size_bytes"])
            or sha256_file(dp_solution) != str(dp_spec["sha256"])
        ):
            raise ValueError("pinned off-grid DP solution fingerprint drift")
        reliability = ReliabilityConfig(**frozen_protocol["evaluation_reliability"])
        reference_detector = UprightDetector(
            "Pendulum-v1",
            cos_threshold=reliability.near_upright_cos_threshold,
            abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
        )
        reference_returns = validation_reference_returns(
            theta,
            velocity,
            reference_detector,
            reliability,
            dp_solution,
        )
        epsilon_return = float(reference_spec["epsilon_return"])
    else:
        if args.points is None or int(args.points) <= 0:
            raise SystemExit("--points must be positive without a frozen protocol")
        if args.rng_seed is None:
            raise SystemExit("--rng-seed is required without a frozen protocol")
        rng = np.random.default_rng(int(args.rng_seed))
        theta = rng.uniform(-math.pi, math.pi, int(args.points))
        velocity = rng.uniform(-1.0, 1.0, int(args.points))
        reliability = None
        epsilon_return = 5.0

    specs = variant_specs(args)
    results: list[dict[str, Any]] = []

    for run_arg in args.runs:
        run = Path(run_arg)
        agent, config, _payload = load_agent_from_run(run, device=args.device)
        evaluation_reliability = reliability or config.reliability
        detector = UprightDetector(
            "Pendulum-v1",
            cos_threshold=evaluation_reliability.near_upright_cos_threshold,
            abs_velocity_threshold=evaluation_reliability.near_upright_abs_velocity_threshold,
        )
        baseline_rows = rollout_pendulum_grid_vectorized(
            agent, theta, velocity, detector, evaluation_reliability, horizon=200
        )
        baseline_returns = np.asarray(
            [float(row["return"]) for row in baseline_rows], dtype=np.float64
        )
        baseline_metrics = metrics(
            baseline_rows,
            reference_returns=reference_returns,
            epsilon_return=epsilon_return,
        )
        results.append(
            {
                "run": str(run),
                "seed": int(config.seed),
                "variant": "actor",
                "spec": specs[0],
                "metrics": baseline_metrics,
                "selection": {},
            }
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

        for spec in specs[1:]:
            policy = make_policy(
                agent, spec, critic_search_batch_size=int(args.critic_search_batch_size)
            )
            rows = rollout_pendulum_grid_vectorized(
                policy, theta, velocity, detector, evaluation_reliability, horizon=200
            )
            selection = (
                policy.selection_metrics()
                if hasattr(policy, "selection_metrics")
                else {}
            )
            results.append(
                {
                    "run": str(run),
                    "seed": int(config.seed),
                    "variant": str(spec["name"]),
                    "spec": spec,
                    "metrics": metrics(
                        rows,
                        baseline_returns=baseline_returns,
                        reference_returns=reference_returns,
                        epsilon_return=epsilon_return,
                    ),
                    "selection": selection,
                }
            )
            print(json.dumps(results[-1], sort_keys=True), flush=True)

    payload = {
        "protocol": {
            "reference_free": reference_returns is None,
            "reference_used_during_candidate_inference": False,
            "reference_used_only_for_post_rollout_scoring": (
                reference_returns is not None
            ),
            "authoritative_grid_queried": False,
            "state_distribution": (
                "fixed_offgrid_midpoint_plus_continuous_reset_support"
                if frozen_protocol is not None
                else "continuous_uniform_reset_support"
            ),
            "points": int(len(theta)),
            "continuous_points": (
                int(frozen_protocol["validation_dataset"]["continuous_points"])
                if frozen_protocol is not None
                else int(args.points)
            ),
            "rng_seed": (
                int(frozen_protocol["validation_dataset"]["continuous_rng_seed"])
                if frozen_protocol is not None
                else int(args.rng_seed)
            ),
            "state_sha256": (
                frozen_dataset_sha256
                if frozen_dataset_sha256 is not None
                else state_sha256(theta, velocity)
            ),
            "raw_state_array_sha256": state_sha256(theta, velocity),
            "epsilon_return": float(epsilon_return),
            "runs": [str(run) for run in args.runs],
            "selection_metrics": (
                [
                    "near_reference_eps_successes",
                    "task_successes",
                    "strict_beats_reference_successes",
                    "bottom10_conditional_mean_return",
                    "mean_return",
                ]
                if reference_returns is not None
                else [
                    "mean_return",
                    "bottom10_conditional_mean_return",
                    "task_success",
                ]
            ),
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(theta: np.ndarray, velocity: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(theta, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(velocity, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
