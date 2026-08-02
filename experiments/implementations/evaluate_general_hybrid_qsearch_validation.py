from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.envs import UprightDetector
from last_nine_rl.hybrid_qsearch import (
    FixedGlobalCriticQSearchPolicy,
    FixedLocalCriticQSearchPolicy,
)
from last_nine_rl.pendulum_grid import rollout_pendulum_grid_vectorized

from train_pendulum_qregularized_dagger import (
    midpoint_grid,
    validation_reference_returns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Off-grid validation for one fixed supervised + pure-RL critic Q-search rule."
    )
    parser.add_argument("--actor-run", required=True)
    parser.add_argument("--critic-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-actions", type=int, default=41)
    parser.add_argument("--search-radius", type=float, required=True)
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--max-action-delta", type=float, default=4.0)
    parser.add_argument("--margins", type=float, nargs="+", required=True)
    parser.add_argument("--theta-bins", type=int, default=47)
    parser.add_argument("--velocity-bins", type=int, default=31)
    parser.add_argument("--random-points", type=int, default=5001)
    parser.add_argument("--reset-velocity-limit", type=float, default=1.0)
    parser.add_argument("--symmetric-actor-fallback", action="store_true")
    parser.add_argument(
        "--dp-solution",
        default=(
            "data/reference/pendulum_dp_solution.npz"
        ),
    )
    return parser.parse_args()


def metrics(rows: list[dict[str, Any]], reference: np.ndarray) -> dict[str, float]:
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    task = np.asarray([row["task_success"] for row in rows], dtype=np.float64)
    return {
        "points": float(len(rows)),
        "near_reference_eps": float((returns >= reference - 5.0).mean()),
        "task_success": float(task.mean()),
        "strict_beats_reference": float((returns > reference).mean()),
        "mean_return": float(returns.mean()),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    actor, config, _actor_payload = load_agent_from_run(
        Path(args.actor_run), device=args.device
    )
    critic, _critic_config, _critic_payload = load_agent_from_run(
        Path(args.critic_run), device=args.device
    )
    actor_checkpoint = Path(args.actor_run) / "checkpoints" / "final.pt"
    critic_checkpoint = Path(args.critic_run) / "checkpoints" / "final.pt"
    reliability = config.reliability
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    midpoint_theta, midpoint_velocity = midpoint_grid(
        int(args.theta_bins), int(args.velocity_bins)
    )
    random_rng = np.random.default_rng(9_700_000 + int(args.seed))
    random_theta = random_rng.uniform(-math.pi, math.pi, int(args.random_points))
    random_velocity = random_rng.uniform(
        -float(args.reset_velocity_limit),
        float(args.reset_velocity_limit),
        int(args.random_points),
    )
    all_theta = np.concatenate([midpoint_theta, random_theta])
    all_velocity = np.concatenate([midpoint_velocity, random_velocity])
    reference = validation_reference_returns(
        all_theta,
        all_velocity,
        detector,
        reliability,
        Path(args.dp_solution),
    )
    split = len(midpoint_theta)
    variants: list[dict[str, Any]] = []
    for margin in args.margins:
        if args.global_search:
            policy = FixedGlobalCriticQSearchPolicy(
                actor_agent=actor,
                critic_agent=critic,
                num_actions=int(args.num_actions),
                margin=float(margin),
                max_action_delta=float(args.max_action_delta),
                symmetric_actor_fallback=bool(args.symmetric_actor_fallback),
            )
        else:
            policy = FixedLocalCriticQSearchPolicy(
                actor_agent=actor,
                critic_agent=critic,
                num_actions=int(args.num_actions),
                margin=float(margin),
                search_radius=float(args.search_radius),
                symmetric_actor_fallback=bool(args.symmetric_actor_fallback),
            )
        rows = rollout_pendulum_grid_vectorized(
            policy,
            all_theta,
            all_velocity,
            detector,
            reliability,
            horizon=200,
        )
        variant = {
            "margin": float(margin),
            "num_actions": int(args.num_actions),
            "search_mode": "global" if args.global_search else "local",
            "search_radius": (
                None if args.global_search else float(args.search_radius)
            ),
            "max_action_delta": (
                float(args.max_action_delta) if args.global_search else None
            ),
            "filter": "unanimous_advantage",
            "symmetric_actor_fallback": bool(args.symmetric_actor_fallback),
            "midpoint": metrics(rows[:split], reference[:split]),
            "continuous_uniform_reset": metrics(rows[split:], reference[split:]),
            "selection": policy.selection_metrics(),
        }
        variants.append(variant)
        print(json.dumps(variant, sort_keys=True), flush=True)
    payload = {
        "protocol": {
            "authoritative_grid_queried": False,
            "actor_run": str(args.actor_run),
            "actor_checkpoint": str(actor_checkpoint),
            "critic_run": str(args.critic_run),
            "critic_checkpoint": str(critic_checkpoint),
            "seed": int(args.seed),
            "midpoint_grid": [int(args.theta_bins), int(args.velocity_bins)],
            "continuous_uniform_reset_points": int(args.random_points),
            "continuous_uniform_reset_rng_seed": 9_700_000 + int(args.seed),
            "same_global_rule_for_every_state": True,
            "search_mode": "global" if args.global_search else "local",
            "symmetric_actor_fallback": bool(args.symmetric_actor_fallback),
            "reference_used_at_inference": False,
        },
        "variants": variants,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
