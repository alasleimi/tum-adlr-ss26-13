from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.evaluate import evaluate_policy
from last_nine_rl.hybrid_qsearch import (
    FixedGlobalCriticQSearchPolicy,
    FixedLocalCriticQSearchPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIXED_CRITIC = (
    PROJECT_ROOT
    / "artifacts"
    / "report_reproduction"
    / "models"
    / "mixed_shared_critic"
    / "seed1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="last-nine-eval",
        description="Run a deterministic Gym-reset smoke evaluation of one retained checkpoint.",
    )
    parser.add_argument("run", type=Path, help="Run directory containing config.json and checkpoints/final.pt")
    parser.add_argument(
        "--deployment",
        choices=("actor", "pure-qsearch", "mixed-qsearch"),
        default="actor",
        help="Action-selection policy to evaluate (default: actor)",
    )
    parser.add_argument(
        "--critic-run",
        type=Path,
        help="Critic run for mixed-qsearch (default: retained shared critic)",
    )
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--critic-checkpoint", default="final.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit Gym reset seeds; overrides --episodes and --seed-base",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _batch_action(policy: object) -> Callable[[np.ndarray], np.ndarray]:
    def act(observation: np.ndarray) -> np.ndarray:
        batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return np.asarray(
            policy.act_batch(batch, deterministic=True), dtype=np.float32
        ).reshape(1, -1)[0]

    return act


def _policy_for(args: argparse.Namespace, agent: object) -> tuple[Callable[[np.ndarray], np.ndarray], str | None]:
    if args.deployment == "actor":
        return lambda observation: agent.act(observation, deterministic=True), None
    if args.deployment == "pure-qsearch":
        policy = FixedGlobalCriticQSearchPolicy(
            agent,
            agent,
            num_actions=41,
            margin=0.005,
            max_action_delta=4.0,
            symmetric_actor_fallback=True,
        )
        return _batch_action(policy), None
    critic_run = (args.critic_run or DEFAULT_MIXED_CRITIC).expanduser().resolve()
    critic, _, _ = load_agent_from_run(
        critic_run,
        device=args.device,
        checkpoint=args.critic_checkpoint,
        load_optimizers=False,
    )
    policy = FixedLocalCriticQSearchPolicy(
        agent,
        critic,
        num_actions=5,
        margin=0.0,
        search_radius=0.10,
        symmetric_actor_fallback=False,
    )
    return _batch_action(policy), str(critic_run)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = args.run.expanduser().resolve()
    agent, config, _ = load_agent_from_run(
        run,
        device=args.device,
        checkpoint=args.checkpoint,
        load_optimizers=False,
    )
    act, critic_run = _policy_for(args, agent)
    if args.seeds is not None:
        explicit_seeds = tuple(args.seeds)
    elif args.episodes is not None or args.seed_base is not None:
        explicit_seeds = None
    else:
        explicit_seeds = config.eval.seeds
    episodes = (
        len(explicit_seeds)
        if explicit_seeds is not None
        else int(args.episodes if args.episodes is not None else config.eval.episodes)
    )
    if episodes <= 0:
        raise ValueError("--episodes must be positive")
    seed_base = int(
        args.seed_base if args.seed_base is not None else config.eval.seed_base
    )
    metrics = evaluate_policy(
        act,
        config.env,
        episodes=episodes,
        reliability=config.reliability,
        seeds=explicit_seeds,
        seed=seed_base,
    )
    result = {
        "schema": "last-nine-checkpoint-evaluation/v1",
        "qualification": (
            "Gym-reset smoke evaluation. Report-grade common-grid counts are "
            "verified separately by `last-nine evaluate`."
        ),
        "run": str(run),
        "checkpoint": args.checkpoint,
        "deployment": args.deployment,
        "critic_run": critic_run,
        "device": config.sac.device,
        "metrics": metrics,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to replace {output}; pass --overwrite explicitly"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
