from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from last_nine_rl.config import EnvConfig, EvalConfig, ExperimentConfig, ReliabilityConfig, SACConfig, TelemetryConfig, resolve_device
from last_nine_rl.envs import make_env
from last_nine_rl.evaluate import evaluate_agent, fixed_eval_seeds
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from last_nine_rl.sac import SACAgent


def main() -> None:
    args = parse_args()
    result = distill_reference(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill Pendulum max(DP, controller) actions into a SimbaV2-sized actor.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--policy", choices=["controller", "dp", "best"], default="best")
    parser.add_argument("--dp-solution-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--actor-backbone",
        choices=["cleanrl_mlp", "simba_v2"],
        default="simba_v2",
        help="Actor architecture used by the supervised DAgger learner.",
    )
    parser.add_argument("--actor-hidden-dim", type=int, default=32)
    parser.add_argument("--actor-blocks", type=int, default=1)
    parser.add_argument("--dataset-size", type=int, default=200_000)
    parser.add_argument("--eval-dataset-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--velocity-limit", type=float, default=8.0)
    parser.add_argument("--reset-support-fraction", type=float, default=0.0)
    parser.add_argument("--reset-support-velocity-limit", type=float, default=1.0)
    parser.add_argument("--near-down-fraction", type=float, default=0.0)
    parser.add_argument("--near-down-abs-theta-low-deg", type=float, default=150.0)
    parser.add_argument("--near-upright-fraction", type=float, default=0.0)
    parser.add_argument("--near-upright-abs-theta-high-deg", type=float, default=35.0)
    parser.add_argument("--near-upright-velocity-limit", type=float, default=1.0)
    parser.add_argument("--initial-dataset-source", choices=["state_sampler", "expert_rollout"], default="state_sampler")
    parser.add_argument("--initial-expert-episodes", type=int, default=0)
    parser.add_argument("--initial-expert-seed-base", type=int, default=150_000)
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--selection-metric", choices=["eval_action_mae", "last"], default="eval_action_mae")
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-episodes-per-iteration", type=int, default=0)
    parser.add_argument("--dagger-train-epochs-per-iteration", type=int, default=20)
    parser.add_argument("--dagger-max-dataset-size", type=int, default=0)
    parser.add_argument("--dagger-rollout-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--dagger-seed-base", type=int, default=200_000)
    parser.add_argument("--dagger-max-episode-steps", type=int, default=200)
    parser.add_argument("--rollout-backend", choices=["gym", "vectorized_pendulum"], default="gym")
    parser.add_argument("--dagger-expert-beta-start", type=float, default=0.0)
    parser.add_argument("--dagger-expert-beta-final", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-dataset", action="store_true")
    return parser.parse_args()


def seed_training_rngs(seed: int) -> None:
    """Seed every RNG used to initialize and optimize the distilled actor."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distill_reference(args: argparse.Namespace) -> dict[str, Any]:
    seed_training_rngs(int(args.seed))
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    device = resolve_device(str(args.device))
    config = distilled_config(args, device)
    config.to_json(run_dir / "config.json")

    env = make_env(config.env.env_id, seed=config.seed, max_episode_steps=config.env.max_episode_steps)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_low = env.action_space.low
        action_high = env.action_space.high
    finally:
        env.close()

    reference = PendulumReferenceGuidance(
        policy=args.policy,
        dp_solution_path=args.dp_solution_path,
        horizon=200,
    )
    initial_collection_metrics: dict[str, float] | None = None
    if str(args.initial_dataset_source) == "expert_rollout":
        if int(args.initial_expert_episodes) <= 0:
            raise ValueError("--initial-expert-episodes must be positive for expert_rollout initial data.")
        train_obs, train_actions, initial_collection_metrics = collect_reference_rollout_dataset(
            reference=reference,
            env_id=config.env.env_id,
            episodes=int(args.initial_expert_episodes),
            seed_base=int(args.initial_expert_seed_base) + 10_000 * int(args.seed),
            max_episode_steps=int(args.dagger_max_episode_steps),
            vectorized=bool(args.rollout_backend == "vectorized_pendulum"),
        )
    else:
        train_obs, train_actions = reference_dataset(
            reference=reference,
            size=int(args.dataset_size),
            rng=rng,
            velocity_limit=float(args.velocity_limit),
            reset_support_fraction=float(args.reset_support_fraction),
            reset_support_velocity_limit=float(args.reset_support_velocity_limit),
            near_down_fraction=float(args.near_down_fraction),
            near_down_abs_theta_low_deg=float(args.near_down_abs_theta_low_deg),
            near_upright_fraction=float(args.near_upright_fraction),
            near_upright_abs_theta_high_deg=float(args.near_upright_abs_theta_high_deg),
            near_upright_velocity_limit=float(args.near_upright_velocity_limit),
        )
    initial_reference_label_queries = int(train_obs.shape[0])
    eval_obs, eval_actions = reference_dataset(
        reference=reference,
        size=int(args.eval_dataset_size),
        rng=rng,
        velocity_limit=float(args.velocity_limit),
        reset_support_fraction=float(args.reset_support_fraction),
        reset_support_velocity_limit=float(args.reset_support_velocity_limit),
        near_down_fraction=float(args.near_down_fraction),
        near_down_abs_theta_low_deg=float(args.near_down_abs_theta_low_deg),
        near_upright_fraction=float(args.near_upright_fraction),
        near_upright_abs_theta_high_deg=float(args.near_upright_abs_theta_high_deg),
        near_upright_velocity_limit=float(args.near_upright_velocity_limit),
    )

    agent = SACAgent(obs_dim, action_low, action_high, config.sac, device=device)
    if agent.obs_rms is not None:
        agent.obs_rms.update(train_obs)
    init_payload_extra: dict[str, Any] | None = None
    if args.init_checkpoint:
        init_payload = agent.load_checkpoint(args.init_checkpoint, load_optimizers=False)
        extra = init_payload.get("extra", {})
        init_payload_extra = extra if isinstance(extra, dict) else {"extra": str(extra)}
    optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    action_scale = torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=device).reshape(1, -1)
    train_metrics: list[dict[str, float]] = []
    actor_optimizer_steps = 0
    best_eval_mae = math.inf
    best_epoch = 0
    best_actor_state: dict[str, torch.Tensor] | None = None
    total_epochs = 0
    if args.init_checkpoint:
        init_eval_metrics = action_fit_metrics(agent, eval_obs, eval_actions, int(args.batch_size), action_scale)
        best_eval_mae = init_eval_metrics["action_mae"]
        best_actor_state = {
            key: value.detach().cpu().clone()
            for key, value in agent.actor.state_dict().items()
        }
        train_metrics.append(
            {
                "epoch": 0.0,
                "dagger_iteration": 0.0,
                **{f"eval_{key}": value for key, value in init_eval_metrics.items()},
            }
        )
    for epoch in range(1, int(args.epochs) + 1):
        total_epochs = epoch
        epoch_metrics = train_epoch(
            agent,
            optimizer,
            train_obs,
            train_actions,
            batch_size=int(args.batch_size),
            action_scale=action_scale,
            rng=rng,
        )
        actor_optimizer_steps += int(math.ceil(train_obs.shape[0] / int(args.batch_size)))
        should_eval = epoch == 1 or epoch == int(args.epochs) or epoch % int(args.eval_every_epochs) == 0
        if should_eval:
            eval_metrics = action_fit_metrics(agent, eval_obs, eval_actions, int(args.batch_size), action_scale)
            epoch_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            if eval_metrics["action_mae"] < best_eval_mae:
                best_eval_mae = eval_metrics["action_mae"]
                best_epoch = epoch
                best_actor_state = {
                    key: value.detach().cpu().clone()
                    for key, value in agent.actor.state_dict().items()
                }
        epoch_metrics["epoch"] = float(epoch)
        epoch_metrics["dagger_iteration"] = 0.0
        train_metrics.append(epoch_metrics)

    dagger_collection_metrics: list[dict[str, float]] = []
    if int(args.dagger_iterations) > 0:
        if int(args.dagger_episodes_per_iteration) <= 0:
            raise ValueError("--dagger-episodes-per-iteration must be positive when --dagger-iterations is positive.")
        for dagger_iteration in range(1, int(args.dagger_iterations) + 1):
            expert_beta = dagger_expert_beta(
                dagger_iteration,
                int(args.dagger_iterations),
                start=float(args.dagger_expert_beta_start),
                final=float(args.dagger_expert_beta_final),
            )
            dagger_obs, dagger_actions, collection_metrics = collect_dagger_dataset(
                agent=agent,
                reference=reference,
                env_id=config.env.env_id,
                episodes=int(args.dagger_episodes_per_iteration),
                seed_base=int(args.dagger_seed_base) + 10_000 * (dagger_iteration - 1),
                max_episode_steps=int(args.dagger_max_episode_steps),
                deterministic=bool(args.dagger_rollout_mode == "deterministic"),
                expert_beta=expert_beta,
                vectorized=bool(args.rollout_backend == "vectorized_pendulum"),
            )
            collection_metrics["dagger_iteration"] = float(dagger_iteration)
            dagger_collection_metrics.append(collection_metrics)
            train_obs, train_actions = append_capped_dataset(
                train_obs,
                train_actions,
                dagger_obs,
                dagger_actions,
                max_size=int(args.dagger_max_dataset_size),
                rng=rng,
            )
            if agent.obs_rms is not None:
                agent.obs_rms.update(dagger_obs)

            for local_epoch in range(1, int(args.dagger_train_epochs_per_iteration) + 1):
                total_epochs += 1
                epoch_metrics = train_epoch(
                    agent,
                    optimizer,
                    train_obs,
                    train_actions,
                    batch_size=int(args.batch_size),
                    action_scale=action_scale,
                    rng=rng,
                )
                actor_optimizer_steps += int(math.ceil(train_obs.shape[0] / int(args.batch_size)))
                should_eval = (
                    local_epoch == 1
                    or local_epoch == int(args.dagger_train_epochs_per_iteration)
                    or local_epoch % int(args.eval_every_epochs) == 0
                )
                if should_eval:
                    eval_metrics = action_fit_metrics(agent, eval_obs, eval_actions, int(args.batch_size), action_scale)
                    epoch_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
                    if eval_metrics["action_mae"] < best_eval_mae:
                        best_eval_mae = eval_metrics["action_mae"]
                        best_epoch = total_epochs
                        best_actor_state = {
                            key: value.detach().cpu().clone()
                            for key, value in agent.actor.state_dict().items()
                        }
                if local_epoch == 1:
                    epoch_metrics.update({f"dagger_collect_{key}": value for key, value in collection_metrics.items()})
                epoch_metrics["epoch"] = float(total_epochs)
                epoch_metrics["dagger_iteration"] = float(dagger_iteration)
                epoch_metrics["dataset_size"] = float(train_obs.shape[0])
                train_metrics.append(epoch_metrics)

    selected_epoch = total_epochs if args.selection_metric == "last" else best_epoch
    if args.selection_metric == "eval_action_mae" and best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
    eval_seeds = fixed_eval_seeds(int(args.eval_seed_base), int(args.eval_episodes))
    final_eval = evaluate_agent(
        agent,
        config.env,
        episodes=int(args.eval_episodes),
        reliability=config.reliability,
        deterministic=True,
        seeds=eval_seeds,
    )
    final_eval = {key: value for key, value in final_eval.items() if key not in {"returns", "lengths"}}

    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    agent.save_checkpoint(
        checkpoint_path,
        extra={
            "global_step": 0,
            "distillation": True,
            "reference_policy": args.policy,
            "best_epoch": best_epoch,
            "best_eval_action_mae": best_eval_mae,
            "selection_metric": args.selection_metric,
            "selected_epoch": selected_epoch,
            "dagger_iterations": int(args.dagger_iterations),
            "initial_dataset_source": str(args.initial_dataset_source),
            "initial_collection_metrics": initial_collection_metrics,
            "rollout_backend": str(args.rollout_backend),
            "training_environment_steps": int(
                (initial_collection_metrics or {}).get("samples", 0.0)
                + sum(item["samples"] for item in dagger_collection_metrics)
            ),
            "reference_label_queries": int(
                initial_reference_label_queries
                + sum(item["samples"] for item in dagger_collection_metrics)
            ),
            "actor_optimizer_steps": int(actor_optimizer_steps),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        },
    )
    write_metrics_csv(run_dir / "metrics.csv", train_metrics, final_eval)
    write_events(
        run_dir / "events.jsonl",
        args,
        train_metrics,
        final_eval,
        checkpoint_path,
        config,
        reference.metadata(),
        dagger_collection_metrics,
    )
    if args.save_dataset:
        np.savez_compressed(
            run_dir / "distillation_dataset.npz",
            train_obs=train_obs,
            train_actions=train_actions,
            eval_obs=eval_obs,
            eval_actions=eval_actions,
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "actor_backbone": str(args.actor_backbone),
        "reference": reference.metadata(),
        "dataset_size": int(train_obs.shape[0]),
        "eval_dataset_size": int(eval_obs.shape[0]),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "init_checkpoint_extra": init_payload_extra,
        "dagger_iterations": int(args.dagger_iterations),
        "dagger_collection_metrics": dagger_collection_metrics,
        "initial_dataset_source": str(args.initial_dataset_source),
        "initial_collection_metrics": initial_collection_metrics,
        "rollout_backend": str(args.rollout_backend),
        "training_environment_steps": int(
            (initial_collection_metrics or {}).get("samples", 0.0)
            + sum(item["samples"] for item in dagger_collection_metrics)
        ),
        "reference_label_queries": int(
            initial_reference_label_queries
            + sum(item["samples"] for item in dagger_collection_metrics)
        ),
        "actor_optimizer_steps": int(actor_optimizer_steps),
        "initial_training_epochs": int(args.epochs),
        "dagger_training_epochs_per_iteration": int(args.dagger_train_epochs_per_iteration),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "best_epoch": int(best_epoch),
        "best_eval_action_mae": float(best_eval_mae),
        "selection_metric": str(args.selection_metric),
        "selected_epoch": int(selected_epoch),
        "final_eval": final_eval,
    }
    (run_dir / "distillation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def distilled_config(args: argparse.Namespace, device: str) -> ExperimentConfig:
    use_simba = str(args.actor_backbone) == "simba_v2"
    return ExperimentConfig(
        name=f"pendulum_distill_{args.policy}_{args.actor_backbone}",
        seed=int(args.seed),
        env=EnvConfig(env_id="Pendulum-v1", max_episode_steps=None),
        sac=SACConfig(
            total_steps=1,
            learning_starts=0,
            buffer_size=max(2, int(args.dataset_size)),
            batch_size=int(args.batch_size),
            device=device,
            simba_backbone=use_simba,
            simba_weight_projection=use_simba,
            simba_distributional_critic=use_simba,
            simba_reward_scaling=use_simba,
            simba_feature_norm=use_simba,
            simba_input_shift=use_simba,
            simba_observation_norm=use_simba,
            simba_actor_hidden_dim=int(getattr(args, "actor_hidden_dim", 32)),
            simba_actor_blocks=int(getattr(args, "actor_blocks", 1)),
            simba_critic_hidden_dim=64,
            simba_critic_num_bins=51,
            policy_lr=float(args.lr),
            q_lr=float(args.lr),
            policy_lr_final=None,
            q_lr_final=None,
            alpha_initial_value=0.01,
            target_entropy_scale=-0.5,
            update_diagnostics=False,
        ),
        eval=EvalConfig(
            every_steps=1,
            episodes=int(args.eval_episodes),
            deterministic=True,
            seed_base=int(args.eval_seed_base),
        ),
        reliability=ReliabilityConfig(),
        telemetry=TelemetryConfig(
            run_root=str(Path(args.run_dir).parent),
            tensorboard=False,
            overwrite=bool(args.overwrite),
            save_model=True,
            save_replay=False,
        ),
    )


def reference_dataset(
    reference: PendulumReferenceGuidance,
    size: int,
    rng: np.random.Generator,
    velocity_limit: float,
    reset_support_fraction: float = 0.0,
    reset_support_velocity_limit: float = 1.0,
    near_down_fraction: float = 0.0,
    near_down_abs_theta_low_deg: float = 150.0,
    near_upright_fraction: float = 0.0,
    near_upright_abs_theta_high_deg: float = 35.0,
    near_upright_velocity_limit: float = 1.0,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(-math.pi, math.pi, size=size).astype(np.float32)
    theta_dot = rng.uniform(-velocity_limit, velocity_limit, size=size).astype(np.float32)
    reset_support_count = int(round(size * min(max(reset_support_fraction, 0.0), 1.0)))
    near_down_count = int(round(size * min(max(near_down_fraction, 0.0), 1.0)))
    near_upright_count = int(round(size * min(max(near_upright_fraction, 0.0), 1.0)))
    allocated_count = reset_support_count + near_down_count + near_upright_count
    if allocated_count > size:
        scale = size / float(allocated_count)
        reset_support_count = int(math.floor(reset_support_count * scale))
        near_down_count = int(math.floor(near_down_count * scale))
        near_upright_count = size - reset_support_count - near_down_count
    available_indices = rng.permutation(size)
    cursor = 0
    if reset_support_count > 0:
        reset_indices = available_indices[cursor : cursor + reset_support_count]
        cursor += reset_support_count
        theta[reset_indices] = rng.uniform(-math.pi, math.pi, size=reset_support_count).astype(np.float32)
        theta_dot[reset_indices] = rng.uniform(
            -reset_support_velocity_limit,
            reset_support_velocity_limit,
            size=reset_support_count,
        ).astype(np.float32)
    if near_down_count > 0:
        low = math.radians(float(near_down_abs_theta_low_deg))
        hard_abs_theta = rng.uniform(low, math.pi, size=near_down_count).astype(np.float32)
        hard_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=near_down_count)
        hard_indices = available_indices[cursor : cursor + near_down_count]
        cursor += near_down_count
        theta[hard_indices] = hard_abs_theta * hard_sign
        theta_dot[hard_indices] = rng.uniform(-velocity_limit, velocity_limit, size=near_down_count).astype(np.float32)
    if near_upright_count > 0:
        high = math.radians(float(near_upright_abs_theta_high_deg))
        local_abs_theta = rng.uniform(0.0, high, size=near_upright_count).astype(np.float32)
        local_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=near_upright_count)
        local_indices = available_indices[cursor : cursor + near_upright_count]
        theta[local_indices] = local_abs_theta * local_sign
        theta_dot[local_indices] = rng.uniform(
            -near_upright_velocity_limit,
            near_upright_velocity_limit,
            size=near_upright_count,
        ).astype(np.float32)
    obs = np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)
    actions = np.empty((size, 1), dtype=np.float32)
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        actions[start:end] = reference.act_batch(obs[start:end])
    return obs, actions


def collect_reference_rollout_dataset(
    reference: PendulumReferenceGuidance,
    env_id: str,
    episodes: int,
    seed_base: int,
    max_episode_steps: int,
    vectorized: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if vectorized:
        return collect_reference_rollout_dataset_vectorized(
            reference=reference,
            env_id=env_id,
            episodes=episodes,
            seed_base=seed_base,
            max_episode_steps=max_episode_steps,
        )
    env = make_env(env_id, seed=int(seed_base), max_episode_steps=int(max_episode_steps))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[float] = []
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=int(seed_base) + episode)
            episode_return = 0.0
            episode_length = 0
            for step in range(int(max_episode_steps)):
                remaining_steps = int(max_episode_steps) - step
                reference_action = reference.act(obs, remaining_steps=remaining_steps)
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(np.asarray(reference_action, dtype=np.float32).reshape(-1).copy())
                next_obs, reward, terminated, truncated, _info = env.step(
                    np.asarray(reference_action, dtype=np.float32).reshape(env.action_space.shape)
                )
                episode_return += float(reward)
                episode_length += 1
                obs = next_obs
                if bool(terminated or truncated):
                    break
            returns.append(episode_return)
            lengths.append(float(episode_length))
    finally:
        env.close()

    if not observations:
        raise RuntimeError("Reference rollout collection produced no observations.")
    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32).reshape(obs_array.shape[0], -1)
    metrics = {
        "episodes": float(episodes),
        "samples": float(obs_array.shape[0]),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
    }
    return obs_array, action_array, metrics


def pendulum_initial_states(
    env_id: str,
    episodes: int,
    seed_base: int,
    max_episode_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read Gym's seeded reset distribution, then return its latent Pendulum states."""
    if not str(env_id).startswith("Pendulum"):
        raise ValueError("The vectorized rollout backend only supports Pendulum environments.")
    env = make_env(env_id, seed=int(seed_base), max_episode_steps=int(max_episode_steps))
    theta = np.empty(int(episodes), dtype=np.float64)
    theta_dot = np.empty(int(episodes), dtype=np.float64)
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=int(seed_base) + episode)
            theta[episode] = float(np.arctan2(obs[1], obs[0]))
            theta_dot[episode] = float(obs[2])
    finally:
        env.close()
    return theta, theta_dot


def pendulum_obs_batch(theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)


def pendulum_step_batch(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    action: np.ndarray,
    g: float = 10.0,
    m: float = 1.0,
    length: float = 1.0,
    dt: float = 0.05,
    max_speed: float = 8.0,
    max_torque: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torque = np.clip(np.asarray(action, dtype=np.float64), -max_torque, max_torque)
    normalized_theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi
    costs = normalized_theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2
    new_theta_dot = theta_dot + (
        3.0 * g / (2.0 * length) * np.sin(theta) + 3.0 / (m * length**2) * torque
    ) * dt
    new_theta_dot = np.clip(new_theta_dot, -max_speed, max_speed)
    new_theta = theta + new_theta_dot * dt
    return new_theta, new_theta_dot, -costs


def collect_reference_rollout_dataset_vectorized(
    reference: PendulumReferenceGuidance,
    env_id: str,
    episodes: int,
    seed_base: int,
    max_episode_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    theta, theta_dot = pendulum_initial_states(env_id, episodes, seed_base, max_episode_steps)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    returns = np.zeros(int(episodes), dtype=np.float64)
    for step in range(int(max_episode_steps)):
        obs = pendulum_obs_batch(theta, theta_dot)
        reference_action = reference.act_batch(obs, remaining_steps=int(max_episode_steps) - step)
        observations.append(obs)
        actions.append(np.asarray(reference_action, dtype=np.float32).reshape(int(episodes), -1))
        theta, theta_dot, reward = pendulum_step_batch(theta, theta_dot, reference_action.reshape(-1))
        returns += reward
    obs_array = np.concatenate(observations, axis=0)
    action_array = np.concatenate(actions, axis=0)
    metrics = {
        "episodes": float(episodes),
        "samples": float(obs_array.shape[0]),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(max_episode_steps),
    }
    return obs_array, action_array, metrics


def dagger_expert_beta(iteration: int, iterations: int, start: float, final: float) -> float:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if iteration < 1 or iteration > iterations:
        raise ValueError("iteration must be in [1, iterations]")
    if not (0.0 <= start <= 1.0 and 0.0 <= final <= 1.0):
        raise ValueError("DAgger expert beta endpoints must be in [0, 1]")
    if iterations == 1:
        return float(start)
    progress = float(iteration - 1) / float(iterations - 1)
    return float(start + progress * (final - start))


def train_epoch(
    agent: SACAgent,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    action_scale: torch.Tensor,
    rng: np.random.Generator,
) -> dict[str, float]:
    agent.actor.train()
    indices = rng.permutation(observations.shape[0])
    losses: list[float] = []
    maes: list[float] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        obs = torch.as_tensor(observations[batch_idx], dtype=torch.float32, device=agent.device)
        target = torch.as_tensor(actions[batch_idx], dtype=torch.float32, device=agent.device)
        norm_obs = agent._normalize_obs_tensor(obs)
        _sampled, _log_prob, pred = agent.actor.get_action(norm_obs)
        scaled_error = (pred - target) / action_scale.clamp_min(1e-6)
        loss = F.smooth_l1_loss(scaled_error, torch.zeros_like(scaled_error), beta=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=10.0)
        optimizer.step()
        if agent.cfg.simba_weight_projection:
            agent._project_weights()
        losses.append(float(loss.detach().cpu()))
        maes.append(float(torch.mean(torch.abs(pred.detach() - target)).cpu()))
    return {
        "train_loss": float(np.mean(losses)),
        "train_action_mae": float(np.mean(maes)),
    }


def collect_dagger_dataset(
    agent: SACAgent,
    reference: PendulumReferenceGuidance,
    env_id: str,
    episodes: int,
    seed_base: int,
    max_episode_steps: int,
    deterministic: bool,
    expert_beta: float = 0.0,
    vectorized: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not 0.0 <= float(expert_beta) <= 1.0:
        raise ValueError("expert_beta must be in [0, 1]")
    if vectorized:
        return collect_dagger_dataset_vectorized(
            agent=agent,
            reference=reference,
            env_id=env_id,
            episodes=episodes,
            seed_base=seed_base,
            max_episode_steps=max_episode_steps,
            deterministic=deterministic,
            expert_beta=expert_beta,
        )
    env = make_env(env_id, seed=int(seed_base), max_episode_steps=int(max_episode_steps))
    mixture_rng = np.random.default_rng(int(seed_base) + 7919)
    observations: list[np.ndarray] = []
    policy_actions: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    returns: list[float] = []
    lengths: list[float] = []
    reference_executions = 0
    try:
        for episode in range(int(episodes)):
            obs, _info = env.reset(seed=int(seed_base) + episode)
            episode_return = 0.0
            episode_length = 0
            for step in range(int(max_episode_steps)):
                remaining_steps = int(max_episode_steps) - step
                policy_action = agent.act(obs, deterministic=deterministic)
                reference_action = reference.act(obs, remaining_steps=remaining_steps)
                execute_reference = bool(mixture_rng.random() < float(expert_beta))
                execution_action = reference_action if execute_reference else policy_action
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                policy_actions.append(np.asarray(policy_action, dtype=np.float32).reshape(-1).copy())
                reference_actions.append(np.asarray(reference_action, dtype=np.float32).reshape(-1).copy())
                reference_executions += int(execute_reference)
                next_obs, reward, terminated, truncated, _info = env.step(
                    np.asarray(execution_action, dtype=np.float32).reshape(env.action_space.shape)
                )
                episode_return += float(reward)
                episode_length += 1
                obs = next_obs
                if bool(terminated or truncated):
                    break
            returns.append(episode_return)
            lengths.append(float(episode_length))
    finally:
        env.close()

    if not observations:
        raise RuntimeError("DAgger rollout collection produced no observations.")
    obs_array = np.asarray(observations, dtype=np.float32)
    policy_action_array = np.asarray(policy_actions, dtype=np.float32).reshape(obs_array.shape[0], -1)
    action_array = np.asarray(reference_actions, dtype=np.float32).reshape(obs_array.shape[0], -1)
    policy_ref_abs_error = np.abs(policy_action_array - action_array)
    metrics = {
        "episodes": float(episodes),
        "samples": float(obs_array.shape[0]),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "policy_ref_action_mae": float(np.mean(policy_ref_abs_error)),
        "deterministic": float(deterministic),
        "expert_beta": float(expert_beta),
        "executed_reference_fraction": float(reference_executions / obs_array.shape[0]),
    }
    return obs_array, action_array, metrics


def collect_dagger_dataset_vectorized(
    agent: SACAgent,
    reference: PendulumReferenceGuidance,
    env_id: str,
    episodes: int,
    seed_base: int,
    max_episode_steps: int,
    deterministic: bool,
    expert_beta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    theta, theta_dot = pendulum_initial_states(env_id, episodes, seed_base, max_episode_steps)
    mixture_rng = np.random.default_rng(int(seed_base) + 7919)
    observations: list[np.ndarray] = []
    policy_actions: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    returns = np.zeros(int(episodes), dtype=np.float64)
    reference_executions = 0
    for step in range(int(max_episode_steps)):
        obs = pendulum_obs_batch(theta, theta_dot)
        policy_action = agent.act_batch(obs, deterministic=deterministic).reshape(int(episodes), -1)
        reference_action = reference.act_batch(obs, remaining_steps=int(max_episode_steps) - step).reshape(
            int(episodes), -1
        )
        execute_reference = mixture_rng.random(int(episodes)) < float(expert_beta)
        execution_action = np.where(execute_reference[:, None], reference_action, policy_action)
        observations.append(obs)
        policy_actions.append(np.asarray(policy_action, dtype=np.float32))
        reference_actions.append(np.asarray(reference_action, dtype=np.float32))
        reference_executions += int(np.sum(execute_reference))
        theta, theta_dot, reward = pendulum_step_batch(theta, theta_dot, execution_action.reshape(-1))
        returns += reward

    obs_array = np.concatenate(observations, axis=0)
    policy_action_array = np.concatenate(policy_actions, axis=0)
    action_array = np.concatenate(reference_actions, axis=0)
    metrics = {
        "episodes": float(episodes),
        "samples": float(obs_array.shape[0]),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(max_episode_steps),
        "policy_ref_action_mae": float(np.mean(np.abs(policy_action_array - action_array))),
        "deterministic": float(deterministic),
        "expert_beta": float(expert_beta),
        "executed_reference_fraction": float(reference_executions / obs_array.shape[0]),
    }
    return obs_array, action_array, metrics


def append_capped_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    new_observations: np.ndarray,
    new_actions: np.ndarray,
    max_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    combined_observations = np.concatenate([observations, new_observations], axis=0)
    combined_actions = np.concatenate([actions, new_actions], axis=0)
    if int(max_size) <= 0 or combined_observations.shape[0] <= int(max_size):
        return combined_observations, combined_actions
    indices = rng.choice(combined_observations.shape[0], size=int(max_size), replace=False)
    return combined_observations[indices], combined_actions[indices]


def action_fit_metrics(
    agent: SACAgent,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    action_scale: torch.Tensor,
) -> dict[str, float]:
    agent.actor.eval()
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    match_01: list[float] = []
    match_025: list[float] = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            end = min(start + batch_size, observations.shape[0])
            obs = torch.as_tensor(observations[start:end], dtype=torch.float32, device=agent.device)
            target = torch.as_tensor(actions[start:end], dtype=torch.float32, device=agent.device)
            norm_obs = agent._normalize_obs_tensor(obs)
            _sampled, _log_prob, pred = agent.actor.get_action(norm_obs)
            error = pred - target
            abs_error = torch.abs(error)
            abs_errors.append(float(torch.mean(abs_error).cpu()))
            sq_errors.append(float(torch.mean(error.square()).cpu()))
            match_01.append(float(torch.mean((abs_error <= 0.1).to(torch.float32)).cpu()))
            match_025.append(float(torch.mean((abs_error <= 0.25).to(torch.float32)).cpu()))
    return {
        "action_mae": float(np.mean(abs_errors)),
        "action_rmse": float(math.sqrt(float(np.mean(sq_errors)))),
        "action_match_abs_le_0_10": float(np.mean(match_01)),
        "action_match_abs_le_0_25": float(np.mean(match_025)),
        "action_scale": float(action_scale.detach().cpu().view(-1)[0]),
    }


def write_metrics_csv(path: Path, train_metrics: list[dict[str, float]], final_eval: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for item in train_metrics:
        epoch = int(item["epoch"])
        for key, value in item.items():
            if key == "epoch":
                continue
            rows.append({"step": epoch, "split": "distill", "name": key, "value": value})
    for key, value in final_eval.items():
        if isinstance(value, int | float):
            rows.append({"step": int(train_metrics[-1]["epoch"]) if train_metrics else 0, "split": "eval", "name": key, "value": value})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "split", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_events(
    path: Path,
    args: argparse.Namespace,
    train_metrics: list[dict[str, float]],
    final_eval: dict[str, Any],
    checkpoint_path: Path,
    config: ExperimentConfig,
    reference_metadata: dict[str, Any],
    dagger_collection_metrics: list[dict[str, float]],
) -> None:
    final_step = int(train_metrics[-1]["epoch"]) if train_metrics else 0
    events = [
        {
            "step": 0,
            "type": "run_start",
            "payload": {
                "mode": "distillation",
                "actor_backbone": str(args.actor_backbone),
                "reference": reference_metadata,
                "dataset_size": int(args.dataset_size),
                "eval_dataset_size": int(args.eval_dataset_size),
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                "selection_metric": str(args.selection_metric),
                "dagger_iterations": int(args.dagger_iterations),
                "dagger_episodes_per_iteration": int(args.dagger_episodes_per_iteration),
                "dagger_train_epochs_per_iteration": int(args.dagger_train_epochs_per_iteration),
                "dagger_rollout_mode": str(args.dagger_rollout_mode),
                "rollout_backend": str(args.rollout_backend),
                "dagger_expert_beta_start": float(args.dagger_expert_beta_start),
                "dagger_expert_beta_final": float(args.dagger_expert_beta_final),
                "initial_dataset_source": str(args.initial_dataset_source),
                "initial_expert_episodes": int(args.initial_expert_episodes),
                "initial_expert_seed_base": int(args.initial_expert_seed_base),
                "config": config.to_dict(),
            },
        },
    ]
    for item in dagger_collection_metrics:
        events.append(
            {
                "step": int(item["dagger_iteration"]),
                "type": "dagger_collection",
                "payload": item,
            }
        )
    for item in train_metrics:
        events.append({"step": int(item["epoch"]), "type": "distillation", "payload": item})
    events.extend(
        [
            {"step": final_step, "type": "evaluation", "payload": final_eval},
            {"step": final_step, "type": "checkpoint_saved", "payload": {"path": str(checkpoint_path)}},
            {"step": final_step, "type": "run_complete", "payload": {"epochs": final_step}},
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
