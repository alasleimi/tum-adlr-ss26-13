from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from last_nine_rl.config import ExperimentConfig, resolve_device
from last_nine_rl.envs import UprightDetector, make_env
from last_nine_rl.evaluate import evaluate_agent, fixed_eval_seeds, threshold_fractions
from last_nine_rl.replay import InstrumentedReplayBuffer
from last_nine_rl.sac import SACAgent
from last_nine_rl.telemetry import TelemetryLogger, default_run_dir


EVAL_SERIES_KEYS = {
    "returns",
    "lengths",
    "near_upright_fractions",
    "min_step_rewards",
    "not_near_upright_streaks",
    "seeds",
}


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    apply_overrides(config, args)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(config)
    train(config, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Week 1 SAC baseline with reliability telemetry.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", default=None, help="Output directory. Defaults to runs/<name>/<timestamp>_seed<seed>.")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed-base", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--replay-inspection-interval", type=int, default=None)
    parser.add_argument("--diagnostics-interval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-replay", action="store_true", help="Write replay_final.npz at the end of training.")
    parser.add_argument("--overwrite", action="store_true", help="Delete known telemetry files in run-dir before training.")
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.seed is not None:
        config.seed = args.seed
    if args.env_id is not None:
        config.env.env_id = args.env_id
    if args.total_steps is not None:
        config.sac.total_steps = args.total_steps
    if getattr(args, "buffer_size", None) is not None:
        config.sac.buffer_size = args.buffer_size
    if getattr(args, "batch_size", None) is not None:
        config.sac.batch_size = args.batch_size
    if args.learning_starts is not None:
        config.sac.learning_starts = args.learning_starts
    if getattr(args, "updates_per_step", None) is not None:
        config.sac.updates_per_step = args.updates_per_step
    if args.eval_every_steps is not None:
        config.eval.every_steps = args.eval_every_steps
    if args.eval_episodes is not None:
        config.eval.episodes = args.eval_episodes
        config.eval.seeds = None
    if args.eval_seed_base is not None:
        config.eval.seed_base = args.eval_seed_base
        config.eval.seeds = None
    if args.log_interval is not None:
        config.telemetry.log_interval_steps = args.log_interval
    if args.replay_inspection_interval is not None:
        config.telemetry.replay_inspection_interval_steps = args.replay_inspection_interval
    if args.diagnostics_interval is not None:
        config.telemetry.diagnostics_interval_steps = args.diagnostics_interval
    if args.device is not None:
        config.sac.device = args.device
    if getattr(args, "save_replay", False):
        config.telemetry.save_replay = True
    if args.overwrite:
        config.telemetry.overwrite = True


def train(config: ExperimentConfig, run_dir: Path) -> Path:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.sac.device)
    config.sac.device = device

    logger = TelemetryLogger(run_dir, config)
    env = None
    try:
        env = make_env(config.env.env_id, seed=config.seed, max_episode_steps=config.env.max_episode_steps)
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, config.sac, device=device)
        
        # =====================================================================
        # 1. SETUP FEATURE NORM TRACKING VIA HOOKS
        # =====================================================================
        feature_norms = {}
        def register_feature_hooks(network: torch.nn.Module, prefix: str):
            def hook_fn(module, input, output):
                if isinstance(output, tuple): 
                    output = output[0]
                # Store the L2 norm of the layer's output activations
                feature_norms[f"{prefix}_{name}"] = torch.norm(output.detach(), p=2).item()
            
            for name, module in network.named_modules():
                if len(list(module.children())) == 0: # Targets leaf layers (Linear, LayerNorm, etc.)
                    module.register_forward_hook(hook_fn)

        # Register hooks for Actor and Critic if they are accessible attributes
        if hasattr(agent, "actor"): register_feature_hooks(agent.actor, "feat_actor")
        if hasattr(agent, "critic"): register_feature_hooks(agent.critic, "feat_critic")
        # =====================================================================

        replay = InstrumentedReplayBuffer(
            config.sac.buffer_size,
            env.observation_space,
            env.action_space,
            device,
            n_envs=1,
            handle_timeout_termination=False,
        )
        detector = UprightDetector(
            config.env.env_id,
            cos_threshold=config.reliability.near_upright_cos_threshold,
            abs_velocity_threshold=config.reliability.near_upright_abs_velocity_threshold,
        )
        eval_seeds = fixed_eval_seeds(config.eval.seed_base, config.eval.episodes, config.eval.seeds)

        logger.log_event(
            "run_start",
            0,
            {
                "run_dir": str(run_dir),
                "device": device,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "action_low": env.action_space.low.tolist(),
                "action_high": env.action_space.high.tolist(),
                "eval_seeds": eval_seeds,
            },
        )

        obs, _ = env.reset(seed=config.seed)
        episode_id = 0
        episode_return = 0.0
        episode_length = 0
        update_step = 0
        update_metrics_window: list[dict[str, float]] = []
        last_eval_step = log_evaluation(logger, agent, config, eval_seeds, step=0)

        for global_step in range(1, config.sac.total_steps + 1):
            if global_step <= config.sac.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            terminal_for_bootstrap = bool(terminated)
            episode_done = bool(terminated or truncated)
            replay.add(
                np.asarray(obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                np.asarray(next_obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                np.asarray(action, dtype=np.float32).reshape(1, *env.action_space.shape),
                np.asarray([float(reward)], dtype=np.float32),
                np.asarray([terminal_for_bootstrap], dtype=bool),
                [info],
                step=global_step,
                episode_id=episode_id,
            )

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if episode_done:
                episode_metrics = {
                    "return": episode_return,
                    "length": float(episode_length),
                    "success": float(episode_return >= config.reliability.success_return_threshold),
                    "collapse": float(episode_return <= config.reliability.collapse_return_threshold),
                }
                logger.log_event("episode", global_step, {"episode_id": episode_id, **episode_metrics})
                logger.log_metrics(global_step, "train_episode", episode_metrics)
                episode_id += 1
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()

            if global_step > config.sac.learning_starts and replay.size() >= config.sac.batch_size:
                for _ in range(config.sac.updates_per_step):
                    batch = replay.sample(config.sac.batch_size)
                    update_step += 1
                    
                    # Run the forward & backward passes inside the agent
                    metrics = agent.update(batch, update_step)

                    # =====================================================================
                    # 2. INJECT FEATURE AND PARAMETER NORMS INTO THE METRICS DICT
                    # =====================================================================
                    # Inject captured feature norms
                    for k, v in feature_norms.items():
                        metrics[k] = v

                    # Inject parameter (weight) norms for the Actor
                    if hasattr(agent, "actor"):
                        for name, param in agent.actor.named_parameters():
                            if param.requires_grad:
                                metrics[f"param_actor_{name}"] = torch.norm(param.data, p=2).item()
                                
                    # Inject parameter (weight) norms for the Critic
                    if hasattr(agent, "critic"):
                        for name, param in agent.critic.named_parameters():
                            if param.requires_grad:
                                metrics[f"param_critic_{name}"] = torch.norm(param.data, p=2).item()
                    # =====================================================================

                    update_metrics_window.append(metrics)

            if should_run(global_step, config.telemetry.log_interval_steps):
                log_update_window(logger, global_step, update_metrics_window)

            if should_run(global_step, config.telemetry.replay_inspection_interval_steps):
                replay_metrics = replay.summary(detector, global_step, action_high=env.action_space.high)
                logger.log_metrics(global_step, "replay", replay_metrics)
                logger.log_event("replay_inspection", global_step, replay_metrics)

            if (
                should_run(global_step, config.telemetry.diagnostics_interval_steps)
                and replay.size() >= config.sac.batch_size
            ):
                diag_batch = replay.sample(config.sac.batch_size, count=False)
                diagnostics = agent.diagnostics(diag_batch, config.reliability.dormant_relative_threshold)
                logger.log_metrics(global_step, "diagnostics", diagnostics)
                logger.log_event("diagnostics", global_step, diagnostics)

            if should_run(global_step, config.eval.every_steps):
                last_eval_step = log_evaluation(logger, agent, config, eval_seeds, step=global_step)

        log_update_window(logger, config.sac.total_steps, update_metrics_window)
        if last_eval_step != config.sac.total_steps:
            log_evaluation(logger, agent, config, eval_seeds, step=config.sac.total_steps)

        if config.telemetry.save_replay:
            replay_path = run_dir / "replay_final.npz"
            replay.save_npz(replay_path)
            logger.log_event("replay_saved", config.sac.total_steps, {"path": str(replay_path)})

        if config.telemetry.save_model:
            checkpoint_path = run_dir / "checkpoints" / "final.pt"
            agent.save_checkpoint(
                checkpoint_path,
                extra={
                    "global_step": config.sac.total_steps,
                    "update_step": update_step,
                    "run_dir": str(run_dir),
                },
            )
            logger.log_event("checkpoint_saved", config.sac.total_steps, {"path": str(checkpoint_path)})

        logger.log_event("run_complete", config.sac.total_steps, {"episodes": episode_id, "updates": update_step})
        return run_dir
    finally:
        if env is not None:
            env.close()
        logger.close()


def should_run(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def log_evaluation(
    logger: TelemetryLogger,
    agent: SACAgent,
    config: ExperimentConfig,
    eval_seeds: list[int],
    step: int,
) -> int:
    evaluation = evaluate_agent(
        agent,
        config.env,
        episodes=config.eval.episodes,
        reliability=config.reliability,
        deterministic=config.eval.deterministic,
        seeds=eval_seeds,
    )
    scalar_eval = {k: v for k, v in evaluation.items() if k not in EVAL_SERIES_KEYS}
    scalar_eval.update(threshold_fractions(evaluation["returns"], config.reliability.strict_return_thresholds))
    logger.log_metrics(step, "eval", scalar_eval)
    logger.log_eval_episodes(
        step,
        evaluation,
        success_threshold=config.reliability.success_return_threshold,
        collapse_threshold=config.reliability.collapse_return_threshold,
    )
    logger.log_event(
        "evaluation",
        step,
        {
            **scalar_eval,
            "seeds": evaluation["seeds"],
            "eval_episodes_csv": str(logger.run_dir / "eval_episodes.csv")
            if config.telemetry.write_eval_returns_csv
            else None,
        },
    )
    return step


def log_update_window(
    logger: TelemetryLogger,
    step: int,
    update_metrics_window: list[dict[str, float]],
) -> None:
    summary = summarize_metric_window(update_metrics_window)
    if not summary:
        return
    logger.log_metrics(step, "update", summary)
    logger.log_event("update", step, summary)
    update_metrics_window.clear()


def summarize_metric_window(metrics_window: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_window:
        return {}
    by_name: dict[str, list[float]] = {}
    for metrics in metrics_window:
        for name, value in metrics.items():
            by_name.setdefault(name, []).append(float(value))

    summary: dict[str, float] = {"num_optimizer_updates": float(len(metrics_window))}
    for name, values in sorted(by_name.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary[name] = float(arr[-1])
        summary[f"{name}_mean"] = float(np.mean(arr))
        summary[f"{name}_min"] = float(np.min(arr))
        summary[f"{name}_max"] = float(np.max(arr))
    return summary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()