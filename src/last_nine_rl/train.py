from __future__ import annotations

import argparse
from contextlib import contextmanager
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from last_nine_rl.config import (
    ExperimentConfig,
    needs_actor_reference_actions,
    reference_auxiliary_loss_can_be_active,
    resolve_device,
)
from last_nine_rl.distill_reference import reference_dataset
from last_nine_rl.envs import (
    UprightDetector,
    make_env,
    pendulum_failure_reset_stats,
    replace_pendulum_failure_reset_states,
    set_pendulum_hard_reset_probability,
)
from last_nine_rl.evaluate import evaluate_agent, fixed_eval_seeds
from last_nine_rl.failure_curriculum import PendulumFailureStartCurriculum
from last_nine_rl.pendulum_dp import PendulumDPParams, pendulum_step_model
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from last_nine_rl.replay import InstrumentedReplayBuffer, concatenate_replay_samples, concatenate_sacn_replay_samples
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


def pendulum_hard_reset_probability_at_step(config: ExperimentConfig, step: int) -> float:
    start_step = int(config.env.pendulum_hard_reset_start_step)
    if step < start_step:
        return 0.0
    initial = float(config.env.pendulum_hard_reset_prob)
    decay_steps = int(config.env.pendulum_hard_reset_decay_steps)
    if decay_steps <= 0:
        return initial
    final = float(config.env.pendulum_hard_reset_final_prob)
    fraction = min(max(float(step - start_step) / float(decay_steps), 0.0), 1.0)
    return initial + fraction * (final - initial)


def pendulum_hard_reset_schedule_enabled(config: ExperimentConfig) -> bool:
    return max(float(config.env.pendulum_hard_reset_prob), float(config.env.pendulum_hard_reset_final_prob)) > 0.0


def pendulum_failure_curriculum_enabled(config: ExperimentConfig) -> bool:
    return float(config.env.pendulum_failure_reset_prob) > 0.0


def uniform_exploration_probability_at_step(config: ExperimentConfig, step: int) -> float:
    start_step = int(config.sac.uniform_exploration_start_step)
    if step < start_step:
        return 0.0
    initial = float(config.sac.uniform_exploration_initial_probability)
    decay_steps = int(config.sac.uniform_exploration_decay_steps)
    if decay_steps <= 0:
        return initial
    final = float(config.sac.uniform_exploration_final_probability)
    fraction = min(max(float(step - start_step) / float(decay_steps), 0.0), 1.0)
    return initial + fraction * (final - initial)


def pendulum_hard_replay_fraction_at_step(config: ExperimentConfig, step: int) -> float:
    start_step = int(config.sac.pendulum_hard_replay_start_step)
    if step < start_step:
        return 0.0
    initial = float(config.sac.pendulum_hard_replay_fraction)
    decay_steps = int(config.sac.pendulum_hard_replay_decay_steps)
    if decay_steps <= 0:
        return initial
    final = float(config.sac.pendulum_hard_replay_final_fraction)
    fraction = min(max(float(step - start_step) / float(decay_steps), 0.0), 1.0)
    return initial + fraction * (final - initial)


def populate_reference_prior_rollout_dataset(
    replay: InstrumentedReplayBuffer,
    reference: PendulumReferenceGuidance,
    config: ExperimentConfig,
) -> dict[str, float]:
    dataset_steps = int(config.sac.reference_prior_dataset_steps)
    if dataset_steps <= 0:
        return {"reference_prior_dataset_steps": 0.0, "reference_prior_dataset_episodes": 0.0}

    horizon = int(config.env.max_episode_steps or 200)
    seed = int(config.seed + config.sac.reference_prior_dataset_seed_offset)
    env = make_env(
        config.env.env_id,
        seed=seed,
        max_episode_steps=config.env.max_episode_steps,
        pendulum_hard_reset_prob=pendulum_hard_reset_probability_at_step(config, 0),
        pendulum_hard_reset_enabled=pendulum_hard_reset_schedule_enabled(config),
        pendulum_hard_reset_abs_theta_low=config.env.pendulum_hard_reset_abs_theta_low,
        pendulum_hard_reset_abs_theta_high=config.env.pendulum_hard_reset_abs_theta_high,
        pendulum_hard_reset_velocity_limit=config.env.pendulum_hard_reset_velocity_limit,
    )
    returns: list[float] = []
    lengths: list[int] = []
    episode_id = 0
    episode_return = 0.0
    episode_length = 0
    steps_added = 0
    obs, _ = env.reset(seed=seed)
    try:
        while steps_added < dataset_steps:
            remaining_steps = max(1, horizon - episode_length)
            action = reference.act(obs, remaining_steps=remaining_steps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            terminal_for_bootstrap = bool(terminated)
            replay.add(
                np.asarray(obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                np.asarray(next_obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                np.asarray(action, dtype=np.float32).reshape(1, *env.action_space.shape),
                np.asarray([float(reward)], dtype=np.float32),
                np.asarray([terminal_for_bootstrap], dtype=bool),
                [info],
                step=steps_added + 1,
                episode_id=episode_id,
            )
            steps_added += 1
            episode_return += float(reward)
            episode_length += 1
            obs = next_obs
            if bool(terminated or truncated):
                returns.append(episode_return)
                lengths.append(episode_length)
                episode_id += 1
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()
    finally:
        env.close()

    if episode_length > 0:
        returns.append(episode_return)
        lengths.append(episode_length)

    returns_array = np.asarray(returns, dtype=np.float64)
    lengths_array = np.asarray(lengths, dtype=np.float64)
    return {
        "reference_prior_dataset_steps": float(steps_added),
        "reference_prior_dataset_size": float(replay.size()),
        "reference_prior_dataset_episodes": float(len(returns)),
        "reference_prior_dataset_mean_return": float(np.mean(returns_array)) if len(returns_array) else 0.0,
        "reference_prior_dataset_min_return": float(np.min(returns_array)) if len(returns_array) else 0.0,
        "reference_prior_dataset_max_return": float(np.max(returns_array)) if len(returns_array) else 0.0,
        "reference_prior_dataset_mean_length": float(np.mean(lengths_array)) if len(lengths_array) else 0.0,
    }


def populate_pendulum_policy_model_replay(
    replay: InstrumentedReplayBuffer,
    agent: SACAgent,
    config: ExperimentConfig,
    rng: np.random.Generator,
    step: int,
    episode_id: int,
) -> dict[str, float]:
    """Add one-step model transitions from synthetic hard Pendulum states under the current actor."""

    count = int(config.sac.pendulum_model_replay_steps_per_step)
    if count <= 0:
        return {"pendulum_model_replay_transitions": 0.0, "pendulum_model_replay_size": float(replay.size())}

    low = float(config.sac.pendulum_model_replay_abs_theta_low)
    high = float(config.sac.pendulum_model_replay_abs_theta_high)
    velocity_limit = float(config.sac.pendulum_model_replay_velocity_limit)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=count)
    abs_theta = rng.uniform(low, high, size=count) if high > low else np.full(count, low, dtype=np.float64)
    theta = signs * abs_theta
    theta_dot = rng.uniform(-velocity_limit, velocity_limit, size=count)
    observations = np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)

    agent.observe(observations)
    params = PendulumDPParams(horizon=int(config.env.max_episode_steps or 200))
    actions, action_log_probs = agent.act_batch_with_log_prob(observations)
    random_action_fraction = float(config.sac.pendulum_model_replay_random_action_fraction)
    random_action_mask = np.zeros(count, dtype=bool)
    if random_action_fraction > 0.0:
        random_action_mask = rng.random(count) < random_action_fraction
        if np.any(random_action_mask):
            random_actions = rng.uniform(
                -params.max_torque,
                params.max_torque,
                size=np.asarray(actions).shape,
            ).astype(np.float32)
            actions = np.asarray(actions, dtype=np.float32).copy()
            actions[random_action_mask] = random_actions[random_action_mask]
            action_log_probs = np.asarray(action_log_probs, dtype=np.float32).copy()
            action_dim = int(np.asarray(actions).reshape(count, -1).shape[1])
            action_log_probs[random_action_mask] = -float(action_dim) * float(np.log(2.0 * params.max_torque))
    rewards, next_theta, next_theta_dot = pendulum_step_model(
        theta,
        theta_dot,
        np.asarray(actions, dtype=np.float64).reshape(count, -1)[:, 0],
        params,
    )
    next_observations = np.stack([np.cos(next_theta), np.sin(next_theta), next_theta_dot], axis=1).astype(np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    action_log_probs = np.asarray(action_log_probs, dtype=np.float32).reshape(-1)

    for idx in range(count):
        replay.add(
            observations[idx].reshape(1, -1),
            next_observations[idx].reshape(1, -1),
            np.asarray(actions[idx], dtype=np.float32).reshape(1, -1),
            rewards[idx : idx + 1],
            np.asarray([False], dtype=bool),
            [{"pendulum_model_replay": True}],
            step=step,
            episode_id=episode_id,
            action_log_prob=float(action_log_probs[idx]),
        )

    return {
        "pendulum_model_replay_transitions": float(count),
        "pendulum_model_replay_size": float(replay.size()),
        "pendulum_model_replay_reward_mean": float(np.mean(rewards)),
        "pendulum_model_replay_abs_theta_mean": float(np.mean(np.abs(theta))),
        "pendulum_model_replay_abs_velocity_mean": float(np.mean(np.abs(theta_dot))),
        "pendulum_model_replay_abs_action_mean": float(np.mean(np.abs(actions))),
        "pendulum_model_replay_random_action_fraction": float(np.mean(random_action_mask)),
    }


def populate_pendulum_policy_model_rollouts(
    replay: InstrumentedReplayBuffer,
    agent: SACAgent,
    config: ExperimentConfig,
    rng: np.random.Generator,
    step: int,
) -> dict[str, float]:
    starts = int(config.sac.pendulum_model_rollout_starts_per_step)
    horizon = int(config.sac.pendulum_model_rollout_horizon)
    if starts <= 0 or horizon <= 0:
        return {
            "pendulum_model_rollout_transitions": 0.0,
            "pendulum_model_rollout_size": float(replay.size()),
        }

    low = float(config.sac.pendulum_model_rollout_abs_theta_low)
    high = float(config.sac.pendulum_model_rollout_abs_theta_high)
    velocity_limit = float(config.sac.pendulum_model_rollout_velocity_limit)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=starts)
    abs_theta = rng.uniform(low, high, size=starts) if high > low else np.full(starts, low, dtype=np.float64)
    theta = signs * abs_theta
    theta_dot = rng.uniform(-velocity_limit, velocity_limit, size=starts)

    params = PendulumDPParams(horizon=int(config.env.max_episode_steps or 200))
    rewards_out: list[np.ndarray] = []
    actions_out: list[np.ndarray] = []
    for rollout_step in range(horizon):
        observations = np.stack([np.cos(theta), np.sin(theta), theta_dot], axis=1).astype(np.float32)
        actions, action_log_probs = agent.act_batch_with_log_prob(observations)
        flat_actions = np.asarray(actions, dtype=np.float64).reshape(starts, -1)[:, 0]
        rewards, next_theta, next_theta_dot = pendulum_step_model(theta, theta_dot, flat_actions, params)
        next_observations = np.stack([np.cos(next_theta), np.sin(next_theta), next_theta_dot], axis=1).astype(np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        action_log_probs = np.asarray(action_log_probs, dtype=np.float32).reshape(-1)

        for rollout_index in range(starts):
            replay.add(
                observations[rollout_index].reshape(1, -1),
                next_observations[rollout_index].reshape(1, -1),
                np.asarray(actions[rollout_index], dtype=np.float32).reshape(1, -1),
                rewards[rollout_index : rollout_index + 1],
                np.asarray([False], dtype=bool),
                [{"pendulum_model_rollout": True}],
                step=step * starts * horizon + rollout_index * horizon + rollout_step,
                episode_id=2_000_000_000 + step * starts + rollout_index,
                action_log_prob=float(action_log_probs[rollout_index]),
            )

        rewards_out.append(rewards)
        actions_out.append(np.asarray(actions, dtype=np.float32).reshape(starts, -1))
        theta = next_theta
        theta_dot = next_theta_dot

    all_rewards = np.concatenate(rewards_out, axis=0) if rewards_out else np.asarray([], dtype=np.float32)
    all_actions = np.concatenate(actions_out, axis=0) if actions_out else np.asarray([], dtype=np.float32)
    final_abs_theta = np.abs(theta)
    return {
        "pendulum_model_rollout_transitions": float(starts * horizon),
        "pendulum_model_rollout_starts": float(starts),
        "pendulum_model_rollout_horizon": float(horizon),
        "pendulum_model_rollout_size": float(replay.size()),
        "pendulum_model_rollout_reward_mean": float(np.mean(all_rewards)) if all_rewards.size else 0.0,
        "pendulum_model_rollout_abs_action_mean": float(np.mean(np.abs(all_actions))) if all_actions.size else 0.0,
        "pendulum_model_rollout_initial_abs_theta_mean": float(np.mean(abs_theta)),
        "pendulum_model_rollout_final_abs_theta_mean": float(np.mean(final_abs_theta)),
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
    parser.add_argument("--pendulum-hard-reset-prob", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-final-prob", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-decay-steps", type=int, default=None)
    parser.add_argument("--pendulum-hard-reset-start-step", type=int, default=None)
    parser.add_argument("--pendulum-hard-reset-abs-theta-low", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-abs-theta-high", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-velocity-limit", type=float, default=None)
    parser.add_argument(
        "--pendulum-failure-reset-prob",
        type=float,
        default=None,
        help="Fraction of later training episodes reset from the automatically discovered worst-state bank.",
    )
    parser.add_argument("--pendulum-failure-curriculum-start-step", type=int, default=None)
    parser.add_argument(
        "--pendulum-failure-curriculum-refresh-interval-steps",
        type=int,
        default=None,
    )
    parser.add_argument("--pendulum-failure-curriculum-candidate-count", type=int, default=None)
    parser.add_argument("--pendulum-failure-curriculum-worst-fraction", type=float, default=None)
    parser.add_argument(
        "--pendulum-failure-curriculum-rollouts-per-candidate",
        type=int,
        default=None,
    )
    parser.add_argument("--pendulum-failure-curriculum-rollout-horizon", type=int, default=None)
    parser.add_argument("--pendulum-failure-curriculum-seed-offset", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument(
        "--random-action-steps",
        type=int,
        default=None,
        help="Random-behavior warmup length; defaults to learning_starts when omitted.",
    )
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--policy-lr", type=float, default=None)
    parser.add_argument("--q-lr", type=float, default=None)
    parser.add_argument("--policy-lr-final", type=float, default=None)
    parser.add_argument("--q-lr-final", type=float, default=None)
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=None,
        help="Optional entropy-temperature optimizer learning rate; defaults to q-lr.",
    )
    parser.add_argument("--alpha-lr-final", type=float, default=None)
    parser.add_argument("--alpha-initial-value", type=float, default=None)
    parser.add_argument(
        "--alpha-min-value",
        type=float,
        default=None,
        help="Clamp SAC's learned entropy temperature to this minimum value; 0 disables the floor.",
    )
    parser.add_argument("--target-entropy-scale", type=float, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument(
        "--policy-frequency",
        type=int,
        default=None,
        help="Run an actor/temperature update trigger every N critic updates.",
    )
    parser.add_argument(
        "--actor-updates-per-trigger",
        type=int,
        default=None,
        help=(
            "Actor/temperature optimizer steps per trigger. 0 preserves the CleanRL default "
            "of repeating --policy-frequency times."
        ),
    )
    parser.add_argument(
        "--actor-q-aggregation",
        choices=["min", "mean", "max"],
        default=None,
        help="Critic aggregation used only for the actor objective; critic targets still use clipped/min Q.",
    )
    parser.add_argument(
        "--actor-q-aggregation-late",
        choices=["min", "mean", "max"],
        default=None,
        help="Optional actor critic aggregation to switch to after --actor-q-aggregation-switch-step.",
    )
    parser.add_argument(
        "--actor-q-aggregation-switch-step",
        type=int,
        default=None,
        help="Environment/update step at which --actor-q-aggregation-late becomes active; 0 disables switching.",
    )
    parser.add_argument(
        "--target-q-aggregation",
        choices=["min", "mean", "max"],
        default=None,
        help="Critic aggregation used for Bellman bootstrap targets.",
    )
    parser.add_argument(
        "--redq-num-critics",
        type=int,
        default=None,
        help="Number of critic networks for REDQ-style ensembling. 2 keeps standard clipped double Q.",
    )
    parser.add_argument(
        "--redq-target-subset-size",
        type=int,
        default=None,
        help="Number of target critics sampled for REDQ clipped target backups.",
    )
    parser.add_argument(
        "--redo-interval-updates",
        type=int,
        default=None,
        help="Run ReDo-style dormant critic neuron recycling every N optimizer updates; 0 disables it.",
    )
    parser.add_argument(
        "--redo-dormant-threshold",
        type=float,
        default=None,
        help="Relative activation threshold for ReDo-style dormant neuron detection.",
    )
    parser.add_argument(
        "--swd-linear-decay-steps",
        type=int,
        default=None,
        help=(
            "Enable SWD age-biased replay sampling. 0 disables it; positive values prefer newer "
            "transitions; negative values prefer older transitions."
        ),
    )
    parser.add_argument(
        "--swd-min-weight",
        type=float,
        default=None,
        help="Minimum SWD sampling weight for de-emphasized transitions.",
    )
    parser.add_argument(
        "--replay-priority-mode",
        choices=["none", "bellman_residual", "critic_disagreement", "max"],
        default=None,
        help=(
            "Automatic coordinate-agnostic replay priority signal. 'max' uses the larger of the "
            "clipped Bellman residual and online-critic disagreement; 'none' preserves uniform replay."
        ),
    )
    parser.add_argument("--replay-priority-alpha", type=float, default=None)
    parser.add_argument("--replay-priority-beta-initial", type=float, default=None)
    parser.add_argument("--replay-priority-beta-final", type=float, default=None)
    parser.add_argument("--replay-priority-beta-anneal-steps", type=int, default=None)
    parser.add_argument(
        "--replay-priority-uniform-fraction",
        type=float,
        default=None,
        help="Minimum uniform component in the sampling mixture; must be at least 0.5 when enabled.",
    )
    parser.add_argument("--replay-priority-epsilon", type=float, default=None)
    parser.add_argument("--replay-priority-clip", type=float, default=None)
    parser.add_argument(
        "--pendulum-hard-replay-fraction",
        type=float,
        default=None,
        help="Fraction of each update batch sampled from Pendulum hard-boundary states; 0 disables it.",
    )
    parser.add_argument(
        "--pendulum-hard-replay-final-fraction",
        type=float,
        default=None,
        help="Final hard-boundary replay fraction after --pendulum-hard-replay-decay-steps.",
    )
    parser.add_argument(
        "--pendulum-hard-replay-decay-steps",
        type=int,
        default=None,
        help="Linearly decay hard-boundary replay fraction after its start step; 0 keeps it constant.",
    )
    parser.add_argument(
        "--pendulum-hard-replay-start-step",
        type=int,
        default=None,
        help="Start sampling Pendulum hard-boundary replay batches at this environment step.",
    )
    parser.add_argument("--pendulum-hard-replay-abs-theta-low", type=float, default=None)
    parser.add_argument("--pendulum-hard-replay-abs-theta-high", type=float, default=None)
    parser.add_argument("--pendulum-hard-replay-velocity-limit", type=float, default=None)
    parser.add_argument(
        "--pendulum-model-replay-ratio",
        type=float,
        default=None,
        help=(
            "Fraction of each one-step SAC update batch sampled from current-policy Pendulum model "
            "transitions seeded at hard states; 0 disables it."
        ),
    )
    parser.add_argument("--pendulum-model-replay-steps-per-step", type=int, default=None)
    parser.add_argument("--pendulum-model-replay-start-step", type=int, default=None)
    parser.add_argument("--pendulum-model-replay-random-action-fraction", type=float, default=None)
    parser.add_argument("--pendulum-model-replay-abs-theta-low", type=float, default=None)
    parser.add_argument("--pendulum-model-replay-abs-theta-high", type=float, default=None)
    parser.add_argument("--pendulum-model-replay-velocity-limit", type=float, default=None)
    parser.add_argument(
        "--pendulum-model-rollout-ratio",
        type=float,
        default=None,
        help="Fraction of each SAC-N update batch sampled from exact-model hard-state rollouts.",
    )
    parser.add_argument("--pendulum-model-rollout-starts-per-step", type=int, default=None)
    parser.add_argument("--pendulum-model-rollout-horizon", type=int, default=None)
    parser.add_argument("--pendulum-model-rollout-interval-steps", type=int, default=None)
    parser.add_argument("--pendulum-model-rollout-start-step", type=int, default=None)
    parser.add_argument("--pendulum-model-rollout-abs-theta-low", type=float, default=None)
    parser.add_argument("--pendulum-model-rollout-abs-theta-high", type=float, default=None)
    parser.add_argument("--pendulum-model-rollout-velocity-limit", type=float, default=None)
    parser.add_argument(
        "--sacn-n-step",
        type=int,
        default=None,
        help="Enable SACn critic targets with this maximum n-step horizon; 1 is standard one-step SAC.",
    )
    parser.add_argument(
        "--sacn-importance-quantile",
        type=float,
        default=None,
        help="Quantile q_b used to clip SACn importance ratios before batch-max normalization.",
    )
    parser.add_argument(
        "--sacn-no-tau-entropy",
        action="store_true",
        help="Use one entropy sample per SACn target instead of the paper's tau-dependent sample count.",
    )
    parser.add_argument(
        "--sacn-max-entropy-samples",
        type=int,
        default=None,
        help="Cap for SACn tau-sampled entropy estimates.",
    )
    parser.add_argument(
        "--sacn-recent-max-age-steps",
        type=int,
        default=None,
        help="Only sample SACn sequences whose start transition is this many environment steps old or newer; 0 disables.",
    )
    parser.add_argument(
        "--sacn-min-horizon-ess-fraction",
        type=float,
        default=None,
        help="Drop SACn horizon columns whose normalized effective sample size is below this fraction; 0 disables.",
    )
    parser.add_argument(
        "--sacn-importance-mode",
        choices=("density", "none"),
        default=None,
        help="SACn horizon weighting: paper density-product weights, or no importance weighting.",
    )
    parser.add_argument(
        "--sacn-non-soft-targets",
        action="store_true",
        help="Use non-soft SACn critic targets by omitting entropy terms from the multi-step Bellman target.",
    )
    parser.add_argument(
        "--sacn-stop-after-steps",
        type=int,
        default=None,
        help="Use SACn sequence updates only through this environment step; 0 keeps SACn active for all training.",
    )
    parser.add_argument(
        "--sacn-target-mode",
        choices=("all", "fast_last"),
        default=None,
        help="SACn critic target set: all horizons, or FastSACN-style 1-step plus last active horizon.",
    )
    parser.add_argument(
        "--sacn-horizon-lambda",
        type=float,
        default=None,
        help="Optional FastSACN-style lambda decay applied across selected SACn horizons.",
    )
    parser.add_argument(
        "--fast-updates",
        action="store_true",
        help="Skip per-optimizer-step parameter/gradient norm telemetry for faster architecture ablation runs.",
    )
    parser.add_argument(
        "--l2-feature-norm",
        action="store_true",
        help=(
            "Use the archived CleanRL MLP variant that L2-normalizes each ReLU hidden feature vector "
            "and rescales it by sqrt(hidden_dim)."
        ),
    )
    parser.add_argument(
        "--simba-backbone",
        action="store_true",
        help="Swap CleanRL MLPs for the SimbaV2 HyperDense/Scaler/LERP scalar-critic backbone.",
    )
    parser.add_argument(
        "--simba-no-feature-norm",
        action="store_true",
        help="Disable SimbaV2 L2 feature normalization inside the Simba backbone for a design-study ablation.",
    )
    parser.add_argument(
        "--simba-no-observation-norm",
        action="store_true",
        help="Disable SimbaV2 running observation normalization before the shifted/L2 input embedding.",
    )
    parser.add_argument(
        "--simba-no-input-shift",
        action="store_true",
        help="Disable SimbaV2's positive shifted input coordinate before L2 normalization.",
    )
    parser.add_argument("--simba-actor-blocks", type=int, default=None)
    parser.add_argument("--simba-actor-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--simba-actor-log-std-floor",
        type=float,
        default=None,
        help=(
            "Clamp the Simba actor's learned log-standard deviation after its unchanged "
            "historical [-10, 2] mapping. This default-off exploration intervention is "
            "valid only with --simba-backbone."
        ),
    )
    parser.add_argument("--simba-critic-blocks", type=int, default=None)
    parser.add_argument("--simba-critic-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--simba-distributional-critic",
        action="store_true",
        help="Use SimbaV2's categorical critic loss instead of scalar MSE Q regression.",
    )
    parser.add_argument(
        "--simba-reward-scaling",
        action="store_true",
        help="Scale rewards with running discounted-return statistics before critic updates.",
    )
    parser.add_argument("--simba-critic-num-bins", type=int, default=None)
    parser.add_argument("--simba-critic-min-v", type=float, default=None)
    parser.add_argument("--simba-critic-max-v", type=float, default=None)
    parser.add_argument(
        "--simba-weight-projection",
        action="store_true",
        help="Enable SimbaV2's HyperDense weight projection after initialization and actor/critic optimizer steps.",
    )
    parser.add_argument(
        "--reference-guidance-mode",
        choices=("none", "replay_injection", "interleaved_execution"),
        default=None,
        help="Use a Pendulum reference policy by adding synthetic replay or by mixing it into executed actions.",
    )
    parser.add_argument(
        "--reference-guidance-policy",
        choices=("controller", "dp", "best"),
        default=None,
        help="Reference source for guidance; best chooses the better finite-horizon DP/controller return estimate.",
    )
    parser.add_argument("--reference-guidance-probability", type=float, default=None)
    parser.add_argument(
        "--reference-guidance-start-step",
        type=int,
        default=None,
        help="First environment step where reference guidance may act or inject replay.",
    )
    parser.add_argument("--reference-guidance-dp-solution", default=None)
    parser.add_argument(
        "--reference-auxiliary-mode",
        choices=("none", "bc", "q_filtered_bc", "q_filtered_replay_bc"),
        default=None,
        help=(
            "Add an actor imitation loss toward a Pendulum reference policy; "
            "q_filtered_bc only clones when critics prefer the reference action, "
            "while q_filtered_replay_bc applies that filter to replay samples and "
            "keeps reference-anchor samples unconditional."
        ),
    )
    parser.add_argument(
        "--reference-auxiliary-policy",
        choices=("controller", "dp", "best"),
        default=None,
        help="Reference source for the auxiliary actor loss. DP is the cheap default for batch updates.",
    )
    parser.add_argument("--reference-auxiliary-weight", type=float, default=None)
    parser.add_argument("--reference-auxiliary-weight-final", type=float, default=None)
    parser.add_argument("--reference-auxiliary-decay-updates", type=int, default=None)
    parser.add_argument(
        "--reference-auxiliary-stop-update",
        type=int,
        default=None,
        help=(
            "Exclusive optimizer-update boundary for reference actor loss: a positive "
            "value disables BC at and after this update; zero leaves BC unbounded."
        ),
    )
    parser.add_argument("--reference-auxiliary-margin", type=float, default=None)
    parser.add_argument("--reference-auxiliary-filter-start-update", type=int, default=None)
    parser.add_argument(
        "--reference-auxiliary-q-filter-mode",
        choices=("twin_min_difference", "online_target_unanimous"),
        default=None,
        help=(
            "Critic evidence used by Q-filtered reference BC. The legacy/default "
            "twin_min_difference compares the two online clipped values; "
            "online_target_unanimous requires every online and target critic to "
            "prefer the reference action by the configured margin."
        ),
    )
    parser.add_argument(
        "--reference-auxiliary-replay-normalization",
        choices=("selected_mean", "full_batch_mean"),
        default=None,
        help=(
            "Normalize masked replay BC by selected replay rows (legacy/default) "
            "or by the full replay batch. Anchor rows remain unconditional."
        ),
    )
    parser.add_argument("--reference-anchor-ratio", type=float, default=None)
    parser.add_argument("--reference-anchor-size", type=int, default=None)
    parser.add_argument("--reference-anchor-velocity-limit", type=float, default=None)
    parser.add_argument("--reference-anchor-reset-support-fraction", type=float, default=None)
    parser.add_argument("--reference-anchor-reset-velocity-limit", type=float, default=None)
    parser.add_argument(
        "--sac-actor-loss-weight",
        type=float,
        default=None,
        help="Weight of the reward SAC term inside a joint actor objective.",
    )
    parser.add_argument(
        "--sac-actor-loss-start-step",
        type=int,
        default=None,
        help="Optimizer update where the reward SAC term starts; BC may update before this.",
    )
    parser.add_argument(
        "--sac-actor-objective-mode",
        choices=("stochastic", "deterministic_mean"),
        default=None,
        help=(
            "Use the ordinary entropy-regularized sampled SAC actor objective, or optimize "
            "the deterministic mean action against Q. The latter aligns a joint BC+RL update "
            "with deterministic deployment and leaves the policy std head without a direct loss."
        ),
    )
    parser.add_argument(
        "--actor-mean-logit-l2-weight",
        type=float,
        default=None,
        help=(
            "Weight on the mean squared deterministic pre-tanh actor mean. "
            "This default-off regularizer supplies an unsquashed restoring gradient "
            "when tanh action gradients are saturated."
        ),
    )
    parser.add_argument(
        "--actor-mean-logit-excess-threshold",
        type=float,
        default=None,
        help=(
            "When positive, apply the actor mean-logit L2 weight only to squared excess "
            "above this absolute pre-tanh threshold. This preserves useful bounded torque "
            "while retaining an unsquashed gradient on extreme saturation."
        ),
    )
    parser.add_argument(
        "--sac-actor-gradient-balance-mode",
        choices=("none", "match_reference"),
        default=None,
        help=(
            "Optionally rescale only the SAC actor term to match the weighted reference-BC "
            "gradient norm."
        ),
    )
    parser.add_argument("--sac-actor-gradient-balance-min-multiplier", type=float, default=None)
    parser.add_argument("--sac-actor-gradient-balance-max-multiplier", type=float, default=None)
    parser.add_argument(
        "--sac-actor-gradient-conflict-mode",
        choices=("none", "project_sac"),
        default=None,
        help=(
            "When SAC and reference-BC gradients conflict, optionally project only the weighted "
            "SAC gradient off the weighted BC gradient before their single joint optimizer step."
        ),
    )
    parser.add_argument(
        "--sac-actor-filter-mode",
        choices=("none", "reference_online_unanimous", "reference_online_target_unanimous"),
        default=None,
        help="Optionally admit SAC actor gradients only when critics prefer the actor to the teacher.",
    )
    parser.add_argument("--sac-actor-filter-margin", type=float, default=None)
    parser.add_argument(
        "--reference-critic-mode",
        choices=("none", "margin"),
        default=None,
        help="Add a critic ranking/calibration loss that prefers a Pendulum reference action over the actor action.",
    )
    parser.add_argument(
        "--reference-critic-policy",
        choices=("controller", "dp", "best"),
        default=None,
        help="Reference source for the critic ranking/calibration loss.",
    )
    parser.add_argument("--reference-critic-weight", type=float, default=None)
    parser.add_argument("--reference-critic-margin", type=float, default=None)
    parser.add_argument(
        "--reference-prior-mode",
        choices=("none", "rlpd"),
        default=None,
        help="Mix model-generated Pendulum reference transitions into ordinary SAC update batches, RLPD-style.",
    )
    parser.add_argument(
        "--reference-prior-policy",
        choices=("controller", "dp", "best"),
        default=None,
        help="Reference source for the RLPD-style prior replay buffer.",
    )
    parser.add_argument("--reference-prior-ratio", type=float, default=None)
    parser.add_argument(
        "--reference-prior-source",
        choices=("online_one_step", "rollout_dataset", "rollout_plus_online"),
        default=None,
        help=(
            "How to populate the RLPD-style prior buffer. online_one_step keeps the old synthetic transition "
            "at visited online states; rollout_dataset pre-fills an offline reference rollout dataset; "
            "rollout_plus_online does both."
        ),
    )
    parser.add_argument("--reference-prior-dataset-steps", type=int, default=None)
    parser.add_argument("--reference-prior-dataset-seed-offset", type=int, default=None)
    parser.add_argument(
        "--actor-init-checkpoint",
        default=None,
        help="Initialize only the actor, and optionally observation normalization, from a SAC-format checkpoint.",
    )
    parser.add_argument(
        "--actor-init-no-obs-rms",
        action="store_true",
        help="When using --actor-init-checkpoint, do not load observation normalization statistics.",
    )
    parser.add_argument(
        "--actor-update-start-step",
        type=int,
        default=None,
        help="Do not update the actor/temperature until this optimizer update step; 0 keeps standard SAC behavior.",
    )
    parser.add_argument(
        "--actor-update-stop-step",
        type=int,
        default=None,
        help="Stop actor/temperature updates after this optimizer update step; 0 keeps them enabled after start.",
    )
    parser.add_argument("--uniform-exploration-initial-probability", type=float, default=None)
    parser.add_argument("--uniform-exploration-final-probability", type=float, default=None)
    parser.add_argument("--uniform-exploration-decay-steps", type=int, default=None)
    parser.add_argument("--uniform-exploration-start-step", type=int, default=None)
    parser.add_argument(
        "--freeze-obs-rms",
        action="store_true",
        help="Keep observation-normalization statistics fixed during training.",
    )
    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=None,
        help="Weight for a CQL-style conservative critic penalty; 0 disables it.",
    )
    parser.add_argument("--cql-temperature", type=float, default=None)
    parser.add_argument("--cql-num-random-actions", type=int, default=None)
    parser.add_argument(
        "--cql-interval-updates",
        type=int,
        default=None,
        help="Apply the CQL critic penalty every N optimizer updates; 1 keeps the original every-update behavior.",
    )
    parser.add_argument(
        "--cql-no-policy-actions",
        action="store_true",
        help="Use only random actions in the conservative critic penalty.",
    )
    parser.add_argument(
        "--critic-search-actor-weight",
        type=float,
        default=None,
        help="Weight for critic-guided global action-search policy improvement; 0 disables it.",
    )
    parser.add_argument("--critic-search-num-actions", type=int, default=None)
    parser.add_argument(
        "--critic-search-margin",
        type=float,
        default=None,
        help="Minimum clipped-double-Q gain required before cloning a searched action.",
    )
    parser.add_argument(
        "--critic-search-start-update",
        type=int,
        default=None,
        help="First optimizer update where critic-guided action search is active.",
    )
    parser.add_argument(
        "--critic-search-filter-mode",
        choices=(
            "clipped_value",
            "unanimous_advantage",
            "online_target_unanimous_advantage",
        ),
        default=None,
        help=(
            "Acceptance rule for critic-search actor targets. The online-target rule lets "
            "either online or lagged target critics veto a pseudo-label."
        ),
    )
    parser.add_argument(
        "--critic-search-actor-loss-type",
        choices=("mse", "log_prob"),
        default=None,
        help=(
            "Distill critic-selected actions with post-tanh action MSE or their policy "
            "log likelihood. Log probability retains a useful gradient for saturated actions."
        ),
    )
    parser.add_argument(
        "--self-imitation-weight",
        type=float,
        default=None,
        help="Weight for critic-filtered replay-action self-imitation; 0 disables it.",
    )
    parser.add_argument(
        "--self-imitation-loss-type",
        choices=("mse", "log_prob"),
        default=None,
        help="Actor target for self-imitation: bounded-action MSE to replay actions or tanh log-prob.",
    )
    parser.add_argument(
        "--self-imitation-start-step",
        type=int,
        default=None,
        help="Do not apply self-imitation until this optimizer update step; 0 enables it immediately.",
    )
    parser.add_argument("--self-imitation-temperature", type=float, default=None)
    parser.add_argument("--self-imitation-margin", type=float, default=None)
    parser.add_argument("--self-imitation-max-weight", type=float, default=None)
    parser.add_argument(
        "--pendulum-potential-shaping-weight",
        type=float,
        default=None,
        help="Weight for potential-based reward shaping from the Pendulum DP/controller value grid; 0 disables it.",
    )
    parser.add_argument(
        "--pendulum-potential-shaping-start-update",
        type=int,
        default=None,
        help="First optimizer update where Pendulum potential shaping is active; 0 enables it from the start.",
    )
    parser.add_argument(
        "--pendulum-potential-shaping-abs-theta-low",
        type=float,
        default=None,
        help="Only apply Pendulum potential shaping from states with abs(theta) at or above this value.",
    )
    parser.add_argument(
        "--pendulum-potential-shaping-abs-theta-high",
        type=float,
        default=None,
        help="Only apply Pendulum potential shaping from states with abs(theta) at or below this value.",
    )
    parser.add_argument(
        "--pendulum-potential-shaping-velocity-limit",
        type=float,
        default=None,
        help="If positive, only apply Pendulum potential shaping when abs(theta_dot) is at or below this value.",
    )
    parser.add_argument(
        "--pendulum-potential-shaping-source",
        choices=("best", "dp_policy", "dp_value", "controller"),
        default=None,
        help="State-value source for Pendulum potential shaping.",
    )
    parser.add_argument("--pendulum-potential-shaping-dp-grid", default=None)
    parser.add_argument("--pendulum-potential-shaping-controller-grid", default=None)
    parser.add_argument(
        "--pendulum-symmetry-augmentation",
        action="store_true",
        help=(
            "Duplicate Pendulum update batches with the exact mirror symmetry "
            "(sin(theta), theta_dot, action) -> (-sin(theta), -theta_dot, -action)."
        ),
    )
    parser.add_argument(
        "--pendulum-actor-symmetry-weight",
        type=float,
        default=None,
        help="Weight for deterministic actor-mean mirror-equivariance consistency.",
    )
    parser.add_argument(
        "--pendulum-critic-symmetry-weight",
        type=float,
        default=None,
        help="Weight for Q(s, a) = Q(mirror(s), -a) consistency on every online critic.",
    )
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed-base", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=None,
        help="Save an additional model checkpoint every N environment steps; 0 disables periodic checkpoints.",
    )
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
    if getattr(args, "pendulum_hard_reset_prob", None) is not None:
        config.env.pendulum_hard_reset_prob = args.pendulum_hard_reset_prob
    if getattr(args, "pendulum_hard_reset_final_prob", None) is not None:
        config.env.pendulum_hard_reset_final_prob = args.pendulum_hard_reset_final_prob
    if getattr(args, "pendulum_hard_reset_decay_steps", None) is not None:
        config.env.pendulum_hard_reset_decay_steps = args.pendulum_hard_reset_decay_steps
    if getattr(args, "pendulum_hard_reset_start_step", None) is not None:
        config.env.pendulum_hard_reset_start_step = args.pendulum_hard_reset_start_step
    if getattr(args, "pendulum_hard_reset_abs_theta_low", None) is not None:
        config.env.pendulum_hard_reset_abs_theta_low = args.pendulum_hard_reset_abs_theta_low
    if getattr(args, "pendulum_hard_reset_abs_theta_high", None) is not None:
        config.env.pendulum_hard_reset_abs_theta_high = args.pendulum_hard_reset_abs_theta_high
    if getattr(args, "pendulum_hard_reset_velocity_limit", None) is not None:
        config.env.pendulum_hard_reset_velocity_limit = args.pendulum_hard_reset_velocity_limit
    if getattr(args, "pendulum_failure_reset_prob", None) is not None:
        config.env.pendulum_failure_reset_prob = args.pendulum_failure_reset_prob
    if getattr(args, "pendulum_failure_curriculum_start_step", None) is not None:
        config.env.pendulum_failure_curriculum_start_step = (
            args.pendulum_failure_curriculum_start_step
        )
    if getattr(args, "pendulum_failure_curriculum_refresh_interval_steps", None) is not None:
        config.env.pendulum_failure_curriculum_refresh_interval_steps = (
            args.pendulum_failure_curriculum_refresh_interval_steps
        )
    if getattr(args, "pendulum_failure_curriculum_candidate_count", None) is not None:
        config.env.pendulum_failure_curriculum_candidate_count = (
            args.pendulum_failure_curriculum_candidate_count
        )
    if getattr(args, "pendulum_failure_curriculum_worst_fraction", None) is not None:
        config.env.pendulum_failure_curriculum_worst_fraction = (
            args.pendulum_failure_curriculum_worst_fraction
        )
    if getattr(args, "pendulum_failure_curriculum_rollouts_per_candidate", None) is not None:
        config.env.pendulum_failure_curriculum_rollouts_per_candidate = (
            args.pendulum_failure_curriculum_rollouts_per_candidate
        )
    if getattr(args, "pendulum_failure_curriculum_rollout_horizon", None) is not None:
        config.env.pendulum_failure_curriculum_rollout_horizon = (
            args.pendulum_failure_curriculum_rollout_horizon
        )
    if getattr(args, "pendulum_failure_curriculum_seed_offset", None) is not None:
        config.env.pendulum_failure_curriculum_seed_offset = (
            args.pendulum_failure_curriculum_seed_offset
        )
    if args.total_steps is not None:
        config.sac.total_steps = args.total_steps
    if getattr(args, "buffer_size", None) is not None:
        config.sac.buffer_size = args.buffer_size
    if getattr(args, "batch_size", None) is not None:
        config.sac.batch_size = args.batch_size
    if args.learning_starts is not None:
        config.sac.learning_starts = args.learning_starts
    if getattr(args, "random_action_steps", None) is not None:
        config.sac.random_action_steps = args.random_action_steps
    if getattr(args, "gamma", None) is not None:
        config.sac.gamma = args.gamma
    if getattr(args, "policy_lr", None) is not None:
        config.sac.policy_lr = args.policy_lr
    if getattr(args, "q_lr", None) is not None:
        config.sac.q_lr = args.q_lr
    if getattr(args, "policy_lr_final", None) is not None:
        config.sac.policy_lr_final = args.policy_lr_final
    if getattr(args, "q_lr_final", None) is not None:
        config.sac.q_lr_final = args.q_lr_final
    if getattr(args, "alpha_lr", None) is not None:
        config.sac.alpha_lr = args.alpha_lr
    if getattr(args, "alpha_lr_final", None) is not None:
        config.sac.alpha_lr_final = args.alpha_lr_final
    if getattr(args, "alpha_initial_value", None) is not None:
        config.sac.alpha_initial_value = args.alpha_initial_value
    if getattr(args, "alpha_min_value", None) is not None:
        config.sac.alpha_min_value = args.alpha_min_value
    if getattr(args, "target_entropy_scale", None) is not None:
        config.sac.target_entropy_scale = args.target_entropy_scale
    if getattr(args, "updates_per_step", None) is not None:
        config.sac.updates_per_step = args.updates_per_step
    if getattr(args, "policy_frequency", None) is not None:
        config.sac.policy_frequency = args.policy_frequency
    if getattr(args, "actor_updates_per_trigger", None) is not None:
        config.sac.actor_updates_per_trigger = args.actor_updates_per_trigger
    if getattr(args, "actor_q_aggregation", None) is not None:
        config.sac.actor_q_aggregation = args.actor_q_aggregation
    if getattr(args, "actor_q_aggregation_late", None) is not None:
        config.sac.actor_q_aggregation_late = args.actor_q_aggregation_late
    if getattr(args, "actor_q_aggregation_switch_step", None) is not None:
        config.sac.actor_q_aggregation_switch_step = args.actor_q_aggregation_switch_step
    if getattr(args, "target_q_aggregation", None) is not None:
        config.sac.target_q_aggregation = args.target_q_aggregation
    if getattr(args, "redq_num_critics", None) is not None:
        config.sac.redq_num_critics = args.redq_num_critics
    if getattr(args, "redq_target_subset_size", None) is not None:
        config.sac.redq_target_subset_size = args.redq_target_subset_size
    if getattr(args, "redo_interval_updates", None) is not None:
        config.sac.redo_interval_updates = args.redo_interval_updates
    if getattr(args, "redo_dormant_threshold", None) is not None:
        config.sac.redo_dormant_threshold = args.redo_dormant_threshold
    if getattr(args, "swd_linear_decay_steps", None) is not None:
        config.sac.swd_linear_decay_steps = args.swd_linear_decay_steps
    if getattr(args, "swd_min_weight", None) is not None:
        config.sac.swd_min_weight = args.swd_min_weight
    if getattr(args, "replay_priority_mode", None) is not None:
        config.sac.replay_priority_mode = args.replay_priority_mode
    if getattr(args, "replay_priority_alpha", None) is not None:
        config.sac.replay_priority_alpha = args.replay_priority_alpha
    if getattr(args, "replay_priority_beta_initial", None) is not None:
        config.sac.replay_priority_beta_initial = args.replay_priority_beta_initial
    if getattr(args, "replay_priority_beta_final", None) is not None:
        config.sac.replay_priority_beta_final = args.replay_priority_beta_final
    if getattr(args, "replay_priority_beta_anneal_steps", None) is not None:
        config.sac.replay_priority_beta_anneal_steps = args.replay_priority_beta_anneal_steps
    if getattr(args, "replay_priority_uniform_fraction", None) is not None:
        config.sac.replay_priority_uniform_fraction = args.replay_priority_uniform_fraction
    if getattr(args, "replay_priority_epsilon", None) is not None:
        config.sac.replay_priority_epsilon = args.replay_priority_epsilon
    if getattr(args, "replay_priority_clip", None) is not None:
        config.sac.replay_priority_clip = args.replay_priority_clip
    if getattr(args, "pendulum_hard_replay_fraction", None) is not None:
        config.sac.pendulum_hard_replay_fraction = args.pendulum_hard_replay_fraction
    if getattr(args, "pendulum_hard_replay_final_fraction", None) is not None:
        config.sac.pendulum_hard_replay_final_fraction = args.pendulum_hard_replay_final_fraction
    if getattr(args, "pendulum_hard_replay_decay_steps", None) is not None:
        config.sac.pendulum_hard_replay_decay_steps = args.pendulum_hard_replay_decay_steps
    if getattr(args, "pendulum_hard_replay_start_step", None) is not None:
        config.sac.pendulum_hard_replay_start_step = args.pendulum_hard_replay_start_step
    if getattr(args, "pendulum_hard_replay_abs_theta_low", None) is not None:
        config.sac.pendulum_hard_replay_abs_theta_low = args.pendulum_hard_replay_abs_theta_low
    if getattr(args, "pendulum_hard_replay_abs_theta_high", None) is not None:
        config.sac.pendulum_hard_replay_abs_theta_high = args.pendulum_hard_replay_abs_theta_high
    if getattr(args, "pendulum_hard_replay_velocity_limit", None) is not None:
        config.sac.pendulum_hard_replay_velocity_limit = args.pendulum_hard_replay_velocity_limit
    if getattr(args, "pendulum_model_replay_ratio", None) is not None:
        config.sac.pendulum_model_replay_ratio = args.pendulum_model_replay_ratio
    if getattr(args, "pendulum_model_replay_steps_per_step", None) is not None:
        config.sac.pendulum_model_replay_steps_per_step = args.pendulum_model_replay_steps_per_step
    if getattr(args, "pendulum_model_replay_start_step", None) is not None:
        config.sac.pendulum_model_replay_start_step = args.pendulum_model_replay_start_step
    if getattr(args, "pendulum_model_replay_random_action_fraction", None) is not None:
        config.sac.pendulum_model_replay_random_action_fraction = args.pendulum_model_replay_random_action_fraction
    if getattr(args, "pendulum_model_replay_abs_theta_low", None) is not None:
        config.sac.pendulum_model_replay_abs_theta_low = args.pendulum_model_replay_abs_theta_low
    if getattr(args, "pendulum_model_replay_abs_theta_high", None) is not None:
        config.sac.pendulum_model_replay_abs_theta_high = args.pendulum_model_replay_abs_theta_high
    if getattr(args, "pendulum_model_replay_velocity_limit", None) is not None:
        config.sac.pendulum_model_replay_velocity_limit = args.pendulum_model_replay_velocity_limit
    if getattr(args, "pendulum_model_rollout_ratio", None) is not None:
        config.sac.pendulum_model_rollout_ratio = args.pendulum_model_rollout_ratio
    if getattr(args, "pendulum_model_rollout_starts_per_step", None) is not None:
        config.sac.pendulum_model_rollout_starts_per_step = args.pendulum_model_rollout_starts_per_step
    if getattr(args, "pendulum_model_rollout_horizon", None) is not None:
        config.sac.pendulum_model_rollout_horizon = args.pendulum_model_rollout_horizon
    if getattr(args, "pendulum_model_rollout_interval_steps", None) is not None:
        config.sac.pendulum_model_rollout_interval_steps = args.pendulum_model_rollout_interval_steps
    if getattr(args, "pendulum_model_rollout_start_step", None) is not None:
        config.sac.pendulum_model_rollout_start_step = args.pendulum_model_rollout_start_step
    if getattr(args, "pendulum_model_rollout_abs_theta_low", None) is not None:
        config.sac.pendulum_model_rollout_abs_theta_low = args.pendulum_model_rollout_abs_theta_low
    if getattr(args, "pendulum_model_rollout_abs_theta_high", None) is not None:
        config.sac.pendulum_model_rollout_abs_theta_high = args.pendulum_model_rollout_abs_theta_high
    if getattr(args, "pendulum_model_rollout_velocity_limit", None) is not None:
        config.sac.pendulum_model_rollout_velocity_limit = args.pendulum_model_rollout_velocity_limit
    if getattr(args, "sacn_n_step", None) is not None:
        config.sac.sacn_n_step = args.sacn_n_step
    if getattr(args, "sacn_importance_quantile", None) is not None:
        config.sac.sacn_importance_quantile = args.sacn_importance_quantile
    if getattr(args, "sacn_no_tau_entropy", False):
        config.sac.sacn_tau_entropy = False
    if getattr(args, "sacn_max_entropy_samples", None) is not None:
        config.sac.sacn_max_entropy_samples = args.sacn_max_entropy_samples
    if getattr(args, "sacn_recent_max_age_steps", None) is not None:
        config.sac.sacn_recent_max_age_steps = args.sacn_recent_max_age_steps
    if getattr(args, "sacn_min_horizon_ess_fraction", None) is not None:
        config.sac.sacn_min_horizon_ess_fraction = args.sacn_min_horizon_ess_fraction
    if getattr(args, "sacn_importance_mode", None) is not None:
        config.sac.sacn_importance_mode = args.sacn_importance_mode
    if getattr(args, "sacn_non_soft_targets", False):
        config.sac.sacn_non_soft_targets = True
    if getattr(args, "sacn_stop_after_steps", None) is not None:
        config.sac.sacn_stop_after_steps = args.sacn_stop_after_steps
    if getattr(args, "sacn_target_mode", None) is not None:
        config.sac.sacn_target_mode = args.sacn_target_mode
    if getattr(args, "sacn_horizon_lambda", None) is not None:
        config.sac.sacn_horizon_lambda = args.sacn_horizon_lambda
    if getattr(args, "fast_updates", False):
        config.sac.update_diagnostics = False
    if getattr(args, "l2_feature_norm", False):
        config.sac.l2_feature_norm = True
    if getattr(args, "simba_backbone", False):
        config.sac.simba_backbone = True
    if getattr(args, "simba_no_feature_norm", False):
        config.sac.simba_feature_norm = False
    if getattr(args, "simba_no_observation_norm", False):
        config.sac.simba_observation_norm = False
    if getattr(args, "simba_no_input_shift", False):
        config.sac.simba_input_shift = False
    if getattr(args, "simba_actor_blocks", None) is not None:
        config.sac.simba_actor_blocks = args.simba_actor_blocks
    if getattr(args, "simba_actor_hidden_dim", None) is not None:
        config.sac.simba_actor_hidden_dim = args.simba_actor_hidden_dim
    if getattr(args, "simba_actor_log_std_floor", None) is not None:
        config.sac.simba_actor_log_std_floor = args.simba_actor_log_std_floor
    if getattr(args, "simba_critic_blocks", None) is not None:
        config.sac.simba_critic_blocks = args.simba_critic_blocks
    if getattr(args, "simba_critic_hidden_dim", None) is not None:
        config.sac.simba_critic_hidden_dim = args.simba_critic_hidden_dim
    if getattr(args, "simba_distributional_critic", False):
        config.sac.simba_distributional_critic = True
    if getattr(args, "simba_reward_scaling", False):
        config.sac.simba_reward_scaling = True
    if getattr(args, "simba_critic_num_bins", None) is not None:
        config.sac.simba_critic_num_bins = args.simba_critic_num_bins
    if getattr(args, "simba_critic_min_v", None) is not None:
        config.sac.simba_critic_min_v = args.simba_critic_min_v
    if getattr(args, "simba_critic_max_v", None) is not None:
        config.sac.simba_critic_max_v = args.simba_critic_max_v
    if getattr(args, "simba_weight_projection", False):
        config.sac.simba_weight_projection = True
    if getattr(args, "reference_guidance_mode", None) is not None:
        config.sac.reference_guidance_mode = args.reference_guidance_mode
    if getattr(args, "reference_guidance_policy", None) is not None:
        config.sac.reference_guidance_policy = args.reference_guidance_policy
    if getattr(args, "reference_guidance_probability", None) is not None:
        config.sac.reference_guidance_probability = args.reference_guidance_probability
    if getattr(args, "reference_guidance_start_step", None) is not None:
        config.sac.reference_guidance_start_step = args.reference_guidance_start_step
    if getattr(args, "reference_guidance_dp_solution", None) is not None:
        config.sac.reference_guidance_dp_solution_path = args.reference_guidance_dp_solution
    if getattr(args, "reference_auxiliary_mode", None) is not None:
        config.sac.reference_auxiliary_mode = args.reference_auxiliary_mode
    if getattr(args, "reference_auxiliary_policy", None) is not None:
        config.sac.reference_auxiliary_policy = args.reference_auxiliary_policy
    if getattr(args, "reference_auxiliary_weight", None) is not None:
        config.sac.reference_auxiliary_weight = args.reference_auxiliary_weight
    if getattr(args, "reference_auxiliary_weight_final", None) is not None:
        config.sac.reference_auxiliary_weight_final = args.reference_auxiliary_weight_final
    if getattr(args, "reference_auxiliary_decay_updates", None) is not None:
        config.sac.reference_auxiliary_decay_updates = args.reference_auxiliary_decay_updates
    if getattr(args, "reference_auxiliary_stop_update", None) is not None:
        config.sac.reference_auxiliary_stop_update = args.reference_auxiliary_stop_update
    if getattr(args, "reference_auxiliary_margin", None) is not None:
        config.sac.reference_auxiliary_margin = args.reference_auxiliary_margin
    if getattr(args, "reference_auxiliary_filter_start_update", None) is not None:
        config.sac.reference_auxiliary_filter_start_update = (
            args.reference_auxiliary_filter_start_update
        )
    if getattr(args, "reference_auxiliary_q_filter_mode", None) is not None:
        config.sac.reference_auxiliary_q_filter_mode = (
            args.reference_auxiliary_q_filter_mode
        )
    if getattr(args, "reference_auxiliary_replay_normalization", None) is not None:
        config.sac.reference_auxiliary_replay_normalization = (
            args.reference_auxiliary_replay_normalization
        )
    if getattr(args, "reference_anchor_ratio", None) is not None:
        config.sac.reference_anchor_ratio = args.reference_anchor_ratio
    if getattr(args, "reference_anchor_size", None) is not None:
        config.sac.reference_anchor_size = args.reference_anchor_size
    if getattr(args, "reference_anchor_velocity_limit", None) is not None:
        config.sac.reference_anchor_velocity_limit = args.reference_anchor_velocity_limit
    if getattr(args, "reference_anchor_reset_support_fraction", None) is not None:
        config.sac.reference_anchor_reset_support_fraction = (
            args.reference_anchor_reset_support_fraction
        )
    if getattr(args, "reference_anchor_reset_velocity_limit", None) is not None:
        config.sac.reference_anchor_reset_velocity_limit = (
            args.reference_anchor_reset_velocity_limit
        )
    if getattr(args, "sac_actor_loss_weight", None) is not None:
        config.sac.sac_actor_loss_weight = args.sac_actor_loss_weight
    if getattr(args, "sac_actor_loss_start_step", None) is not None:
        config.sac.sac_actor_loss_start_step = args.sac_actor_loss_start_step
    if getattr(args, "sac_actor_objective_mode", None) is not None:
        config.sac.sac_actor_objective_mode = args.sac_actor_objective_mode
    if getattr(args, "actor_mean_logit_l2_weight", None) is not None:
        config.sac.actor_mean_logit_l2_weight = args.actor_mean_logit_l2_weight
    if getattr(args, "actor_mean_logit_excess_threshold", None) is not None:
        config.sac.actor_mean_logit_excess_threshold = (
            args.actor_mean_logit_excess_threshold
        )
    if getattr(args, "sac_actor_gradient_balance_mode", None) is not None:
        config.sac.sac_actor_gradient_balance_mode = args.sac_actor_gradient_balance_mode
    if getattr(args, "sac_actor_gradient_balance_min_multiplier", None) is not None:
        config.sac.sac_actor_gradient_balance_min_multiplier = (
            args.sac_actor_gradient_balance_min_multiplier
        )
    if getattr(args, "sac_actor_gradient_balance_max_multiplier", None) is not None:
        config.sac.sac_actor_gradient_balance_max_multiplier = (
            args.sac_actor_gradient_balance_max_multiplier
        )
    if getattr(args, "sac_actor_gradient_conflict_mode", None) is not None:
        config.sac.sac_actor_gradient_conflict_mode = args.sac_actor_gradient_conflict_mode
    if getattr(args, "sac_actor_filter_mode", None) is not None:
        config.sac.sac_actor_filter_mode = args.sac_actor_filter_mode
    if getattr(args, "sac_actor_filter_margin", None) is not None:
        config.sac.sac_actor_filter_margin = args.sac_actor_filter_margin
    if getattr(args, "reference_critic_mode", None) is not None:
        config.sac.reference_critic_mode = args.reference_critic_mode
    if getattr(args, "reference_critic_policy", None) is not None:
        config.sac.reference_critic_policy = args.reference_critic_policy
    if getattr(args, "reference_critic_weight", None) is not None:
        config.sac.reference_critic_weight = args.reference_critic_weight
    if getattr(args, "reference_critic_margin", None) is not None:
        config.sac.reference_critic_margin = args.reference_critic_margin
    if getattr(args, "reference_prior_mode", None) is not None:
        config.sac.reference_prior_mode = args.reference_prior_mode
    if getattr(args, "reference_prior_policy", None) is not None:
        config.sac.reference_prior_policy = args.reference_prior_policy
    if getattr(args, "reference_prior_ratio", None) is not None:
        config.sac.reference_prior_ratio = args.reference_prior_ratio
    if getattr(args, "reference_prior_source", None) is not None:
        config.sac.reference_prior_source = args.reference_prior_source
    if getattr(args, "reference_prior_dataset_steps", None) is not None:
        config.sac.reference_prior_dataset_steps = args.reference_prior_dataset_steps
    if getattr(args, "reference_prior_dataset_seed_offset", None) is not None:
        config.sac.reference_prior_dataset_seed_offset = args.reference_prior_dataset_seed_offset
    if getattr(args, "actor_init_checkpoint", None) is not None:
        config.sac.actor_init_checkpoint_path = args.actor_init_checkpoint
    if getattr(args, "actor_init_no_obs_rms", False):
        config.sac.actor_init_load_obs_rms = False
    if getattr(args, "actor_update_start_step", None) is not None:
        config.sac.actor_update_start_step = args.actor_update_start_step
    if getattr(args, "actor_update_stop_step", None) is not None:
        config.sac.actor_update_stop_step = args.actor_update_stop_step
    if getattr(args, "uniform_exploration_initial_probability", None) is not None:
        config.sac.uniform_exploration_initial_probability = (
            args.uniform_exploration_initial_probability
        )
    if getattr(args, "uniform_exploration_final_probability", None) is not None:
        config.sac.uniform_exploration_final_probability = (
            args.uniform_exploration_final_probability
        )
    if getattr(args, "uniform_exploration_decay_steps", None) is not None:
        config.sac.uniform_exploration_decay_steps = args.uniform_exploration_decay_steps
    if getattr(args, "uniform_exploration_start_step", None) is not None:
        config.sac.uniform_exploration_start_step = args.uniform_exploration_start_step
    if getattr(args, "freeze_obs_rms", False):
        config.sac.obs_rms_update_enabled = False
    if getattr(args, "cql_alpha", None) is not None:
        config.sac.cql_alpha = args.cql_alpha
    if getattr(args, "cql_temperature", None) is not None:
        config.sac.cql_temperature = args.cql_temperature
    if getattr(args, "cql_num_random_actions", None) is not None:
        config.sac.cql_num_random_actions = args.cql_num_random_actions
    if getattr(args, "cql_interval_updates", None) is not None:
        config.sac.cql_interval_updates = args.cql_interval_updates
    if getattr(args, "cql_no_policy_actions", False):
        config.sac.cql_include_policy_actions = False
    if getattr(args, "critic_search_actor_weight", None) is not None:
        config.sac.critic_search_actor_weight = args.critic_search_actor_weight
    if getattr(args, "critic_search_num_actions", None) is not None:
        config.sac.critic_search_num_actions = args.critic_search_num_actions
    if getattr(args, "critic_search_margin", None) is not None:
        config.sac.critic_search_margin = args.critic_search_margin
    if getattr(args, "critic_search_start_update", None) is not None:
        config.sac.critic_search_start_update = args.critic_search_start_update
    if getattr(args, "critic_search_filter_mode", None) is not None:
        config.sac.critic_search_filter_mode = args.critic_search_filter_mode
    if getattr(args, "critic_search_actor_loss_type", None) is not None:
        config.sac.critic_search_actor_loss_type = args.critic_search_actor_loss_type
    if getattr(args, "self_imitation_weight", None) is not None:
        config.sac.self_imitation_weight = args.self_imitation_weight
    if getattr(args, "self_imitation_loss_type", None) is not None:
        config.sac.self_imitation_loss_type = args.self_imitation_loss_type
    if getattr(args, "self_imitation_start_step", None) is not None:
        config.sac.self_imitation_start_step = args.self_imitation_start_step
    if getattr(args, "self_imitation_temperature", None) is not None:
        config.sac.self_imitation_temperature = args.self_imitation_temperature
    if getattr(args, "self_imitation_margin", None) is not None:
        config.sac.self_imitation_margin = args.self_imitation_margin
    if getattr(args, "self_imitation_max_weight", None) is not None:
        config.sac.self_imitation_max_weight = args.self_imitation_max_weight
    if getattr(args, "pendulum_potential_shaping_weight", None) is not None:
        config.sac.pendulum_potential_shaping_weight = args.pendulum_potential_shaping_weight
    if getattr(args, "pendulum_potential_shaping_start_update", None) is not None:
        config.sac.pendulum_potential_shaping_start_update = args.pendulum_potential_shaping_start_update
    if getattr(args, "pendulum_potential_shaping_abs_theta_low", None) is not None:
        config.sac.pendulum_potential_shaping_abs_theta_low = args.pendulum_potential_shaping_abs_theta_low
    if getattr(args, "pendulum_potential_shaping_abs_theta_high", None) is not None:
        config.sac.pendulum_potential_shaping_abs_theta_high = args.pendulum_potential_shaping_abs_theta_high
    if getattr(args, "pendulum_potential_shaping_velocity_limit", None) is not None:
        config.sac.pendulum_potential_shaping_velocity_limit = args.pendulum_potential_shaping_velocity_limit
    if getattr(args, "pendulum_potential_shaping_source", None) is not None:
        config.sac.pendulum_potential_shaping_source = args.pendulum_potential_shaping_source
    if getattr(args, "pendulum_potential_shaping_dp_grid", None) is not None:
        config.sac.pendulum_potential_shaping_dp_grid_path = args.pendulum_potential_shaping_dp_grid
    if getattr(args, "pendulum_potential_shaping_controller_grid", None) is not None:
        config.sac.pendulum_potential_shaping_controller_grid_path = args.pendulum_potential_shaping_controller_grid
    if getattr(args, "pendulum_symmetry_augmentation", False):
        config.sac.pendulum_symmetry_augmentation = True
    if getattr(args, "pendulum_actor_symmetry_weight", None) is not None:
        config.sac.pendulum_actor_symmetry_weight = args.pendulum_actor_symmetry_weight
    if getattr(args, "pendulum_critic_symmetry_weight", None) is not None:
        config.sac.pendulum_critic_symmetry_weight = args.pendulum_critic_symmetry_weight
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
    if getattr(args, "checkpoint_interval_steps", None) is not None:
        config.telemetry.checkpoint_interval_steps = args.checkpoint_interval_steps
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
    failure_curriculum = None
    try:
        env = make_env(
            config.env.env_id,
            seed=config.seed,
            max_episode_steps=config.env.max_episode_steps,
            pendulum_hard_reset_prob=pendulum_hard_reset_probability_at_step(config, 0),
            pendulum_hard_reset_enabled=pendulum_hard_reset_schedule_enabled(config),
            pendulum_hard_reset_abs_theta_low=config.env.pendulum_hard_reset_abs_theta_low,
            pendulum_hard_reset_abs_theta_high=config.env.pendulum_hard_reset_abs_theta_high,
            pendulum_hard_reset_velocity_limit=config.env.pendulum_hard_reset_velocity_limit,
            pendulum_failure_reset_prob=config.env.pendulum_failure_reset_prob,
            pendulum_failure_reset_enabled=pendulum_failure_curriculum_enabled(config),
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, config.sac, device=device)
        if pendulum_failure_curriculum_enabled(config):
            failure_curriculum = PendulumFailureStartCurriculum(
                env_id=config.env.env_id,
                max_episode_steps=config.env.max_episode_steps,
                seed=int(config.seed + config.env.pendulum_failure_curriculum_seed_offset),
                start_step=config.env.pendulum_failure_curriculum_start_step,
                refresh_interval_steps=(
                    config.env.pendulum_failure_curriculum_refresh_interval_steps
                ),
                candidate_count=config.env.pendulum_failure_curriculum_candidate_count,
                worst_fraction=config.env.pendulum_failure_curriculum_worst_fraction,
                rollouts_per_candidate=(
                    config.env.pendulum_failure_curriculum_rollouts_per_candidate
                ),
                rollout_horizon=config.env.pendulum_failure_curriculum_rollout_horizon,
            )
        actor_init_metadata = None
        if config.sac.actor_init_checkpoint_path:
            actor_init_path = Path(config.sac.actor_init_checkpoint_path)
            actor_init_payload = agent.load_actor_checkpoint(
                actor_init_path,
                load_obs_rms=bool(config.sac.actor_init_load_obs_rms),
            )
            actor_init_metadata = {
                "path": str(actor_init_path),
                "source": actor_init_payload.get("source"),
                "loaded_obs_rms": bool(config.sac.actor_init_load_obs_rms and actor_init_payload.get("obs_rms") is not None),
            }
        replay = InstrumentedReplayBuffer(
            config.sac.buffer_size,
            env.observation_space,
            env.action_space,
            device,
            n_envs=1,
            handle_timeout_termination=False,
            swd_linear_decay_steps=config.sac.swd_linear_decay_steps,
            swd_min_weight=config.sac.swd_min_weight,
            priority_mode=config.sac.replay_priority_mode,
            priority_alpha=config.sac.replay_priority_alpha,
            priority_beta_initial=config.sac.replay_priority_beta_initial,
            priority_beta_final=config.sac.replay_priority_beta_final,
            priority_beta_anneal_steps=config.sac.replay_priority_beta_anneal_steps,
            priority_uniform_fraction=config.sac.replay_priority_uniform_fraction,
            priority_epsilon=config.sac.replay_priority_epsilon,
            priority_clip=config.sac.replay_priority_clip,
        )
        pendulum_model_replay = None
        pendulum_model_rng = None
        if config.sac.pendulum_model_replay_ratio > 0.0:
            pendulum_model_replay = InstrumentedReplayBuffer(
                config.sac.buffer_size,
                env.observation_space,
                env.action_space,
                device,
                n_envs=1,
                handle_timeout_termination=False,
            )
            pendulum_model_rng = np.random.default_rng(int(config.seed) + 2_000_000)
        pendulum_model_rollout_replay = None
        pendulum_model_rollout_rng = None
        if config.sac.pendulum_model_rollout_ratio > 0.0:
            pendulum_model_rollout_replay = InstrumentedReplayBuffer(
                config.sac.buffer_size,
                env.observation_space,
                env.action_space,
                device,
                n_envs=1,
                handle_timeout_termination=False,
            )
            pendulum_model_rollout_rng = np.random.default_rng(int(config.seed) + 3_000_000)
        reference_prior = None
        reference_prior_replay = None
        reference_prior_dataset_metrics = None
        if config.sac.reference_prior_mode != "none" and config.sac.reference_prior_ratio > 0.0:
            reference_prior = PendulumReferenceGuidance(
                policy=config.sac.reference_prior_policy,
                dp_solution_path=config.sac.reference_guidance_dp_solution_path,
                horizon=int(config.env.max_episode_steps or 200),
            )
            reference_prior_replay = InstrumentedReplayBuffer(
                config.sac.buffer_size,
                env.observation_space,
                env.action_space,
                device,
                n_envs=1,
                handle_timeout_termination=False,
            )
            if config.sac.reference_prior_source in {"rollout_dataset", "rollout_plus_online"}:
                reference_prior_dataset_metrics = populate_reference_prior_rollout_dataset(
                    reference_prior_replay,
                    reference_prior,
                    config,
                )
        detector = UprightDetector(
            config.env.env_id,
            cos_threshold=config.reliability.near_upright_cos_threshold,
            abs_velocity_threshold=config.reliability.near_upright_abs_velocity_threshold,
        )
        reference_guidance = None
        if config.sac.reference_guidance_mode != "none" and config.sac.reference_guidance_probability > 0.0:
            reference_guidance = PendulumReferenceGuidance(
                policy=config.sac.reference_guidance_policy,
                dp_solution_path=config.sac.reference_guidance_dp_solution_path,
                horizon=int(config.env.max_episode_steps or 200),
            )
        reference_auxiliary = None
        reference_anchor_observations = None
        reference_anchor_actions = None
        reference_anchor_rng = None
        if needs_actor_reference_actions(config.sac):
            reference_auxiliary = PendulumReferenceGuidance(
                policy=config.sac.reference_auxiliary_policy,
                dp_solution_path=config.sac.reference_guidance_dp_solution_path,
                horizon=int(config.env.max_episode_steps or 200),
            )
            if (
                reference_auxiliary_loss_can_be_active(config.sac)
                and config.sac.reference_anchor_ratio > 0.0
            ):
                reference_anchor_rng = np.random.default_rng(int(config.seed) + 4_000_000)
                reference_anchor_observations, reference_anchor_actions = reference_dataset(
                    reference_auxiliary,
                    size=int(config.sac.reference_anchor_size),
                    rng=reference_anchor_rng,
                    velocity_limit=float(config.sac.reference_anchor_velocity_limit),
                    reset_support_fraction=float(
                        config.sac.reference_anchor_reset_support_fraction
                    ),
                    reset_support_velocity_limit=float(
                        config.sac.reference_anchor_reset_velocity_limit
                    ),
                )
        reference_critic = None
        if config.sac.reference_critic_mode != "none" and config.sac.reference_critic_weight > 0.0:
            reference_critic = PendulumReferenceGuidance(
                policy=config.sac.reference_critic_policy,
                dp_solution_path=config.sac.reference_guidance_dp_solution_path,
                horizon=int(config.env.max_episode_steps or 200),
            )
        eval_seeds = fixed_eval_seeds(config.eval.seed_base, config.eval.episodes, config.eval.seeds)
        planned_failure_refresh_steps = (
            failure_curriculum.planned_refresh_steps(config.sac.total_steps)
            if failure_curriculum is not None
            else ()
        )
        planned_failure_discovery_upper_bound = int(
            failure_curriculum.planned_environment_steps_upper_bound(config.sac.total_steps)
            if failure_curriculum is not None
            else 0
        )

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
                "actor_q_aggregation": config.sac.actor_q_aggregation,
                "sac_actor_objective_mode": config.sac.sac_actor_objective_mode,
                "actor_mean_logit_l2_weight": float(
                    config.sac.actor_mean_logit_l2_weight
                ),
                "actor_mean_logit_excess_threshold": float(
                    config.sac.actor_mean_logit_excess_threshold
                ),
                "simba_actor_log_std_floor": config.sac.simba_actor_log_std_floor,
                "actor_q_aggregation_late": config.sac.actor_q_aggregation_late,
                "actor_q_aggregation_switch_step": int(config.sac.actor_q_aggregation_switch_step),
                "target_q_aggregation": config.sac.target_q_aggregation,
                "replay_priority_mode": config.sac.replay_priority_mode,
                "replay_priority_alpha": float(config.sac.replay_priority_alpha),
                "replay_priority_beta_initial": float(config.sac.replay_priority_beta_initial),
                "replay_priority_beta_final": float(config.sac.replay_priority_beta_final),
                "replay_priority_beta_anneal_steps": int(config.sac.replay_priority_beta_anneal_steps),
                "replay_priority_uniform_fraction": float(config.sac.replay_priority_uniform_fraction),
                "replay_priority_epsilon": float(config.sac.replay_priority_epsilon),
                "replay_priority_clip": float(config.sac.replay_priority_clip),
                "reference_guidance": reference_guidance.metadata() if reference_guidance is not None else None,
                "reference_guidance_start_step": int(config.sac.reference_guidance_start_step),
                "reference_auxiliary": reference_auxiliary.metadata() if reference_auxiliary is not None else None,
                "reference_critic": reference_critic.metadata() if reference_critic is not None else None,
                "reference_prior": reference_prior.metadata() if reference_prior is not None else None,
                "reference_prior_ratio": float(config.sac.reference_prior_ratio),
                "reference_prior_source": config.sac.reference_prior_source,
                "reference_prior_dataset_steps": int(config.sac.reference_prior_dataset_steps),
                "reference_prior_initial_size": (
                    float(reference_prior_replay.size()) if reference_prior_replay is not None else 0.0
                ),
                "reference_prior_dataset": reference_prior_dataset_metrics,
                "critic_search_actor_loss_type": config.sac.critic_search_actor_loss_type,
                "self_imitation_weight": float(config.sac.self_imitation_weight),
                "self_imitation_loss_type": config.sac.self_imitation_loss_type,
                "self_imitation_start_step": int(config.sac.self_imitation_start_step),
                "self_imitation_temperature": float(config.sac.self_imitation_temperature),
                "self_imitation_margin": float(config.sac.self_imitation_margin),
                "self_imitation_max_weight": float(config.sac.self_imitation_max_weight),
                "pendulum_hard_reset_initial_prob": float(config.env.pendulum_hard_reset_prob),
                "pendulum_hard_reset_final_prob": float(config.env.pendulum_hard_reset_final_prob),
                "pendulum_hard_reset_decay_steps": int(config.env.pendulum_hard_reset_decay_steps),
                "pendulum_hard_reset_start_step": int(config.env.pendulum_hard_reset_start_step),
                "pendulum_failure_curriculum_enabled": bool(failure_curriculum is not None),
                "pendulum_failure_reset_probability": float(
                    config.env.pendulum_failure_reset_prob
                ),
                "pendulum_failure_curriculum_start_step": int(
                    config.env.pendulum_failure_curriculum_start_step
                ),
                "pendulum_failure_curriculum_refresh_interval_steps": int(
                    config.env.pendulum_failure_curriculum_refresh_interval_steps
                ),
                "pendulum_failure_curriculum_candidate_count": int(
                    config.env.pendulum_failure_curriculum_candidate_count
                ),
                "pendulum_failure_curriculum_worst_fraction": float(
                    config.env.pendulum_failure_curriculum_worst_fraction
                ),
                "pendulum_failure_curriculum_rollouts_per_candidate": int(
                    config.env.pendulum_failure_curriculum_rollouts_per_candidate
                ),
                "pendulum_failure_curriculum_rollout_horizon": int(
                    config.env.pendulum_failure_curriculum_rollout_horizon
                ),
                "pendulum_failure_curriculum_seed_offset": int(
                    config.env.pendulum_failure_curriculum_seed_offset
                ),
                "learning_environment_step_budget": int(config.sac.total_steps),
                "failure_discovery_environment_steps_initial": 0,
                "failure_curriculum_planned_refresh_steps": list(
                    planned_failure_refresh_steps
                ),
                "failure_curriculum_planned_refresh_count": len(
                    planned_failure_refresh_steps
                ),
                "failure_discovery_environment_steps_planned_upper_bound": int(
                    planned_failure_discovery_upper_bound
                ),
                "learning_plus_failure_discovery_steps_planned_upper_bound": int(
                    config.sac.total_steps + planned_failure_discovery_upper_bound
                ),
                "pendulum_hard_replay_fraction": float(config.sac.pendulum_hard_replay_fraction),
                "pendulum_hard_replay_final_fraction": float(config.sac.pendulum_hard_replay_final_fraction),
                "pendulum_hard_replay_decay_steps": int(config.sac.pendulum_hard_replay_decay_steps),
                "pendulum_hard_replay_start_step": int(config.sac.pendulum_hard_replay_start_step),
                "pendulum_model_replay_ratio": float(config.sac.pendulum_model_replay_ratio),
                "pendulum_model_replay_steps_per_step": int(config.sac.pendulum_model_replay_steps_per_step),
                "pendulum_model_replay_start_step": int(config.sac.pendulum_model_replay_start_step),
                "pendulum_model_replay_random_action_fraction": float(
                    config.sac.pendulum_model_replay_random_action_fraction
                ),
                "pendulum_model_replay_abs_theta_low": float(config.sac.pendulum_model_replay_abs_theta_low),
                "pendulum_model_replay_abs_theta_high": float(config.sac.pendulum_model_replay_abs_theta_high),
                "pendulum_model_replay_velocity_limit": float(config.sac.pendulum_model_replay_velocity_limit),
                "pendulum_model_rollout_ratio": float(config.sac.pendulum_model_rollout_ratio),
                "pendulum_model_rollout_starts_per_step": int(config.sac.pendulum_model_rollout_starts_per_step),
                "pendulum_model_rollout_horizon": int(config.sac.pendulum_model_rollout_horizon),
                "pendulum_model_rollout_interval_steps": int(config.sac.pendulum_model_rollout_interval_steps),
                "pendulum_model_rollout_start_step": int(config.sac.pendulum_model_rollout_start_step),
                "pendulum_model_rollout_abs_theta_low": float(config.sac.pendulum_model_rollout_abs_theta_low),
                "pendulum_model_rollout_abs_theta_high": float(config.sac.pendulum_model_rollout_abs_theta_high),
                "pendulum_model_rollout_velocity_limit": float(config.sac.pendulum_model_rollout_velocity_limit),
                "pendulum_potential_shaping_weight": float(config.sac.pendulum_potential_shaping_weight),
                "pendulum_potential_shaping_start_update": int(
                    config.sac.pendulum_potential_shaping_start_update
                ),
                "pendulum_potential_shaping_abs_theta_low": float(
                    config.sac.pendulum_potential_shaping_abs_theta_low
                ),
                "pendulum_potential_shaping_abs_theta_high": float(
                    config.sac.pendulum_potential_shaping_abs_theta_high
                ),
                "pendulum_potential_shaping_velocity_limit": float(
                    config.sac.pendulum_potential_shaping_velocity_limit
                ),
                "pendulum_potential_shaping_source": config.sac.pendulum_potential_shaping_source,
                "pendulum_potential_shaping_dp_grid_path": config.sac.pendulum_potential_shaping_dp_grid_path,
                "pendulum_potential_shaping_controller_grid_path": (
                    config.sac.pendulum_potential_shaping_controller_grid_path
                ),
                "pendulum_symmetry_augmentation": bool(config.sac.pendulum_symmetry_augmentation),
                "pendulum_actor_symmetry_weight": float(
                    config.sac.pendulum_actor_symmetry_weight
                ),
                "pendulum_critic_symmetry_weight": float(
                    config.sac.pendulum_critic_symmetry_weight
                ),
                "actor_init": actor_init_metadata,
            },
        )
        if reference_prior_dataset_metrics is not None:
            logger.log_event("reference_prior_dataset", 0, reference_prior_dataset_metrics)
            logger.log_metrics(0, "reference_guidance", reference_prior_dataset_metrics)
        if reference_anchor_observations is not None:
            logger.log_event(
                "reference_anchor_dataset",
                0,
                {
                    "size": int(len(reference_anchor_observations)),
                    "ratio": float(config.sac.reference_anchor_ratio),
                    "reset_support_fraction": float(
                        config.sac.reference_anchor_reset_support_fraction
                    ),
                },
            )

        obs, _ = env.reset(seed=config.seed)
        random_action_log_prob = uniform_box_action_log_prob(env.action_space)
        episode_id = 0
        episode_return = 0.0
        episode_length = 0
        update_step = 0
        update_metrics_window: list[dict[str, float]] = []
        guidance_metrics_window: list[dict[str, float]] = []
        # ``eval.every_steps == 0`` is the explicit no-online-evaluation
        # contract used by budget-audited workflows.  In particular, do not
        # silently spend environment transitions on the initial or terminal
        # evaluations when periodic evaluation is disabled.
        last_eval_step: int | None = None
        if config.eval.every_steps > 0:
            last_eval_step = log_evaluation(logger, agent, config, eval_seeds, step=0)
        last_diagnostics_step: int | None = None
        random_action_steps = (
            int(config.sac.learning_starts)
            if config.sac.random_action_steps is None
            else int(config.sac.random_action_steps)
        )

        for global_step in range(1, config.sac.total_steps + 1):
            agent.observe(obs)
            remaining_steps = int(config.env.max_episode_steps or 200) - episode_length
            replay_reference_action = (
                reference_auxiliary.act(obs, remaining_steps=remaining_steps)
                if reference_auxiliary is not None
                else None
            )
            replay_reference_critic_action = (
                reference_critic.act(obs, remaining_steps=remaining_steps)
                if reference_critic is not None
                else None
            )
            guidance_draw = random.random()
            uniform_exploration_probability = uniform_exploration_probability_at_step(
                config, global_step
            )
            uniform_exploration = (
                global_step > random_action_steps
                and random.random() < uniform_exploration_probability
            )
            reference_action = None
            reference_guidance_active = (
                reference_guidance is not None and global_step >= config.sac.reference_guidance_start_step
            )
            if reference_guidance_active and guidance_draw < config.sac.reference_guidance_probability:
                reference_action = reference_guidance.act(obs, remaining_steps=remaining_steps)
            if global_step <= random_action_steps or uniform_exploration:
                action = env.action_space.sample()
                behavior_action_log_prob: float | None = random_action_log_prob
            else:
                action, behavior_action_log_prob = agent.act_with_log_prob(obs)
            guidance_metrics_window.append(
                {
                    "uniform_exploration_probability": float(uniform_exploration_probability),
                    "uniform_exploration_action": float(uniform_exploration),
                    "random_warmup_action": float(global_step <= random_action_steps),
                }
            )
            if reference_action is not None and config.sac.reference_guidance_mode == "interleaved_execution":
                action = reference_action
                behavior_action_log_prob = None
                guidance_metrics_window.append({"interleaved_reference_actions": 1.0})
            elif reference_guidance is not None:
                guidance_metrics_window.append({"interleaved_reference_actions": 0.0})

            next_obs, reward, terminated, truncated, info = env.step(action)
            terminal_for_bootstrap = bool(terminated)
            episode_done = bool(terminated or truncated)
            agent.observe_reward(float(reward), episode_done)
            replay.add(
                np.asarray(obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                np.asarray(next_obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                np.asarray(action, dtype=np.float32).reshape(1, *env.action_space.shape),
                np.asarray([float(reward)], dtype=np.float32),
                np.asarray([terminal_for_bootstrap], dtype=bool),
                [info],
                step=global_step,
                episode_id=episode_id,
                action_log_prob=behavior_action_log_prob,
                reference_action=replay_reference_action,
                reference_critic_action=replay_reference_critic_action,
            )
            if reference_action is not None and config.sac.reference_guidance_mode == "replay_injection":
                reference_next_obs, reference_reward = reference_guidance.model_transition(obs, reference_action)
                replay.add(
                    np.asarray(obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                    np.asarray(reference_next_obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                    np.asarray(reference_action, dtype=np.float32).reshape(1, *env.action_space.shape),
                    np.asarray([float(reference_reward)], dtype=np.float32),
                    np.asarray([False], dtype=bool),
                    [{"reference_guidance": True}],
                    step=global_step,
                    episode_id=episode_id,
                    reference_action=replay_reference_action,
                    reference_critic_action=replay_reference_critic_action,
                )
                guidance_metrics_window.append({"injected_reference_transitions": 1.0})
            elif reference_guidance is not None:
                guidance_metrics_window.append({"injected_reference_transitions": 0.0})

            if (
                reference_prior is not None
                and reference_prior_replay is not None
                and config.sac.reference_prior_source in {"online_one_step", "rollout_plus_online"}
            ):
                prior_action = reference_prior.act(obs, remaining_steps=remaining_steps)
                prior_next_obs, prior_reward = reference_prior.model_transition(obs, prior_action)
                reference_prior_replay.add(
                    np.asarray(obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                    np.asarray(prior_next_obs, dtype=np.float32).reshape(1, *env.observation_space.shape),
                    np.asarray(prior_action, dtype=np.float32).reshape(1, *env.action_space.shape),
                    np.asarray([float(prior_reward)], dtype=np.float32),
                    np.asarray([False], dtype=bool),
                    [{"reference_prior": True}],
                    step=global_step,
                    episode_id=episode_id,
                    reference_action=replay_reference_action,
                    reference_critic_action=replay_reference_critic_action,
                )
                guidance_metrics_window.append(
                    {
                        "reference_prior_transitions": 1.0,
                        "reference_prior_size": float(reference_prior_replay.size()),
                    }
                )
            if (
                pendulum_model_replay is not None
                and pendulum_model_rng is not None
                and global_step > config.sac.learning_starts
                and global_step >= config.sac.pendulum_model_replay_start_step
            ):
                guidance_metrics_window.append(
                    populate_pendulum_policy_model_replay(
                        pendulum_model_replay,
                        agent,
                        config,
                        pendulum_model_rng,
                        step=global_step,
                        episode_id=episode_id,
                    )
                )
            if (
                pendulum_model_rollout_replay is not None
                and pendulum_model_rollout_rng is not None
                and global_step > config.sac.learning_starts
                and global_step >= config.sac.pendulum_model_rollout_start_step
                and global_step % config.sac.pendulum_model_rollout_interval_steps == 0
            ):
                guidance_metrics_window.append(
                    populate_pendulum_policy_model_rollouts(
                        pendulum_model_rollout_replay,
                        agent,
                        config,
                        pendulum_model_rollout_rng,
                        step=global_step,
                    )
                )

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if episode_done:
                episode_metrics = {
                    "return": episode_return,
                    "length": float(episode_length),
                    "collapse": float(episode_return <= config.reliability.collapse_return_threshold),
                }
                logger.log_event("episode", global_step, {"episode_id": episode_id, **episode_metrics})
                logger.log_metrics(global_step, "train_episode", episode_metrics)
                episode_id += 1
                episode_return = 0.0
                episode_length = 0
                if (
                    config.env.pendulum_hard_reset_decay_steps > 0
                    or config.env.pendulum_hard_reset_start_step > 0
                ):
                    hard_reset_prob = pendulum_hard_reset_probability_at_step(config, global_step)
                    if set_pendulum_hard_reset_probability(env, hard_reset_prob):
                        guidance_metrics_window.append({"pendulum_hard_reset_probability": hard_reset_prob})
                obs, reset_info = env.reset()
                if failure_curriculum is not None:
                    guidance_metrics_window.append(
                        {
                            "failure_curriculum_reset": float(
                                bool(reset_info.get("failure_curriculum_reset", False))
                            ),
                            "failure_curriculum_bank_size": float(
                                reset_info.get("failure_curriculum_bank_size", 0)
                            ),
                        }
                    )

            if global_step > config.sac.learning_starts and replay.size() >= config.sac.batch_size:
                for _ in range(config.sac.updates_per_step):
                    sacn_active = config.sac.sacn_n_step > 1 and (
                        config.sac.sacn_stop_after_steps <= 0
                        or global_step <= config.sac.sacn_stop_after_steps
                    )
                    hard_replay_fraction = pendulum_hard_replay_fraction_at_step(config, global_step)
                    hard_replay_active = hard_replay_fraction > 0.0
                    if sacn_active:
                        require_action_log_probs = config.sac.sacn_importance_mode == "density"
                        prior_sacn_batch_size = 0
                        if (
                            not require_action_log_probs
                            and reference_prior_replay is not None
                            and config.sac.reference_prior_ratio > 0.0
                            and config.sac.reference_prior_source == "rollout_dataset"
                            and reference_prior_replay.size() >= config.sac.sacn_n_step
                        ):
                            desired_prior = int(round(config.sac.batch_size * config.sac.reference_prior_ratio))
                            prior_sacn_batch_size = min(
                                desired_prior,
                                max(config.sac.batch_size - 1, 0),
                                reference_prior_replay.size(),
                            )
                        model_rollout_batch_size = 0
                        if (
                            pendulum_model_rollout_replay is not None
                            and config.sac.pendulum_model_rollout_ratio > 0.0
                            and pendulum_model_rollout_replay.size() >= config.sac.sacn_n_step
                        ):
                            desired_model_rollout = int(
                                round(config.sac.batch_size * config.sac.pendulum_model_rollout_ratio)
                            )
                            model_rollout_batch_size = min(
                                desired_model_rollout,
                                max(config.sac.batch_size - prior_sacn_batch_size - 1, 0),
                            )
                        online_sacn_batch_size = config.sac.batch_size - prior_sacn_batch_size - model_rollout_batch_size
                        if hard_replay_active:
                            batch = replay.sample_sacn_pendulum_hard_states(
                                online_sacn_batch_size,
                                n_step=config.sac.sacn_n_step,
                                fraction=hard_replay_fraction,
                                abs_theta_low=config.sac.pendulum_hard_replay_abs_theta_low,
                                abs_theta_high=config.sac.pendulum_hard_replay_abs_theta_high,
                                velocity_limit=config.sac.pendulum_hard_replay_velocity_limit,
                                max_age_steps=config.sac.sacn_recent_max_age_steps,
                                require_action_log_probs=require_action_log_probs,
                            )
                        else:
                            batch = replay.sample_sacn(
                                online_sacn_batch_size,
                                n_step=config.sac.sacn_n_step,
                                max_age_steps=config.sac.sacn_recent_max_age_steps,
                                require_action_log_probs=require_action_log_probs,
                            )
                        if prior_sacn_batch_size > 0 and reference_prior_replay is not None:
                            prior_sacn_batch = reference_prior_replay.sample_sacn(
                                prior_sacn_batch_size,
                                n_step=config.sac.sacn_n_step,
                                require_action_log_probs=require_action_log_probs,
                            )
                            batch = concatenate_sacn_replay_samples(batch, prior_sacn_batch)
                        if model_rollout_batch_size > 0 and pendulum_model_rollout_replay is not None:
                            model_rollout_batch = pendulum_model_rollout_replay.sample_sacn(
                                model_rollout_batch_size,
                                n_step=config.sac.sacn_n_step,
                                require_action_log_probs=require_action_log_probs,
                            )
                            batch = concatenate_sacn_replay_samples(batch, model_rollout_batch)
                        guidance_metrics_window.append(
                            {
                                "reference_prior_batch_fraction": float(prior_sacn_batch_size)
                                / float(config.sac.batch_size),
                                "reference_prior_batch_size": float(prior_sacn_batch_size),
                                "reference_prior_size": float(
                                    reference_prior_replay.size() if reference_prior_replay is not None else 0
                                ),
                                "pendulum_model_rollout_batch_fraction": float(model_rollout_batch_size)
                                / float(config.sac.batch_size),
                                "pendulum_model_rollout_batch_size": float(model_rollout_batch_size),
                                "pendulum_model_rollout_size": float(
                                    pendulum_model_rollout_replay.size()
                                    if pendulum_model_rollout_replay is not None
                                    else 0
                                ),
                                "pendulum_hard_replay_active": float(hard_replay_active),
                                "pendulum_hard_replay_fraction": float(hard_replay_fraction),
                            }
                        )
                    else:
                        model_batch_size = 0
                        if pendulum_model_replay is not None and config.sac.pendulum_model_replay_ratio > 0.0:
                            desired_model = int(round(config.sac.batch_size * config.sac.pendulum_model_replay_ratio))
                            model_batch_size = min(
                                desired_model,
                                max(config.sac.batch_size - 1, 0),
                                pendulum_model_replay.size(),
                            )
                        prior_batch_size = 0
                        if reference_prior_replay is not None and config.sac.reference_prior_ratio > 0.0:
                            desired_prior = int(round(config.sac.batch_size * config.sac.reference_prior_ratio))
                            prior_batch_size = min(
                                desired_prior,
                                max(config.sac.batch_size - model_batch_size - 1, 0),
                                reference_prior_replay.size(),
                            )
                        online_batch_size = config.sac.batch_size - prior_batch_size - model_batch_size
                        if hard_replay_active:
                            online_batch = replay.sample_pendulum_hard_states(
                                online_batch_size,
                                fraction=hard_replay_fraction,
                                abs_theta_low=config.sac.pendulum_hard_replay_abs_theta_low,
                                abs_theta_high=config.sac.pendulum_hard_replay_abs_theta_high,
                                velocity_limit=config.sac.pendulum_hard_replay_velocity_limit,
                            )
                        else:
                            online_batch = replay.sample(online_batch_size)
                        batches = [online_batch]
                        if prior_batch_size > 0 and reference_prior_replay is not None:
                            prior_batch = reference_prior_replay.sample(prior_batch_size)
                            batches.append(prior_batch)
                        if model_batch_size > 0 and pendulum_model_replay is not None:
                            batches.append(pendulum_model_replay.sample(model_batch_size))
                        batch = concatenate_replay_samples(*batches)
                        guidance_metrics_window.append(
                            {
                                "reference_prior_batch_fraction": float(prior_batch_size)
                                / float(config.sac.batch_size),
                                "reference_prior_batch_size": float(prior_batch_size),
                                "pendulum_model_replay_batch_fraction": float(model_batch_size)
                                / float(config.sac.batch_size),
                                "pendulum_model_replay_batch_size": float(model_batch_size),
                                "pendulum_model_replay_size": float(
                                    pendulum_model_replay.size() if pendulum_model_replay is not None else 0
                                ),
                                "pendulum_hard_replay_active": float(hard_replay_active),
                                "pendulum_hard_replay_fraction": float(hard_replay_fraction),
                            }
                        )
                    reference_actions_batch = None
                    reference_actions_replay_fraction = None
                    if reference_auxiliary is not None:
                        reference_actions_batch, reference_actions_replay_fraction = (
                            resolve_replay_reference_actions(
                                batch,
                                field="reference_actions",
                                reference=reference_auxiliary,
                            )
                        )
                    reference_critic_actions_batch = None
                    reference_critic_actions_replay_fraction = None
                    if reference_critic is not None:
                        reference_critic_actions_batch, reference_critic_actions_replay_fraction = (
                            resolve_replay_reference_actions(
                                batch,
                                field="reference_critic_actions",
                                reference=reference_critic,
                            )
                        )
                    reference_anchor_observations_batch = None
                    reference_anchor_actions_batch = None
                    if (
                        reference_anchor_observations is not None
                        and reference_anchor_actions is not None
                        and reference_anchor_rng is not None
                    ):
                        ratio = float(config.sac.reference_anchor_ratio)
                        anchor_batch_size = max(
                            1,
                            int(round(config.sac.batch_size * ratio / max(1.0 - ratio, 1e-8))),
                        )
                        anchor_indices = reference_anchor_rng.integers(
                            0,
                            len(reference_anchor_observations),
                            size=anchor_batch_size,
                        )
                        reference_anchor_observations_batch = reference_anchor_observations[
                            anchor_indices
                        ]
                        reference_anchor_actions_batch = reference_anchor_actions[anchor_indices]
                    update_step += 1
                    collect_update_metrics = bool(
                        config.sac.update_diagnostics
                        or should_run(global_step, config.telemetry.log_interval_steps)
                    )
                    update_metrics = agent.update(
                        batch,
                        update_step,
                        reference_actions=reference_actions_batch,
                        reference_critic_actions=reference_critic_actions_batch,
                        reference_anchor_observations=reference_anchor_observations_batch,
                        reference_anchor_actions=reference_anchor_actions_batch,
                        collect_metrics=collect_update_metrics,
                    )
                    if config.sac.replay_priority_mode != "none":
                        replay_indices = getattr(batch, "replay_indices", None)
                        priority_values = agent.last_replay_priority_values
                        if replay_indices is None or priority_values is None:
                            raise RuntimeError(
                                "Prioritized replay update did not produce aligned replay indices and priorities."
                            )
                        replay.update_priorities(
                            replay_indices.detach().cpu().numpy(),
                            priority_values,
                        )
                    if update_metrics:
                        if reference_actions_replay_fraction is not None:
                            update_metrics["reference_actions_from_replay_fraction"] = float(
                                reference_actions_replay_fraction
                            )
                        if reference_critic_actions_replay_fraction is not None:
                            update_metrics["reference_critic_actions_from_replay_fraction"] = float(
                                reference_critic_actions_replay_fraction
                            )
                        update_metrics_window.append(update_metrics)

            if (
                failure_curriculum is not None
                and global_step < config.sac.total_steps
                and failure_curriculum.should_refresh(global_step)
            ):
                # SACAgent's deterministic action path currently samples before
                # returning its mean. Preserve all learner RNG streams so this
                # reward-only diagnostic intervention does not perturb training.
                with preserve_training_rng_state():
                    discovery = failure_curriculum.refresh(
                        agent,
                        learning_step=global_step,
                    )
                if not replace_pendulum_failure_reset_states(env, discovery.selected_states):
                    raise RuntimeError(
                        "Failure-start discovery completed but the training reset wrapper was absent."
                    )
                discovery_metrics = discovery.scalar_metrics()
                discovery_metrics.update(
                    {
                        "refresh_count": float(failure_curriculum.refresh_count),
                        "discovery_environment_steps_cumulative": float(
                            failure_curriculum.discovery_environment_steps
                        ),
                        "learning_environment_steps_so_far": float(global_step),
                        "learning_plus_failure_discovery_environment_steps": float(
                            global_step + failure_curriculum.discovery_environment_steps
                        ),
                        "failure_curriculum_reset_probability": float(
                            config.env.pendulum_failure_reset_prob
                        ),
                    }
                )
                logger.log_metrics(global_step, "failure_curriculum", discovery_metrics)
                logger.log_event(
                    "pendulum_failure_curriculum_refresh",
                    global_step,
                    {
                        **discovery_metrics,
                        "selection_signal": "undiscounted_environment_return_only",
                        "candidate_source": "environment_reset_distribution",
                        "policy_mode": "deterministic_actor_mean",
                        "replay_transitions_added": 0,
                        "observation_normalization_updates": 0,
                        "candidate_states_theta_theta_dot": discovery.candidate_states.tolist(),
                        "candidate_returns": discovery.candidate_returns.tolist(),
                        "selected_states_theta_theta_dot": discovery.selected_states.tolist(),
                        "selected_returns": discovery.selected_returns.tolist(),
                    },
                )

            if should_run(global_step, config.telemetry.log_interval_steps):
                log_update_window(logger, global_step, update_metrics_window)
                log_guidance_window(logger, global_step, guidance_metrics_window)

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
                last_diagnostics_step = global_step

            if should_run(global_step, config.eval.every_steps):
                last_eval_step = log_evaluation(logger, agent, config, eval_seeds, step=global_step)

            if config.telemetry.save_model and should_run(global_step, config.telemetry.checkpoint_interval_steps):
                save_training_checkpoint(
                    logger,
                    agent,
                    run_dir,
                    step=global_step,
                    update_step=update_step,
                    filename=f"step_{global_step}.pt",
                )

        log_update_window(logger, config.sac.total_steps, update_metrics_window)
        log_guidance_window(logger, config.sac.total_steps, guidance_metrics_window)
        if (
            config.eval.every_steps > 0
            and last_eval_step != config.sac.total_steps
        ):
            log_evaluation(logger, agent, config, eval_seeds, step=config.sac.total_steps)

        if config.telemetry.save_replay:
            replay_path = run_dir / "replay_final.npz"
            replay.save_npz(replay_path)
            logger.log_event("replay_saved", config.sac.total_steps, {"path": str(replay_path)})

        if config.telemetry.save_model:
            save_training_checkpoint(
                logger,
                agent,
                run_dir,
                step=config.sac.total_steps,
                update_step=update_step,
                filename="final.pt",
            )

        # Always expose representation health at the actual terminal model when
        # diagnostics are enabled.  This runs after replay/checkpoint persistence,
        # so the extra read-only batch sample cannot alter either the trained
        # artifact or the RNG state from which a checkpoint would resume.
        if (
            config.telemetry.diagnostics_interval_steps > 0
            and last_diagnostics_step != config.sac.total_steps
            and replay.size() >= config.sac.batch_size
        ):
            diag_batch = replay.sample(config.sac.batch_size, count=False)
            diagnostics = agent.diagnostics(
                diag_batch, config.reliability.dormant_relative_threshold
            )
            diagnostics["terminal_diagnostic"] = 1.0
            logger.log_metrics(config.sac.total_steps, "diagnostics", diagnostics)
            logger.log_event(
                "diagnostics", config.sac.total_steps, diagnostics
            )
            last_diagnostics_step = config.sac.total_steps

        failure_reset_usage = pendulum_failure_reset_stats(env)
        failure_discovery_steps = int(
            failure_curriculum.discovery_environment_steps
            if failure_curriculum is not None
            else 0
        )
        logger.log_event(
            "run_complete",
            config.sac.total_steps,
            {
                "episodes": episode_id,
                "updates": update_step,
                "learning_environment_steps": int(config.sac.total_steps),
                "failure_discovery_environment_steps": failure_discovery_steps,
                "learning_plus_failure_discovery_environment_steps": int(
                    config.sac.total_steps + failure_discovery_steps
                ),
                "failure_discovery_environment_steps_planned_upper_bound": int(
                    planned_failure_discovery_upper_bound
                ),
                "learning_plus_failure_discovery_steps_planned_upper_bound": int(
                    config.sac.total_steps + planned_failure_discovery_upper_bound
                ),
                "failure_curriculum_refreshes": int(
                    failure_curriculum.refresh_count if failure_curriculum is not None else 0
                ),
                "failure_reset_usage": failure_reset_usage,
            },
        )
        return run_dir
    finally:
        if failure_curriculum is not None:
            failure_curriculum.close()
        if env is not None:
            env.close()
        logger.close()


def should_run(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def resolve_replay_reference_actions(
    batch: Any,
    *,
    field: str,
    reference: PendulumReferenceGuidance,
) -> tuple[np.ndarray, float]:
    """Use collection-time DAgger labels, filling only legacy/unlabeled rows.

    Collection-time labels preserve the actual remaining episode horizon.  The
    fallback keeps old replay/synthetic-transition configurations loadable, but
    reports its fraction so a supposedly clean joint run cannot hide it.
    """

    observations = batch.observations.detach().cpu().numpy()
    stored = getattr(batch, field, None)
    if stored is None:
        return reference.act_batch(observations), 0.0
    actions = stored.detach().cpu().numpy().copy()
    finite_rows = np.isfinite(actions).all(axis=1)
    if not np.all(finite_rows):
        actions[~finite_rows] = reference.act_batch(observations[~finite_rows])
    return actions, float(np.mean(finite_rows))


def uniform_box_action_log_prob(action_space: Any) -> float:
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    width = high - low
    if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
        raise ValueError("SACn behavior log-prob storage requires finite Box action bounds.")
    return float(-np.sum(np.log(width)))


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
    logger.log_metrics(step, "eval", scalar_eval)
    logger.log_eval_episodes(
        step,
        evaluation,
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


def save_training_checkpoint(
    logger: TelemetryLogger,
    agent: SACAgent,
    run_dir: Path,
    *,
    step: int,
    update_step: int,
    filename: str,
) -> Path:
    checkpoint_path = run_dir / "checkpoints" / filename
    agent.save_checkpoint(
        checkpoint_path,
        extra={
            "global_step": step,
            "update_step": update_step,
            "run_dir": str(run_dir),
        },
    )
    logger.log_event("checkpoint_saved", step, {"path": str(checkpoint_path)})
    return checkpoint_path


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


def log_guidance_window(
    logger: TelemetryLogger,
    step: int,
    guidance_metrics_window: list[dict[str, float]],
) -> None:
    summary = summarize_metric_window(guidance_metrics_window)
    if not summary:
        return
    logger.log_metrics(step, "reference_guidance", summary)
    logger.log_event("reference_guidance", step, summary)
    guidance_metrics_window.clear()


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


@contextmanager
def preserve_training_rng_state():
    """Keep auxiliary deterministic rollouts from advancing learner RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.random.get_rng_state()
    torch_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_cpu_state)
        if torch_cuda_states is not None:
            torch.cuda.set_rng_state_all(torch_cuda_states)


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
