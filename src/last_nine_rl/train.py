from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from last_nine_rl.config import ExperimentConfig, resolve_device
from last_nine_rl.envs import UprightDetector, make_env
from last_nine_rl.evaluate import evaluate_agent, fixed_eval_seeds, threshold_fractions
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
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
    parser.add_argument("--pendulum-hard-reset-prob", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-abs-theta-low", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-abs-theta-high", type=float, default=None)
    parser.add_argument("--pendulum-hard-reset-velocity-limit", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--policy-lr", type=float, default=None)
    parser.add_argument("--q-lr", type=float, default=None)
    parser.add_argument("--policy-lr-final", type=float, default=None)
    parser.add_argument("--q-lr-final", type=float, default=None)
    parser.add_argument("--alpha-initial-value", type=float, default=None)
    parser.add_argument("--target-entropy-scale", type=float, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
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
        "--pendulum-hard-replay-fraction",
        type=float,
        default=None,
        help="Fraction of each update batch sampled from Pendulum hard-boundary states; 0 disables it.",
    )
    parser.add_argument("--pendulum-hard-replay-abs-theta-low", type=float, default=None)
    parser.add_argument("--pendulum-hard-replay-abs-theta-high", type=float, default=None)
    parser.add_argument("--pendulum-hard-replay-velocity-limit", type=float, default=None)
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
        "--fast-updates",
        action="store_true",
        help="Skip per-optimizer-step parameter/gradient norm telemetry for faster architecture ablation runs.",
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
    parser.add_argument("--reference-guidance-dp-solution", default=None)
    parser.add_argument(
        "--reference-auxiliary-mode",
        choices=("none", "bc", "q_filtered_bc"),
        default=None,
        help="Add an actor imitation loss toward a Pendulum reference policy; q_filtered_bc only clones when critics prefer the reference action.",
    )
    parser.add_argument(
        "--reference-auxiliary-policy",
        choices=("controller", "dp", "best"),
        default=None,
        help="Reference source for the auxiliary actor loss. DP is the cheap default for batch updates.",
    )
    parser.add_argument("--reference-auxiliary-weight", type=float, default=None)
    parser.add_argument("--reference-auxiliary-margin", type=float, default=None)
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
    if getattr(args, "pendulum_hard_reset_prob", None) is not None:
        config.env.pendulum_hard_reset_prob = args.pendulum_hard_reset_prob
    if getattr(args, "pendulum_hard_reset_abs_theta_low", None) is not None:
        config.env.pendulum_hard_reset_abs_theta_low = args.pendulum_hard_reset_abs_theta_low
    if getattr(args, "pendulum_hard_reset_abs_theta_high", None) is not None:
        config.env.pendulum_hard_reset_abs_theta_high = args.pendulum_hard_reset_abs_theta_high
    if getattr(args, "pendulum_hard_reset_velocity_limit", None) is not None:
        config.env.pendulum_hard_reset_velocity_limit = args.pendulum_hard_reset_velocity_limit
    if args.total_steps is not None:
        config.sac.total_steps = args.total_steps
    if getattr(args, "buffer_size", None) is not None:
        config.sac.buffer_size = args.buffer_size
    if getattr(args, "batch_size", None) is not None:
        config.sac.batch_size = args.batch_size
    if args.learning_starts is not None:
        config.sac.learning_starts = args.learning_starts
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
    if getattr(args, "alpha_initial_value", None) is not None:
        config.sac.alpha_initial_value = args.alpha_initial_value
    if getattr(args, "target_entropy_scale", None) is not None:
        config.sac.target_entropy_scale = args.target_entropy_scale
    if getattr(args, "updates_per_step", None) is not None:
        config.sac.updates_per_step = args.updates_per_step
    if getattr(args, "redo_interval_updates", None) is not None:
        config.sac.redo_interval_updates = args.redo_interval_updates
    if getattr(args, "redo_dormant_threshold", None) is not None:
        config.sac.redo_dormant_threshold = args.redo_dormant_threshold
    if getattr(args, "swd_linear_decay_steps", None) is not None:
        config.sac.swd_linear_decay_steps = args.swd_linear_decay_steps
    if getattr(args, "swd_min_weight", None) is not None:
        config.sac.swd_min_weight = args.swd_min_weight
    if getattr(args, "pendulum_hard_replay_fraction", None) is not None:
        config.sac.pendulum_hard_replay_fraction = args.pendulum_hard_replay_fraction
    if getattr(args, "pendulum_hard_replay_abs_theta_low", None) is not None:
        config.sac.pendulum_hard_replay_abs_theta_low = args.pendulum_hard_replay_abs_theta_low
    if getattr(args, "pendulum_hard_replay_abs_theta_high", None) is not None:
        config.sac.pendulum_hard_replay_abs_theta_high = args.pendulum_hard_replay_abs_theta_high
    if getattr(args, "pendulum_hard_replay_velocity_limit", None) is not None:
        config.sac.pendulum_hard_replay_velocity_limit = args.pendulum_hard_replay_velocity_limit
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
    if getattr(args, "fast_updates", False):
        config.sac.update_diagnostics = False
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
    if getattr(args, "reference_guidance_dp_solution", None) is not None:
        config.sac.reference_guidance_dp_solution_path = args.reference_guidance_dp_solution
    if getattr(args, "reference_auxiliary_mode", None) is not None:
        config.sac.reference_auxiliary_mode = args.reference_auxiliary_mode
    if getattr(args, "reference_auxiliary_policy", None) is not None:
        config.sac.reference_auxiliary_policy = args.reference_auxiliary_policy
    if getattr(args, "reference_auxiliary_weight", None) is not None:
        config.sac.reference_auxiliary_weight = args.reference_auxiliary_weight
    if getattr(args, "reference_auxiliary_margin", None) is not None:
        config.sac.reference_auxiliary_margin = args.reference_auxiliary_margin
    if getattr(args, "reference_critic_mode", None) is not None:
        config.sac.reference_critic_mode = args.reference_critic_mode
    if getattr(args, "reference_critic_policy", None) is not None:
        config.sac.reference_critic_policy = args.reference_critic_policy
    if getattr(args, "reference_critic_weight", None) is not None:
        config.sac.reference_critic_weight = args.reference_critic_weight
    if getattr(args, "reference_critic_margin", None) is not None:
        config.sac.reference_critic_margin = args.reference_critic_margin
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
        env = make_env(
            config.env.env_id,
            seed=config.seed,
            max_episode_steps=config.env.max_episode_steps,
            pendulum_hard_reset_prob=config.env.pendulum_hard_reset_prob,
            pendulum_hard_reset_abs_theta_low=config.env.pendulum_hard_reset_abs_theta_low,
            pendulum_hard_reset_abs_theta_high=config.env.pendulum_hard_reset_abs_theta_high,
            pendulum_hard_reset_velocity_limit=config.env.pendulum_hard_reset_velocity_limit,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, config.sac, device=device)
        replay = InstrumentedReplayBuffer(
            config.sac.buffer_size,
            env.observation_space,
            env.action_space,
            device,
            n_envs=1,
            handle_timeout_termination=False,
            swd_linear_decay_steps=config.sac.swd_linear_decay_steps,
            swd_min_weight=config.sac.swd_min_weight,
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
        if config.sac.reference_auxiliary_mode != "none" and config.sac.reference_auxiliary_weight > 0.0:
            reference_auxiliary = PendulumReferenceGuidance(
                policy=config.sac.reference_auxiliary_policy,
                dp_solution_path=config.sac.reference_guidance_dp_solution_path,
                horizon=int(config.env.max_episode_steps or 200),
            )
        reference_critic = None
        if config.sac.reference_critic_mode != "none" and config.sac.reference_critic_weight > 0.0:
            reference_critic = PendulumReferenceGuidance(
                policy=config.sac.reference_critic_policy,
                dp_solution_path=config.sac.reference_guidance_dp_solution_path,
                horizon=int(config.env.max_episode_steps or 200),
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
                "reference_guidance": reference_guidance.metadata() if reference_guidance is not None else None,
                "reference_auxiliary": reference_auxiliary.metadata() if reference_auxiliary is not None else None,
                "reference_critic": reference_critic.metadata() if reference_critic is not None else None,
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
        last_eval_step = log_evaluation(logger, agent, config, eval_seeds, step=0)

        for global_step in range(1, config.sac.total_steps + 1):
            agent.observe(obs)
            remaining_steps = int(config.env.max_episode_steps or 200) - episode_length
            guidance_draw = random.random()
            reference_action = None
            if reference_guidance is not None and guidance_draw < config.sac.reference_guidance_probability:
                reference_action = reference_guidance.act(obs, remaining_steps=remaining_steps)
            if global_step <= config.sac.learning_starts:
                action = env.action_space.sample()
                behavior_action_log_prob: float | None = random_action_log_prob
            else:
                action, behavior_action_log_prob = agent.act_with_log_prob(obs)
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
                )
                guidance_metrics_window.append({"injected_reference_transitions": 1.0})
            elif reference_guidance is not None:
                guidance_metrics_window.append({"injected_reference_transitions": 0.0})

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
                    sacn_active = config.sac.sacn_n_step > 1 and (
                        config.sac.sacn_stop_after_steps <= 0
                        or global_step <= config.sac.sacn_stop_after_steps
                    )
                    if sacn_active:
                        require_action_log_probs = config.sac.sacn_importance_mode == "density"
                        if config.sac.pendulum_hard_replay_fraction > 0.0:
                            batch = replay.sample_sacn_pendulum_hard_states(
                                config.sac.batch_size,
                                n_step=config.sac.sacn_n_step,
                                fraction=config.sac.pendulum_hard_replay_fraction,
                                abs_theta_low=config.sac.pendulum_hard_replay_abs_theta_low,
                                abs_theta_high=config.sac.pendulum_hard_replay_abs_theta_high,
                                velocity_limit=config.sac.pendulum_hard_replay_velocity_limit,
                                max_age_steps=config.sac.sacn_recent_max_age_steps,
                                require_action_log_probs=require_action_log_probs,
                            )
                        else:
                            batch = replay.sample_sacn(
                                config.sac.batch_size,
                                n_step=config.sac.sacn_n_step,
                                max_age_steps=config.sac.sacn_recent_max_age_steps,
                                require_action_log_probs=require_action_log_probs,
                            )
                    elif config.sac.pendulum_hard_replay_fraction > 0.0:
                        batch = replay.sample_pendulum_hard_states(
                            config.sac.batch_size,
                            fraction=config.sac.pendulum_hard_replay_fraction,
                            abs_theta_low=config.sac.pendulum_hard_replay_abs_theta_low,
                            abs_theta_high=config.sac.pendulum_hard_replay_abs_theta_high,
                            velocity_limit=config.sac.pendulum_hard_replay_velocity_limit,
                        )
                    else:
                        batch = replay.sample(config.sac.batch_size)
                    reference_actions_batch = None
                    if reference_auxiliary is not None:
                        reference_actions_batch = reference_auxiliary.act_batch(
                            batch.observations.detach().cpu().numpy()
                        )
                    reference_critic_actions_batch = None
                    if reference_critic is not None:
                        reference_critic_actions_batch = reference_critic.act_batch(
                            batch.observations.detach().cpu().numpy()
                        )
                    update_step += 1
                    update_metrics_window.append(
                        agent.update(
                            batch,
                            update_step,
                            reference_actions=reference_actions_batch,
                            reference_critic_actions=reference_critic_actions_batch,
                        )
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

            if should_run(global_step, config.eval.every_steps):
                last_eval_step = log_evaluation(logger, agent, config, eval_seeds, step=global_step)

        log_update_window(logger, config.sac.total_steps, update_metrics_window)
        log_guidance_window(logger, config.sac.total_steps, guidance_metrics_window)
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
