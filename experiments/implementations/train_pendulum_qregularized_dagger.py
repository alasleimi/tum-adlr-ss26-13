from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import ReliabilityConfig, resolve_device
from last_nine_rl.critic_policy_extraction import configure_trainable_actor, train_actor_epoch
from last_nine_rl.distill_reference import collect_dagger_dataset
from last_nine_rl.envs import UprightDetector
from last_nine_rl.hybrid_qsearch import FixedLocalCriticQSearchPolicy
from last_nine_rl.pendulum_grid import (
    pendulum_obs_batch,
    pendulum_step_batch,
    rollout_pendulum_grid_vectorized,
)
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from last_nine_rl.sac import SACAgent


DEFAULT_DP_SOLUTION = Path("data/reference/pendulum_dp_solution.npz")


class FiniteHorizonReferencePolicy:
    def __init__(self, policy: str, dp_solution: Path, horizon: int = 200) -> None:
        self.guidance = PendulumReferenceGuidance(
            policy,
            dp_solution_path=str(dp_solution) if policy in {"dp", "best"} else None,
            horizon=horizon,
        )
        self.remaining = int(horizon)

    def act_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        actions = self.guidance.act_batch(observations, remaining_steps=self.remaining)
        self.remaining = max(1, self.remaining - 1)
        return actions


class TimeConditionedPolicy:
    """Expose a four-input actor through the ordinary three-state rollout interface."""

    def __init__(self, agent: SACAgent, horizon: int = 200) -> None:
        self.agent = agent
        self.horizon = int(horizon)
        self.remaining = int(horizon)

    def act_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        time_feature = np.full(
            (len(observations), 1),
            self.remaining / float(self.horizon),
            dtype=np.float32,
        )
        augmented = np.concatenate(
            [np.asarray(observations, dtype=np.float32), time_feature], axis=1
        )
        actions = self.agent.act_batch(augmented, deterministic=deterministic)
        self.remaining = max(1, self.remaining - 1)
        return actions


def initialize_time_conditioned_actor(
    source: SACAgent, config: Any, device: str
) -> SACAgent:
    """Add a zero-weight time input while preserving the source actor exactly."""
    student = SACAgent(
        4,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        config.sac,
        device=device,
    )
    source_state = source.actor.state_dict()
    student_state = student.actor.state_dict()
    embedder_key = "backbone.embedder.w.linear.weight"
    for key, target in student_state.items():
        source_value = source_state[key]
        if key == embedder_key:
            if source_value.shape[1] + 1 != target.shape[1]:
                raise ValueError("unexpected SimbaV2 embedder shape for time extension")
            target.zero_()
            target[:, :3].copy_(source_value[:, :3])
            target[:, 4].copy_(source_value[:, 3])
        else:
            if source_value.shape != target.shape:
                raise ValueError(f"unexpected actor shape change for {key}")
            target.copy_(source_value)
    student.actor.load_state_dict(student_state)
    if source.obs_rms is not None and student.obs_rms is not None:
        student.obs_rms.mean[:3] = source.obs_rms.mean
        student.obs_rms.var[:3] = source.obs_rms.var
        student.obs_rms.mean[3] = 0.5
        student.obs_rms.var[3] = 1.0 / 12.0
        student.obs_rms.count = source.obs_rms.count
    return student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one DAgger actor with conservative, training-only targets proposed by "
            "seed-matched pure-RL critics."
        )
    )
    parser.add_argument("--dagger-run", required=True)
    parser.add_argument("--rl-run", required=True)
    parser.add_argument(
        "--hard-teacher-run",
        default=None,
        help="Optional supervised hard-policy checkpoint used only for failure-band labels.",
    )
    parser.add_argument(
        "--failure-hard-teacher",
        action="store_true",
        help=(
            "For hard-diagonal failure-mixture episodes, label positive starts with the "
            "hard teacher and negative starts with its Pendulum-symmetric action."
        ),
    )
    parser.add_argument(
        "--hard-teacher-prefix-steps",
        type=int,
        default=200,
        help=(
            "Use the optional hard teacher only for this many initial rollout steps; "
            "the best reference labels the remaining learner-visited states."
        ),
    )
    parser.add_argument(
        "--dagger-save-prefix-steps",
        type=int,
        default=200,
        help=(
            "Aggregate only this many initial states from each failure-mixture "
            "rollout while still executing and scoring the full 200-step episode."
        ),
    )
    parser.add_argument(
        "--failure-expert-beta",
        type=float,
        default=0.0,
        help=(
            "Probability that the supervised target action, rather than the learner "
            "action, is executed at each failure-mixture transition. Zero is clean "
            "learner-only DAgger; positive values are explicit expert augmentation."
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dp-solution", default=str(DEFAULT_DP_SOLUTION))
    parser.add_argument("--static-size", type=int, default=240_000)
    parser.add_argument(
        "--static-target-policy",
        choices=("reference", "source_actor"),
        default="reference",
        help=(
            "Label the static support with the best reference, or preserve the "
            "initial accepted actor on that support while targeted states use expert labels."
        ),
    )
    parser.add_argument("--broad-fraction", type=float, default=0.20)
    parser.add_argument("--reset-support-fraction", type=float, default=0.50)
    parser.add_argument("--hard120-fraction", type=float, default=0.15)
    parser.add_argument("--near-down-fraction", type=float, default=0.15)
    parser.add_argument("--broad-velocity-limit", type=float, default=8.0)
    parser.add_argument("--reset-velocity-limit", type=float, default=1.0)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=50)
    parser.add_argument(
        "--dagger-initial-mode",
        choices=(
            "standard_reset",
            "priority_uniform",
            "hard120_edge",
            "failure_mixture",
        ),
        default="standard_reset",
    )
    parser.add_argument(
        "--priority-candidate-multiplier",
        type=int,
        default=10,
        help="Uniform candidate-pool size divided by the requested DAgger episodes.",
    )
    parser.add_argument(
        "--priority-fraction",
        type=float,
        default=0.75,
        help=(
            "Fraction of DAgger starts selected by largest reference regret; the "
            "remainder is sampled uniformly from the candidate pool."
        ),
    )
    parser.add_argument(
        "--priority-expert-demonstrations-per-round",
        type=int,
        default=0,
        help=(
            "For priority_uniform only, additionally execute this many best-reference "
            "recovery demonstrations from the automatically selected starts. Learner-only "
            "DAgger collection is retained in full."
        ),
    )
    parser.add_argument("--hard-dagger-velocity-low", type=float, default=0.75)
    parser.add_argument("--failure-hard-fraction", type=float, default=0.40)
    parser.add_argument("--failure-near-down-fraction", type=float, default=0.40)
    parser.add_argument("--failure-wrap-fraction", type=float, default=0.20)
    parser.add_argument(
        "--tight-failure-bands",
        action="store_true",
        help=(
            "Concentrate failure-mixture starts in continuous neighborhoods around "
            "the residual 126.9-degree and -174.1-degree basin boundaries."
        ),
    )
    parser.add_argument("--targeted-validation-size", type=int, default=0)
    parser.add_argument(
        "--targeted-validation-mode",
        choices=("failure_mixture", "reset_uniform"),
        default="failure_mixture",
        help=(
            "Distribution for the optional large validation holdout. reset_uniform "
            "draws continuously from the full Pendulum reset support and contains no "
            "hand-written failure regions."
        ),
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Run rollout checkpoint selection every N supervised epochs.",
    )
    parser.add_argument("--epochs-per-round", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-actions", type=int, default=41)
    parser.add_argument("--rl-margin", type=float, default=0.05)
    parser.add_argument("--rl-blend", type=float, default=0.02)
    parser.add_argument("--max-target-shift", type=float, default=0.10)
    parser.add_argument("--selected-weight", type=float, default=4.0)
    parser.add_argument(
        "--weight-all-dagger-samples",
        action="store_true",
        help="Apply --selected-weight to every learner-visited DAgger sample.",
    )
    parser.add_argument(
        "--dagger-disagreement-threshold",
        type=float,
        default=0.0,
        help=(
            "If positive, apply --selected-weight only where the current actor and "
            "the supervised DAgger target differ by more than this action amount."
        ),
    )
    parser.add_argument("--model-rl-updates-per-epoch", type=int, default=0)
    parser.add_argument("--model-rl-batch-size", type=int, default=64)
    parser.add_argument("--model-rl-horizon", type=int, default=200)
    parser.add_argument("--model-rl-weight", type=float, default=1.0)
    parser.add_argument("--model-rl-bc-weight", type=float, default=100.0)
    parser.add_argument(
        "--trainable-actor",
        choices=("all", "mean_head", "last_layer", "time_input"),
        default="last_layer",
    )
    parser.add_argument(
        "--student-actor-hidden-dim",
        type=int,
        default=0,
        help="If positive, train a newly initialized SimbaV2 student of this width.",
    )
    parser.add_argument(
        "--student-actor-blocks",
        type=int,
        default=2,
        help="Residual blocks for a newly initialized SimbaV2 student.",
    )
    parser.add_argument(
        "--time-conditioned-student",
        action="store_true",
        help=(
            "Extend the accepted actor with normalized remaining time as a fourth "
            "input, initialized to reproduce the source actor exactly."
        ),
    )
    parser.add_argument("--validation-theta-bins", type=int, default=17)
    parser.add_argument("--validation-velocity-bins", type=int, default=11)
    parser.add_argument(
        "--validation-qsearch-radius",
        type=float,
        default=0.0,
        help=(
            "If positive, select checkpoints using the deployed fixed local pure-RL "
            "critic Q-search policy rather than the raw supervised actor."
        ),
    )
    parser.add_argument("--validation-qsearch-num-actions", type=int, default=5)
    parser.add_argument("--validation-qsearch-margin", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "static_size",
        "broad_velocity_limit",
        "reset_velocity_limit",
        "dagger_rounds",
        "dagger_episodes",
        "hard_dagger_velocity_low",
        "epochs_per_round",
        "batch_size",
        "lr",
        "num_actions",
        "selected_weight",
        "validation_theta_bins",
        "validation_velocity_bins",
        "validation_every_epochs",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    fractions = np.asarray(
        [
            args.broad_fraction,
            args.reset_support_fraction,
            args.hard120_fraction,
            args.near_down_fraction,
        ],
        dtype=np.float64,
    )
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("state mixture fractions must be nonnegative and sum to one")
    if not 0.0 <= float(args.rl_blend) <= 1.0:
        raise ValueError("--rl-blend must be in [0, 1]")
    if float(args.rl_margin) < 0.0 or float(args.max_target_shift) < 0.0:
        raise ValueError("RL margin and maximum target shift must be nonnegative")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be nonnegative")
    if float(args.dagger_disagreement_threshold) < 0.0:
        raise ValueError("--dagger-disagreement-threshold must be nonnegative")
    if bool(args.weight_all_dagger_samples) and float(args.dagger_disagreement_threshold) > 0.0:
        raise ValueError("all-sample and disagreement DAgger weighting are exclusive")
    if float(args.hard_dagger_velocity_low) >= float(args.reset_velocity_limit):
        raise ValueError("hard DAgger velocity lower bound must be below reset velocity limit")
    failure_fractions = np.asarray(
        [
            args.failure_hard_fraction,
            args.failure_near_down_fraction,
            args.failure_wrap_fraction,
        ],
        dtype=np.float64,
    )
    if np.any(failure_fractions < 0.0) or not np.isclose(failure_fractions.sum(), 1.0):
        raise ValueError("failure-mixture fractions must be nonnegative and sum to one")
    if int(args.targeted_validation_size) < 0:
        raise ValueError("--targeted-validation-size must be nonnegative")
    if int(args.priority_candidate_multiplier) < 1:
        raise ValueError("--priority-candidate-multiplier must be at least one")
    if not 0.0 <= float(args.priority_fraction) <= 1.0:
        raise ValueError("--priority-fraction must be in [0, 1]")
    if int(args.priority_expert_demonstrations_per_round) < 0:
        raise ValueError("--priority-expert-demonstrations-per-round must be nonnegative")
    if int(args.priority_expert_demonstrations_per_round) > int(args.dagger_episodes):
        raise ValueError("priority expert demonstrations cannot exceed DAgger episodes")
    if (
        int(args.priority_expert_demonstrations_per_round) > 0
        and str(args.dagger_initial_mode) != "priority_uniform"
    ):
        raise ValueError("priority expert demonstrations require priority_uniform starts")
    if bool(args.failure_hard_teacher) and not args.hard_teacher_run:
        raise ValueError("--failure-hard-teacher requires --hard-teacher-run")
    if not 0 < int(args.hard_teacher_prefix_steps) <= 200:
        raise ValueError("--hard-teacher-prefix-steps must be in [1, 200]")
    if not 0 < int(args.dagger_save_prefix_steps) <= 200:
        raise ValueError("--dagger-save-prefix-steps must be in [1, 200]")
    if not 0.0 <= float(args.failure_expert_beta) <= 1.0:
        raise ValueError("--failure-expert-beta must be in [0, 1]")
    if float(args.validation_qsearch_radius) < 0.0:
        raise ValueError("--validation-qsearch-radius must be nonnegative")
    if int(args.validation_qsearch_num_actions) < 3:
        raise ValueError("--validation-qsearch-num-actions must be at least three")
    if int(args.validation_qsearch_num_actions) % 2 == 0:
        raise ValueError("--validation-qsearch-num-actions must be odd to include the actor action")
    if float(args.validation_qsearch_margin) < 0.0:
        raise ValueError("--validation-qsearch-margin must be nonnegative")
    if int(args.model_rl_updates_per_epoch) < 0:
        raise ValueError("--model-rl-updates-per-epoch must be nonnegative")
    if int(args.model_rl_updates_per_epoch) > 0:
        if int(args.model_rl_batch_size) <= 0 or int(args.model_rl_horizon) <= 0:
            raise ValueError("model-RL batch size and horizon must be positive")
        if float(args.model_rl_weight) <= 0.0 or float(args.model_rl_bc_weight) < 0.0:
            raise ValueError("model-RL weight must be positive and BC weight nonnegative")
    if int(args.student_actor_hidden_dim) < 0 or int(args.student_actor_blocks) <= 0:
        raise ValueError("student actor width must be nonnegative and blocks positive")
    if int(args.student_actor_hidden_dim) > 0 and str(args.static_target_policy) != "source_actor":
        raise ValueError("a new student requires --static-target-policy source_actor")
    if bool(args.time_conditioned_student):
        if int(args.student_actor_hidden_dim) > 0:
            raise ValueError("time-conditioned and randomly initialized students are exclusive")
        if str(args.static_target_policy) != "source_actor":
            raise ValueError("a time-conditioned student requires source-actor anchors")
        if str(args.dagger_initial_mode) != "failure_mixture":
            raise ValueError("time-conditioned training currently requires failure_mixture")
    if str(args.trainable_actor) == "time_input" and not bool(args.time_conditioned_student):
        raise ValueError("time_input training requires --time-conditioned-student")


def sample_training_states(
    size: int,
    rng: np.random.Generator,
    broad_fraction: float,
    reset_support_fraction: float,
    hard120_fraction: float,
    near_down_fraction: float,
    broad_velocity_limit: float,
    reset_velocity_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.asarray(
        [broad_fraction, reset_support_fraction, hard120_fraction, near_down_fraction],
        dtype=np.float64,
    )
    raw_counts = fractions * int(size)
    counts = np.floor(raw_counts).astype(np.int64)
    counts[np.argmax(raw_counts - counts)] += int(size) - int(counts.sum())
    names = np.asarray(["broad", "reset_support", "hard120", "near_down"])
    theta_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for name, count in zip(names, counts):
        count = int(count)
        if name == "broad":
            theta = rng.uniform(-math.pi, math.pi, count)
            velocity = rng.uniform(-broad_velocity_limit, broad_velocity_limit, count)
        elif name == "reset_support":
            theta = rng.uniform(-math.pi, math.pi, count)
            velocity = rng.uniform(-reset_velocity_limit, reset_velocity_limit, count)
        else:
            low = math.radians(120.0 if name == "hard120" else 150.0)
            high = math.radians(135.0) if name == "hard120" else math.pi
            abs_theta = rng.uniform(low, high, count)
            theta = abs_theta * rng.choice(np.asarray([-1.0, 1.0]), count)
            velocity = rng.uniform(-reset_velocity_limit, reset_velocity_limit, count)
        theta_parts.append(theta)
        velocity_parts.append(velocity)
        source_parts.append(np.full(count, name, dtype="U16"))
    theta = np.concatenate(theta_parts)
    velocity = np.concatenate(velocity_parts)
    sources = np.concatenate(source_parts)
    order = rng.permutation(len(theta))
    return pendulum_obs_batch(theta[order], velocity[order]), sources[order]


def reference_actions_batched(
    reference: PendulumReferenceGuidance,
    observations: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        outputs.append(
            np.asarray(
                reference.act_batch(observations[start : start + batch_size]),
                dtype=np.float32,
            ).reshape(-1, 1)
        )
    return np.concatenate(outputs, axis=0)


def sample_hard120_edge_initial_states(
    episodes: int,
    rng: np.random.Generator,
    velocity_low: float,
    velocity_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    abs_theta = rng.uniform(math.radians(120.0), math.radians(135.0), int(episodes))
    theta = abs_theta * rng.choice(np.asarray([-1.0, 1.0]), int(episodes))
    abs_velocity = rng.uniform(float(velocity_low), float(velocity_high), int(episodes))
    velocity = abs_velocity * rng.choice(np.asarray([-1.0, 1.0]), int(episodes))
    return theta, velocity


def sample_failure_mixture_initial_states(
    episodes: int,
    rng: np.random.Generator,
    hard_fraction: float,
    near_down_fraction: float,
    wrap_fraction: float,
    tight_failure_bands: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample continuous neighborhoods around the three shared residual failure bands."""
    fractions = np.asarray(
        [hard_fraction, near_down_fraction, wrap_fraction], dtype=np.float64
    )
    raw_counts = fractions * int(episodes)
    counts = np.floor(raw_counts).astype(np.int64)
    counts[np.argmax(raw_counts - counts)] += int(episodes) - int(counts.sum())
    theta_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []

    hard_count, near_down_count, wrap_count = (int(value) for value in counts)
    if hard_count:
        signs = rng.choice(np.asarray([-1.0, 1.0]), hard_count)
        hard_angle_low, hard_angle_high = (
            (126.4, 127.4) if tight_failure_bands else (123.0, 131.0)
        )
        hard_velocity_low, hard_velocity_high = (
            (0.83, 0.87) if tight_failure_bands else (0.78, 0.92)
        )
        theta_parts.append(
            signs
            * rng.uniform(
                math.radians(hard_angle_low),
                math.radians(hard_angle_high),
                hard_count,
            )
        )
        # The shared failures have matching theta and velocity signs.
        velocity_parts.append(
            signs * rng.uniform(hard_velocity_low, hard_velocity_high, hard_count)
        )
        source_parts.append(np.full(hard_count, "hard_diagonal", dtype="U20"))
    if near_down_count:
        near_angle_low, near_angle_high = (
            (173.7, 174.5) if tight_failure_bands else (170.0, 178.0)
        )
        near_velocity_low, near_velocity_high = (
            (-0.065, -0.035) if tight_failure_bands else (-0.18, 0.02)
        )
        theta_parts.append(
            -rng.uniform(
                math.radians(near_angle_low),
                math.radians(near_angle_high),
                near_down_count,
            )
        )
        velocity_parts.append(
            rng.uniform(near_velocity_low, near_velocity_high, near_down_count)
        )
        source_parts.append(np.full(near_down_count, "near_down_slow", dtype="U20"))
    if wrap_count:
        signs = rng.choice(np.asarray([-1.0, 1.0]), wrap_count)
        theta_parts.append(
            signs * rng.uniform(math.radians(175.0), math.pi, wrap_count)
        )
        velocity_parts.append(rng.uniform(0.72, 0.92, wrap_count))
        source_parts.append(np.full(wrap_count, "wrap_fast", dtype="U20"))

    theta = np.concatenate(theta_parts)
    velocity = np.concatenate(velocity_parts)
    sources = np.concatenate(source_parts)
    order = rng.permutation(len(theta))
    return theta[order], velocity[order], sources[order]


def collect_hard_dagger_dataset(
    agent: SACAgent,
    reference: PendulumReferenceGuidance,
    episodes: int,
    rng: np.random.Generator,
    velocity_low: float,
    velocity_high: float,
    horizon: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    theta, theta_dot = sample_hard120_edge_initial_states(
        episodes,
        rng,
        velocity_low=velocity_low,
        velocity_high=velocity_high,
    )
    obs_array, reference_array, metrics = collect_dagger_from_initial_states(
        agent,
        reference,
        theta,
        theta_dot,
        horizon=horizon,
    )
    metrics["hard120_edge_initials"] = 1.0
    return obs_array, reference_array, metrics


def collect_dagger_from_initial_states(
    agent: SACAgent,
    reference: PendulumReferenceGuidance,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    horizon: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    theta = np.asarray(theta, dtype=np.float64).copy()
    theta_dot = np.asarray(theta_dot, dtype=np.float64).copy()
    episodes = int(len(theta))
    if episodes <= 0 or len(theta_dot) != episodes:
        raise ValueError("initial theta and velocity arrays must have matching positive lengths")
    observations: list[np.ndarray] = []
    policy_actions: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    returns = np.zeros(int(episodes), dtype=np.float64)
    for step in range(int(horizon)):
        obs = pendulum_obs_batch(theta, theta_dot)
        policy_action = agent.act_batch(obs, deterministic=True).reshape(episodes, -1)
        reference_action = reference.act_batch(
            obs, remaining_steps=int(horizon) - step
        ).reshape(episodes, -1)
        observations.append(obs)
        policy_actions.append(np.asarray(policy_action, dtype=np.float32))
        reference_actions.append(np.asarray(reference_action, dtype=np.float32))
        theta, theta_dot, reward = pendulum_step_batch(
            theta, theta_dot, policy_action.reshape(-1)
        )
        returns += reward
    obs_array = np.concatenate(observations, axis=0)
    policy_array = np.concatenate(policy_actions, axis=0)
    reference_array = np.concatenate(reference_actions, axis=0)
    metrics = {
        "episodes": float(episodes),
        "samples": float(len(obs_array)),
        "mean_return": float(returns.mean()),
        "mean_length": float(horizon),
        "policy_ref_action_mae": float(np.abs(policy_array - reference_array).mean()),
        "deterministic": 1.0,
        "expert_beta": 0.0,
        "executed_reference_fraction": 0.0,
    }
    return obs_array, reference_array, metrics


def collect_reference_demonstrations_from_initial_states(
    reference: PendulumReferenceGuidance,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    horizon: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Execute the reference from automatically selected starts and save its trajectory."""
    theta = np.asarray(theta, dtype=np.float64).copy()
    theta_dot = np.asarray(theta_dot, dtype=np.float64).copy()
    episodes = int(len(theta))
    if episodes <= 0 or len(theta_dot) != episodes:
        raise ValueError("initial theta and velocity arrays must have matching positive lengths")
    observations: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    returns = np.zeros(episodes, dtype=np.float64)
    for step in range(int(horizon)):
        obs = pendulum_obs_batch(theta, theta_dot)
        action = reference.act_batch(
            obs, remaining_steps=int(horizon) - step
        ).reshape(episodes, -1)
        observations.append(obs)
        reference_actions.append(np.asarray(action, dtype=np.float32))
        theta, theta_dot, reward = pendulum_step_batch(
            theta, theta_dot, action.reshape(-1)
        )
        returns += reward
    obs_array = np.concatenate(observations, axis=0)
    reference_array = np.concatenate(reference_actions, axis=0)
    return obs_array, reference_array, {
        "expert_demonstration_episodes": float(episodes),
        "expert_demonstration_samples": float(len(obs_array)),
        "expert_demonstration_mean_return": float(returns.mean()),
        "expert_demonstration_executed_reference_fraction": 1.0,
    }


def collect_failure_mixture_dagger_dataset(
    agent: SACAgent,
    reference: PendulumReferenceGuidance,
    episodes: int,
    rng: np.random.Generator,
    hard_fraction: float,
    near_down_fraction: float,
    wrap_fraction: float,
    tight_failure_bands: bool = False,
    hard_teacher: SACAgent | None = None,
    hard_teacher_prefix_steps: int = 200,
    save_prefix_steps: int = 200,
    expert_beta: float = 0.0,
    time_conditioned: bool = False,
    horizon: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    theta, theta_dot, sources = sample_failure_mixture_initial_states(
        episodes,
        rng,
        hard_fraction=hard_fraction,
        near_down_fraction=near_down_fraction,
        wrap_fraction=wrap_fraction,
        tight_failure_bands=tight_failure_bands,
    )
    observations: list[np.ndarray] = []
    policy_actions: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    returns = np.zeros(int(episodes), dtype=np.float64)
    executed_reference = 0
    positive_hard = (sources == "hard_diagonal") & (theta > 0.0)
    negative_hard = (sources == "hard_diagonal") & (theta < 0.0)
    for step in range(int(horizon)):
        obs = pendulum_obs_batch(theta, theta_dot)
        if time_conditioned:
            time_feature = np.full(
                (int(episodes), 1),
                (int(horizon) - step) / float(horizon),
                dtype=np.float32,
            )
            policy_obs = np.concatenate([obs, time_feature], axis=1)
        else:
            policy_obs = obs
        policy_action = agent.act_batch(policy_obs, deterministic=True).reshape(
            int(episodes), -1
        )
        reference_action = reference.act_batch(
            obs, remaining_steps=int(horizon) - step
        ).reshape(int(episodes), -1)
        use_hard_teacher = hard_teacher is not None and step < int(
            hard_teacher_prefix_steps
        )
        if use_hard_teacher and (positive_hard.any() or negative_hard.any()):
            direct_action = hard_teacher.act_batch(obs, deterministic=True).reshape(
                int(episodes), -1
            )
            mirrored_obs = np.asarray(obs, dtype=np.float32).copy()
            mirrored_obs[:, 1] *= -1.0
            mirrored_obs[:, 2] *= -1.0
            mirrored_action = -hard_teacher.act_batch(
                mirrored_obs, deterministic=True
            ).reshape(int(episodes), -1)
            reference_action[positive_hard] = direct_action[positive_hard]
            reference_action[negative_hard] = mirrored_action[negative_hard]
        if step < int(save_prefix_steps):
            observations.append(policy_obs)
            policy_actions.append(np.asarray(policy_action, dtype=np.float32))
            reference_actions.append(np.asarray(reference_action, dtype=np.float32))
        execute_reference = rng.random(int(episodes)) < float(expert_beta)
        executed_action = np.where(
            execute_reference[:, None], reference_action, policy_action
        ).reshape(-1)
        executed_reference += int(execute_reference.sum())
        theta, theta_dot, reward = pendulum_step_batch(
            theta, theta_dot, executed_action
        )
        returns += reward
    obs_array = np.concatenate(observations, axis=0)
    policy_array = np.concatenate(policy_actions, axis=0)
    reference_array = np.concatenate(reference_actions, axis=0)
    source_counts = {
        str(name): int(np.sum(sources == name)) for name in np.unique(sources)
    }
    return obs_array, reference_array, {
        "episodes": float(episodes),
        "samples": float(len(obs_array)),
        "mean_return": float(returns.mean()),
        "mean_length": float(horizon),
        "policy_ref_action_mae": float(np.abs(policy_array - reference_array).mean()),
        "deterministic": 1.0,
        "expert_beta": float(expert_beta),
        "executed_reference_fraction": float(
            executed_reference / (int(episodes) * int(horizon))
        ),
        "failure_mixture_initials": 1.0,
        "hard_teacher_labels": float(
            (
                min(
                    int(horizon),
                    int(hard_teacher_prefix_steps),
                    int(save_prefix_steps),
                )
                * int(positive_hard.sum() + negative_hard.sum())
            )
            if hard_teacher is not None
            else 0
        ),
        **{f"initial_{name}": float(count) for name, count in source_counts.items()},
    }


def q_regularized_targets(
    rl_agent: SACAgent,
    observations: np.ndarray,
    reference_actions: np.ndarray,
    num_actions: int,
    margin: float,
    blend: float,
    max_target_shift: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    targets: list[np.ndarray] = []
    selections: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    raw_deltas: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        raw_obs = torch.as_tensor(
            observations[start : start + batch_size],
            dtype=torch.float32,
            device=rl_agent.device,
        )
        ref = torch.as_tensor(
            reference_actions[start : start + batch_size],
            dtype=torch.float32,
            device=rl_agent.device,
        )
        normalized = rl_agent._normalize_obs_tensor(raw_obs)
        with torch.no_grad():
            best, _best_q, _ref_q = rl_agent._critic_search_best_actions(
                normalized,
                ref,
                num_actions=int(num_actions),
            )
            per_critic_advantage = torch.stack(
                [
                    critic(normalized, best).view(-1)
                    - critic(normalized, ref).view(-1)
                    for critic in rl_agent.q_networks
                ],
                dim=0,
            )
            unanimous_advantage = per_critic_advantage.min(dim=0).values
            selected = unanimous_advantage > float(margin)
            raw_delta = best - ref
            shift = torch.clamp(
                float(blend) * raw_delta,
                min=-float(max_target_shift),
                max=float(max_target_shift),
            )
            target = torch.where(selected[:, None], ref + shift, ref)
            target = torch.clamp(
                target,
                min=rl_agent.actor.action_bias - rl_agent.actor.action_scale,
                max=rl_agent.actor.action_bias + rl_agent.actor.action_scale,
            )
        targets.append(target.cpu().numpy())
        selections.append(selected.cpu().numpy())
        advantages.append(unanimous_advantage.cpu().numpy())
        raw_deltas.append(raw_delta.view(-1).cpu().numpy())
    target_array = np.concatenate(targets).astype(np.float32)
    selected_array = np.concatenate(selections).astype(bool)
    advantage_array = np.concatenate(advantages)
    raw_delta_array = np.concatenate(raw_deltas)
    applied_shift = target_array.reshape(-1) - reference_actions.reshape(-1)
    selected_shift = np.abs(applied_shift[selected_array])
    metrics = {
        "samples": float(len(observations)),
        "selected": float(selected_array.sum()),
        "selected_fraction": float(selected_array.mean()),
        "unanimous_advantage_mean": float(advantage_array.mean()),
        "unanimous_advantage_max": float(advantage_array.max()),
        "selected_raw_action_delta_abs_mean": float(
            np.abs(raw_delta_array[selected_array]).mean() if selected_array.any() else 0.0
        ),
        "applied_shift_abs_mean_selected": float(
            selected_shift.mean() if selected_shift.size else 0.0
        ),
        "applied_shift_abs_max": float(np.abs(applied_shift).max()),
    }
    return target_array, selected_array, metrics


def actor_mean(agent: SACAgent, observations: torch.Tensor) -> torch.Tensor:
    normalized = agent._normalize_obs_tensor(observations)
    raw_mean, _log_std = agent.actor(normalized)
    return torch.tanh(raw_mean) * agent.actor.action_scale + agent.actor.action_bias


def differentiable_pendulum_mean_cost(
    agent: SACAgent,
    theta: torch.Tensor,
    theta_dot: torch.Tensor,
    horizon: int,
    time_conditioned: bool = False,
) -> torch.Tensor:
    total_cost = torch.zeros_like(theta)
    for step in range(int(horizon)):
        obs = torch.stack([torch.cos(theta), torch.sin(theta), theta_dot], dim=1)
        if time_conditioned:
            time_feature = torch.full_like(
                theta[:, None], (int(horizon) - step) / float(horizon)
            )
            obs = torch.cat([obs, time_feature], dim=1)
        torque = actor_mean(agent, obs).reshape(-1).clamp(-2.0, 2.0)
        normalized_theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        total_cost = total_cost + (
            normalized_theta.square() + 0.1 * theta_dot.square() + 0.001 * torque.square()
        )
        theta_dot = theta_dot + (15.0 * torch.sin(theta) + 3.0 * torque) * 0.05
        theta_dot = theta_dot.clamp(-8.0, 8.0)
        theta = theta + theta_dot * 0.05
    return total_cost.mean() / float(horizon)


def model_rl_updates(
    agent: SACAgent,
    optimizer: torch.optim.Optimizer,
    anchor_observations: np.ndarray,
    anchor_targets: np.ndarray,
    rng: np.random.Generator,
    updates: int,
    batch_size: int,
    horizon: int,
    velocity_low: float,
    velocity_high: float,
    rl_weight: float,
    bc_weight: float,
    trainable_parameters: list[torch.nn.Parameter],
    tight_failure_bands: bool = False,
    time_conditioned: bool = False,
) -> dict[str, float]:
    if int(updates) <= 0:
        return {}
    costs: list[float] = []
    bc_losses: list[float] = []
    total_losses: list[float] = []
    grad_norms: list[float] = []
    action_scale = agent.actor.action_scale.detach().abs().clamp_min(1e-6)
    for _update in range(int(updates)):
        if tight_failure_bands:
            theta_np, velocity_np, _sources = sample_failure_mixture_initial_states(
                int(batch_size),
                rng,
                hard_fraction=1.0,
                near_down_fraction=0.0,
                wrap_fraction=0.0,
                tight_failure_bands=True,
            )
        else:
            theta_np, velocity_np = sample_hard120_edge_initial_states(
                int(batch_size),
                rng,
                velocity_low=float(velocity_low),
                velocity_high=float(velocity_high),
            )
        theta = torch.as_tensor(theta_np, dtype=torch.float32, device=agent.device)
        velocity = torch.as_tensor(velocity_np, dtype=torch.float32, device=agent.device)
        cost = differentiable_pendulum_mean_cost(
            agent,
            theta,
            velocity,
            int(horizon),
            time_conditioned=time_conditioned,
        )
        anchor_indices = rng.integers(0, len(anchor_observations), size=int(batch_size))
        anchor_obs = torch.as_tensor(
            anchor_observations[anchor_indices], dtype=torch.float32, device=agent.device
        )
        anchor_target = torch.as_tensor(
            anchor_targets[anchor_indices], dtype=torch.float32, device=agent.device
        )
        anchor_prediction = actor_mean(agent, anchor_obs)
        bc_loss = F.mse_loss(
            (anchor_prediction - anchor_target) / action_scale,
            torch.zeros_like(anchor_prediction),
        )
        loss = float(rl_weight) * cost + float(bc_weight) * bc_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=10.0)
        optimizer.step()
        if agent.cfg.simba_weight_projection:
            agent._project_weights()
        costs.append(float(cost.detach().cpu()))
        bc_losses.append(float(bc_loss.detach().cpu()))
        total_losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(torch.as_tensor(grad_norm).detach().cpu()))
    return {
        "model_rl_mean_step_cost": float(np.mean(costs)),
        "model_rl_bc_loss": float(np.mean(bc_losses)),
        "model_rl_total_loss": float(np.mean(total_losses)),
        "model_rl_grad_norm": float(np.mean(grad_norms)),
        "model_rl_updates": float(updates),
    }


def midpoint_grid(theta_bins: int, velocity_bins: int) -> tuple[np.ndarray, np.ndarray]:
    theta = -math.pi + (np.arange(theta_bins, dtype=np.float64) + 0.5) * (
        2.0 * math.pi / theta_bins
    )
    velocity = -1.0 + (np.arange(velocity_bins, dtype=np.float64) + 0.5) * (
        2.0 / velocity_bins
    )
    return np.tile(theta, velocity_bins), np.repeat(velocity, theta_bins)


def validation_reference_returns(
    theta: np.ndarray,
    velocity: np.ndarray,
    detector: UprightDetector,
    reliability: ReliabilityConfig,
    dp_solution: Path,
) -> np.ndarray:
    policies = (
        FiniteHorizonReferencePolicy("dp", dp_solution),
        FiniteHorizonReferencePolicy("controller", dp_solution),
    )
    returns = []
    for policy in policies:
        rows = rollout_pendulum_grid_vectorized(
            policy,
            theta,
            velocity,
            detector,
            reliability,
            horizon=200,
        )
        returns.append(np.asarray([row["return"] for row in rows], dtype=np.float64))
    return np.maximum(returns[0], returns[1])


def evaluate_validation(
    agent: Any,
    theta: np.ndarray,
    velocity: np.ndarray,
    reference_returns: np.ndarray,
    detector: UprightDetector,
    reliability: ReliabilityConfig,
    time_conditioned: bool = False,
    critic_agent: Any | None = None,
    qsearch_radius: float = 0.0,
    qsearch_num_actions: int = 5,
    qsearch_margin: float = 0.0,
) -> dict[str, float]:
    rollout_agent: Any = (
        TimeConditionedPolicy(agent, horizon=200) if time_conditioned else agent
    )
    qsearch_policy = None
    if float(qsearch_radius) > 0.0:
        if critic_agent is None:
            raise ValueError("critic_agent is required when qsearch_radius is positive")
        qsearch_policy = FixedLocalCriticQSearchPolicy(
            actor_agent=rollout_agent,
            critic_agent=critic_agent,
            num_actions=int(qsearch_num_actions),
            margin=float(qsearch_margin),
            search_radius=float(qsearch_radius),
        )
        rollout_agent = qsearch_policy
    rows = rollout_pendulum_grid_vectorized(
        rollout_agent,
        theta,
        velocity,
        detector,
        reliability,
        horizon=200,
    )
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    tasks = np.asarray([row["task_success"] for row in rows], dtype=np.float64)
    result = {
        "near_reference_eps": float((returns >= reference_returns - 5.0).mean()),
        "task_success": float(tasks.mean()),
        "strict_beats_reference": float((returns > reference_returns).mean()),
        "mean_return": float(returns.mean()),
    }
    if qsearch_policy is not None:
        result.update(
            {
                f"qsearch_{key}": value
                for key, value in qsearch_policy.selection_metrics().items()
            }
        )
    return result


def prioritized_uniform_initial_states(
    agent: SACAgent,
    episodes: int,
    candidate_multiplier: int,
    priority_fraction: float,
    velocity_limit: float,
    rng: np.random.Generator,
    detector: UprightDetector,
    reliability: ReliabilityConfig,
    dp_solution: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Choose DAgger starts by a coordinate-free rollout-regret ranking."""
    candidate_count = max(int(episodes), int(episodes) * int(candidate_multiplier))
    theta = rng.uniform(-math.pi, math.pi, candidate_count)
    velocity = rng.uniform(-float(velocity_limit), float(velocity_limit), candidate_count)
    actor_rows = rollout_pendulum_grid_vectorized(
        agent,
        theta,
        velocity,
        detector,
        reliability,
        horizon=200,
    )
    actor_returns = np.asarray([row["return"] for row in actor_rows], dtype=np.float64)
    reference_returns = validation_reference_returns(
        theta,
        velocity,
        detector,
        reliability,
        dp_solution,
    )
    regret = reference_returns - actor_returns
    priority_count = min(
        int(episodes),
        int(round(int(episodes) * float(priority_fraction))),
    )
    ranked = np.argsort(-regret, kind="stable")
    priority_indices = ranked[:priority_count]
    remaining = ranked[priority_count:]
    uniform_count = int(episodes) - priority_count
    if uniform_count:
        uniform_indices = rng.choice(remaining, size=uniform_count, replace=False)
        selected = np.concatenate([priority_indices, uniform_indices])
    else:
        selected = priority_indices
    selected = selected[rng.permutation(len(selected))]
    metrics = {
        "priority_candidate_count": float(candidate_count),
        "priority_selected_count": float(priority_count),
        "uniform_selected_count": float(uniform_count),
        "candidate_regret_mean": float(regret.mean()),
        "candidate_regret_max": float(regret.max()),
        "selected_regret_mean": float(regret[selected].mean()),
        "selected_regret_min": float(regret[selected].min()),
        "candidate_near_reference_rate": float((regret <= 5.0).mean()),
        "selected_near_reference_rate": float((regret[selected] <= 5.0).mean()),
        "reference_rollout_queries": float(2 * candidate_count),
    }
    return theta[selected], velocity[selected], metrics


def copy_actor_state(agent: SACAgent) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in agent.actor.state_dict().items()}


def selection_key(row: dict[str, float], targeted: bool = False) -> tuple[float, ...]:
    if targeted:
        return (
            row["near_reference_eps"],
            row["targeted_near_reference_eps"],
            row["task_success"],
            row["targeted_mean_return"],
            row["mean_return"],
        )
    return row["near_reference_eps"], row["task_success"], row["mean_return"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    actor_agent, config, actor_payload = load_agent_from_run(
        Path(args.dagger_run), device=device, load_optimizers=False
    )
    rl_agent, _rl_config, rl_payload = load_agent_from_run(
        Path(args.rl_run), device=device, load_optimizers=False
    )
    hard_teacher_agent = None
    if args.hard_teacher_run:
        hard_teacher_agent, _hard_teacher_config, _hard_teacher_payload = (
            load_agent_from_run(
                Path(args.hard_teacher_run), device=device, load_optimizers=False
            )
        )
    config.seed = int(args.seed)
    config.sac.device = device
    config.to_json(run_dir / "config.json")
    for critic in rl_agent.q_networks:
        critic.requires_grad_(False)
        critic.eval()

    rng = np.random.default_rng(int(args.seed) + 44_711)
    dp_solution = Path(args.dp_solution)
    reference = PendulumReferenceGuidance(
        "best", dp_solution_path=str(dp_solution), horizon=200
    )
    static_obs, static_sources = sample_training_states(
        int(args.static_size),
        rng,
        float(args.broad_fraction),
        float(args.reset_support_fraction),
        float(args.hard120_fraction),
        float(args.near_down_fraction),
        float(args.broad_velocity_limit),
        float(args.reset_velocity_limit),
    )
    if str(args.static_target_policy) == "source_actor":
        static_reference = np.asarray(
            actor_agent.act_batch(static_obs, deterministic=True), dtype=np.float32
        ).reshape(-1, 1)
    else:
        static_reference = reference_actions_batched(
            reference, static_obs, batch_size=int(args.batch_size)
        )
    train_targets, train_selected, static_rl_metrics = q_regularized_targets(
        rl_agent,
        static_obs,
        static_reference,
        num_actions=int(args.num_actions),
        margin=float(args.rl_margin),
        blend=float(args.rl_blend),
        max_target_shift=float(args.max_target_shift),
        batch_size=int(args.batch_size),
    )
    train_obs = static_obs
    time_conditioned = bool(args.time_conditioned_student)
    new_student = int(args.student_actor_hidden_dim) > 0
    if time_conditioned:
        actor_agent = initialize_time_conditioned_actor(actor_agent, config, device)
        static_remaining = rng.integers(1, 201, size=len(static_obs), endpoint=False)
        train_obs = np.concatenate(
            [
                static_obs,
                (static_remaining.astype(np.float32) / 200.0).reshape(-1, 1),
            ],
            axis=1,
        )
        config.to_json(run_dir / "config.json")
    elif new_student:
        config.sac.simba_backbone = True
        config.sac.simba_actor_hidden_dim = int(args.student_actor_hidden_dim)
        config.sac.simba_actor_blocks = int(args.student_actor_blocks)
        actor_agent = SACAgent(
            3,
            np.asarray([-2.0], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
            config.sac,
            device=device,
        )
        if actor_agent.obs_rms is not None:
            actor_agent.obs_rms.update(static_obs)
        config.to_json(run_dir / "config.json")

    reliability = config.reliability
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    validation_theta, validation_velocity = midpoint_grid(
        int(args.validation_theta_bins), int(args.validation_velocity_bins)
    )
    reference_returns = validation_reference_returns(
        validation_theta,
        validation_velocity,
        detector,
        reliability,
        dp_solution,
    )
    targeted_validation = int(args.targeted_validation_size) > 0
    if targeted_validation:
        targeted_rng = np.random.default_rng(9_700_000 + int(args.seed))
        if str(args.targeted_validation_mode) == "reset_uniform":
            targeted_theta = targeted_rng.uniform(
                -math.pi, math.pi, int(args.targeted_validation_size)
            )
            targeted_velocity = targeted_rng.uniform(
                -float(args.reset_velocity_limit),
                float(args.reset_velocity_limit),
                int(args.targeted_validation_size),
            )
        else:
            targeted_theta, targeted_velocity, _targeted_sources = (
                sample_failure_mixture_initial_states(
                int(args.targeted_validation_size),
                targeted_rng,
                hard_fraction=float(args.failure_hard_fraction),
                near_down_fraction=float(args.failure_near_down_fraction),
                wrap_fraction=float(args.failure_wrap_fraction),
                tight_failure_bands=bool(args.tight_failure_bands),
            )
            )
        targeted_reference_returns = validation_reference_returns(
            targeted_theta,
            targeted_velocity,
            detector,
            reliability,
            dp_solution,
        )
    else:
        targeted_theta = np.empty(0, dtype=np.float64)
        targeted_velocity = np.empty(0, dtype=np.float64)
        targeted_reference_returns = np.empty(0, dtype=np.float64)
    initial_validation = evaluate_validation(
        actor_agent,
        validation_theta,
        validation_velocity,
        reference_returns,
        detector,
        reliability,
        time_conditioned=time_conditioned,
        critic_agent=rl_agent,
        qsearch_radius=float(args.validation_qsearch_radius),
        qsearch_num_actions=int(args.validation_qsearch_num_actions),
        qsearch_margin=float(args.validation_qsearch_margin),
    )
    if targeted_validation:
        targeted_result = evaluate_validation(
            actor_agent,
            targeted_theta,
            targeted_velocity,
            targeted_reference_returns,
            detector,
            reliability,
            time_conditioned=time_conditioned,
            critic_agent=rl_agent,
            qsearch_radius=float(args.validation_qsearch_radius),
            qsearch_num_actions=int(args.validation_qsearch_num_actions),
            qsearch_margin=float(args.validation_qsearch_margin),
        )
        initial_validation.update(
            {f"targeted_{key}": value for key, value in targeted_result.items()}
        )
    initial_validation.update({"epoch": 0.0, "dagger_round": 0.0})
    validation_rows: list[dict[str, float]] = [initial_validation]
    best_row = dict(initial_validation)
    best_actor_state = copy_actor_state(actor_agent)

    if str(args.trainable_actor) == "time_input":
        actor_agent.actor.requires_grad_(False)
        time_weight = actor_agent.actor.backbone.embedder.w.linear.weight
        time_weight.requires_grad_(True)
        time_mask = torch.zeros_like(time_weight)
        time_mask[:, 3] = 1.0
        time_weight.register_hook(lambda gradient: gradient * time_mask)
        parameters = [time_weight]
        # Projection would renormalize the frozen source columns after every edit.
        actor_agent.cfg.simba_weight_projection = False
        config.sac.simba_weight_projection = False
        config.to_json(run_dir / "config.json")
    else:
        parameters = configure_trainable_actor(actor_agent, str(args.trainable_actor))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    training_rows: list[dict[str, float]] = []
    collection_rows: list[dict[str, float]] = []
    rl_target_rows: list[dict[str, float]] = [
        {"dagger_round": 0.0, **static_rl_metrics}
    ]
    total_epoch = 0
    for dagger_round in range(1, int(args.dagger_rounds) + 1):
        if str(args.dagger_initial_mode) == "hard120_edge":
            dagger_obs, dagger_reference, collection = collect_hard_dagger_dataset(
                actor_agent,
                reference,
                episodes=int(args.dagger_episodes),
                rng=np.random.default_rng(
                    700_000 + 10_000 * int(args.seed) + 1_000 * dagger_round
                ),
                velocity_low=float(args.hard_dagger_velocity_low),
                velocity_high=float(args.reset_velocity_limit),
            )
        elif str(args.dagger_initial_mode) == "priority_uniform":
            priority_rng = np.random.default_rng(
                700_000 + 10_000 * int(args.seed) + 1_000 * dagger_round
            )
            initial_theta, initial_velocity, priority_metrics = (
                prioritized_uniform_initial_states(
                    actor_agent,
                    episodes=int(args.dagger_episodes),
                    candidate_multiplier=int(args.priority_candidate_multiplier),
                    priority_fraction=float(args.priority_fraction),
                    velocity_limit=float(args.reset_velocity_limit),
                    rng=priority_rng,
                    detector=detector,
                    reliability=reliability,
                    dp_solution=dp_solution,
                )
            )
            dagger_obs, dagger_reference, collection = collect_dagger_from_initial_states(
                actor_agent,
                reference,
                initial_theta,
                initial_velocity,
            )
            collection.update(priority_metrics)
            collection["priority_uniform_initials"] = 1.0
            demonstration_episodes = int(
                args.priority_expert_demonstrations_per_round
            )
            if demonstration_episodes > 0:
                demonstration_obs, demonstration_reference, demonstration_metrics = (
                    collect_reference_demonstrations_from_initial_states(
                        reference,
                        initial_theta[:demonstration_episodes],
                        initial_velocity[:demonstration_episodes],
                    )
                )
                dagger_obs = np.concatenate(
                    [dagger_obs, demonstration_obs], axis=0
                )
                dagger_reference = np.concatenate(
                    [dagger_reference, demonstration_reference], axis=0
                )
                collection.update(demonstration_metrics)
                collection["samples"] = float(len(dagger_obs))
                collection["learner_dagger_samples"] = float(
                    len(dagger_obs) - len(demonstration_obs)
                )
                collection["aggregate_executed_reference_fraction"] = float(
                    len(demonstration_obs) / len(dagger_obs)
                )
        elif str(args.dagger_initial_mode) == "failure_mixture":
            dagger_obs, dagger_reference, collection = (
                collect_failure_mixture_dagger_dataset(
                    actor_agent,
                    reference,
                    episodes=int(args.dagger_episodes),
                    rng=np.random.default_rng(
                        700_000 + 10_000 * int(args.seed) + 1_000 * dagger_round
                    ),
                    hard_fraction=float(args.failure_hard_fraction),
                    near_down_fraction=float(args.failure_near_down_fraction),
                    wrap_fraction=float(args.failure_wrap_fraction),
                    tight_failure_bands=bool(args.tight_failure_bands),
                    hard_teacher=(
                        hard_teacher_agent if bool(args.failure_hard_teacher) else None
                    ),
                    hard_teacher_prefix_steps=int(args.hard_teacher_prefix_steps),
                    save_prefix_steps=int(args.dagger_save_prefix_steps),
                    expert_beta=float(args.failure_expert_beta),
                    time_conditioned=time_conditioned,
                )
            )
        else:
            dagger_obs, dagger_reference, collection = collect_dagger_dataset(
                actor_agent,
                reference,
                env_id="Pendulum-v1",
                episodes=int(args.dagger_episodes),
                seed_base=700_000 + 10_000 * int(args.seed) + 1_000 * dagger_round,
                max_episode_steps=200,
                deterministic=True,
                expert_beta=0.0,
                vectorized=True,
            )
        dagger_targets, dagger_selected, dagger_rl_metrics = q_regularized_targets(
            rl_agent,
            dagger_obs[:, :3] if time_conditioned else dagger_obs,
            dagger_reference,
            num_actions=int(args.num_actions),
            margin=float(args.rl_margin),
            blend=float(args.rl_blend),
            max_target_shift=float(args.max_target_shift),
            batch_size=int(args.batch_size),
        )
        if bool(args.weight_all_dagger_samples):
            dagger_selected = np.ones(len(dagger_selected), dtype=bool)
            dagger_rl_metrics["all_dagger_samples_weighted"] = 1.0
            dagger_rl_metrics["weighted_samples"] = float(len(dagger_selected))
        elif float(args.dagger_disagreement_threshold) > 0.0:
            current_actions = np.asarray(
                actor_agent.act_batch(dagger_obs, deterministic=True), dtype=np.float32
            ).reshape(-1, 1)
            disagreement = np.abs(current_actions - dagger_targets).reshape(-1)
            dagger_selected = disagreement > float(args.dagger_disagreement_threshold)
            dagger_rl_metrics["all_dagger_samples_weighted"] = 0.0
            dagger_rl_metrics["disagreement_weighting"] = 1.0
            dagger_rl_metrics["disagreement_abs_mean"] = float(disagreement.mean())
            dagger_rl_metrics["disagreement_abs_max"] = float(disagreement.max())
            dagger_rl_metrics["selected"] = float(dagger_selected.sum())
            dagger_rl_metrics["selected_fraction"] = float(dagger_selected.mean())
            dagger_rl_metrics["weighted_samples"] = float(dagger_selected.sum())
        else:
            dagger_rl_metrics["all_dagger_samples_weighted"] = 0.0
            dagger_rl_metrics["disagreement_weighting"] = 0.0
            dagger_rl_metrics["weighted_samples"] = float(dagger_selected.sum())
        train_obs = np.concatenate([train_obs, dagger_obs], axis=0)
        train_targets = np.concatenate([train_targets, dagger_targets], axis=0)
        train_selected = np.concatenate([train_selected, dagger_selected], axis=0)
        if new_student and actor_agent.obs_rms is not None:
            actor_agent.obs_rms.update(dagger_obs)
        collection_rows.append({"dagger_round": float(dagger_round), **collection})
        rl_target_rows.append({"dagger_round": float(dagger_round), **dagger_rl_metrics})

        for local_epoch in range(1, int(args.epochs_per_round) + 1):
            total_epoch += 1
            train_metrics = train_actor_epoch(
                actor_agent,
                optimizer,
                train_obs,
                train_targets,
                train_selected,
                batch_size=int(args.batch_size),
                selected_weight=float(args.selected_weight),
                rng=rng,
            )
            train_metrics.update(
                model_rl_updates(
                    actor_agent,
                    optimizer,
                    train_obs,
                    train_targets,
                    rng=rng,
                    updates=int(args.model_rl_updates_per_epoch),
                    batch_size=int(args.model_rl_batch_size),
                    horizon=int(args.model_rl_horizon),
                    velocity_low=float(args.hard_dagger_velocity_low),
                    velocity_high=float(args.reset_velocity_limit),
                    rl_weight=float(args.model_rl_weight),
                    bc_weight=float(args.model_rl_bc_weight),
                    trainable_parameters=parameters,
                    tight_failure_bands=bool(args.tight_failure_bands),
                    time_conditioned=time_conditioned,
                )
            )
            train_metrics.update(
                {
                    "epoch": float(total_epoch),
                    "dagger_round": float(dagger_round),
                    "local_epoch": float(local_epoch),
                    "dataset_size": float(len(train_obs)),
                    "rl_selected_fraction": float(train_selected.mean()),
                }
            )
            training_rows.append(train_metrics)
            should_validate = (
                total_epoch % int(args.validation_every_epochs) == 0
                or local_epoch == int(args.epochs_per_round)
            )
            if not should_validate:
                continue
            validation = evaluate_validation(
                actor_agent,
                validation_theta,
                validation_velocity,
                reference_returns,
                detector,
                reliability,
                time_conditioned=time_conditioned,
                critic_agent=rl_agent,
                qsearch_radius=float(args.validation_qsearch_radius),
                qsearch_num_actions=int(args.validation_qsearch_num_actions),
                qsearch_margin=float(args.validation_qsearch_margin),
            )
            if targeted_validation:
                targeted_result = evaluate_validation(
                    actor_agent,
                    targeted_theta,
                    targeted_velocity,
                    targeted_reference_returns,
                    detector,
                    reliability,
                    time_conditioned=time_conditioned,
                    critic_agent=rl_agent,
                    qsearch_radius=float(args.validation_qsearch_radius),
                    qsearch_num_actions=int(args.validation_qsearch_num_actions),
                    qsearch_margin=float(args.validation_qsearch_margin),
                )
                validation.update(
                    {f"targeted_{key}": value for key, value in targeted_result.items()}
                )
            validation.update(
                {"epoch": float(total_epoch), "dagger_round": float(dagger_round)}
            )
            validation_rows.append(validation)
            if selection_key(validation, targeted=targeted_validation) > selection_key(
                best_row, targeted=targeted_validation
            ):
                best_row = dict(validation)
                best_actor_state = copy_actor_state(actor_agent)

    actor_agent.actor.load_state_dict(best_actor_state)
    checkpoint = run_dir / "checkpoints" / "final.pt"
    actor_agent.save_checkpoint(
        checkpoint,
        extra={
            "global_step": 0,
            "qregularized_dagger": True,
            "single_actor_inference": float(args.validation_qsearch_radius) <= 0.0,
            "uses_fixed_pure_rl_critic_at_inference": (
                float(args.validation_qsearch_radius) > 0.0
            ),
            "inference_qsearch_radius": float(args.validation_qsearch_radius),
            "inference_qsearch_num_actions": int(args.validation_qsearch_num_actions),
            "inference_qsearch_margin": float(args.validation_qsearch_margin),
            "dagger_source_run": str(args.dagger_run),
            "rl_critic_source_run": str(args.rl_run),
            "dagger_source_extra": actor_payload.get("extra", {}),
            "rl_source_extra": rl_payload.get("extra", {}),
            "selected_epoch": int(best_row["epoch"]),
            "rl_margin": float(args.rl_margin),
            "rl_blend": float(args.rl_blend),
            "max_target_shift": float(args.max_target_shift),
            "hard_teacher_source_run": (
                str(args.hard_teacher_run) if args.hard_teacher_run else None
            ),
            "failure_hard_teacher": bool(args.failure_hard_teacher),
            "hard_teacher_prefix_steps": int(args.hard_teacher_prefix_steps),
            "dagger_save_prefix_steps": int(args.dagger_save_prefix_steps),
            "failure_expert_beta": float(args.failure_expert_beta),
            "tight_failure_bands": bool(args.tight_failure_bands),
            "static_target_policy": str(args.static_target_policy),
            "student_actor_hidden_dim": int(args.student_actor_hidden_dim),
            "student_actor_blocks": int(args.student_actor_blocks),
            "time_conditioned_student": bool(args.time_conditioned_student),
            "dagger_disagreement_threshold": float(args.dagger_disagreement_threshold),
            "weight_all_dagger_samples": bool(args.weight_all_dagger_samples),
        },
    )
    write_csv(run_dir / "training_metrics.csv", training_rows)
    write_csv(run_dir / "validation_metrics.csv", validation_rows)
    write_csv(run_dir / "dagger_collection_metrics.csv", collection_rows)
    write_csv(run_dir / "rl_target_metrics.csv", rl_target_rows)
    source_counts = {
        str(name): int(np.sum(static_sources == name)) for name in np.unique(static_sources)
    }
    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "device": device,
        "seed": int(args.seed),
        "single_actor_inference": float(args.validation_qsearch_radius) <= 0.0,
        "uses_fixed_pure_rl_critic_at_inference": (
            float(args.validation_qsearch_radius) > 0.0
        ),
        "uses_reference_labels": True,
        "uses_fixed_pure_rl_critics_during_training": bool(
            float(args.rl_blend) > 0.0 or float(args.selected_weight) != 1.0
        ),
        "pure_rl_critic_influence": (
            "target_shift_and_sample_weight"
            if float(args.rl_blend) > 0.0 and float(args.selected_weight) != 1.0
            else "target_shift"
            if float(args.rl_blend) > 0.0
            else "sample_weight"
            if float(args.selected_weight) != 1.0
            else "diagnostics_only"
        ),
        "dagger_source_run": str(args.dagger_run),
        "rl_critic_source_run": str(args.rl_run),
        "hard_teacher_source_run": (
            str(args.hard_teacher_run) if args.hard_teacher_run else None
        ),
        "failure_hard_teacher": bool(args.failure_hard_teacher),
        "hard_teacher_prefix_steps": int(args.hard_teacher_prefix_steps),
        "dagger_save_prefix_steps": int(args.dagger_save_prefix_steps),
        "failure_expert_beta": float(args.failure_expert_beta),
        "tight_failure_bands": bool(args.tight_failure_bands),
        "static_size": int(args.static_size),
        "static_target_policy": str(args.static_target_policy),
        "static_source_counts": source_counts,
        "final_dataset_size": int(len(train_obs)),
        "dagger_rounds": int(args.dagger_rounds),
        "dagger_episodes_per_round": int(args.dagger_episodes),
        "dagger_initial_mode": str(args.dagger_initial_mode),
        "priority_candidate_multiplier": int(args.priority_candidate_multiplier),
        "priority_fraction": float(args.priority_fraction),
        "priority_expert_demonstrations_per_round": int(
            args.priority_expert_demonstrations_per_round
        ),
        "epochs_per_round": int(args.epochs_per_round),
        "actor_optimizer_steps": int(
            sum(math.ceil(row["dataset_size"] / int(args.batch_size)) for row in training_rows)
        ),
        "trainable_actor": str(args.trainable_actor),
        "student_actor_hidden_dim": int(args.student_actor_hidden_dim),
        "student_actor_blocks": int(args.student_actor_blocks),
        "time_conditioned_student": bool(args.time_conditioned_student),
        "validation_every_epochs": int(args.validation_every_epochs),
        "inference_qsearch": {
            "enabled": float(args.validation_qsearch_radius) > 0.0,
            "radius": float(args.validation_qsearch_radius),
            "num_actions": int(args.validation_qsearch_num_actions),
            "margin": float(args.validation_qsearch_margin),
            "filter": "unanimous_advantage",
        },
        "learning_rate": float(args.lr),
        "rl_margin": float(args.rl_margin),
        "rl_blend": float(args.rl_blend),
        "max_target_shift": float(args.max_target_shift),
        "selected_weight": float(args.selected_weight),
        "weight_all_dagger_samples": bool(args.weight_all_dagger_samples),
        "dagger_disagreement_threshold": float(args.dagger_disagreement_threshold),
        "model_rl": {
            "updates_per_epoch": int(args.model_rl_updates_per_epoch),
            "batch_size": int(args.model_rl_batch_size),
            "horizon": int(args.model_rl_horizon),
            "rl_weight": float(args.model_rl_weight),
            "bc_weight": float(args.model_rl_bc_weight),
        },
        "validation_grid": {
            "theta_bins": int(args.validation_theta_bins),
            "velocity_bins": int(args.validation_velocity_bins),
            "points": int(len(validation_theta)),
        },
        "targeted_validation": {
            "enabled": bool(targeted_validation),
            "mode": str(args.targeted_validation_mode),
            "points": int(len(targeted_theta)),
            "failure_hard_fraction": float(args.failure_hard_fraction),
            "failure_near_down_fraction": float(args.failure_near_down_fraction),
            "failure_wrap_fraction": float(args.failure_wrap_fraction),
        },
        "selection_rule": (
            [
                "near_reference_eps",
                "targeted_near_reference_eps",
                "task_success",
                "targeted_mean_return",
                "mean_return",
            ]
            if targeted_validation
            else ["near_reference_eps", "task_success", "mean_return"]
        ),
        "initial_validation": initial_validation,
        "selected_validation": best_row,
        "selected_epoch": int(best_row["epoch"]),
        "static_rl_targets": static_rl_metrics,
        "dagger_collection": collection_rows,
        "dagger_rl_targets": rl_target_rows[1:],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
