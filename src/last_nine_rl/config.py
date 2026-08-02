from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass
class EnvConfig:
    env_id: str = "Pendulum-v1"
    max_episode_steps: int | None = None
    pendulum_hard_reset_prob: float = 0.0
    pendulum_hard_reset_final_prob: float = 0.0
    pendulum_hard_reset_decay_steps: int = 0
    pendulum_hard_reset_start_step: int = 0
    pendulum_hard_reset_abs_theta_low: float = 2.0943951023931953
    pendulum_hard_reset_abs_theta_high: float = 2.356194490192345
    pendulum_hard_reset_velocity_limit: float = 1.0
    pendulum_failure_reset_prob: float = 0.0
    pendulum_failure_curriculum_start_step: int = 20_000
    pendulum_failure_curriculum_refresh_interval_steps: int = 20_000
    pendulum_failure_curriculum_candidate_count: int = 32
    pendulum_failure_curriculum_worst_fraction: float = 0.25
    pendulum_failure_curriculum_rollouts_per_candidate: int = 1
    pendulum_failure_curriculum_rollout_horizon: int = 200
    pendulum_failure_curriculum_seed_offset: int = 5_000_000


@dataclass
class SACConfig:
    total_steps: int = 100_000
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    random_action_steps: int | None = None
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_lr_final: float | None = None
    q_lr_final: float | None = None
    alpha_lr: float | None = None
    alpha_lr_final: float | None = None
    alpha_initial_value: float = 1.0
    alpha_min_value: float = 0.0
    target_entropy_scale: float = -1.0
    updates_per_step: int = 1
    policy_frequency: int = 2
    actor_updates_per_trigger: int = 0
    actor_q_aggregation: str = "min"
    actor_q_aggregation_late: str | None = None
    actor_q_aggregation_switch_step: int = 0
    target_q_aggregation: str = "min"
    target_network_frequency: int = 1
    redq_num_critics: int = 2
    redq_target_subset_size: int = 2
    update_diagnostics: bool = True
    redo_interval_updates: int = 0
    redo_dormant_threshold: float = 0.025
    swd_linear_decay_steps: int = 0
    swd_min_weight: float = 0.1
    replay_priority_mode: str = "none"
    replay_priority_alpha: float = 0.6
    replay_priority_beta_initial: float = 0.4
    replay_priority_beta_final: float = 1.0
    replay_priority_beta_anneal_steps: int = 100_000
    replay_priority_uniform_fraction: float = 0.5
    replay_priority_epsilon: float = 1e-3
    replay_priority_clip: float = 10.0
    pendulum_hard_replay_fraction: float = 0.0
    pendulum_hard_replay_final_fraction: float = 0.0
    pendulum_hard_replay_decay_steps: int = 0
    pendulum_hard_replay_start_step: int = 0
    pendulum_hard_replay_abs_theta_low: float = 2.0943951023931953
    pendulum_hard_replay_abs_theta_high: float = 2.356194490192345
    pendulum_hard_replay_velocity_limit: float = 1.0
    pendulum_model_replay_ratio: float = 0.0
    pendulum_model_replay_steps_per_step: int = 1
    pendulum_model_replay_start_step: int = 0
    pendulum_model_replay_random_action_fraction: float = 0.0
    pendulum_model_replay_abs_theta_low: float = 2.6179938779914944
    pendulum_model_replay_abs_theta_high: float = math.pi
    pendulum_model_replay_velocity_limit: float = 1.0
    pendulum_model_rollout_ratio: float = 0.0
    pendulum_model_rollout_starts_per_step: int = 1
    pendulum_model_rollout_horizon: int = 8
    pendulum_model_rollout_interval_steps: int = 1
    pendulum_model_rollout_start_step: int = 0
    pendulum_model_rollout_abs_theta_low: float = 2.6179938779914944
    pendulum_model_rollout_abs_theta_high: float = math.pi
    pendulum_model_rollout_velocity_limit: float = 1.0
    pendulum_potential_shaping_weight: float = 0.0
    pendulum_potential_shaping_start_update: int = 0
    pendulum_potential_shaping_abs_theta_low: float = 0.0
    pendulum_potential_shaping_abs_theta_high: float = math.pi
    pendulum_potential_shaping_velocity_limit: float = 0.0
    pendulum_potential_shaping_source: str = "best"
    pendulum_potential_shaping_dp_grid_path: str | None = None
    pendulum_potential_shaping_controller_grid_path: str | None = None
    pendulum_symmetry_augmentation: bool = False
    pendulum_actor_symmetry_weight: float = 0.0
    pendulum_critic_symmetry_weight: float = 0.0
    device: str = "auto"
    l2_feature_norm: bool = False
    simba_backbone: bool = False
    simba_observation_norm: bool = True
    simba_feature_norm: bool = True
    simba_input_shift: bool = True
    simba_c_shift: float = 3.0
    simba_actor_blocks: int = 1
    simba_actor_hidden_dim: int = 128
    # ``None`` preserves the historical SimbaV2 mapping to [-10, 2].  A finite
    # floor is applied *after* that mapping, so it changes only states whose
    # learned log-standard deviation would otherwise fall below the floor.
    simba_actor_log_std_floor: float | None = None
    simba_critic_blocks: int = 2
    simba_critic_hidden_dim: int = 512
    simba_block_expansion: int = 4
    simba_weight_projection: bool = False
    simba_distributional_critic: bool = False
    simba_critic_num_bins: int = 101
    simba_critic_min_v: float = -5.0
    simba_critic_max_v: float = 5.0
    simba_reward_scaling: bool = False
    simba_reward_scale_g_max: float = 5.0
    sacn_n_step: int = 1
    sacn_importance_quantile: float = 0.75
    sacn_tau_entropy: bool = True
    sacn_max_entropy_samples: int = 32
    sacn_recent_max_age_steps: int = 0
    sacn_min_horizon_ess_fraction: float = 0.0
    sacn_importance_mode: str = "density"
    sacn_non_soft_targets: bool = False
    sacn_stop_after_steps: int = 0
    sacn_target_mode: str = "all"
    sacn_horizon_lambda: float = 1.0
    reference_guidance_mode: str = "none"
    reference_guidance_policy: str = "best"
    reference_guidance_probability: float = 0.0
    reference_guidance_start_step: int = 0
    reference_guidance_dp_solution_path: str | None = None
    reference_auxiliary_mode: str = "none"
    reference_auxiliary_policy: str = "dp"
    reference_auxiliary_weight: float = 0.0
    reference_auxiliary_weight_final: float | None = None
    reference_auxiliary_decay_updates: int = 0
    reference_auxiliary_stop_update: int = 0
    reference_auxiliary_margin: float = 0.0
    reference_auxiliary_filter_start_update: int = 0
    reference_auxiliary_q_filter_mode: str = "twin_min_difference"
    reference_auxiliary_replay_normalization: str = "selected_mean"
    reference_anchor_ratio: float = 0.0
    reference_anchor_size: int = 0
    reference_anchor_velocity_limit: float = 8.0
    reference_anchor_reset_support_fraction: float = 0.6
    reference_anchor_reset_velocity_limit: float = 1.0
    sac_actor_loss_weight: float = 1.0
    sac_actor_loss_start_step: int = 0
    sac_actor_objective_mode: str = "stochastic"
    actor_mean_logit_l2_weight: float = 0.0
    actor_mean_logit_excess_threshold: float = 0.0
    sac_actor_gradient_balance_mode: str = "none"
    sac_actor_gradient_balance_min_multiplier: float = 1.0
    sac_actor_gradient_balance_max_multiplier: float = 1.0
    sac_actor_gradient_conflict_mode: str = "none"
    sac_actor_filter_mode: str = "none"
    sac_actor_filter_margin: float = 0.0
    reference_critic_mode: str = "none"
    reference_critic_policy: str = "best"
    reference_critic_weight: float = 0.0
    reference_critic_margin: float = 0.0
    reference_prior_mode: str = "none"
    reference_prior_policy: str = "best"
    reference_prior_ratio: float = 0.0
    reference_prior_source: str = "online_one_step"
    reference_prior_dataset_steps: int = 0
    reference_prior_dataset_seed_offset: int = 1_000_000
    actor_init_checkpoint_path: str | None = None
    actor_init_load_obs_rms: bool = True
    actor_update_start_step: int = 0
    actor_update_stop_step: int = 0
    uniform_exploration_initial_probability: float = 0.0
    uniform_exploration_final_probability: float = 0.0
    uniform_exploration_decay_steps: int = 0
    uniform_exploration_start_step: int = 0
    obs_rms_update_enabled: bool = True
    cql_alpha: float = 0.0
    cql_temperature: float = 1.0
    cql_num_random_actions: int = 10
    cql_interval_updates: int = 1
    cql_include_policy_actions: bool = True
    critic_search_actor_weight: float = 0.0
    critic_search_num_actions: int = 41
    critic_search_margin: float = 0.0
    critic_search_start_update: int = 0
    critic_search_filter_mode: str = "clipped_value"
    critic_search_actor_loss_type: str = "mse"
    self_imitation_weight: float = 0.0
    self_imitation_loss_type: str = "mse"
    self_imitation_start_step: int = 0
    self_imitation_temperature: float = 1.0
    self_imitation_margin: float = 0.0
    self_imitation_max_weight: float = 20.0


def reference_auxiliary_loss_can_be_active(config: SACConfig) -> bool:
    """Whether the configured BC schedule has a positive weight at any point."""

    final_weight = (
        float(config.reference_auxiliary_weight)
        if config.reference_auxiliary_weight_final is None
        else float(config.reference_auxiliary_weight_final)
    )
    return (
        config.reference_auxiliary_mode != "none"
        and max(float(config.reference_auxiliary_weight), final_weight) > 0.0
    )


def needs_actor_reference_actions(config: SACConfig) -> bool:
    """Whether actor training needs teacher actions for BC and/or SAC filtering."""

    return (
        reference_auxiliary_loss_can_be_active(config)
        or config.sac_actor_filter_mode != "none"
    )


@dataclass
class EvalConfig:
    every_steps: int = 5_000
    episodes: int = 10
    deterministic: bool = True
    seed_base: int = 100_000
    seeds: tuple[int, ...] | None = None


@dataclass
class ReliabilityConfig:
    collapse_return_threshold: float = -1_000.0
    success_near_upright_fraction_threshold: float = 0.8
    success_max_not_near_upright_streak: int = 50
    near_upright_cos_threshold: float = 0.95
    near_upright_abs_velocity_threshold: float = 1.0
    dormant_relative_threshold: float = 0.1


@dataclass
class TelemetryConfig:
    run_root: str = "runs"
    log_interval_steps: int = 1_000
    replay_inspection_interval_steps: int = 5_000
    diagnostics_interval_steps: int = 5_000
    checkpoint_interval_steps: int = 0
    tensorboard: bool = True
    write_eval_returns_csv: bool = True
    overwrite: bool = False
    save_replay: bool = False
    save_model: bool = True


@dataclass
class ExperimentConfig:
    name: str = "week1_pendulum_sac"
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        env = EnvConfig(**raw.get("env", {}))

        sac_raw = dict(raw.get("sac", {}))
        legacy_network_variant = sac_raw.pop("network_variant", None)
        if legacy_network_variant is not None:
            if legacy_network_variant != "l2_feature_norm":
                raise ValueError(f"unsupported legacy sac.network_variant: {legacy_network_variant!r}")
            sac_raw["l2_feature_norm"] = True
        sac = SACConfig(**sac_raw)

        eval_raw = dict(raw.get("eval", {}))
        if eval_raw.get("seeds") is not None:
            eval_raw["seeds"] = tuple(int(v) for v in eval_raw["seeds"])
        eval_cfg = EvalConfig(**eval_raw)
        if eval_cfg.seeds is not None:
            eval_cfg.episodes = len(eval_cfg.seeds)

        reliability_keys = {item.name for item in fields(ReliabilityConfig)}
        reliability = ReliabilityConfig(
            **{key: value for key, value in raw.get("reliability", {}).items() if key in reliability_keys}
        )

        telemetry = TelemetryConfig(**raw.get("telemetry", {}))
        return cls(
            name=raw.get("name", "week1_pendulum_sac"),
            seed=int(raw.get("seed", 0)),
            env=env,
            sac=sac,
            eval=eval_cfg,
            reliability=reliability,
            telemetry=telemetry,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def validate(self) -> None:
        errors: list[str] = []

        if not self.name:
            errors.append("name must be non-empty")
        if not self.env.env_id:
            errors.append("env.env_id must be non-empty")
        if self.env.max_episode_steps is not None and self.env.max_episode_steps <= 0:
            errors.append("env.max_episode_steps must be positive when set")
        if not (0.0 <= self.env.pendulum_hard_reset_prob <= 1.0):
            errors.append("env.pendulum_hard_reset_prob must be in [0, 1]")
        if not (0.0 <= self.env.pendulum_hard_reset_final_prob <= 1.0):
            errors.append("env.pendulum_hard_reset_final_prob must be in [0, 1]")
        _require_nonnegative(
            "env.pendulum_hard_reset_decay_steps",
            self.env.pendulum_hard_reset_decay_steps,
            errors,
        )
        _require_nonnegative(
            "env.pendulum_hard_reset_start_step",
            self.env.pendulum_hard_reset_start_step,
            errors,
        )
        if not (0.0 <= self.env.pendulum_hard_reset_abs_theta_low <= math.pi):
            errors.append("env.pendulum_hard_reset_abs_theta_low must be in [0, pi]")
        if not (0.0 <= self.env.pendulum_hard_reset_abs_theta_high <= math.pi):
            errors.append("env.pendulum_hard_reset_abs_theta_high must be in [0, pi]")
        if self.env.pendulum_hard_reset_abs_theta_low > self.env.pendulum_hard_reset_abs_theta_high:
            errors.append(
                "env.pendulum_hard_reset_abs_theta_low must be <= env.pendulum_hard_reset_abs_theta_high"
            )
        _require_nonnegative(
            "env.pendulum_hard_reset_velocity_limit",
            self.env.pendulum_hard_reset_velocity_limit,
            errors,
        )
        if (
            max(self.env.pendulum_hard_reset_prob, self.env.pendulum_hard_reset_final_prob) > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("env.pendulum_hard_reset_prob currently supports only Pendulum environments")
        if self.env.pendulum_hard_reset_final_prob > 0.0 and self.env.pendulum_hard_reset_prob <= 0.0:
            errors.append(
                "env.pendulum_hard_reset_prob must be > 0 when env.pendulum_hard_reset_final_prob is > 0"
            )
        if not (0.0 <= self.env.pendulum_failure_reset_prob <= 1.0):
            errors.append("env.pendulum_failure_reset_prob must be in [0, 1]")
        _require_nonnegative(
            "env.pendulum_failure_curriculum_start_step",
            self.env.pendulum_failure_curriculum_start_step,
            errors,
        )
        _require_positive(
            "env.pendulum_failure_curriculum_refresh_interval_steps",
            self.env.pendulum_failure_curriculum_refresh_interval_steps,
            errors,
        )
        _require_positive(
            "env.pendulum_failure_curriculum_candidate_count",
            self.env.pendulum_failure_curriculum_candidate_count,
            errors,
        )
        if not (0.0 < self.env.pendulum_failure_curriculum_worst_fraction <= 1.0):
            errors.append("env.pendulum_failure_curriculum_worst_fraction must be in (0, 1]")
        _require_positive(
            "env.pendulum_failure_curriculum_rollouts_per_candidate",
            self.env.pendulum_failure_curriculum_rollouts_per_candidate,
            errors,
        )
        _require_positive(
            "env.pendulum_failure_curriculum_rollout_horizon",
            self.env.pendulum_failure_curriculum_rollout_horizon,
            errors,
        )
        _require_nonnegative(
            "env.pendulum_failure_curriculum_seed_offset",
            self.env.pendulum_failure_curriculum_seed_offset,
            errors,
        )
        if (
            self.env.pendulum_failure_reset_prob > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append(
                "env.pendulum_failure_reset_prob currently supports only Pendulum environments"
            )

        _require_positive("sac.total_steps", self.sac.total_steps, errors)
        _require_positive("sac.buffer_size", self.sac.buffer_size, errors)
        _require_nonnegative("sac.learning_starts", self.sac.learning_starts, errors)
        _require_positive("sac.batch_size", self.sac.batch_size, errors)
        _require_positive("sac.updates_per_step", self.sac.updates_per_step, errors)
        _require_positive("sac.policy_frequency", self.sac.policy_frequency, errors)
        _require_nonnegative("sac.actor_updates_per_trigger", self.sac.actor_updates_per_trigger, errors)
        if self.sac.actor_q_aggregation not in {"min", "mean", "max"}:
            errors.append("sac.actor_q_aggregation must be one of min, mean, max")
        if self.sac.actor_q_aggregation_late is not None and self.sac.actor_q_aggregation_late not in {
            "min",
            "mean",
            "max",
        }:
            errors.append("sac.actor_q_aggregation_late must be one of min, mean, max when set")
        _require_nonnegative("sac.actor_q_aggregation_switch_step", self.sac.actor_q_aggregation_switch_step, errors)
        if self.sac.target_q_aggregation not in {"min", "mean", "max"}:
            errors.append("sac.target_q_aggregation must be one of min, mean, max")
        _require_positive("sac.target_network_frequency", self.sac.target_network_frequency, errors)
        if self.sac.redq_num_critics < 2:
            errors.append("sac.redq_num_critics must be >= 2")
        if self.sac.redq_target_subset_size < 2:
            errors.append("sac.redq_target_subset_size must be >= 2")
        if self.sac.redq_target_subset_size > self.sac.redq_num_critics:
            errors.append("sac.redq_target_subset_size must be <= sac.redq_num_critics")
        _require_nonnegative("sac.redo_interval_updates", self.sac.redo_interval_updates, errors)
        if not (0.0 <= self.sac.redo_dormant_threshold <= 1.0):
            errors.append("sac.redo_dormant_threshold must be in [0, 1]")
        if not (0.0 <= self.sac.swd_min_weight <= 1.0):
            errors.append("sac.swd_min_weight must be in [0, 1]")
        if self.sac.replay_priority_mode not in {
            "none",
            "bellman_residual",
            "critic_disagreement",
            "max",
        }:
            errors.append(
                "sac.replay_priority_mode must be one of none, bellman_residual, "
                "critic_disagreement, max"
            )
        _require_nonnegative("sac.replay_priority_alpha", self.sac.replay_priority_alpha, errors)
        if not (0.0 <= self.sac.replay_priority_beta_initial <= 1.0):
            errors.append("sac.replay_priority_beta_initial must be in [0, 1]")
        if not (0.0 <= self.sac.replay_priority_beta_final <= 1.0):
            errors.append("sac.replay_priority_beta_final must be in [0, 1]")
        _require_nonnegative(
            "sac.replay_priority_beta_anneal_steps",
            self.sac.replay_priority_beta_anneal_steps,
            errors,
        )
        if not (0.0 <= self.sac.replay_priority_uniform_fraction <= 1.0):
            errors.append("sac.replay_priority_uniform_fraction must be in [0, 1]")
        _require_positive("sac.replay_priority_epsilon", self.sac.replay_priority_epsilon, errors)
        _require_positive("sac.replay_priority_clip", self.sac.replay_priority_clip, errors)
        if self.sac.replay_priority_clip < self.sac.replay_priority_epsilon:
            errors.append("sac.replay_priority_clip must be >= sac.replay_priority_epsilon")
        if self.sac.replay_priority_mode != "none":
            if self.sac.replay_priority_uniform_fraction < 0.5:
                errors.append(
                    "sac.replay_priority_uniform_fraction must be >= 0.5 when prioritized replay is enabled"
                )
            if self.sac.swd_linear_decay_steps != 0:
                errors.append("Prioritized replay cannot be combined with SWD age-biased replay")
            if max(
                self.sac.pendulum_hard_replay_fraction,
                self.sac.pendulum_hard_replay_final_fraction,
            ) > 0.0:
                errors.append("Prioritized replay cannot be combined with hard-range replay sampling")
            if max(
                self.sac.pendulum_model_replay_ratio,
                self.sac.pendulum_model_rollout_ratio,
                self.sac.reference_prior_ratio,
            ) > 0.0:
                errors.append(
                    "Prioritized replay currently requires a single online replay source"
                )
            if self.sac.pendulum_symmetry_augmentation:
                errors.append(
                    "Prioritized replay currently cannot be combined with batch symmetry augmentation"
                )
        if not (0.0 <= self.sac.pendulum_hard_replay_fraction <= 1.0):
            errors.append("sac.pendulum_hard_replay_fraction must be in [0, 1]")
        if not (0.0 <= self.sac.pendulum_hard_replay_final_fraction <= 1.0):
            errors.append("sac.pendulum_hard_replay_final_fraction must be in [0, 1]")
        _require_nonnegative(
            "sac.pendulum_hard_replay_decay_steps",
            self.sac.pendulum_hard_replay_decay_steps,
            errors,
        )
        _require_nonnegative("sac.pendulum_hard_replay_start_step", self.sac.pendulum_hard_replay_start_step, errors)
        if not (0.0 <= self.sac.pendulum_hard_replay_abs_theta_low <= math.pi):
            errors.append("sac.pendulum_hard_replay_abs_theta_low must be in [0, pi]")
        if not (0.0 <= self.sac.pendulum_hard_replay_abs_theta_high <= math.pi):
            errors.append("sac.pendulum_hard_replay_abs_theta_high must be in [0, pi]")
        if self.sac.pendulum_hard_replay_abs_theta_low > self.sac.pendulum_hard_replay_abs_theta_high:
            errors.append(
                "sac.pendulum_hard_replay_abs_theta_low must be <= sac.pendulum_hard_replay_abs_theta_high"
            )
        _require_nonnegative(
            "sac.pendulum_hard_replay_velocity_limit",
            self.sac.pendulum_hard_replay_velocity_limit,
            errors,
        )
        if (
            max(self.sac.pendulum_hard_replay_fraction, self.sac.pendulum_hard_replay_final_fraction) > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("sac.pendulum_hard_replay_fraction currently supports only Pendulum environments")
        if not (0.0 <= self.sac.pendulum_model_replay_ratio < 1.0):
            errors.append("sac.pendulum_model_replay_ratio must be in [0, 1)")
        _require_positive(
            "sac.pendulum_model_replay_steps_per_step",
            self.sac.pendulum_model_replay_steps_per_step,
            errors,
        )
        _require_nonnegative("sac.pendulum_model_replay_start_step", self.sac.pendulum_model_replay_start_step, errors)
        if not (0.0 <= self.sac.pendulum_model_replay_random_action_fraction <= 1.0):
            errors.append("sac.pendulum_model_replay_random_action_fraction must be in [0, 1]")
        if not (0.0 <= self.sac.pendulum_model_replay_abs_theta_low <= math.pi):
            errors.append("sac.pendulum_model_replay_abs_theta_low must be in [0, pi]")
        if not (0.0 <= self.sac.pendulum_model_replay_abs_theta_high <= math.pi):
            errors.append("sac.pendulum_model_replay_abs_theta_high must be in [0, pi]")
        if self.sac.pendulum_model_replay_abs_theta_low > self.sac.pendulum_model_replay_abs_theta_high:
            errors.append(
                "sac.pendulum_model_replay_abs_theta_low must be <= "
                "sac.pendulum_model_replay_abs_theta_high"
            )
        _require_nonnegative(
            "sac.pendulum_model_replay_velocity_limit",
            self.sac.pendulum_model_replay_velocity_limit,
            errors,
        )
        if self.sac.pendulum_model_replay_ratio > 0.0 and not self.env.env_id.startswith("Pendulum"):
            errors.append("sac.pendulum_model_replay_ratio currently supports only Pendulum environments")
        if not (0.0 <= self.sac.pendulum_model_rollout_ratio < 1.0):
            errors.append("sac.pendulum_model_rollout_ratio must be in [0, 1)")
        _require_positive(
            "sac.pendulum_model_rollout_starts_per_step",
            self.sac.pendulum_model_rollout_starts_per_step,
            errors,
        )
        _require_positive("sac.pendulum_model_rollout_horizon", self.sac.pendulum_model_rollout_horizon, errors)
        _require_positive(
            "sac.pendulum_model_rollout_interval_steps",
            self.sac.pendulum_model_rollout_interval_steps,
            errors,
        )
        _require_nonnegative("sac.pendulum_model_rollout_start_step", self.sac.pendulum_model_rollout_start_step, errors)
        if not (0.0 <= self.sac.pendulum_model_rollout_abs_theta_low <= math.pi):
            errors.append("sac.pendulum_model_rollout_abs_theta_low must be in [0, pi]")
        if not (0.0 <= self.sac.pendulum_model_rollout_abs_theta_high <= math.pi):
            errors.append("sac.pendulum_model_rollout_abs_theta_high must be in [0, pi]")
        if self.sac.pendulum_model_rollout_abs_theta_low > self.sac.pendulum_model_rollout_abs_theta_high:
            errors.append(
                "sac.pendulum_model_rollout_abs_theta_low must be <= "
                "sac.pendulum_model_rollout_abs_theta_high"
            )
        _require_nonnegative(
            "sac.pendulum_model_rollout_velocity_limit",
            self.sac.pendulum_model_rollout_velocity_limit,
            errors,
        )
        if self.sac.pendulum_model_rollout_ratio > 0.0 and not self.env.env_id.startswith("Pendulum"):
            errors.append("sac.pendulum_model_rollout_ratio currently supports only Pendulum environments")
        _require_nonnegative(
            "sac.pendulum_potential_shaping_weight",
            self.sac.pendulum_potential_shaping_weight,
            errors,
        )
        _require_nonnegative(
            "sac.pendulum_potential_shaping_start_update",
            self.sac.pendulum_potential_shaping_start_update,
            errors,
        )
        if not (0.0 <= self.sac.pendulum_potential_shaping_abs_theta_low <= math.pi):
            errors.append("sac.pendulum_potential_shaping_abs_theta_low must be in [0, pi]")
        if not (0.0 <= self.sac.pendulum_potential_shaping_abs_theta_high <= math.pi):
            errors.append("sac.pendulum_potential_shaping_abs_theta_high must be in [0, pi]")
        if self.sac.pendulum_potential_shaping_abs_theta_low > self.sac.pendulum_potential_shaping_abs_theta_high:
            errors.append(
                "sac.pendulum_potential_shaping_abs_theta_low must be <= "
                "sac.pendulum_potential_shaping_abs_theta_high"
            )
        _require_nonnegative(
            "sac.pendulum_potential_shaping_velocity_limit",
            self.sac.pendulum_potential_shaping_velocity_limit,
            errors,
        )
        if self.sac.pendulum_potential_shaping_source not in {"best", "dp_policy", "dp_value", "controller"}:
            errors.append(
                "sac.pendulum_potential_shaping_source must be one of best, dp_policy, dp_value, controller"
            )
        if (
            self.sac.pendulum_potential_shaping_weight > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("sac.pendulum_potential_shaping_weight currently supports only Pendulum environments")
        if self.sac.pendulum_symmetry_augmentation and not self.env.env_id.startswith("Pendulum"):
            errors.append("sac.pendulum_symmetry_augmentation currently supports only Pendulum environments")
        _require_nonnegative(
            "sac.pendulum_actor_symmetry_weight",
            self.sac.pendulum_actor_symmetry_weight,
            errors,
        )
        _require_nonnegative(
            "sac.pendulum_critic_symmetry_weight",
            self.sac.pendulum_critic_symmetry_weight,
            errors,
        )
        if (
            max(
                self.sac.pendulum_actor_symmetry_weight,
                self.sac.pendulum_critic_symmetry_weight,
            )
            > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("Pendulum symmetry-consistency losses currently support only Pendulum environments")
        if (
            self.sac.pendulum_symmetry_augmentation
            and self.sac.sacn_n_step > 1
            and self.sac.sacn_importance_mode != "none"
        ):
            errors.append(
                "sac.pendulum_symmetry_augmentation with SACn requires sac.sacn_importance_mode='none'"
            )
        _require_positive("sac.policy_lr", self.sac.policy_lr, errors)
        _require_positive("sac.q_lr", self.sac.q_lr, errors)
        if self.sac.policy_lr_final is not None:
            _require_positive("sac.policy_lr_final", self.sac.policy_lr_final, errors)
        if self.sac.q_lr_final is not None:
            _require_positive("sac.q_lr_final", self.sac.q_lr_final, errors)
        if self.sac.alpha_lr is not None:
            _require_positive("sac.alpha_lr", self.sac.alpha_lr, errors)
        if self.sac.alpha_lr_final is not None:
            _require_positive("sac.alpha_lr_final", self.sac.alpha_lr_final, errors)
            if self.sac.alpha_lr is None:
                errors.append("sac.alpha_lr_final requires sac.alpha_lr")
        _require_positive("sac.alpha_initial_value", self.sac.alpha_initial_value, errors)
        _require_nonnegative("sac.alpha_min_value", self.sac.alpha_min_value, errors)
        if not math.isfinite(self.sac.alpha_min_value):
            errors.append("sac.alpha_min_value must be finite")
        if not math.isfinite(self.sac.target_entropy_scale):
            errors.append("sac.target_entropy_scale must be finite")
        _require_positive("sac.simba_c_shift", self.sac.simba_c_shift, errors)
        _require_positive("sac.simba_actor_blocks", self.sac.simba_actor_blocks, errors)
        _require_positive("sac.simba_actor_hidden_dim", self.sac.simba_actor_hidden_dim, errors)
        if self.sac.simba_actor_log_std_floor is not None:
            if not math.isfinite(self.sac.simba_actor_log_std_floor):
                errors.append("sac.simba_actor_log_std_floor must be finite")
            elif self.sac.simba_actor_log_std_floor >= 2.0:
                errors.append("sac.simba_actor_log_std_floor must be less than 2")
            if not self.sac.simba_backbone:
                errors.append("sac.simba_actor_log_std_floor requires sac.simba_backbone")
        _require_positive("sac.simba_critic_blocks", self.sac.simba_critic_blocks, errors)
        _require_positive("sac.simba_critic_hidden_dim", self.sac.simba_critic_hidden_dim, errors)
        _require_positive("sac.simba_block_expansion", self.sac.simba_block_expansion, errors)
        _require_positive("sac.simba_critic_num_bins", self.sac.simba_critic_num_bins, errors)
        _require_positive("sac.simba_reward_scale_g_max", self.sac.simba_reward_scale_g_max, errors)
        _require_positive("sac.sacn_n_step", self.sac.sacn_n_step, errors)
        if not (0.0 < self.sac.sacn_importance_quantile <= 1.0):
            errors.append("sac.sacn_importance_quantile must be in (0, 1]")
        _require_positive("sac.sacn_max_entropy_samples", self.sac.sacn_max_entropy_samples, errors)
        _require_nonnegative("sac.sacn_recent_max_age_steps", self.sac.sacn_recent_max_age_steps, errors)
        if not (0.0 <= self.sac.sacn_min_horizon_ess_fraction <= 1.0):
            errors.append("sac.sacn_min_horizon_ess_fraction must be in [0, 1]")
        if self.sac.sacn_importance_mode not in {"density", "none"}:
            errors.append("sac.sacn_importance_mode must be one of density, none")
        _require_nonnegative("sac.sacn_stop_after_steps", self.sac.sacn_stop_after_steps, errors)
        if self.sac.sacn_target_mode not in {"all", "fast_last"}:
            errors.append("sac.sacn_target_mode must be one of all, fast_last")
        if not (0.0 < self.sac.sacn_horizon_lambda <= 1.0):
            errors.append("sac.sacn_horizon_lambda must be in (0, 1]")
        if self.sac.sacn_n_step > self.sac.buffer_size:
            errors.append("sac.sacn_n_step must be <= sac.buffer_size")
        if self.sac.pendulum_model_rollout_ratio > 0.0 and self.sac.sacn_n_step <= 1:
            errors.append("sac.pendulum_model_rollout_ratio requires sac.sacn_n_step > 1")
        if (
            self.sac.pendulum_model_rollout_ratio > 0.0
            and self.sac.pendulum_model_rollout_horizon < self.sac.sacn_n_step
        ):
            errors.append("sac.pendulum_model_rollout_horizon must be >= sac.sacn_n_step")
        if self.sac.batch_size > self.sac.buffer_size:
            errors.append("sac.batch_size must be <= sac.buffer_size")
        if self.sac.learning_starts >= self.sac.total_steps:
            errors.append("sac.learning_starts must be < sac.total_steps")
        if self.sac.random_action_steps is not None:
            _require_nonnegative("sac.random_action_steps", self.sac.random_action_steps, errors)
            if self.sac.random_action_steps >= self.sac.total_steps:
                errors.append("sac.random_action_steps must be < sac.total_steps when set")
        if not (0.0 <= self.sac.gamma <= 1.0):
            errors.append("sac.gamma must be in [0, 1]")
        if not (0.0 < self.sac.tau <= 1.0):
            errors.append("sac.tau must be in (0, 1]")
        if self.sac.simba_weight_projection and not self.sac.simba_backbone:
            errors.append(
                "sac.simba_weight_projection requires sac.simba_backbone; "
                "SimbaV2 projects only bias-free HyperDense weights, not CleanRL Linear layers"
            )
        if self.sac.simba_distributional_critic and not self.sac.simba_backbone:
            errors.append("sac.simba_distributional_critic requires sac.simba_backbone")
        if self.sac.simba_distributional_critic and self.sac.simba_critic_num_bins < 2:
            errors.append("sac.simba_critic_num_bins must be >= 2")
        if self.sac.simba_critic_min_v >= self.sac.simba_critic_max_v:
            errors.append("sac.simba_critic_min_v must be < sac.simba_critic_max_v")
        if self.sac.reference_guidance_mode not in {"none", "replay_injection", "interleaved_execution"}:
            errors.append(
                "sac.reference_guidance_mode must be one of none, replay_injection, interleaved_execution"
            )
        if self.sac.l2_feature_norm and self.sac.simba_backbone:
            errors.append("sac.l2_feature_norm is a CleanRL MLP variant and cannot be combined with sac.simba_backbone")
        if self.sac.reference_guidance_policy not in {"controller", "dp", "best"}:
            errors.append("sac.reference_guidance_policy must be one of controller, dp, best")
        if not (0.0 <= self.sac.reference_guidance_probability <= 1.0):
            errors.append("sac.reference_guidance_probability must be in [0, 1]")
        _require_nonnegative("sac.reference_guidance_start_step", self.sac.reference_guidance_start_step, errors)
        if self.sac.reference_guidance_mode != "none" and not self.env.env_id.startswith("Pendulum"):
            errors.append("reference guidance currently supports only Pendulum environments")
        delayed_reference_replay_after_sacn = (
            self.sac.reference_guidance_mode == "replay_injection"
            and self.sac.sacn_stop_after_steps > 0
            and self.sac.reference_guidance_start_step > self.sac.sacn_stop_after_steps
        )
        if (
            self.sac.sacn_n_step > 1
            and self.sac.reference_guidance_mode != "none"
            and not delayed_reference_replay_after_sacn
            and (
                self.sac.sacn_importance_mode == "density"
                or self.sac.reference_guidance_mode != "interleaved_execution"
            )
        ):
            errors.append(
                "sac.sacn_n_step > 1 with reference guidance requires "
                "sac.reference_guidance_mode='interleaved_execution' and sac.sacn_importance_mode='none', "
                "or replay_injection with sac.reference_guidance_start_step > sac.sacn_stop_after_steps; "
                "density-weighted SACn needs behavior-policy densities, and replay injection does not form "
                "contiguous real trajectories"
            )
        if (
            self.sac.sacn_n_step > 1
            and self.sac.sacn_stop_after_steps <= 0
            and self.sac.pendulum_model_replay_ratio > 0.0
        ):
            errors.append(
                "sac.pendulum_model_replay_ratio with sac.sacn_n_step > 1 requires "
                "sac.sacn_stop_after_steps > 0 because model replay is mixed into one-step SAC updates"
            )
        if self.sac.reference_auxiliary_mode not in {
            "none",
            "bc",
            "q_filtered_bc",
            "q_filtered_replay_bc",
        }:
            errors.append(
                "sac.reference_auxiliary_mode must be one of none, bc, "
                "q_filtered_bc, q_filtered_replay_bc"
            )
        if self.sac.reference_auxiliary_policy not in {"controller", "dp", "best"}:
            errors.append("sac.reference_auxiliary_policy must be one of controller, dp, best")
        _require_nonnegative("sac.reference_auxiliary_weight", self.sac.reference_auxiliary_weight, errors)
        if self.sac.reference_auxiliary_weight_final is not None:
            _require_nonnegative(
                "sac.reference_auxiliary_weight_final",
                self.sac.reference_auxiliary_weight_final,
                errors,
            )
        _require_nonnegative(
            "sac.reference_auxiliary_decay_updates",
            self.sac.reference_auxiliary_decay_updates,
            errors,
        )
        _require_nonnegative(
            "sac.reference_auxiliary_stop_update",
            self.sac.reference_auxiliary_stop_update,
            errors,
        )
        _require_nonnegative("sac.reference_auxiliary_margin", self.sac.reference_auxiliary_margin, errors)
        _require_nonnegative(
            "sac.reference_auxiliary_filter_start_update",
            self.sac.reference_auxiliary_filter_start_update,
            errors,
        )
        if self.sac.reference_auxiliary_q_filter_mode not in {
            "twin_min_difference",
            "online_target_unanimous",
        }:
            errors.append(
                "sac.reference_auxiliary_q_filter_mode must be one of "
                "twin_min_difference, online_target_unanimous"
            )
        if self.sac.reference_auxiliary_replay_normalization not in {
            "selected_mean",
            "full_batch_mean",
        }:
            errors.append(
                "sac.reference_auxiliary_replay_normalization must be one of "
                "selected_mean, full_batch_mean"
            )
        if (
            self.sac.reference_auxiliary_q_filter_mode != "twin_min_difference"
            and self.sac.reference_auxiliary_mode
            not in {"q_filtered_bc", "q_filtered_replay_bc"}
        ):
            errors.append(
                "sac.reference_auxiliary_q_filter_mode is active only for "
                "q_filtered_bc or q_filtered_replay_bc"
            )
        if (
            self.sac.reference_auxiliary_replay_normalization != "selected_mean"
            and self.sac.reference_auxiliary_mode != "q_filtered_replay_bc"
        ):
            errors.append(
                "sac.reference_auxiliary_replay_normalization is active only for "
                "q_filtered_replay_bc"
            )
        if not (0.0 <= self.sac.reference_anchor_ratio < 1.0):
            errors.append("sac.reference_anchor_ratio must be in [0, 1)")
        _require_nonnegative("sac.reference_anchor_size", self.sac.reference_anchor_size, errors)
        _require_positive(
            "sac.reference_anchor_velocity_limit",
            self.sac.reference_anchor_velocity_limit,
            errors,
        )
        if not (0.0 <= self.sac.reference_anchor_reset_support_fraction <= 1.0):
            errors.append("sac.reference_anchor_reset_support_fraction must be in [0, 1]")
        _require_positive(
            "sac.reference_anchor_reset_velocity_limit",
            self.sac.reference_anchor_reset_velocity_limit,
            errors,
        )
        if self.sac.reference_anchor_ratio > 0.0 and self.sac.reference_anchor_size <= 0:
            errors.append("sac.reference_anchor_size must be positive when reference_anchor_ratio > 0")
        _require_nonnegative("sac.sac_actor_loss_weight", self.sac.sac_actor_loss_weight, errors)
        _require_nonnegative("sac.sac_actor_loss_start_step", self.sac.sac_actor_loss_start_step, errors)
        _require_nonnegative(
            "sac.actor_mean_logit_l2_weight",
            self.sac.actor_mean_logit_l2_weight,
            errors,
        )
        _require_nonnegative(
            "sac.actor_mean_logit_excess_threshold",
            self.sac.actor_mean_logit_excess_threshold,
            errors,
        )
        if self.sac.sac_actor_objective_mode not in {"stochastic", "deterministic_mean"}:
            errors.append(
                "sac.sac_actor_objective_mode must be one of stochastic, deterministic_mean"
            )
        if self.sac.sac_actor_gradient_balance_mode not in {"none", "match_reference"}:
            errors.append(
                "sac.sac_actor_gradient_balance_mode must be one of none, match_reference"
            )
        _require_nonnegative(
            "sac.sac_actor_gradient_balance_min_multiplier",
            self.sac.sac_actor_gradient_balance_min_multiplier,
            errors,
        )
        _require_nonnegative(
            "sac.sac_actor_gradient_balance_max_multiplier",
            self.sac.sac_actor_gradient_balance_max_multiplier,
            errors,
        )
        if (
            self.sac.sac_actor_gradient_balance_min_multiplier
            > self.sac.sac_actor_gradient_balance_max_multiplier
        ):
            errors.append(
                "sac.sac_actor_gradient_balance_min_multiplier must be less than or equal to "
                "sac.sac_actor_gradient_balance_max_multiplier"
            )
        if self.sac.sac_actor_gradient_conflict_mode not in {"none", "project_sac"}:
            errors.append(
                "sac.sac_actor_gradient_conflict_mode must be one of none, project_sac"
            )
        if self.sac.sac_actor_gradient_conflict_mode != "none":
            incompatible_actor_auxiliaries = {
                "sac.critic_search_actor_weight": self.sac.critic_search_actor_weight,
                "sac.self_imitation_weight": self.sac.self_imitation_weight,
                "sac.pendulum_actor_symmetry_weight": self.sac.pendulum_actor_symmetry_weight,
                "sac.actor_mean_logit_l2_weight": self.sac.actor_mean_logit_l2_weight,
            }
            for field_name, weight in incompatible_actor_auxiliaries.items():
                if float(weight) > 0.0:
                    errors.append(
                        "sac.sac_actor_gradient_conflict_mode cannot be combined with "
                        f"{field_name}; asymmetric projection currently supports exactly the "
                        "SAC and reference-BC actor objectives"
                    )
        if self.sac.sac_actor_filter_mode not in {
            "none",
            "reference_online_unanimous",
            "reference_online_target_unanimous",
        }:
            errors.append(
                "sac.sac_actor_filter_mode must be one of none, reference_online_unanimous, "
                "reference_online_target_unanimous"
            )
        _require_nonnegative("sac.sac_actor_filter_margin", self.sac.sac_actor_filter_margin, errors)
        if needs_actor_reference_actions(self.sac) and not self.env.env_id.startswith("Pendulum"):
            errors.append("actor reference actions currently support only Pendulum environments")
        if self.sac.reference_critic_mode not in {"none", "margin"}:
            errors.append("sac.reference_critic_mode must be one of none, margin")
        if self.sac.reference_critic_policy not in {"controller", "dp", "best"}:
            errors.append("sac.reference_critic_policy must be one of controller, dp, best")
        _require_nonnegative("sac.reference_critic_weight", self.sac.reference_critic_weight, errors)
        _require_nonnegative("sac.reference_critic_margin", self.sac.reference_critic_margin, errors)
        if (
            self.sac.reference_critic_mode != "none"
            and self.sac.reference_critic_weight > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("reference critic calibration currently supports only Pendulum environments")
        if self.sac.reference_prior_mode not in {"none", "rlpd"}:
            errors.append("sac.reference_prior_mode must be one of none, rlpd")
        if self.sac.reference_prior_policy not in {"controller", "dp", "best"}:
            errors.append("sac.reference_prior_policy must be one of controller, dp, best")
        if not (0.0 <= self.sac.reference_prior_ratio < 1.0):
            errors.append("sac.reference_prior_ratio must be in [0, 1)")
        if self.sac.reference_prior_source not in {"online_one_step", "rollout_dataset", "rollout_plus_online"}:
            errors.append(
                "sac.reference_prior_source must be one of online_one_step, rollout_dataset, rollout_plus_online"
            )
        _require_nonnegative("sac.reference_prior_dataset_steps", self.sac.reference_prior_dataset_steps, errors)
        _require_nonnegative(
            "sac.reference_prior_dataset_seed_offset", self.sac.reference_prior_dataset_seed_offset, errors
        )
        if self.sac.reference_prior_dataset_steps > self.sac.buffer_size:
            errors.append("sac.reference_prior_dataset_steps must be <= sac.buffer_size")
        if (
            self.sac.reference_prior_source in {"rollout_dataset", "rollout_plus_online"}
            and self.sac.reference_prior_dataset_steps <= 0
        ):
            errors.append("sac.reference_prior_dataset_steps must be > 0 for rollout reference prior sources")
        if (
            self.sac.reference_prior_mode != "none"
            and self.sac.reference_prior_ratio > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("reference prior replay currently supports only Pendulum environments")
        if self.sac.actor_init_checkpoint_path is not None and not str(self.sac.actor_init_checkpoint_path).strip():
            errors.append("sac.actor_init_checkpoint_path cannot be empty when set")
        _require_nonnegative("sac.actor_update_start_step", self.sac.actor_update_start_step, errors)
        _require_nonnegative("sac.actor_update_stop_step", self.sac.actor_update_stop_step, errors)
        if (
            self.sac.actor_update_stop_step > 0
            and self.sac.actor_update_stop_step < self.sac.actor_update_start_step
        ):
            errors.append("sac.actor_update_stop_step must be >= sac.actor_update_start_step when set")
        for name, value in (
            (
                "sac.uniform_exploration_initial_probability",
                self.sac.uniform_exploration_initial_probability,
            ),
            (
                "sac.uniform_exploration_final_probability",
                self.sac.uniform_exploration_final_probability,
            ),
        ):
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name} must be in [0, 1]")
        _require_nonnegative(
            "sac.uniform_exploration_decay_steps",
            self.sac.uniform_exploration_decay_steps,
            errors,
        )
        _require_nonnegative(
            "sac.uniform_exploration_start_step",
            self.sac.uniform_exploration_start_step,
            errors,
        )
        if (
            self.sac.sacn_n_step > 1
            and self.sac.sacn_importance_mode == "density"
            and max(
                self.sac.uniform_exploration_initial_probability,
                self.sac.uniform_exploration_final_probability,
            )
            > 0.0
        ):
            errors.append(
                "uniform exploration with SACn requires sacn_importance_mode='none' because the "
                "epsilon-mixture behavior density is not stored"
            )
        _require_nonnegative("sac.cql_alpha", self.sac.cql_alpha, errors)
        _require_positive("sac.cql_temperature", self.sac.cql_temperature, errors)
        _require_positive("sac.cql_num_random_actions", self.sac.cql_num_random_actions, errors)
        _require_positive("sac.cql_interval_updates", self.sac.cql_interval_updates, errors)
        _require_nonnegative("sac.critic_search_actor_weight", self.sac.critic_search_actor_weight, errors)
        _require_positive("sac.critic_search_num_actions", self.sac.critic_search_num_actions, errors)
        _require_nonnegative("sac.critic_search_margin", self.sac.critic_search_margin, errors)
        _require_nonnegative("sac.critic_search_start_update", self.sac.critic_search_start_update, errors)
        if self.sac.critic_search_filter_mode not in {
            "clipped_value",
            "unanimous_advantage",
            "online_target_unanimous_advantage",
        }:
            errors.append(
                "sac.critic_search_filter_mode must be one of clipped_value, "
                "unanimous_advantage, online_target_unanimous_advantage"
            )
        if self.sac.critic_search_actor_loss_type not in {"mse", "log_prob"}:
            errors.append(
                "sac.critic_search_actor_loss_type must be one of: mse, log_prob"
            )
        _require_nonnegative("sac.self_imitation_weight", self.sac.self_imitation_weight, errors)
        if self.sac.self_imitation_loss_type not in {"mse", "log_prob"}:
            errors.append("sac.self_imitation_loss_type must be one of: mse, log_prob")
        _require_nonnegative("sac.self_imitation_start_step", self.sac.self_imitation_start_step, errors)
        _require_positive("sac.self_imitation_temperature", self.sac.self_imitation_temperature, errors)
        _require_nonnegative("sac.self_imitation_margin", self.sac.self_imitation_margin, errors)
        _require_positive("sac.self_imitation_max_weight", self.sac.self_imitation_max_weight, errors)

        _require_positive("eval.episodes", self.eval.episodes, errors)
        _require_nonnegative("eval.every_steps", self.eval.every_steps, errors)
        if self.eval.seeds is not None and len(self.eval.seeds) == 0:
            errors.append("eval.seeds cannot be empty")
        if self.eval.seeds is not None and self.eval.episodes != len(self.eval.seeds):
            errors.append("eval.episodes must match len(eval.seeds) when explicit seeds are set")

        _require_nonnegative("telemetry.log_interval_steps", self.telemetry.log_interval_steps, errors)
        _require_nonnegative(
            "telemetry.checkpoint_interval_steps", self.telemetry.checkpoint_interval_steps, errors
        )
        _require_nonnegative(
            "telemetry.replay_inspection_interval_steps",
            self.telemetry.replay_inspection_interval_steps,
            errors,
        )
        _require_nonnegative("telemetry.diagnostics_interval_steps", self.telemetry.diagnostics_interval_steps, errors)

        for name, value in self.reliability.__dict__.items():
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"reliability.{name} must be finite")
        if not (0.0 <= self.reliability.success_near_upright_fraction_threshold <= 1.0):
            errors.append("reliability.success_near_upright_fraction_threshold must be in [0, 1]")
        _require_nonnegative(
            "reliability.success_max_not_near_upright_streak",
            self.reliability.success_max_not_near_upright_streak,
            errors,
        )
        if not (0.0 <= self.reliability.near_upright_cos_threshold <= 1.0):
            errors.append("reliability.near_upright_cos_threshold must be in [0, 1]")
        _require_nonnegative(
            "reliability.near_upright_abs_velocity_threshold",
            self.reliability.near_upright_abs_velocity_threshold,
            errors,
        )
        if not (0.0 <= self.reliability.dormant_relative_threshold <= 1.0):
            errors.append("reliability.dormant_relative_threshold must be in [0, 1]")

        if errors:
            raise ValueError("Invalid experiment config: " + "; ".join(errors))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _require_positive(name: str, value: int | float, errors: list[str]) -> None:
    if value <= 0:
        errors.append(f"{name} must be positive")


def _require_nonnegative(name: str, value: int | float, errors: list[str]) -> None:
    if value < 0:
        errors.append(f"{name} must be non-negative")
