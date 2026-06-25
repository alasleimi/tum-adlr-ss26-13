from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass
class EnvConfig:
    env_id: str = "Pendulum-v1"
    max_episode_steps: int | None = None
    pendulum_hard_reset_prob: float = 0.0
    pendulum_hard_reset_abs_theta_low: float = 2.0943951023931953
    pendulum_hard_reset_abs_theta_high: float = 2.356194490192345
    pendulum_hard_reset_velocity_limit: float = 1.0


@dataclass
class SACConfig:
    total_steps: int = 100_000
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_lr_final: float | None = None
    q_lr_final: float | None = None
    alpha_initial_value: float = 1.0
    target_entropy_scale: float = -1.0
    updates_per_step: int = 1
    policy_frequency: int = 2
    target_network_frequency: int = 1
    update_diagnostics: bool = True
    redo_interval_updates: int = 0
    redo_dormant_threshold: float = 0.025
    swd_linear_decay_steps: int = 0
    swd_min_weight: float = 0.1
    pendulum_hard_replay_fraction: float = 0.0
    pendulum_hard_replay_abs_theta_low: float = 2.0943951023931953
    pendulum_hard_replay_abs_theta_high: float = 2.356194490192345
    pendulum_hard_replay_velocity_limit: float = 1.0
    device: str = "auto"
    simba_backbone: bool = False
    simba_observation_norm: bool = True
    simba_feature_norm: bool = True
    simba_input_shift: bool = True
    simba_c_shift: float = 3.0
    simba_actor_blocks: int = 1
    simba_actor_hidden_dim: int = 128
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
    reference_guidance_mode: str = "none"
    reference_guidance_policy: str = "best"
    reference_guidance_probability: float = 0.0
    reference_guidance_dp_solution_path: str | None = None
    reference_auxiliary_mode: str = "none"
    reference_auxiliary_policy: str = "dp"
    reference_auxiliary_weight: float = 0.0
    reference_auxiliary_margin: float = 0.0
    reference_critic_mode: str = "none"
    reference_critic_policy: str = "best"
    reference_critic_weight: float = 0.0
    reference_critic_margin: float = 0.0


@dataclass
class EvalConfig:
    every_steps: int = 5_000
    episodes: int = 10
    deterministic: bool = True
    seed_base: int = 100_000
    seeds: tuple[int, ...] | None = None


@dataclass
class ReliabilityConfig:
    success_return_threshold: float = -200.0
    strict_return_thresholds: tuple[float, ...] = (-250.0, -200.0, -150.0, -100.0)
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

        sac = SACConfig(**raw.get("sac", {}))

        eval_raw = dict(raw.get("eval", {}))
        if eval_raw.get("seeds") is not None:
            eval_raw["seeds"] = tuple(int(v) for v in eval_raw["seeds"])
        eval_cfg = EvalConfig(**eval_raw)
        if eval_cfg.seeds is not None:
            eval_cfg.episodes = len(eval_cfg.seeds)

        rel_raw = dict(raw.get("reliability", {}))
        if "strict_return_thresholds" in rel_raw:
            rel_raw["strict_return_thresholds"] = tuple(float(v) for v in rel_raw["strict_return_thresholds"])
        reliability = ReliabilityConfig(**rel_raw)

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
        if self.env.pendulum_hard_reset_prob > 0.0 and not self.env.env_id.startswith("Pendulum"):
            errors.append("env.pendulum_hard_reset_prob currently supports only Pendulum environments")

        _require_positive("sac.total_steps", self.sac.total_steps, errors)
        _require_positive("sac.buffer_size", self.sac.buffer_size, errors)
        _require_nonnegative("sac.learning_starts", self.sac.learning_starts, errors)
        _require_positive("sac.batch_size", self.sac.batch_size, errors)
        _require_positive("sac.updates_per_step", self.sac.updates_per_step, errors)
        _require_positive("sac.policy_frequency", self.sac.policy_frequency, errors)
        _require_positive("sac.target_network_frequency", self.sac.target_network_frequency, errors)
        _require_nonnegative("sac.redo_interval_updates", self.sac.redo_interval_updates, errors)
        if not (0.0 <= self.sac.redo_dormant_threshold <= 1.0):
            errors.append("sac.redo_dormant_threshold must be in [0, 1]")
        if not (0.0 <= self.sac.swd_min_weight <= 1.0):
            errors.append("sac.swd_min_weight must be in [0, 1]")
        if not (0.0 <= self.sac.pendulum_hard_replay_fraction <= 1.0):
            errors.append("sac.pendulum_hard_replay_fraction must be in [0, 1]")
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
        if self.sac.pendulum_hard_replay_fraction > 0.0 and not self.env.env_id.startswith("Pendulum"):
            errors.append("sac.pendulum_hard_replay_fraction currently supports only Pendulum environments")
        _require_positive("sac.policy_lr", self.sac.policy_lr, errors)
        _require_positive("sac.q_lr", self.sac.q_lr, errors)
        if self.sac.policy_lr_final is not None:
            _require_positive("sac.policy_lr_final", self.sac.policy_lr_final, errors)
        if self.sac.q_lr_final is not None:
            _require_positive("sac.q_lr_final", self.sac.q_lr_final, errors)
        _require_positive("sac.alpha_initial_value", self.sac.alpha_initial_value, errors)
        if not math.isfinite(self.sac.target_entropy_scale):
            errors.append("sac.target_entropy_scale must be finite")
        _require_positive("sac.simba_c_shift", self.sac.simba_c_shift, errors)
        _require_positive("sac.simba_actor_blocks", self.sac.simba_actor_blocks, errors)
        _require_positive("sac.simba_actor_hidden_dim", self.sac.simba_actor_hidden_dim, errors)
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
        if self.sac.sacn_n_step > self.sac.buffer_size:
            errors.append("sac.sacn_n_step must be <= sac.buffer_size")
        if self.sac.batch_size > self.sac.buffer_size:
            errors.append("sac.batch_size must be <= sac.buffer_size")
        if self.sac.learning_starts >= self.sac.total_steps:
            errors.append("sac.learning_starts must be < sac.total_steps")
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
        if self.sac.reference_guidance_policy not in {"controller", "dp", "best"}:
            errors.append("sac.reference_guidance_policy must be one of controller, dp, best")
        if not (0.0 <= self.sac.reference_guidance_probability <= 1.0):
            errors.append("sac.reference_guidance_probability must be in [0, 1]")
        if self.sac.reference_guidance_mode != "none" and not self.env.env_id.startswith("Pendulum"):
            errors.append("reference guidance currently supports only Pendulum environments")
        if (
            self.sac.sacn_n_step > 1
            and self.sac.reference_guidance_mode != "none"
            and (
                self.sac.sacn_importance_mode == "density"
                or self.sac.reference_guidance_mode != "interleaved_execution"
            )
        ):
            errors.append(
                "sac.sacn_n_step > 1 with reference guidance requires "
                "sac.reference_guidance_mode='interleaved_execution' and sac.sacn_importance_mode='none'; "
                "density-weighted SACn needs behavior-policy densities, and replay injection does not form "
                "contiguous real trajectories"
            )
        if self.sac.reference_auxiliary_mode not in {"none", "bc", "q_filtered_bc"}:
            errors.append("sac.reference_auxiliary_mode must be one of none, bc, q_filtered_bc")
        if self.sac.reference_auxiliary_policy not in {"controller", "dp", "best"}:
            errors.append("sac.reference_auxiliary_policy must be one of controller, dp, best")
        _require_nonnegative("sac.reference_auxiliary_weight", self.sac.reference_auxiliary_weight, errors)
        _require_nonnegative("sac.reference_auxiliary_margin", self.sac.reference_auxiliary_margin, errors)
        if (
            self.sac.reference_auxiliary_mode != "none"
            and self.sac.reference_auxiliary_weight > 0.0
            and not self.env.env_id.startswith("Pendulum")
        ):
            errors.append("reference auxiliary loss currently supports only Pendulum environments")
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

        _require_positive("eval.episodes", self.eval.episodes, errors)
        _require_nonnegative("eval.every_steps", self.eval.every_steps, errors)
        if self.eval.seeds is not None and len(self.eval.seeds) == 0:
            errors.append("eval.seeds cannot be empty")
        if self.eval.seeds is not None and self.eval.episodes != len(self.eval.seeds):
            errors.append("eval.episodes must match len(eval.seeds) when explicit seeds are set")

        _require_nonnegative("telemetry.log_interval_steps", self.telemetry.log_interval_steps, errors)
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
