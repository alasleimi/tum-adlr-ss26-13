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
    updates_per_step: int = 1
    policy_frequency: int = 2
    target_network_frequency: int = 1
    device: str = "auto"


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

        _require_positive("sac.total_steps", self.sac.total_steps, errors)
        _require_positive("sac.buffer_size", self.sac.buffer_size, errors)
        _require_nonnegative("sac.learning_starts", self.sac.learning_starts, errors)
        _require_positive("sac.batch_size", self.sac.batch_size, errors)
        _require_positive("sac.updates_per_step", self.sac.updates_per_step, errors)
        _require_positive("sac.policy_frequency", self.sac.policy_frequency, errors)
        _require_positive("sac.target_network_frequency", self.sac.target_network_frequency, errors)
        _require_positive("sac.policy_lr", self.sac.policy_lr, errors)
        _require_positive("sac.q_lr", self.sac.q_lr, errors)
        if self.sac.batch_size > self.sac.buffer_size:
            errors.append("sac.batch_size must be <= sac.buffer_size")
        if self.sac.learning_starts >= self.sac.total_steps:
            errors.append("sac.learning_starts must be < sac.total_steps")
        if not (0.0 <= self.sac.gamma <= 1.0):
            errors.append("sac.gamma must be in [0, 1]")
        if not (0.0 < self.sac.tau <= 1.0):
            errors.append("sac.tau must be in (0, 1]")

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
