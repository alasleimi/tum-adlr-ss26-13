from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from last_nine_rl.config import ExperimentConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard import is environment-dependent.
    SummaryWriter = None


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class TelemetryLogger:
    def __init__(self, run_dir: str | Path, config: ExperimentConfig):
        if config.telemetry.tensorboard and SummaryWriter is None:
            raise RuntimeError("TensorBoard logging is enabled, but torch.utils.tensorboard.SummaryWriter is unavailable.")

        self.run_dir = Path(run_dir)
        self._prepare_run_dir(config)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        config.to_json(self.run_dir / "config.json")
        self.config = config

        self.events_file = (self.run_dir / "events.jsonl").open("w", encoding="utf-8")
        self.metrics_file = (self.run_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=["step", "split", "name", "value"])
        self.metrics_writer.writeheader()

        self.eval_returns_file = None
        self.eval_returns_writer = None
        if config.telemetry.write_eval_returns_csv:
            self.eval_returns_file = (self.run_dir / "eval_episodes.csv").open("w", newline="", encoding="utf-8")
            self.eval_returns_writer = csv.DictWriter(
                self.eval_returns_file,
                fieldnames=[
                    "step",
                    "episode_index",
                    "seed",
                    "return",
                    "length",
                    "near_upright_fraction",
                    "min_step_reward",
                    "not_near_upright_streak",
                    "success",
                    "return_success",
                    "stability_success",
                    "streak_success",
                    "strict_success",
                    "collapse",
                ],
            )
            self.eval_returns_writer.writeheader()

        self.tb_writer = None
        if config.telemetry.tensorboard and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(str(self.run_dir / "tensorboard"))
            self.tb_writer.add_text(
                "config/json",
                "```json\n" + json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n```",
                global_step=0,
            )

    def _prepare_run_dir(self, config: ExperimentConfig) -> None:
        known_outputs = [
            "config.json",
            "events.jsonl",
            "metrics.csv",
            "eval_episodes.csv",
            "replay_final.npz",
        ]
        known_dirs = ["checkpoints", "tensorboard"]
        existing = [name for name in known_outputs if (self.run_dir / name).exists()]
        existing.extend(name for name in known_dirs if (self.run_dir / name).exists())
        if existing and not config.telemetry.overwrite:
            joined = ", ".join(sorted(existing))
            raise FileExistsError(
                f"Run directory {self.run_dir} already contains telemetry outputs ({joined}). "
                "Use --overwrite or choose a new --run-dir."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if config.telemetry.overwrite:
            for name in known_outputs:
                path = self.run_dir / name
                if path.exists():
                    path.unlink()
            for name in known_dirs:
                path = self.run_dir / name
                if path.exists():
                    shutil.rmtree(path)

    def log_event(self, event_type: str, step: int, payload: dict[str, Any]) -> None:
        row = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(step),
            "type": event_type,
            "payload": _jsonable(payload),
        }
        self.events_file.write(json.dumps(row, sort_keys=True) + "\n")
        self.events_file.flush()

    def log_metrics(self, step: int, split: str, metrics: dict[str, float]) -> None:
        for name, value in sorted(metrics.items()):
            if value is None:
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            self.metrics_writer.writerow({"step": int(step), "split": split, "name": name, "value": scalar})
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f"{split}/{name}", scalar, int(step))
        self.metrics_file.flush()

    def log_eval_episodes(
        self,
        step: int,
        evaluation: dict[str, Any],
        success_threshold: float,
        collapse_threshold: float,
    ) -> None:
        if self.eval_returns_writer is None:
            return
        returns = evaluation.get("returns", [])
        lengths = evaluation.get("lengths", [])
        seeds = evaluation.get("seeds", [])
        near = evaluation.get("near_upright_fractions", [])
        min_rewards = evaluation.get("min_step_rewards", [])
        streaks = evaluation.get("not_near_upright_streaks", [])
        for idx, episode_return in enumerate(returns):
            return_success = float(float(episode_return) >= success_threshold)
            stability_success = float(float(near[idx]) >= self.config.reliability.success_near_upright_fraction_threshold)
            streak_success = float(int(streaks[idx]) <= self.config.reliability.success_max_not_near_upright_streak)
            row = {
                "step": int(step),
                "episode_index": idx,
                "seed": int(seeds[idx]),
                "return": float(episode_return),
                "length": int(lengths[idx]),
                "near_upright_fraction": float(near[idx]),
                "min_step_reward": float(min_rewards[idx]),
                "not_near_upright_streak": int(streaks[idx]),
                "success": return_success,
                "return_success": return_success,
                "stability_success": stability_success,
                "streak_success": streak_success,
                "strict_success": float(bool(return_success and stability_success and streak_success)),
                "collapse": float(float(episode_return) <= collapse_threshold),
            }
            self.eval_returns_writer.writerow(row)
        self.eval_returns_file.flush()

    def close(self) -> None:
        self.events_file.close()
        self.metrics_file.close()
        if self.eval_returns_file is not None:
            self.eval_returns_file.close()
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()


def default_run_dir(config: ExperimentConfig) -> Path:
    return Path(config.telemetry.run_root) / config.name / f"{timestamp_utc()}_seed{config.seed}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
