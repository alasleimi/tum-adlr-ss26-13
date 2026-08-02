from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from last_nine_rl.config import ExperimentConfig
from last_nine_rl.evaluate import evaluate_policy, fixed_eval_seeds


class PendulumEnergySwingupController:
    """Energy-shaping swing-up plus local PD stabilization for Gymnasium Pendulum."""

    def __init__(
        self,
        energy_gain: float = 2.0,
        kp: float = 9.0,
        kd: float = 3.0,
        switch_angle: float = 0.4,
        switch_velocity: float = 3.0,
        max_torque: float = 2.0,
        gravity: float = 10.0,
        length: float = 1.0,
        mass: float = 1.0,
    ):
        self.energy_gain = energy_gain
        self.kp = kp
        self.kd = kd
        self.switch_angle = switch_angle
        self.switch_velocity = switch_velocity
        self.max_torque = max_torque
        self.upright_energy = 3.0 * gravity / (2.0 * length)
        self.input_gain = 3.0 / (mass * length * length)

    def act(self, observation: np.ndarray) -> np.ndarray:
        theta = float(np.arctan2(observation[1], observation[0]))
        theta_dot = float(observation[2])
        if abs(theta) <= self.switch_angle and abs(theta_dot) <= self.switch_velocity:
            torque = -self.kp * theta - self.kd * theta_dot
        else:
            energy = 0.5 * theta_dot * theta_dot + self.upright_energy * np.cos(theta)
            torque = -self.energy_gain * (energy - self.upright_energy) * theta_dot
            if abs(theta_dot) < 0.1:
                torque += -0.5 * np.sign(theta if theta != 0.0 else 1.0)
        return np.asarray([np.clip(torque, -self.max_torque, self.max_torque)], dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        return {
            "energy_gain": self.energy_gain,
            "kp": self.kp,
            "kd": self.kd,
            "switch_angle": self.switch_angle,
            "switch_velocity": self.switch_velocity,
            "max_torque": self.max_torque,
        }


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.episodes is not None:
        config.eval.episodes = args.episodes
        config.eval.seeds = None
    if args.seed_base is not None:
        config.eval.seed_base = args.seed_base
        config.eval.seeds = None
    config.validate()

    result = evaluate_pendulum_reference(config)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Week 1 task reference controllers for threshold calibration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def evaluate_pendulum_reference(config: ExperimentConfig) -> dict[str, Any]:
    if not config.env.env_id.startswith("Pendulum"):
        raise ValueError("Only Pendulum reference control is implemented.")
    controller = PendulumEnergySwingupController()
    eval_seeds = fixed_eval_seeds(config.eval.seed_base, config.eval.episodes, config.eval.seeds)
    evaluation = evaluate_policy(
        controller.act,
        config.env,
        episodes=config.eval.episodes,
        reliability=config.reliability,
        seeds=eval_seeds,
    )
    scalar_eval = {k: v for k, v in evaluation.items() if not isinstance(v, list)}
    return {
        "controller": "pendulum_energy_swingup_pd",
        "controller_params": controller.to_dict(),
        "env_id": config.env.env_id,
        "eval_seeds": eval_seeds,
        "success_definition": {
            "near_upright_fraction_threshold": config.reliability.success_near_upright_fraction_threshold,
            "max_not_near_upright_streak": config.reliability.success_max_not_near_upright_streak,
        },
        "metrics": scalar_eval,
    }


if __name__ == "__main__":
    main()
