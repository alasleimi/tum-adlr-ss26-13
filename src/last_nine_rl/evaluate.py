from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import numpy as np

from last_nine_rl.config import EnvConfig, ReliabilityConfig
from last_nine_rl.envs import UprightDetector, make_env
from last_nine_rl.sac import SACAgent


def evaluate_agent(
    agent: SACAgent,
    env_cfg: EnvConfig,
    episodes: int,
    reliability: ReliabilityConfig,
    deterministic: bool = True,
    seeds: Sequence[int] | None = None,
    seed: int | None = None,
) -> dict[str, float | list[float]]:
    return evaluate_policy(
        lambda observation: agent.act(observation, deterministic=deterministic),
        env_cfg,
        episodes=episodes,
        reliability=reliability,
        seeds=seeds,
        seed=seed,
    )


def evaluate_policy(
    act: Callable[[np.ndarray], np.ndarray],
    env_cfg: EnvConfig,
    episodes: int,
    reliability: ReliabilityConfig,
    seeds: Sequence[int] | None = None,
    seed: int | None = None,
) -> dict[str, float | list[float]]:
    eval_seeds = fixed_eval_seeds(int(seed or 0), episodes, seeds)
    env_seed = eval_seeds[0] if eval_seeds else int(seed or 0)
    env = make_env(env_cfg.env_id, seed=env_seed, max_episode_steps=env_cfg.max_episode_steps)
    detector = UprightDetector(
        env_cfg.env_id,
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    returns: list[float] = []
    lengths: list[int] = []
    near_fractions: list[float] = []
    min_rewards: list[float] = []
    worst_near_upright_streaks: list[int] = []
    try:
        for episode, eval_seed in enumerate(eval_seeds):
            obs, _ = env.reset(seed=int(eval_seed))
            done = False
            episode_return = 0.0
            length = 0
            near_count = 0
            episode_min_reward = math.inf
            current_not_near_streak = 0
            longest_not_near_streak = 0
            while not done:
                action = act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                length += 1
                episode_min_reward = min(episode_min_reward, float(reward))
                near = bool(detector.near_upright(np.asarray(next_obs))[0])
                near_count += int(near)
                if near:
                    current_not_near_streak = 0
                else:
                    current_not_near_streak += 1
                    longest_not_near_streak = max(longest_not_near_streak, current_not_near_streak)
                done = bool(terminated or truncated)
                obs = next_obs
            returns.append(episode_return)
            lengths.append(length)
            near_fractions.append(near_count / max(length, 1))
            min_rewards.append(episode_min_reward)
            worst_near_upright_streaks.append(longest_not_near_streak)
    finally:
        env.close()

    returns_np = np.asarray(returns, dtype=np.float64)
    outcomes = episode_outcome_metrics(returns, near_fractions, worst_near_upright_streaks, reliability)
    return {
        "num_eval_episodes": float(len(returns)),
        "mean_return": float(np.mean(returns_np)),
        "median_return": float(np.median(returns_np)),
        "worst_return": float(np.min(returns_np)),
        "best_return": float(np.max(returns_np)),
        "std_return": float(np.std(returns_np)),
        "return_p05": float(np.percentile(returns_np, 5)),
        "return_p10": float(np.percentile(returns_np, 10)),
        "return_p25": float(np.percentile(returns_np, 25)),
        "return_p75": float(np.percentile(returns_np, 75)),
        "return_p90": float(np.percentile(returns_np, 90)),
        "return_p95": float(np.percentile(returns_np, 95)),
        **outcomes,
        "mean_episode_length": float(np.mean(np.asarray(lengths, dtype=np.float64))),
        "near_upright_fraction": float(np.mean(np.asarray(near_fractions, dtype=np.float64))),
        "worst_episode_near_upright_fraction": float(np.min(np.asarray(near_fractions, dtype=np.float64))),
        "mean_min_step_reward": float(np.mean(np.asarray(min_rewards, dtype=np.float64))),
        "worst_min_step_reward": float(np.min(np.asarray(min_rewards, dtype=np.float64))),
        "max_not_near_upright_streak": float(np.max(worst_near_upright_streaks)),
        "returns": returns,
        "lengths": lengths,
        "near_upright_fractions": near_fractions,
        "min_step_rewards": min_rewards,
        "not_near_upright_streaks": worst_near_upright_streaks,
        "seeds": [int(s) for s in eval_seeds],
    }


def episode_outcome_metrics(
    returns: Sequence[float],
    near_upright_fractions: Sequence[float],
    not_near_upright_streaks: Sequence[int],
    reliability: ReliabilityConfig,
) -> dict[str, float]:
    returns_np = np.asarray(returns, dtype=np.float64)
    near_np = np.asarray(near_upright_fractions, dtype=np.float64)
    streak_np = np.asarray(not_near_upright_streaks, dtype=np.float64)

    return_success = returns_np >= reliability.success_return_threshold
    stability_success = near_np >= reliability.success_near_upright_fraction_threshold
    streak_success = streak_np <= reliability.success_max_not_near_upright_streak
    task_success = stability_success & streak_success
    strict_success = return_success & stability_success & streak_success
    collapse = returns_np <= reliability.collapse_return_threshold

    return_success_low, return_success_high = wilson_interval(int(np.sum(return_success)), len(return_success))
    task_success_low, task_success_high = wilson_interval(int(np.sum(task_success)), len(task_success))
    strict_success_low, strict_success_high = wilson_interval(int(np.sum(strict_success)), len(strict_success))
    return_failure_rate = float(1.0 - np.mean(return_success))
    task_failure_rate = float(1.0 - np.mean(task_success))
    strict_failure_rate = float(1.0 - np.mean(strict_success))
    return_failure_high = float(1.0 - return_success_low)
    task_failure_high = float(1.0 - task_success_low)
    strict_failure_high = float(1.0 - strict_success_low)
    total = len(return_success)

    return {
        "num_successes": float(np.sum(return_success)),
        "success_rate": float(np.mean(return_success)),
        "failure_rate": return_failure_rate,
        "success_rate_wilson95_low": return_success_low,
        "success_rate_wilson95_high": return_success_high,
        "return_success_rate": float(np.mean(return_success)),
        "return_failure_rate": return_failure_rate,
        "return_reliability_nines_empirical": empirical_reliability_nines(return_failure_rate, total),
        "return_reliability_nines_wilson95_low": reliability_nines(return_failure_high),
        "stability_success_rate": float(np.mean(stability_success)),
        "streak_success_rate": float(np.mean(streak_success)),
        "num_task_successes": float(np.sum(task_success)),
        "task_success_rate": float(np.mean(task_success)),
        "task_failure_rate": task_failure_rate,
        "task_success_rate_wilson95_low": task_success_low,
        "task_success_rate_wilson95_high": task_success_high,
        "task_reliability_nines_empirical": empirical_reliability_nines(task_failure_rate, total),
        "task_reliability_nines_wilson95_low": reliability_nines(task_failure_high),
        "num_strict_successes": float(np.sum(strict_success)),
        "strict_success_rate": float(np.mean(strict_success)),
        "strict_failure_rate": strict_failure_rate,
        "strict_success_rate_wilson95_low": strict_success_low,
        "strict_success_rate_wilson95_high": strict_success_high,
        "strict_reliability_nines_empirical": empirical_reliability_nines(strict_failure_rate, total),
        "strict_reliability_nines_wilson95_low": reliability_nines(strict_failure_high),
        "num_collapses": float(np.sum(collapse)),
        "collapse_rate": float(np.mean(collapse)),
    }


def empirical_reliability_nines(failure_rate: float, total: int) -> float:
    if failure_rate <= 0.0:
        failure_rate = 1.0 / (total + 1.0)
    return reliability_nines(failure_rate)


def reliability_nines(failure_rate: float) -> float:
    value = float(-math.log10(max(failure_rate, 1e-12)))
    return 0.0 if abs(value) < 1e-12 else value


def threshold_fractions(returns: Sequence[float], thresholds: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(returns, dtype=np.float64)
    return {f"fraction_return_ge_{threshold:g}": float(np.mean(arr >= threshold)) for threshold in thresholds}


def fixed_eval_seeds(seed_base: int, episodes: int, explicit_seeds: Sequence[int] | None = None) -> list[int]:
    if explicit_seeds is not None:
        out = [int(seed) for seed in explicit_seeds]
        if not out:
            raise ValueError("Explicit evaluation seed list cannot be empty.")
        return out
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive.")
    return [int(seed_base + i) for i in range(episodes)]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))
