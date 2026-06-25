from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from last_nine_rl.checkpoints import expand_run_dirs, load_agent_from_run
from last_nine_rl.evaluate import evaluate_agent, episode_outcome_metrics, fixed_eval_seeds, threshold_fractions


SERIES_KEYS = {
    "returns",
    "lengths",
    "near_upright_fractions",
    "min_step_rewards",
    "not_near_upright_streaks",
    "seeds",
}


def main() -> None:
    args = parse_args()
    run_dirs = expand_run_dirs([Path(path) for path in args.runs], require_checkpoint=True)
    if not run_dirs:
        raise SystemExit("No run directories with config.json and checkpoints/final.pt were found.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_checkpoints(
        run_dirs=run_dirs,
        episodes=args.episodes,
        seed_base=args.seed_base,
        output_dir=output_dir,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large fixed-seed evaluation from saved SAC checkpoints.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parents containing checkpoints.")
    parser.add_argument("--out", required=True, help="Output directory for JSON and CSV diagnostics.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=200000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="final.pt")
    return parser.parse_args()


def evaluate_checkpoints(
    run_dirs: list[Path],
    episodes: int,
    seed_base: int,
    output_dir: Path,
    device: str | None,
    checkpoint: str,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    first_reliability = None
    first_thresholds = None

    for run_dir in run_dirs:
        agent, config, payload = load_agent_from_run(run_dir, device=device, checkpoint=checkpoint)
        eval_seeds = fixed_eval_seeds(seed_base, episodes)
        evaluation = evaluate_agent(
            agent,
            config.env,
            episodes=episodes,
            reliability=config.reliability,
            deterministic=config.eval.deterministic,
            seeds=eval_seeds,
        )
        if first_reliability is None:
            first_reliability = config.reliability
            first_thresholds = config.reliability.strict_return_thresholds

        scalar_eval = {key: value for key, value in evaluation.items() if key not in SERIES_KEYS}
        scalar_eval.update(threshold_fractions(evaluation["returns"], config.reliability.strict_return_thresholds))
        run_summaries.append(
            {
                "run_dir": str(run_dir),
                "actual_seed": int(config.seed),
                "checkpoint_extra": payload.get("extra", {}),
                **scalar_eval,
            }
        )
        all_rows.extend(episode_rows(run_dir, config.seed, evaluation, config.reliability))

    summary = summarize_runs(run_summaries, all_rows, first_reliability, first_thresholds)
    write_episode_csv(output_dir / "posthoc_eval_episodes.csv", all_rows)
    (output_dir / "posthoc_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "rows": all_rows}


def episode_rows(run_dir: Path, actual_seed: int, evaluation: dict[str, Any], reliability: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, episode_return in enumerate(evaluation["returns"]):
        near_fraction = float(evaluation["near_upright_fractions"][idx])
        streak = int(evaluation["not_near_upright_streaks"][idx])
        return_success = float(episode_return) >= reliability.success_return_threshold
        stability_success = near_fraction >= reliability.success_near_upright_fraction_threshold
        streak_success = streak <= reliability.success_max_not_near_upright_streak
        task_success = stability_success and streak_success
        rows.append(
            {
                "run_dir": str(run_dir),
                "actual_seed": int(actual_seed),
                "episode_index": idx,
                "eval_seed": int(evaluation["seeds"][idx]),
                "return": float(episode_return),
                "length": int(evaluation["lengths"][idx]),
                "near_upright_fraction": near_fraction,
                "min_step_reward": float(evaluation["min_step_rewards"][idx]),
                "not_near_upright_streak": streak,
                "return_success": float(return_success),
                "stability_success": float(stability_success),
                "streak_success": float(streak_success),
                "task_success": float(task_success),
                "strict_success": float(return_success and task_success),
                "collapse": float(float(episode_return) <= reliability.collapse_return_threshold),
            }
        )
    return rows


def summarize_runs(
    run_summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    reliability: Any,
    thresholds: tuple[float, ...] | None,
) -> dict[str, Any]:
    returns = [float(row["return"]) for row in rows]
    near = [float(row["near_upright_fraction"]) for row in rows]
    streaks = [int(row["not_near_upright_streak"]) for row in rows]
    pooled = episode_outcome_metrics(returns, near, streaks, reliability)
    if thresholds is not None:
        pooled.update(threshold_fractions(returns, thresholds))

    seed_success_rates = np.asarray([run["success_rate"] for run in run_summaries], dtype=np.float64)
    seed_strict_rates = np.asarray([run["strict_success_rate"] for run in run_summaries], dtype=np.float64)
    seed_mean_returns = np.asarray([run["mean_return"] for run in run_summaries], dtype=np.float64)

    return {
        "num_runs": len(run_summaries),
        "num_eval_episodes_per_run": int(len(rows) / max(len(run_summaries), 1)),
        "num_pooled_episodes": len(rows),
        "aggregation_note": "Pooled episode intervals reuse the same eval seeds across training seeds; seed means are the primary cross-run unit.",
        "mean_seed_mean_return": float(np.mean(seed_mean_returns)),
        "worst_seed_mean_return": float(np.min(seed_mean_returns)),
        "mean_seed_success_rate": float(np.mean(seed_success_rates)),
        "mean_seed_strict_success_rate": float(np.mean(seed_strict_rates)),
        "pooled": pooled,
        "runs": run_summaries,
    }


def write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
