from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from last_nine_rl.evaluate import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate last-nine SAC evaluation telemetry across seed runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or parent directories containing runs.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None, help="Return thresholds for seed fractions.")
    args = parser.parse_args()

    run_dirs = expand_run_dirs([Path(p) for p in args.runs])
    summary = aggregate_runs(run_dirs, args.thresholds)
    print(json.dumps(summary, indent=2, sort_keys=True))


def expand_run_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if (path / "events.jsonl").exists():
            out.append(path)
            continue
        out.extend(sorted(child for child in path.glob("**/events.jsonl") if child.is_file()))
    return sorted({p if p.name != "events.jsonl" else p.parent for p in out})


def aggregate_runs(run_dirs: list[Path], thresholds: list[float] | None = None) -> dict[str, Any]:
    evals_by_run = []
    for run_dir in run_dirs:
        evals = load_evaluations(run_dir)
        if evals:
            evals_by_run.append(
                {
                    "run_dir": str(run_dir),
                    "actual_seed": load_actual_seed(run_dir),
                    "evaluations": evals,
                    "final_eval_episodes": load_eval_episode_rows(run_dir, evals[-1]["step"]),
                }
            )
    if not evals_by_run:
        return {"num_runs": 0, "error": "no evaluation events found"}

    grouped_runs = group_runs_by_seed(evals_by_run)
    final_run_evals = [final_run_eval(run) for run in evals_by_run]
    final_seed_evals = [combine_seed_group(group) for group in grouped_runs]
    representative_evals = [final_run_eval(group[0]) for group in grouped_runs]

    seed_mean_returns = np.asarray([e["mean_return"] for e in final_seed_evals], dtype=np.float64)
    seed_worst_returns = np.asarray([e["worst_return"] for e in final_seed_evals], dtype=np.float64)
    seed_success_rates = np.asarray([e["success_rate"] for e in final_seed_evals], dtype=np.float64)
    seed_collapse_rates = np.asarray([e["collapse_rate"] for e in final_seed_evals], dtype=np.float64)
    pooled_successes = int(sum(e.get("num_successes", 0.0) for e in representative_evals))
    pooled_eval_episodes = int(sum(e.get("num_eval_episodes", 0.0) for e in representative_evals))
    pooled_success_low, pooled_success_high = wilson_interval(pooled_successes, pooled_eval_episodes)
    seed_strict_success_rates = np.asarray(
        [e.get("strict_success_rate", 0.0) for e in final_seed_evals],
        dtype=np.float64,
    )
    pooled_strict_successes = int(sum(e.get("num_strict_successes", 0.0) for e in representative_evals))
    pooled_strict_low, pooled_strict_high = wilson_interval(pooled_strict_successes, pooled_eval_episodes)
    post_success_collapse = np.asarray(
        [any(has_post_success_collapse(run["evaluations"]) for run in group) for group in grouped_runs]
    )
    ever_success = np.asarray(
        [any(max(e.get("success_rate", 0.0) for e in run["evaluations"]) > 0.0 for run in group) for group in grouped_runs]
    )
    mean_return_ci_low, mean_return_ci_high = bootstrap_mean_ci(seed_mean_returns)
    success_rate_ci_low, success_rate_ci_high = bootstrap_mean_ci(seed_success_rates)
    duplicate_seed_groups = duplicate_actual_seed_groups(grouped_runs)
    final_episode_rows = [row for group in grouped_runs for row in group[0]["final_eval_episodes"]]

    summary: dict[str, Any] = {
        "num_runs": len(final_run_evals),
        "num_actual_seeds": len(final_seed_evals),
        "duplicate_actual_seed_count": sum(group["num_runs"] - 1 for group in duplicate_seed_groups),
        "duplicate_actual_seeds": duplicate_seed_groups,
        "aggregation_unit": "actual_seed",
        "pooled_success_unit": "first_run_per_actual_seed",
        "final_mean_seed_mean_return": float(np.mean(seed_mean_returns)),
        "final_mean_seed_mean_return_bootstrap95_low": mean_return_ci_low,
        "final_mean_seed_mean_return_bootstrap95_high": mean_return_ci_high,
        "final_median_seed_mean_return": float(np.median(seed_mean_returns)),
        "final_best_seed_mean_return": float(np.max(seed_mean_returns)),
        "final_worst_seed_mean_return": float(np.min(seed_mean_returns)),
        "final_worst_seed_worst_episode_return": float(np.min(seed_worst_returns)),
        "final_mean_success_rate": float(np.mean(seed_success_rates)),
        "final_mean_success_rate_bootstrap95_low": success_rate_ci_low,
        "final_mean_success_rate_bootstrap95_high": success_rate_ci_high,
        "final_pooled_eval_episodes": pooled_eval_episodes,
        "final_pooled_successes": pooled_successes,
        "final_pooled_success_rate": float(pooled_successes / pooled_eval_episodes) if pooled_eval_episodes else 0.0,
        "final_pooled_success_rate_wilson95_low": pooled_success_low,
        "final_pooled_success_rate_wilson95_high": pooled_success_high,
        "final_mean_strict_success_rate": float(np.mean(seed_strict_success_rates)),
        "final_pooled_strict_successes": pooled_strict_successes,
        "final_pooled_strict_success_rate": float(pooled_strict_successes / pooled_eval_episodes)
        if pooled_eval_episodes
        else 0.0,
        "final_pooled_strict_success_rate_wilson95_low": pooled_strict_low,
        "final_pooled_strict_success_rate_wilson95_high": pooled_strict_high,
        "final_collapse_frequency_any_episode": float(np.mean(seed_collapse_rates > 0.0)),
        "final_mean_collapse_rate": float(np.mean(seed_collapse_rates)),
        "ever_success_frequency": float(np.mean(ever_success)),
        "post_success_collapse_frequency": float(np.mean(post_success_collapse)),
        "runs": final_run_evals,
        "seed_units": final_seed_evals,
    }
    summary.update(eval_episode_summary(final_episode_rows))
    summary.update(eval_seed_difficulty_summary(final_episode_rows))

    if thresholds:
        for threshold in thresholds:
            summary[f"final_fraction_seeds_mean_return_ge_{threshold:g}"] = float(np.mean(seed_mean_returns >= threshold))
            summary[f"final_fraction_seeds_worst_return_ge_{threshold:g}"] = float(np.mean(seed_worst_returns >= threshold))
    return summary


def load_evaluations(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    evals = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "evaluation":
                payload = dict(event["payload"])
                payload["step"] = event["step"]
                evals.append(payload)
    return evals


def load_actual_seed(run_dir: Path) -> int | None:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return int(json.load(f)["seed"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    match = re.search(r"seed(-?\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def load_eval_episode_rows(run_dir: Path, step: int) -> list[dict[str, float]]:
    path = run_dir / "eval_episodes.csv"
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["step"]) == int(step):
                rows.append({key: float(value) for key, value in row.items() if key != "step"})
    return rows


def group_runs_by_seed(runs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for run in runs:
        key = ("seed", run["actual_seed"]) if run["actual_seed"] is not None else ("run", run["run_dir"])
        groups.setdefault(key, []).append(run)
    return [sorted(group, key=lambda run: run["run_dir"]) for _, group in sorted(groups.items(), key=lambda item: str(item[0]))]


def final_run_eval(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": run["run_dir"],
        "actual_seed": run["actual_seed"],
        **run["evaluations"][-1],
    }


def combine_seed_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    finals = [final_run_eval(run) for run in group]
    combined = dict(finals[0])
    combined["run_dirs"] = [run["run_dir"] for run in group]
    combined["num_runs_for_seed"] = len(group)

    for name in numeric_metric_names(finals):
        values = np.asarray([float(final[name]) for final in finals], dtype=np.float64)
        if name in {"worst_return", "worst_episode_near_upright_fraction", "worst_min_step_reward"}:
            combined[name] = float(np.min(values))
        elif name in {"best_return"}:
            combined[name] = float(np.max(values))
        elif name in {"collapse_rate"}:
            combined[name] = float(np.max(values))
        else:
            combined[name] = float(np.mean(values))
    return combined


def numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    names = set(rows[0])
    for row in rows[1:]:
        names &= set(row)
    metadata = {"actual_seed", "step"}
    return sorted(name for name in names if name not in metadata and isinstance(rows[0][name], int | float))


def duplicate_actual_seed_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    duplicates = []
    for group in groups:
        seed = group[0]["actual_seed"]
        if seed is not None and len(group) > 1:
            duplicates.append(
                {
                    "actual_seed": int(seed),
                    "num_runs": len(group),
                    "run_dirs": [run["run_dir"] for run in group],
                }
            )
    return duplicates


def eval_episode_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"final_eval_episode_rows": 0}
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    near = np.asarray([row["near_upright_fraction"] for row in rows], dtype=np.float64)
    streaks = np.asarray([row["not_near_upright_streak"] for row in rows], dtype=np.float64)
    return {
        "final_eval_episode_rows": int(len(rows)),
        "final_eval_episode_return_min": float(np.min(returns)),
        "final_eval_episode_return_mean": float(np.mean(returns)),
        "final_eval_episode_near_upright_fraction_mean": float(np.mean(near)),
        "final_eval_episode_not_near_upright_streak_max": float(np.max(streaks)),
    }


def eval_seed_difficulty_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {
            "final_eval_unique_seed_count": 0,
            "final_eval_hardest_seeds": [],
        }

    by_eval_seed: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        eval_seed = int(row.get("seed", row.get("episode_index", len(by_eval_seed))))
        by_eval_seed.setdefault(eval_seed, []).append(row)

    eval_seed_rows = []
    for eval_seed, seed_rows in sorted(by_eval_seed.items()):
        returns = np.asarray([row["return"] for row in seed_rows], dtype=np.float64)
        success = np.asarray([row.get("success", row.get("return_success", 0.0)) for row in seed_rows], dtype=np.float64)
        strict_success = np.asarray([row.get("strict_success", row.get("success", 0.0)) for row in seed_rows], dtype=np.float64)
        eval_seed_rows.append(
            {
                "eval_seed": int(eval_seed),
                "num_actual_seeds": int(len(seed_rows)),
                "mean_return": float(np.mean(returns)),
                "min_return": float(np.min(returns)),
                "success_rate": float(np.mean(success)),
                "strict_success_rate": float(np.mean(strict_success)),
            }
        )

    mean_returns = np.asarray([row["mean_return"] for row in eval_seed_rows], dtype=np.float64)
    success_rates = np.asarray([row["success_rate"] for row in eval_seed_rows], dtype=np.float64)
    strict_success_rates = np.asarray([row["strict_success_rate"] for row in eval_seed_rows], dtype=np.float64)
    hardest = sorted(eval_seed_rows, key=lambda row: (row["strict_success_rate"], row["success_rate"], row["mean_return"]))[:10]

    return {
        "final_eval_unique_seed_count": int(len(eval_seed_rows)),
        "final_eval_seed_return_mean_min": float(np.min(mean_returns)),
        "final_eval_seed_return_mean_p10": float(np.percentile(mean_returns, 10)),
        "final_eval_seed_success_rate_min": float(np.min(success_rates)),
        "final_eval_seed_success_rate_p10": float(np.percentile(success_rates, 10)),
        "final_eval_seed_strict_success_rate_min": float(np.min(strict_success_rates)),
        "final_eval_seed_strict_success_rate_p10": float(np.percentile(strict_success_rates, 10)),
        "final_eval_hardest_seeds": hardest,
    }


def has_post_success_collapse(evaluations: list[dict[str, Any]]) -> bool:
    seen_success = False
    for evaluation in evaluations:
        if seen_success and evaluation.get("collapse_rate", 0.0) > 0.0:
            return True
        if evaluation.get("success_rate", 0.0) > 0.0:
            seen_success = True
    return False


def bootstrap_mean_ci(values: np.ndarray, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


if __name__ == "__main__":
    main()
