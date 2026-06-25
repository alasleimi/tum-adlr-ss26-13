from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.simba_v2 import SimbaCategoricalQNetwork


class _SACNBatch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    trajectory_observations: torch.Tensor
    trajectory_actions: torch.Tensor
    trajectory_next_observations: torch.Tensor
    trajectory_rewards: torch.Tensor
    trajectory_dones: torch.Tensor
    trajectory_action_log_probs: torch.Tensor


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent, config, _payload = load_agent_from_run(run_dir, device=args.device)
    replay = load_replay(run_dir / "replay_final.npz")
    horizons = [int(value) for value in args.horizons]
    summaries = []
    rows = []
    rng = np.random.default_rng(args.seed)

    for horizon in horizons:
        starts = sample_valid_starts(replay, horizon, args.samples, rng, start_filter=args.start_filter)
        target_parts = []
        weight_parts = []
        log_omega_parts = []
        q_pred_parts = []
        for chunk_start in range(0, len(starts), int(args.batch_size)):
            chunk_starts = starts[chunk_start : chunk_start + int(args.batch_size)]
            batch = make_batch(replay, chunk_starts, horizon, agent.device)
            with torch.no_grad():
                common = agent._sacn_common_target_inputs(batch)
                targets = sacn_targets(agent, common)
                start_observations = agent._normalize_obs_tensor(batch.observations)
                q_pred = torch.minimum(
                    agent.q1(start_observations, batch.actions).view(-1),
                    agent.q2(start_observations, batch.actions).view(-1),
                )
            target_parts.append(targets.detach().cpu().numpy())
            weight_parts.append(common["weights"].detach().cpu().numpy())
            log_omega_parts.append(common["log_omega"].detach().cpu().numpy())
            q_pred_parts.append(q_pred.detach().cpu().numpy())
            if agent.device.type == "cuda":
                torch.cuda.empty_cache()

        target_np = np.concatenate(target_parts, axis=0)
        weight_np = np.concatenate(weight_parts, axis=0)
        log_omega_np = np.concatenate(log_omega_parts, axis=0)
        q_pred_np = np.concatenate(q_pred_parts, axis=0)
        hard_mask = hard_state_mask(replay["observations"][starts])
        age_fraction = (replay["steps"].max() - replay["steps"][starts]) / max(float(replay["steps"].max()), 1.0)

        summary = summarize_horizon(
            horizon=horizon,
            targets=target_np,
            weights=weight_np,
            log_omega=log_omega_np,
            q_pred=q_pred_np,
            hard_mask=hard_mask,
            age_fraction=age_fraction,
        )
        summaries.append(summary)
        rows.extend(flatten_rows(horizon, summary))

    (out_dir / "sacn_horizon_summary.json").write_text(
        json.dumps(
            {
                "run": str(run_dir),
                "seed": config.seed,
                "samples": args.samples,
                "horizons": horizons,
                "summary": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "sacn_horizon_summary.csv", rows)
    write_plot(out_dir / "sacn_horizon_summary.png", summaries)
    print(json.dumps({"run": str(run_dir), "summary": summaries}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SACn target and importance-weight behavior by horizon.")
    parser.add_argument("--run", required=True, help="Run directory with config.json, final checkpoint, and replay_final.npz.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--start-filter",
        choices=("all", "hard", "recent_quarter", "oldest_quarter"),
        default="all",
        help="Restrict sampled sequence starts before computing grouped diagnostics.",
    )
    return parser.parse_args()


def load_replay(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"Replay file not found: {path}")
    z = np.load(path)
    return {
        "observations": np.asarray(z["observations"], dtype=np.float32).reshape(-1, 3),
        "next_observations": np.asarray(z["next_observations"], dtype=np.float32).reshape(-1, 3),
        "actions": np.asarray(z["actions"], dtype=np.float32).reshape(-1, 1),
        "rewards": np.asarray(z["rewards"], dtype=np.float32).reshape(-1, 1),
        "dones": np.asarray(z["dones"], dtype=np.float32).reshape(-1, 1),
        "episode_ids": np.asarray(z["episode_ids"], dtype=np.int64).reshape(-1),
        "steps": np.asarray(z["steps"], dtype=np.int64).reshape(-1),
        "action_log_probs": np.asarray(z["action_log_probs"], dtype=np.float32).reshape(-1),
    }


def sample_valid_starts(
    replay: dict[str, np.ndarray],
    horizon: int,
    samples: int,
    rng: np.random.Generator,
    start_filter: str = "all",
) -> np.ndarray:
    size = int(replay["steps"].shape[0])
    offsets = np.arange(horizon, dtype=np.int64).reshape(1, -1)
    candidates = np.arange(0, max(0, size - horizon + 1), dtype=np.int64)
    positions = candidates.reshape(-1, 1) + offsets
    same_episode = np.all(replay["episode_ids"][positions] == replay["episode_ids"][positions[:, :1]], axis=1)
    sequential = np.all(replay["steps"][positions] == replay["steps"][positions[:, :1]] + offsets, axis=1)
    finite_logp = np.all(np.isfinite(replay["action_log_probs"][positions]), axis=1)
    valid = candidates[same_episode & sequential & finite_logp]
    if start_filter != "all":
        valid = filter_valid_starts(replay, valid, start_filter)
    if valid.size == 0:
        raise SystemExit(f"No valid SACn sequences for horizon={horizon}, start_filter={start_filter}")
    return rng.choice(valid, size=int(samples), replace=valid.size < int(samples)).astype(np.int64, copy=False)


def filter_valid_starts(replay: dict[str, np.ndarray], valid: np.ndarray, start_filter: str) -> np.ndarray:
    if start_filter == "hard":
        return valid[hard_state_mask(replay["observations"][valid])]
    current_step = float(np.max(replay["steps"]))
    age_fraction = (current_step - replay["steps"][valid]) / max(current_step, 1.0)
    if start_filter == "recent_quarter":
        return valid[age_fraction <= 0.25]
    if start_filter == "oldest_quarter":
        return valid[age_fraction >= 0.75]
    raise ValueError(f"Unknown start_filter: {start_filter}")


def make_batch(
    replay: dict[str, np.ndarray],
    starts: np.ndarray,
    horizon: int,
    device: torch.device,
) -> _SACNBatch:
    offsets = np.arange(horizon, dtype=np.int64).reshape(1, -1)
    positions = starts.reshape(-1, 1) + offsets
    tensors = (
        replay["observations"][starts],
        replay["actions"][starts],
        replay["observations"][positions],
        replay["actions"][positions],
        replay["next_observations"][positions],
        replay["rewards"][positions],
        replay["dones"][positions],
        replay["action_log_probs"][positions][..., None],
    )
    return _SACNBatch(*[torch.as_tensor(x, dtype=torch.float32, device=device) for x in tensors])


def sacn_targets(agent: Any, common: dict[str, Any]) -> torch.Tensor:
    if isinstance(agent.q1, SimbaCategoricalQNetwork):
        next_q1 = agent.q1_target(common["successor_observations"], common["successor_actions"])
        next_q2 = agent.q2_target(common["successor_observations"], common["successor_actions"])
    else:
        next_q1 = agent.q1_target(common["successor_observations"], common["successor_actions"])
        next_q2 = agent.q2_target(common["successor_observations"], common["successor_actions"])
    bootstrap = torch.minimum(next_q1, next_q2).view(common["batch_size"], common["n_step"])
    return common["offsets"] + common["discounts"] * bootstrap


def hard_state_mask(observations: np.ndarray) -> np.ndarray:
    theta = np.arctan2(observations[:, 1], observations[:, 0])
    theta_dot = observations[:, 2]
    return (np.abs(theta) >= math.radians(150.0)) & (np.abs(theta_dot) <= 1.0)


def summarize_horizon(
    horizon: int,
    targets: np.ndarray,
    weights: np.ndarray,
    log_omega: np.ndarray,
    q_pred: np.ndarray,
    hard_mask: np.ndarray,
    age_fraction: np.ndarray,
) -> dict[str, Any]:
    return {
        "horizon": horizon,
        "all": summarize_group(targets, weights, log_omega, q_pred, np.ones(targets.shape[0], dtype=bool), age_fraction),
        "hard_down_abs_theta_ge_150_abs_vel_le_1": summarize_group(
            targets, weights, log_omega, q_pred, hard_mask, age_fraction
        ),
        "recent_quarter": summarize_group(targets, weights, log_omega, q_pred, age_fraction <= 0.25, age_fraction),
        "oldest_quarter": summarize_group(targets, weights, log_omega, q_pred, age_fraction >= 0.75, age_fraction),
    }


def summarize_group(
    targets: np.ndarray,
    weights: np.ndarray,
    log_omega: np.ndarray,
    q_pred: np.ndarray,
    mask: np.ndarray,
    age_fraction: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not bool(np.any(mask)):
        return {"count": 0.0}
    t = targets[mask]
    w = weights[mask]
    lo = log_omega[mask]
    q = q_pred[mask]
    age = age_fraction[mask]
    target_delta = t[:, -1] - t[:, 0]
    horizon_ess_fraction = effective_sample_fraction(w, axis=0)
    return {
        "count": float(t.shape[0]),
        "target_first_mean": fmean(t[:, 0]),
        "target_last_mean": fmean(t[:, -1]),
        "target_last_minus_first_mean": fmean(target_delta),
        "target_last_minus_first_weighted_by_last_mean": fweighted_mean(target_delta, w[:, -1]),
        "target_last_minus_first_median": fmedian(target_delta),
        "target_last_minus_first_p10": fpct(target_delta, 10),
        "target_last_minus_first_p90": fpct(target_delta, 90),
        "q_pred_mean": fmean(q),
        "target_first_minus_q_mean": fmean(t[:, 0] - q),
        "target_last_minus_q_mean": fmean(t[:, -1] - q),
        "weight_mean": fmean(w),
        "weight_last_mean": fmean(w[:, -1]),
        "weight_last_median": fmedian(w[:, -1]),
        "weight_last_p10": fpct(w[:, -1], 10),
        "weight_last_p90": fpct(w[:, -1], 90),
        "weight_last_le_1e_3_rate": float(np.mean(w[:, -1] <= 1e-3)),
        "weight_ess_first_fraction": float(horizon_ess_fraction[0]),
        "weight_ess_last_fraction": float(horizon_ess_fraction[-1]),
        "weight_ess_min_fraction": float(np.min(horizon_ess_fraction)),
        "weight_active_horizon_count": float(np.sum(np.mean(np.abs(w), axis=0) > 1e-8)),
        "log_omega_last_mean": fmean(lo[:, -1]),
        "log_omega_last_median": fmedian(lo[:, -1]),
        "log_omega_last_p10": fpct(lo[:, -1], 10),
        "log_omega_last_p90": fpct(lo[:, -1], 90),
        "age_fraction_mean": fmean(age),
    }


def fmean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def fweighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return float("nan")
    return float(np.sum(values64 * weights64) / weight_sum)


def effective_sample_fraction(weights: np.ndarray, axis: int) -> np.ndarray:
    weights64 = np.asarray(weights, dtype=np.float64)
    numerator = np.square(np.sum(weights64, axis=axis))
    denominator = np.sum(np.square(weights64), axis=axis)
    count = weights64.shape[axis]
    ess = numerator / np.maximum(denominator, 1e-12)
    return ess / max(float(count), 1.0)


def fmedian(values: np.ndarray) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def fpct(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def flatten_rows(horizon: int, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group, stats in summary.items():
        if group == "horizon":
            continue
        row = {"horizon": horizon, "group": group}
        row.update(stats)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


def write_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    horizons = np.asarray([item["horizon"] for item in summaries], dtype=np.float64)
    groups = [
        ("all", "all"),
        ("hard_down_abs_theta_ge_150_abs_vel_le_1", "hard"),
        ("recent_quarter", "recent"),
        ("oldest_quarter", "old"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for group, label in groups:
        deltas = [item[group].get("target_last_minus_first_mean", np.nan) for item in summaries]
        weight_last = [item[group].get("weight_last_mean", np.nan) for item in summaries]
        collapsed = [item[group].get("weight_last_le_1e_3_rate", np.nan) for item in summaries]
        ess_last = [item[group].get("weight_ess_last_fraction", np.nan) for item in summaries]
        axes[0].plot(horizons, deltas, marker="o", label=label)
        axes[1].plot(horizons, weight_last, marker="o", label=label)
        axes[2].plot(horizons, collapsed, marker="o", label=label)
        axes[3].plot(horizons, ess_last, marker="o", label=label)
    axes[0].set_title("Target last - first")
    axes[1].set_title("Last-horizon weight mean")
    axes[2].set_title("Last weight <= 1e-3")
    axes[3].set_title("Last-horizon ESS fraction")
    for ax in axes:
        ax.set_xlabel("horizon")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("scaled Q units")
    axes[1].set_ylabel("weight")
    axes[2].set_ylabel("rate")
    axes[3].set_ylabel("ESS / batch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
