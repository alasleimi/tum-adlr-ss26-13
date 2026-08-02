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

    agent, config, payload = load_agent_from_run(run_dir, device=args.device)
    update_step = resolve_update_step(payload, args.update_step)
    require_action_log_probs = config.sac.sacn_importance_mode == "density"
    replay = load_replay(
        run_dir / "replay_final.npz",
        require_action_log_probs=require_action_log_probs,
    )
    horizons = [int(value) for value in args.horizons]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise SystemExit("--horizons must contain positive integers")
    if int(args.samples) <= 0:
        raise SystemExit("--samples must be positive")
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    summaries = []
    rows = []
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(int(args.seed))
    if agent.device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    # Use the same starts for every requested horizon so changes across horizons
    # are not confounded by a different replay sample.
    paired_starts = sample_valid_starts(
        replay,
        max(horizons),
        args.samples,
        rng,
        start_filter=args.start_filter,
        require_action_log_probs=require_action_log_probs,
    )

    for horizon in horizons:
        starts = paired_starts
        target_parts = []
        effective_weight_parts = []
        importance_weight_parts = []
        horizon_mask_parts = []
        horizon_loss_weight_parts = []
        log_omega_parts = []
        q_pred_parts = []
        for chunk_start in range(0, len(starts), int(args.batch_size)):
            chunk_starts = starts[chunk_start : chunk_start + int(args.batch_size)]
            batch = make_batch(replay, chunk_starts, horizon, agent.device)
            with torch.no_grad():
                importance_weights, _raw_log_omega, _raw_log_importance_clip = (
                    agent._sacn_importance_weights(batch)
                )
                common = agent._sacn_common_target_inputs(batch, update_step=update_step)
                targets = sacn_targets(agent, common, update_step=update_step)
                start_observations = agent._normalize_obs_tensor(batch.observations)
                q_pred = torch.stack(
                    [
                        q_network(start_observations, batch.actions).view(-1)
                        for q_network in agent.q_networks
                    ],
                    dim=0,
                ).mean(dim=0)
            target_parts.append(targets.detach().cpu().numpy())
            effective_weight_parts.append(common["weights"].detach().cpu().numpy())
            importance_weight_parts.append(importance_weights.detach().cpu().numpy())
            chunk_size = len(chunk_starts)
            horizon_mask_parts.append(
                common["horizon_mask"].detach().cpu().numpy()[None, :].repeat(chunk_size, axis=0)
            )
            horizon_loss_weight_parts.append(
                common["horizon_loss_weights"].detach().cpu().numpy()[None, :].repeat(chunk_size, axis=0)
            )
            log_omega_parts.append(common["log_omega"].detach().cpu().numpy())
            q_pred_parts.append(q_pred.detach().cpu().numpy())
            if agent.device.type == "cuda":
                torch.cuda.empty_cache()

        target_np = np.concatenate(target_parts, axis=0)
        effective_weight_np = np.concatenate(effective_weight_parts, axis=0)
        importance_weight_np = np.concatenate(importance_weight_parts, axis=0)
        horizon_mask_np = np.concatenate(horizon_mask_parts, axis=0)
        horizon_loss_weight_np = np.concatenate(horizon_loss_weight_parts, axis=0)
        log_omega_np = np.concatenate(log_omega_parts, axis=0)
        q_pred_np = np.concatenate(q_pred_parts, axis=0)
        hard_mask = hard_state_mask(replay["observations"][starts])
        age_fraction = (replay["steps"].max() - replay["steps"][starts]) / max(float(replay["steps"].max()), 1.0)

        summary = summarize_horizon(
            horizon=horizon,
            targets=target_np,
            effective_weights=effective_weight_np,
            importance_weights=importance_weight_np,
            horizon_mask=horizon_mask_np,
            horizon_loss_weights=horizon_loss_weight_np,
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
                "diagnostic_batch_size": args.batch_size,
                "horizons": horizons,
                "paired_starts": True,
                "start_filter": args.start_filter,
                "require_action_log_probs": require_action_log_probs,
                "checkpoint_global_step": checkpoint_extra_value(payload, "global_step"),
                "checkpoint_update_step": update_step,
                "target_q_aggregation": config.sac.target_q_aggregation,
                "q_prediction_aggregation": "online_ensemble_mean",
                "redq_num_critics": len(agent.q_target_networks),
                "redq_target_subset_size": agent._redq_target_subset_size(),
                "sacn_importance_mode": config.sac.sacn_importance_mode,
                "sacn_target_mode": config.sac.sacn_target_mode,
                "sacn_horizon_lambda": config.sac.sacn_horizon_lambda,
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
        "--update-step",
        type=int,
        default=None,
        help=(
            "Training update index used for target-ensemble selection and scheduled shaping. "
            "Defaults to checkpoint extra.update_step."
        ),
    )
    parser.add_argument(
        "--start-filter",
        choices=("all", "hard", "recent_quarter", "oldest_quarter"),
        default="all",
        help="Restrict sampled sequence starts before computing grouped diagnostics.",
    )
    return parser.parse_args()


def resolve_update_step(payload: dict[str, Any], override: int | None) -> int:
    if override is not None:
        if int(override) < 0:
            raise SystemExit("--update-step must be nonnegative")
        return int(override)
    extra = payload.get("extra", {})
    value = extra.get("update_step") if isinstance(extra, dict) else None
    if value is None:
        raise SystemExit(
            "Checkpoint has no extra.update_step; pass --update-step explicitly so REDQ target selection "
            "and scheduled target construction can be reproduced."
        )
    return int(value)


def checkpoint_extra_value(payload: dict[str, Any], key: str) -> Any | None:
    extra = payload.get("extra", {})
    return extra.get(key) if isinstance(extra, dict) else None


def load_replay(path: Path, require_action_log_probs: bool = True) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"Replay file not found: {path}")
    with np.load(path) as z:
        observations = np.asarray(z["observations"], dtype=np.float32).reshape(-1, 3)
        if "action_log_probs" in z.files:
            action_log_probs = np.asarray(z["action_log_probs"], dtype=np.float32).reshape(-1)
        elif require_action_log_probs:
            raise SystemExit(
                f"Replay file has no action_log_probs required by SACn density weighting: {path}"
            )
        else:
            action_log_probs = np.full(observations.shape[0], np.nan, dtype=np.float32)
        return {
            "observations": observations,
            "next_observations": np.asarray(z["next_observations"], dtype=np.float32).reshape(-1, 3),
            "actions": np.asarray(z["actions"], dtype=np.float32).reshape(-1, 1),
            "rewards": np.asarray(z["rewards"], dtype=np.float32).reshape(-1, 1),
            "dones": np.asarray(z["dones"], dtype=np.float32).reshape(-1, 1),
            "episode_ids": np.asarray(z["episode_ids"], dtype=np.int64).reshape(-1),
            "steps": np.asarray(z["steps"], dtype=np.int64).reshape(-1),
            "action_log_probs": action_log_probs,
        }


def sample_valid_starts(
    replay: dict[str, np.ndarray],
    horizon: int,
    samples: int,
    rng: np.random.Generator,
    start_filter: str = "all",
    require_action_log_probs: bool = True,
) -> np.ndarray:
    size = int(replay["steps"].shape[0])
    offsets = np.arange(horizon, dtype=np.int64).reshape(1, -1)
    candidates = np.arange(0, max(0, size - horizon + 1), dtype=np.int64)
    positions = candidates.reshape(-1, 1) + offsets
    same_episode = np.all(replay["episode_ids"][positions] == replay["episode_ids"][positions[:, :1]], axis=1)
    sequential = np.all(replay["steps"][positions] == replay["steps"][positions[:, :1]] + offsets, axis=1)
    valid_mask = same_episode & sequential
    if require_action_log_probs:
        finite_logp = np.all(np.isfinite(replay["action_log_probs"][positions]), axis=1)
        valid_mask = valid_mask & finite_logp
    valid = candidates[valid_mask]
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


def sacn_targets(agent: Any, common: dict[str, Any], update_step: int) -> torch.Tensor:
    bootstrap = agent._target_q_aggregate(
        common["successor_observations"],
        common["successor_actions"],
        update_step=update_step,
    ).view(common["batch_size"], common["n_step"])
    return common["offsets"] + common["discounts"] * bootstrap


def hard_state_mask(observations: np.ndarray) -> np.ndarray:
    theta = np.arctan2(observations[:, 1], observations[:, 0])
    theta_dot = observations[:, 2]
    return (np.abs(theta) >= math.radians(150.0)) & (np.abs(theta_dot) <= 1.0)


def summarize_horizon(
    horizon: int,
    targets: np.ndarray,
    effective_weights: np.ndarray,
    importance_weights: np.ndarray,
    horizon_mask: np.ndarray,
    horizon_loss_weights: np.ndarray,
    log_omega: np.ndarray,
    q_pred: np.ndarray,
    hard_mask: np.ndarray,
    age_fraction: np.ndarray,
) -> dict[str, Any]:
    return {
        "horizon": horizon,
        "all": summarize_group(
            targets,
            effective_weights,
            importance_weights,
            horizon_mask,
            horizon_loss_weights,
            log_omega,
            q_pred,
            np.ones(targets.shape[0], dtype=bool),
            age_fraction,
        ),
        "hard_down_abs_theta_ge_150_abs_vel_le_1": summarize_group(
            targets,
            effective_weights,
            importance_weights,
            horizon_mask,
            horizon_loss_weights,
            log_omega,
            q_pred,
            hard_mask,
            age_fraction,
        ),
        "recent_quarter": summarize_group(
            targets,
            effective_weights,
            importance_weights,
            horizon_mask,
            horizon_loss_weights,
            log_omega,
            q_pred,
            age_fraction <= 0.25,
            age_fraction,
        ),
        "oldest_quarter": summarize_group(
            targets,
            effective_weights,
            importance_weights,
            horizon_mask,
            horizon_loss_weights,
            log_omega,
            q_pred,
            age_fraction >= 0.75,
            age_fraction,
        ),
    }


def summarize_group(
    targets: np.ndarray,
    effective_weights: np.ndarray,
    importance_weights: np.ndarray,
    horizon_mask: np.ndarray,
    horizon_loss_weights: np.ndarray,
    log_omega: np.ndarray,
    q_pred: np.ndarray,
    mask: np.ndarray,
    age_fraction: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not bool(np.any(mask)):
        return {"count": 0.0}
    t = targets[mask]
    w = effective_weights[mask]
    iw = importance_weights[mask]
    hm = horizon_mask[mask]
    hlw = horizon_loss_weights[mask]
    lo = log_omega[mask]
    q = q_pred[mask]
    age = age_fraction[mask]
    target_delta = t[:, -1] - t[:, 0]
    importance_ess_fraction = effective_sample_fraction(iw, axis=0)
    effective_ess_fraction = effective_sample_fraction(w, axis=0)
    horizon_loss_weight_sum = np.sum(hlw, axis=1)
    nominal_shares = hlw / np.maximum(horizon_loss_weight_sum[:, None], 1e-12)
    empirical_weight_sums = np.sum(w, axis=0)
    empirical_weight_total = float(np.sum(empirical_weight_sums))
    empirical_shares = empirical_weight_sums / max(empirical_weight_total, 1e-12)
    first_horizon_loss_weight = hlw[:, 0]
    last_horizon_loss_weight = hlw[:, -1]
    last_to_first = last_horizon_loss_weight / np.maximum(first_horizon_loss_weight, 1e-12)
    selected_horizons = np.mean(hlw > 0.0, axis=0) > 0.0
    selected_effective_ess = effective_ess_fraction[selected_horizons]
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
        # "importance" is the off-policy density correction before horizon
        # selection/decay. "effective" is the multiplier used in the loss.
        "importance_weight_mean": fmean(iw),
        "importance_weight_last_mean": fmean(iw[:, -1]),
        "effective_weight_mean": fmean(w),
        "effective_weight_last_mean": fmean(w[:, -1]),
        "effective_weight_last_median": fmedian(w[:, -1]),
        "effective_weight_last_p10": fpct(w[:, -1], 10),
        "effective_weight_last_p90": fpct(w[:, -1], 90),
        "effective_weight_last_le_1e_3_rate": float(np.mean(w[:, -1] <= 1e-3)),
        # Backward-compatible aliases for older consumers. These are effective
        # weights, not raw importance weights.
        "weight_mean": fmean(w),
        "weight_last_mean": fmean(w[:, -1]),
        "weight_last_median": fmedian(w[:, -1]),
        "weight_last_p10": fpct(w[:, -1], 10),
        "weight_last_p90": fpct(w[:, -1], 90),
        "weight_last_le_1e_3_rate": float(np.mean(w[:, -1] <= 1e-3)),
        "importance_weight_ess_first_fraction": float(importance_ess_fraction[0]),
        "importance_weight_ess_last_fraction": float(importance_ess_fraction[-1]),
        "importance_weight_ess_min_fraction": float(np.min(importance_ess_fraction)),
        "effective_weight_ess_first_fraction": float(effective_ess_fraction[0]),
        "effective_weight_ess_last_fraction": float(effective_ess_fraction[-1]),
        "effective_weight_ess_selected_min_fraction": float(np.min(selected_effective_ess)),
        # Backward-compatible aliases now exclude deliberately deselected
        # horizons from the minimum, so FastSACN's zero middle columns are not
        # misreported as importance-weight collapse.
        "weight_ess_first_fraction": float(effective_ess_fraction[0]),
        "weight_ess_last_fraction": float(effective_ess_fraction[-1]),
        "weight_ess_min_fraction": float(np.min(selected_effective_ess)),
        "weight_active_horizon_count": float(np.sum(np.mean(np.abs(w), axis=0) > 1e-8)),
        "horizon_support_active_count_mean": fmean(np.sum(hm > 0.0, axis=1)),
        "horizon_loss_selected_count_mean": fmean(np.sum(hlw > 0.0, axis=1)),
        "horizon_loss_weight_sum_mean": fmean(horizon_loss_weight_sum),
        "horizon_loss_weight_first_mean": fmean(first_horizon_loss_weight),
        "horizon_loss_weight_last_mean": fmean(last_horizon_loss_weight),
        "horizon_loss_weight_last_to_first_ratio_mean": fmean(last_to_first),
        "nominal_first_horizon_objective_share_mean": fmean(nominal_shares[:, 0]),
        "nominal_last_horizon_objective_share_mean": fmean(nominal_shares[:, -1]),
        "nominal_non_one_step_objective_share_mean": fmean(np.sum(nominal_shares[:, 1:], axis=1)),
        "empirical_first_horizon_objective_share": float(empirical_shares[0]),
        "empirical_last_horizon_objective_share": float(empirical_shares[-1]),
        "empirical_non_one_step_objective_share": float(np.sum(empirical_shares[1:])),
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
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for group, label in groups:
        deltas = [item[group].get("target_last_minus_first_mean", np.nan) for item in summaries]
        nominal_last_share = [
            item[group].get("nominal_last_horizon_objective_share_mean", np.nan) for item in summaries
        ]
        empirical_last_share = [
            item[group].get("empirical_last_horizon_objective_share", np.nan) for item in summaries
        ]
        collapsed = [item[group].get("effective_weight_last_le_1e_3_rate", np.nan) for item in summaries]
        ess_last = [item[group].get("weight_ess_last_fraction", np.nan) for item in summaries]
        axes[0].plot(horizons, deltas, marker="o", label=label)
        axes[1].plot(horizons, nominal_last_share, marker="o", label=label)
        axes[2].plot(horizons, empirical_last_share, marker="o", label=label)
        axes[3].plot(horizons, collapsed, marker="o", label=label)
        axes[4].plot(horizons, ess_last, marker="o", label=label)
    axes[0].set_title("Target last - first")
    axes[1].set_title("Nominal last objective share")
    axes[2].set_title("Empirical last objective share")
    axes[3].set_title("Effective last weight <= 1e-3")
    axes[4].set_title("Last-horizon ESS fraction")
    for ax in axes:
        ax.set_xlabel("horizon")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("scaled Q units")
    axes[1].set_ylabel("share")
    axes[2].set_ylabel("share")
    axes[3].set_ylabel("rate")
    axes[4].set_ylabel("ESS / batch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
