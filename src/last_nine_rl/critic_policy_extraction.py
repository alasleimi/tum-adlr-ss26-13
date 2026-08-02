from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import resolve_device
from last_nine_rl.pendulum_grid import pendulum_obs_batch, pendulum_step_batch
from last_nine_rl.sac import SACAgent
from last_nine_rl.simba_v2 import project_simba_weights_to_unit_norm


def main() -> None:
    args = parse_args()
    result = extract_critic_policy(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a same-size actor from a learned continuous-action critic without reference labels."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-size", type=int, default=100_000)
    parser.add_argument("--eval-dataset-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-actions", type=int, default=41)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--selected-weight", type=float, default=4.0)
    parser.add_argument("--trainable-actor", choices=["all", "mean_head", "last_layer"], default="all")
    parser.add_argument("--velocity-limit", type=float, default=8.0)
    parser.add_argument("--near-down-fraction", type=float, default=0.25)
    parser.add_argument("--near-down-velocity-limit", type=float, default=1.0)
    parser.add_argument("--near-upright-fraction", type=float, default=0.25)
    parser.add_argument("--near-upright-velocity-limit", type=float, default=2.0)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-trajectories", type=int, default=512)
    parser.add_argument("--dagger-horizon", type=int, default=200)
    parser.add_argument("--dagger-train-epochs", type=int, default=5)
    parser.add_argument("--dagger-initial-velocity-limit", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-dataset", action="store_true")
    return parser.parse_args()


def extract_critic_policy(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    agent, config, source_payload = load_agent_from_run(
        args.source_run,
        device=device,
        checkpoint=str(args.checkpoint),
        load_optimizers=False,
    )
    config.sac.device = device
    config.seed = int(args.seed)
    config.to_json(run_dir / "config.json")
    for critic in agent.q_networks:
        critic.requires_grad_(False)
        critic.eval()

    rng = np.random.default_rng(int(args.seed))
    train_observations = sample_pendulum_states(
        size=int(args.dataset_size),
        rng=rng,
        velocity_limit=float(args.velocity_limit),
        near_down_fraction=float(args.near_down_fraction),
        near_down_velocity_limit=float(args.near_down_velocity_limit),
        near_upright_fraction=float(args.near_upright_fraction),
        near_upright_velocity_limit=float(args.near_upright_velocity_limit),
    )
    eval_observations = sample_pendulum_states(
        size=int(args.eval_dataset_size),
        rng=rng,
        velocity_limit=float(args.velocity_limit),
        near_down_fraction=float(args.near_down_fraction),
        near_down_velocity_limit=float(args.near_down_velocity_limit),
        near_upright_fraction=float(args.near_upright_fraction),
        near_upright_velocity_limit=float(args.near_upright_velocity_limit),
    )
    train_targets, train_selected = critic_search_targets(
        agent,
        train_observations,
        num_actions=int(args.num_actions),
        margin=float(args.margin),
        batch_size=int(args.batch_size),
    )
    eval_targets, eval_selected = critic_search_targets(
        agent,
        eval_observations,
        num_actions=int(args.num_actions),
        margin=float(args.margin),
        batch_size=int(args.batch_size),
    )

    best_metric = math.inf
    best_epoch = 0
    best_actor_state = _copy_actor_state(agent)
    metrics: list[dict[str, float]] = []
    initial_metrics = action_fit_metrics(
        agent,
        eval_observations,
        eval_targets,
        eval_selected,
        batch_size=int(args.batch_size),
        selected_weight=float(args.selected_weight),
    )
    initial_metrics["epoch"] = 0.0
    metrics.append(initial_metrics)

    trainable_parameters = configure_trainable_actor(agent, str(args.trainable_actor))
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_actor_epoch(
            agent,
            optimizer,
            train_observations,
            train_targets,
            train_selected,
            batch_size=int(args.batch_size),
            selected_weight=float(args.selected_weight),
            rng=rng,
        )
        train_metrics["epoch"] = float(epoch)
        if epoch == 1 or epoch == int(args.epochs) or epoch % int(args.eval_every_epochs) == 0:
            eval_metrics = action_fit_metrics(
                agent,
                eval_observations,
                eval_targets,
                eval_selected,
                batch_size=int(args.batch_size),
                selected_weight=float(args.selected_weight),
            )
            train_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            selection_metric = eval_metrics["weighted_mse"]
            if selection_metric < best_metric:
                best_metric = selection_metric
                best_epoch = epoch
                best_actor_state = _copy_actor_state(agent)
        metrics.append(train_metrics)

    if int(args.epochs) > 0:
        agent.actor.load_state_dict(best_actor_state)
    dagger_summaries: list[dict[str, float]] = []
    total_epochs = int(args.epochs)
    for dagger_iteration in range(1, int(args.dagger_iterations) + 1):
        dagger_observations = collect_actor_rollout_states(
            agent,
            trajectories=int(args.dagger_trajectories),
            horizon=int(args.dagger_horizon),
            initial_velocity_limit=float(args.dagger_initial_velocity_limit),
            rng=rng,
        )
        dagger_targets, dagger_selected = critic_search_targets(
            agent,
            dagger_observations,
            num_actions=int(args.num_actions),
            margin=float(args.margin),
            batch_size=int(args.batch_size),
        )
        before = action_fit_metrics(
            agent,
            dagger_observations,
            dagger_targets,
            dagger_selected,
            batch_size=int(args.batch_size),
            selected_weight=float(args.selected_weight),
        )
        for local_epoch in range(1, int(args.dagger_train_epochs) + 1):
            total_epochs += 1
            dagger_metrics = train_actor_epoch(
                agent,
                optimizer,
                dagger_observations,
                dagger_targets,
                dagger_selected,
                batch_size=int(args.batch_size),
                selected_weight=float(args.selected_weight),
                rng=rng,
            )
            dagger_metrics.update(
                {
                    "epoch": float(total_epochs),
                    "dagger_iteration": float(dagger_iteration),
                    "dagger_local_epoch": float(local_epoch),
                    "dagger_selected_fraction": float(dagger_selected.mean()),
                }
            )
            metrics.append(dagger_metrics)
        after = action_fit_metrics(
            agent,
            dagger_observations,
            dagger_targets,
            dagger_selected,
            batch_size=int(args.batch_size),
            selected_weight=float(args.selected_weight),
        )
        dagger_summaries.append(
            {
                "iteration": float(dagger_iteration),
                "states": float(len(dagger_observations)),
                "selected_fraction": float(dagger_selected.mean()),
                "before_weighted_mse": before["weighted_mse"],
                "before_selected_action_mae": before["selected_action_mae"],
                "after_weighted_mse": after["weighted_mse"],
                "after_selected_action_mae": after["selected_action_mae"],
            }
        )
    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    agent.save_checkpoint(
        checkpoint_path,
        extra={
            "global_step": int(source_payload.get("extra", {}).get("global_step", 0)),
            "critic_policy_extraction": True,
            "uses_reference_actions": False,
            "source_run": str(args.source_run),
            "source_checkpoint": str(args.checkpoint),
            "num_actions": int(args.num_actions),
            "margin": float(args.margin),
            "selected_weight": float(args.selected_weight),
            "trainable_actor": str(args.trainable_actor),
            "best_epoch": int(best_epoch),
            "best_eval_weighted_mse": float(best_metric),
            "dagger_iterations": int(args.dagger_iterations),
        },
    )
    _write_metrics(run_dir / "metrics.csv", metrics)
    if args.save_dataset:
        np.savez_compressed(
            run_dir / "critic_policy_dataset.npz",
            train_observations=train_observations,
            train_targets=train_targets,
            train_selected=train_selected,
            eval_observations=eval_observations,
            eval_targets=eval_targets,
            eval_selected=eval_selected,
        )

    final_metrics = action_fit_metrics(
        agent,
        eval_observations,
        eval_targets,
        eval_selected,
        batch_size=int(args.batch_size),
        selected_weight=float(args.selected_weight),
    )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "source_run": str(args.source_run),
        "source_checkpoint": str(args.checkpoint),
        "uses_reference_actions": False,
        "dataset_size": int(args.dataset_size),
        "eval_dataset_size": int(args.eval_dataset_size),
        "num_actions": int(args.num_actions),
        "margin": float(args.margin),
        "selected_weight": float(args.selected_weight),
        "trainable_actor": str(args.trainable_actor),
        "train_selected_fraction": float(train_selected.mean()),
        "eval_selected_fraction": float(eval_selected.mean()),
        "best_epoch": int(best_epoch),
        "dagger_iterations": int(args.dagger_iterations),
        "dagger": dagger_summaries,
        "initial_eval": initial_metrics,
        "final_eval": final_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def collect_actor_rollout_states(
    agent: SACAgent,
    trajectories: int,
    horizon: int,
    initial_velocity_limit: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if trajectories <= 0 or horizon <= 0:
        raise ValueError("trajectories and horizon must be positive")
    theta = rng.uniform(-math.pi, math.pi, size=trajectories)
    theta_dot = rng.uniform(-initial_velocity_limit, initial_velocity_limit, size=trajectories)
    observations: list[np.ndarray] = []
    for _step in range(horizon):
        obs = pendulum_obs_batch(theta, theta_dot)
        observations.append(obs)
        actions = agent.act_batch(obs, deterministic=True).reshape(-1)
        theta, theta_dot, _reward = pendulum_step_batch(theta, theta_dot, actions)
    return np.concatenate(observations, axis=0)


def configure_trainable_actor(agent: SACAgent, mode: str) -> list[torch.nn.Parameter]:
    if mode not in {"all", "mean_head", "last_layer"}:
        raise ValueError(f"unknown trainable actor mode: {mode}")
    for name, parameter in agent.actor.named_parameters():
        if mode == "all":
            trainable = True
        elif mode == "mean_head":
            trainable = name.startswith("mean_")
        else:
            trainable = name.startswith("mean_w2.") or name == "mean_bias"
        parameter.requires_grad_(trainable)
    parameters = [parameter for parameter in agent.actor.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"trainable actor mode selected no parameters: {mode}")
    return parameters


def sample_pendulum_states(
    size: int,
    rng: np.random.Generator,
    velocity_limit: float,
    near_down_fraction: float,
    near_down_velocity_limit: float,
    near_upright_fraction: float,
    near_upright_velocity_limit: float,
) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive")
    if near_down_fraction < 0.0 or near_upright_fraction < 0.0:
        raise ValueError("mixture fractions must be nonnegative")
    if near_down_fraction + near_upright_fraction > 1.0:
        raise ValueError("mixture fractions must sum to at most one")

    down_count = int(round(size * near_down_fraction))
    upright_count = int(round(size * near_upright_fraction))
    uniform_count = size - down_count - upright_count
    theta_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    if uniform_count:
        theta_parts.append(rng.uniform(-math.pi, math.pi, size=uniform_count))
        velocity_parts.append(rng.uniform(-velocity_limit, velocity_limit, size=uniform_count))
    if down_count:
        abs_theta = rng.uniform(math.radians(150.0), math.pi, size=down_count)
        theta_parts.append(abs_theta * rng.choice(np.asarray([-1.0, 1.0]), size=down_count))
        velocity_parts.append(rng.uniform(-near_down_velocity_limit, near_down_velocity_limit, size=down_count))
    if upright_count:
        theta_parts.append(rng.uniform(-math.radians(35.0), math.radians(35.0), size=upright_count))
        velocity_parts.append(rng.uniform(-near_upright_velocity_limit, near_upright_velocity_limit, size=upright_count))

    theta = np.concatenate(theta_parts)
    velocity = np.concatenate(velocity_parts)
    order = rng.permutation(size)
    theta = theta[order]
    velocity = velocity[order]
    return np.column_stack((np.cos(theta), np.sin(theta), velocity)).astype(np.float32)


def critic_search_targets(
    agent: SACAgent,
    observations: np.ndarray,
    num_actions: int,
    margin: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_actions = _batched_actor_actions(agent, observations, batch_size=batch_size)
    target_actions = agent.act_batch_critic_search(
        observations,
        num_actions=num_actions,
        margin=margin,
        batch_size=batch_size,
    )
    selected = np.max(np.abs(target_actions - base_actions), axis=1) > 1e-6
    return target_actions.astype(np.float32, copy=False), selected


def critic_teacher_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
    action_scale: torch.Tensor,
    selected_weight: float,
) -> torch.Tensor:
    per_sample = ((predictions - targets) / action_scale).pow(2).mean(dim=1)
    weights = 1.0 + selected.to(per_sample.dtype) * (float(selected_weight) - 1.0)
    return (weights * per_sample).sum() / weights.sum()


def train_actor_epoch(
    agent: SACAgent,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    batch_size: int,
    selected_weight: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    agent.actor.train()
    action_scale = torch.clamp(agent.actor.action_scale.abs(), min=1e-6).view(1, -1)
    losses: list[float] = []
    for start in range(0, len(observations), batch_size):
        indices = rng.integers(0, len(observations), size=min(batch_size, len(observations) - start))
        obs = torch.as_tensor(observations[indices], dtype=torch.float32, device=agent.device)
        obs = agent._normalize_obs_tensor(obs)
        target = torch.as_tensor(targets[indices], dtype=torch.float32, device=agent.device)
        selected_tensor = torch.as_tensor(selected[indices], dtype=torch.bool, device=agent.device)
        predictions = _actor_mean(agent, obs)
        loss = critic_teacher_loss(
            predictions,
            target,
            selected_tensor,
            action_scale=action_scale,
            selected_weight=selected_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if agent.cfg.simba_weight_projection:
            project_simba_weights_to_unit_norm(agent.actor)
        losses.append(float(loss.detach().cpu()))
    return {"weighted_mse": float(np.mean(losses))}


def action_fit_metrics(
    agent: SACAgent,
    observations: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    batch_size: int,
    selected_weight: float,
) -> dict[str, float]:
    predictions = _batched_actor_actions(agent, observations, batch_size=batch_size)
    error = np.abs(predictions - targets)
    squared = np.square(error / np.asarray(agent.actor.action_scale.detach().cpu()))
    per_sample_mse = squared.mean(axis=1)
    weights = 1.0 + selected.astype(np.float32) * (float(selected_weight) - 1.0)
    selected_error = error[selected] if selected.any() else np.zeros((1, error.shape[1]), dtype=np.float32)
    return {
        "weighted_mse": float(np.sum(weights * per_sample_mse) / np.sum(weights)),
        "action_mae": float(error.mean()),
        "selected_action_mae": float(selected_error.mean()),
        "selected_fraction": float(selected.mean()),
    }


def _actor_mean(agent: SACAgent, normalized_observations: torch.Tensor) -> torch.Tensor:
    raw_mean, _log_std = agent.actor(normalized_observations)
    return torch.tanh(raw_mean) * agent.actor.action_scale + agent.actor.action_bias


def _batched_actor_actions(agent: SACAgent, observations: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    agent.actor.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=agent.device,
            )
            outputs.append(_actor_mean(agent, agent._normalize_obs_tensor(obs)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _copy_actor_state(agent: SACAgent) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in agent.actor.state_dict().items()}


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = [
        "dataset_size",
        "eval_dataset_size",
        "batch_size",
        "lr",
        "num_actions",
        "selected_weight",
        "velocity_limit",
        "near_down_velocity_limit",
        "near_upright_velocity_limit",
        "eval_every_epochs",
    ]
    for name in positive_names:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if float(args.margin) < 0.0:
        raise ValueError("--margin must be nonnegative")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be nonnegative")
    if float(args.near_down_fraction) + float(args.near_upright_fraction) > 1.0:
        raise ValueError("state mixture fractions must sum to at most one")
    if int(args.epochs) < 0 or int(args.dagger_iterations) < 0:
        raise ValueError("epoch and DAgger counts must be nonnegative")
    if int(args.epochs) == 0 and int(args.dagger_iterations) == 0:
        raise ValueError("at least one static epoch or DAgger iteration is required")
    if int(args.dagger_iterations) > 0:
        for name in ("dagger_trajectories", "dagger_horizon", "dagger_train_epochs"):
            if int(getattr(args, name)) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if float(args.dagger_initial_velocity_limit) <= 0.0:
            raise ValueError("--dagger-initial-velocity-limit must be positive")


def _write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
