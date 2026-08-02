from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.envs import UprightDetector
from last_nine_rl.qsearch_lock import build_validation_dataset
try:
    from scripts.diagnose_critic_flatness import (
        pendulum_obs,
        rollout_after_first_action,
    )
    from scripts.train_pendulum_qregularized_dagger import (
        validation_reference_returns,
    )
except ModuleNotFoundError:
    from diagnose_critic_flatness import (  # type: ignore[no-redef]
        pendulum_obs,
        rollout_after_first_action,
    )
    from train_pendulum_qregularized_dagger import (  # type: ignore[no-redef]
        validation_reference_returns,
    )


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen critic action gradients with finite-difference "
            "realized-return gradients on a locked off-grid state sample."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments" / "protocols" / "pure_rl_offgrid_validation_protocol_20260722.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=230723)
    parser.add_argument("--action-delta", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    conditions = payload["conditions"]
    if len(conditions) < 2:
        raise ValueError("The comparison requires at least two conditions.")
    names = [str(item["name"]) for item in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Condition names must be unique.")
    for item in conditions:
        actors = list(item["actor_runs"])
        critics = item.get("critic_runs", "same")
        if len(actors) != 5:
            raise ValueError(f"{item['name']} must contain five actor runs.")
        if critics != "same" and len(critics) not in {1, 5}:
            raise ValueError(
                f"{item['name']} critic_runs must be 'same', one shared run, or five runs."
            )
    return conditions


def critic_runs(condition: dict[str, Any]) -> list[str]:
    actors = list(condition["actor_runs"])
    critics = condition.get("critic_runs", "same")
    if critics == "same":
        return actors
    values = list(critics)
    return values * 5 if len(values) == 1 else values


def critic_gradients(
    critic: Any, observations: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = torch.as_tensor(
        observations, dtype=torch.float32, device=critic.device
    )
    obs = critic._normalize_obs_tensor(obs)
    action = (
        torch.as_tensor(actions[:, None], dtype=torch.float32, device=critic.device)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    q1 = critic.q1(obs, action).view(-1)
    q2 = critic.q2(obs, action).view(-1)
    grad1 = torch.autograd.grad(
        q1.sum(), action, retain_graph=True, create_graph=False
    )[0]
    grad2 = torch.autograd.grad(
        q2.sum(), action, retain_graph=True, create_graph=False
    )[0]
    conservative = torch.where(q1[:, None] <= q2[:, None], grad1, grad2)
    return (
        conservative.detach().cpu().numpy().reshape(-1),
        grad1.detach().cpu().numpy().reshape(-1),
        grad2.detach().cpu().numpy().reshape(-1),
    )


def actor_rollout(
    actor: Any, theta: np.ndarray, velocity: np.ndarray, horizon: int
) -> dict[str, np.ndarray]:
    obs = pendulum_obs(theta, velocity)
    first = np.asarray(
        actor.act_batch(obs, deterministic=True), dtype=np.float64
    ).reshape(-1)
    returns = rollout_after_first_action(
        actor, theta, velocity, first, horizon, gamma=0.99
    )
    return {"action": first, **returns}


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    value = spearmanr(x[mask], y[mask]).statistic
    return float(value)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    q = np.asarray([float(row["critic_gradient"]) for row in rows])
    realized = np.asarray([float(row["realized_gradient"]) for row in rows])
    gain = np.asarray([float(row["critic_step_discounted_gain"]) for row in rows])
    near = np.asarray([int(row["near_reference"]) for row in rows], dtype=bool)
    twin = np.asarray([int(row["twin_gradient_sign_agreement"]) for row in rows])
    nonzero = np.abs(realized) > 1e-8
    sign = np.sign(q[nonzero]) == np.sign(realized[nonzero])
    result: dict[str, float | int] = {
        "rows": len(rows),
        "near_reference_rate": float(near.mean()),
        "gradient_spearman": safe_spearman(q, realized),
        "gradient_sign_agreement": float(sign.mean()) if len(sign) else float("nan"),
        "critic_step_beneficial_rate": float(np.mean(gain > 1e-8)),
        "critic_step_harmful_rate": float(np.mean(gain < -1e-8)),
        "critic_step_discounted_gain_mean": float(gain.mean()),
        "twin_gradient_sign_agreement": float(twin.mean()),
        "mean_abs_critic_gradient": float(np.mean(np.abs(q))),
        "mean_abs_realized_gradient": float(np.mean(np.abs(realized))),
        "actor_saturation_rate": float(
            np.mean([int(row["actor_saturated"]) for row in rows])
        ),
    }
    for label, mask in (("success", near), ("failure", ~near)):
        subset = np.flatnonzero(mask & nonzero)
        result[f"{label}_count"] = int(mask.sum())
        result[f"{label}_gradient_sign_agreement"] = (
            float(np.mean(np.sign(q[subset]) == np.sign(realized[subset])))
            if len(subset)
            else float("nan")
        )
        result[f"{label}_critic_step_beneficial_rate"] = (
            float(np.mean(gain[mask] > 1e-8)) if int(mask.sum()) else float("nan")
        )
    return result


def evaluate_condition(
    condition: dict[str, Any],
    theta: np.ndarray,
    velocity: np.ndarray,
    reference_returns: np.ndarray,
    epsilon: float,
    horizon: int,
    delta: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    critic_paths = critic_runs(condition)
    for replicate, (actor_value, critic_value) in enumerate(
        zip(condition["actor_runs"], critic_paths, strict=True)
    ):
        actor_path = resolve(actor_value)
        critic_path = resolve(critic_value)
        actor, actor_config, _ = load_agent_from_run(actor_path, device=device)
        critic = (
            actor
            if actor_path.resolve() == critic_path.resolve()
            else load_agent_from_run(critic_path, device=device)[0]
        )
        actions = actor_rollout(actor, theta, velocity, horizon)
        plus = np.clip(actions["action"] + delta, -2.0, 2.0)
        minus = np.clip(actions["action"] - delta, -2.0, 2.0)
        plus_returns = rollout_after_first_action(
            actor, theta, velocity, plus, horizon, gamma=0.99
        )
        minus_returns = rollout_after_first_action(
            actor, theta, velocity, minus, horizon, gamma=0.99
        )
        denominator = np.maximum(plus - minus, 1e-12)
        realized_gradient = (
            plus_returns["discounted"] - minus_returns["discounted"]
        ) / denominator
        q_gradient, q1_gradient, q2_gradient = critic_gradients(
            critic, pendulum_obs(theta, velocity), actions["action"]
        )
        critic_step = np.clip(
            actions["action"] + delta * np.sign(q_gradient), -2.0, 2.0
        )
        step_returns = rollout_after_first_action(
            actor, theta, velocity, critic_step, horizon, gamma=0.99
        )
        normalized_action = np.clip(actions["action"] / 2.0, -1.0, 1.0)
        tanh_derivative = 1.0 - normalized_action**2
        seed = int(actor_config.seed)
        for index in range(len(theta)):
            rows.append(
                {
                    "condition": str(condition["name"]),
                    "replicate": replicate,
                    "seed": seed,
                    "state_index": int(index),
                    "theta": float(theta[index]),
                    "velocity": float(velocity[index]),
                    "actor_action": float(actions["action"][index]),
                    "actor_raw_return": float(actions["raw"][index]),
                    "actor_discounted_return": float(actions["discounted"][index]),
                    "reference_raw_return": float(reference_returns[index]),
                    "near_reference": int(
                        actions["raw"][index] >= reference_returns[index] - epsilon
                    ),
                    "critic_gradient": float(q_gradient[index]),
                    "q1_gradient": float(q1_gradient[index]),
                    "q2_gradient": float(q2_gradient[index]),
                    "twin_gradient_sign_agreement": int(
                        np.sign(q1_gradient[index]) == np.sign(q2_gradient[index])
                    ),
                    "realized_gradient": float(realized_gradient[index]),
                    "critic_gradient_sign_agreement": int(
                        np.sign(q_gradient[index])
                        == np.sign(realized_gradient[index])
                    ),
                    "critic_step_action": float(critic_step[index]),
                    "critic_step_raw_gain": float(
                        step_returns["raw"][index] - actions["raw"][index]
                    ),
                    "critic_step_discounted_gain": float(
                        step_returns["discounted"][index]
                        - actions["discounted"][index]
                    ),
                    "tanh_derivative": float(tanh_derivative[index]),
                    "actor_saturated": int(abs(actions["action"][index]) >= 1.98),
                }
            )
        provenance.append(
            {
                "condition": str(condition["name"]),
                "replicate": replicate,
                "actor_run": str(actor_path.relative_to(ROOT)),
                "critic_run": str(critic_path.relative_to(ROOT)),
                "actor_checkpoint_sha256": sha256(
                    actor_path / "checkpoints" / "final.pt"
                ),
                "critic_checkpoint_sha256": sha256(
                    critic_path / "checkpoints" / "final.pt"
                ),
            }
        )
    return rows, provenance


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    output: Path,
    rows: list[dict[str, Any]],
    summaries: dict[str, Any],
) -> None:
    conditions = list(summaries["conditions"])
    colors = ["#0f766e", "#b45309", "#2563eb", "#7c3aed"]
    display = {
        "pure RL actor and own critic": "pure RL",
        "mixed actor and shared FastSACN critic": "mixed + shared FastSACN",
        "P0 one-step SimbaV2 actor and own critic": "P0 one-step",
        "P1 FastSACN8 SimbaV2 actor and own critic": "P1 FastSACN8",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for condition, color in zip(conditions, colors, strict=False):
        subset = [row for row in rows if row["condition"] == condition]
        q = np.asarray([float(row["critic_gradient"]) for row in subset])
        realized = np.asarray([float(row["realized_gradient"]) for row in subset])
        gain = np.asarray(
            [float(row["critic_step_discounted_gain"]) for row in subset]
        )
        axes[0, 0].scatter(
            q,
            realized,
            s=5,
            alpha=0.12,
            color=color,
            label=display.get(condition, condition),
            rasterized=True,
        )
        seed_rows = summaries["conditions"][condition]["per_seed"]
        axes[0, 2].plot(
            [item["seed"] for item in seed_rows],
            [100.0 * item["gradient_sign_agreement"] for item in seed_rows],
            marker="o",
            color=color,
            label=display.get(condition, condition),
        )
        pooled = summaries["conditions"][condition]["pooled"]
        if int(pooled["failure_count"]) >= 20:
            axes[1, 0].plot(
                [0, 1],
                [
                    100.0 * pooled["failure_gradient_sign_agreement"],
                    100.0 * pooled["success_gradient_sign_agreement"],
                ],
                marker="o",
                color=color,
                label=display.get(condition, condition),
            )
    axes[0, 0].axhline(0, color="#64748b", linewidth=0.7)
    axes[0, 0].axvline(0, color="#64748b", linewidth=0.7)
    axes[0, 0].set(
        title="A. Critic gradient versus realized local gradient",
        xlabel="critic dQ/da",
        ylabel="finite-difference dG/da",
    )
    x = np.arange(len(conditions))
    beneficial = [
        100.0
        * summaries["conditions"][condition]["pooled"][
            "critic_step_beneficial_rate"
        ]
        for condition in conditions
    ]
    harmful = [
        100.0
        * summaries["conditions"][condition]["pooled"]["critic_step_harmful_rate"]
        for condition in conditions
    ]
    axes[0, 1].bar(x, beneficial, color="#15803d", label="beneficial")
    axes[0, 1].bar(
        x, harmful, bottom=beneficial, color="#b91c1c", label="harmful"
    )
    axes[0, 1].set_xticks(
        x,
        [display.get(condition, condition) for condition in conditions],
        rotation=15,
        ha="right",
    )
    axes[0, 1].set(
        title="B. Return change after a critic-gradient step",
        ylabel="states (%)",
    )
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend(frameon=False, loc="lower right")
    axes[0, 2].set(
        title="C. Direction agreement by trained seed",
        xlabel="actor seed",
        ylabel="sign agreement (%)",
    )
    axes[0, 2].set_ylim(0, 100)
    axes[1, 0].set_xticks([0, 1], ["near-reference failure", "success"])
    axes[1, 0].set(
        title="D. Direction agreement by outcome",
        ylabel="sign agreement (%)",
    )
    axes[1, 0].set_ylim(0, 100)
    omitted = [
        condition
        for condition in conditions
        if int(
            summaries["conditions"][condition]["pooled"]["failure_count"]
        )
        < 20
    ]
    if omitted:
        axes[1, 0].text(
            0.03,
            0.06,
            "Mixed failure comparison omitted: fewer than 20 failures",
            transform=axes[1, 0].transAxes,
            fontsize=8,
            color="#475569",
        )
    condition_labels = []
    twin_values = []
    saturation_values = []
    for condition in conditions:
        pooled = summaries["conditions"][condition]["pooled"]
        condition_labels.append(condition)
        twin_values.append(100.0 * pooled["twin_gradient_sign_agreement"])
        saturation_values.append(100.0 * pooled["actor_saturation_rate"])
    axes[1, 1].bar(x, twin_values, color=colors[: len(conditions)])
    axes[1, 1].set_xticks(
        x,
        [display.get(condition, condition) for condition in conditions],
        rotation=15,
        ha="right",
    )
    axes[1, 1].set(
        title="E. Twin critics agree on gradient direction",
        ylabel="agreement (%)",
    )
    axes[1, 1].set_ylim(0, 100)
    axes[1, 2].bar(x, saturation_values, color=colors[: len(conditions)])
    axes[1, 2].set_xticks(
        x,
        [display.get(condition, condition) for condition in conditions],
        rotation=15,
        ha="right",
    )
    axes[1, 2].set(
        title="F. Actor actions at the torque boundary",
        ylabel="saturated actions (%)",
    )
    axes[1, 2].set_ylim(0, 100)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(conditions),
    )
    fig.suptitle(
        "Frozen critic gradients versus realized return directions\n"
        "Five actors per family on a locked off-grid sample",
        y=0.992,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.845))
    fig.savefig(output / "critic_action_gradient_alignment.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.sample_count < 64:
        raise SystemExit("--sample-count must be at least 64.")
    if not 0.0 < args.action_delta <= 0.25:
        raise SystemExit("--action-delta must be in (0, 0.25].")
    protocol = json.loads(resolve(args.protocol).read_text(encoding="utf-8"))
    dataset = build_validation_dataset(protocol["validation_dataset"])
    if args.sample_count > dataset["points"]:
        raise SystemExit("sample count exceeds the locked dataset size.")
    rng = np.random.default_rng(args.sample_seed)
    selected = np.sort(
        rng.choice(dataset["points"], size=args.sample_count, replace=False)
    )
    theta = np.asarray(dataset["theta"])[selected]
    velocity = np.asarray(dataset["velocity"])[selected]
    reference_spec = protocol["reference_protocol"]
    dp_solution = resolve(reference_spec["dp_solution"]["path"])
    if (
        dp_solution.stat().st_size
        != int(reference_spec["dp_solution"]["size_bytes"])
        or sha256(dp_solution) != str(reference_spec["dp_solution"]["sha256"])
    ):
        raise ValueError("Pinned off-grid DP solution fingerprint drift.")
    reliability = ReliabilityConfig(**protocol["evaluation_reliability"])
    detector = UprightDetector(
        "Pendulum-v1",
        cos_threshold=reliability.near_upright_cos_threshold,
        abs_velocity_threshold=reliability.near_upright_abs_velocity_threshold,
    )
    reference = validation_reference_returns(
        theta, velocity, detector, reliability, dp_solution
    )
    epsilon = float(reference_spec["epsilon_return"])
    horizon = 200
    conditions = load_spec(resolve(args.spec))
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for condition in conditions:
        rows, identities = evaluate_condition(
            condition,
            theta,
            velocity,
            reference,
            epsilon,
            horizon,
            args.action_delta,
            args.device,
        )
        all_rows.extend(rows)
        provenance.extend(identities)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_scope": (
            "Read-only frozen-policy diagnostic. The critic action derivative is "
            "compared with a central finite difference of the realized discounted "
            "return obtained by changing only the first action and then returning "
            "control to the same frozen actor."
        ),
        "protocol": {
            "locked_dataset_sha256": dataset["sha256"],
            "locked_dataset_points": dataset["points"],
            "sample_count": args.sample_count,
            "sample_seed": args.sample_seed,
            "selected_indices_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "action_delta": args.action_delta,
            "horizon": horizon,
            "gamma": 0.99,
            "reference_access_during_inference": False,
            "authority_grid_used": False,
        },
        "conditions": {},
        "provenance": provenance,
        "interpretation_limits": [
            "A finite first-action perturbation tests the local policy-evaluation landscape, not the full policy-gradient objective.",
            "Agreement is descriptive after checkpoint selection and does not identify which training update created an error.",
            "Critic gradient magnitudes are not compared across reward scalings; direction and realized intervention outcomes are the primary quantities.",
        ],
    }
    for condition in [str(item["name"]) for item in conditions]:
        subset = [row for row in all_rows if row["condition"] == condition]
        seeds = sorted({int(row["seed"]) for row in subset})
        summary["conditions"][condition] = {
            "pooled": summarize_group(subset),
            "per_seed": [
                {
                    "seed": seed,
                    **summarize_group(
                        [row for row in subset if int(row["seed"]) == seed]
                    ),
                }
                for seed in seeds
            ],
        }
    write_rows(output / "critic_action_gradient_rows.csv", all_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    make_plots(output, all_rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
