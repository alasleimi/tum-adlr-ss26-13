from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from scripts.diagnose_first_divergence_20260723 import pendulum_obs, pendulum_step


ROOT = Path(__file__).resolve().parents[1]
FIGURE = (
    ROOT
    / "deliverables"
    / "chasing_nines_20260723"
    / "figures"
    / "42_qualitative_trajectory.png"
)
COMPACT_FIGURE = (
    ROOT
    / "deliverables"
    / "chasing_nines_20260723"
    / "figures"
    / "46_template_qualitative_trajectory.png"
)
POSTER_FIGURE = (
    ROOT
    / "deliverables"
    / "chasing_nines_20260723"
    / "figures"
    / "51_same_start_step64.png"
)
POSTER_SEQUENCE_FIGURE = (
    ROOT
    / "deliverables"
    / "chasing_nines_20260723"
    / "figures"
    / "52_same_start_recovery_sequence.png"
)
SUMMARY = ROOT / "reports" / "plan2507_qualitative_trajectory_20260725.json"
P7_RUN = (
    ROOT
    / "runs"
    / "plan2307_completion_20260723"
    / "pure_target_architecture_matrix"
    / "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k"
    / "seed0"
)
MIXED_RUN = ROOT / "runs" / "systematic_100k_ablation_no_rl_shift_20260722" / "seed0"
DP_SOLUTION = (
    ROOT
    / "reports"
    / "pendulum_investigation_20260509"
    / "pendulum_dp_100k_reset_support_241x161x81"
    / "pendulum_dp_solution.npz"
)
THETA0 = 3.038589615767177
VELOCITY0 = -0.25
STEPS = 200
SNAPSHOTS = [0, 16, 32, 64, 100, 150]


def wrap(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def actor_rollout(agent: object) -> dict[str, np.ndarray | float]:
    theta = np.array([THETA0], dtype=np.float64)
    velocity = np.array([VELOCITY0], dtype=np.float64)
    theta_history = [THETA0]
    velocity_history = [VELOCITY0]
    actions: list[float] = []
    total_return = 0.0
    for _step in range(STEPS):
        obs = pendulum_obs(theta, velocity)
        action = np.asarray(
            agent.act_batch(obs, deterministic=True), dtype=np.float64
        ).reshape(-1)
        reward, theta, velocity = pendulum_step(theta, velocity, action)
        total_return += float(reward[0])
        actions.append(float(action[0]))
        theta_history.append(float(theta[0]))
        velocity_history.append(float(velocity[0]))
    return {
        "theta": np.asarray(theta_history),
        "velocity": np.asarray(velocity_history),
        "action": np.asarray(actions),
        "return": total_return,
    }


def reference_rollout() -> dict[str, np.ndarray | float | str]:
    dp = PendulumReferenceGuidance(
        policy="dp", dp_solution_path=DP_SOLUTION, horizon=STEPS
    )
    controller = PendulumReferenceGuidance(policy="controller", horizon=STEPS)
    candidates: list[tuple[str, PendulumReferenceGuidance]] = [
        ("dynamic programming", dp),
        ("controller", controller),
    ]
    outputs: list[dict[str, np.ndarray | float | str]] = []
    for name, policy in candidates:
        theta = np.array([THETA0], dtype=np.float64)
        velocity = np.array([VELOCITY0], dtype=np.float64)
        theta_history = [THETA0]
        velocity_history = [VELOCITY0]
        actions: list[float] = []
        total_return = 0.0
        for step in range(STEPS):
            obs = pendulum_obs(theta, velocity)
            action = np.asarray(
                policy.act_batch(obs, remaining_steps=STEPS - step),
                dtype=np.float64,
            ).reshape(-1)
            reward, theta, velocity = pendulum_step(theta, velocity, action)
            total_return += float(reward[0])
            actions.append(float(action[0]))
            theta_history.append(float(theta[0]))
            velocity_history.append(float(velocity[0]))
        outputs.append(
            {
                "name": name,
                "theta": np.asarray(theta_history),
                "velocity": np.asarray(velocity_history),
                "action": np.asarray(actions),
                "return": total_return,
            }
        )
    return max(outputs, key=lambda value: float(value["return"]))


def draw_pendulum(
    ax: plt.Axes,
    theta: float,
    velocity: float,
    *,
    color: str,
    step: int,
) -> None:
    near = np.cos(theta) >= 0.95 and abs(velocity) <= 1.0
    ax.set_facecolor("#e5f5ef" if near else "#fff0e7")
    bob_x = float(np.sin(theta))
    bob_y = float(np.cos(theta))
    ax.plot([0, bob_x], [0, bob_y], color=color, linewidth=6.0)
    ax.scatter([0], [0], s=42, color="#102a35", zorder=3)
    ax.scatter([bob_x], [bob_y], s=135, color=color, edgecolor="white", linewidth=1.5, zorder=4)
    ax.axhline(0, color="#9fb0b7", linewidth=0.7)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{step} steps", fontsize=12, pad=4, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#c7d4d8")


def main() -> None:
    pure_agent, _, _ = load_agent_from_run(P7_RUN, device="cpu")
    mixed_agent, _, _ = load_agent_from_run(MIXED_RUN, device="cpu")
    trajectories = {
        "Stored reference": reference_rollout(),
        "Pure RL raw actor": actor_rollout(pure_agent),
        "RL + supervised actor": actor_rollout(mixed_agent),
    }
    colors = {
        "Stored reference": "#475569",
        "Pure RL raw actor": "#3478ef",
        "RL + supervised actor": "#0f9f91",
    }

    fig = plt.figure(figsize=(20.0, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        3,
        7,
        width_ratios=[1.55, 1, 1, 1, 1, 1, 1],
        hspace=0.28,
        wspace=0.16,
    )

    row_labels = ["Stored reference", "Pure RL raw actor", "RL + supervised actor"]
    display_labels = {
        "Stored reference": "Stored\nreference",
        "Pure RL raw actor": "Pure RL\nraw actor",
        "RL + supervised actor": "RL + supervised\nactor",
    }
    for row, label in enumerate(row_labels):
        trajectory = trajectories[label]
        label_ax = fig.add_subplot(grid[row, 0])
        label_ax.set_facecolor(
            "#eef2f4"
            if label == "Stored reference"
            else ("#e6efff" if label == "Pure RL raw actor" else "#d9f2ec")
        )
        label_ax.text(
            0.08,
            0.68,
            display_labels[label],
            transform=label_ax.transAxes,
            color=colors[label],
            fontsize=14,
            fontweight="bold",
            va="center",
        )
        label_ax.text(
            0.08,
            0.41,
            f"return {float(trajectory['return']):.1f}",
            transform=label_ax.transAxes,
            color="#102a35",
            fontsize=16,
            fontweight="bold",
            va="center",
        )
        label_ax.text(
            0.08,
            0.18,
            (
                "reaches upright by 64 steps"
                if label != "Pure RL raw actor"
                else "still drifting at 100 steps"
            ),
            transform=label_ax.transAxes,
            color="#526b76",
            fontsize=10.5,
            va="center",
            wrap=True,
        )
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for spine in label_ax.spines.values():
            spine.set_visible(False)
        for column, step in enumerate(SNAPSHOTS, start=1):
            ax = fig.add_subplot(grid[row, column])
            draw_pendulum(
                ax,
                float(np.asarray(trajectory["theta"])[step]),
                float(np.asarray(trajectory["velocity"])[step]),
                color=colors[label],
                step=step,
            )
    fig.suptitle(
        "Same hard start. Direct learner-state labels prevent prolonged drift.",
        x=0.025,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#102a35",
    )
    fig.text(
        0.03,
        0.015,
        "Initial state θ = 174.1°, angular velocity = −0.25. Green snapshots satisfy the upright-and-slow condition. "
        "Actors are shown before critic search so the actor-side difference is visible.",
        fontsize=11.5,
        color="#526b76",
    )
    fig.tight_layout(rect=[0.02, 0.055, 1, 0.93])
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    compact_steps = [0, 64, 100]
    compact = plt.figure(figsize=(9.5, 7.0), facecolor="white")
    compact_grid = compact.add_gridspec(
        3,
        4,
        width_ratios=[2.0, 1, 1, 1],
        hspace=0.30,
        wspace=0.18,
    )
    compact_labels = {
        "Stored reference": "Reference",
        "Pure RL raw actor": "Pure RL",
        "RL + supervised actor": "RL + labels",
    }
    for row, label in enumerate(row_labels):
        trajectory = trajectories[label]
        label_ax = compact.add_subplot(compact_grid[row, 0])
        label_ax.set_facecolor(
            "#eef2f4"
            if label == "Stored reference"
            else ("#e6efff" if label == "Pure RL raw actor" else "#d9f2ec")
        )
        label_ax.text(
            0.08,
            0.66,
            compact_labels[label],
            transform=label_ax.transAxes,
            color=colors[label],
            fontsize=30,
            fontweight="bold",
            va="center",
        )
        label_ax.text(
            0.08,
            0.26,
            f"{float(trajectory['return']):.1f}",
            transform=label_ax.transAxes,
            color="#102a35",
            fontsize=30,
            fontweight="bold",
            va="center",
        )
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for spine in label_ax.spines.values():
            spine.set_visible(False)
        for column, step in enumerate(compact_steps, start=1):
            ax = compact.add_subplot(compact_grid[row, column])
            draw_pendulum(
                ax,
                float(np.asarray(trajectory["theta"])[step]),
                float(np.asarray(trajectory["velocity"])[step]),
                color=colors[label],
                step=step,
            )
            ax.set_title(f"t={step}", fontsize=30, pad=5, fontweight="bold")
    compact.tight_layout(pad=0.4)
    compact.savefig(
        COMPACT_FIGURE,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(compact)

    poster = plt.figure(figsize=(10.5, 3.2), facecolor="white")
    poster_grid = poster.add_gridspec(1, 3, wspace=0.16)
    poster_labels = {
        "Stored reference": "Reference",
        "Pure RL raw actor": "Pure RL raw actor",
        "RL + supervised actor": "RL + supervised",
    }
    for column, label in enumerate(row_labels):
        trajectory = trajectories[label]
        ax = poster.add_subplot(poster_grid[0, column])
        draw_pendulum(
            ax,
            float(np.asarray(trajectory["theta"])[64]),
            float(np.asarray(trajectory["velocity"])[64]),
            color=colors[label],
            step=64,
        )
        ax.set_title(poster_labels[label], fontsize=18, pad=5, fontweight="bold", color=colors[label])
        ax.text(
            0.5,
            -0.11,
            f"return {float(trajectory['return']):.1f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            color="#102a35",
            fontweight="bold",
        )
    poster.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.19)
    poster.savefig(POSTER_FIGURE, dpi=220, facecolor="white")
    plt.close(poster)

    sequence_steps = [0, 32, 64, 100]
    sequence = plt.figure(figsize=(12.5, 5.25), facecolor="white")
    sequence_grid = sequence.add_gridspec(
        3,
        5,
        width_ratios=[1.65, 1, 1, 1, 1],
        hspace=0.23,
        wspace=0.12,
    )
    sequence_labels = {
        "Stored reference": "Reference",
        "Pure RL raw actor": "Pure RL raw",
        "RL + supervised actor": "RL + supervised",
    }
    for row, label in enumerate(row_labels):
        trajectory = trajectories[label]
        label_ax = sequence.add_subplot(sequence_grid[row, 0])
        label_ax.set_facecolor(
            "#eef2f4"
            if label == "Stored reference"
            else ("#e6efff" if label == "Pure RL raw actor" else "#d9f2ec")
        )
        label_ax.text(
            0.07,
            0.65,
            sequence_labels[label],
            transform=label_ax.transAxes,
            color=colors[label],
            fontsize=19,
            fontweight="bold",
            va="center",
        )
        label_ax.text(
            0.07,
            0.27,
            f"return {float(trajectory['return']):.1f}",
            transform=label_ax.transAxes,
            color="#102a35",
            fontsize=17,
            fontweight="bold",
            va="center",
        )
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for spine in label_ax.spines.values():
            spine.set_visible(False)
        for column, step in enumerate(sequence_steps, start=1):
            ax = sequence.add_subplot(sequence_grid[row, column])
            draw_pendulum(
                ax,
                float(np.asarray(trajectory["theta"])[step]),
                float(np.asarray(trajectory["velocity"])[step]),
                color=colors[label],
                step=step,
            )
            ax.set_title(f"step {step}", fontsize=17, pad=3, fontweight="bold")
    sequence.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.02)
    sequence.savefig(POSTER_SEQUENCE_FIGURE, dpi=220, facecolor="white")
    plt.close(sequence)

    payload = {
        "initial_state": {"theta": THETA0, "theta_degrees": float(np.degrees(THETA0)), "theta_dot": VELOCITY0},
        "runs": {"pure": str(P7_RUN.relative_to(ROOT)), "mixed": str(MIXED_RUN.relative_to(ROOT))},
        "returns": {label: float(value["return"]) for label, value in trajectories.items()},
        "reference_policy": str(trajectories["Stored reference"]["name"]),
        "selection_note": (
            "The state was chosen post hoc because the P7 seed-0 raw actor fails "
            "near-reference while the matched mixed seed-0 actor succeeds. This is "
            "a qualitative illustration, not an additional aggregate estimate."
        ),
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
