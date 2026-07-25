from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from last_nine_rl.checkpoints import load_agent_from_run

try:
    from scripts.rank_pure_rl_improvement_screen import (
        FIXED_STATE_GEOMETRY_SPEC,
        evaluation_protocol,
        fixed_state_policy_geometry,
        frozen_validation_states,
    )
except ModuleNotFoundError:
    from rank_pure_rl_improvement_screen import (  # type: ignore[no-redef]
        FIXED_STATE_GEOMETRY_SPEC,
        evaluation_protocol,
        fixed_state_policy_geometry,
        frozen_validation_states,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "reports"
    / "plan2307_mechanistic_20260723"
    / "completed_target_actor_geometry"
)
ARM_LABELS = {
    "p0_simba_onestep_utd1_100k": "One-step SAC",
    "p1_simba_fastsacn8_lambda1_utd1_100k": "FastSACN8",
    "p2_simba_sacn8_lambda1_utd1_100k": "SACn8",
    "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k": (
        r"FastSACN8 $\lambda$=.5, critic UTD2"
    ),
}
METRICS = (
    (
        "deterministic_action_saturation_fraction_abs_ge_0p995",
        "Actor saturation",
        "fraction at ≥99.5% of torque bound",
    ),
    (
        "mean_tanh_derivative",
        "Actor action sensitivity",
        r"mean $1-\tanh^2(z)$",
    ),
    (
        "reflection_action_abs_error_mean",
        "Reflection error",
        r"mean $|a(s)+a(Ms)|$",
    ),
)


def run_paths() -> dict[str, list[Path]]:
    p0 = [
        ROOT
        / "runs"
        / (
            "week3_simbav2_scale_100k_20260526"
            if seed < 3
            else "week3_100k_component_ablation_20260527"
        )
        / "simba_full_official_opt"
        / f"seed{seed}"
        for seed in range(5)
    ]
    p1 = [
        (
            ROOT
            / "runs"
            / "systematic_pure_rl_improvements_20260722"
            / "p51_clean_sacn_fastlast8_lambda1_utd1_100k"
            / "seed0"
            if seed == 0
            else ROOT
            / "runs"
            / "plan2307_completion_20260723"
            / "pure_target_architecture_matrix"
            / "p1_simba_fastsacn8_lambda1_utd1_100k"
            / f"seed{seed}"
        )
        for seed in range(5)
    ]
    p2 = [
        ROOT
        / "runs"
        / "plan2307_completion_20260723"
        / "pure_target_architecture_matrix"
        / "p2_simba_sacn8_lambda1_utd1_100k"
        / f"seed{seed}"
        for seed in range(5)
    ]
    p7 = [
        ROOT
        / "runs"
        / "plan2307_completion_20260723"
        / "pure_target_architecture_matrix"
        / "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k"
        / f"seed{seed}"
        for seed in range(5)
    ]
    return {
        "p0_simba_onestep_utd1_100k": p0,
        "p1_simba_fastsacn8_lambda1_utd1_100k": p1,
        "p2_simba_sacn8_lambda1_utd1_100k": p2,
        "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k": p7,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = evaluation_protocol(critic_search_batch_size=512)
    theta, velocity = frozen_validation_states(protocol=protocol)
    rows: list[dict[str, object]] = []
    paths = run_paths()
    for arm, runs in paths.items():
        for seed, run in enumerate(runs):
            checkpoint = run / "checkpoints" / "final.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            agent, _config, _payload = load_agent_from_run(run, device="cpu")
            actor, _critic = fixed_state_policy_geometry(
                agent,
                theta,
                velocity,
                state_hash=str(protocol["state_sha256"]),
                spec=FIXED_STATE_GEOMETRY_SPEC,
            )
            row: dict[str, object] = {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "seed": seed,
                "run_dir": str(run.relative_to(ROOT)).replace("\\", "/"),
                "points": int(actor["points"]),
                "state_sha256": actor["state_sha256"],
            }
            for metric, _title, _ylabel in METRICS:
                row[metric] = float(actor[metric])
            row["mean_logit_abs_mean"] = float(actor["mean_logit_abs_mean"])
            row["mean_logit_abs_gt_4p15_fraction"] = float(
                actor["mean_logit_abs_gt_4p15_fraction"]
            )
            rows.append(row)

    with (OUT / "seed_actor_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arm_order = list(ARM_LABELS)
    contrasts = []
    for treatment, control, label in (
        (
            "p1_simba_fastsacn8_lambda1_utd1_100k",
            "p0_simba_onestep_utd1_100k",
            "FastSACN8 minus one-step",
        ),
        (
            "p2_simba_sacn8_lambda1_utd1_100k",
            "p1_simba_fastsacn8_lambda1_utd1_100k",
            "SACn8 minus FastSACN8",
        ),
        (
            "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k",
            "p0_simba_onestep_utd1_100k",
            "Matched FastSACN8 combination minus one-step",
        ),
    ):
        for metric, _title, _ylabel in METRICS:
            treatment_values = np.array(
                [
                    float(row[metric])
                    for row in rows
                    if row["arm"] == treatment
                ]
            )
            control_values = np.array(
                [float(row[metric]) for row in rows if row["arm"] == control]
            )
            delta = treatment_values - control_values
            contrasts.append(
                {
                    "contrast": label,
                    "metric": metric,
                    "paired_seed_deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "same_direction_seeds": int(
                        max((delta > 0).sum(), (delta < 0).sum())
                    ),
                }
            )

    colors = ["#334155", "#2563eb", "#0f766e", "#b91c1c"]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
    x = np.arange(len(arm_order))
    for axis, (metric, title, ylabel) in zip(axes, METRICS, strict=True):
        for index, (arm, color) in enumerate(zip(arm_order, colors, strict=True)):
            values = np.array(
                [float(row[metric]) for row in rows if row["arm"] == arm]
            )
            jitter = np.linspace(-0.11, 0.11, len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=44,
                color=color,
                alpha=0.9,
                zorder=3,
            )
            axis.plot(
                [index - 0.22, index + 0.22],
                [np.median(values), np.median(values)],
                color="#0f172a",
                linewidth=2.5,
                zorder=4,
            )
        axis.set_xticks(x, [ARM_LABELS[arm] for arm in arm_order], rotation=16)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in {
            "deterministic_action_saturation_fraction_abs_ge_0p995",
            "mean_tanh_derivative",
        }:
            axis.set_ylim(bottom=0)
    fig.suptitle(
        "Temporal-target recipes change the learned SimbaV2 actor",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "Five independently trained actors per method on the same 5,553 "
        "locked off-grid initial states. No rollouts or reference returns are used.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.savefig(OUT / "completed_target_actor_geometry.png", dpi=260)
    plt.close(fig)

    summary = {
        "schema_version": 1,
        "scope": (
            "Frozen-state actor geometry for the four completed five-seed "
            "SimbaV2 target arms. This diagnostic does not score policy returns."
        ),
        "independent_seeds_per_arm": 5,
        "points_per_seed": int(len(theta)),
        "state_sha256": str(protocol["state_sha256"]),
        "arms": {
            arm: {
                "label": ARM_LABELS[arm],
                "metrics": {
                    metric: {
                        "seed_values": [
                            float(row[metric])
                            for row in rows
                            if row["arm"] == arm
                        ],
                        "median": float(
                            np.median(
                                [
                                    float(row[metric])
                                    for row in rows
                                    if row["arm"] == arm
                                ]
                            )
                        ),
                    }
                    for metric, _title, _ylabel in METRICS
                },
            }
            for arm in arm_order
        },
        "paired_contrasts": contrasts,
        "limitation": (
            "The target is assigned at training time, but the frozen geometry is "
            "post-training association. It does not by itself establish that "
            "saturation or reflection error causes a reliability failure."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
