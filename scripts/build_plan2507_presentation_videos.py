from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.hybrid_qsearch import reflection_averaged_actor_actions
from last_nine_rl.reference_guidance import PendulumReferenceGuidance
from scripts.diagnose_first_divergence_20260723 import pendulum_obs, pendulum_step


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "chasing_nines_20260723" / "videos"
P7_ROOT = (
    ROOT
    / "runs"
    / "plan2307_completion_20260723"
    / "pure_target_architecture_matrix"
    / "p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k"
)
MIXED_RUN = ROOT / "runs" / "systematic_100k_ablation_no_rl_shift_20260722" / "seed0"
P7_REFLECTION_CSV = (
    ROOT
    / "reports"
    / "plan2507_p7_reflection_authority_20260725"
    / "grid"
    / "pendulum_grid_rollouts.csv"
)
P7_DEPLOY_CSV = (
    ROOT
    / "reports"
    / "plan2507_p7_authority_20260725"
    / "grid"
    / "pendulum_grid_rollouts.csv"
)
QUALITATIVE_SUMMARY = ROOT / "reports" / "plan2507_qualitative_trajectory_20260725.json"
DP_SOLUTION = (
    ROOT
    / "reports"
    / "pendulum_investigation_20260509"
    / "pendulum_dp_100k_reset_support_241x161x81"
    / "pendulum_dp_solution.npz"
)

WIDTH = 1920
HEIGHT = 1080
FPS = 20
STEPS = 200
FREEZE_FRAMES = 40

COLORS = {
    "ink": "#102a35",
    "muted": "#526b76",
    "line": "#c7d4d8",
    "soft": "#eef2f4",
    "reference": "#64748b",
    "pure": "#3478ef",
    "mixed": "#0f9f91",
    "good": "#0f9f91",
    "bad": "#e85d45",
    "amber": "#f5a623",
}


@dataclass
class Trajectory:
    label: str
    color: str
    theta: np.ndarray
    velocity: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    cumulative_return: np.ndarray
    near: np.ndarray
    final_return: float
    near_fraction: float
    longest_miss_streak: int
    task_success: bool
    note: str
    switched: np.ndarray | None = None
    result_label: str | None = None
    result_positive: bool | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def read_csv_row(path: Path, *, seed: int, theta: float, theta_dot: float) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(float(row["actual_seed"])) == seed
                and abs(float(row["theta"]) - theta) < 1e-11
                and abs(float(row["theta_dot"]) - theta_dot) < 1e-11
            ):
                return row
    raise ValueError(f"No matching row in {path}: seed={seed}, theta={theta}, theta_dot={theta_dot}")


def longest_false_streak(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def rollout(
    *,
    label: str,
    color: str,
    theta0: float,
    velocity0: float,
    action_fn: Callable[[np.ndarray, int], tuple[float, bool]],
    note: str,
) -> Trajectory:
    theta = np.zeros(STEPS + 1, dtype=np.float64)
    velocity = np.zeros(STEPS + 1, dtype=np.float64)
    action = np.zeros(STEPS, dtype=np.float64)
    reward = np.zeros(STEPS, dtype=np.float64)
    switched = np.zeros(STEPS, dtype=bool)
    theta[0] = theta0
    velocity[0] = velocity0
    for step in range(STEPS):
        obs = pendulum_obs(theta[step : step + 1], velocity[step : step + 1])
        action[step], switched[step] = action_fn(obs, step)
        r, next_theta, next_velocity = pendulum_step(
            theta[step : step + 1],
            velocity[step : step + 1],
            action[step : step + 1],
        )
        reward[step] = float(r[0])
        theta[step + 1] = float(next_theta[0])
        velocity[step + 1] = float(next_velocity[0])
    near = (np.cos(theta[1:]) >= 0.95) & (np.abs(velocity[1:]) <= 1.0)
    cumulative = np.cumsum(reward)
    longest = longest_false_streak(near)
    fraction = float(np.mean(near))
    return Trajectory(
        label=label,
        color=color,
        theta=theta,
        velocity=velocity,
        action=action,
        reward=reward,
        cumulative_return=cumulative,
        near=near,
        final_return=float(cumulative[-1]),
        near_fraction=fraction,
        longest_miss_streak=longest,
        task_success=bool(fraction >= 0.8 and longest <= 50),
        note=note,
        switched=switched,
    )


def actor_fn(agent: object) -> Callable[[np.ndarray, int], tuple[float, bool]]:
    def apply(obs: np.ndarray, _step: int) -> tuple[float, bool]:
        action = np.asarray(agent.act_batch(obs, deterministic=True), dtype=np.float64).reshape(-1)
        return float(action[0]), False

    return apply


def reflection_fn(agent: object) -> Callable[[np.ndarray, int], tuple[float, bool]]:
    def apply(obs: np.ndarray, _step: int) -> tuple[float, bool]:
        action = np.asarray(
            reflection_averaged_actor_actions(agent, obs), dtype=np.float64
        ).reshape(-1)
        return float(action[0]), False

    return apply


def reference_fn(policy: PendulumReferenceGuidance) -> Callable[[np.ndarray, int], tuple[float, bool]]:
    def apply(obs: np.ndarray, step: int) -> tuple[float, bool]:
        action = np.asarray(
            policy.act_batch(obs, remaining_steps=STEPS - step), dtype=np.float64
        ).reshape(-1)
        return float(action[0]), False

    return apply


def deployed_p7_fn(agent: object) -> Callable[[np.ndarray, int], tuple[float, bool]]:
    def apply(obs: np.ndarray, _step: int) -> tuple[float, bool]:
        fallback = np.asarray(
            reflection_averaged_actor_actions(agent, obs), dtype=np.float64
        ).reshape(-1)
        action = np.asarray(
            agent.act_batch_critic_search(
                obs,
                num_actions=41,
                margin=0.005,
                filter_mode="symmetric_actor_unanimous_advantage",
            ),
            dtype=np.float64,
        ).reshape(-1)
        return float(action[0]), bool(abs(action[0] - fallback[0]) > 1e-7)

    return apply


def rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=text_font, fill=fill)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    trajectory: Trajectory,
    step: int,
    final: bool,
) -> None:
    x0, y0, x1, y1 = box
    rounded_rectangle(draw, box, radius=28, fill="#ffffff", outline=COLORS["line"], width=3)
    draw.rounded_rectangle((x0, y0, x1, y0 + 88), radius=28, fill=trajectory.color)
    draw.rectangle((x0, y0 + 50, x1, y0 + 88), fill=trajectory.color)
    draw.text((x0 + 28, y0 + 20), trajectory.label, font=font(36, bold=True), fill="#ffffff")
    draw.text((x0 + 28, y0 + 98), trajectory.note, font=font(22), fill=COLORS["muted"])
    if trajectory.switched is not None and bool(trajectory.switched[min(step, STEPS - 1)]):
        rounded_rectangle(
            draw,
            (x1 - 250, y0 + 96, x1 - 28, y0 + 143),
            radius=13,
            fill="#fff2d6",
            outline=COLORS["amber"],
            width=2,
        )
        draw_centered(
            draw,
            (x1 - 139, y0 + 117),
            "Q SEARCH SWITCH",
            font(19, bold=True),
            COLORS["ink"],
        )

    state_index = min(step + 1, STEPS)
    theta = float(trajectory.theta[state_index])
    velocity = float(trajectory.velocity[state_index])
    action = float(trajectory.action[min(step, STEPS - 1)])
    cumulative = float(trajectory.cumulative_return[min(step, STEPS - 1)])
    near = bool(trajectory.near[min(step, STEPS - 1)])

    cx = (x0 + x1) // 2
    cy = y0 + 330
    radius = min((x1 - x0) * 0.24, 150)
    draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=COLORS["ink"])
    bob_x = cx + radius * math.sin(theta)
    bob_y = cy - radius * math.cos(theta)
    draw.line((cx, cy, bob_x, bob_y), fill=trajectory.color, width=18)
    draw.ellipse((bob_x - 31, bob_y - 31, bob_x + 31, bob_y + 31), fill=trajectory.color, outline="#ffffff", width=5)
    draw.arc((cx - radius, cy - radius, cx + radius, cy + radius), 200, 340, fill=COLORS["line"], width=3)
    draw.line((cx - 225, cy + 175, cx + 225, cy + 175), fill=COLORS["line"], width=3)
    status = "UPRIGHT + SLOW" if near else "RECOVERING"
    status_color = COLORS["good"] if near else COLORS["bad"]
    draw_centered(draw, (cx, cy + 220), status, font(26, bold=True), status_color)

    metric_y = y0 + 580
    metrics = [
        ("step", f"{min(step + 1, STEPS):03d} / 200"),
        ("return", f"{cumulative:7.1f}"),
        ("angle", f"{math.degrees(((theta + math.pi) % (2 * math.pi)) - math.pi):+6.1f}°"),
        ("speed", f"{velocity:+5.2f}"),
    ]
    metric_width = (x1 - x0 - 56) // len(metrics)
    for index, (name, value) in enumerate(metrics):
        mx = x0 + 28 + metric_width * index
        draw.text((mx, metric_y), name.upper(), font=font(17, bold=True), fill=COLORS["muted"])
        draw.text((mx, metric_y + 26), value, font=font(26, bold=True), fill=COLORS["ink"])

    action_y = y0 + 665
    draw.text((x0 + 28, action_y), "TORQUE", font=font(17, bold=True), fill=COLORS["muted"])
    bar_x0, bar_x1 = x0 + 150, x1 - 28
    bar_mid = (bar_x0 + bar_x1) / 2
    draw.rounded_rectangle((bar_x0, action_y + 2, bar_x1, action_y + 25), radius=11, fill="#e6ecef")
    action_x = bar_mid + np.clip(action / 2.0, -1.0, 1.0) * (bar_x1 - bar_x0) / 2
    draw.line((bar_mid, action_y - 2, bar_mid, action_y + 30), fill=COLORS["ink"], width=2)
    draw.ellipse((action_x - 11, action_y + 2, action_x + 11, action_y + 24), fill=trajectory.color)

    if final:
        final_y = y1 - 82
        positive = (
            trajectory.task_success
            if trajectory.result_positive is None
            else trajectory.result_positive
        )
        result = trajectory.result_label or (
            "TASK SUCCESS" if trajectory.task_success else "TASK FAILURE"
        )
        result_color = COLORS["good"] if positive else COLORS["bad"]
        rounded_rectangle(
            draw,
            (x0 + 28, final_y, x1 - 28, y1 - 26),
            radius=15,
            fill="#e6f6f2" if positive else "#fff0ec",
        )
        draw.text((x0 + 48, final_y + 12), result, font=font(22, bold=True), fill=result_color)
        draw.text(
            (x1 - 315, final_y + 16),
            f"near {100 * trajectory.near_fraction:.1f}%  ·  streak {trajectory.longest_miss_streak}",
            font=font(20, bold=True),
            fill=COLORS["ink"],
        )


def render_video(
    *,
    path: Path,
    title: str,
    subtitle: str,
    trajectories: list[Trajectory],
    footer: str,
) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        quality=8,
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    count = len(trajectories)
    left = 54
    right = WIDTH - 54
    gap = 30
    panel_width = int((right - left - gap * (count - 1)) / count)
    boxes = [
        (left + index * (panel_width + gap), 175, left + index * (panel_width + gap) + panel_width, 970)
        for index in range(count)
    ]
    for frame_index in range(STEPS + FREEZE_FRAMES):
        step = min(frame_index, STEPS - 1)
        final = frame_index >= STEPS
        image = Image.new("RGB", (WIDTH, HEIGHT), "#f7fafb")
        draw = ImageDraw.Draw(image)
        draw.text((58, 30), title, font=font(50, bold=True), fill=COLORS["ink"])
        draw.text((60, 96), subtitle, font=font(27), fill=COLORS["muted"])
        for box, trajectory in zip(boxes, trajectories):
            draw_panel(draw, box=box, trajectory=trajectory, step=step, final=final)
        draw.text((58, 1012), footer, font=font(21), fill=COLORS["muted"])
        writer.send(np.asarray(image, dtype=np.uint8))
    writer.close()


def verify_row(trajectory: Trajectory, row: dict[str, str], *, label: str) -> dict[str, float | int | bool]:
    expected = {
        "return": float(row["return"]),
        "near_upright_fraction": float(row["near_upright_fraction"]),
        "not_near_upright_streak": int(float(row["not_near_upright_streak"])),
        "task_success": bool(int(float(row["task_success"]))),
    }
    observed = {
        "return": trajectory.final_return,
        "near_upright_fraction": trajectory.near_fraction,
        "not_near_upright_streak": trajectory.longest_miss_streak,
        "task_success": trajectory.task_success,
    }
    # Scalar replay can differ slightly from the batched authority pass because
    # the neural-network kernels accumulate floating-point products differently.
    if abs(observed["return"] - expected["return"]) > 5e-4:
        raise ValueError(f"{label} return mismatch: {observed} vs {expected}")
    if abs(observed["near_upright_fraction"] - expected["near_upright_fraction"]) > 1e-12:
        raise ValueError(f"{label} near fraction mismatch: {observed} vs {expected}")
    if observed["not_near_upright_streak"] != expected["not_near_upright_streak"]:
        raise ValueError(f"{label} streak mismatch: {observed} vs {expected}")
    if observed["task_success"] != expected["task_success"]:
        raise ValueError(f"{label} task-success mismatch: {observed} vs {expected}")
    return {
        "return_abs_error": abs(observed["return"] - expected["return"]),
        "near_fraction_abs_error": abs(
            observed["near_upright_fraction"] - expected["near_upright_fraction"]
        ),
        "streak_exact": True,
        "task_success_exact": True,
    }


def probe_video(path: Path) -> dict[str, float | int | str]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    reader.close()
    return {
        "duration_seconds": float(meta["duration"]),
        "fps": float(meta["fps"]),
        "width": int(meta["size"][0]),
        "height": int(meta["size"][1]),
        "codec": str(meta["codec"]),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qualitative = json.loads(QUALITATIVE_SUMMARY.read_text(encoding="utf-8"))
    theta0 = float(qualitative["initial_state"]["theta"])
    velocity0 = float(qualitative["initial_state"]["theta_dot"])

    p7_seed0, _, _ = load_agent_from_run(P7_ROOT / "seed0", device="cpu")
    mixed_seed0, _, _ = load_agent_from_run(MIXED_RUN, device="cpu")
    dp = PendulumReferenceGuidance(
        policy="dp", dp_solution_path=DP_SOLUTION, horizon=STEPS
    )

    reference = rollout(
        label="Diagnostic reference",
        color=COLORS["reference"],
        theta0=theta0,
        velocity0=velocity0,
        action_fn=reference_fn(dp),
        note="Dynamic-programming controller",
    )
    pure_raw = rollout(
        label="Pure RL raw actor",
        color=COLORS["pure"],
        theta0=theta0,
        velocity0=velocity0,
        action_fn=actor_fn(p7_seed0),
        note="SimbaV2 + FastSACN8 · 100k reward transitions",
    )
    mixed_raw = rollout(
        label="RL + supervised actor",
        color=COLORS["mixed"],
        theta0=theta0,
        velocity0=velocity0,
        action_fn=actor_fn(mixed_seed0),
        note="DAgger learner-state labels + automatic weak-start follow-up",
    )
    reference.result_label = "REFERENCE"
    reference.result_positive = True
    pure_raw.result_label = "MISSES REFERENCE"
    pure_raw.result_positive = False
    mixed_raw.result_label = "NEAR REFERENCE"
    mixed_raw.result_positive = True
    expected_qualitative = qualitative["returns"]
    for trajectory, key in (
        (reference, "Stored reference"),
        (pure_raw, "Pure RL raw actor"),
        (mixed_raw, "RL + supervised actor"),
    ):
        if abs(trajectory.final_return - float(expected_qualitative[key])) > 1e-4:
            raise ValueError(
                f"Qualitative replay mismatch for {key}: "
                f"{trajectory.final_return} vs {expected_qualitative[key]}"
            )
    video1 = OUT / "01_same_hard_start_learning_gap.mp4"
    render_video(
        path=video1,
        title="Same hard start: supervision prevents drift",
        subtitle=(
            "Same initial state · three deterministic controllers · 200 Pendulum-v1 steps · "
            "θ = 174.1°, angular velocity = −0.25"
        ),
        trajectories=[reference, pure_raw, mixed_raw],
        footer=(
            "By step 64, the learner-state-labelled actor matches the reference swing-up; "
            "the raw pure-RL actor has drifted to the opposite side."
        ),
    )

    repair_seed = 4
    repair_theta = -3.038589615767177
    repair_velocity = -0.9
    p7_seed4, _, _ = load_agent_from_run(P7_ROOT / f"seed{repair_seed}", device="cpu")
    repair_reflection = rollout(
        label="Reflection only",
        color=COLORS["pure"],
        theta0=repair_theta,
        velocity0=repair_velocity,
        action_fn=reflection_fn(p7_seed4),
        note="Same seed-4 actor · exact symmetry projection",
    )
    repair_deployed = rollout(
        label="Deployed pure RL",
        color=COLORS["good"],
        theta0=repair_theta,
        velocity0=repair_velocity,
        action_fn=deployed_p7_fn(p7_seed4),
        note="Reflection + 41-action Q search · margin 0.005",
    )
    reflection_row = read_csv_row(
        P7_REFLECTION_CSV,
        seed=repair_seed,
        theta=repair_theta,
        theta_dot=repair_velocity,
    )
    deployed_row = read_csv_row(
        P7_DEPLOY_CSV,
        seed=repair_seed,
        theta=repair_theta,
        theta_dot=repair_velocity,
    )
    repair_verification = {
        "reflection_only": verify_row(
            repair_reflection, reflection_row, label="reflection only"
        ),
        "deployed_pure_rl": verify_row(
            repair_deployed, deployed_row, label="deployed pure RL"
        ),
    }
    video2 = OUT / "02_pure_rl_qsearch_repairs_failure.mp4"
    render_video(
        path=video2,
        title="A critic-approved action changes the trajectory",
        subtitle=(
            "Same seed-4 actor · same start θ = −174.10°, angular velocity = −0.90 · "
            "reflection alone versus reflection plus twin-Q search"
        ),
        trajectories=[repair_reflection, repair_deployed],
        footer=(
            "At step 13, both learned critics prefer another torque by more than 0.005. "
            "That single accepted switch changes task failure into task success."
        ),
    )

    videos = {
        video1.name: probe_video(video1),
        video2.name: probe_video(video2),
    }
    manifest = {
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "protocol": {
            "environment": "Pendulum-v1 exact project dynamics",
            "horizon": STEPS,
            "frames_per_second": FPS,
            "environment_step_seconds": 0.05,
            "final_freeze_seconds": FREEZE_FRAMES / FPS,
            "reference_at_deployment": False,
            "training_performed": False,
        },
        "videos": videos,
        "video_1": {
            "initial_state": {"theta": theta0, "theta_dot": velocity0},
            "runs": {
                "pure_rl": str((P7_ROOT / "seed0").relative_to(ROOT)),
                "rl_plus_supervised": str(MIXED_RUN.relative_to(ROOT)),
            },
            "reference": {
                "role": "diagnostic comparison only",
                "policy": "dynamic programming",
                "solution": str(DP_SOLUTION.relative_to(ROOT)),
            },
            "returns": {
                "diagnostic_reference": reference.final_return,
                "pure_rl_raw_actor": pure_raw.final_return,
                "rl_plus_supervised_actor": mixed_raw.final_return,
            },
            "source_summary": str(QUALITATIVE_SUMMARY.relative_to(ROOT)),
            "selection": qualitative["selection_note"],
        },
        "video_2": {
            "initial_state": {
                "theta": repair_theta,
                "theta_degrees": math.degrees(repair_theta),
                "theta_dot": repair_velocity,
            },
            "run": str((P7_ROOT / "seed4").relative_to(ROOT)),
            "deployment": {
                "actor_fallback": "reflection-averaged actor",
                "candidate_actions": 41,
                "candidate_support": "full legal action range [-2, 2]",
                "acceptance": "all online critics show advantage > 0.005",
                "hardcoded_state_ranges": False,
                "reference_access": False,
                "switch_fraction_this_rollout": float(np.mean(repair_deployed.switched)),
                "accepted_switch_steps_one_indexed": [
                    int(index + 1)
                    for index in np.flatnonzero(repair_deployed.switched)
                ],
            },
            "source_rows": {
                "reflection_only": {
                    "csv": str(P7_REFLECTION_CSV.relative_to(ROOT)),
                    "row": reflection_row,
                },
                "deployed_pure_rl": {
                    "csv": str(P7_DEPLOY_CSV.relative_to(ROOT)),
                    "row": deployed_row,
                },
            },
            "replay_verification": repair_verification,
            "selection_note": (
                "Selected post hoc from the 250 grid seed-state cells where reflection-only "
                "failed task success and the deployed policy passed. It is an illustrative "
                "case and is not an additional aggregate estimate."
            ),
        },
        "checkpoint_hashes": {
            "p7_seed0": sha256_file(P7_ROOT / "seed0" / "checkpoints" / "final.pt"),
            "p7_seed4": sha256_file(P7_ROOT / "seed4" / "checkpoints" / "final.pt"),
            "mixed_seed0": sha256_file(MIXED_RUN / "checkpoints" / "final.pt"),
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
