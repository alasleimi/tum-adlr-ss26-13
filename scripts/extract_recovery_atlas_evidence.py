"""Extract compact poster trajectory evidence from the retained checkpoints.

The default output is disposable. The canonical copy under ``data/report`` is
already checked by the artifact manifest and normally does not need rebuilding.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.hybrid_qsearch import (
    FixedGlobalCriticQSearchPolicy,
    FixedLocalCriticQSearchPolicy,
)


STEPS = 200
DESIRED_STARTS = ((-174.098361, -0.35), (-44.262295, 1.0), (44.262295, -1.0))


def choose_starts(data_dir: Path) -> list[dict[str, float]]:
    keys = ["theta", "theta_degrees", "theta_dot"]
    pure = pd.read_csv(data_dir / "rollouts" / "pure_selected_q41.csv")
    mixed = pd.read_csv(data_dir / "rollouts" / "mixed_selected.csv")
    pure_grid = (
        pure.groupby(keys)
        .agg(
            pure_success=("near_best_known_return_eps", "sum"),
            pure_mean=("return", "mean"),
        )
        .reset_index()
    )
    mixed_grid = (
        mixed.groupby(keys)
        .agg(
            mixed_success=("near_best_known_return_eps", "sum"),
            mixed_mean=("return", "mean"),
        )
        .reset_index()
    )
    joined = pure_grid.merge(mixed_grid, on=keys, validate="one_to_one")
    candidates = joined.query("mixed_success == 5 and pure_success == 0").copy()
    if candidates.empty:
        raise ValueError("No starts satisfy mixed 5/5 and pure 0/5 in retained evidence")
    starts: list[dict[str, float]] = []
    for desired_angle, desired_velocity in DESIRED_STARTS:
        distance = (candidates["theta_degrees"] - desired_angle).abs() + 10 * (
            candidates["theta_dot"] - desired_velocity
        ).abs()
        row = candidates.loc[distance.idxmin()]
        starts.append(
            {
                "theta": float(row["theta"]),
                "theta_degrees": float(row["theta_degrees"]),
                "theta_dot": float(row["theta_dot"]),
                "mixed_success": int(row["mixed_success"]),
                "pure_success": int(row["pure_success"]),
            }
        )
    return starts


def rollout(policy: object, theta0: float, velocity0: float) -> tuple[np.ndarray, np.ndarray, float]:
    theta = np.asarray([theta0], dtype=np.float64)
    velocity = np.asarray([velocity0], dtype=np.float64)
    theta_history = [float(theta[0])]
    velocity_history = [float(velocity[0])]
    total_return = 0.0
    for _ in range(STEPS):
        observation = np.stack(
            [np.cos(theta), np.sin(theta), velocity], axis=1
        ).astype(np.float32)
        action = np.asarray(
            policy.act_batch(observation, deterministic=True), dtype=np.float64
        ).reshape(-1)
        torque = np.clip(action, -2.0, 2.0)
        wrapped = ((theta + np.pi) % (2.0 * np.pi)) - np.pi
        reward = -(wrapped**2 + 0.1 * velocity**2 + 0.001 * torque**2)
        velocity = np.clip(
            velocity + (15.0 * np.sin(theta) + 3.0 * torque) * 0.05,
            -8.0,
            8.0,
        )
        theta = theta + velocity * 0.05
        total_return += float(reward[0])
        theta_history.append(float(theta[0]))
        velocity_history.append(float(velocity[0]))
    return (
        np.asarray(theta_history, dtype=np.float64),
        np.asarray(velocity_history, dtype=np.float64),
        total_return,
    )


def extract(root: Path, output: Path) -> tuple[Path, Path]:
    data_dir = root / "data" / "report"
    models = root / "artifacts" / "report_reproduction" / "models"
    starts = choose_starts(data_dir)
    shape = (len(starts), 5, STEPS + 1)
    arrays = {
        "mixed_theta": np.empty(shape, dtype=np.float64),
        "mixed_velocity": np.empty(shape, dtype=np.float64),
        "pure_theta": np.empty(shape, dtype=np.float64),
        "pure_velocity": np.empty(shape, dtype=np.float64),
        "mixed_return": np.empty((len(starts), 5), dtype=np.float64),
        "pure_return": np.empty((len(starts), 5), dtype=np.float64),
    }
    mixed_critic, _, _ = load_agent_from_run(
        models / "mixed_shared_critic" / "seed1", device="cpu"
    )
    for seed in range(5):
        pure_agent, _, _ = load_agent_from_run(
            models / "pure_selected" / f"seed{seed}", device="cpu"
        )
        mixed_actor, _, _ = load_agent_from_run(
            models / "mixed_selected" / f"seed{seed}", device="cpu"
        )
        policies = {
            "pure": FixedGlobalCriticQSearchPolicy(
                pure_agent,
                pure_agent,
                num_actions=41,
                margin=0.005,
                max_action_delta=4.0,
                symmetric_actor_fallback=True,
            ),
            "mixed": FixedLocalCriticQSearchPolicy(
                mixed_actor,
                mixed_critic,
                num_actions=5,
                margin=0.0,
                search_radius=0.10,
                symmetric_actor_fallback=False,
            ),
        }
        for start_index, start in enumerate(starts):
            for method, policy in policies.items():
                theta, velocity, total_return = rollout(
                    policy, start["theta"], start["theta_dot"]
                )
                arrays[f"{method}_theta"][start_index, seed] = theta
                arrays[f"{method}_velocity"][start_index, seed] = velocity
                arrays[f"{method}_return"][start_index, seed] = total_return

    return_validation: dict[str, dict[str, float]] = {}
    for method, filename in (
        ("mixed", "mixed_selected.csv"),
        ("pure", "pure_selected_q41.csv"),
    ):
        frame = pd.read_csv(data_dir / "rollouts" / filename)
        deltas: list[float] = []
        for start_index, start in enumerate(starts):
            selected = frame.loc[
                np.isclose(frame["theta_degrees"], start["theta_degrees"])
                & np.isclose(frame["theta_dot"], start["theta_dot"])
            ].sort_values("actual_seed")
            if selected["actual_seed"].tolist() != list(range(5)):
                raise ValueError(
                    f"{method} retained rollout rows are incomplete for start {start_index + 1}"
                )
            deltas.extend(
                np.abs(
                    arrays[f"{method}_return"][start_index]
                    - selected["return"].to_numpy(dtype=np.float64)
                ).tolist()
            )
        maximum = float(max(deltas))
        if maximum > 1e-3:
            raise ValueError(
                f"{method} extracted returns differ from retained rollouts by {maximum}"
            )
        return_validation[method] = {
            "maximum_absolute_return_delta": maximum,
            "required_tolerance": 1e-3,
        }

    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "trajectories.npz"
    np.savez_compressed(
        archive_path,
        starts_theta=np.asarray([row["theta"] for row in starts]),
        starts_theta_degrees=np.asarray([row["theta_degrees"] for row in starts]),
        starts_theta_dot=np.asarray([row["theta_dot"] for row in starts]),
        **arrays,
    )
    provenance = {
        "schema_version": 1,
        "role": "derived_poster_trajectory_evidence",
        "description": (
            "Deterministic 200-step trajectories used by poster v115's recovery "
            "atlas. Extracted from retained checkpoints so routine figure builds "
            "do not repeatedly execute neural-network policies."
        ),
        "sources": {
            "start_selection": [
                "data/report/rollouts/mixed_selected.csv",
                "data/report/rollouts/pure_selected_q41.csv",
            ],
            "mixed_actors": "artifacts/report_reproduction/models/mixed_selected/seed0..seed4",
            "mixed_critic": "artifacts/report_reproduction/models/mixed_shared_critic/seed1",
            "pure_actor_critics": "artifacts/report_reproduction/models/pure_selected/seed0..seed4",
        },
        "policies": {
            "mixed": {
                "type": "FixedLocalCriticQSearchPolicy",
                "num_actions": 5,
                "margin": 0.0,
                "search_radius": 0.1,
            },
            "pure": {
                "type": "FixedGlobalCriticQSearchPolicy",
                "num_actions": 41,
                "margin": 0.005,
                "max_action_delta": 4.0,
                "symmetric_actor_fallback": True,
            },
        },
        "dynamics": {
            "steps": STEPS,
            "dt": 0.05,
            "torque_bounds": [-2.0, 2.0],
            "velocity_bounds": [-8.0, 8.0],
        },
        "starts": starts,
        "validation_against_retained_rollouts": return_validation,
        "extraction_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "serialization": "UTF-8 JSON with LF line endings",
        "qualification": (
            "This is derived display evidence, not an original training artifact. "
            "The manifest protects the checked-in archive byte-for-byte."
        ),
    }
    provenance_path = output / "provenance.json"
    with provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(provenance, indent=2) + "\n")
    return archive_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / ".build" / "recovery-atlas-evidence",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement when the requested output already contains evidence.",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    existing = [output / "trajectories.npz", output / "provenance.json"]
    if any(path.exists() for path in existing) and not args.overwrite:
        raise FileExistsError(
            f"Evidence already exists under {output}; pass --overwrite explicitly"
        )
    archive, provenance = extract(args.root.expanduser().resolve(), output)
    print(archive)
    print(provenance)


if __name__ == "__main__":
    main()
