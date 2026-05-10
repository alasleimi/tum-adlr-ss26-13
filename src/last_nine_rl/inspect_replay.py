from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from last_nine_rl.envs import UprightDetector
from last_nine_rl.replay import summarize_saved_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved replay_final.npz file.")
    parser.add_argument("--replay", required=True, help="Path to replay npz file.")
    parser.add_argument("--env-id", required=True, help="Environment ID used to collect the replay.")
    parser.add_argument("--current-step", type=int, default=None)
    parser.add_argument("--action-high", nargs="*", type=float, default=None)
    parser.add_argument("--near-upright-cos-threshold", type=float, default=0.95)
    parser.add_argument("--near-upright-abs-velocity-threshold", type=float, default=1.0)
    args = parser.parse_args()

    action_high = np.asarray(args.action_high, dtype=np.float32) if args.action_high else None
    detector = UprightDetector(
        args.env_id,
        cos_threshold=args.near_upright_cos_threshold,
        abs_velocity_threshold=args.near_upright_abs_velocity_threshold,
    )
    summary = summarize_saved_replay(
        Path(args.replay),
        detector=detector,
        current_step=args.current_step,
        action_high=action_high,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
