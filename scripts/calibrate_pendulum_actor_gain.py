from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from last_nine_rl.checkpoints import load_agent_from_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake a validated scalar gain into one actor checkpoint.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.gain) <= 0.0:
        raise ValueError("gain must be positive")
    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    agent, config, payload = load_agent_from_run(
        Path(args.source_run), device=str(args.device), load_optimizers=False
    )
    config.seed = int(args.seed)
    config.to_json(run_dir / "config.json")
    original_scale = agent.actor.action_scale.detach().cpu().clone()
    with torch.no_grad():
        agent.actor.action_scale.mul_(float(args.gain))
    checkpoint = run_dir / "checkpoints" / "final.pt"
    agent.save_checkpoint(
        checkpoint,
        extra={
            "global_step": 0,
            "single_actor_inference": True,
            "inference_router": False,
            "calibrated_action_gain": float(args.gain),
            "source_run": str(args.source_run),
            "source_extra": payload.get("extra", {}),
        },
    )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "single_actor_inference": True,
        "inference_router": False,
        "source_run": str(args.source_run),
        "gain": float(args.gain),
        "original_action_scale": original_scale.tolist(),
        "calibrated_action_scale": agent.actor.action_scale.detach().cpu().tolist(),
        "environment_action_clip": [-2.0, 2.0],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
