from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from last_nine_rl.config import ExperimentConfig, resolve_device
from last_nine_rl.envs import make_env
from last_nine_rl.sac import SACAgent


def expand_run_dirs(paths: list[Path], require_checkpoint: bool = True) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if is_run_dir(path, require_checkpoint=require_checkpoint):
            out.append(path)
            continue
        for config_path in path.glob("**/config.json"):
            run_dir = config_path.parent
            if is_run_dir(run_dir, require_checkpoint=require_checkpoint):
                out.append(run_dir)
    return sorted({path.resolve() for path in out})


def is_run_dir(path: Path, require_checkpoint: bool = True) -> bool:
    if not (path / "config.json").is_file():
        return False
    return not require_checkpoint or checkpoint_path(path).is_file()


def checkpoint_path(run_dir: Path, checkpoint: str = "final.pt") -> Path:
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    return run_dir / "checkpoints" / checkpoint


def load_agent_from_run(
    run_dir: str | Path,
    device: str | None = None,
    checkpoint: str = "final.pt",
    load_optimizers: bool = False,
) -> tuple[SACAgent, ExperimentConfig, dict[str, Any]]:
    run_path = Path(run_dir)
    return load_agent_from_config_checkpoint(
        run_path / "config.json",
        checkpoint_path(run_path, checkpoint),
        device=device,
        load_optimizers=load_optimizers,
    )


def load_agent_from_config_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    device: str | None = None,
    load_optimizers: bool = False,
) -> tuple[SACAgent, ExperimentConfig, dict[str, Any]]:
    """Load a checkpoint with an explicitly pinned compatible config file.

    Actor-only continuation workflows intentionally keep their model in a new
    run directory without copying the source run's ``config.json``.  Accepting
    both paths directly avoids model or config copies while retaining the same
    construction and checkpoint-loading semantics as ``load_agent_from_run``.
    """
    config = ExperimentConfig.from_json(Path(config_path))
    if device is not None:
        config.sac.device = device
    resolved_device = resolve_device(config.sac.device)
    config.sac.device = resolved_device

    env = make_env(config.env.env_id, seed=config.seed, max_episode_steps=config.env.max_episode_steps)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, config.sac, device=resolved_device)
    finally:
        env.close()

    payload = agent.load_checkpoint(
        Path(checkpoint),
        load_optimizers=load_optimizers,
    )
    return agent, config, payload
