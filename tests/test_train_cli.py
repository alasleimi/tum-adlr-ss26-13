from types import SimpleNamespace

from last_nine_rl.config import ExperimentConfig
from last_nine_rl.train import apply_overrides


def test_eval_episode_override_replaces_explicit_eval_seed_list():
    config = ExperimentConfig.from_dict({"eval": {"seeds": [7, 11]}})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        eval_every_steps=None,
        eval_episodes=3,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.eval.episodes == 3
    assert config.eval.seeds is None


def test_save_replay_cli_override_enables_replay_snapshot():
    config = ExperimentConfig.from_dict({"telemetry": {"save_replay": False}})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        save_replay=True,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.telemetry.save_replay is True
