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
        fast_updates=False,
        simba_weight_projection=False,
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
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=True,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.telemetry.save_replay is True


def test_pendulum_hard_reset_cli_overrides_env_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        pendulum_hard_reset_prob=0.75,
        pendulum_hard_reset_abs_theta_low=2.1,
        pendulum_hard_reset_abs_theta_high=2.3,
        pendulum_hard_reset_velocity_limit=0.5,
        total_steps=None,
        learning_starts=None,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.env.pendulum_hard_reset_prob == 0.75
    assert config.env.pendulum_hard_reset_abs_theta_low == 2.1
    assert config.env.pendulum_hard_reset_abs_theta_high == 2.3
    assert config.env.pendulum_hard_reset_velocity_limit == 0.5


def test_pendulum_hard_replay_cli_overrides_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        pendulum_hard_replay_fraction=0.25,
        pendulum_hard_replay_abs_theta_low=2.0,
        pendulum_hard_replay_abs_theta_high=2.4,
        pendulum_hard_replay_velocity_limit=0.75,
        total_steps=None,
        learning_starts=None,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.pendulum_hard_replay_fraction == 0.25
    assert config.sac.pendulum_hard_replay_abs_theta_low == 2.0
    assert config.sac.pendulum_hard_replay_abs_theta_high == 2.4
    assert config.sac.pendulum_hard_replay_velocity_limit == 0.75


def test_simba_weight_projection_cli_override_sets_sac_flag():
    config = ExperimentConfig.from_dict({})
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
        fast_updates=False,
        simba_weight_projection=True,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.simba_weight_projection is True


def test_simba_backbone_cli_overrides_set_ablation_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        policy_lr=1e-4,
        q_lr=1e-4,
        policy_lr_final=5e-5,
        q_lr_final=5e-5,
        alpha_initial_value=0.01,
        target_entropy_scale=-0.5,
        redo_interval_updates=None,
        redo_dormant_threshold=None,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=True,
        simba_backbone=True,
        simba_no_observation_norm=True,
        simba_no_feature_norm=True,
        simba_no_input_shift=True,
        simba_actor_blocks=2,
        simba_actor_hidden_dim=32,
        simba_critic_blocks=3,
        simba_critic_hidden_dim=64,
        simba_distributional_critic=True,
        simba_reward_scaling=True,
        simba_critic_num_bins=51,
        simba_critic_min_v=-3.0,
        simba_critic_max_v=3.0,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.simba_backbone is True
    assert config.sac.simba_observation_norm is False
    assert config.sac.simba_feature_norm is False
    assert config.sac.simba_input_shift is False
    assert config.sac.simba_actor_blocks == 2
    assert config.sac.simba_actor_hidden_dim == 32
    assert config.sac.simba_critic_blocks == 3
    assert config.sac.simba_critic_hidden_dim == 64
    assert config.sac.simba_distributional_critic is True
    assert config.sac.simba_reward_scaling is True
    assert config.sac.simba_critic_num_bins == 51
    assert config.sac.simba_critic_min_v == -3.0
    assert config.sac.simba_critic_max_v == 3.0
    assert config.sac.update_diagnostics is False
    assert config.sac.policy_lr == 1e-4
    assert config.sac.q_lr == 1e-4
    assert config.sac.policy_lr_final == 5e-5
    assert config.sac.q_lr_final == 5e-5
    assert config.sac.alpha_initial_value == 0.01
    assert config.sac.target_entropy_scale == -0.5


def test_redo_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        redo_interval_updates=2000,
        redo_dormant_threshold=0.025,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.redo_interval_updates == 2000
    assert config.sac.redo_dormant_threshold == 0.025


def test_swd_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        swd_linear_decay_steps=80000,
        swd_min_weight=0.1,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.swd_linear_decay_steps == 80000
    assert config.sac.swd_min_weight == 0.1


def test_sacn_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        sacn_n_step=16,
        sacn_importance_quantile=0.5,
        sacn_no_tau_entropy=True,
        sacn_max_entropy_samples=8,
        sacn_recent_max_age_steps=4000,
        sacn_min_horizon_ess_fraction=0.2,
        sacn_importance_mode="none",
        sacn_non_soft_targets=True,
        sacn_stop_after_steps=5000,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.sacn_n_step == 16
    assert config.sac.sacn_importance_quantile == 0.5
    assert config.sac.sacn_tau_entropy is False
    assert config.sac.sacn_max_entropy_samples == 8
    assert config.sac.sacn_recent_max_age_steps == 4000
    assert config.sac.sacn_min_horizon_ess_fraction == 0.2
    assert config.sac.sacn_importance_mode == "none"
    assert config.sac.sacn_non_soft_targets is True
    assert config.sac.sacn_stop_after_steps == 5000


def test_reference_auxiliary_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        reference_auxiliary_mode="q_filtered_bc",
        reference_auxiliary_policy="controller",
        reference_auxiliary_weight=0.25,
        reference_auxiliary_margin=0.05,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.reference_auxiliary_mode == "q_filtered_bc"
    assert config.sac.reference_auxiliary_policy == "controller"
    assert config.sac.reference_auxiliary_weight == 0.25
    assert config.sac.reference_auxiliary_margin == 0.05


def test_reference_critic_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        reference_critic_mode="margin",
        reference_critic_policy="best",
        reference_critic_weight=0.25,
        reference_critic_margin=0.02,
        eval_every_steps=None,
        eval_episodes=None,
        eval_seed_base=None,
        log_interval=None,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.reference_critic_mode == "margin"
    assert config.sac.reference_critic_policy == "best"
    assert config.sac.reference_critic_weight == 0.25
    assert config.sac.reference_critic_margin == 0.02
