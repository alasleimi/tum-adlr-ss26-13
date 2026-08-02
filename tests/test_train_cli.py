from types import SimpleNamespace
import sys

import pytest

from last_nine_rl.config import ExperimentConfig
from last_nine_rl.train import (
    apply_overrides,
    parse_args,
    pendulum_hard_replay_fraction_at_step,
    pendulum_hard_reset_probability_at_step,
)


def test_priority_replay_cli_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "unused.json",
            "--replay-priority-mode",
            "max",
            "--replay-priority-alpha",
            "0.7",
            "--replay-priority-beta-initial",
            "0.5",
            "--replay-priority-beta-final",
            "0.9",
            "--replay-priority-beta-anneal-steps",
            "50000",
            "--replay-priority-uniform-fraction",
            "0.6",
            "--replay-priority-epsilon",
            "0.002",
            "--replay-priority-clip",
            "4.0",
        ],
    )
    args = parse_args()
    config = ExperimentConfig()
    apply_overrides(config, args)

    assert config.sac.replay_priority_mode == "max"
    assert config.sac.replay_priority_alpha == pytest.approx(0.7)
    assert config.sac.replay_priority_beta_initial == pytest.approx(0.5)
    assert config.sac.replay_priority_beta_final == pytest.approx(0.9)
    assert config.sac.replay_priority_beta_anneal_steps == 50_000
    assert config.sac.replay_priority_uniform_fraction == pytest.approx(0.6)
    assert config.sac.replay_priority_epsilon == pytest.approx(0.002)
    assert config.sac.replay_priority_clip == pytest.approx(4.0)


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


def test_checkpoint_interval_cli_override_updates_telemetry_config():
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
        checkpoint_interval_steps=10_000,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.telemetry.checkpoint_interval_steps == 10_000


def test_cql_interval_cli_override_updates_sac_config():
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
        cql_interval_updates=8,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.cql_interval_updates == 8


def test_critic_search_cli_overrides_update_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        critic_search_actor_weight=0.05,
        critic_search_num_actions=41,
        critic_search_margin=0.001,
        critic_search_start_update=40_000,
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

    assert config.sac.critic_search_actor_weight == pytest.approx(0.05)
    assert config.sac.critic_search_num_actions == 41
    assert config.sac.critic_search_margin == pytest.approx(0.001)
    assert config.sac.critic_search_start_update == 40_000


def test_self_imitation_cli_overrides_update_sac_config():
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
        self_imitation_weight=0.05,
        self_imitation_loss_type="log_prob",
        self_imitation_start_step=12_000,
        self_imitation_temperature=0.7,
        self_imitation_margin=0.02,
        self_imitation_max_weight=12.0,
        replay_inspection_interval=None,
        diagnostics_interval=None,
        device=None,
        fast_updates=False,
        simba_weight_projection=False,
        save_replay=False,
        overwrite=False,
    )

    apply_overrides(config, args)

    assert config.sac.self_imitation_weight == pytest.approx(0.05)
    assert config.sac.self_imitation_loss_type == "log_prob"
    assert config.sac.self_imitation_start_step == 12_000
    assert config.sac.self_imitation_temperature == pytest.approx(0.7)
    assert config.sac.self_imitation_margin == pytest.approx(0.02)
    assert config.sac.self_imitation_max_weight == pytest.approx(12.0)


def test_actor_q_aggregation_cli_override_updates_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        actor_q_aggregation="mean",
        actor_q_aggregation_late="max",
        actor_q_aggregation_switch_step=20_000,
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

    assert config.sac.actor_q_aggregation == "mean"
    assert config.sac.actor_q_aggregation_late == "max"
    assert config.sac.actor_q_aggregation_switch_step == 20_000


def test_actor_update_rate_cli_overrides_update_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        policy_frequency=4,
        actor_updates_per_trigger=1,
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

    assert config.sac.policy_frequency == 4
    assert config.sac.actor_updates_per_trigger == 1


def test_alpha_min_value_cli_override_updates_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        alpha_min_value=0.002,
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

    assert config.sac.alpha_min_value == 0.002


def test_alpha_lr_cli_override_updates_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        alpha_lr=2e-5,
        alpha_lr_final=1e-5,
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

    assert config.sac.alpha_lr == pytest.approx(2e-5)
    assert config.sac.alpha_lr_final == pytest.approx(1e-5)


def test_sac_actor_objective_mode_cli_override_updates_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        sac_actor_objective_mode="deterministic_mean",
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

    assert config.sac.sac_actor_objective_mode == "deterministic_mean"


def test_critic_search_actor_loss_type_cli_override_updates_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        critic_search_actor_loss_type="log_prob",
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

    assert config.sac.critic_search_actor_loss_type == "log_prob"


def test_redq_cli_overrides_update_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        redq_num_critics=4,
        redq_target_subset_size=2,
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

    assert config.sac.redq_num_critics == 4
    assert config.sac.redq_target_subset_size == 2


def test_pendulum_hard_reset_cli_overrides_env_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        pendulum_hard_reset_prob=0.75,
        pendulum_hard_reset_final_prob=0.05,
        pendulum_hard_reset_decay_steps=1234,
        pendulum_hard_reset_start_step=5000,
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
    assert config.env.pendulum_hard_reset_final_prob == 0.05
    assert config.env.pendulum_hard_reset_decay_steps == 1234
    assert config.env.pendulum_hard_reset_start_step == 5000
    assert config.env.pendulum_hard_reset_abs_theta_low == 2.1
    assert config.env.pendulum_hard_reset_abs_theta_high == 2.3
    assert config.env.pendulum_hard_reset_velocity_limit == 0.5


def test_pendulum_hard_reset_probability_respects_start_step():
    config = ExperimentConfig.from_dict(
        {
            "env": {
                "pendulum_hard_reset_prob": 0.1,
                "pendulum_hard_reset_final_prob": 0.3,
                "pendulum_hard_reset_decay_steps": 100,
                "pendulum_hard_reset_start_step": 1000,
            }
        }
    )

    assert pendulum_hard_reset_probability_at_step(config, 999) == 0.0
    assert pendulum_hard_reset_probability_at_step(config, 1000) == pytest.approx(0.1)
    assert pendulum_hard_reset_probability_at_step(config, 1050) == pytest.approx(0.2)
    assert pendulum_hard_reset_probability_at_step(config, 1100) == pytest.approx(0.3)
    assert pendulum_hard_reset_probability_at_step(config, 1200) == pytest.approx(0.3)


def test_pendulum_hard_replay_cli_overrides_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        pendulum_hard_replay_fraction=0.25,
        pendulum_hard_replay_final_fraction=0.05,
        pendulum_hard_replay_decay_steps=5000,
        pendulum_hard_replay_start_step=1234,
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
    assert config.sac.pendulum_hard_replay_final_fraction == 0.05
    assert config.sac.pendulum_hard_replay_decay_steps == 5000
    assert config.sac.pendulum_hard_replay_start_step == 1234
    assert config.sac.pendulum_hard_replay_abs_theta_low == 2.0
    assert config.sac.pendulum_hard_replay_abs_theta_high == 2.4
    assert config.sac.pendulum_hard_replay_velocity_limit == 0.75


def test_pendulum_hard_replay_fraction_schedule_respects_start_and_decay():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "pendulum_hard_replay_fraction": 0.02,
                "pendulum_hard_replay_final_fraction": 0.005,
                "pendulum_hard_replay_decay_steps": 10_000,
                "pendulum_hard_replay_start_step": 10_000,
            }
        }
    )

    assert pendulum_hard_replay_fraction_at_step(config, 9999) == 0.0
    assert pendulum_hard_replay_fraction_at_step(config, 10_000) == pytest.approx(0.02)
    assert pendulum_hard_replay_fraction_at_step(config, 15_000) == pytest.approx(0.0125)
    assert pendulum_hard_replay_fraction_at_step(config, 20_000) == pytest.approx(0.005)
    assert pendulum_hard_replay_fraction_at_step(config, 30_000) == pytest.approx(0.005)


def test_pendulum_model_replay_cli_overrides_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        pendulum_model_replay_ratio=0.25,
        pendulum_model_replay_steps_per_step=8,
        pendulum_model_replay_start_step=6000,
        pendulum_model_replay_random_action_fraction=0.5,
        pendulum_model_replay_abs_theta_low=2.6,
        pendulum_model_replay_abs_theta_high=3.1,
        pendulum_model_replay_velocity_limit=0.5,
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

    assert config.sac.pendulum_model_replay_ratio == 0.25
    assert config.sac.pendulum_model_replay_steps_per_step == 8
    assert config.sac.pendulum_model_replay_start_step == 6000
    assert config.sac.pendulum_model_replay_random_action_fraction == 0.5
    assert config.sac.pendulum_model_replay_abs_theta_low == 2.6
    assert config.sac.pendulum_model_replay_abs_theta_high == 3.1
    assert config.sac.pendulum_model_replay_velocity_limit == 0.5


def test_pendulum_model_rollout_cli_overrides_sac_config():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        pendulum_model_rollout_ratio=0.25,
        pendulum_model_rollout_starts_per_step=3,
        pendulum_model_rollout_horizon=8,
        pendulum_model_rollout_interval_steps=5,
        pendulum_model_rollout_start_step=2000,
        pendulum_model_rollout_abs_theta_low=2.6,
        pendulum_model_rollout_abs_theta_high=3.1,
        pendulum_model_rollout_velocity_limit=0.5,
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

    assert config.sac.pendulum_model_rollout_ratio == 0.25
    assert config.sac.pendulum_model_rollout_starts_per_step == 3
    assert config.sac.pendulum_model_rollout_horizon == 8
    assert config.sac.pendulum_model_rollout_interval_steps == 5
    assert config.sac.pendulum_model_rollout_start_step == 2000
    assert config.sac.pendulum_model_rollout_abs_theta_low == 2.6
    assert config.sac.pendulum_model_rollout_abs_theta_high == 3.1
    assert config.sac.pendulum_model_rollout_velocity_limit == 0.5


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
        simba_actor_log_std_floor=-1.5,
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
    assert config.sac.simba_actor_log_std_floor == pytest.approx(-1.5)
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
        sacn_target_mode="fast_last",
        sacn_horizon_lambda=0.5,
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
    assert config.sac.sacn_target_mode == "fast_last"
    assert config.sac.sacn_horizon_lambda == 0.5


def test_reference_auxiliary_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        reference_auxiliary_mode="q_filtered_replay_bc",
        reference_auxiliary_policy="controller",
        reference_auxiliary_weight=0.25,
        reference_auxiliary_stop_update=456,
        reference_auxiliary_margin=0.05,
        reference_auxiliary_filter_start_update=123,
        reference_auxiliary_q_filter_mode="online_target_unanimous",
        reference_auxiliary_replay_normalization="full_batch_mean",
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

    assert config.sac.reference_auxiliary_mode == "q_filtered_replay_bc"
    assert config.sac.reference_auxiliary_policy == "controller"
    assert config.sac.reference_auxiliary_weight == 0.25
    assert config.sac.reference_auxiliary_stop_update == 456
    assert config.sac.reference_auxiliary_margin == 0.05
    assert config.sac.reference_auxiliary_filter_start_update == 123
    assert config.sac.reference_auxiliary_q_filter_mode == "online_target_unanimous"
    assert config.sac.reference_auxiliary_replay_normalization == "full_batch_mean"


def test_actor_gradient_balance_cli_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "unused.json",
            "--sac-actor-gradient-balance-mode",
            "match_reference",
            "--sac-actor-gradient-balance-min-multiplier",
            "0.2",
            "--sac-actor-gradient-balance-max-multiplier",
            "3.0",
        ],
    )
    args = parse_args()
    config = ExperimentConfig()
    apply_overrides(config, args)

    assert config.sac.sac_actor_gradient_balance_mode == "match_reference"
    assert config.sac.sac_actor_gradient_balance_min_multiplier == pytest.approx(0.2)
    assert config.sac.sac_actor_gradient_balance_max_multiplier == pytest.approx(3.0)


def test_actor_mean_logit_l2_cli_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "unused.json",
            "--actor-mean-logit-l2-weight",
            "0.001",
        ],
    )
    args = parse_args()
    config = ExperimentConfig()
    apply_overrides(config, args)

    assert config.sac.actor_mean_logit_l2_weight == pytest.approx(0.001)


def test_actor_mean_logit_excess_threshold_cli_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "unused.json",
            "--actor-mean-logit-excess-threshold",
            "4.15",
        ],
    )
    args = parse_args()
    config = ExperimentConfig()
    apply_overrides(config, args)

    assert config.sac.actor_mean_logit_excess_threshold == pytest.approx(4.15)


def test_actor_gradient_conflict_cli_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "unused.json",
            "--sac-actor-gradient-conflict-mode",
            "project_sac",
        ],
    )
    args = parse_args()
    config = ExperimentConfig()
    apply_overrides(config, args)

    assert config.sac.sac_actor_gradient_conflict_mode == "project_sac"


def test_reference_guidance_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        reference_guidance_mode="replay_injection",
        reference_guidance_policy="best",
        reference_guidance_probability=0.25,
        reference_guidance_start_step=10_001,
        reference_guidance_dp_solution="dp.pkl",
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

    assert config.sac.reference_guidance_mode == "replay_injection"
    assert config.sac.reference_guidance_policy == "best"
    assert config.sac.reference_guidance_probability == 0.25
    assert config.sac.reference_guidance_start_step == 10_001
    assert config.sac.reference_guidance_dp_solution_path == "dp.pkl"


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


def test_pendulum_potential_shaping_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        pendulum_potential_shaping_weight=0.05,
        pendulum_potential_shaping_start_update=123,
        pendulum_potential_shaping_abs_theta_low=2.6,
        pendulum_potential_shaping_abs_theta_high=3.14,
        pendulum_potential_shaping_velocity_limit=1.0,
        pendulum_potential_shaping_source="best",
        pendulum_potential_shaping_dp_grid="dp.csv",
        pendulum_potential_shaping_controller_grid="controller.csv",
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

    assert config.sac.pendulum_potential_shaping_weight == 0.05
    assert config.sac.pendulum_potential_shaping_start_update == 123
    assert config.sac.pendulum_potential_shaping_abs_theta_low == 2.6
    assert config.sac.pendulum_potential_shaping_abs_theta_high == 3.14
    assert config.sac.pendulum_potential_shaping_velocity_limit == 1.0
    assert config.sac.pendulum_potential_shaping_source == "best"
    assert config.sac.pendulum_potential_shaping_dp_grid_path == "dp.csv"
    assert config.sac.pendulum_potential_shaping_controller_grid_path == "controller.csv"


def test_pendulum_symmetry_augmentation_cli_override_enables_sac_flag():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        pendulum_symmetry_augmentation=True,
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

    assert config.sac.pendulum_symmetry_augmentation is True


def test_pendulum_symmetry_consistency_cli_overrides_set_weights():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        pendulum_actor_symmetry_weight=0.25,
        pendulum_critic_symmetry_weight=0.5,
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

    assert config.sac.pendulum_actor_symmetry_weight == pytest.approx(0.25)
    assert config.sac.pendulum_critic_symmetry_weight == pytest.approx(0.5)


def test_reference_prior_cli_overrides_set_sac_flags():
    config = ExperimentConfig.from_dict({})
    args = SimpleNamespace(
        seed=None,
        env_id=None,
        total_steps=None,
        learning_starts=None,
        reference_prior_mode="rlpd",
        reference_prior_policy="best",
        reference_prior_ratio=0.5,
        reference_prior_source="rollout_dataset",
        reference_prior_dataset_steps=10000,
        reference_prior_dataset_seed_offset=123,
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

    assert config.sac.reference_prior_mode == "rlpd"
    assert config.sac.reference_prior_policy == "best"
    assert config.sac.reference_prior_ratio == 0.5
    assert config.sac.reference_prior_source == "rollout_dataset"
    assert config.sac.reference_prior_dataset_steps == 10000
    assert config.sac.reference_prior_dataset_seed_offset == 123
