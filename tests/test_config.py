import pytest

from last_nine_rl.config import EnvConfig, ExperimentConfig, SACConfig


def test_config_validation_rejects_late_failures():
    config = ExperimentConfig(
        sac=SACConfig(
            total_steps=10,
            learning_starts=10,
            buffer_size=4,
            batch_size=8,
            updates_per_step=0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.batch_size" in message
    assert "sac.learning_starts" in message
    assert "sac.updates_per_step" in message


def test_config_validation_rejects_cleanrl_weight_projection():
    config = ExperimentConfig(sac=SACConfig(simba_weight_projection=True))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.simba_weight_projection requires sac.simba_backbone" in str(exc_info.value)


def test_config_validation_rejects_cleanrl_distributional_critic():
    config = ExperimentConfig(sac=SACConfig(simba_distributional_critic=True))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.simba_distributional_critic requires sac.simba_backbone" in str(exc_info.value)


def test_config_validation_rejects_invalid_reference_auxiliary():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "reference_auxiliary_mode": "bad",
                "reference_auxiliary_policy": "bad",
                "reference_auxiliary_weight": -1.0,
                "reference_auxiliary_margin": -0.1,
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.reference_auxiliary_mode" in message
    assert "sac.reference_auxiliary_policy" in message
    assert "sac.reference_auxiliary_weight" in message
    assert "sac.reference_auxiliary_margin" in message


def test_config_validation_rejects_invalid_reference_critic():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "reference_critic_mode": "bad",
                "reference_critic_policy": "bad",
                "reference_critic_weight": -1.0,
                "reference_critic_margin": -0.1,
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.reference_critic_mode" in message
    assert "sac.reference_critic_policy" in message
    assert "sac.reference_critic_weight" in message
    assert "sac.reference_critic_margin" in message


def test_config_validation_rejects_invalid_optimizer_overrides():
    config = ExperimentConfig(
        sac=SACConfig(
            policy_lr_final=0.0,
            q_lr_final=-1e-4,
            alpha_initial_value=0.0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.policy_lr_final" in message
    assert "sac.q_lr_final" in message
    assert "sac.alpha_initial_value" in message


def test_config_validation_rejects_invalid_redo_overrides():
    config = ExperimentConfig(
        sac=SACConfig(
            redo_interval_updates=-1,
            redo_dormant_threshold=1.5,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.redo_interval_updates" in message
    assert "sac.redo_dormant_threshold" in message


def test_config_validation_rejects_invalid_pendulum_hard_replay_overrides():
    config = ExperimentConfig(
        sac=SACConfig(
            pendulum_hard_replay_fraction=1.5,
            pendulum_hard_replay_abs_theta_low=-0.1,
            pendulum_hard_replay_abs_theta_high=4.0,
            pendulum_hard_replay_velocity_limit=-1.0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.pendulum_hard_replay_fraction" in message
    assert "sac.pendulum_hard_replay_abs_theta_low" in message
    assert "sac.pendulum_hard_replay_abs_theta_high" in message
    assert "sac.pendulum_hard_replay_velocity_limit" in message


def test_config_validation_rejects_invalid_swd_min_weight():
    config = ExperimentConfig.from_dict({"sac": {"swd_min_weight": 1.5}})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.swd_min_weight" in str(exc_info.value)


def test_config_validation_rejects_invalid_sacn_settings():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=0,
            sacn_importance_quantile=1.5,
            sacn_max_entropy_samples=0,
            sacn_recent_max_age_steps=-1,
            sacn_min_horizon_ess_fraction=1.5,
            sacn_importance_mode="bogus",
            sacn_stop_after_steps=-1,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.sacn_n_step" in message
    assert "sac.sacn_importance_quantile" in message
    assert "sac.sacn_max_entropy_samples" in message
    assert "sac.sacn_recent_max_age_steps" in message
    assert "sac.sacn_min_horizon_ess_fraction" in message
    assert "sac.sacn_importance_mode" in message
    assert "sac.sacn_stop_after_steps" in message


def test_config_validation_allows_sacn_with_hard_replay_sequence_sampling():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            pendulum_hard_replay_fraction=0.1,
        )
    )

    config.validate()


def test_config_validation_allows_sacn_with_swd_sequence_sampling():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            swd_linear_decay_steps=10,
        )
    )

    config.validate()


def test_config_validation_rejects_sacn_with_reference_guidance():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            reference_guidance_mode="interleaved_execution",
            reference_guidance_probability=0.5,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "density-weighted SACn needs behavior-policy densities" in str(exc_info.value)


def test_config_validation_allows_no_importance_sacn_with_interleaved_reference_guidance():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            sacn_importance_mode="none",
            reference_guidance_mode="interleaved_execution",
            reference_guidance_probability=0.5,
        )
    )

    config.validate()


def test_config_validation_rejects_no_importance_sacn_with_reference_replay_injection():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            sacn_importance_mode="none",
            reference_guidance_mode="replay_injection",
            reference_guidance_probability=0.5,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "replay injection does not form contiguous real trajectories" in str(exc_info.value)


def test_config_validation_rejects_inverted_pendulum_hard_replay_band():
    config = ExperimentConfig(
        sac=SACConfig(pendulum_hard_replay_abs_theta_low=2.5, pendulum_hard_replay_abs_theta_high=2.0)
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_hard_replay_abs_theta_low must be <=" in str(exc_info.value)


def test_config_validation_rejects_pendulum_hard_replay_on_non_pendulum_env():
    config = ExperimentConfig(
        env=EnvConfig(env_id="dm_control/cartpole-swingup-v0"),
        sac=SACConfig(pendulum_hard_replay_fraction=0.2),
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_hard_replay_fraction currently supports only Pendulum" in str(exc_info.value)


def test_config_validation_rejects_invalid_pendulum_hard_reset_overrides():
    config = ExperimentConfig(
        env=EnvConfig(
            env_id="dm_control/cartpole-swingup-v0",
            pendulum_hard_reset_prob=1.5,
            pendulum_hard_reset_abs_theta_low=-0.1,
            pendulum_hard_reset_abs_theta_high=4.0,
            pendulum_hard_reset_velocity_limit=-1.0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "env.pendulum_hard_reset_prob" in message
    assert "env.pendulum_hard_reset_abs_theta_low" in message
    assert "env.pendulum_hard_reset_abs_theta_high" in message
    assert "env.pendulum_hard_reset_velocity_limit" in message


def test_config_validation_rejects_inverted_pendulum_hard_reset_band():
    config = ExperimentConfig(
        env=EnvConfig(pendulum_hard_reset_abs_theta_low=2.5, pendulum_hard_reset_abs_theta_high=2.0)
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "env.pendulum_hard_reset_abs_theta_low must be <=" in str(exc_info.value)


def test_config_validation_rejects_pendulum_hard_reset_on_non_pendulum_env():
    config = ExperimentConfig(
        env=EnvConfig(env_id="dm_control/cartpole-swingup-v0", pendulum_hard_reset_prob=0.5)
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "env.pendulum_hard_reset_prob currently supports only Pendulum" in str(exc_info.value)


def test_config_validation_accepts_redo_on_simba_backbone():
    config = ExperimentConfig(sac=SACConfig(simba_backbone=True, redo_interval_updates=100))

    config.validate()
