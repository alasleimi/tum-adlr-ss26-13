import pytest

from last_nine_rl.config import (
    EnvConfig,
    ExperimentConfig,
    SACConfig,
    needs_actor_reference_actions,
    reference_auxiliary_loss_can_be_active,
)


def test_config_validates_conservative_automatic_prioritized_replay():
    valid = ExperimentConfig(
        sac=SACConfig(
            replay_priority_mode="max",
            replay_priority_uniform_fraction=0.5,
        )
    )
    valid.validate()

    invalid_floor = ExperimentConfig(
        sac=SACConfig(
            replay_priority_mode="bellman_residual",
            replay_priority_uniform_fraction=0.49,
        )
    )
    with pytest.raises(ValueError, match="must be >= 0.5"):
        invalid_floor.validate()

    incompatible = ExperimentConfig(
        sac=SACConfig(
            replay_priority_mode="critic_disagreement",
            swd_linear_decay_steps=10,
            pendulum_hard_replay_fraction=0.1,
        )
    )
    with pytest.raises(ValueError) as exc_info:
        incompatible.validate()
    assert "cannot be combined with SWD" in str(exc_info.value)
    assert "cannot be combined with hard-range" in str(exc_info.value)


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


def test_config_validation_rejects_negative_actor_updates_per_trigger():
    config = ExperimentConfig.from_dict({"sac": {"actor_updates_per_trigger": -1}})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.actor_updates_per_trigger" in str(exc_info.value)


def test_config_validation_rejects_negative_checkpoint_interval():
    config = ExperimentConfig.from_dict({"telemetry": {"checkpoint_interval_steps": -1}})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "telemetry.checkpoint_interval_steps" in str(exc_info.value)


def test_config_validation_rejects_nonpositive_cql_interval():
    config = ExperimentConfig.from_dict({"sac": {"cql_interval_updates": 0}})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.cql_interval_updates" in str(exc_info.value)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"critic_search_actor_weight": -0.1}, "sac.critic_search_actor_weight"),
        ({"critic_search_num_actions": 0}, "sac.critic_search_num_actions"),
        ({"critic_search_margin": -0.1}, "sac.critic_search_margin"),
        ({"critic_search_start_update": -1}, "sac.critic_search_start_update"),
    ],
)
def test_config_validation_rejects_invalid_critic_search_parameters(overrides, expected):
    config = ExperimentConfig.from_dict({"sac": overrides})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert expected in str(exc_info.value)


@pytest.mark.parametrize(
    ("sac_overrides", "expected"),
    [
        ({"self_imitation_weight": -0.1}, "sac.self_imitation_weight"),
        ({"self_imitation_loss_type": "nll"}, "sac.self_imitation_loss_type"),
        ({"self_imitation_start_step": -1}, "sac.self_imitation_start_step"),
        ({"self_imitation_temperature": 0.0}, "sac.self_imitation_temperature"),
        ({"self_imitation_margin": -0.1}, "sac.self_imitation_margin"),
        ({"self_imitation_max_weight": 0.0}, "sac.self_imitation_max_weight"),
    ],
)
def test_config_validation_rejects_invalid_self_imitation_parameters(sac_overrides, expected):
    config = ExperimentConfig.from_dict({"sac": sac_overrides})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert expected in str(exc_info.value)


def test_config_validation_rejects_invalid_actor_q_aggregation():
    config = ExperimentConfig.from_dict({"sac": {"actor_q_aggregation": "median"}})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.actor_q_aggregation" in str(exc_info.value)


def test_config_validation_rejects_invalid_late_actor_q_aggregation():
    config = ExperimentConfig.from_dict(
        {"sac": {"actor_q_aggregation_late": "median", "actor_q_aggregation_switch_step": -1}}
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.actor_q_aggregation_late" in message
    assert "sac.actor_q_aggregation_switch_step" in message


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
                "reference_auxiliary_stop_update": -1,
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
    assert "sac.reference_auxiliary_stop_update" in message


def test_config_validation_accepts_q_filtered_replay_bc():
    config = ExperimentConfig.from_dict(
        {"sac": {"reference_auxiliary_mode": "q_filtered_replay_bc"}}
    )

    config.validate()


def test_config_validation_accepts_robust_q_filter_and_full_batch_replay_normalization():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "reference_auxiliary_mode": "q_filtered_replay_bc",
                "reference_auxiliary_q_filter_mode": "online_target_unanimous",
                "reference_auxiliary_replay_normalization": "full_batch_mean",
            }
        }
    )

    config.validate()


def test_config_validation_rejects_inapplicable_reference_filter_options():
    robust_without_q_filter = ExperimentConfig(
        sac=SACConfig(
            reference_auxiliary_mode="bc",
            reference_auxiliary_q_filter_mode="online_target_unanimous",
        )
    )
    with pytest.raises(ValueError, match="q_filter_mode is active only"):
        robust_without_q_filter.validate()

    full_batch_without_replay_filter = ExperimentConfig(
        sac=SACConfig(
            reference_auxiliary_mode="q_filtered_bc",
            reference_auxiliary_replay_normalization="full_batch_mean",
        )
    )
    with pytest.raises(ValueError, match="replay_normalization is active only"):
        full_batch_without_replay_filter.validate()


def test_config_validation_rejects_invalid_independent_alpha_lr():
    config = ExperimentConfig(
        sac=SACConfig(alpha_lr=0.0, alpha_lr_final=1e-5)
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.alpha_lr must be positive" in message

    missing_initial = ExperimentConfig(sac=SACConfig(alpha_lr_final=1e-5))
    with pytest.raises(ValueError, match="alpha_lr_final requires sac.alpha_lr"):
        missing_initial.validate()


def test_legacy_config_without_alpha_lr_keeps_q_schedule_fallback():
    config = ExperimentConfig.from_dict(
        {"sac": {"q_lr": 1e-4, "q_lr_final": 5e-5}}
    )

    config.validate()

    assert config.sac.alpha_lr is None
    assert config.sac.alpha_lr_final is None


def test_config_validation_rejects_invalid_actor_gradient_balance():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "sac_actor_gradient_balance_mode": "bad",
                "sac_actor_gradient_balance_min_multiplier": 2.0,
                "sac_actor_gradient_balance_max_multiplier": 1.0,
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.sac_actor_gradient_balance_mode" in message
    assert "sac.sac_actor_gradient_balance_min_multiplier" in message
    assert "sac.sac_actor_gradient_balance_max_multiplier" in message


def test_config_validates_actor_gradient_conflict_projection_and_auxiliary_exclusion():
    ExperimentConfig(
        sac=SACConfig(
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=1.0,
            sac_actor_gradient_balance_mode="match_reference",
            sac_actor_gradient_conflict_mode="project_sac",
        )
    ).validate()

    invalid_mode = ExperimentConfig(
        sac=SACConfig(sac_actor_gradient_conflict_mode="not_a_projection")
    )
    with pytest.raises(ValueError, match="sac.sac_actor_gradient_conflict_mode"):
        invalid_mode.validate()

    for auxiliary_name in (
        "critic_search_actor_weight",
        "self_imitation_weight",
        "pendulum_actor_symmetry_weight",
        "actor_mean_logit_l2_weight",
    ):
        incompatible = ExperimentConfig(
            sac=SACConfig(
                sac_actor_gradient_conflict_mode="project_sac",
                **{auxiliary_name: 0.1},
            )
        )
        with pytest.raises(ValueError) as exc_info:
            incompatible.validate()
        assert "supports exactly the SAC and reference-BC actor objectives" in str(
            exc_info.value
        )
        assert auxiliary_name in str(exc_info.value)


def test_config_validation_rejects_negative_actor_mean_logit_l2_weight():
    config = ExperimentConfig(
        sac=SACConfig(actor_mean_logit_l2_weight=-1e-4)
    )

    with pytest.raises(ValueError, match="sac.actor_mean_logit_l2_weight"):
        config.validate()


def test_config_validation_rejects_negative_actor_mean_logit_excess_threshold():
    config = ExperimentConfig(
        sac=SACConfig(actor_mean_logit_excess_threshold=-1e-4)
    )

    with pytest.raises(
        ValueError, match="sac.actor_mean_logit_excess_threshold"
    ):
        config.validate()


def test_simba_actor_log_std_floor_is_default_off_and_validates_scope():
    assert SACConfig().simba_actor_log_std_floor is None
    ExperimentConfig(
        sac=SACConfig(simba_backbone=True, simba_actor_log_std_floor=-1.5)
    ).validate()

    with pytest.raises(ValueError, match="simba_actor_log_std_floor requires"):
        ExperimentConfig(
            sac=SACConfig(simba_backbone=False, simba_actor_log_std_floor=-1.5)
        ).validate()
    with pytest.raises(ValueError, match="simba_actor_log_std_floor must be less than 2"):
        ExperimentConfig(
            sac=SACConfig(simba_backbone=True, simba_actor_log_std_floor=2.0)
        ).validate()


def test_config_validation_accepts_online_target_critic_search_filter():
    ExperimentConfig(
        sac=SACConfig(
            critic_search_filter_mode="online_target_unanimous_advantage"
        )
    ).validate()


def test_actor_reference_requirements_include_increasing_bc_schedule_and_filter_only():
    increasing_bc = SACConfig(
        reference_auxiliary_mode="bc",
        reference_auxiliary_weight=0.0,
        reference_auxiliary_weight_final=1.0,
        reference_auxiliary_decay_updates=10,
    )
    assert reference_auxiliary_loss_can_be_active(increasing_bc)
    assert needs_actor_reference_actions(increasing_bc)

    filter_only = SACConfig(
        reference_auxiliary_mode="none",
        reference_auxiliary_weight=0.0,
        sac_actor_filter_mode="reference_online_unanimous",
    )
    assert not reference_auxiliary_loss_can_be_active(filter_only)
    assert needs_actor_reference_actions(filter_only)


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
            alpha_min_value=-1e-4,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.policy_lr_final" in message
    assert "sac.q_lr_final" in message
    assert "sac.alpha_initial_value" in message
    assert "sac.alpha_min_value" in message


def test_config_validation_rejects_invalid_sac_actor_objective_mode():
    config = ExperimentConfig(sac=SACConfig(sac_actor_objective_mode="bad"))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.sac_actor_objective_mode" in str(exc_info.value)


def test_config_validation_rejects_invalid_critic_search_actor_loss_type():
    config = ExperimentConfig(sac=SACConfig(critic_search_actor_loss_type="bad"))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.critic_search_actor_loss_type" in str(exc_info.value)


def test_config_validation_rejects_invalid_redq_overrides():
    config = ExperimentConfig(
        sac=SACConfig(
            redq_num_critics=1,
            redq_target_subset_size=3,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.redq_num_critics" in message
    assert "sac.redq_target_subset_size" in message


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
            pendulum_hard_replay_final_fraction=-0.1,
            pendulum_hard_replay_decay_steps=-1,
            pendulum_hard_replay_start_step=-1,
            pendulum_hard_replay_abs_theta_low=-0.1,
            pendulum_hard_replay_abs_theta_high=4.0,
            pendulum_hard_replay_velocity_limit=-1.0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.pendulum_hard_replay_fraction" in message
    assert "sac.pendulum_hard_replay_final_fraction" in message
    assert "sac.pendulum_hard_replay_decay_steps" in message
    assert "sac.pendulum_hard_replay_start_step" in message
    assert "sac.pendulum_hard_replay_abs_theta_low" in message
    assert "sac.pendulum_hard_replay_abs_theta_high" in message
    assert "sac.pendulum_hard_replay_velocity_limit" in message


def test_config_validation_rejects_invalid_pendulum_potential_shaping():
    config = ExperimentConfig(
        sac=SACConfig(
            pendulum_potential_shaping_weight=-0.1,
            pendulum_potential_shaping_start_update=-1,
            pendulum_potential_shaping_abs_theta_low=3.2,
            pendulum_potential_shaping_abs_theta_high=-0.1,
            pendulum_potential_shaping_velocity_limit=-1.0,
            pendulum_potential_shaping_source="bad",
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.pendulum_potential_shaping_weight" in message
    assert "sac.pendulum_potential_shaping_start_update" in message
    assert "sac.pendulum_potential_shaping_abs_theta_low" in message
    assert "sac.pendulum_potential_shaping_abs_theta_high" in message
    assert "sac.pendulum_potential_shaping_velocity_limit" in message
    assert "sac.pendulum_potential_shaping_source" in message


def test_config_validation_rejects_invalid_pendulum_hard_reset_schedule():
    config = ExperimentConfig(
        env=EnvConfig(
            pendulum_hard_reset_prob=0.0,
            pendulum_hard_reset_final_prob=1.5,
            pendulum_hard_reset_decay_steps=-1,
            pendulum_hard_reset_start_step=-1,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "env.pendulum_hard_reset_final_prob" in message
    assert "env.pendulum_hard_reset_decay_steps" in message
    assert "env.pendulum_hard_reset_start_step" in message


def test_config_validation_rejects_hard_reset_ramp_from_zero():
    config = ExperimentConfig(
        env=EnvConfig(
            pendulum_hard_reset_prob=0.0,
            pendulum_hard_reset_final_prob=0.1,
            pendulum_hard_reset_decay_steps=100,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "env.pendulum_hard_reset_prob must be > 0" in str(exc_info.value)


def test_config_validation_rejects_invalid_pendulum_model_replay_overrides():
    config = ExperimentConfig(
        sac=SACConfig(
            pendulum_model_replay_ratio=1.0,
            pendulum_model_replay_steps_per_step=0,
            pendulum_model_replay_start_step=-1,
            pendulum_model_replay_random_action_fraction=1.5,
            pendulum_model_replay_abs_theta_low=-0.1,
            pendulum_model_replay_abs_theta_high=4.0,
            pendulum_model_replay_velocity_limit=-1.0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.pendulum_model_replay_ratio" in message
    assert "sac.pendulum_model_replay_steps_per_step" in message
    assert "sac.pendulum_model_replay_start_step" in message
    assert "sac.pendulum_model_replay_random_action_fraction" in message
    assert "sac.pendulum_model_replay_abs_theta_low" in message
    assert "sac.pendulum_model_replay_abs_theta_high" in message
    assert "sac.pendulum_model_replay_velocity_limit" in message


def test_config_validation_rejects_invalid_pendulum_model_rollout_overrides():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=1,
            pendulum_model_rollout_ratio=1.0,
            pendulum_model_rollout_starts_per_step=0,
            pendulum_model_rollout_horizon=0,
            pendulum_model_rollout_interval_steps=0,
            pendulum_model_rollout_start_step=-1,
            pendulum_model_rollout_abs_theta_low=-0.1,
            pendulum_model_rollout_abs_theta_high=4.0,
            pendulum_model_rollout_velocity_limit=-1.0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.pendulum_model_rollout_ratio" in message
    assert "sac.pendulum_model_rollout_starts_per_step" in message
    assert "sac.pendulum_model_rollout_horizon" in message
    assert "sac.pendulum_model_rollout_interval_steps" in message
    assert "sac.pendulum_model_rollout_start_step" in message
    assert "sac.pendulum_model_rollout_abs_theta_low" in message
    assert "sac.pendulum_model_rollout_abs_theta_high" in message
    assert "sac.pendulum_model_rollout_velocity_limit" in message


def test_config_validation_rejects_model_rollout_without_sacn():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=1,
            pendulum_model_rollout_ratio=0.25,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_model_rollout_ratio requires sac.sacn_n_step > 1" in str(exc_info.value)


def test_config_validation_rejects_short_model_rollout_horizon():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=8,
            pendulum_model_rollout_ratio=0.25,
            pendulum_model_rollout_horizon=4,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_model_rollout_horizon must be >= sac.sacn_n_step" in str(exc_info.value)


def test_config_validation_allows_sacn_model_rollout():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=8,
            pendulum_model_rollout_ratio=0.25,
            pendulum_model_rollout_horizon=8,
        )
    )

    config.validate()


def test_config_validation_rejects_invalid_swd_min_weight():
    config = ExperimentConfig.from_dict({"sac": {"swd_min_weight": 1.5}})

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.swd_min_weight" in str(exc_info.value)


def test_config_validation_rejects_invalid_reference_prior():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "reference_prior_mode": "bad",
                "reference_prior_policy": "bad",
                "reference_prior_ratio": 1.0,
                "reference_prior_source": "bad",
                "reference_prior_dataset_steps": -1,
                "reference_prior_dataset_seed_offset": -1,
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.reference_prior_mode" in message
    assert "sac.reference_prior_policy" in message
    assert "sac.reference_prior_ratio" in message
    assert "sac.reference_prior_source" in message
    assert "sac.reference_prior_dataset_steps" in message
    assert "sac.reference_prior_dataset_seed_offset" in message


def test_config_validation_rejects_rollout_reference_prior_without_dataset():
    config = ExperimentConfig.from_dict(
        {
            "sac": {
                "reference_prior_mode": "rlpd",
                "reference_prior_ratio": 0.5,
                "reference_prior_source": "rollout_dataset",
                "reference_prior_dataset_steps": 0,
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.reference_prior_dataset_steps must be > 0" in str(exc_info.value)


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
            sacn_target_mode="bogus",
            sacn_horizon_lambda=0.0,
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
    assert "sac.sacn_target_mode" in message
    assert "sac.sacn_horizon_lambda" in message


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


def test_config_validation_allows_delayed_reference_replay_injection_after_sacn_stops():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            sacn_importance_mode="none",
            sacn_stop_after_steps=10_000,
            reference_guidance_mode="replay_injection",
            reference_guidance_probability=0.5,
            reference_guidance_start_step=10_001,
        )
    )

    config.validate()


def test_config_validation_rejects_reference_replay_injection_on_sacn_stop_step():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            sacn_importance_mode="none",
            sacn_stop_after_steps=10_000,
            reference_guidance_mode="replay_injection",
            reference_guidance_probability=0.5,
            reference_guidance_start_step=10_000,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "reference_guidance_start_step > sac.sacn_stop_after_steps" in str(exc_info.value)


def test_config_validation_rejects_negative_reference_guidance_start_step():
    config = ExperimentConfig(sac=SACConfig(reference_guidance_start_step=-1))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.reference_guidance_start_step" in str(exc_info.value)


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


def test_config_validation_rejects_pendulum_model_replay_on_non_pendulum_env():
    config = ExperimentConfig(
        env=EnvConfig(env_id="dm_control/cartpole-swingup-v0"),
        sac=SACConfig(pendulum_model_replay_ratio=0.2),
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_model_replay_ratio currently supports only Pendulum" in str(exc_info.value)


def test_config_validation_rejects_pendulum_model_replay_with_always_on_sacn():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            sacn_stop_after_steps=0,
            pendulum_model_replay_ratio=0.2,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "model replay is mixed into one-step SAC updates" in str(exc_info.value)


def test_config_validation_allows_pendulum_model_replay_after_sacn_warmup():
    config = ExperimentConfig(
        sac=SACConfig(
            sacn_n_step=3,
            sacn_stop_after_steps=10_000,
            pendulum_model_replay_ratio=0.2,
        )
    )

    config.validate()


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


def test_config_validation_rejects_pendulum_symmetry_on_non_pendulum_env():
    config = ExperimentConfig(
        env=EnvConfig(env_id="dm_control/cartpole-swingup-v0"),
        sac=SACConfig(pendulum_symmetry_augmentation=True),
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_symmetry_augmentation currently supports only Pendulum" in str(exc_info.value)


def test_config_validation_rejects_pendulum_symmetry_with_sacn_density_importance():
    config = ExperimentConfig(
        sac=SACConfig(
            pendulum_symmetry_augmentation=True,
            sacn_n_step=3,
            sacn_importance_mode="density",
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.pendulum_symmetry_augmentation with SACn requires" in str(exc_info.value)


@pytest.mark.parametrize(
    "field_name",
    ["pendulum_actor_symmetry_weight", "pendulum_critic_symmetry_weight"],
)
def test_config_validation_rejects_negative_pendulum_symmetry_weights(field_name):
    config = ExperimentConfig(sac=SACConfig(**{field_name: -0.1}))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert f"sac.{field_name} must be non-negative" in str(exc_info.value)


@pytest.mark.parametrize(
    "field_name",
    ["pendulum_actor_symmetry_weight", "pendulum_critic_symmetry_weight"],
)
def test_config_validation_rejects_symmetry_consistency_on_non_pendulum(field_name):
    config = ExperimentConfig(
        env=EnvConfig(env_id="dm_control/cartpole-swingup-v0"),
        sac=SACConfig(**{field_name: 0.1}),
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "symmetry-consistency losses currently support only Pendulum" in str(exc_info.value)


def test_config_validation_accepts_redo_on_simba_backbone():
    config = ExperimentConfig(sac=SACConfig(simba_backbone=True, redo_interval_updates=100))

    config.validate()


def test_legacy_l2_feature_norm_network_variant_migrates():
    config = ExperimentConfig.from_dict({"sac": {"network_variant": "l2_feature_norm"}})

    assert config.sac.l2_feature_norm is True


def test_config_validation_rejects_l2_feature_norm_with_simba_backbone():
    config = ExperimentConfig(sac=SACConfig(l2_feature_norm=True, simba_backbone=True))

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    assert "sac.l2_feature_norm" in str(exc_info.value)
