import numpy as np
import gymnasium as gym
import pytest

from last_nine_rl.envs import UprightDetector
from last_nine_rl.replay import (
    InstrumentedReplayBuffer,
    concatenate_replay_samples,
    concatenate_sacn_replay_samples,
    pendulum_hard_state_mask,
    summarize_saved_replay,
)


def _pendulum_obs(theta: float, velocity: float) -> np.ndarray:
    return np.asarray([np.cos(theta), np.sin(theta), velocity], dtype=np.float32)


def _add_indexed_transitions(replay: InstrumentedReplayBuffer, count: int) -> None:
    for index in range(count):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
            action_log_prob=np.asarray([0.0], dtype=np.float32),
        )


def test_disabled_priority_sampler_preserves_uniform_randint_path():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    _add_indexed_transitions(replay, 4)

    np.random.seed(731)
    expected_indices = np.random.randint(0, 4, size=32)
    np.random.seed(731)
    batch = replay.sample(32)

    np.testing.assert_array_equal(batch.observations[:, 0].cpu().numpy(), expected_indices)
    assert np.isnan(batch.sampling_probabilities.cpu().numpy()).all()
    np.testing.assert_array_equal(batch.importance_weights.cpu().numpy(), np.ones((32, 1)))
    summary = replay.summary(UprightDetector("Pendulum-v1"), current_step=4)
    assert summary["priority_enabled"] == 0.0


def test_automatic_priority_sampler_uses_uniform_mixture_and_exact_is_weights():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
        priority_mode="max",
        priority_alpha=1.0,
        priority_beta_initial=0.4,
        priority_beta_final=0.4,
        priority_uniform_fraction=0.5,
        priority_epsilon=1e-3,
        priority_clip=10.0,
    )
    _add_indexed_transitions(replay, 4)
    replay.update_priorities(np.arange(4), np.asarray([10.0, 1.0, 1.0, 1.0]))

    priorities = replay.priorities[:4, 0].astype(np.float64)
    expected_probabilities = 0.5 / 4.0 + 0.5 * priorities / priorities.sum()
    expected_max_weight = np.power(4.0 * expected_probabilities.min(), -0.4)
    np.random.seed(991)
    batch = replay.sample(20_000)
    sampled_indices = batch.replay_indices.cpu().numpy().astype(np.int64).reshape(-1)
    sampled_probabilities = batch.sampling_probabilities.cpu().numpy().reshape(-1)
    sampled_weights = batch.importance_weights.cpu().numpy().reshape(-1)

    assert np.mean(sampled_indices == 0) == pytest.approx(expected_probabilities[0], abs=0.015)
    np.testing.assert_allclose(sampled_probabilities, expected_probabilities[sampled_indices], rtol=1e-6)
    expected_weights = np.power(4.0 * sampled_probabilities, -0.4) / expected_max_weight
    np.testing.assert_allclose(sampled_weights, expected_weights, rtol=1e-6)
    assert expected_probabilities.min() >= 0.5 / 4.0
    assert sampled_weights.max() <= 1.0 + 1e-6


def test_priority_updates_clip_and_sacn_sampling_keeps_valid_starts():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
        priority_mode="bellman_residual",
        priority_uniform_fraction=0.5,
        priority_clip=3.0,
    )
    _add_indexed_transitions(replay, 6)
    replay.update_priorities([2, 2, 5], [0.25, 99.0, np.nan])

    assert replay.priorities[2, 0] == pytest.approx(3.0)
    assert replay.priorities[5, 0] == pytest.approx(replay.priority_epsilon)
    batch = replay.sample_sacn(batch_size=128, n_step=3)
    starts = batch.replay_indices.cpu().numpy().astype(np.int64).reshape(-1)
    assert starts.min() >= 0
    assert starts.max() <= 3
    assert np.isfinite(batch.sampling_probabilities.cpu().numpy()).all()
    assert np.isfinite(batch.importance_weights.cpu().numpy()).all()


def test_replay_summary_reports_near_upright_fraction_and_sampling(tmp_path):
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    near = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    far = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    replay.add(
        near.reshape(1, 3),
        near.reshape(1, 3),
        np.asarray([[0.1]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        [{}],
        step=1,
        episode_id=0,
    )
    replay.add(
        far.reshape(1, 3),
        far.reshape(1, 3),
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([True]),
        [{}],
        step=2,
        episode_id=0,
    )

    summary = replay.summary(
        UprightDetector("Pendulum-v1", cos_threshold=0.95, abs_velocity_threshold=1.0),
        current_step=3,
        action_high=np.asarray([1.0], dtype=np.float32),
    )

    assert summary["size"] == 2.0
    assert summary["near_upright_obs_fraction"] == 0.5
    assert summary["done_fraction"] == 0.5
    assert summary["action_saturation_fraction"] == 0.5

    batch = replay.sample(batch_size=2)
    assert batch.observations.shape == (2, 3)
    assert batch.actions.shape == (2, 1)
    sample_counts_after_training_sample = replay.sample_counts.copy()
    replay.sample(batch_size=2, count=False)
    np.testing.assert_array_equal(replay.sample_counts, sample_counts_after_training_sample)

    replay_path = tmp_path / "replay.npz"
    replay.save_npz(replay_path)
    saved_summary = summarize_saved_replay(
        replay_path,
        UprightDetector("Pendulum-v1", cos_threshold=0.95, abs_velocity_threshold=1.0),
        action_high=np.asarray([1.0], dtype=np.float32),
    )
    assert saved_summary["near_upright_obs_fraction"] == 0.5


def test_concatenate_replay_samples_preserves_batch_fields():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for i in range(4):
        obs = np.asarray([float(i), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            (obs + 1.0).reshape(1, 3),
            np.asarray([[0.1 * i]], dtype=np.float32),
            np.asarray([float(i)], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=i,
            episode_id=0,
        )

    first = replay.sample(batch_size=2)
    second = replay.sample(batch_size=1)
    combined = concatenate_replay_samples(first, second)

    assert combined.observations.shape[0] == 3
    assert combined.actions.shape[0] == 3
    assert combined.next_observations.shape[0] == 3
    assert combined.dones.shape[0] == 3
    assert combined.rewards.shape[0] == 3


def test_replay_teacher_labels_stay_aligned_and_overwrite_clears_them(tmp_path):
    replay = InstrumentedReplayBuffer(
        2,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index, label in enumerate((0.25, -0.75)):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
            reference_action=np.asarray([label], dtype=np.float32),
        )

    batch = replay._get_samples(np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(batch.observations[:, 0].numpy(), [0.0, 1.0])
    np.testing.assert_allclose(batch.reference_actions[:, 0].numpy(), [0.25, -0.75])

    replacement = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    replay.add(
        replacement.reshape(1, 3),
        replacement.reshape(1, 3),
        np.asarray([[0.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        [{}],
        step=3,
        episode_id=1,
    )
    overwritten = replay._get_samples(np.asarray([0], dtype=np.int64))
    assert np.isnan(overwritten.reference_actions.numpy()).all()

    replay_path = tmp_path / "labeled_replay.npz"
    replay.save_npz(replay_path)
    saved = np.load(replay_path)
    assert "reference_actions" in saved
    assert "reference_critic_actions" in saved


def test_concatenate_sacn_replay_samples_preserves_trajectory_fields():
    replay = InstrumentedReplayBuffer(
        16,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(8):
        obs = np.asarray([1.0, 0.0, float(index)], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            (obs + np.asarray([0.0, 0.0, 1.0], dtype=np.float32)).reshape(1, 3),
            np.asarray([[0.1]], dtype=np.float32),
            np.asarray([float(index)], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index,
            episode_id=0,
            action_log_prob=-0.5,
        )

    first = replay.sample_sacn(batch_size=2, n_step=3)
    second = replay.sample_sacn(batch_size=3, n_step=3)
    combined = concatenate_sacn_replay_samples(first, second)

    assert combined.observations.shape == (5, 3)
    assert combined.actions.shape == (5, 1)
    assert combined.trajectory_rewards.shape == (5, 3, 1)
    assert combined.trajectory_action_log_probs.shape == (5, 3, 1)


def test_sacn_replay_uses_teacher_label_from_sequence_start():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(5):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index,
            episode_id=0,
            action_log_prob=-0.5,
            reference_action=np.asarray([index + 0.5], dtype=np.float32),
        )
    batch = replay._get_sacn_samples(np.asarray([1, 2], dtype=np.int64), n_step=3)
    np.testing.assert_allclose(batch.reference_actions[:, 0].numpy(), [1.5, 2.5])


def test_pendulum_hard_state_sampler_oversamples_configured_band():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    low = 2.0943951023931953
    high = 2.356194490192345
    observations = [
        _pendulum_obs(0.0, 0.0),
        _pendulum_obs(low + 0.01, 0.2),
        _pendulum_obs(-(high - 0.01), -0.3),
        _pendulum_obs(np.pi, 0.0),
    ]
    for index, obs in enumerate(observations):
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
        )

    batch = replay.sample_pendulum_hard_states(
        batch_size=24,
        fraction=1.0,
        abs_theta_low=low,
        abs_theta_high=high,
        velocity_limit=1.0,
    )
    sampled_obs = batch.observations.cpu().numpy()

    assert np.all(
        pendulum_hard_state_mask(
            sampled_obs,
            abs_theta_low=low,
            abs_theta_high=high,
            velocity_limit=1.0,
        )
    )
    assert int(replay.sample_counts.sum()) == 24

    summary = replay.summary(
        UprightDetector("Pendulum-v1", cos_threshold=0.95, abs_velocity_threshold=1.0),
        current_step=5,
    )
    assert summary["pendulum_hard_state_fraction"] == 0.5
    assert summary["pendulum_hard_state_sample_count_mean"] > 0.0


def test_pendulum_hard_state_sampler_falls_back_when_band_is_absent():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(3):
        obs = _pendulum_obs(0.1 * index, 0.0)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
        )

    batch = replay.sample_pendulum_hard_states(
        batch_size=5,
        fraction=1.0,
        abs_theta_low=2.0943951023931953,
        abs_theta_high=2.356194490192345,
        velocity_limit=1.0,
    )

    assert batch.observations.shape == (5, 3)
    assert int(replay.sample_counts.sum()) == 5


def test_swd_sampler_prefers_recent_transitions_with_positive_decay():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
        swd_linear_decay_steps=2,
        swd_min_weight=0.1,
    )
    for index in range(5):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
        )

    np.random.seed(123)
    batch = replay.sample(batch_size=2000)

    assert batch.observations.shape == (2000, 3)
    assert replay.sample_counts[4, 0] > replay.sample_counts[0, 0] * 5


def test_sacn_sequence_sampler_returns_contiguous_logged_policy_density():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(6):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        next_obs = np.asarray([float(index) + 0.5, 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            next_obs.reshape(1, 3),
            np.asarray([[float(index) / 10.0]], dtype=np.float32),
            np.asarray([float(index)], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
            action_log_prob=np.asarray([-float(index)], dtype=np.float32),
        )

    np.random.seed(0)
    batch = replay.sample_sacn(batch_size=4, n_step=3)

    assert batch.observations.shape == (4, 3)
    assert batch.trajectory_observations.shape == (4, 3, 3)
    assert batch.trajectory_actions.shape == (4, 3, 1)
    assert batch.trajectory_action_log_probs.shape == (4, 3, 1)
    rewards = batch.trajectory_rewards.squeeze(-1).cpu().numpy()
    np.testing.assert_allclose(rewards[:, 1] - rewards[:, 0], np.ones(4))
    np.testing.assert_allclose(rewards[:, 2] - rewards[:, 1], np.ones(4))


def test_sacn_sequence_sampler_supports_swd_recency_weighting():
    replay = InstrumentedReplayBuffer(
        32,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
        swd_linear_decay_steps=8,
        swd_min_weight=0.05,
    )
    for index in range(24):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        next_obs = np.asarray([float(index) + 0.5, 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            next_obs.reshape(1, 3),
            np.asarray([[float(index) / 10.0]], dtype=np.float32),
            np.asarray([float(index)], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
            action_log_prob=np.asarray([-float(index)], dtype=np.float32),
        )

    np.random.seed(123)
    batch = replay.sample_sacn(batch_size=2000, n_step=3)
    start_indices = batch.observations[:, 0].cpu().numpy()

    assert start_indices.mean() > 14.0
    assert start_indices.max() <= 21.0
    rewards = batch.trajectory_rewards.squeeze(-1).cpu().numpy()
    np.testing.assert_allclose(rewards[:, 1] - rewards[:, 0], np.ones(2000))
    np.testing.assert_allclose(rewards[:, 2] - rewards[:, 1], np.ones(2000))


def test_sacn_sequence_sampler_can_cap_sequence_start_age():
    replay = InstrumentedReplayBuffer(
        32,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(24):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        next_obs = np.asarray([float(index) + 0.5, 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            next_obs.reshape(1, 3),
            np.asarray([[float(index) / 10.0]], dtype=np.float32),
            np.asarray([float(index)], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
            action_log_prob=np.asarray([-float(index)], dtype=np.float32),
        )

    np.random.seed(123)
    batch = replay.sample_sacn(batch_size=200, n_step=3, max_age_steps=5)
    start_indices = batch.observations[:, 0].cpu().numpy()

    assert start_indices.min() >= 18.0
    assert start_indices.max() <= 21.0
    rewards = batch.trajectory_rewards.squeeze(-1).cpu().numpy()
    np.testing.assert_allclose(rewards[:, 1] - rewards[:, 0], np.ones(200))
    np.testing.assert_allclose(rewards[:, 2] - rewards[:, 1], np.ones(200))


def test_sacn_sequence_sampler_supports_pendulum_hard_state_fraction():
    replay = InstrumentedReplayBuffer(
        32,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(24):
        obs = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if index in {4, 5, 6, 14, 15, 16}:
            obs = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
        next_obs = obs.copy()
        replay.add(
            obs.reshape(1, 3),
            next_obs.reshape(1, 3),
            np.asarray([[float(index) / 10.0]], dtype=np.float32),
            np.asarray([float(index)], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
            action_log_prob=np.asarray([-float(index)], dtype=np.float32),
        )

    np.random.seed(123)
    batch = replay.sample_sacn_pendulum_hard_states(
        batch_size=100,
        n_step=3,
        fraction=0.5,
        abs_theta_low=np.pi * 0.9,
        abs_theta_high=np.pi,
        velocity_limit=1.0,
    )

    hard = pendulum_hard_state_mask(
        batch.observations.cpu().numpy(),
        abs_theta_low=np.pi * 0.9,
        abs_theta_high=np.pi,
        velocity_limit=1.0,
    )
    assert hard.mean() >= 0.5
    assert np.isfinite(batch.trajectory_action_log_probs.cpu().numpy()).all()
    rewards = batch.trajectory_rewards.squeeze(-1).cpu().numpy()
    np.testing.assert_allclose(rewards[:, 1] - rewards[:, 0], np.ones(100))
    np.testing.assert_allclose(rewards[:, 2] - rewards[:, 1], np.ones(100))


def test_sacn_sequence_sampler_rejects_missing_behavior_log_probs():
    replay = InstrumentedReplayBuffer(
        4,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(3):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
        )

    with pytest.raises(ValueError, match="No valid contiguous replay trajectories"):
        replay.sample_sacn(batch_size=1, n_step=2)


def test_sacn_sequence_sampler_can_ignore_missing_behavior_log_probs():
    replay = InstrumentedReplayBuffer(
        8,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False,
    )
    for index in range(4):
        obs = np.asarray([float(index), 0.0, 0.0], dtype=np.float32)
        replay.add(
            obs.reshape(1, 3),
            obs.reshape(1, 3),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            [{}],
            step=index + 1,
            episode_id=0,
        )

    batch = replay.sample_sacn(
        batch_size=2,
        n_step=3,
        require_action_log_probs=False,
    )

    assert batch.trajectory_actions.shape == (2, 3, 1)
    assert np.isnan(batch.trajectory_action_log_probs.cpu().numpy()).all()
