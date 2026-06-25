import numpy as np
import gymnasium as gym
import pytest

from last_nine_rl.envs import UprightDetector
from last_nine_rl.replay import InstrumentedReplayBuffer, pendulum_hard_state_mask, summarize_saved_replay


def _pendulum_obs(theta: float, velocity: float) -> np.ndarray:
    return np.asarray([np.cos(theta), np.sin(theta), velocity], dtype=np.float32)


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
