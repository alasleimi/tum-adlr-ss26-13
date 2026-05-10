import numpy as np
import gymnasium as gym

from last_nine_rl.envs import UprightDetector
from last_nine_rl.replay import InstrumentedReplayBuffer, summarize_saved_replay


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
