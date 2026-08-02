import importlib.util

import numpy as np
import pytest

from last_nine_rl.envs import UprightDetector, make_env, set_pendulum_hard_reset_probability


def test_pendulum_env_is_flat_box_and_float32():
    env = make_env("Pendulum-v1", seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        assert obs.shape == (3,)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert next_obs.dtype == np.float32
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_pendulum_hard_reset_targets_configured_abs_theta_band():
    low = 2.1
    high = 2.2
    env = make_env(
        "Pendulum-v1",
        seed=0,
        pendulum_hard_reset_prob=1.0,
        pendulum_hard_reset_abs_theta_low=low,
        pendulum_hard_reset_abs_theta_high=high,
        pendulum_hard_reset_velocity_limit=0.2,
    )
    try:
        obs, info = env.reset(seed=123)
        theta = abs(float(np.arctan2(obs[1], obs[0])))
        assert info["hard_reset"] is True
        assert obs.dtype == np.float32
        assert low <= theta <= high
        assert abs(float(obs[2])) <= 0.2
    finally:
        env.close()


def test_pendulum_hard_reset_can_be_enabled_after_zero_probability_start():
    env = make_env(
        "Pendulum-v1",
        seed=0,
        pendulum_hard_reset_prob=0.0,
        pendulum_hard_reset_enabled=True,
        pendulum_hard_reset_abs_theta_low=2.1,
        pendulum_hard_reset_abs_theta_high=2.2,
        pendulum_hard_reset_velocity_limit=0.2,
    )
    try:
        _, info = env.reset(seed=123)
        assert "hard_reset" not in info
        assert set_pendulum_hard_reset_probability(env, 1.0) is True

        obs, info = env.reset()
        theta = abs(float(np.arctan2(obs[1], obs[0])))
        assert info["hard_reset"] is True
        assert 2.1 <= theta <= 2.2
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("shimmy") is None, reason="shimmy is not installed")
def test_dmcontrol_cartpole_swingup_env_is_flattened():
    env = make_env("dm_control/cartpole-swingup-v0", seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        assert obs.ndim == 1
        assert obs.shape[0] == 5
        assert env.action_space.shape == (1,)
    finally:
        env.close()


def test_near_upright_detector_pendulum():
    detector = UprightDetector("Pendulum-v1", cos_threshold=0.95, abs_velocity_threshold=1.0)
    observations = np.asarray(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.0, 0.1],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(detector.near_upright(observations), np.asarray([True, False, False]))
