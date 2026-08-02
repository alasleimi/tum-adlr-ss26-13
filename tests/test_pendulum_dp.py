import gymnasium as gym
import numpy as np

from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.pendulum_dp import (
    PendulumDPParams,
    angle_normalize,
    evaluate_dp_on_grid,
    join_sac_grid,
    pendulum_step_model,
    solve_finite_horizon_dp,
)


def test_pendulum_step_model_matches_gymnasium():
    params = PendulumDPParams()
    env = gym.make("Pendulum-v1")
    try:
        env.reset(seed=0)
        env.unwrapped.state = np.asarray([0.4, -0.7], dtype=np.float64)
        obs, reward, _terminated, _truncated, _info = env.step(np.asarray([1.2], dtype=np.float32))
        model_reward, model_theta, model_velocity = pendulum_step_model(0.4, -0.7, 1.2, params)
    finally:
        env.close()

    assert np.isclose(model_reward, reward)
    assert np.allclose(obs, np.asarray([np.cos(model_theta), np.sin(model_theta), model_velocity]))


def test_angle_normalize_range():
    values = angle_normalize(np.asarray([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi]))
    assert np.all(values >= -np.pi)
    assert np.all(values < np.pi)
    assert np.isclose(values[2], 0.0)


def test_tiny_dp_prefers_upright_state():
    params = PendulumDPParams(horizon=4, theta_bins=17, velocity_bins=9, action_bins=5, eval_theta_bins=3, eval_velocity_bins=3)
    solution = solve_finite_horizon_dp(params)
    rows = evaluate_dp_on_grid(
        solution,
        theta_values=np.asarray([0.0, -np.pi]),
        velocity_values=np.asarray([0.0]),
        reliability=ReliabilityConfig(),
    )
    upright = rows[0]
    downward = rows[1]

    assert solution.values_by_remaining.shape == (5, 17 * 9)
    assert upright["dp_policy_return"] > downward["dp_policy_return"]
    assert upright["dp_policy_task_success"] == 1.0


def test_dp_comparison_separates_signed_gap_from_regret():
    dp_rows = [
        {
            "theta": "0.0",
            "theta_degrees": "0.0",
            "theta_dot": "0.0",
            "dp_policy_return": "-100.0",
            "dp_policy_task_success": "1.0",
        }
    ]
    sac_rows = [
        {
            "theta": "0.0",
            "theta_dot": "0.0",
            "mean_return": "-90.0",
            "task_success_rate": "1.0",
        }
    ]

    joined = join_sac_grid(dp_rows, sac_rows, ReliabilityConfig())

    assert joined[0]["sac_signed_gap_to_dp_policy"] == -10.0
    assert joined[0]["sac_mean_regret_to_dp_policy"] == 0.0
    assert joined[0]["sac_advantage_over_dp_policy"] == 10.0
