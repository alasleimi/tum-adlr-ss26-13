from last_nine_rl.pendulum_relative import enrich_rollouts, mean_t_interval, wilson_interval
from last_nine_rl.config import ReliabilityConfig


def test_relative_success_uses_higher_return_as_better():
    sac_rows = [
        {
            "actual_seed": "0",
            "theta": "0.0",
            "theta_degrees": "0.0",
            "theta_dot": "0.0",
            "return": "-100.0",
            "stability_success": "1.0",
            "streak_success": "1.0",
            "strict_success": "1.0",
        }
    ]
    dp_rows = [
        {
            "theta": "0.0",
            "theta_dot": "0.0",
            "dp_policy_return": "-90.0",
            "dp_policy_return_success": "1.0",
            "dp_policy_strict_success": "1.0",
        }
    ]
    controller_rows = [
        {
            "theta": "0.0",
            "theta_dot": "0.0",
            "controller_return": "-120.0",
            "controller_return_success": "1.0",
            "controller_task_success": "1.0",
            "controller_strict_success": "1.0",
        }
    ]

    rows = enrich_rollouts(sac_rows, dp_rows, controller_rows, epsilon_return=5.0, reliability=ReliabilityConfig())

    assert rows[0]["beats_dp_return"] == 0.0
    assert rows[0]["near_dp_return_eps"] == 0.0
    assert rows[0]["beats_controller_return"] == 1.0
    assert rows[0]["beats_best_known_return"] == 0.0
    assert rows[0]["signed_gap_to_dp"] == 10.0
    assert rows[0]["regret_to_dp"] == 10.0
    assert rows[0]["signed_gap_to_controller"] == -20.0
    assert rows[0]["regret_to_controller"] == 0.0
    assert rows[0]["advantage_over_controller"] == 20.0


def test_relative_success_epsilon_margin():
    sac_rows = [
        {
            "actual_seed": "0",
            "theta": "0.0",
            "theta_degrees": "0.0",
            "theta_dot": "0.0",
            "return": "-94.0",
            "stability_success": "1.0",
            "streak_success": "1.0",
            "strict_success": "1.0",
        }
    ]
    dp_rows = [
        {
            "theta": "0.0",
            "theta_dot": "0.0",
            "dp_policy_return": "-90.0",
            "dp_policy_return_success": "1.0",
            "dp_policy_strict_success": "1.0",
        }
    ]
    controller_rows = [
        {
            "theta": "0.0",
            "theta_dot": "0.0",
            "controller_return": "-100.0",
            "controller_return_success": "1.0",
            "controller_task_success": "1.0",
            "controller_strict_success": "1.0",
        }
    ]

    rows = enrich_rollouts(sac_rows, dp_rows, controller_rows, epsilon_return=5.0, reliability=ReliabilityConfig())

    assert rows[0]["beats_dp_return"] == 0.0
    assert rows[0]["near_dp_return_eps"] == 1.0
    assert rows[0]["near_best_known_return_eps"] == 1.0


def test_intervals_are_bounded():
    wilson = wilson_interval(0, 10)
    t_interval = mean_t_interval([0.5, 0.7, 0.9])

    assert wilson["low"] == 0.0
    assert 0.0 < wilson["high"] < 1.0
    assert t_interval["low"] < t_interval["mean"] < t_interval["high"]
