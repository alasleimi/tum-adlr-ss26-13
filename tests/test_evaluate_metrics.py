from last_nine_rl.config import ReliabilityConfig
from last_nine_rl.evaluate import episode_outcome_metrics, fixed_eval_seeds, wilson_interval


def test_fixed_eval_seeds_are_step_independent():
    assert fixed_eval_seeds(seed_base=1000, episodes=4) == [1000, 1001, 1002, 1003]
    assert fixed_eval_seeds(seed_base=1000, episodes=4, explicit_seeds=[7, 11]) == [7, 11]


def test_wilson_interval_is_bounded_and_informative():
    low, high = wilson_interval(successes=10, total=10)
    assert 0.0 <= low <= high <= 1.0
    assert low < 1.0

    low, high = wilson_interval(successes=0, total=10)
    assert low == 0.0
    assert high > 0.0


def test_episode_outcomes_keep_return_task_and_strict_success_separate():
    reliability = ReliabilityConfig(
        success_return_threshold=-200.0,
        success_near_upright_fraction_threshold=0.8,
        success_max_not_near_upright_streak=50,
    )

    metrics = episode_outcome_metrics(
        returns=[-100.0, -100.0, -300.0],
        near_upright_fractions=[0.9, 0.4, 0.9],
        not_near_upright_streaks=[20, 20, 20],
        reliability=reliability,
    )

    assert metrics["success_rate"] == 2 / 3
    assert metrics["return_success_rate"] == 2 / 3
    assert metrics["stability_success_rate"] == 2 / 3
    assert metrics["streak_success_rate"] == 1.0
    assert metrics["task_success_rate"] == 2 / 3
    assert metrics["num_task_successes"] == 2.0
    assert metrics["strict_success_rate"] == 1 / 3
    assert metrics["task_reliability_nines_wilson95_low"] >= 0.0
    assert metrics["strict_reliability_nines_wilson95_low"] >= 0.0
