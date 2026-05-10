import json

from last_nine_rl.aggregate import aggregate_runs


def test_aggregate_detects_post_success_collapse_and_pooled_ci(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        {
            "step": 10,
            "type": "evaluation",
            "payload": {
                "mean_return": 1.0,
                "worst_return": 1.0,
                "success_rate": 1.0,
                "collapse_rate": 0.0,
                "num_successes": 2.0,
                "num_eval_episodes": 2.0,
            },
        },
        {
            "step": 20,
            "type": "evaluation",
            "payload": {
                "mean_return": -1.0,
                "worst_return": -2.0,
                "success_rate": 0.5,
                "collapse_rate": 0.5,
                "num_successes": 1.0,
                "num_eval_episodes": 2.0,
            },
        },
    ]
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = aggregate_runs([run_dir], thresholds=[0.0])
    assert summary["num_runs"] == 1
    assert summary["post_success_collapse_frequency"] == 1.0
    assert summary["final_pooled_eval_episodes"] == 2
    assert summary["final_pooled_successes"] == 1
    assert summary["final_fraction_seeds_mean_return_ge_0"] == 0.0


def test_aggregate_groups_same_actual_seed_repeats(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    for run_dir, mean_return, success_rate in [(run_a, 10.0, 1.0), (run_b, -10.0, 0.0)]:
        run_dir.mkdir()
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump({"seed": 5}, f)
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": 20,
                        "type": "evaluation",
                        "payload": {
                            "mean_return": mean_return,
                            "worst_return": mean_return,
                            "best_return": mean_return,
                            "success_rate": success_rate,
                            "collapse_rate": 0.0,
                            "num_successes": success_rate,
                            "num_eval_episodes": 1.0,
                        },
                    }
                )
                + "\n"
            )

    summary = aggregate_runs([run_a, run_b], thresholds=[0.0])

    assert summary["num_runs"] == 2
    assert summary["num_actual_seeds"] == 1
    assert summary["duplicate_actual_seed_count"] == 1
    assert summary["duplicate_actual_seeds"][0]["actual_seed"] == 5
    assert summary["final_mean_seed_mean_return"] == 0.0
    assert summary["final_fraction_seeds_mean_return_ge_0"] == 1.0
    assert summary["final_pooled_eval_episodes"] == 1


def test_aggregate_reports_fixed_eval_seed_difficulty(tmp_path):
    run_dirs = []
    for actual_seed in [0, 1]:
        run_dir = tmp_path / f"seed{actual_seed}"
        run_dirs.append(run_dir)
        run_dir.mkdir()
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump({"seed": actual_seed}, f)
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": 20,
                        "type": "evaluation",
                        "payload": {
                            "mean_return": 0.0,
                            "worst_return": -10.0,
                            "best_return": 10.0,
                            "success_rate": 0.5,
                            "collapse_rate": 0.0,
                            "num_successes": 1.0,
                            "num_eval_episodes": 2.0,
                        },
                    }
                )
                + "\n"
            )
        with (run_dir / "eval_episodes.csv").open("w", encoding="utf-8") as f:
            f.write(
                "step,episode_index,seed,return,length,near_upright_fraction,min_step_reward,"
                "not_near_upright_streak,success,return_success,stability_success,streak_success,"
                "strict_success,collapse\n"
            )
            f.write("20,0,100,-10,200,0.1,-1,200,0,0,0,0,0,0\n")
            f.write("20,1,101,10,200,1.0,-1,0,1,1,1,1,1,0\n")

    summary = aggregate_runs(run_dirs)

    assert summary["final_eval_unique_seed_count"] == 2
    assert summary["final_eval_seed_strict_success_rate_min"] == 0.0
    assert summary["final_eval_hardest_seeds"][0]["eval_seed"] == 100
    assert summary["final_eval_hardest_seeds"][0]["num_actual_seeds"] == 2
