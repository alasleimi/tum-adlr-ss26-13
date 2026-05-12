import csv
import json

from last_nine_rl.replay_diagnostics_report import write_replay_diagnostics_report


def test_replay_diagnostics_report_writes_summary_and_csv(tmp_path):
    run_dir = tmp_path / "runs" / "seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"seed": 0}), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"step": 10, "type": "run_complete", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "name", "value"])
        writer.writerow([10, "eval", "mean_return", -100.0])
        writer.writerow([10, "eval", "return_success_rate", 1.0])
        writer.writerow([10, "eval", "strict_success_rate", 0.5])
        writer.writerow([10, "replay", "near_upright_any_transition_fraction", 0.75])
        writer.writerow([10, "replay", "action_saturation_fraction", 0.1])
        writer.writerow([10, "diagnostics", "q1_layer1_dormant_fraction", 0.4])
        writer.writerow([10, "diagnostics", "q1_layer1_effective_rank_fraction", 0.2])
        writer.writerow([10, "update", "q_grad_norm_mean", 3.0])

    out = tmp_path / "report"
    result = write_replay_diagnostics_report([("condition", tmp_path / "runs")], out)

    assert result["summary"]["condition"]["num_complete_runs"] == 1
    assert result["summary"]["condition"]["eval_strict_success_rate_mean"] == 0.5
    assert (out / "replay_diagnostics_snapshot.csv").is_file()
    assert (out / "replay_diagnostics_summary.json").is_file()
    assert (out / "replay_eval_over_steps.png").is_file()
