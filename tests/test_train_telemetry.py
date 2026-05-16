import csv
import json

import pytest

from last_nine_rl.config import EnvConfig, EvalConfig, ExperimentConfig, SACConfig, TelemetryConfig
from last_nine_rl.train import train


def test_train_writes_complete_telemetry_and_uses_fixed_eval_seeds(tmp_path):
    run_dir = tmp_path / "run"
    config = ExperimentConfig(
        name="telemetry_smoke",
        seed=0,
        env=EnvConfig(env_id="Pendulum-v1", max_episode_steps=5),
        sac=SACConfig(
            total_steps=13,
            buffer_size=32,
            learning_starts=4,
            batch_size=4,
            updates_per_step=1,
            device="cpu",
        ),
        eval=EvalConfig(every_steps=6, episodes=2, deterministic=True, seed_base=123),
        telemetry=TelemetryConfig(
            run_root=str(tmp_path),
            log_interval_steps=6,
            replay_inspection_interval_steps=6,
            diagnostics_interval_steps=6,
            tensorboard=True,
            write_eval_returns_csv=True,
            overwrite=False,
            save_replay=True,
            save_model=True,
        ),
    )

    assert train(config, run_dir) == run_dir

    assert (run_dir / "config.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "eval_episodes.csv").is_file()
    assert (run_dir / "replay_final.npz").is_file()
    assert (run_dir / "checkpoints" / "final.pt").is_file()
    assert any((run_dir / "tensorboard").glob("events.out.tfevents.*"))

    with (run_dir / "eval_episodes.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(int(row["step"]), int(row["seed"])) for row in rows] == [
        (0, 123),
        (0, 124),
        (6, 123),
        (6, 124),
        (12, 123),
        (12, 124),
        (13, 123),
        (13, 124),
    ]

    with (run_dir / "events.jsonl").open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    evaluation_events = [event for event in events if event.get("type") == "evaluation"]
    assert [event["step"] for event in evaluation_events] == [0, 6, 12, 13]
    assert "returns" not in evaluation_events[0]["payload"]
    assert evaluation_events[0]["payload"]["seeds"] == [123, 124]
    assert evaluation_events[0]["payload"]["eval_episodes_csv"].endswith("eval_episodes.csv")

    with (run_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as f:
        metric_rows = list(csv.DictReader(f))
    assert any(row["split"] == "eval" and row["name"] == "strict_success_rate" for row in metric_rows)
    assert any(row["split"] == "eval" and row["name"] == "return_reliability_nines_wilson95_low" for row in metric_rows)
    update_rows = [row for row in metric_rows if row["split"] == "update"]
    update_names = {row["name"] for row in update_rows}
    assert "num_optimizer_updates" in update_names
    assert "q_loss_mean" in update_names

    with pytest.raises(FileExistsError):
        train(config, run_dir)
