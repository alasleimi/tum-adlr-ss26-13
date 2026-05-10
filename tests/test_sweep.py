from types import SimpleNamespace

from last_nine_rl.config import ExperimentConfig, SACConfig
from last_nine_rl.sweep import build_sweep_entries


def test_sweep_distinguishes_same_seed_repeats_from_new_seed_offsets(tmp_path):
    base = ExperimentConfig(
        name="sweep_test",
        sac=SACConfig(total_steps=10, learning_starts=1, batch_size=4, buffer_size=32),
    )
    args = SimpleNamespace(
        seeds=[5],
        same_seed_repeats=2,
        repeat_offsets=[0, 100],
        total_steps=[10],
        updates_per_step=[1],
        batch_sizes=[4],
        buffer_sizes=[32],
        overwrite=True,
    )

    entries = build_sweep_entries(base, args, tmp_path)

    assert [
        (entry["base_seed"], entry["repeat_offset"], entry["repeat_index"], entry["actual_seed"])
        for entry in entries
    ] == [
        (5, 0, 0, 5),
        (5, 0, 1, 5),
        (5, 100, 0, 105),
        (5, 100, 1, 105),
    ]
    assert [entry["config"]["seed"] for entry in entries] == [5, 5, 105, 105]
    assert all(entry["config"]["telemetry"]["overwrite"] for entry in entries)
    assert "base5_offset0_repeat1_seed5" in entries[1]["run_dir"]
