from __future__ import annotations

import json
from pathlib import Path

from last_nine_rl.eval_cli import build_parser, main


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_evaluator_parser_is_simple_and_explicit() -> None:
    args = build_parser().parse_args(["run-dir"])
    assert args.deployment == "actor"
    assert args.device == "auto"
    assert args.episodes is None


def test_checkpoint_evaluator_runs_one_cpu_episode(tmp_path: Path, capsys) -> None:
    run = (
        ROOT
        / "artifacts"
        / "report_reproduction"
        / "models"
        / "pure_onestep"
        / "seed0"
    )
    output = tmp_path / "evaluation.json"
    assert main(
        [
            str(run),
            "--device",
            "cpu",
            "--episodes",
            "1",
            "--seed-base",
            "731000",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "last-nine-checkpoint-evaluation/v1"
    assert payload["deployment"] == "actor"
    assert payload["metrics"]["seeds"] == [731000]
    assert json.loads(capsys.readouterr().out)["metrics"]["num_eval_episodes"] == 1.0
