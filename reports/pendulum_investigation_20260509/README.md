# Pendulum Investigation 2026-05-09

This directory contains the post-hoc Pendulum reliability investigation for the 100k CleanRL SAC baseline and the ongoing longer compute sweep.

## Completed Diagnostics

- `posthoc_100k_1000eps/posthoc_eval_summary.json`: 1000 deterministic fixed eval seeds per training seed.
- `pendulum_grid_100k_61x41/index.html`: full-state initial-condition map over velocity range `[-8, 8]`.
- `pendulum_grid_100k_reset_support_61x41/index.html`: Gymnasium reset-support map over velocity range `[-1, 1]`.

Key numbers from completed diagnostics:

- 100k checkpoints, 5000 pooled post-hoc episodes: return success `0.7012`, strict success `0.6734`.
- Reset-support grid: mean cell return success `0.6918`, mean cell strict success `0.6692`.
- Full-state grid: mean cell return success `0.9226`, mean cell strict success `0.9056`.
- Reset-support region `|theta| >= 150 degrees`: mean return success `0.0`, mean strict success `0.0`.
- Reset-support region `60 <= |theta| <= 120 degrees`: mean return success `1.0`, mean strict success `1.0`.

Interpretation: the Pendulum reliability gap is concentrated near downward starts with low or opposing velocity. Full-state maps look better because many high-velocity states already contain enough momentum for easy swing-up.

## Ongoing Sweep

- Runner metadata: `logs/runner_process.json`.
- Script: `run_pendulum_investigation.ps1`.
- Run root: `..\..\runs\pendulum_investigation_20260509`.
- First condition: `pendulum_500k_utd1_buffer500k`.

The sweep is sequential. Each condition writes raw runs, TensorBoard data, aggregate reports, comparison plots, 1000-episode post-hoc eval, and a reset-support initial-condition grid after that condition finishes.

Partial result as of 2026-05-10 13:16 local time:

- Completed 500k UTD1 seeds: `0, 1, 2`.
- Running 500k UTD1 seed: `3`, checked at step `187800 / 500000`.
- 500k UTD1 seeds 0-2 post-hoc 1000-episode eval: return success `0.7023`, strict success `0.6773`.
- Report: `pendulum_500k_utd1_partial_seed0_2/posthoc_1000eps/posthoc_eval_summary.json`.

This is not materially better than the 100k five-seed post-hoc result: return success `0.7012`, strict success `0.6734`.
