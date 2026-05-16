# TUM ADLR SS26 Project 15: Chasing the Nines in Deep RL

This repository contains the Week 1 infrastructure and experiment artifacts for Project 15. The goal is to study the gap between high average return and high-reliability success in continuous-control deep RL.

The current implementation uses SAC copied from CleanRL and wraps it with reliability evaluation, replay inspection, optimizer telemetry, checkpointing, TensorBoard logging, and post-hoc initial-condition analysis.

CleanRL provenance:

- Copied SAC source: `src/cleanrl/sac_continuous_action.py`
- Copied replay-buffer dependency: `src/cleanrl_utils/buffers.py`
- Provenance and license: `vendor_licenses/CLEANRL_PROVENANCE.md`

## Clone

This repo includes experiment artifacts through Git LFS. After cloning, pull LFS objects explicitly:

```bash
git clone https://github.com/alasleimi/tum-adlr-ss26-13.git
cd tum-adlr-ss26-13
git lfs install
git lfs pull
```

If you only want code and not the large artifacts:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/alasleimi/tum-adlr-ss26-13.git
```

## Setup

Python 3.11 was used locally.

```powershell
python -m pip install -e ".[test]"
```

For DeepMind Control through Gymnasium/Shimmy:

```powershell
python -m pip install -e ".[test,dm-control]"
```

Pendulum uses only the core dependencies. DMC CartPole Swingup requires the `dm-control` extra.

Run tests:

```powershell
python -m pytest
```

## Repository Layout

- `configs/`: checked-in experiment configs.
- `src/cleanrl/`: copied CleanRL SAC implementation.
- `src/last_nine_rl/`: project code for training, evaluation, aggregation, plotting, and diagnostics.
- `tests/`: unit and smoke tests.
- `docs/`: design notes, audit notes, and result summaries.
- `runs/`: raw run outputs, including checkpoints and TensorBoard files.
- `reports/`: aggregate JSON, HTML reports, plots, and post-hoc analysis.
- `proposal.pdf`: project proposal.

Large artifacts are tracked with Git LFS, including `*.pt`, TensorBoard event files, large CSV/JSONL telemetry, and generated PNG plots.

## Result Summaries

Detailed result writeups live under `docs/`:

- Week 1 design: [docs/week1_design.md](docs/week1_design.md)
- Week 1 remaining work: [docs/week1_remaining.md](docs/week1_remaining.md)
- CleanRL audit: [docs/cleanrl_audit.md](docs/cleanrl_audit.md)
- 100k Pendulum result summary with plots: [docs/pendulum_100k_results.md](docs/pendulum_100k_results.md)
- 500k Pendulum UTD1 result summary: [docs/pendulum_500k_results.md](docs/pendulum_500k_results.md)
- Pendulum dynamic-programming calibration: [docs/pendulum_dp_calibration.md](docs/pendulum_dp_calibration.md)
- Pendulum model equations and success criteria: [docs/pendulum_models_and_success_criteria.md](docs/pendulum_models_and_success_criteria.md)
- Pendulum relative success results: [docs/pendulum_relative_success_results.md](docs/pendulum_relative_success_results.md)
- Pendulum replay and representation diagnostics: [docs/pendulum_replay_diagnostics.md](docs/pendulum_replay_diagnostics.md)
- Experiment log: [docs/experiment_log.md](docs/experiment_log.md)

The short version of the Pendulum result is that the legacy `return >= -200` diagnostic barely changes from 100k to 500k, but 500k is closer to the DP/controller references under relative metrics. The detailed analysis is in [docs/pendulum_relative_success_results.md](docs/pendulum_relative_success_results.md).

## Baseline Parameters

The 100k Pendulum baseline uses CleanRL SAC defaults for the core algorithmic parameters:

| Parameter | Value |
| --- | --- |
| Environment | `Pendulum-v1` |
| Algorithm | CleanRL SAC |
| Actor/critic architecture | `256 x 256` MLPs |
| Training seeds | `0, 1, 2, 3, 4` |
| Environment steps | `100000` |
| Replay buffer size | `100000` |
| Batch size | `256` |
| Learning starts | `5000` |
| Updates per environment step | `1` |
| Discount `gamma` | `0.99` |
| Target smoothing `tau` | `0.005` |
| Actor learning rate | `3e-4` |
| Critic learning rate | `1e-3` |
| Alpha learning rate | `1e-3` |
| Policy update frequency | `2` |
| Target network update frequency | `1` |
| Device | `cuda` |
| During-training eval | 50 fixed episodes every 25000 steps |
| Post-hoc eval | 1000 fixed episodes per checkpoint |

The longer 500k investigation keeps the same SAC parameters and changes:

| Parameter | 100k baseline | 500k investigation |
| --- | ---: | ---: |
| Environment steps | `100000` | `500000` |
| Replay buffer size | `100000` | `500000` |
| During-training eval interval | `25000` | `100000` |
| During-training eval episodes | `50` | `20` |

The completed 500k UTD1 result does not materially improve the legacy fixed-threshold diagnostic over 100k, but it improves exact DP-relative and best-known-relative rates. See [docs/pendulum_500k_results.md](docs/pendulum_500k_results.md).

Replay diagnostics show that the longer runs contain more near-upright replay and better replay rewards, while critic dormant fractions increase and actor update ratios shrink. See [docs/pendulum_replay_diagnostics.md](docs/pendulum_replay_diagnostics.md).

## Success Criteria

Pendulum return is the sum of Gymnasium rewards over a 200-step episode. A larger return is better and the best possible return is near `0`.

The current primary Pendulum success criteria are:

- Task-only success: near-upright fraction `>= 0.8` and max not-upright streak `<= 50`, without using return.
- Relative success: SAC return compared with DP, the energy-shaping controller, and `max(DP, controller)`.

Legacy return-threshold diagnostics are still reported for continuity:

- Diagnostic fixed threshold: episode return `>= -200`.
- Diagnostic strict threshold: fixed threshold plus task-only success.
- Threshold ladder: `-250`, `-200`, `-150`, `-100`.

Important caveat: `-200` is not treated as a scientific success oracle because it is not feasible from every reset-support initial state. The repo includes approximate finite-horizon dynamic-programming calibration and an energy-shaping controller baseline for Pendulum initial states. See [docs/pendulum_models_and_success_criteria.md](docs/pendulum_models_and_success_criteria.md).

## Running Training

Pendulum:

```powershell
python -m last_nine_rl.train --config configs/week1_pendulum.json --seed 0
```

DMC CartPole Swingup:

```powershell
python -m last_nine_rl.train --config configs/week1_cartpole_swingup.json --seed 0
```

Short smoke run:

```powershell
python -m last_nine_rl.train `
  --config configs/week1_pendulum.json `
  --seed 0 `
  --total-steps 1000 `
  --learning-starts 100 `
  --eval-every-steps 500 `
  --replay-inspection-interval 250 `
  --diagnostics-interval 250 `
  --run-dir runs/smoke_pendulum `
  --overwrite
```

Add `--save-replay` when a run should preserve `replay_final.npz` for state-conditioned replay-buffer inspection. The main 100k and 500k datasets logged replay summaries but did not save the raw replay arrays.

Each run writes:

- `config.json`: resolved configuration.
- `events.jsonl`: structured events.
- `metrics.csv`: scalar metrics in long format.
- `eval_episodes.csv`: per-evaluation episode outcomes.
- `tensorboard/`: TensorBoard event files.
- `checkpoints/final.pt`: actor, critics, target critics, entropy state, and optimizer states.

## TensorBoard

```powershell
python -m tensorboard.main --logdir runs
```

Then open the printed local URL, usually `http://127.0.0.1:6006`.

## Post-Hoc Evaluation

Saved checkpoints can be re-evaluated without retraining.

Large fixed-seed eval:

```powershell
python -m last_nine_rl.posthoc_eval `
  --runs runs/week1_real_gpu_20260509/pendulum_100k `
  --out reports/recheck_pendulum_100k/posthoc_1000eps `
  --episodes 1000 `
  --seed-base 200000 `
  --device cpu
```

Exact Pendulum initial-condition grid:

```powershell
python -m last_nine_rl.pendulum_grid `
  --runs runs/week1_real_gpu_20260509/pendulum_100k `
  --out reports/recheck_pendulum_100k/grid_reset_support_61x41 `
  --theta-bins 61 `
  --velocity-bins 41 `
  --velocity-limit 1.0 `
  --device cpu
```

Approximate finite-horizon dynamic-programming calibration joined to the 100k checkpoint grid:

```powershell
python -m last_nine_rl.pendulum_dp `
  --out reports/recheck_pendulum_100k/dp_reset_support_241x161x81 `
  --sac-grid reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_summary.csv `
  --horizon 200 `
  --theta-bins 241 `
  --velocity-bins 161 `
  --action-bins 81 `
  --eval-theta-bins 61 `
  --eval-velocity-bins 41 `
  --eval-velocity-limit 1.0 `
  --save-solution
```

## Aggregation And Reports

Aggregate runs:

```powershell
python -m last_nine_rl.aggregate `
  --runs runs/week1_real_gpu_20260509/pendulum_100k `
  --thresholds -250 -200 -150 -100
```

Generate per-run HTML plots:

```powershell
python -m last_nine_rl.visualize `
  --runs runs/week1_real_gpu_20260509/pendulum_100k `
  --out reports/recheck_pendulum_100k/html
```

Generate cross-seed comparison plots:

```powershell
python -m last_nine_rl.compare `
  --runs runs/week1_real_gpu_20260509/pendulum_100k `
  --out reports/recheck_pendulum_100k/compare
```

## Sweep Runs

```powershell
python -m last_nine_rl.sweep `
  --config configs/week1_pendulum.json `
  --run-root runs/week1_pendulum_sweep `
  --seeds 0 1 2 3 4 `
  --same-seed-repeats 1 `
  --repeat-offsets 0 `
  --total-steps 25000 100000 `
  --updates-per-step 1 2
```

The sweep launcher writes `sweep_manifest.json`. Same-seed repeats are nondeterminism audits and should not be counted as independent training seeds.

## Reference Controller

Pendulum threshold calibration reference:

```powershell
python -m last_nine_rl.reference `
  --config configs/week1_pendulum.json `
  --episodes 100 `
  --seed-base 100000 `
  --out reports/pendulum_reference.json
```

The reference controller uses energy shaping for swing-up and local PD near upright. It is a sanity reference, not an optimal oracle.
