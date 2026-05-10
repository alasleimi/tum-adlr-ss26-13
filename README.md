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
- CleanRL audit: [docs/cleanrl_audit.md](docs/cleanrl_audit.md)
- 100k Pendulum result summary with plots: [docs/pendulum_100k_results.md](docs/pendulum_100k_results.md)
- Experiment log: [docs/experiment_log.md](docs/experiment_log.md)

The short version of the 100k Pendulum result is that CleanRL SAC reaches good average return but not high reliability. The detailed analysis, including the initial-condition maps, is in [docs/pendulum_100k_results.md](docs/pendulum_100k_results.md).

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

The current partial 500k result does not materially improve reliability over 100k. See `docs/experiment_log.md`.

## Success Criteria

Pendulum return is the sum of Gymnasium rewards over a 200-step episode. A larger return is better and the best possible return is near `0`.

The current operational success criteria are:

- Return success: episode return `>= -200`.
- Strict success: return success, near-upright fraction `>= 0.8`, and maximum not-near-upright streak `<= 50`.
- Threshold ladder: `-250`, `-200`, `-150`, `-100`.

Important caveat: `-200` is not an oracle-derived feasibility threshold. The repo includes an energy-shaping plus local-PD reference controller for calibration, but future work should add a near-oracle planner or dynamic-programming calibration for Pendulum initial states.

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
