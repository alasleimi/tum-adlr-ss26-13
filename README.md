# Chasing the Nines in Deep RL: Week 1

This repository implements the Week 1 milestone from `proposal.pdf`: a reproducible CleanRL SAC baseline with reliability metrics and replay inspection for Gymnasium Pendulum and DeepMind Control CartPole Swingup through Shimmy.

The SAC implementation is vendored from CleanRL commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`; see `vendor_licenses/CLEANRL_PROVENANCE.md`.

## Setup

```powershell
python -m pip install -e ".[dm-control,test]"
```

`dm-control` is only required for `dm_control/cartpole-swingup-v0`. Pendulum runs with the core dependencies.

## Run Baselines

```powershell
python -m last_nine_rl.train --config configs/week1_pendulum.json --seed 0
python -m last_nine_rl.train --config configs/week1_cartpole_swingup.json --seed 0
```

For a short smoke run:

```powershell
python -m last_nine_rl.train --config configs/week1_pendulum.json --seed 0 --total-steps 1000 --learning-starts 100 --eval-every-steps 500 --replay-inspection-interval 250 --diagnostics-interval 250 --run-dir runs/smoke_pendulum --overwrite
```

If reusing a run directory intentionally, pass `--overwrite`; otherwise training fails instead of appending incompatible telemetry.

Each run writes:

- `config.json`: exact resolved configuration.
- `events.jsonl`: structured telemetry for episodes, updates, replay inspections, diagnostics, evaluation, and checkpoint paths.
- `metrics.csv`: long-format scalar metrics for plotting.
- `eval_episodes.csv`: per-evaluation episode seeds, returns, lengths, near-upright fractions, and collapse/success labels.
- `tensorboard/`: TensorBoard event files for live scalar inspection.
- `checkpoints/final.pt`: SAC actor, critics, target critics, entropy state, and optimizer states.

Training always evaluates the untrained policy at step 0 and guarantees a final evaluation at `total_steps`, even when `total_steps` is not divisible by the configured evaluation interval.
`success_rate` remains return-threshold success for comparability. `strict_success_rate` adds the Week 1 reliability checks: return threshold, near-upright fraction, and maximum not-near-upright streak.

DeepMind Control tasks are run through Gymnasium IDs registered by Shimmy, e.g. `dm_control/cartpole-swingup-v0`. The code uses Gymnasium everywhere; Shimmy adapts `dm_control` environments into Gymnasium spaces.

The checked-in Week 1 configs expose only knobs that the CleanRL-backed baseline actually uses. Architecture sweeps should be added as a separate CleanRL-derived variant, because the copied CleanRL SAC networks are fixed at `256 x 256`.

Live TensorBoard:

```powershell
python -m tensorboard.main --logdir runs
```

If `save_replay` is enabled in a config, inspect the saved buffer with:

```powershell
python -m last_nine_rl.inspect_replay --replay runs/<run>/replay_final.npz --env-id Pendulum-v1 --action-high 2.0
```

Pendulum threshold calibration reference:

```powershell
python -m last_nine_rl.reference --config configs/week1_pendulum.json --episodes 100 --seed-base 100000 --out reports/pendulum_reference.json
```

## Aggregate Runs

```powershell
python -m last_nine_rl.aggregate --runs runs/week1_pendulum --thresholds -200 -150 -100
```

## Sweep Runs

```powershell
python -m last_nine_rl.sweep --config configs/week1_pendulum.json --run-root runs/week1_pendulum_sweep --seeds 0 1 2 3 4 --same-seed-repeats 1 --repeat-offsets 0 --total-steps 25000 100000 --updates-per-step 1 2
```

The sweep launcher writes `sweep_manifest.json` with the base seed, same-seed repeat index, repeat offset, actual seed, and scale knobs for every run. `--same-seed-repeats` repeats an actual seed to audit nondeterminism; it is not an independent statistical replicate. Nonzero `--repeat-offsets` intentionally creates additional random seeds while preserving the base-seed grouping. Pass `--overwrite` only when replacing previous sweep outputs intentionally.

## Visual Reports

```powershell
python -m last_nine_rl.visualize --runs runs/week1_pendulum_sweep --out reports/week1_pendulum
```

Open `reports/week1_pendulum/index.html` to inspect evaluation returns, success/collapse, replay coverage, update health, and dormant/rank diagnostics.

The aggregator reports mean return, worst-seed return, mean success rate, collapse frequency, and the fraction of seeds crossing each strict return threshold.
Aggregation uses actual seeds as the statistical unit and reports duplicate same-seed runs separately so nondeterminism audits do not silently count as independent seeds.

For proposal-level cross-seed figures:

```powershell
python -m last_nine_rl.compare --runs runs/week1_pendulum_sweep --out reports/week1_pendulum_compare
```

This generates learning curves, reliability nines, threshold ladders, final eval heatmaps, replay-vs-eval plots, and Pendulum initial-state maps when applicable.

## Verification

```powershell
python -m pytest
```

See `docs/week1_design.md` for the scientific design, success criteria, reliability criteria, telemetry schema, and recommended scale grid.
