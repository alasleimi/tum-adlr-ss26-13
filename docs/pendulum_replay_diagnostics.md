# Pendulum Replay And Representation Diagnostics

Generated report: `reports/pendulum_investigation_20260509/replay_diagnostics_comparison/index.html`

Command:

```powershell
python -m last_nine_rl.replay_diagnostics_report `
  --condition 100k_utd1=runs/week1_real_gpu_20260509/pendulum_100k `
  --condition 500k_utd1=runs/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k `
  --condition 250k_utd2_partial=runs/pendulum_investigation_20260509/pendulum_250k_utd2_buffer500k `
  --out reports/pendulum_investigation_20260509/replay_diagnostics_comparison
```

The terminal summaries and plots use complete runs when available. The `250k_utd2_partial`
condition has only seeds 0 and 1 complete; seed 2 stopped near 76000 steps and is kept in the
CSV for failure accounting, not as a terminal result.

## Important Limitation

The completed 100k and 500k runs did not save `replay_final.npz`; their configs have
`telemetry.save_replay=false`. Therefore this pass checks the logged replay summaries and
sample-count telemetry, not a reconstructed state-conditioned replay buffer. Future diagnostic
or SimbaV2 scale runs should pass `--save-replay` when storage is acceptable.

## Main Findings

| Condition | Complete seeds | Built-in eval strict success | Replay near-upright transitions | Q1 layer1 dormant | Q2 layer1 dormant | Actor update ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k UTD1 | 5 | 0.588, 95% CI [0.555, 0.621] | 0.7948, 95% CI [0.7875, 0.8021] | 0.460, 95% CI [0.403, 0.517] | 0.480, 95% CI [0.455, 0.505] | 2.28e-4, 95% CI [2.09e-4, 2.48e-4] |
| 500k UTD1 | 5 | 0.550, 95% CI [0.550, 0.550] | 0.8471, 95% CI [0.8453, 0.8490] | 0.773, 95% CI [0.722, 0.823] | 0.780, 95% CI [0.756, 0.805] | 9.51e-5, 95% CI [8.93e-5, 1.01e-4] |
| 250k UTD2 partial | 2 | 0.550, 95% CI [0.550, 0.550] | 0.8378, 95% CI [0.8004, 0.8753] | 0.707, 95% CI [0.211, 1.000] | 0.787, 95% CI [0.613, 0.961] | 9.01e-5, 95% CI [8.07e-5, 9.96e-5] |

The confidence intervals are seed-level t intervals clipped to the logical `[0, 1]` range for
fraction metrics. They are descriptive because five seeds are enough for debugging, not for a
final reliability claim.

Additional replay checks:

- Replay reward mean improves from -1.111 at 100k to -0.789 at 500k, so the buffer is not merely full of bad transitions.
- Replay action saturation stays low, about 7-8%, so the current failure is not obviously an always-clipped-action artifact.
- Mean sample count is mechanically consistent with the update budget: 243.2 at 100k UTD1, 253.44 at 500k UTD1 after the larger buffer fills, and 501.76 for the completed 250k UTD2 seeds.
- The 500k and UTD2 actor update norm ratios are less than half the 100k value, while critic dormant fractions rise sharply.

## Current Hypothesis

The Pendulum reliability gap is probably not explained by a lack of near-upright replay coverage.
The buffer contains more near-upright transitions at 500k than at 100k, and replay rewards improve,
but the built-in fixed-eval diagnostic does not improve.

The stronger concern is representation and plasticity. Longer training and higher update density
coincide with high dormant fractions in critic hidden layers, persistently low effective-rank
fractions, lower entropy temperature, and much smaller actor parameter updates. That pattern is
consistent with the project motivation for testing SimbaV2-style representation and value-learning
changes, but it is not proof that those changes will fix reliability.

## Week 1 Consequences

- Keep `return >= -200` as a diagnostic only. The more meaningful Pendulum criteria remain
  task-state success and reference-relative success against DP/controller baselines.
- Complete or explicitly close the `250k_utd2_partial` condition before using it in any main figure.
- Enable `--save-replay` on future diagnostic runs so state-conditioned replay and sampling maps can
  be computed directly.
- If Week 1 needs another baseline artifact, the highest-value addition is not another 100k-style
  run; it is either completing UTD2 cleanly or adding raw-replay snapshots to the next intervention run.
