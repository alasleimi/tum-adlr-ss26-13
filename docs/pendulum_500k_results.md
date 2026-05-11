# Pendulum 500k UTD1 Results

Status: **complete** for the 500000-environment-step UTD1 condition.

This page summarizes the completed CleanRL SAC Pendulum run with a larger interaction budget than the 100k baseline.

## Experimental Setup

Condition:

- Run root: `runs/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k`
- Environment: Gymnasium `Pendulum-v1`
- Algorithm: copied CleanRL SAC
- Training seeds: `0, 1, 2, 3, 4`
- Environment steps per seed: `500000`
- Replay buffer size: `500000`
- Updates per environment step: `1`
- Device: CUDA

Parameter changes relative to the 100k baseline:

| Parameter | 100k baseline | 500k UTD1 |
| --- | ---: | ---: |
| Environment steps | `100000` | `500000` |
| Replay buffer size | `100000` | `500000` |
| Updates per environment step | `1` | `1` |
| Batch size | `256` | `256` |
| Learning starts | `5000` | `5000` |
| Actor learning rate | `3e-4` | `3e-4` |
| Critic learning rate | `1e-3` | `1e-3` |
| During-training eval interval | `25000` | `100000` |
| During-training eval episodes | `50` | `20` |

## Completed Artifacts

- Aggregate: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/aggregate.json`
- Post-hoc eval: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/posthoc_1000eps/posthoc_eval_summary.json`
- Initial-state grid: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/pendulum_grid_summary.json`
- Cross-seed plots: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/compare/index.html`
- Relative success report: `reports/pendulum_investigation_20260509/relative_success_500k_utd1/relative_summary.json`

## Post-Hoc Reliability

Post-hoc eval uses 1000 fixed deterministic eval seeds per training seed, seed base `200000`.

| Metric | Value |
| --- | ---: |
| Training seeds | `5` |
| Eval episodes per seed | `1000` |
| Pooled eval episodes | `5000` |
| Mean seed mean return | `-139.3497` |
| Mean seed diagnostic fixed threshold | `0.7028` |
| Mean seed diagnostic strict threshold | `0.6802` |
| Pooled fixed-threshold passes | `3514 / 5000` |
| Pooled strict-threshold passes | `3401 / 5000` |
| Pooled fixed-threshold Wilson 95% | `[0.6900, 0.7153]` |
| Pooled strict-threshold Wilson 95% | `[0.6671, 0.6930]` |
| Collapse rate | `0.0` |

Interpretation: 500k UTD1 does not materially improve the legacy fixed-threshold diagnostic over the 100k baseline. It improves the diagnostic strict-threshold rate slightly, but the main success story should use task-only and reference-relative metrics.

## Initial-State Grid

The reset-support grid evaluates the five saved checkpoints over 61 theta bins by 41 angular-velocity bins over `theta_dot in [-1, 1]`.

| Metric | 500k UTD1 |
| --- | ---: |
| Initial-condition cells | `2501` |
| Seed trials | `12505` |
| Mean fixed-threshold cell rate | `0.6927` |
| Mean strict-threshold cell rate | `0.6748` |
| Cells where all five seeds pass fixed threshold | `0.6905` |
| Cells where all five seeds pass strict threshold | `0.6657` |
| Cells where any seed passes strict threshold | `0.6805` |

Diagnostic fixed-threshold map:

![500k reset-support return success map](../reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/return_success_rate_map.png)

Diagnostic strict-threshold map:

![500k reset-support strict success map](../reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/strict_success_rate_map.png)

Mean return map:

![500k reset-support mean return map](../reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/mean_return_map.png)

Near-upright fraction map:

![500k reset-support near-upright map](../reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/near_upright_fraction_map.png)

## Relative And Task Success

Detailed definitions and equations are in `docs/pendulum_models_and_success_criteria.md`. Combined 100k/500k relative results are in `docs/pendulum_relative_success_results.md`.

Rates below are seed means over the 61 x 41 grid, with 95% t intervals.

| Criterion | 500k UTD1 |
| --- | ---: |
| Diagnostic fixed threshold, `return >= -200` | `0.6927 [0.6912, 0.6942]` |
| Task-only stability | `0.8861 [0.8231, 0.9492]` |
| Diagnostic strict threshold | `0.6748 [0.6693, 0.6803]` |
| Beats DP | `0.0804 [0.0364, 0.1243]` |
| Near DP, within 5 return points | `0.9642 [0.9457, 0.9827]` |
| Beats controller | `0.5421 [0.4701, 0.6141]` |
| Near controller, within 5 return points | `0.9958 [0.9941, 0.9975]` |
| Beats best known, `max(DP, controller)` | `0.0774 [0.0341, 0.1207]` |
| Near best known, within 5 return points | `0.9632 [0.9448, 0.9816]` |

![500k relative success with intervals](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/criterion_success_rates_ci.png)

Representative initial-state maps:

![500k task-only success map](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/task_success_map.png)

![500k near best-known success map](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/near_best_known_return_eps_map.png)

All relative maps are stored under `reports/pendulum_investigation_20260509/relative_success_500k_utd1`.

## Paired Comparison To 100k

The most relevant comparison is paired by training seed. The plot and table below compare success criteria over the exact reset-support grid.

![Relative success 100k vs 500k](../reports/pendulum_investigation_20260509/relative_success_comparison_100k_500k/relative_success_100k_vs_500k_ci.png)

![Relative success paired differences](../reports/pendulum_investigation_20260509/relative_success_comparison_100k_500k/relative_success_500k_minus_100k_paired_ci.png)

| Criterion | 500k minus 100k | Paired t-test |
| --- | ---: | ---: |
| Diagnostic fixed threshold | `+0.0009`, 95% CI `[-0.0007, +0.0024]` | uncorrected `p = 0.1894` |
| Task-only | `+0.0268`, 95% CI `[-0.0873, +0.1409]` | uncorrected `p = 0.5499` |
| Diagnostic strict threshold | `+0.0056`, 95% CI `[-0.0192, +0.0304]` | uncorrected `p = 0.5645` |
| Beats DP | `+0.0500`, 95% CI `[+0.0090, +0.0909]` | uncorrected `p = 0.0275` |
| Near DP | `+0.0741`, 95% CI `[-0.0815, +0.2296]` | uncorrected `p = 0.2567` |
| Beats controller | `+0.0318`, 95% CI `[-0.0959, +0.1595]` | uncorrected `p = 0.5270` |
| Near controller | `+0.0256`, 95% CI `[-0.0384, +0.0896]` | uncorrected `p = 0.3290` |
| Beats best known | `+0.0509`, 95% CI `[+0.0087, +0.0931]` | uncorrected `p = 0.0287` |
| Near best known | `+0.0742`, 95% CI `[-0.0812, +0.2296]` | uncorrected `p = 0.2555` |

The exact beats-DP and beats-best-known improvements are statistically visible in this five-seed paired comparison, but these are not corrected for multiple comparisons. The robust near-DP and near-best-known rates are high for both conditions, with wide seed intervals.

## Conclusion

500k UTD1 does not change the fixed `return >= -200` diagnostic much. It does improve the learned policy's closeness to the DP/controller references, especially exact beats-DP and beats-best-known rates. The fixed-threshold diagnostic and the relative-success story therefore differ:

- Fixed threshold diagnostic: 100k and 500k are almost identical.
- Task-only stability: 500k is slightly higher, but the seed interval is wide.
- Relative to DP/controller: 500k is closer to the best-known reference on many cells, but exact superiority over DP remains rare because DP is a near-oracle.
