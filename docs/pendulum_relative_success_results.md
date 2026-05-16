# Pendulum Relative Success Results

This page adds task-only, DP-relative, controller-relative, and best-known-baseline-relative success metrics for the completed 100k and 500k UTD1 Pendulum runs. The legacy `return >= -200` quantity is reported only as a diagnostic threshold, not as the main definition of success.

Definitions and equations are in `docs/pendulum_models_and_success_criteria.md`.

## Artifacts

Main inputs:

- 100k SAC grid: `reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_rollouts.csv`
- 500k SAC grid: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/pendulum_grid_rollouts.csv`
- DP grid: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv`
- Controller grid: `reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_grid.csv`

Generated outputs:

- 100k relative report: `reports/pendulum_investigation_20260509/relative_success_100k`
- 500k relative report: `reports/pendulum_investigation_20260509/relative_success_500k_utd1`
- Paired comparison: `reports/pendulum_investigation_20260509/relative_success_comparison_100k_500k`

Regret/shortfall convention:

- `mean_regret_to_*` is now nonnegative shortfall: `max(0, reference_return - SAC_return)`, averaged over training seeds.
- `mean_signed_gap_to_*` is the signed return gap: `reference_return - SAC_return`. Negative values mean SAC outperformed the reference on that initial state.
- The state-grid join was checked directly from the source CSVs: DP, controller, 100k SAC, and 500k SAC all contain the same 2501 reset-support cells with no missing or extra keys.

## Criteria

All relative metrics use returns, where larger is better. `best known` means `max(DP return, energy-controller return)` per initial state.

| Criterion | Definition |
| --- | --- |
| Task-only | near-upright fraction `>= 0.8` and max not-upright streak `<= 50` |
| Beats DP | `SAC_return >= DP_return` |
| Near DP | `SAC_return >= DP_return - 5` |
| Beats controller | `SAC_return >= controller_return` |
| Near controller | `SAC_return >= controller_return - 5` |
| Beats best known | `SAC_return >= max(DP_return, controller_return)` |
| Near best known | `SAC_return >= max(DP_return, controller_return) - 5` |

Diagnostics kept for continuity:

| Diagnostic | Definition |
| --- | --- |
| Fixed threshold | `SAC_return >= -200` |
| Strict threshold | fixed threshold and task-only |

The exact "beats DP" metric is intentionally strict. The `-5` tolerance is the preferred DP-relative diagnostic because DP is approximate and discretized.

## Controller Grid

The energy-shaping controller is deterministic, so these are cell fractions rather than seed intervals:

| Metric | Value |
| --- | ---: |
| Mean return | `-152.2637` |
| Task-only success | `0.7621` |
| Diagnostic fixed threshold | `0.6901` |
| Diagnostic strict threshold | `0.6901` |

![Controller return map](../reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_return_map.png)

![Controller task-only success map](../reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_task_success_map.png)

![Controller strict success map](../reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_strict_success_map.png)

## 100k Vs 500k

Rates below are seed means over five SAC seeds with 95% t intervals. The paired difference is `500k - 100k`, paired by seed, with an uncorrected paired t-test.

![Relative success 100k vs 500k](../reports/pendulum_investigation_20260509/relative_success_comparison_100k_500k/relative_success_100k_vs_500k_ci.png)

![Relative success paired differences](../reports/pendulum_investigation_20260509/relative_success_comparison_100k_500k/relative_success_500k_minus_100k_paired_ci.png)

| Criterion | 100k | 500k UTD1 | Difference | Paired p |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic fixed threshold | `0.6918 [0.6905, 0.6931]` | `0.6927 [0.6912, 0.6942]` | `+0.0009 [-0.0007, +0.0024]` | `0.1894` |
| Task-only | `0.8593 [0.7673, 0.9513]` | `0.8861 [0.8231, 0.9492]` | `+0.0268 [-0.0873, +0.1409]` | `0.5499` |
| Diagnostic strict threshold | `0.6692 [0.6475, 0.6908]` | `0.6748 [0.6693, 0.6803]` | `+0.0056 [-0.0192, +0.0304]` | `0.5645` |
| Beats DP | `0.0304 [0.0105, 0.0503]` | `0.0804 [0.0364, 0.1243]` | `+0.0500 [+0.0090, +0.0909]` | `0.0275` |
| Near DP | `0.8901 [0.7383, 1.0419]` | `0.9642 [0.9457, 0.9827]` | `+0.0741 [-0.0815, +0.2296]` | `0.2567` |
| Beats controller | `0.5103 [0.4292, 0.5914]` | `0.5421 [0.4701, 0.6141]` | `+0.0318 [-0.0959, +0.1595]` | `0.5270` |
| Near controller | `0.9702 [0.9053, 1.0350]` | `0.9958 [0.9941, 0.9975]` | `+0.0256 [-0.0384, +0.0896]` | `0.3290` |
| Beats best known | `0.0265 [0.0149, 0.0382]` | `0.0774 [0.0341, 0.1207]` | `+0.0509 [+0.0087, +0.0931]` | `0.0287` |
| Near best known | `0.8890 [0.7375, 1.0406]` | `0.9632 [0.9448, 0.9816]` | `+0.0742 [-0.0812, +0.2296]` | `0.2555` |

Some t intervals extend outside `[0, 1]`; they are unbounded seed-level t intervals, not clipped binomial intervals.

## 100k Maps

Overall success rates:

![100k relative success with intervals](../reports/pendulum_investigation_20260509/relative_success_100k/criterion_success_rates_ci.png)

Initial-state maps:

![100k diagnostic fixed-threshold map](../reports/pendulum_investigation_20260509/relative_success_100k/fixed_return_success_map.png)

![100k task-only success](../reports/pendulum_investigation_20260509/relative_success_100k/task_success_map.png)

![100k diagnostic strict-threshold map](../reports/pendulum_investigation_20260509/relative_success_100k/strict_success_map.png)

![100k beats DP](../reports/pendulum_investigation_20260509/relative_success_100k/beats_dp_return_map.png)

![100k near DP](../reports/pendulum_investigation_20260509/relative_success_100k/near_dp_return_eps_map.png)

![100k beats controller](../reports/pendulum_investigation_20260509/relative_success_100k/beats_controller_return_map.png)

![100k near controller](../reports/pendulum_investigation_20260509/relative_success_100k/near_controller_return_eps_map.png)

![100k beats best known](../reports/pendulum_investigation_20260509/relative_success_100k/beats_best_known_return_map.png)

![100k near best known](../reports/pendulum_investigation_20260509/relative_success_100k/near_best_known_return_eps_map.png)

Shortfall and signed-gap diagnostics:

![100k nonnegative shortfall to DP](../reports/pendulum_investigation_20260509/relative_success_100k/mean_regret_to_dp_map.png)

![100k signed gap to DP](../reports/pendulum_investigation_20260509/relative_success_100k/mean_signed_gap_to_dp_map.png)

## 500k Maps

Overall success rates:

![500k relative success with intervals](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/criterion_success_rates_ci.png)

Initial-state maps:

![500k diagnostic fixed-threshold map](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/fixed_return_success_map.png)

![500k task-only success](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/task_success_map.png)

![500k diagnostic strict-threshold map](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/strict_success_map.png)

![500k beats DP](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/beats_dp_return_map.png)

![500k near DP](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/near_dp_return_eps_map.png)

![500k beats controller](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/beats_controller_return_map.png)

![500k near controller](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/near_controller_return_eps_map.png)

![500k beats best known](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/beats_best_known_return_map.png)

![500k near best known](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/near_best_known_return_eps_map.png)

Shortfall and signed-gap diagnostics:

![500k nonnegative shortfall to DP](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/mean_regret_to_dp_map.png)

![500k signed gap to DP](../reports/pendulum_investigation_20260509/relative_success_500k_utd1/mean_signed_gap_to_dp_map.png)

## Interpretation

The fixed-threshold diagnostic barely changes from 100k to 500k, which is one reason not to treat it as the main success definition. The relative metrics reveal a more useful distinction: 500k improves exact beats-DP and beats-best-known rates, but those exact metrics remain low because DP is a near-oracle. The robust near-best-known metric is high for both conditions and higher for 500k (`0.8890` to `0.9632`), but the paired confidence interval is wide because seed 4 in the 100k run is an outlier.

The task-only metric is important because it measures the actual swing-up/stabilization behavior without using return. It is higher than the diagnostic strict threshold for both conditions, showing that some episodes look task-successful by posture/stability even when the arbitrary return threshold is not met.
