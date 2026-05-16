# Pendulum Dynamic-Programming Calibration

This page records the finite-horizon dynamic-programming calibration added after the initial 100k SAC result. It answers a specific question: whether the legacy fixed Pendulum threshold `return >= -200` is feasible from every reset-support initial state.

The exact dynamics, DP approximation, energy-shaping controller equations, and relative-success criteria are specified in `docs/pendulum_models_and_success_criteria.md`.

## Method

Source: `src/last_nine_rl/pendulum_dp.py`

Primary run:

```powershell
python -m last_nine_rl.pendulum_dp `
  --out reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81 `
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

The DP model uses Gymnasium `Pendulum-v1` dynamics, a 200-step finite horizon, torque grid `[-2, 2]`, full angular-velocity grid `[-8, 8]`, bilinear interpolation in state space, and greedy rollout of the resulting value function on the same 61 by 41 reset-support grid used for the SAC checkpoint map.

The 100k SAC checkpoints are not needed to solve DP. They are used in the joined comparison CSV so each SAC initial-state cell can be compared to the DP-calibrated feasible value for the same cell.

## Main Result

Primary output: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_summary.json`

| Metric | DP calibration | SAC 100k checkpoint grid |
| --- | ---: | ---: |
| Reset-support cells | `2501` | `2501` |
| Fixed-threshold cell fraction | `0.6941` | `0.6918` |
| Strict-threshold cell fraction | `0.6933` | `0.6692` |
| Fixed-threshold-feasible cells | `1736 / 2501` | n/a |
| Strict-threshold-feasible cells | `1734 / 2501` | n/a |
| SAC failure rate among DP fixed-threshold-feasible cells | n/a | `0.0033` |
| SAC strict-threshold failure rate among DP strict-threshold-feasible cells | n/a | `0.0348` |
| Mean SAC nonnegative shortfall to DP policy return | n/a | `3.11` return points |
| Mean SAC signed gap to DP policy return | n/a | `3.03` return points |

The fixed `-200` threshold is not feasible everywhere. Under the DP calibration, the same fraction of reset-support cells is feasible as the SAC policy passes by fixed threshold. This is evidence that `-200` should be treated as a diagnostic threshold, not as the main Pendulum success criterion.

## Initial-State Evidence

DP policy return:

![DP policy return map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/dp_policy_return_map.png)

DP fixed-threshold-feasible cells:

![DP return success map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/dp_return_success_map.png)

SAC nonnegative shortfall to DP:

![SAC regret to DP map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_regret_to_dp_map.png)

SAC signed gap to DP, where negative values mean SAC outperformed the approximate DP rollout:

![SAC signed gap to DP map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_signed_gap_to_dp_map.png)

SAC failure rate on DP fixed-threshold-feasible starts:

![SAC failure on DP feasible map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_failure_on_dp_feasible_map.png)

SAC strict-threshold failure rate on DP strict-threshold-feasible starts:

![SAC strict failure on DP strict feasible map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_strict_failure_on_dp_strict_feasible_map.png)

## Near-Down Region

The previous SAC map showed complete failure near the downward low-velocity region. DP changes the interpretation:

| Region | Cells | DP fixed-threshold-feasible cells | DP mean return | SAC fixed-threshold rate |
| --- | ---: | ---: | ---: | ---: |
| `|theta| >= 150 deg` | `451` | `0` | `-240.94` | `0.0000` |
| `|theta| >= 150 deg` and `|theta_dot| <= 0.5` | `231` | `0` | `-241.92` | `0.0000` |

This means the near-down failure is not good evidence that SAC missed an easy solution under the `-200` threshold. It is evidence that the fixed threshold itself is too strict for that region.

## Sensitivity Check

A finer grid was run as a discretization check:

- Output: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_361x241x101_check`
- State grid: 361 theta bins by 241 velocity bins
- Action grid: 101 torque bins

Key comparison:

| Metric | Primary grid | Finer check |
| --- | ---: | ---: |
| DP fixed-threshold-feasible cells | `1736` | `1736` |
| DP strict-threshold-feasible cells | `1734` | `1734` |
| Near-down fixed-threshold-feasible cells | `0` | `0` |
| Mean return, `|theta| >= 150 deg` | `-240.94` | `-239.51` |
| Mean return, `|theta| >= 150 deg` and `|theta_dot| <= 0.5` | `-241.92` | `-240.77` |

The exact return values move by about 1 to 1.5 points in the hard region, but the feasibility classification is unchanged.

## Caveats

This is an approximate finite-horizon planner, not a mathematical proof of optimality. The conclusion is strong enough for Week 1 calibration because the feasible mask is stable under a finer discretization check and because the near-down region is far below `-200` on average.

For later project phases, success should be reported as task-state success plus DP/controller-relative metrics. The original fixed-threshold quantity should remain a reproducibility diagnostic and threshold-ladder point.
