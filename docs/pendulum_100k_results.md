# Pendulum 100k Results

This page summarizes the 100000-environment-step CleanRL SAC baseline on Gymnasium `Pendulum-v1`.

## Status

This result is complete for the Week 1 baseline:

- 5/5 training seeds completed.
- All five final checkpoints were post-hoc evaluated on 1000 fixed eval seeds.
- Exact initial-condition maps were generated from the saved checkpoints.
- Approximate finite-horizon dynamic-programming calibration was generated for the same reset-support grid.
- Large artifacts and plots are committed through Git LFS.

## Experimental Setup

Runs:

- Run root: `runs/week1_real_gpu_20260509/pendulum_100k`
- Training seeds: `0, 1, 2, 3, 4`
- Environment: Gymnasium `Pendulum-v1`
- Algorithm: copied CleanRL SAC
- Environment steps per seed: `100000`
- Replay buffer size: `100000`
- Updates per environment step: `1`
- Device: CUDA

Core SAC parameters:

| Parameter | Value |
| --- | ---: |
| Actor/critic hidden layers | `256, 256` |
| Batch size | `256` |
| Learning starts | `5000` |
| Discount `gamma` | `0.99` |
| Target smoothing `tau` | `0.005` |
| Actor learning rate | `3e-4` |
| Critic learning rate | `1e-3` |
| Alpha learning rate | `1e-3` |
| Policy update frequency | `2` |
| Target network update frequency | `1` |

Evaluation:

- During training: 50 fixed eval episodes at steps `0`, `25000`, `50000`, `75000`, and `100000`.
- Post-hoc: 1000 deterministic fixed eval seeds per training seed, seed base `200000`.
- Exact reset-support grid: 61 theta bins by 41 angular-velocity bins over `theta_dot in [-1, 1]`.

## Success Definitions

Pendulum return is the 200-step sum of Gymnasium rewards. Larger is better and the best possible return is near `0`.

Operational success metrics:

- Return success: return `>= -200`.
- Strict success: return success, near-upright fraction `>= 0.8`, and max not-near-upright streak `<= 50`.
- Threshold ladder: `-250`, `-200`, `-150`, `-100`.

The `-200` threshold is a strict operational criterion, not an oracle feasibility claim. The dynamic-programming calibration below shows that this threshold is not feasible from every reset-support initial state.

## Statistical Reporting

Two uncertainty views are reported:

- **Seed-level intervals**: mean across training seeds plus a 95% t interval. This is the primary uncertainty estimate for SAC because training seed is the experimental unit.
- **Pooled episode Wilson intervals**: binomial Wilson interval over all evaluated episodes. These are useful for operational failure rates, but the same eval seeds are reused across training seeds, so they should not replace seed-level uncertainty.

Comparisons involving the energy-shaping controller are marked exploratory because it is a single deterministic controller, not a distribution over training seeds.

Review note: plots that were generated before this review pass, such as learning curves and heatmaps, are treated as diagnostic visualizations. Formal uncertainty is reported in the interval plots and tables in this document. Heatmaps do not draw cellwise confidence intervals; the region table below reports Wilson intervals for the main initial-state regions.

## Post-Hoc Reliability

Source files:

- `reports/pendulum_investigation_20260509/posthoc_100k_1000eps/posthoc_eval_summary.json`
- `reports/pendulum_investigation_20260509/analysis_stats.json`

| Metric | Value |
| --- | ---: |
| Training seeds | `5` |
| Eval episodes per seed | `1000` |
| Pooled eval episodes | `5000` |
| Mean seed mean return | `-140.6066 +/- 1.5295` |
| Mean seed return success | `0.7012 +/- 0.0027` |
| Mean seed strict success | `0.6734 +/- 0.0247` |
| Pooled return successes | `3506 / 5000` |
| Pooled strict successes | `3367 / 5000` |
| Pooled return success Wilson 95% | `[0.6884, 0.7137]` |
| Pooled strict success Wilson 95% | `[0.6603, 0.6863]` |
| Collapse rate | `0.0` |

Interpretation: the policy is not collapsing, but it is far from high-reliability. About 30% of evaluation starts remain failures under the `-200` return threshold.

## Energy-Shaping Baseline

Controller:

- Source: `src/last_nine_rl/reference.py`
- Type: energy-shaping swing-up plus local PD near upright
- Eval seeds: 1000 fixed eval seeds, seed base `200000`, matching the SAC post-hoc eval distribution
- Output: `reports/pendulum_investigation_20260509/pendulum_reference_1000_seed200000.json`

Energy-shaping formula:

```text
E = 0.5 * theta_dot^2 + a * cos(theta)
E* = a
u = -k * (E - E*) * theta_dot
```

Near upright it switches to:

```text
u = -Kp * theta - Kd * theta_dot
```

Matched 1000-seed result:

| Metric | Energy shaping |
| --- | ---: |
| Mean return | `-151.0905 +/- 5.3816` |
| Return success | `701 / 1000 = 0.7010`, Wilson `[0.6719, 0.7286]` |
| Strict success | `701 / 1000 = 0.7010`, Wilson `[0.6719, 0.7286]` |
| Collapse rate | `0.0` |

SAC 100k and energy shaping have nearly identical return success on this matched eval distribution. The energy controller has slightly higher strict success than the average SAC checkpoint, but an exploratory pooled two-proportion test gives `p = 0.0882` for strict success. Because SAC uses repeated eval seeds across five trained policies and energy shaping is one deterministic controller, this test is descriptive rather than definitive.

![Pendulum success comparison with intervals](../reports/pendulum_investigation_20260509/success_comparison_100k_500k_energy.png)

## Threshold Ladder

The threshold ladder is important because the conclusion depends on how strict the success threshold is.

![Pendulum threshold ladder with CI](../reports/pendulum_investigation_20260509/threshold_ladder_100k_energy_ci.png)

| Return threshold | SAC 100k pooled | Energy shaping |
| --- | ---: | ---: |
| `>= -250` | `0.9686`, Wilson `[0.9634, 0.9731]` | `0.8710`, Wilson `[0.8488, 0.8904]` |
| `>= -200` | `0.7012`, Wilson `[0.6884, 0.7137]` | `0.7010`, Wilson `[0.6719, 0.7286]` |
| `>= -150` | `0.7012`, Wilson `[0.6884, 0.7137]` | `0.7010`, Wilson `[0.6719, 0.7286]` |
| `>= -100` | `0.1230`, Wilson `[0.1142, 0.1324]` | `0.1100`, Wilson `[0.0921, 0.1309]` |

SAC is much better than the hand-designed controller at avoiding the very bad tail below `-250`, but both methods have essentially the same pass rate at `-200` and `-150`.

## Learning Curves

![Pendulum 100k learning curves](../reports/week1_real_gpu_20260509/pendulum_100k_compare/learning_curves.png)

The five seeds converge to a stable partial solution. This diagnostic plot shows individual seed traces and cross-seed spread, not a formal confidence band. The final-checkpoint seed-level intervals are reported in the post-hoc reliability table above.

## Reliability Nines

![Pendulum 100k reliability nines](../reports/week1_real_gpu_20260509/pendulum_100k_compare/reliability_nines.png)

The "nines" view makes the failure-rate problem explicit. A 30% failure rate is about 0.5 nines, not close to the project goal of pushing reliability into the tail. This is a diagnostic learning-curve plot; the post-hoc Wilson and seed-level intervals above are the inferential final-checkpoint estimates.

## Fixed Eval Seeds

![Pendulum 100k final eval return heatmap](../reports/week1_real_gpu_20260509/pendulum_100k_compare/final_eval_return_heatmap.png)

![Pendulum 100k final eval strict success heatmap](../reports/week1_real_gpu_20260509/pendulum_100k_compare/final_eval_strict_success_heatmap.png)

Failures are concentrated on the same fixed evaluation seeds across training seeds. These heatmaps show raw final-eval outcomes, without separate error bars per cell. The repeated failure pattern is diagnostic evidence for a structured initial-state failure mode rather than independent random noise.

## Initial-State Maps

The fixed-seed scatter first revealed the hard-start pattern:

![Pendulum 100k final eval initial-state scatter](../reports/week1_real_gpu_20260509/pendulum_100k_compare/pendulum_initial_state_map.png)

The scatter is diagnostic and uses the fixed eval seeds. The exact reset-support grid below evaluates the saved policies from exact initial states, not only from sampled reset seeds, and the region table supplies uncertainty for the main regions.

Source: `reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_summary.json`

Grid:

- Theta bins: `61`
- Angular-velocity bins: `41`
- Angular-velocity range: `[-1, 1]`, matching Gymnasium Pendulum's reset support.
- Initial-condition cells: `2501`
- Training seeds per cell: `5`

Return success map:

![Pendulum reset-support return success map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/return_success_rate_map.png)

Strict success map:

![Pendulum reset-support strict success map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/strict_success_rate_map.png)

Mean return map:

![Pendulum reset-support mean return map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/mean_return_map.png)

Near-upright fraction map:

![Pendulum reset-support near-upright fraction map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/near_upright_fraction_map.png)

Region summaries treat each cell-by-training-seed rollout as a trial and report Wilson intervals. Because adjacent initial-condition cells are correlated, use these intervals as descriptive diagnostics, not as independent-cell hypothesis tests.

![Pendulum reset-support region success with intervals](../reports/pendulum_investigation_20260509/reset_support_region_success_ci.png)

| Region | Cells | Seed trials | Return success Wilson 95% | Strict success Wilson 95% |
| --- | ---: | ---: | ---: | ---: |
| All reset support | `2501` | `12505` | `0.6918 [0.6837, 0.6998]` | `0.6692 [0.6609, 0.6774]` |
| `|theta| >= 150 deg` | `451` | `2255` | `0.0000 [0.0000, 0.0017]` | `0.0000 [0.0000, 0.0017]` |
| `|theta| >= 150 deg` and `|theta_dot| <= 0.5` | `231` | `1155` | `0.0000 [0.0000, 0.0033]` | `0.0000 [0.0000, 0.0033]` |
| `60 deg <= |theta| <= 120 deg` | `820` | `4100` | `1.0000 [0.9991, 1.0000]` | `1.0000 [0.9991, 1.0000]` |

Interpretation: the SAC policy reliably solves mid-angle starts and reliably fails near downward starts under the reset distribution. The reliability gap is therefore not simply average instability; it is a localized hard-start failure.

## Dynamic-Programming Calibration

Detailed writeup: `docs/pendulum_dp_calibration.md`

Primary output: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_summary.json`

Method:

- Finite-horizon DP for Gymnasium `Pendulum-v1`, horizon `200`.
- State grid: `241` theta bins by `161` angular-velocity bins over `theta_dot in [-8, 8]`.
- Action grid: `81` torque bins over `[-2, 2]`.
- Value interpolation: bilinear in theta and angular velocity.
- Evaluation grid: same 61 by 41 reset-support initial-state grid used for the SAC checkpoint map.
- Sensitivity check: finer `361 x 241 x 101` DP grid.

DP policy return map:

![DP policy return map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/dp_policy_return_map.png)

DP return-feasible map under `return >= -200`:

![DP return success map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/dp_return_success_map.png)

SAC regret to DP:

![SAC regret to DP map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_regret_to_dp_map.png)

SAC failure rate on DP-feasible starts:

![SAC failure on DP feasible map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_failure_on_dp_feasible_map.png)

SAC strict failure rate on DP-strict-feasible starts:

![SAC strict failure on DP strict feasible map](../reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/sac_strict_failure_on_dp_strict_feasible_map.png)

| Metric | DP calibration | SAC checkpoint grid |
| --- | ---: | ---: |
| Return-success cell fraction | `0.6941` | `0.6918` |
| Strict-success cell fraction | `0.6933` | `0.6692` |
| DP return-feasible cells | `1736 / 2501` | n/a |
| DP strict-feasible cells | `1734 / 2501` | n/a |
| SAC failure rate among DP return-feasible cells | n/a | `0.0033` |
| SAC strict-failure rate among DP strict-feasible cells | n/a | `0.0348` |
| Mean SAC regret to DP policy return | n/a | `3.03` return points |

The DP calibration changes the interpretation of the hard-start map. Near downward starts are not solved by SAC, but the DP planner also does not classify them as feasible under `return >= -200`:

| Region | Cells | DP return-feasible cells | DP mean return | SAC return success |
| --- | ---: | ---: | ---: | ---: |
| `|theta| >= 150 deg` | `451` | `0` | `-240.94` | `0.0000` |
| `|theta| >= 150 deg` and `|theta_dot| <= 0.5` | `231` | `0` | `-241.92` | `0.0000` |

Sensitivity check:

| Metric | Primary DP grid | Finer DP check |
| --- | ---: | ---: |
| DP return-feasible cells | `1736` | `1736` |
| DP strict-feasible cells | `1734` | `1734` |
| Near-down return-feasible cells | `0` | `0` |
| Mean return, `|theta| >= 150 deg` | `-240.94` | `-239.51` |
| Mean return, `|theta| >= 150 deg` and `|theta_dot| <= 0.5` | `-241.92` | `-240.77` |

Interpretation: the fixed `-200` threshold overstates SAC failure in the near-downward low-velocity region. For Week 1, the stronger scientific claim is now that 100k SAC nearly matches the DP return-feasible mask, while still leaving a small strict-stabilization gap on DP-strict-feasible cells.

## Full-State Grid Caveat

The full Pendulum grid over angular velocity `[-8, 8]` looks much better:

| Grid | Mean return success | Mean strict success |
| --- | ---: | ---: |
| Full state range, `theta_dot in [-8, 8]` | `0.9226` | `0.9056` |
| Reset support, `theta_dot in [-1, 1]` | `0.6918` | `0.6692` |

High angular velocity often makes swing-up easier because the pendulum already has useful momentum. For claims about Gymnasium evaluation reliability, the reset-support grid is the relevant one.

## Replay And Optimization Context

![Pendulum 100k replay vs eval](../reports/week1_real_gpu_20260509/pendulum_100k_compare/replay_vs_eval.png)

The infrastructure logs replay coverage, action saturation, dormant units, effective ranks, parameter norms, gradient norms, and actual optimizer update norms. The current result does not point to a missing terminal-state or CleanRL implementation bug; see `docs/cleanrl_audit.md`.

## Relation To 500k Partial Run

The partial 500k UTD1 result is tracked separately in `docs/pendulum_500k_partial_results.md`. It is incomplete and should not be quoted as the final 500k condition. For the completed seeds 0-2, paired comparisons against their 100k counterparts show no practical improvement:

| Metric | 500k minus 100k, seeds 0-2 | Paired t-test |
| --- | ---: | ---: |
| Mean return | `+0.3874`, 95% CI `[-1.0583, +1.8331]` | paired t-test, uncorrected `p = 0.3681` |
| Return success | `-0.0003`, 95% CI `[-0.0018, +0.0011]` | paired t-test, uncorrected `p = 0.4226` |
| Strict success | `-0.0060`, 95% CI `[-0.0103, -0.0017]` | paired t-test, uncorrected `p = 0.0267` |

The strict-success decrease is tiny and comes from only three completed seeds, so it should be treated as a warning sign rather than a settled conclusion.

## Conclusion

The 100k CleanRL SAC baseline learns a strong Pendulum policy under the reset distribution. The dynamic-programming calibration shows that the largest fixed-threshold failure region is not feasible under the `return >= -200` criterion, so the original hard-start map should not be interpreted as a pure RL failure.

The next metric improvement is to report reliability relative to DP feasibility or DP regret, not only the fixed `-200` threshold. Under that view, the main remaining 100k Pendulum gap is the strict-stabilization failure rate of about `3.5%` on DP-strict-feasible cells.
