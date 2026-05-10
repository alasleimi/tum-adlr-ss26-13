# Pendulum 100k Results

This page summarizes the 100000-environment-step CleanRL SAC baseline on `Pendulum-v1`.

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

The `-200` threshold is a strict operational criterion, not an oracle feasibility claim. The energy-shaping reference controller reaches `return >= -200` on 68/100 calibration seeds, so future work should calibrate against a stronger near-oracle controller or planner.

## Post-Hoc Reliability

Source: `reports/pendulum_investigation_20260509/posthoc_100k_1000eps/posthoc_eval_summary.json`

| Metric | Value |
| --- | ---: |
| Training seeds | `5` |
| Eval episodes per seed | `1000` |
| Pooled eval episodes | `5000` |
| Mean seed mean return | `-140.6066` |
| Worst seed mean return | `-142.7925` |
| Mean seed return success | `0.7012` |
| Mean seed strict success | `0.6734` |
| Pooled return successes | `3506 / 5000` |
| Pooled strict successes | `3367 / 5000` |
| Pooled return success | `0.7012` |
| Pooled strict success | `0.6734` |
| Pooled return success Wilson 95% | `[0.6884, 0.7137]` |
| Pooled strict success Wilson 95% | `[0.6603, 0.6863]` |
| Collapse rate | `0.0` |

Threshold ladder:

| Return threshold | Fraction passing |
| --- | ---: |
| `>= -250` | `0.9686` |
| `>= -200` | `0.7012` |
| `>= -150` | `0.7012` |
| `>= -100` | `0.1230` |

Interpretation: the policy is not collapsing, but it is far from high-reliability. About 30% of evaluation starts remain failures under the `-200` return threshold.

## Learning Curves

![Pendulum 100k learning curves](../reports/week1_real_gpu_20260509/pendulum_100k_compare/learning_curves.png)

The five seeds converge to a stable partial solution. Mean return improves substantially from random initialization, but final success rates plateau well below reliable control.

## Reliability Nines

![Pendulum 100k reliability nines](../reports/week1_real_gpu_20260509/pendulum_100k_compare/reliability_nines.png)

The "nines" view makes the failure-rate problem explicit. A 30% failure rate is about 0.5 nines, not close to the project goal of pushing reliability into the tail.

## Threshold Ladder

![Pendulum 100k threshold ladder](../reports/week1_real_gpu_20260509/pendulum_100k_compare/threshold_ladder.png)

The threshold ladder shows that most episodes are above `-250`, about 70% are above `-200`/`-150`, and only about 12% clear `-100`. This means the exact threshold matters, and a single binary success definition can hide important structure.

## Fixed Eval Seeds

![Pendulum 100k final eval return heatmap](../reports/week1_real_gpu_20260509/pendulum_100k_compare/final_eval_return_heatmap.png)

![Pendulum 100k final eval strict success heatmap](../reports/week1_real_gpu_20260509/pendulum_100k_compare/final_eval_strict_success_heatmap.png)

Failures are concentrated on the same fixed evaluation seeds across training seeds. That is evidence for a structured initial-state failure mode rather than independent random noise.

## Initial-State Scatter

![Pendulum 100k final eval initial-state scatter](../reports/week1_real_gpu_20260509/pendulum_100k_compare/pendulum_initial_state_map.png)

The fixed-seed scatter suggested that downward starts are hard, especially near `theta = +/-pi` with small angular velocity.

## Exact Reset-Support Grid

Source: `reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_summary.json`

Grid:

- Theta bins: `61`
- Angular-velocity bins: `41`
- Angular-velocity range: `[-1, 1]`, matching Gymnasium Pendulum's reset support.
- Initial-condition cells: `2501`
- Training seeds per cell: `5`

Summary:

| Grid metric | Value |
| --- | ---: |
| Mean cell return success | `0.6918` |
| Mean cell strict success | `0.6692` |
| Cells where all five seeds return-succeed | `0.6905` |
| Cells where all five seeds strict-succeed | `0.6385` |
| Cells where any training seed strict-succeeds | `0.6817` |

Return success map:

![Pendulum reset-support return success map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/return_success_rate_map.png)

Strict success map:

![Pendulum reset-support strict success map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/strict_success_rate_map.png)

Mean return map:

![Pendulum reset-support mean return map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/mean_return_map.png)

Near-upright fraction map:

![Pendulum reset-support near-upright fraction map](../reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/near_upright_fraction_map.png)

Important region summaries:

| Region | Cells | Mean return success | Mean strict success |
| --- | ---: | ---: | ---: |
| All reset support | `2501` | `0.6918` | `0.6692` |
| `|theta| >= 150 deg` | `451` | `0.0` | `0.0` |
| `|theta| >= 150 deg` and `|theta_dot| <= 0.5` | `231` | `0.0` | `0.0` |
| `60 deg <= |theta| <= 120 deg` | `820` | `1.0` | `1.0` |

Interpretation: the SAC policy reliably solves mid-angle starts and reliably fails near downward starts under the reset distribution. The reliability gap is therefore not simply average instability; it is a localized hard-start failure.

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

## 500k Follow-Up So Far

Partial 500k UTD1 result, completed seeds 0-2:

- Post-hoc return success: `0.7023`
- Post-hoc strict success: `0.6773`

This is not materially better than the 100k result. The longer sweep is still running locally and should be committed separately after completion.

## Conclusion

The 100k CleanRL SAC baseline learns a strong average-return Pendulum policy, but it does not achieve high-reliability success. The main scientific finding is the mismatch between average performance and tail reliability: the policy solves many starts while consistently failing the near-downward low-velocity region.

The next necessary step is threshold feasibility calibration. A near-oracle Pendulum planner or dynamic-programming calibration should estimate the best achievable return for each initial-condition cell, so success can be evaluated relative to feasible optimal performance rather than only a fixed `-200` threshold.
