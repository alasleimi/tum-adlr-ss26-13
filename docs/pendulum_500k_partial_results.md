# Pendulum 500k Results

Status: **incomplete**.

This page records the partial 500000-environment-step Pendulum investigation. It is not the final 500k result because the full condition has not completed.

## Current Completion State

Condition:

- Run root: `runs/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k`
- Environment: Gymnasium `Pendulum-v1`
- Algorithm: copied CleanRL SAC
- Environment steps per seed: `500000`
- Replay buffer size: `500000`
- Updates per environment step: `1`
- Device: CUDA

Completed:

- `seed0`
- `seed1`
- `seed2`

Not complete at the time this page was written:

- `seed3`: running locally, last checked around step `302800 / 500000`
- `seed4`: not completed

Therefore all numbers below are **partial** and are reported only to decide whether the ongoing run is showing evidence of improvement.

Review note: this page intentionally avoids final 500k claims. All reported comparisons include 95% intervals and uncorrected paired tests where applicable. The comparison plot includes error bars; no 500k initial-state map is shown because the condition is not complete.

## Parameter Changes Relative To 100k

The 500k UTD1 condition keeps the CleanRL SAC algorithmic defaults and changes the interaction budget and buffer size:

| Parameter | 100k baseline | 500k UTD1 condition |
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

## Partial Post-Hoc Reliability

Source files:

- `reports/pendulum_investigation_20260509/pendulum_500k_utd1_partial_seed0_2/posthoc_1000eps/posthoc_eval_summary.json`
- `reports/pendulum_investigation_20260509/analysis_stats.json`

Post-hoc eval:

- Completed training seeds: `0, 1, 2`
- Eval episodes per completed seed: `1000`
- Pooled eval episodes: `3000`
- Seed base: `200000`

| Metric | Partial 500k value |
| --- | ---: |
| Mean seed mean return | `-139.6041 +/- 1.7413` |
| Mean seed return success | `0.7023 +/- 0.0029` |
| Mean seed strict success | `0.6773 +/- 0.0117` |
| Pooled return successes | `2107 / 3000` |
| Pooled strict successes | `2032 / 3000` |
| Pooled return success Wilson 95% | `[0.6857, 0.7184]` |
| Pooled strict success Wilson 95% | `[0.6604, 0.6938]` |
| Collapse rate | `0.0` |

![Pendulum success comparison with intervals](../reports/pendulum_investigation_20260509/success_comparison_100k_500k_energy.png)

The error bars in this plot are:

- SAC 100k and SAC 500k: seed-level 95% t intervals.
- Energy shaping: Wilson interval over 1000 fixed eval episodes.
- The 500k bar is explicitly partial and uses only seeds 0-2.

## Paired Comparison To 100k

The fair early comparison is paired by training seed for the completed seeds `0, 1, 2`.

| Metric | 100k seeds 0-2 | 500k seeds 0-2 | 500k minus 100k | Paired t-test |
| --- | ---: | ---: | ---: | ---: |
| Mean return | `-139.9915 +/- 0.3524` | `-139.6041 +/- 1.7413` | `+0.3874`, 95% CI `[-1.0583, +1.8331]` | paired t-test, uncorrected `p = 0.3681` |
| Return success | `0.7027 +/- 0.0014` | `0.7023 +/- 0.0029` | `-0.0003`, 95% CI `[-0.0018, +0.0011]` | paired t-test, uncorrected `p = 0.4226` |
| Strict success | `0.6833 +/- 0.0160` | `0.6773 +/- 0.0117` | `-0.0060`, 95% CI `[-0.0103, -0.0017]` | paired t-test, uncorrected `p = 0.0267` |

![Pendulum partial 500k paired differences with intervals](../reports/pendulum_investigation_20260509/paired_500k_minus_100k_seed0_2_ci.png)

Interpretation:

- There is no evidence that 500k UTD1 improves return success over 100k on completed seeds.
- The strict-success difference is slightly negative and statistically significant in this very small paired sample before any multiple-comparison correction, but the effect size is only 0.6 percentage points and the condition is incomplete. Treat it as a caution, not as a final conclusion.
- A final 500k claim requires completed seeds 0-4 plus the same post-hoc 1000-episode eval and an exact reset-support grid.

## Missing Before Final 500k Result

Required before quoting this as a completed 500k condition:

- Complete `seed3` and `seed4`.
- Generate the condition aggregate JSON.
- Generate comparison plots.
- Run post-hoc 1000-episode eval on all completed seeds.
- Generate the reset-support initial-condition grid.
- Re-run the uncertainty and paired-comparison analysis.
- Commit the final artifacts.

## Current Takeaway

The partial 500k UTD1 result does not change the scientific conclusion from the 100k run. The hard-start Pendulum reliability issue appears structured and is not obviously solved by increasing environment interaction from 100k to 500k under the same CleanRL SAC settings.
