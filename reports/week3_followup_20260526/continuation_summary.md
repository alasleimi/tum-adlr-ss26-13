# Week 3 Follow-Up Summary

Generated: 2026-05-26.

This is the current scientific read after the follow-up work on real SimbaV2 code, ReDo, replay/performance correlations, no-distributional scale checks, UTD2, and a hard-reset curriculum.

## Main Answer

The strongest mechanism result is still the 100k matched comparison: SAC and full SimbaV2 see essentially the same near-upright replay coverage and have tied return success, but full SimbaV2 has much better strict state-stability reliability. That points away from a pure exploration failure and toward optimization/plasticity/value-estimation failures.

The best current practical recipe for Pendulum reliability is full official-opt SimbaV2. The scalar/no-distributional variant is close on stochastic posthoc strict success, and UTD2 raises its posthoc strict score, but the exact grid still favors the categorical recipe for task reliability and all-seed consistency. The hard-reset p=0.5 curriculum is negative. A lower-probability p=0.2 hard-reset mix on categorical full SimbaV2 is a small task-stability improvement at 50k, but it lowers near-best-return performance and still trails the 100k full-Simba frontier. p=0.1 is not enough to beat the ordinary 50k task baseline. Hard-state replay p=0.2 confirms the same tradeoff from the update-sampling side: rare hard-boundary transitions are heavily replayed, but exact-grid task reliability and near-best return do not improve.

Older five-seed CleanRL scale runs are now included as a compute baseline. They show that 250k UTD2 and 500k UTD1 improve relative-to-DP closeness, but they do not move the strict-success frontier much. More environment steps alone is therefore not the fastest route to more reliability nines under the current diagnostic.

## Posthoc Reliability Results

| Budget | Condition | Seeds | Episodes/seed | Strict | Return success | Task success | Mean return | Stability | Streak | Collapse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | SAC | 3 | 1000 | 0.675 | 0.701 | 0.829 | -140.291 | 0.829 | 0.970 | 0.000 |
| 50k | Backbone | 3 | 1000 | 0.682 | 0.701 | 0.865 | -145.628 | 0.865 | 0.975 | 0.000 |
| 50k | Full SimbaV2 official opt | 3 | 1000 | 0.689 | 0.700 | 0.899 | -141.784 | 0.899 | 0.979 | 0.000 |
| Legacy 100k | CleanRL SAC | 5 | 1000 | 0.673 | 0.701 | 0.873 | -140.607 | 0.873 | 0.982 | 0.000 |
| Legacy 250k UTD2 | CleanRL SAC | 5 | 1000 | 0.680 | 0.702 | 0.904 | -139.544 | 0.904 | 0.989 | 0.000 |
| Legacy 500k UTD1 | CleanRL SAC | 5 | 1000 | 0.680 | 0.703 | 0.902 | -139.350 | 0.902 | 0.991 | 0.000 |
| 100k | SAC | 3 | 1000 | 0.582 | 0.701 | 0.756 | -142.189 | 0.756 | 0.911 | 0.000 |
| 100k | Full SimbaV2 official opt | 3 | 1000 | 0.684 | 0.699 | 0.913 | -140.032 | 0.913 | 0.984 | 0.000 |
| 50k | Full SimbaV2 no distributional | 3 | 1000 | 0.687 | 0.700 | 0.891 | -141.852 | 0.891 | 0.974 | 0.000 |
| 50k UTD2 | Full SimbaV2 no distributional | 3 | 1000 | 0.696 | 0.702 | 0.893 | -142.323 | 0.893 | 0.976 | 0.000 |
| 50k hard reset p=0.5 | SAC | 3 | 1000 | 0.672 | 0.683 | 0.853 | -146.582 | 0.853 | 0.971 | 0.000 |
| 50k hard reset p=0.5 | Full SimbaV2 no distributional | 3 | 1000 | 0.668 | 0.686 | 0.856 | -145.603 | 0.856 | 0.971 | 0.000 |
| 50k hard reset p=0.2 | Full SimbaV2 official opt | 3 | 1000 | 0.686 | 0.691 | 0.909 | -143.510 | 0.909 | 0.980 | 0.000 |
| 50k hard reset p=0.1 | Full SimbaV2 official opt | 3 | 1000 | 0.678 | 0.687 | 0.895 | -143.037 | 0.895 | 0.972 | 0.000 |
| 50k hard replay p=0.2 | Full SimbaV2 official opt | 3 | 1000 | 0.682 | 0.689 | 0.913 | -144.863 | 0.913 | 0.980 | 0.000 |
| 50k ReDo paired | SAC | 3 | 500 | 0.705 | 0.719 | 0.926 | -137.642 | 0.926 | 0.975 | 0.000 |
| 50k ReDo paired | SAC + ReDo 0.025 | 3 | 500 | 0.704 | 0.720 | 0.897 | -139.475 | 0.897 | 0.977 | 0.000 |
| 50k ReDo paired | SAC + ReDo 0.1 | 3 | 500 | 0.701 | 0.716 | 0.890 | -138.872 | 0.890 | 0.967 | 0.000 |

## Replay And Critic Diagnostics

| Condition | Fixed strict | Posthoc strict | Replay near | Q1 dormant | Q1 rank | Q2 rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k hard replay p=0.2 Full SimbaV2 official opt | 0.600 | 0.682 | 0.785 | 0.000 | 0.243 | 0.239 |
| 50k hard reset p=0.1 Full SimbaV2 official opt | 0.600 | 0.678 | 0.780 | 0.000 | 0.247 | 0.235 |
| 50k hard reset p=0.2 Full SimbaV2 official opt | 0.600 | 0.686 | 0.766 | 0.000 | 0.252 | 0.246 |
| 50k hard reset p=0.5 SAC | 0.600 | 0.672 | 0.723 | 0.462 | 0.074 | 0.065 |
| 50k hard reset p=0.5 Full SimbaV2 no distributional | 0.600 | 0.668 | 0.749 | 0.000 | 0.232 | 0.229 |
| 50k Backbone | 0.600 | 0.682 | 0.804 | 0.141 | 0.147 | 0.155 |
| 50k Full SimbaV2 no distributional | 0.600 | 0.687 | 0.787 | 0.000 | 0.214 | 0.210 |
| 50k Full SimbaV2 official opt | 0.600 | 0.689 | 0.785 | 0.000 | 0.238 | 0.237 |
| 50k SAC | 0.600 | 0.675 | 0.794 | 0.443 | 0.062 | 0.062 |
| 50k UTD2 Full SimbaV2 no distributional | 0.600 | 0.696 | 0.816 | 0.000 | 0.201 | 0.193 |
| Legacy 100k CleanRL SAC | 0.588 | 0.673 | 0.795 | 0.460 | 0.060 | 0.061 |
| 100k SAC | 0.533 | 0.582 | 0.826 | 0.527 | 0.059 | 0.056 |
| 100k Full SimbaV2 official opt | 0.600 | 0.684 | 0.827 | 0.000 | 0.236 | 0.229 |
| Legacy 250k UTD2 CleanRL SAC | 0.550 | 0.680 | 0.837 | 0.754 | 0.054 | 0.053 |
| Legacy 500k UTD1 CleanRL SAC | 0.550 | 0.680 | 0.847 | 0.773 | 0.057 | 0.056 |

## SimbaV2 Implementation Check

The official repository is cloned at `external/SimbaV2`, commit `86899c277cdc697b2b02d827243de1ea93f20a1d`. Its config uses actor hidden `128`, critic hidden `512`, `101` categorical bins, reward normalization with `normalized_g_max=5`, learning rate `1e-4 -> 5e-5`, initial alpha `0.01`, target entropy coefficient `-0.5`, replay min length `5000`, max length `1_000_000`, and the TD-MPC-style gamma heuristic. For Pendulum's 200-step horizon this gives gamma about `0.975`.

The official JAX/Flax code is not directly runnable here without installing the SimbaV2 dependency stack and adding a Pendulum environment adapter. The PyTorch port has therefore been audited against the official code and run in both reduced-dimension and exact-default settings. Exact defaults are much slower; CUDA is required for useful turnaround.

Exact-default CUDA 10k completed rows:

| Condition | Seed | Duration s | Strict | Return success | Mean return |
| --- | ---: | ---: | ---: | ---: | ---: |
| sac_gamma_0975 | 0 | 269.333 | 0.600 | 0.600 | -172.573 |
| sac_gamma_0975 | 1 | 299.295 | 0.600 | 0.600 | -173.410 |
| sac_gamma_0975 | 2 | 302.582 | 0.600 | 0.600 | -174.693 |
| simba_full_real_defaults_official_gamma | 0 | 664.936 | 0.580 | 0.580 | -198.152 |
| simba_full_real_defaults_official_gamma | 1 | 574.386 | 0.580 | 0.580 | -180.672 |
| simba_full_real_defaults_official_gamma | 2 | 453.372 | 0.580 | 0.580 | -183.410 |
| simba_full_no_distributional_real_defaults_official_gamma | 0 | 400.870 | 0.580 | 0.580 | -191.364 |
| simba_full_no_distributional_real_defaults_official_gamma | 1 | 399.671 | 0.580 | 0.580 | -179.850 |
| simba_full_no_distributional_real_defaults_official_gamma | 2 | 407.704 | 0.580 | 0.580 | -190.072 |

## What Helped

- HyperDense/Scaler/LERP backbone with observation normalization, shifted L2 input embedding, and feature normalization: consistently improves critic dormancy/rank.
- Full SimbaV2 official optimizer settings: necessary to avoid the earlier false negative from using CleanRL optimizer settings with full SimbaV2.
- Reward scaling with the categorical critic: required. Distributional variants without reward scaling collapse at 10k.

## What Did Not Help

- Projection alone: negative in the scalar CleanRL setting and should not be described as a faithful SimbaV2 ablation.
- ReDo for plain SAC: improves representation health slightly, but at 50k it does not improve posthoc reliability.
- Training-only hard-reset p=0.5 curriculum: preserves replay and Simba representation health but lowers the task-reliability frontier relative to ordinary full SimbaV2.
- Hard-state replay p=0.2: increases update pressure on rare hard-boundary states without changing resets, but does not beat ordinary 50k full SimbaV2 or hard-reset p=0.2 on the exact grid.
- Official low LR/alpha on plain SAC: harmful at 10k in the tracker suite, so the mechanism is not simply smaller updates.
- CleanRL scale alone: 250k UTD2 and 500k UTD1 do not dominate the 50k/100k SimbaV2 recipes on strict nines.

## Why 10k Is Not Enough

Official SimbaV2 is designed for stable scaling, not fastest early learning. The combination of larger models, `5000` warmup steps, lower LR, low alpha, reward scaling, categorical value estimation, and weight projection reduces short-budget speed. The 10k component matrix is useful for attribution and implementation checks; 50k/100k is the right scale for reliability claims.

## Presentation Guidance

Lead with the 100k reliability slide. Use the replay/correlation analysis to argue that replay coverage is necessary but not sufficient. Use ReDo, hard-reset p=0.5, hard-reset p=0.1, hard-replay p=0.2, and the older compute-scale SAC rows as negative/diagnostic results. Use hard-reset p=0.2 as a secondary positive for task stability only. Use the no-distributional rows as an important simplification study, not the lead recipe: scalar full-Simba is close on posthoc strict success, but categorical full-Simba is still better on exact-grid task reliability.

Key files:
- `docs/week3_open_questions.md`
- `reports/week3_replay_performance_correlation_20260526/correlation_summary.md`
- `reports/week3_no_distributional_scale_50k_20260526/replay_diagnostics/index.html`
- `reports/week3_redo_scale_50k_20260526/replay_diagnostics/index.html`
- `reports/week3_simbav2_final_update_20260525/week3_final_update.html`
