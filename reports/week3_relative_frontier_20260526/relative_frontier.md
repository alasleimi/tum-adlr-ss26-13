# DP-Relative Reliability Frontier

Generated: 2026-05-26.

This table compares exact initial-condition grid evaluations against DP and energy-controller references. It is the reliability view to use when the legacy `return >= -200` diagnostic saturates near its DP-calibrated ceiling.

Baseline for delta columns: `SAC100k`.

## Ranked Conditions

| Condition | Seeds | Task | Task nines | Delta task | Near best | Delta near best | Strict | All-seed task cells | Near-down task | Mid-angle task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullSimba100k | 3 | 0.914 | 1.068 | +0.157 | 0.925 | +0.133 | 0.680 | 0.895 | 0.706 | 1.000 |
| FullSimbaHardReset02_50k | 3 | 0.903 | 1.011 | +0.145 | 0.736 | -0.056 | 0.682 | 0.888 | 0.652 | 1.000 |
| FullSimba50k | 3 | 0.898 | 0.992 | +0.140 | 0.816 | +0.024 | 0.681 | 0.865 | 0.647 | 1.000 |
| FullSimbaHardReset01_50k | 3 | 0.895 | 0.978 | +0.137 | 0.773 | -0.018 | 0.680 | 0.876 | 0.652 | 1.000 |
| FullSimbaHardReplay02_50k | 3 | 0.890 | 0.958 | +0.132 | 0.720 | -0.072 | 0.679 | 0.867 | 0.615 | 1.000 |
| Legacy500kUTD1 | 5 | 0.886 | 0.944 | +0.128 | 0.963 | +0.171 | 0.675 | 0.793 | 0.584 | 1.000 |
| FullSimbaNoDist50k | 3 | 0.884 | 0.937 | +0.127 | 0.767 | -0.025 | 0.679 | 0.852 | 0.644 | 1.000 |
| FullSimbaNoDistUTD2_50k | 3 | 0.882 | 0.927 | +0.124 | 0.852 | +0.060 | 0.676 | 0.819 | 0.562 | 1.000 |
| FullSimbaNoDistHardReset50k | 3 | 0.866 | 0.874 | +0.109 | 0.741 | -0.051 | 0.679 | 0.822 | 0.568 | 1.000 |
| Legacy100k | 5 | 0.859 | 0.852 | +0.102 | 0.889 | +0.097 | 0.669 | 0.737 | 0.522 | 1.000 |
| SACHardReset50k | 3 | 0.848 | 0.819 | +0.091 | 0.600 | -0.192 | 0.679 | 0.772 | 0.481 | 1.000 |
| SAC50k | 3 | 0.823 | 0.752 | +0.065 | 0.853 | +0.061 | 0.674 | 0.756 | 0.448 | 1.000 |
| SAC100k | 3 | 0.758 | 0.616 | +0.000 | 0.792 | +0.000 | 0.584 | 0.457 | 0.489 | 0.833 |

## Read

- `Task` is the primary task-only stability rate over reset-support grid rollouts.
- `Near best` is the fraction within the return epsilon of `max(DP, controller)`, using each source report's epsilon.
- `Strict` is still shown for continuity, but it should not be treated as the main route to more nines once it is near the DP-calibrated ceiling.
- `All-seed task cells` measures how much of the initial-condition grid is solved by every training seed, so it is stricter than pooled rollout success.

Raw table: `relative_frontier.csv`.
