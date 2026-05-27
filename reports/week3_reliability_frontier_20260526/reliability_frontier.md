# Reliability Frontier

Generated: 2026-05-26.

This ranks the current saved-checkpoint recipes by strict-success nines, using seed means from the posthoc evaluation summaries. It is a decision aid, not a causal estimate. The main table only includes the 1000-episode posthoc protocol; shorter paired checks are listed separately.

## Main 1000-Episode Ranking

| Budget | Condition | Episodes/seed | Task | Task nines | Strict | Strict nines | Return | Stability | Streak | Collapse | Q1 dormant | Q1 rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | Full SimbaV2 official opt | 1000 | 0.913 | 1.059 | 0.684 | 0.500 | 0.699 | 0.913 | 0.984 | 0.000 | 0.000 | 0.236 |
| 50k hard replay p=0.2 | Full SimbaV2 official opt | 1000 | 0.913 | 1.059 | 0.682 | 0.498 | 0.689 | 0.913 | 0.980 | 0.000 | 0.000 | 0.243 |
| 50k hard reset p=0.2 | Full SimbaV2 official opt | 1000 | 0.909 | 1.043 | 0.686 | 0.503 | 0.691 | 0.909 | 0.980 | 0.000 | 0.000 | 0.252 |
| Legacy 250k UTD2 | CleanRL SAC | 1000 | 0.904 | 1.019 | 0.680 | 0.495 | 0.702 | 0.904 | 0.989 | 0.000 | 0.754 | 0.054 |
| Legacy 500k UTD1 | CleanRL SAC | 1000 | 0.902 | 1.011 | 0.680 | 0.495 | 0.703 | 0.902 | 0.991 | 0.000 | 0.773 | 0.057 |
| 50k | Full SimbaV2 official opt | 1000 | 0.899 | 0.994 | 0.689 | 0.507 | 0.700 | 0.899 | 0.979 | 0.000 | 0.000 | 0.238 |
| 50k hard reset p=0.1 | Full SimbaV2 official opt | 1000 | 0.895 | 0.980 | 0.678 | 0.492 | 0.687 | 0.895 | 0.972 | 0.000 | 0.000 | 0.247 |
| 50k UTD2 | Full SimbaV2 no distributional | 1000 | 0.893 | 0.972 | 0.696 | 0.517 | 0.702 | 0.893 | 0.976 | 0.000 | 0.000 | 0.201 |
| 50k | Full SimbaV2 no distributional | 1000 | 0.891 | 0.963 | 0.687 | 0.505 | 0.700 | 0.891 | 0.974 | 0.000 | 0.000 | 0.214 |
| Legacy 100k | CleanRL SAC | 1000 | 0.873 | 0.896 | 0.673 | 0.486 | 0.701 | 0.873 | 0.982 | 0.000 | 0.460 | 0.060 |

## Shorter Paired Checks

| Budget | Condition | Episodes/seed | Task | Task nines | Strict | Strict nines | Return | Stability | Streak | Collapse | Q1 dormant | Q1 rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k ReDo paired | SAC | 500 | 0.926 | 1.131 | 0.705 | 0.531 | 0.719 | 0.926 | 0.975 | 0.000 | nan | nan |
| 50k ReDo paired | SAC + ReDo 0.025 | 500 | 0.897 | 0.989 | 0.704 | 0.529 | 0.720 | 0.897 | 0.977 | 0.000 | nan | nan |
| 50k ReDo paired | SAC + ReDo 0.1 | 500 | 0.890 | 0.959 | 0.701 | 0.524 | 0.716 | 0.890 | 0.967 | 0.000 | nan | nan |

## Read

- The best measured posthoc strict nines now include the 50k UTD2 scalar row, but the 100k full-Simba row is still the strongest causal comparison because SAC degrades at 100k while replay coverage stays tied.
- 50k no-distributional and UTD2 no-distributional are useful simplification studies. Exact-grid task reliability still favors categorical full SimbaV2, so scalar UTD2 is not the current lead recipe.
- Hard-reset p=0.5 is a negative curriculum result. The lower-probability p=0.2 categorical run improves 50k task success slightly, but at the cost of near-best return. p=0.1 does not beat ordinary 50k full SimbaV2 on task reliability, and hard-state replay p=0.2 does not improve the exact-grid frontier.
- ReDo does not move the frontier; in its paired 50k check, both ReDo thresholds are below paired SAC on strict nines.
- Older five-seed CleanRL scale runs at 250k UTD2 and 500k UTD1 do not beat the reduced-dim SimbaV2 frontier, so more data/updates alone is not the current lead recipe.
- Higher nines should be tracked on task-only and DP/controller-relative metrics. The legacy return-success plateau around `0.700` caps strict nines around `0.52` even when the controller is stable on most feasible starts.

Raw table: `reliability_frontier.csv`.
