# Five-Seed 100k Summary

Generated on 2026-05-27 for the workshop deck refresh.

## Protocol

- Policies: CleanRL SAC 100k and full SimbaV2 official-opt 100k.
- Seeds: 0-4 for each policy.
- Exact grid: 61 x 41 reset-support states, 2501 cells, 12505 deterministic grid rollouts per policy.
- Headline success: return within 5 of `max(DP, hand controller)` from the same initial state.
- Caveat: seeds 3-4 are follow-up runs with diagnostics every 10k steps instead of 100k. Training recipe and final evaluation are otherwise the same.

## Main Numbers

| Policy | Reference success | Task-stability | Known-feasible task | Near-down task | Replay near | Q1 dormant | Q1 rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAC 100k | 83.0% +/- 21.6 pp | 81.2% +/- 24.7 pp | 86.6% +/- 26.2 pp | 53.9% | 82.6% | 52.1% | 5.8% |
| Full SimbaV2 100k | 91.8% +/- 2.3 pp | 91.5% +/- 1.7 pp | 97.2% +/- 1.3 pp | 69.4% | 82.4% | 0.0% | 23.5% |

Use `relative_frontier_n5.csv`, `key_posthoc_results_n5.csv`, and `key_diagnostic_results_n5.csv` as the deck data sources.
