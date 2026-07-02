# SimbaV2, Distillation, and Reference Grid Comparison

This report compares the 5-seed SimbaV2 100k baseline, two same-size supervised distillation policies, and the DP/controller references on the 61x41 Pendulum reset-support grid.

- Near-reference metric: `near_best_known_return_eps`, i.e. return within `5` of `max(DP, controller)`.
- Task success is the stability/task metric, not a fixed-return threshold.
- Distillation policies are single checkpoints; SimbaV2 is the 5-seed aggregate.

## Summary Metrics

| Policy | Type | Seeds | Task success | Near max ref | Near controller | Beats max ref | Mean regret to max ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| Controller reference | reference | 1 | 0.7621 | 0.6829 | 1.0000 | 0.0392 | 13.6024 |
| DP reference | reference | 1 | 0.9332 | 1.0000 | 1.0000 | 0.9608 | 0.0027 |
| Max(DP, controller) reference | reference | 1 | 0.9332 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| SimbaV2 100k (5 seeds) | learned | 5 | 0.9146 | 0.9184 | 0.9926 | 0.0852 | 3.1817 |
| Distill balanced (1 seed) | learned | 1 | 0.9444 | 0.9872 | 0.9940 | 0.1152 | 1.3768 |
| Distill hard120 (1 seed) | learned | 1 | 0.9316 | 0.9912 | 0.9988 | 0.0412 | 1.0416 |

## Bar Charts

![Task success](figures/bar_task_success.png)

![Near max reference](figures/bar_near_max_ref.png)

![Beats max reference](figures/bar_beats_max_ref.png)

## Initial-Condition Maps

Each map uses initial angle in degrees on the x-axis and initial angular velocity on the y-axis.

### Task Success

![Task success maps](figures/map_task_success_panels.png)

### Near Max Reference

![Near max reference maps](figures/map_near_max_ref_panels.png)

### Regret to Max Reference

![Regret maps](figures/map_regret_to_max_ref_panels.png)

## Notes

- `Distill hard120` is the only supervised-only policy above `0.99` near max ref, but it trades off task success relative to `Distill balanced`.
- `Distill balanced` has the best learned-policy task success in this comparison.
- SimbaV2 remains behind both distillation policies on near max ref, but it is a multi-seed RL aggregate rather than a supervised single checkpoint.

## Sources

- Controller reference: `reports/distill_best_simbav2_balanced_400k_20260701/grid_reset_support_61x41_vellim1/controller_grid.csv`
- DP reference: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv`
- Max(DP, controller) reference: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv + reports/distill_best_simbav2_balanced_400k_20260701/grid_reset_support_61x41_vellim1/controller_grid.csv`
- SimbaV2 100k (5 seeds): `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt`
- Distill balanced (1 seed): `reports/distill_best_simbav2_balanced_400k_20260701/relative_success_vellim1`
- Distill hard120 (1 seed): `reports/distill_best_simbav2_hard120_450k_20260701/relative_success_vellim1`
