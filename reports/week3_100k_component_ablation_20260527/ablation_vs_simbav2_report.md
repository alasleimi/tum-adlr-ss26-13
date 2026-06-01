# 100k Ablations vs SimbaV2

Weight projection doesn't seem to be important from the ablations; otherwise not interesting in the report.

Generated on 2026-06-01.

Baseline used for ablation deltas: **Full SimbaV2 official opt 100k (seeds 3+4)**.
All values are from the exact 61x41 reset-support evaluation grid (2501 start states).

## Aggregate Comparison

| Condition | Seeds | Task success | Reference success | Near-down reference | Delta task vs baseline | Delta reference | Delta near-down |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full SimbaV2 (official opt, 100k) | 3+4 | 91.5% | 90.8% | 64.9% | +0.0 pp | +0.0 pp | +0.0 pp |
| Minus feature norm | 0+1 | 82.6% | 70.5% | 21.3% | -8.9 pp | -20.4 pp | -43.6 pp |
| Minus weight projection | 0+1 | 89.9% | 90.2% | 65.7% | -1.6 pp | -0.6 pp | +0.9 pp |
| Minus distributional critic | 0+1 | 89.2% | 80.2% | 66.5% | -2.3 pp | -10.7 pp | +1.7 pp |
| Minus reward scaling | 0+1 | 30.2% | 30.5% | 0.0% | -61.3 pp | -60.3 pp | -64.9 pp |

## Plots

![Ablation metrics](figures_ablation_vs_simbav2/ablation_vs_simbav2_metrics.png)

![Ablation start-state delta maps](figures_ablation_vs_simbav2/ablation_vs_simbav2_start_state_delta_maps.png)

Map notes:
- Left column: delta task-success rate per initial state.
- Right column: delta reference-success rate (`near_best_known_return_eps_rate`) per initial state.
- Positive values mean the ablation is better than the Full SimbaV2 official baseline at that start state.
- Map baseline is `simba_full_official_opt` (seeds 3+4).

## Ablation Definitions

- **Minus feature norm**: keeps full official SimbaV2, but turns off feature normalization via `--simba-no-feature-norm`; features are passed unnormalized into the critic/actor stacks.
- **Minus weight projection**: removes `--simba-weight-projection`; replaced by the standard Simba backbone path without projection, while keeping distributional critic and reward scaling.
- **Minus distributional critic**: removes `--simba-distributional-critic` (and binning); replaced by the scalar Q critic path, while keeping backbone, projection, and reward scaling.
- **Minus reward scaling**: removes `--simba-reward-scaling`; replaced by raw environment rewards (no Simba reward-scaling transform), while keeping backbone, projection, and distributional critic.
