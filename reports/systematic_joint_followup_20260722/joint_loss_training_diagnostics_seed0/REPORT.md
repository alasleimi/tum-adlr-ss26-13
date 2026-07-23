# Joint-loss screen diagnostics

This is a descriptive, read-only report over stored telemetry. It does not rank, select, re-evaluate, route, ensemble, or modify models. Rows are lexicographic.

## Counter semantics

`environment_step` is the step stored in telemetry. Update curves use the separately named `optimizer_update_index_derived`, checked against the final `run_complete` update count. The logged `num_optimizer_updates` field is only a per-window record count and is not treated as cumulative.

## Run overview

| condition | seed | env steps | optimizer updates | ref BC loss | SAC loss | critic loss | grad cosine mean | conflict probes | active gate | ref MAE | eval mean return | task success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h0_bc_only_fast8/seed0 | 0 | 5000 | 8000 | 0.00476721 | 0.750343 | 7.13231 | n/a | n/a | n/a | 0.0272792 | -170.435 | 0.6 |
| h1_bc_plus_one_step/seed0 | 0 | 5000 | 8000 | 0.0111177 | 0.505798 | 7.17461 | -0.110584 | 0.666667 | 1 | 0.0541894 | -170.383 | 0.6 |
| h2b_fast8_unfiltered_sac5/seed0 | 0 | 5000 | 8000 | 0.00616552 | 0.497199 | 7.15725 | 0.0776596 | 0.333333 | 1 | 0.0448181 | -170.036 | 0.6 |
| h2d_fast8_unfiltered_grad_balance/seed0 | 0 | 5000 | 8000 | 0.0071199 | 0.490538 | 7.16122 | -0.151682 | 0.666667 | 1 | 0.0560506 | -169.837 | 0.6 |
| h2f_warm_immediate_joint_grad_balance/seed0 | 0 | 5000 | 8000 | 0.00783604 | 0.0204822 | 7.20481 | -0.147166 | 0.625 | 1 | 0.0677381 | -170.376 | 0.6 |
| h3_bc_plus_fast8_target_gate/seed0 | 0 | 5000 | 8000 | 0.00535978 | 0.586836 | 7.14952 | 0.0438902 | 0.333333 | 0.101562 | 0.0403171 | -170.395 | 0.6 |
| h3b_fast8_target_gate_lr2e6/seed0 | 0 | 5000 | 8000 | 0.00354974 | 0.460529 | 7.13278 | -0.00458256 | 0.333333 | 0.286458 | 0.0295774 | -170.521 | 0.6 |
| h3c_fast8_target_gate_sac5/seed0 | 0 | 5000 | 8000 | 0.00768463 | 0.588855 | 7.15424 | -0.106628 | 0.666667 | 0.197917 | 0.0547825 | -170.231 | 0.6 |
| h_critic_only_fast8/seed0 | 0 | 5000 | 8000 | n/a | n/a | 7.12659 | n/a | n/a | n/a | n/a | -167.691 | 0.7 |
| h_sac_only_fast8/seed0 | 0 | 5000 | 8000 | n/a | 0.490027 | 7.17436 | n/a | n/a | 1 | n/a | -200.481 | 0.7 |

The conflict-probe column is the fraction of logged window-mean gradient cosines below zero; it is not an individual-update conflict rate. `active gate` excludes pre-activation gate probes.

## Telemetry coverage

| metric | runs with samples | total runs |
|---|---:|---:|
| `reference_bc_loss` | 8 | 10 |
| `sac_actor_loss` | 9 | 10 |
| `actor_total_loss` | 9 | 10 |
| `critic_loss` | 10 | 10 |
| `alpha` | 10 | 10 |
| `policy_entropy` | 9 | 10 |
| `bc_gradient_norm` | 7 | 10 |
| `sac_gradient_norm` | 7 | 10 |
| `weighted_bc_gradient_norm` | 7 | 10 |
| `weighted_sac_gradient_norm` | 7 | 10 |
| `gradient_cosine` | 7 | 10 |
| `weighted_gradient_cosine_before` | 0 | 10 |
| `weighted_gradient_cosine_after` | 0 | 10 |
| `pcgrad_projection_fraction` | 6 | 10 |
| `gate_selection_fraction` | 9 | 10 |
| `sac_actor_loss_active_fraction` | 9 | 10 |
| `reference_action_mae` | 8 | 10 |
| `actor_deterministic_action_saturation_fraction` | 3 | 10 |
| `actor_mean_logit_abs_mean` | 3 | 10 |
| `replay_action_saturation_fraction` | 10 | 10 |
| `eval_mean_return` | 10 | 10 |
| `eval_worst_return` | 10 | 10 |
| `eval_task_success_rate` | 10 | 10 |
| `eval_near_upright_fraction` | 10 | 10 |
| `eval_near_reference_success_rate` | 0 | 10 |
| `actor_max_layer_dormant_fraction` | 10 | 10 |
| `q1_max_layer_dormant_fraction` | 10 | 10 |
| `q2_max_layer_dormant_fraction` | 10 | 10 |
| `actor_min_layer_effective_rank_fraction` | 10 | 10 |
| `q1_min_layer_effective_rank_fraction` | 10 | 10 |
| `q2_min_layer_effective_rank_fraction` | 10 | 10 |
| `critic_max_layer_dormant_fraction` | 10 | 10 |
| `critic_min_layer_effective_rank_fraction` | 10 | 10 |
| `gradient_conflict_probe` | 7 | 10 |
| `gate_selection_fraction_active` | 8 | 10 |

## Skipped runs

| run | reason |
|---|---|
| h3d_fast8_target_gate_grad_balance/seed0 | run_complete event is absent |

## Artifacts

- `joint_loss_diagnostics.json`: full provenance, definitions, curves, and summaries
- `joint_loss_run_summary.csv`: one comparable row per run
- `joint_loss_metric_summary.csv`: one row per run and canonical metric
- `joint_loss_curves.csv`: tidy points with both counters
- `loss_curves.png`, `gradient_gate_curves.png`, `representation_health_curves.png`, and `evaluation_curves.png`
