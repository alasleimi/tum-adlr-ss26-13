# Pure-RL gap diagnostics

This is a read-only fixed-checkpoint probe. The original finite-horizon task diagnostic is retained: each local candidate is used for one step, then the deterministic actor controls the remainder. A separate long-horizon probe checks the critic against deterministic reward-only and stochastic-soft checkpoint-policy returns in training units. Neither is an oracle-Q proof.

## Aggregate

Runs: 5 (seeds [3, 4, 0, 1, 2]).

| Measure | Mean | Min | Max |
|---|---:|---:|---:|
| critic_spearman | 0.6931 | 0.5825 | 0.8217 |
| critic_pairwise_accuracy | 0.8437 | 0.7891 | 0.9067 |
| critic_centered_pearson | 0.3901 | 0.2453 | 0.6833 |
| q_selected_true_raw_gain | 0.0102 | -0.1042 | 0.1469 |
| q_selected_harms_actor_fraction | 0.1899 | 0.1111 | 0.2525 |
| q_action_gradient_abs | 0.0165 | 0.0124 | 0.0191 |
| critic_disagreement_to_local_signal | 2.4264 | 1.7173 | 2.9077 |
| critic_max_dormancy | 0 | 0 | 0 |
| critic_min_effective_rank_including_embedding | 0.0477 | 0.0464 | 0.0484 |
| critic_hidden_min_effective_rank | 0.1534 | 0.1389 | 0.1670 |
| actor_saturation_fraction | 0.9313 | 0.9091 | 0.9394 |
| deployed_qsearch_gain_vs_reflected_actor | 16.8289 | 16.6738 | 16.9462 |
| deployed_qsearch_harms_reflected_actor_fraction | 0.4545 | 0.3131 | 0.5455 |
| actor_reflection_error | 0.2432 | 0.1303 | 0.3812 |
| critic_reflection_error | 0.0342 | 0.0198 | 0.0467 |
| critic_training_aligned_deterministic_spearman | 0.7217 | 0.6179 | 0.9535 |
| critic_training_aligned_soft_spearman | 0.7112 | 0.6174 | 0.9436 |
| critic_training_aligned_soft_pairwise_accuracy | 0.8418 | 0.8000 | 0.9511 |
| critic_soft_long_minus_finite_abs_mean | 4.836e-05 | 2.893e-05 | 6.024e-05 |
| critic_hidden_rank_on_replay | 0.2653 | 0.2524 | 0.2706 |
| critic_hidden_rank_on_broad_support | 0.2962 | 0.2883 | 0.3102 |

## Seed 3

Run: `C:\Users\Ala\Desktop\Project 15\runs\week3_100k_component_ablation_20260527\simba_full_official_opt\seed3`<br>
Checkpoint/logged step: 100000<br>
Probe: 99 hard states x 21 local actions, horizon 200.

| Diagnostic | Value |
|---|---:|
| Within-state Q/return Pearson | 0.2516 |
| Mean per-state Spearman | 0.8217 |
| Pairwise action-order accuracy | 0.9067 |
| Median Q / true-scaled range ratio | 1.4865 |
| Critic disagreement / median local Q range | 2.1941 |
| Mean abs(dQ/da) at actor | 0.0168 |
| Q-selected raw gain vs actor | 0.0200 |
| Q-selected harms actor fraction | 0.1111 |
| Critic max dormant fraction | 0 |
| Critic min rank fraction, including rank-limited embedder | 0.0464 |
| Critic min hidden-layer effective-rank fraction | 0.1389 |
| Actor action saturation fraction (>=99.5% bound) | 0.9394 |
| Median normalized tanh derivative | 4.461e-04 |
| Actor reflection MAE | 0.2909 |
| Critic reflection MAE | 0.0198 |
| Deployed reflection+Q-search gain vs reflected actor | 16.8999 |
| Deployed Q-search harms reflected actor fraction | 0.5455 |

Active triage flags: critic_uncertainty_exceeds_local_action_signal, material_actor_reflection_error, critic_reflection_error_exceeds_local_signal, actor_mean_is_heavily_saturated, deployed_qsearch_often_harms_reflected_actor.

Training-aligned critic probe:

| Diagnostic | Value |
|---|---:|
| Long horizon | 917 |
| gamma^H residual multiplier | 9.942e-05 |
| Deterministic/no-entropy Spearman | 0.6179 |
| Stochastic-soft Spearman | 0.6174 |
| Stochastic-soft pairwise accuracy | 0.8000 |
| Soft long-minus-finite abs mean | 6.020e-05 |
| Soft-return MC standard-error mean | 4.119e-04 |

Representation sensitivity by probe distribution:

| Probe | Available | Critic dormant max | Critic hidden-rank min |
|---|---|---:|---:|
| Hard states / actor actions | yes | 0 | 0.1389 |
| Replay state-actions | yes | 0 | 0.2702 |
| Broad uniform state-actions | yes | 0 | 0.2900 |

Training-curve summaries:

- q_loss: 6.6890 -> 1.6511 (early/late quartile means; last 1.2701).
- q1_loss: 3.3445 -> 0.8255 (early/late quartile means; last 0.6350).
- q2_loss: 3.3445 -> 0.8255 (early/late quartile means; last 0.6350).
- actor_loss: 0.2980 -> 0.1020 (early/late quartile means; last 0.1008).
- entropy: 0.7084 -> -0.2845 (early/late quartile means; last -0.4232).
- alpha: 0.0025 -> 2.622e-05 (early/late quartile means; last 1.939e-05).
- eval_mean_return: -1.508e+03 -> -163.4389 (early/late quartile means; last -163.4389).
- eval_task_success: 0 -> 0.8200 (early/late quartile means; last 0.8200).

## Seed 4

Run: `C:\Users\Ala\Desktop\Project 15\runs\week3_100k_component_ablation_20260527\simba_full_official_opt\seed4`<br>
Checkpoint/logged step: 100000<br>
Probe: 99 hard states x 21 local actions, horizon 200.

| Diagnostic | Value |
|---|---:|
| Within-state Q/return Pearson | 0.5052 |
| Mean per-state Spearman | 0.6381 |
| Pairwise action-order accuracy | 0.8162 |
| Median Q / true-scaled range ratio | 1.2120 |
| Critic disagreement / median local Q range | 2.7266 |
| Mean abs(dQ/da) at actor | 0.0124 |
| Q-selected raw gain vs actor | -0.0500 |
| Q-selected harms actor fraction | 0.2525 |
| Critic max dormant fraction | 0 |
| Critic min rank fraction, including rank-limited embedder | 0.0474 |
| Critic min hidden-layer effective-rank fraction | 0.1509 |
| Actor action saturation fraction (>=99.5% bound) | 0.9394 |
| Median normalized tanh derivative | 5.935e-04 |
| Actor reflection MAE | 0.3812 |
| Critic reflection MAE | 0.0238 |
| Deployed reflection+Q-search gain vs reflected actor | 16.9024 |
| Deployed Q-search harms reflected actor fraction | 0.3131 |

Active triage flags: critic_uncertainty_exceeds_local_action_signal, q_local_search_often_harms_actor, material_actor_reflection_error, critic_reflection_error_exceeds_local_signal, actor_mean_is_heavily_saturated, deployed_qsearch_often_harms_reflected_actor.

Training-aligned critic probe:

| Diagnostic | Value |
|---|---:|
| Long horizon | 917 |
| gamma^H residual multiplier | 9.942e-05 |
| Deterministic/no-entropy Spearman | 0.6759 |
| Stochastic-soft Spearman | 0.6556 |
| Stochastic-soft pairwise accuracy | 0.8087 |
| Soft long-minus-finite abs mean | 5.659e-05 |
| Soft-return MC standard-error mean | 1.934e-04 |

Representation sensitivity by probe distribution:

| Probe | Available | Critic dormant max | Critic hidden-rank min |
|---|---|---:|---:|
| Hard states / actor actions | yes | 0 | 0.1509 |
| Replay state-actions | yes | 0 | 0.2640 |
| Broad uniform state-actions | yes | 0 | 0.2883 |

Training-curve summaries:

- q_loss: 6.7150 -> 1.6479 (early/late quartile means; last 1.2713).
- q1_loss: 3.3575 -> 0.8240 (early/late quartile means; last 0.6357).
- q2_loss: 3.3575 -> 0.8239 (early/late quartile means; last 0.6357).
- actor_loss: 0.2758 -> 0.0997 (early/late quartile means; last 0.0988).
- entropy: 0.6280 -> -0.3786 (early/late quartile means; last -0.4689).
- alpha: 0.0025 -> 3.055e-05 (early/late quartile means; last 2.457e-05).
- eval_mean_return: -1.524e+03 -> -160.2781 (early/late quartile means; last -160.2781).
- eval_task_success: 0 -> 0.8400 (early/late quartile means; last 0.8400).

## Seed 0

Run: `C:\Users\Ala\Desktop\Project 15\runs\week3_simbav2_scale_100k_20260526\simba_full_official_opt\seed0`<br>
Checkpoint/logged step: 100000<br>
Probe: 99 hard states x 21 local actions, horizon 200.

| Diagnostic | Value |
|---|---:|
| Within-state Q/return Pearson | 0.2453 |
| Mean per-state Spearman | 0.7096 |
| Pairwise action-order accuracy | 0.8527 |
| Median Q / true-scaled range ratio | 1.3984 |
| Critic disagreement / median local Q range | 2.9077 |
| Mean abs(dQ/da) at actor | 0.0184 |
| Q-selected raw gain vs actor | 0.1469 |
| Q-selected harms actor fraction | 0.1616 |
| Critic max dormant fraction | 0 |
| Critic min rank fraction, including rank-limited embedder | 0.0484 |
| Critic min hidden-layer effective-rank fraction | 0.1556 |
| Actor action saturation fraction (>=99.5% bound) | 0.9091 |
| Median normalized tanh derivative | 4.676e-04 |
| Actor reflection MAE | 0.1303 |
| Critic reflection MAE | 0.0429 |
| Deployed reflection+Q-search gain vs reflected actor | 16.7220 |
| Deployed Q-search harms reflected actor fraction | 0.4646 |

Active triage flags: critic_uncertainty_exceeds_local_action_signal, material_actor_reflection_error, critic_reflection_error_exceeds_local_signal, actor_mean_is_heavily_saturated, deployed_qsearch_often_harms_reflected_actor.

Training-aligned critic probe:

| Diagnostic | Value |
|---|---:|
| Long horizon | 917 |
| gamma^H residual multiplier | 9.942e-05 |
| Deterministic/no-entropy Spearman | 0.6302 |
| Stochastic-soft Spearman | 0.6213 |
| Stochastic-soft pairwise accuracy | 0.8095 |
| Soft long-minus-finite abs mean | 2.893e-05 |
| Soft-return MC standard-error mean | 2.492e-04 |

Representation sensitivity by probe distribution:

| Probe | Available | Critic dormant max | Critic hidden-rank min |
|---|---|---:|---:|
| Hard states / actor actions | yes | 0 | 0.1556 |
| Replay state-actions | yes | 0 | 0.2706 |
| Broad uniform state-actions | yes | 0 | 0.2952 |

Training-curve summaries:

- q_loss: 4.0364 -> 4.0364 (early/late quartile means; last 4.0364).
- q1_loss: 2.0182 -> 2.0182 (early/late quartile means; last 2.0182).
- q2_loss: 2.0182 -> 2.0182 (early/late quartile means; last 2.0182).
- actor_loss: 0.1167 -> 0.1167 (early/late quartile means; last 0.1167).
- entropy: 0.2136 -> 0.2136 (early/late quartile means; last 0.2136).
- alpha: 1.784e-05 -> 1.784e-05 (early/late quartile means; last 1.784e-05).
- eval_mean_return: -1.320e+03 -> -158.9232 (early/late quartile means; last -158.9232).

## Seed 1

Run: `C:\Users\Ala\Desktop\Project 15\runs\week3_simbav2_scale_100k_20260526\simba_full_official_opt\seed1`<br>
Checkpoint/logged step: 100000<br>
Probe: 99 hard states x 21 local actions, horizon 200.

| Diagnostic | Value |
|---|---:|
| Within-state Q/return Pearson | 0.6833 |
| Mean per-state Spearman | 0.7137 |
| Pairwise action-order accuracy | 0.8537 |
| Median Q / true-scaled range ratio | 1.5294 |
| Critic disagreement / median local Q range | 1.7173 |
| Mean abs(dQ/da) at actor | 0.0191 |
| Q-selected raw gain vs actor | 0.0383 |
| Q-selected harms actor fraction | 0.1717 |
| Critic max dormant fraction | 0 |
| Critic min rank fraction, including rank-limited embedder | 0.0484 |
| Critic min hidden-layer effective-rank fraction | 0.1544 |
| Actor action saturation fraction (>=99.5% bound) | 0.9293 |
| Median normalized tanh derivative | 2.654e-04 |
| Actor reflection MAE | 0.1591 |
| Critic reflection MAE | 0.0376 |
| Deployed reflection+Q-search gain vs reflected actor | 16.9462 |
| Deployed Q-search harms reflected actor fraction | 0.4848 |

Active triage flags: critic_uncertainty_exceeds_local_action_signal, material_actor_reflection_error, critic_reflection_error_exceeds_local_signal, actor_mean_is_heavily_saturated, deployed_qsearch_often_harms_reflected_actor, critic_ranking_materially_better_on_training_aligned_soft_target.

Training-aligned critic probe:

| Diagnostic | Value |
|---|---:|
| Long horizon | 917 |
| gamma^H residual multiplier | 9.942e-05 |
| Deterministic/no-entropy Spearman | 0.9535 |
| Stochastic-soft Spearman | 0.9436 |
| Stochastic-soft pairwise accuracy | 0.9511 |
| Soft long-minus-finite abs mean | 6.024e-05 |
| Soft-return MC standard-error mean | 1.249e-04 |

Representation sensitivity by probe distribution:

| Probe | Available | Critic dormant max | Critic hidden-rank min |
|---|---|---:|---:|
| Hard states / actor actions | yes | 0 | 0.1544 |
| Replay state-actions | yes | 0 | 0.2524 |
| Broad uniform state-actions | yes | 0 | 0.2974 |

Training-curve summaries:

- q_loss: 4.0451 -> 4.0451 (early/late quartile means; last 4.0451).
- q1_loss: 2.0225 -> 2.0225 (early/late quartile means; last 2.0225).
- q2_loss: 2.0226 -> 2.0226 (early/late quartile means; last 2.0226).
- actor_loss: 0.0703 -> 0.0703 (early/late quartile means; last 0.0703).
- entropy: 0.3055 -> 0.3055 (early/late quartile means; last 0.3055).
- alpha: 1.432e-05 -> 1.432e-05 (early/late quartile means; last 1.432e-05).
- eval_mean_return: -1.566e+03 -> -160.3265 (early/late quartile means; last -160.3265).

## Seed 2

Run: `C:\Users\Ala\Desktop\Project 15\runs\week3_simbav2_scale_100k_20260526\simba_full_official_opt\seed2`<br>
Checkpoint/logged step: 100000<br>
Probe: 99 hard states x 21 local actions, horizon 200.

| Diagnostic | Value |
|---|---:|
| Within-state Q/return Pearson | 0.2652 |
| Mean per-state Spearman | 0.5825 |
| Pairwise action-order accuracy | 0.7891 |
| Median Q / true-scaled range ratio | 1.3538 |
| Critic disagreement / median local Q range | 2.5864 |
| Mean abs(dQ/da) at actor | 0.0159 |
| Q-selected raw gain vs actor | -0.1042 |
| Q-selected harms actor fraction | 0.2525 |
| Critic max dormant fraction | 0 |
| Critic min rank fraction, including rank-limited embedder | 0.0477 |
| Critic min hidden-layer effective-rank fraction | 0.1670 |
| Actor action saturation fraction (>=99.5% bound) | 0.9394 |
| Median normalized tanh derivative | 3.971e-04 |
| Actor reflection MAE | 0.2545 |
| Critic reflection MAE | 0.0467 |
| Deployed reflection+Q-search gain vs reflected actor | 16.6738 |
| Deployed Q-search harms reflected actor fraction | 0.4646 |

Active triage flags: critic_uncertainty_exceeds_local_action_signal, q_local_search_often_harms_actor, material_actor_reflection_error, critic_reflection_error_exceeds_local_signal, actor_mean_is_heavily_saturated, deployed_qsearch_often_harms_reflected_actor.

Training-aligned critic probe:

| Diagnostic | Value |
|---|---:|
| Long horizon | 917 |
| gamma^H residual multiplier | 9.942e-05 |
| Deterministic/no-entropy Spearman | 0.7313 |
| Stochastic-soft Spearman | 0.7179 |
| Stochastic-soft pairwise accuracy | 0.8396 |
| Soft long-minus-finite abs mean | 3.584e-05 |
| Soft-return MC standard-error mean | 2.352e-04 |

Representation sensitivity by probe distribution:

| Probe | Available | Critic dormant max | Critic hidden-rank min |
|---|---|---:|---:|
| Hard states / actor actions | yes | 0 | 0.1670 |
| Replay state-actions | yes | 0 | 0.2692 |
| Broad uniform state-actions | yes | 0 | 0.3102 |

Training-curve summaries:

- q_loss: 4.0338 -> 4.0338 (early/late quartile means; last 4.0338).
- q1_loss: 2.0169 -> 2.0169 (early/late quartile means; last 2.0169).
- q2_loss: 2.0169 -> 2.0169 (early/late quartile means; last 2.0169).
- actor_loss: 0.0974 -> 0.0974 (early/late quartile means; last 0.0974).
- entropy: 0.2142 -> 0.2142 (early/late quartile means; last 0.2142).
- alpha: 2.170e-05 -> 2.170e-05 (early/late quartile means; last 2.170e-05).
- eval_mean_return: -1.231e+03 -> -160.6587 (early/late quartile means; last -160.6587).

## Interpretation limits

Dormancy and effective rank depend on the probe distribution; hard-policy, replay, and broad-support results are therefore reported separately. The layer-0 critic embedder's centered rank is structurally bounded by the low-dimensional state-action input, so its small rank fraction is not evidence of collapse; hidden-layer rank and matched controls are the relevant checks. The finite task probe and the training-aligned continuing soft probe answer different questions. The soft probe is Monte Carlo and uses a frozen checkpoint policy, while the learned critic reflects the whole nonstationary training history. Threshold flags are descriptive; causal claims require component ablations across seeds. Full point series and layer-level values are retained in `pure_rl_gap_diagnostics.json`.
