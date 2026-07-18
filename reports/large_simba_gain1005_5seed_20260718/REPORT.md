# 99.9% RL + supervised result: calibrated RL-weighted DAgger

Date: 2026-07-18

## Outcome

The goal is achieved with saved, reloadable single-actor checkpoints.

- Near-reference success: **12,500/12,505 = 99.9600%**
- Required for at least 99.9%: **12,493/12,505**
- Margin above the requirement: **7 trials**
- Task success: **11,686/12,505 = 93.4506%**
- Literal strict return wins: **1,809/12,505 = 14.4662%**
- Mean return: **-138.624707**

`Strict` means exactly `policy_return > max(DP_return, controller_return)`. It does not include equality.

## Authoritative evaluation

- Environment: `Pendulum-v1`
- Deterministic horizon: 200 steps
- Initial states per seed: 61 angles in `[-pi, pi)` by 41 velocities in `[-1, 1]` = 2,501
- Training/evaluation seeds: 0, 1, 2, 3, 4
- Total trials: 12,505
- Near-reference criterion: `return >= max(DP_return, controller_return) - 5`
- Task criterion: at least 80% of steps near upright and no non-upright streak longer than 50 steps
- Inference action: one SimbaV2 actor forward pass, followed by its checkpointed action-scale buffer and the environment's ordinary `[-2, 2]` torque clipping
- No inference critic, Q-search, reference query, teacher, policy gate, specialist switch, or router

| Seed | Near reference | Task success | Strict > reference | Mean return |
|---:|---:|---:|---:|---:|
| 0 | 2501/2501 (100.0000%) | 2337/2501 (93.4426%) | 380/2501 (15.1939%) | -138.601240 |
| 1 | 2499/2501 (99.9200%) | 2338/2501 (93.4826%) | 354/2501 (14.1543%) | -138.650307 |
| 2 | 2500/2501 (99.9600%) | 2337/2501 (93.4426%) | 386/2501 (15.4338%) | -138.614709 |
| 3 | 2501/2501 (100.0000%) | 2337/2501 (93.4426%) | 358/2501 (14.3143%) | -138.595572 |
| 4 | 2499/2501 (99.9200%) | 2337/2501 (93.4426%) | 331/2501 (13.2347%) | -138.661706 |

## What was trained and calibrated

Each seed starts from its seed-specific **large SimbaV2 + corrected 100k DAgger** actor. That actor is one 64-wide, two-residual-block SimbaV2 policy. Its earlier training uses max(DP, controller) labels and seed-matched fixed pure-RL critics only for 4x sample weighting; `rl_blend=0`, so critics do not replace the supervised action target.

Each seed then receives the same targeted training stage:

1. Create 240,000 reference-labeled support states: 20% broad, 50% reset support, 15% `120-135 deg`, and 15% near-down.
2. Collect two learner-only DAgger rounds with 500 episodes per round. Every episode has 200 steps, so this adds **200,000 learner-visited states per seed**.
3. Sample DAgger initial states from continuous residual-failure neighborhoods: 40% hard diagonal (`123-131 deg`, matching-sign speed `0.78-0.92`), 40% slow near-down (`-178 to -170 deg`, speed `-0.18 to 0.02`), and 20% fast wrap (`175-180 deg`, speed `0.72-0.92`).
4. At every DAgger step, the current actor action alone is passed to `env.step()`. The visited state is labeled by max(DP, controller) at the correct remaining horizon.
5. Aggregate all states and train the full actor for two epochs per DAgger round, batch size 1,024, learning rate `2e-6`.
6. Fixed seed-matched pure-RL critics mark reference labels for 4x loss weight only when every critic estimates more than `0.05` Q advantage. The target action remains the reference action.
7. Select the DAgger iterate on a disjoint 17x11 broad midpoint grid followed by a 2,001-state continuous failure-mixture holdout.

That targeted stage moves the five-seed result from 12,473 to 12,485 near-reference successes.

The final scalar calibration was selected without the authoritative grid:

1. Evaluate 49 fixed gains from `0.94` through `1.06` in increments of `0.0025` on seed 0's broad 17x11 grid and a disjoint 2,001-state tight hard holdout.
2. Rank by broad near-reference, hard near-reference, broad task success, hard mean return, then broad mean return.
3. Select **gain 1.005**. It keeps broad near-reference at 100% and raises the hard holdout from 94.8026% to 99.9500%.
4. Freeze that one gain and validate it without retuning on seeds 1-4. Broad near-reference is 100% for all four; hard holdout is 100%, 100%, 100%, and 99.9000%.
5. Bake the gain into each actor's checkpointed `action_scale`: `2.0 -> 2.01`. The environment still clips torque to `[-2, 2]`.

The gain changes the actor's final scaling layer. It is not an evaluation wrapper or a policy mixture.

## Comparison

| Method | Trials | Near reference | Task success | Strict > reference | Mean return |
|---|---:|---:|---:|---:|---:|
| **Gain-calibrated targeted RL-weighted DAgger** | 12,505 | **12,500 (99.9600%)** | 11,686 (93.4506%) | **1,809 (14.4662%)** | **-138.624707** |
| Pre-gain targeted RL-weighted DAgger | 12,505 | 12,485 (99.8401%) | 11,689 (93.4746%) | 1,634 (13.0668%) | -138.760903 |
| Large SimbaV2 + corrected 100k DAgger | 12,505 | 12,473 (99.7441%) | 11,700 (93.5626%) | 1,436 (11.4834%) | -138.828336 |
| Prior RL + supervised interaction overlay | 12,505 | 12,438 (99.4642%) | 11,744 (93.9144%) | 1,149 (9.1883%) | -139.016769 |
| Normal SimbaV2 100k | 12,505 | 11,484 (91.8353%) | 11,437 (91.4594%) | 1,066 (8.5246%) | -141.638557 |
| Best pure-RL router | 12,505 | 11,674 (93.3547%) | 11,605 (92.8029%) | 1,437 (11.4914%) | -140.897560 |
| Static reference distillation, historical one seed | 2,501 | 2,469 (98.7205%) | 2,362 (94.4422%) | 288 (11.5154%) | -139.575942 |

Compared with the pre-gain actors, the calibrated actors add 15 near-reference successes, add 175 literal strict wins, improve mean return by 0.136196, and lose 3 task-success trials. Compared with the corrected-100k actors, they add 27 near-reference successes and 373 strict wins, improve mean return by 0.203629, and lose 14 task-success trials.

## Major rejected alternatives

- A new 128x4 SimbaV2 reference actor trained on 400,000 static labels plus exactly 100,000 learner DAgger states reached only 96.8013% near reference on seed 0.
- Full actor RL continuation with the best pure-RL FastSACN8/UTD-2 recipe had previously fallen to 98.4006% on the authoritative seed-0 grid.
- Hard-teacher DAgger, DP demonstration augmentation, source-policy-preserving local edits, a 128x2 student, time-conditioned students, residual adapters, and differentiable model-RL updates were all rejected because their off-grid broad or targeted rollout gate regressed. None was promoted to the five-seed audit.
- The differentiable model-RL pilot also showed unstable full-horizon gradients: its differentiable hard cost increased rather than decreased, so larger compute was not justified.

## Five remaining failures

| Seed | Theta (deg) | Theta dot | Policy return | Reference return | Regret |
|---:|---:|---:|---:|---:|---:|
| 1 | -20.655738 | -0.20 | -127.267191 | -4.382041 | 122.885150 |
| 1 | -180.000000 | 0.00 | -310.662459 | -304.326295 | 6.336164 |
| 2 | -180.000000 | 0.00 | -309.530926 | -304.326295 | 5.204631 |
| 4 | -174.098361 | -0.05 | -372.194633 | -247.642521 | 124.552113 |
| 4 | -180.000000 | 0.85 | -325.999866 | -296.026807 | 29.973059 |

The two universal `+/-126.885 deg, +/-0.85` hard failures are gone in all five seeds. The remaining failures are isolated rather than universal.

## Plots

![Comparison metrics](plots/comparison_metrics.png)

![Near-reference success over initial states](relative/near_best_known_return_eps_map.png)

![Task success over initial states](relative/task_success_map.png)

![Literal strict wins over initial states](relative/beats_best_known_return_map.png)

## Reproducibility paths

- Calibrated checkpoints: `runs/large_simba_gain1005_seed0_20260718/seed0` through `runs/large_simba_gain1005_seed4_20260718/seed4`
- Pre-gain trained actors: `runs/large_simba_targeted_failuremix_qw_seed0_20260718/seed0` through seed 4
- Exact aggregate grid: `reports/large_simba_gain1005_5seed_20260718/grid/pendulum_grid_rollouts.csv`
- Exact relative rollouts: `reports/large_simba_gain1005_5seed_20260718/relative/relative_rollouts.csv`
- Exact metrics: `reports/large_simba_gain1005_5seed_20260718/exact_metrics.csv`
- Per-seed metrics: `reports/large_simba_gain1005_5seed_20260718/per_seed_metrics.csv`
- Remaining failures: `reports/large_simba_gain1005_5seed_20260718/remaining_failures.csv`
- Targeted DAgger trainer: `scripts/train_pendulum_qregularized_dagger.py`
- Gain checkpoint builder: `scripts/calibrate_pendulum_actor_gain.py`
