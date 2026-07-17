# Large SimbaV2 + corrected 100k DAgger: five-seed result

Date: 2026-07-18

## Outcome

The primary goal is achieved. The new method is a **single actor checkpoint at inference** and reaches **12,473/12,505 = 99.7441% near-reference success**. The previous RL+supervised interaction overlay reaches 12,438/12,505 = 99.4642%.

The new actor therefore adds **35** near-reference successes. It also adds **287** literal strict return wins and improves mean return by **0.188433**. It does **not** beat the hybrid on task success: it loses **44** task-success trials.

No result below uses `>=` for the strict metric. A strict win is computed directly as:

```text
policy_return > max(DP_return, controller_return)
```

There are 0 exact return ties in the new result.

## Exact authoritative evaluation

- Environment: `Pendulum-v1`
- Horizon: 200 deterministic steps
- Initial states per seed: 61 angles in `[-pi, pi)` × 41 velocities in `[-1, 1]` = 2,501
- Follow-up seeds: 0, 1, 2, 3, 4
- Total trials: 12,505
- Near-reference success: return at least `max(DP, controller) - 5`
- Task success: at least 80% of steps near upright and no non-upright streak longer than 50 steps

| Method | Trials | Near reference | Task success | Strict return > reference | Mean return |
|---|---:|---:|---:|---:|---:|
| Large SimbaV2 + corrected 100k DAgger | 12,505 | 12,473 (99.7441%) | 11,700 (93.5626%) | 1,436 (11.4834%) | -138.828336 |
| Static distillation (1 seed) | 2,501 | 2,469 (98.7205%) | 2,362 (94.4422%) | 288 (11.5154%) | -139.575942 |
| Normal SimbaV2 100k (5 seeds) | 12,505 | 11,484 (91.8353%) | 11,437 (91.4594%) | 1,066 (8.5246%) | -141.638557 |
| Best pure RL router (5 seeds) | 12,505 | 11,674 (93.3547%) | 11,605 (92.8029%) | 1,437 (11.4914%) | -140.897560 |
| Clean small DAgger (5 seeds) | 12,505 | 12,436 (99.4482%) | 11,746 (93.9304%) | 1,156 (9.2443%) | -139.032602 |
| Prior RL+supervised interaction overlay | 12,505 | 12,438 (99.4642%) | 11,744 (93.9144%) | 1,149 (9.1883%) | -139.016769 |

The static-distillation row has one training seed and is included only as a historical comparator. The other named aggregate rows have five seeds.

## New result by follow-up seed

| Seed | Near reference | Task success | Strict return > reference |
|---:|---:|---:|---:|
| 0 | 2,496/2,501 (99.8001%) | 2,342/2,501 (93.6425%) | 296/2,501 (11.8353%) |
| 1 | 2,494/2,501 (99.7201%) | 2,341/2,501 (93.6026%) | 282/2,501 (11.2755%) |
| 2 | 2,494/2,501 (99.7201%) | 2,337/2,501 (93.4426%) | 247/2,501 (9.8760%) |
| 3 | 2,495/2,501 (99.7601%) | 2,339/2,501 (93.5226%) | 318/2,501 (12.7149%) |
| 4 | 2,494/2,501 (99.7201%) | 2,341/2,501 (93.6026%) | 293/2,501 (11.7153%) |

## What was trained

The deployed object is one SimbaV2 actor network:

- observation: `[cos(theta), sin(theta), theta_dot]`
- action: one bounded torque in `[-2, 2]`
- hidden width: 64
- residual blocks: 2
- SimbaV2 feature normalization, input shift, observation normalization, and weight projection remain part of the actor implementation
- no critic, policy gate, action router, specialist switch, or Q-search is called at inference

The initialization is `runs/reference_assisted_large_simba_reset60_dagger3_seed0_20260717/seed0/checkpoints/final.pt`. That checkpoint was trained from scratch with 400,000 static max(DP, controller) labels, 80 supervised epochs, then three learner-only DAgger rounds of 10,000 visited states each and 10 training epochs per round. Its original one-seed fine-grid result was 99.2003% near reference.

Each new follow-up seed then ran this exact additional recipe:

1. Generate a fresh 240,000-state supervised support set: 20% broad states, 50% reset-support states, 15% `120° <= |theta| <= 135°` states, and 15% near-down states.
2. Label every support state with the finite-horizon best reference, `max(DP, controller)`.
3. Use the seed-matched fixed pure-RL critics to score 41 actions in `[-2, 2]` and the reference action. Mark a state only when **every critic** estimates that the best searched action exceeds the reference action by more than `0.05` Q units.
4. Keep the supervised action target equal to the reference action (`rl_blend=0`), but give marked samples 4× weight in the supervised MSE. Thus RL changes which reference labels receive extra weight; it does not replace the reference label.
5. Run five DAgger rounds. Each round contains 100 learner episodes × 200 steps = 20,000 environment steps.
6. At every DAgger step, save the current state, execute the learned actor action in `env.step()`, and ask the best reference to label the saved state at the correct remaining horizon.
7. Aggregate all new state/reference-action pairs with the support set and apply the same critic-based weighting rule.
8. Train the full 64×2 actor with batch size 1,024, learning rate `1e-5`, and three epochs per DAgger round.
9. Select one actor iterate on the disjoint 17×11 midpoint validation grid, ordered by near-reference success, then task success, then mean return.

The experiment collects exactly 100,000 learner-visited DAgger states per seed. In all five seeds the selected policy is epoch 1 after round 1, so the returned policy was directly updated using the 240,000-state support set plus the first 20,000 DAgger states. The remaining four rounds are still part of the 100k policy sequence and validation selection, but their later actors were rejected because validation near-reference success regressed. This is policy selection from the DAgger iterate sequence, not a claim that the chosen weights consumed all 100,000 labels.

## What happened to the proposed RL cross-pollination

The best pure-RL learning recipe was tested directly before this winner was promoted:

- the corrected DAgger actor initialized the online actor;
- the pure-RL FastSACN8 critic ensemble and UTD 2 update schedule learned from real reward;
- hard-boundary replay used the `120°–135°`, `|theta_dot| <= 1` region;
- max(DP, controller) behavior cloning remained an auxiliary actor loss;
- only one actor was used at inference.

That seed-0 joint checkpoint improved held-out task success, strict wins, and mean return, but its authoritative fine-grid near-reference rate fell to 98.4006%. It was rejected. Conservative fixed-critic target blending, model-differentiated reward updates, hard-only DAgger, and critic-filtered specialist distillation were also rejected by held-out or fine-grid checks.

The winner uses the **larger SimbaV2 network backbone**, corrected DAgger, and **training-only weighting from the best pure-RL critics**. In the training command, `rl_blend=0`, so the critics never shift the reference action target. However, `selected_weight=4` means their unanimous filter gives selected reference-labeled samples four times the loss weight. The correct description is therefore **RL-weighted supervised DAgger**, not pure DAgger and not a runtime actor/critic mixture.

For the selected epoch, the number of 4×-weighted samples in the 240,000-state support set plus first 20,000-state DAgger round was: seed 0, 3,246; seed 1, 795; seed 2, 3,405; seed 3, 2,182; seed 4, 419. Their target actions remained the best-reference actions.

## Trade-offs

Compared with the previous runtime overlay:

- near reference: **+35 trials**
- literal strict wins: **+287 trials**
- mean return: **+0.188433**
- task success: **-44 trials**
- near-down near reference: **+0.8869 percentage points**
- near-down task success: **-0.5322 percentage points**

This is a clear win on the declared near-reference objective and on strict return wins, not a Pareto win over every metric.

## Plots

![Comparison metrics](plots/comparison_metrics.png)

![Near-reference success over initial states](relative/near_best_known_return_eps_map.png)

![Task success over initial states](relative/task_success_map.png)

![Strict return wins over initial states](relative/beats_best_known_return_map.png)

## Reproducibility paths

- Seed checkpoints: `runs/large_simba_corrected_dagger100k_seed0_20260718/seed0` through `runs/large_simba_corrected_dagger100k_seed4_20260718/seed4`
- Aggregate grid: `reports/large_simba_corrected_dagger100k_5seed_20260718/grid/pendulum_grid_rollouts.csv`
- Aggregate relative rollouts: `reports/large_simba_corrected_dagger100k_5seed_20260718/relative/relative_rollouts.csv`
- Exact aggregate metrics: `reports/large_simba_corrected_dagger100k_5seed_20260718/exact_metrics.csv`
- Per-seed metrics: `reports/large_simba_corrected_dagger100k_5seed_20260718/per_seed_metrics.csv`
- Training implementation used for the winner: `scripts/train_pendulum_qregularized_dagger.py`
