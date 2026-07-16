# Exact DAgger Distillation Comparison on Pendulum

Date: 2026-07-16

## Decision summary

The DAgger experiment is `distill_best_simbav2_dagger_2iter_static240k_20260702`. It performs exactly two policy-rollout/data-aggregation rounds. Each round collects 50 deterministic episodes of 200 steps, obtains 10,000 visited states, labels those states with the horizon-aware `best` reference, appends the 10,000 labeled examples to the training set, and trains the actor for five full epochs. The two rounds therefore add exactly 20,000 policy-visited states and perform exactly ten actor-training epochs. The training set grows from 240,000 to 260,000 examples.

On the matched 61 x 41 initial-state grid, the three-seed DAgger policy has the highest near-reference result in this comparison: `0.9928`. Static balanced distillation has slightly higher task success: `0.9444` versus DAgger's `0.9412`, but that static result is from only one training seed. The strongest currently verified five-seed method that uses no DP/controller data is FastSACN8 UTD2 with inference-time Q-filtered critic search: `0.9291` near-reference and `0.9270` task success.

| Method | Training type | Training seeds | Environment steps per seed | Near reference | Task success | Beats reference |
|---|---|---:|---:|---:|---:|---:|
| DAgger distillation, 2 rounds | Reference-supervised, policy-state distribution | 3 | 20,000 policy rollout steps after initialization | **0.9928** | 0.9412 | 0.1037 |
| Static balanced distillation | Reference-supervised, fixed state distribution | 1 | 0 | 0.9872 | **0.9444** | 0.1152 |
| Standard SimbaV2 | Pure online RL | 5 | 100,000 | 0.9184 | 0.9146 | 0.0852 |
| FastSACN8 UTD2 + Q-filtered search | Pure online RL plus inference-time critic search | 5 | 50,000 | 0.9291 | 0.9270 | **0.1166** |

`Environment steps` does not make the supervised and RL rows cost-equivalent. DAgger and static distillation use the privileged DP/controller reference. Standard SimbaV2 and Q-filtered FastSACN do not.

![Matched headline metrics](headline_metrics.png)

## Evaluation protocol

Every number and heatmap in the decision table uses the same deterministic reset grid and the same definitions:

- Environment: `Pendulum-v1` with a 200-step horizon.
- Initial angle grid: 61 values spanning `[-180 degrees, 180 degrees)`.
- Initial angular-velocity grid: 41 values spanning `[-1, 1]`.
- Total initial states: `61 x 41 = 2,501`.
- Each checkpoint is evaluated once from every exact initial state. Pendulum dynamics and the evaluated policies are deterministic.
- For a multi-seed method, the reported grid rate is the mean of the per-seed binary result over all `2,501 x number_of_seeds` rollouts.

The metrics are:

1. **Near reference.** For initial state `s`, let `R_policy(s)` be the 200-step policy return. Let `R_DP(s)` and `R_controller(s)` be the corresponding reference returns. The state succeeds when

   `R_policy(s) >= max(R_DP(s), R_controller(s)) - 5`.

2. **Task success.** A post-transition state is near upright when `cos(theta) >= 0.95` and `abs(theta_dot) <= 1`. A rollout succeeds only when at least `80%` of its 200 post-transition states are near upright and its longest consecutive not-near-upright streak is at most `50` steps. Both conditions must hold.

3. **Beats reference.** The state succeeds when

   `R_policy(s) >= max(R_DP(s), R_controller(s))`.

Near reference and task success are the primary metrics. `Beats reference` is secondary because the DP solution is discretized and the controller is a hand-designed policy; a learned policy can exceed either approximation without changing the task-success criterion.

The heatmap color is the fraction of training seeds that succeed at an initial state. Consequently, DAgger colors occur in increments of `1/3`, the two five-seed RL maps in increments of `1/5`, and the single-seed static-distillation map is binary. The static map is not a multi-seed reliability estimate.

## The DAgger experiment, exactly

### Reference label used at each visited state

The actor does not train on its own rollout action. At a visited observation `[cos(theta), sin(theta), theta_dot]`, `PendulumReferenceGuidance(policy="best")` computes:

1. the finite-horizon DP action and DP value for the number of episode steps remaining;
2. the energy-swing-up/controller action and the model-simulated controller return for the same remaining horizon;
3. the controller action if its predicted return is strictly greater than the DP value, otherwise the DP action.

That selected action is the supervised target. This is what `best` means in the training code. It is not an average of the two actions.

### Initialization and data

All three DAgger seeds load the same actor checkpoint:

`runs/distill_best_simbav2_balanced_400k_20260701/seed0/checkpoints/final.pt`

That actor was produced by static balanced distillation. Each DAgger seed then generates a separate static pool of 240,000 reference-labeled states and a separate fixed evaluation pool of 30,000 reference-labeled states. The DAgger run performs no training epoch on the 240,000-state pool before the first policy rollout; the loaded actor is evaluated at epoch zero and immediately used for DAgger round 1.

The saved run metadata does not persist the static sampler's mixture flags. Therefore this report states the recorded pool sizes and does not invent unrecorded near-down or near-upright sampling fractions. The DAgger rollout counts, rollout mode, episode length, iteration count, training epochs, learning rate, batch size, initialization, and final dataset size are all persisted.

### Exact two-round loop

For each training seed, the loop is:

1. Reset `Pendulum-v1` for a 200-step episode.
2. At the current observation `s_t`, compute the deterministic learned-policy action `a_policy`.
3. Save `s_t` and the number of steps remaining.
4. Execute `env.step(a_policy)`. The learned policy, not the reference, determines `s_(t+1)`.
5. Repeat steps 2-4 for all 200 steps and for all 50 episodes.
6. After collection, label every saved `s_t` with the `best` reference action for the saved remaining horizon.
7. Append all 10,000 `(s_t, a_reference)` pairs to the accumulated dataset.
8. Shuffle the entire accumulated dataset and train for five complete epochs using batches of 1,024.
9. Repeat steps 1-8 once.

The actor loss is Smooth L1 on `(predicted_action - reference_action) / 2`, with `beta=0.05`. Gradients are clipped to norm `10`. SimbaV2 weight projection is applied after optimizer updates. The optimizer learning rate is `3e-5`. The final actor after round 2 is selected (`selection_metric="last"`); there is no best-validation-epoch rollback.

| Quantity | Before round 1 | After round 1 | After round 2 |
|---|---:|---:|---:|
| Static reference-labeled examples | 240,000 | 240,000 | 240,000 |
| Policy-visited, reference-labeled examples added | 0 | 10,000 | 20,000 |
| Total accumulated examples | 240,000 | 250,000 | 260,000 |
| Actor-training epochs completed in this run | 0 | 5 | 10 |

Each seed's collection diagnostics are:

| Seed | DAgger round | Episodes | Samples | Mean episode length | Mean policy-rollout return | Mean absolute policy/reference action difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 50 | 10,000 | 200 | -125.7255 | 0.06182 |
| 0 | 2 | 50 | 10,000 | 200 | -167.0752 | 0.05773 |
| 1 | 1 | 50 | 10,000 | 200 | -166.9875 | 0.06457 |
| 1 | 2 | 50 | 10,000 | 200 | -135.7819 | 0.07268 |
| 2 | 1 | 50 | 10,000 | 200 | -137.7526 | 0.07431 |
| 2 | 2 | 50 | 10,000 | 200 | -124.8797 | 0.06461 |

The action-difference column is diagnostic only. The learned-policy actions determine visitation; the reference actions are the only supervised targets.

## Comparator definitions

### Static balanced distillation

The normal/static comparator is `distill_best_simbav2_balanced_400k_20260701`, seed 0. It creates all reference-labeled states before actor training and never rolls the learned actor out to choose new training states. It uses 400,000 training examples, 40,000 fixed action-fit evaluation examples, batch size 1,024, learning rate `3e-4`, and 160 training epochs. Its best fixed-set action MAE is `0.09106` at epoch 150, and the saved final checkpoint uses that best actor. Only one training seed was run, so comparisons against its `0.9444` task success cannot establish a seed-level statistical difference.

### Standard SimbaV2

The normal RL comparator is the five-seed `simba_full_official_opt` 100k aggregate, seeds 0-4. Each seed uses 100,000 online environment steps, a 100,000-transition replay buffer, learning start at step 1,000, batch size 256, one update per environment step, policy frequency 2, and actor/critic learning rates linearly decayed from `1e-4` to `5e-5`. It uses the SimbaV2 actor and distributional critic configuration with observation normalization, feature normalization, input shift, reward scaling, and weight projection. It uses no DP/controller action labels.

### Current best verified pure-RL version

The selected pure-RL result is the five-seed FastSACN8 UTD2 checkpoint family with Q-filtered action search at inference.

Training is 50,000 online environment steps per seed, learning start at 1,000, batch size 256, replay capacity 100,000, and two gradient-update batches per environment step. The critic uses FastSACN targets with an 8-step maximum horizon, the `fast_last` target set, and horizon weight `lambda=0.5`. From step 10,000, a fraction of each replay batch is forced to come from real replay-buffer states satisfying `120 degrees <= abs(theta) <= 135 degrees` and `abs(theta_dot) <= 1`. That fraction decays linearly from `0.02` to `0.001` over 20,000 steps. These are real environment transitions; no model-generated or reference transitions are used.

At every inference state:

1. Compute the deterministic actor action.
2. Evaluate 41 evenly spaced candidate actions over the full Pendulum action interval `[-2, 2]`.
3. Score each candidate with `min(Q1, Q2)` from the learned critics.
4. Let `a_search` be the highest-scoring candidate.
5. Execute `a_search` only when `min(Q1,Q2)(s,a_search) - min(Q1,Q2)(s,a_actor) > 0.005`.
6. Otherwise execute the original actor action.

The reported result therefore includes online critic search during evaluation. It is not the performance of the actor network alone. It remains a pure-RL method under the project's definition because it uses only environment reward, environment transitions, actor outputs, and learned critics. It uses no DP actions, controller actions, reference replay, distilled initialization, behavior cloning, CQL, or supervised reference loss.

## Results and exact differences

The three-seed DAgger averages are composed of:

| DAgger seed | Near reference | Task success | Beats reference |
|---:|---:|---:|---:|
| 0 | 0.9936 | 0.9408 | 0.1036 |
| 1 | 0.9932 | 0.9408 | 0.1248 |
| 2 | 0.9916 | 0.9420 | 0.0828 |
| Mean | **0.9928** | **0.9412** | **0.1037** |

DAgger minus each comparator is:

| Comparator | Near-reference difference | Task-success difference | Beats-reference difference |
|---|---:|---:|---:|
| Static balanced distillation | +0.0056 | -0.0032 | -0.0115 |
| Standard SimbaV2 | +0.0745 | +0.0266 | +0.0184 |
| Q-filtered FastSACN8 | +0.0637 | +0.0142 | -0.0129 |

The direct pure-RL comparison is also positive for Q-filtered FastSACN versus standard SimbaV2: `+0.0108` near reference, `+0.0124` task success, and `+0.0313` beats reference. Task success improves in all five matched seeds and near-reference improves in four of five. The saved paired-seed 95% t-intervals are `[-0.0014, 0.0230]` for near reference and `[-0.0007, 0.0254]` for task success. Both intervals include zero; this is a five-seed mean win, not a conventional 95%-confidence significance result.

## Initial-state heatmaps

### Near-reference success

![Near-reference success heatmaps](near_reference_success_heatmaps.png)

The DAgger and static policies fail near reference in only narrow boundary cells. Standard SimbaV2 and Q-filtered FastSACN have broader failures around the left/right angle wrap boundary and several narrow angle bands. Q-filtered FastSACN improves the five-seed mean from `0.9184` to `0.9291`, but it does not approach the supervised policies' `0.9872-0.9928` range.

### Task success

![Task-success heatmaps](task_success_heatmaps.png)

All methods have a difficult band at the `-180/180` angle wrap boundary. The supervised maps have fewer isolated interior failures. Static distillation's mean is `0.0032` higher than DAgger's, but the maps have different seed granularity: the static map is one binary checkpoint, while DAgger is the fraction over three separately trained DAgger policies.

## Pure-RL experiments tried

The selected Q-filtered method was not the only pure-RL attempt.

| Pure-RL experiment | Seeds | Near reference | Task success | Beats reference | Decision |
|---|---:|---:|---:|---:|---|
| Standard SimbaV2, 100k | 5 | 0.9184 | 0.9146 | 0.0852 | Stable baseline |
| FastSACN8 UTD2 actor alone, 50k | 5 | 0.9028 | 0.9108 | 0.1126 | Actor alone is below standard SimbaV2 on the two primary metrics |
| FastSACN8 UTD2 + Q-filtered search, 50k | 5 | **0.9291** | **0.9270** | 0.1166 | Selected five-seed pure-RL result |
| FastSACN8 UTD2 + unanimous two-critic filter, 50k | 5 | 0.9273 | 0.9251 | 0.1166 | Better on the defined hard region, slightly worse overall |
| Critic-guided actor-only update, UTD4 | 3 | 0.9380 | 0.9188 | 0.1751 | Promising near-reference result, but only three seeds and lower task success than Q-filtered inference |
| Same critic-guided actor update, UTD2 | 5 | 0.8964 | 0.9000 | 0.1172 | Rejected; failed on seeds 2 and 4 |
| One-step model replay, policy actions | 1 | 0.7317 | 0.8824 | 0.0224 | Rejected |
| One-step model replay, random actions | 1 | 0.7253 | 0.8061 | 0.0212 | Rejected |

The critic-guided actor-only attempt follows the current actor, labels its visited states with the 41-action clipped-double-Q search, and trains the same-size actor on those critic-selected actions. The small UTD4 update used one aggregation iteration, 1,024 trajectories, two epochs, and learning rate `2e-5`. It is DAgger-shaped but contains no external reference. The identical five-seed UTD2 version regressed, so the actor-only extraction is not the selected result.

The one-step model-replay attempt used the known Pendulum transition model but no DP/controller actions. Starting at environment step 8,000, it generated eight one-step transitions per real step from states with `150 degrees <= abs(theta) <= 180 degrees` and `abs(theta_dot) <= 1`; model data supplied 25% of the training batch. One version used the current policy action and one used uniformly random actions. Neither propagated enough long-horizon value to solve the hard states. The policy-action version reached only `0.7317` near reference, while the random-action version also reduced task success to `0.8061`.

## What the evidence supports

- Two DAgger rounds produce the strongest measured near-reference performance here: `0.9928` across three training seeds.
- DAgger does not beat the single static-distillation checkpoint on task success: `0.9412` versus `0.9444`.
- DAgger clearly exceeds both five-seed pure-RL methods on the measured means, but it uses privileged reference labels and is therefore a different training regime.
- Q-filtered FastSACN8 is the current best verified five-seed pure-RL result on the combination of near-reference and task success. Its selected policy includes inference-time critic search.
- The three-seed UTD4 actor-only extraction has a higher near-reference mean than Q-filtered FastSACN8 (`0.9380` versus `0.9291`) but lower task success, fewer seeds, and a failed five-seed UTD2 replication. It is not promoted as the current verified leader.
- No claim of seed-level superiority over static distillation is possible because the static comparator has one seed.

## Reproducibility and sources

Plot construction and recomputed metrics:

- `reports/dagger_exact_comparison_20260716/build_plots.py`
- `reports/dagger_exact_comparison_20260716/computed_metrics.json`

DAgger implementation and run records:

- `src/last_nine_rl/distill_reference.py`, especially `collect_dagger_dataset`
- `src/last_nine_rl/reference_guidance.py`, especially `PendulumReferenceGuidance.act_batch`
- `runs/distill_best_simbav2_dagger_2iter_static240k_20260702/seed{0,1,2}`
- `reports/distill_best_simbav2_dagger_2iter_static240k_20260702/relative_success_vellim1_3seed`

Comparators:

- `runs/distill_best_simbav2_balanced_400k_20260701/seed0`
- `reports/distill_best_simbav2_balanced_400k_20260701/relative_success_vellim1`
- `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt`
- `reports/fastsacn_qfiltered_5seed_20260710/REPORT.md`
- `reports/fastsacn_qfiltered_5seed_20260710/summary.json`

