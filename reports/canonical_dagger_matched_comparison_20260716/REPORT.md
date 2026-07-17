# Clean reference DAgger and DAgger + SimbaV2 at a matched 100k transition budget

Date: 2026-07-17

## Result

The five-seed experiments are complete. All three main conditions use exactly 100,000 training environment transitions per seed:

- clean reference DAgger: 100,000 reference-labeled trajectory transitions;
- DAgger + SimbaV2: 50,000 reference-labeled DAgger transitions, followed by 50,000 reward-only online SimbaV2 transitions;
- standard SimbaV2: 100,000 reward-only online transitions from a randomly initialized actor.

The DAgger + SimbaV2 combination did not improve clean DAgger. It reduced near-reference success from `0.8453` to `0.7679` and task success from `0.8802` to `0.8349`. Standard SimbaV2 remained better than both matched-budget DAgger conditions.

| Method | Seeds | Reference-labeled trajectory steps per seed | Reward-only RL steps per seed | Total training environment steps per seed | Near `max(DP, controller) - 5` | Task success | Beats `max(DP, controller)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean reference DAgger | 5 | 100,000 | 0 | 100,000 | 0.8453 | 0.8802 | 0.0639 |
| Reference DAgger, then SimbaV2 | 5 | 50,000 | 50,000 | 100,000 | 0.7679 | 0.8349 | 0.0376 |
| Standard SimbaV2 | 5 | 0 | 100,000 | 100,000 | 0.9184 | 0.9146 | 0.0852 |
| Static balanced distillation | 1 | 400,000 sampled states, not trajectories | 0 | not comparable | **0.9872** | **0.9444** | 0.1152 |
| FastSACN8 UTD2 + inference-time Q-filter | 5 | 0 | 50,000 | 50,000 | 0.9291 | 0.9270 | **0.1166** |

The static-distillation row is deliberately marked as unmatched: it uses 400,000 privileged labels from a designed state distribution and has only one seed. The FastSACN row is the best currently verified five-seed pure-RL policy, but its reported policy performs a learned-critic action search at every evaluation step.

![Primary comparison](primary_metrics.png)

## Exact differences

All differences below are percentage points on the same 61 by 41 evaluation grid.

| Difference | Near reference | Task success | Beats reference |
|---|---:|---:|---:|
| Clean DAgger minus standard SimbaV2 | -7.30 | -3.44 | -2.14 |
| DAgger 50k + SimbaV2 50k minus standard SimbaV2 | -15.05 | -7.96 | -4.77 |
| DAgger 50k + SimbaV2 50k minus clean DAgger 100k | -7.75 | -4.53 | -2.63 |

This is an equal-environment-transition comparison, not an equal-information or equal-compute comparison. Every DAgger transition receives a privileged reference action label, whereas SimbaV2 sees only environment reward. DAgger also makes multiple supervised passes over its accumulated dataset.

## Clean DAgger: exact implementation

This run follows the defining DAgger data flow. At time `t`:

1. Read the current observation `s_t` before calling `env.step`.
2. Compute the learned actor action `a_actor(s_t)`.
3. Query the horizon-aware `best` reference at the same `s_t`. The reference evaluates the DP and energy-controller choices for the remaining episode horizon and returns the action from the higher-return reference.
4. Draw one Bernoulli mixture decision. Execute the reference action with probability `beta`; otherwise execute the learned actor action.
5. Call `env.step(a_executed)`. The executed action therefore determines `s_(t+1)` and the state distribution visited by later steps.
6. Save `(s_t, a_reference(s_t))`, regardless of whether the reference or actor action was executed.
7. Append every new pair to the cumulative dataset and retrain the actor on the entire accumulated dataset.

There is no replay-buffer Bellman loss, learned critic, reward objective, entropy objective, Q-filter, or SAC update in clean DAgger. The actor is supervised only with reference actions.

Each clean-DAgger seed uses this fixed schedule:

| Phase | Controller used for visitation | Episodes | Steps per episode | New labeled pairs | Cumulative pairs | Supervised epochs |
|---|---|---:|---:|---:|---:|---:|
| Initial collection | reference only | 100 | 200 | 20,000 | 20,000 | 40 |
| DAgger round 1 | reference/actor mixture, `beta=0.75` | 100 | 200 | 20,000 | 40,000 | 20 |
| DAgger round 2 | reference/actor mixture, `beta=0.50` | 100 | 200 | 20,000 | 60,000 | 20 |
| DAgger round 3 | reference/actor mixture, `beta=0.25` | 100 | 200 | 20,000 | 80,000 | 20 |
| DAgger round 4 | actor only, `beta=0.00` | 100 | 200 | 20,000 | 100,000 | 20 |

This produces exactly 100,000 training environment transitions, 100,000 saved current states, and 100,000 reference-label queries per seed. It performs 120 supervised epochs in total. With cumulative sizes and batch size 1,024, that is 6,320 actor optimizer steps per seed.

The actor optimizer is AdamW with learning rate `3e-4` and zero weight decay. The action regression loss is Smooth L1 with `beta=0.05` on action error divided by Pendulum's action scale of 2. Gradients are clipped to norm 10, then SimbaV2 weight projection is applied. The actor after the final epoch is used (`selection_metric=last`); evaluation-grid performance is not used to select an epoch.

The actor network is the compact SimbaV2 actor already in this repository: one residual actor block, hidden dimension 32, expansion factor 4, feature normalization, input shift, observation normalization, and weight projection. Reusing this actor architecture does not turn DAgger into RL: no SimbaV2 critic or RL loss exists during the clean-DAgger phase.

The measured fraction of steps actually controlled by the reference confirms the requested mixture schedule:

| Seed | Round 1, target 0.75 | Round 2, target 0.50 | Round 3, target 0.25 | Round 4, target 0.00 |
|---:|---:|---:|---:|---:|
| 0 | 0.7490 | 0.4979 | 0.2485 | 0.0000 |
| 1 | 0.7541 | 0.4963 | 0.2489 | 0.0000 |
| 2 | 0.7468 | 0.4979 | 0.2467 | 0.0000 |
| 3 | 0.7510 | 0.4964 | 0.2514 | 0.0000 |
| 4 | 0.7482 | 0.5002 | 0.2488 | 0.0000 |

The collector batches independent Pendulum trajectories for speed but applies the exact Gym Pendulum transition equations. The automated equivalence test compares the batched and Gym collectors state by state: maximum observation difference `1.55e-6`, maximum reference-action difference `8.94e-6`, and absolute episode-return difference `6.69e-7` in the tested deterministic trajectory.

## DAgger 50k + SimbaV2 50k: exact implementation

### First 50,000 transitions: DAgger

The hybrid's DAgger half uses the same algorithm, actor, loss, four-round beta schedule, batch size, learning rate, and last-epoch selection as clean DAgger. Only the collection count is halved:

| Phase | Episodes | New pairs | Cumulative pairs | Supervised epochs |
|---|---:|---:|---:|---:|
| Initial reference collection | 50 | 10,000 | 10,000 | 40 |
| DAgger round 1, `beta=0.75` | 50 | 10,000 | 20,000 | 20 |
| DAgger round 2, `beta=0.50` | 50 | 10,000 | 30,000 | 20 |
| DAgger round 3, `beta=0.25` | 50 | 10,000 | 40,000 | 20 |
| DAgger round 4, `beta=0.00` | 50 | 10,000 | 50,000 | 20 |

This half uses exactly 50,000 training transitions and 50,000 reference queries. It performs 3,180 supervised actor optimizer steps per seed.

### Second 50,000 transitions: online SimbaV2

For each seed, the matching 50k DAgger actor and its observation-normalization statistics initialize the SimbaV2 actor. The following objects start fresh: both critics, target critics, critic optimizer, actor optimizer, entropy-temperature parameter and optimizer, and the 100,000-transition replay buffer.

The online phase then uses exactly 50,000 new environment transitions and ordinary environment reward. Its locked settings are:

- learning starts after 1,000 environment steps;
- batch size 256 and replay capacity 100,000;
- UTD 1: one critic update batch per environment step after learning starts;
- actor update trigger every two critic updates; each trigger performs two actor optimizer steps;
- actor updates disabled for critic update steps 1 through 3,999; the first actor trigger is critic update 4,000, which occurs at environment step 5,000 because updates start at environment step 1,001;
- actor and critic learning-rate schedules are configured from `1e-4` to `5e-5`; with 49,000 update steps in this phase, the recorded values run from `9.9999e-5` to `5.1e-5`;
- discount `0.99`, target-network coefficient `0.005`, initial entropy temperature `0.01`, and policy target-entropy scale `-0.5`;
- twin 51-bin distributional critics with hidden dimension 64 and two residual blocks;
- the same SimbaV2 observation normalization, feature normalization, input shift, reward scaling, and weight projection as the standard baseline.

The online phase uses no DP/controller actions, reference labels, reference replay, behavior-cloning loss, CQL, Q-filter, model-generated replay, or expert-action mixture. Each seed ends with `run_complete` at environment step 50,000 and records 49,000 critic update triggers. Triggers 4,000, 4,002, ..., 49,000 each execute two actor optimizer steps, for exactly 45,002 actor optimizer steps per seed.

The measured degradation establishes that this specific continuation is harmful. It does not establish a unique cause. Fresh-critic initialization, entropy-regularized actor updates, the 4,000-update actor threshold, and the 50/50 budget split were not independently ablated, so assigning the loss to one of them would go beyond the experiment.

## Standard SimbaV2 comparator

The normal SimbaV2 comparator is the existing five-seed `simba_full_official_opt` run. Every seed starts from randomly initialized actor and critic networks and trains for 100,000 online environment steps. It uses learning start 1,000, batch 256, replay capacity 100,000, UTD 1, policy frequency 2, actor/critic learning-rate decay from `1e-4` to `5e-5`, twin 51-bin distributional critics, observation and feature normalization, input shift, reward scaling, and weight projection. It uses no DP/controller labels.

## Static distillation comparator

The normal static comparator is `distill_best_simbav2_balanced_400k_20260701`, seed 0. It samples and labels all 400,000 training states before actor training. The learned actor never chooses later training states, so its training distribution does not move toward the actor's own visited states.

It uses 400,000 training labels, 40,000 fixed action-fit evaluation labels, batch size 1,024, learning rate `3e-4`, and 160 epochs. The checkpoint selects the best fixed-evaluation action-MAE epoch. Its near-reference result is `0.9872` and task success is `0.9444`, but this is one seed with four times the privileged-label budget of clean DAgger.

The older result called DAgger in prior reports is also not this clean experiment. It started from a 240,000-state balanced static dataset and added two 10,000-state actor-rollout aggregation rounds, for 260,000 labels across three seeds. Its strong `0.9928` near-reference result demonstrates the value of broad static coverage plus a small policy-state correction; it does not answer whether clean 100k trajectory-only DAgger beats 100k SimbaV2.

## Best verified five-seed pure-RL method

The selected pure-RL result is FastSACN8 UTD2 with inference-time Q-filtered action search. It retains the SimbaV2 actor and critic building blocks. Training uses 50,000 real environment steps, learning start 1,000, batch size 256, replay capacity 100,000, and two gradient-update batches per environment step.

FastSACN uses multi-step critic targets with a maximum horizon of eight transitions, the `fast_last` target set, and horizon weight `lambda=0.5`. From environment step 10,000, `0.02` of each replay batch is forced to come from real replay-buffer transitions satisfying `120 degrees <= abs(theta) <= 135 degrees` and `abs(theta_dot) <= 1`; this fraction decays to `0.001` over 20,000 steps. It is still pure RL: those are previously observed environment transitions, not DP/controller or model-generated data.

At every evaluation state, its policy performs this exact Q-filter:

1. Compute the deterministic actor action.
2. Evaluate 41 evenly spaced candidate actions in `[-2, 2]`.
3. Score every candidate with `min(Q1, Q2)`.
4. Select the highest-scoring candidate as `a_search`.
5. Execute `a_search` only if its clipped-double-Q score exceeds the actor action's score by more than `0.005`.
6. Otherwise execute the actor action.

Therefore, `0.9291` near-reference and `0.9270` task success are results for the actor plus the critics plus inference-time search, not for the actor network alone. The same checkpoints' actor-only result is `0.9028` near-reference and `0.9108` task success.

Other pure-RL attempts are retained for context:

| Pure-RL experiment | Seeds | Near reference | Task success | Beats reference | Status |
|---|---:|---:|---:|---:|---|
| Standard SimbaV2, 100k | 5 | 0.9184 | 0.9146 | 0.0852 | stable baseline |
| FastSACN8 UTD2 actor only, 50k | 5 | 0.9028 | 0.9108 | 0.1126 | below standard SimbaV2 on the two primary metrics |
| FastSACN8 UTD2 + Q-filter, 50k | 5 | **0.9291** | **0.9270** | 0.1166 | selected five-seed pure-RL result |
| FastSACN8 UTD2 + unanimous two-critic filter, 50k | 5 | 0.9273 | 0.9251 | 0.1166 | slightly lower overall; stronger defined hard-region near-reference |
| Critic-guided actor extraction, UTD4 | 3 | 0.9380 | 0.9188 | 0.1751 | promising but only three seeds and lower task success |
| Same actor extraction, UTD2 | 5 | 0.8964 | 0.9000 | 0.1172 | rejected after seed failures |

The three-seed UTD4 actor extraction has a larger near-reference mean than the Q-filter (`0.9380` versus `0.9291`) but is not the verified leader: it has fewer seeds, lower task success, and its five-seed UTD2 replication failed.

## Evaluation definition

Every five-seed condition is evaluated deterministically from the same 61 angle values and 41 angular velocities in `[-1, 1]`: 2,501 initial-state cells per seed and 12,505 rollouts per condition. Every rollout lasts 200 steps.

- Near-reference success: policy return is at least `max(DP return, controller return) - 5` for that exact initial state.
- Beats reference: policy return is at least `max(DP return, controller return)`.
- Near-upright state: `cos(theta) >= 0.95` and `abs(theta_dot) <= 1`.
- Task success: at least 80% of the rollout is near upright and no consecutive not-near-upright streak exceeds 50 steps.
- Near-down region: `abs(theta) >= 150 degrees` within the evaluated velocity range.

## Per-seed results and seed-level uncertainty

The intervals are 95% t-intervals over the five training-seed rates, not binomial intervals over the 12,505 correlated grid rollouts.

| Method | Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | 95% seed CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean DAgger 100k | Near reference | 0.8429 | 0.8581 | 0.8333 | 0.8441 | 0.8485 | 0.8453 | [0.8342, 0.8565] |
| Clean DAgger 100k | Task success | 0.8689 | 0.8689 | 0.8872 | 0.8888 | 0.8872 | 0.8802 | [0.8673, 0.8931] |
| DAgger 50k + SimbaV2 50k | Near reference | 0.8289 | 0.7525 | 0.8285 | 0.6629 | 0.7665 | 0.7679 | [0.6831, 0.8526] |
| DAgger 50k + SimbaV2 50k | Task success | 0.8669 | 0.8565 | 0.8209 | 0.8397 | 0.7909 | 0.8349 | [0.7975, 0.8724] |
| Standard SimbaV2 100k | Near reference | 0.9148 | 0.9512 | 0.9088 | 0.9096 | 0.9072 | 0.9184 | [0.8953, 0.9414] |
| Standard SimbaV2 100k | Task success | 0.8992 | 0.9308 | 0.9132 | 0.9036 | 0.9260 | 0.9146 | [0.8976, 0.9316] |

The hybrid is worse than clean DAgger in every seed on near-reference success. It is also worse in seeds 2-4 on task success and nearly unchanged in seeds 0-1. Seed 3 is the largest hybrid failure, at `0.6629` near-reference success.

![Per-seed primary metrics](per_seed_primary_metrics.png)

## Near-down results

| Method | Near-reference, `abs(theta) >= 150 deg` | Task success, `abs(theta) >= 150 deg` |
|---|---:|---:|
| Clean DAgger 100k | 0.4723 | 0.4359 |
| DAgger 50k + SimbaV2 50k | 0.3290 | 0.4914 |
| Standard SimbaV2 100k | **0.7339** | **0.6945** |

The hybrid raises near-down task stability relative to clean DAgger by 5.54 points but lowers near-down reference-return success by 14.32 points. Standard SimbaV2 is higher on both.

## Initial-state heatmaps: near-reference success

Every heatmap cell is the success fraction over the method's training seeds. The static-distillation map is not shown here because it has one training seed; its cells would be binary.

| Clean DAgger 100k | DAgger 50k + SimbaV2 50k | Standard SimbaV2 100k |
|---|---|---|
| ![Clean DAgger near-reference heatmap](../canonical_reference_dagger_100k_5seed_20260716/relative/near_best_known_return_eps_map.png) | ![DAgger plus SimbaV2 near-reference heatmap](../canonical_reference_dagger50k_simbav250k_5seed_20260716/relative/near_best_known_return_eps_map.png) | ![Standard SimbaV2 near-reference heatmap](../week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/near_best_known_return_eps_map.png) |

## Initial-state heatmaps: task success

| Clean DAgger 100k | DAgger 50k + SimbaV2 50k | Standard SimbaV2 100k |
|---|---|---|
| ![Clean DAgger task heatmap](../canonical_reference_dagger_100k_5seed_20260716/relative/task_success_map.png) | ![DAgger plus SimbaV2 task heatmap](../canonical_reference_dagger50k_simbav250k_5seed_20260716/relative/task_success_map.png) | ![Standard SimbaV2 task heatmap](../week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/task_success_map.png) |

## Accepted and excluded artifacts

Accepted runs contain five complete seeds and final checkpoints:

- clean DAgger: `runs/canonical_reference_dagger_100k_5seed_20260716/seed{0,1,2,3,4}`;
- 50k DAgger initializations: `runs/canonical_reference_dagger50k_for_simbav2_5seed_20260716/seed{0,1,2,3,4}`;
- 50k SimbaV2 continuations: `runs/canonical_reference_dagger50k_simbav250k_5seed_20260716/seed{0,1,2,3,4}`.

The earlier DAgger 100k + SimbaV2 100k run is excluded from every matched-budget number because it uses 200,000 total training transitions per seed. The interrupted 25k pilot is also excluded.

Machine-readable evaluations:

- clean DAgger: `reports/canonical_reference_dagger_100k_5seed_20260716/relative/relative_summary.json`;
- matched hybrid: `reports/canonical_reference_dagger50k_simbav250k_5seed_20260716/relative/relative_summary.json`;
- standard SimbaV2: `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_summary.json`;
- pure-RL Q-filter: `reports/fastsacn_qfiltered_5seed_20260710/summary.json`;
- consolidated values: `reports/canonical_dagger_matched_comparison_20260716/comparison_metrics.json`.
