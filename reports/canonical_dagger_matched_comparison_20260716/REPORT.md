# Correct reference DAgger at 100k, with and without 100k SimbaV2 continuation

## Result

The correctly implemented, trajectory-only reference DAgger experiment did **not** beat the existing five-seed SimbaV2 run. Adding a full 100,000-step standard SimbaV2 phase after DAgger improved the DAgger policy substantially, but it still did not catch SimbaV2 trained from scratch.

| Method | Seeds | Reference-labeled environment steps per seed | Reward-only RL steps per seed | Near `max(DP, controller) - 5` | Task success | Beats `max(DP, controller)` |
|---|---:|---:|---:|---:|---:|---:|
| Clean reference DAgger | 5 | 100,000 | 0 | 0.8093 | 0.8643 | 0.0541 |
| Clean reference DAgger, then standard SimbaV2 | 5 | 100,000 | 100,000 | 0.8808 | 0.8989 | 0.0675 |
| Existing standard SimbaV2 | 5 | 0 | 100,000 | **0.9184** | **0.9146** | **0.0852** |
| FastSACN8 UTD2 with inference-time Q-filter | 5 | 0 | 50,000 | **0.9291** | **0.9270** | **0.1166** |

The standard SimbaV2 row is the direct 100k/five-seed RL comparison requested. Relative to it, clean DAgger is lower by 10.91 percentage points on near-reference success and 5.03 points on task success. DAgger followed by SimbaV2 is lower by 3.76 and 1.57 points, respectively.

The combination is not interaction-matched to standard SimbaV2: it uses 100k privileged, reference-labeled interactions **plus** 100k ordinary RL interactions. The table separates those budgets so this cannot be mistaken for an equal-data claim.

![Primary five-seed comparison](primary_metrics.png)

## What "clean DAgger" means in this run

Each seed starts from a randomly initialized actor. The actor network uses the same small SimbaV2 actor backbone already implemented in this repository, because DAgger still needs a function approximator. "DAgger-only" means that this network is trained only by supervised reference-action regression: it has no critic, reward loss, replay-buffer Bellman targets, entropy objective, or SAC updates.

Each seed executes exactly this fixed protocol:

1. Collect 100 reference-controlled episodes of 200 steps: 20,000 visited states.
2. Label each state with the horizon-aware `best` reference action.
3. Train the actor for 30 complete epochs on those 20,000 pairs.
4. Run four DAgger rounds. Every round collects 100 episodes of 200 steps, or 20,000 new visited states.
5. In rounds 1-4, choose the action sent to the dynamics independently at every state from the reference with probability 0.75, 0.50, 0.25, and 0.00. Otherwise execute the learned actor action.
6. Regardless of which action is executed, save the visited state and the reference action label for that state.
7. Append all 20,000 new pairs to the cumulative dataset, then train for 10 complete epochs on the entire cumulative set.
8. Use the actor after the fourth round. There is no test-grid-based checkpoint selection.

The cumulative supervised dataset sizes are therefore exactly 20k, 40k, 60k, 80k, and 100k. The four aggregation rounds add 80k policy/reference-mixture-visited states to the initial 20k expert-visited states. Batch size is 1,024, actor learning rate is `3e-4`, and the loss is smooth L1 on action error normalized by the Pendulum action scale.

The reference is queried at every saved state. It calculates the DP and energy-controller choices for the remaining episode horizon and labels the state with the action belonging to the higher-return reference. It is not a fixed precomputed action table and it does not label only the initial states.

The requested mixture probabilities were realized correctly in every seed:

| Seed | Round 1, beta 0.75 | Round 2, beta 0.50 | Round 3, beta 0.25 | Round 4, beta 0.00 |
|---:|---:|---:|---:|---:|
| 0 | 0.7490 | 0.4979 | 0.2485 | 0.0000 |
| 1 | 0.7541 | 0.4963 | 0.2489 | 0.0000 |
| 2 | 0.7468 | 0.4979 | 0.2467 | 0.0000 |
| 3 | 0.7510 | 0.4964 | 0.2514 | 0.0000 |
| 4 | 0.7482 | 0.5002 | 0.2488 | 0.0000 |

The collector batches independent Pendulum trajectories for speed, but uses the exact Gym Pendulum transition equations. Unit tests compare the batched and Gym collectors state by state for deterministic trajectories; states, labels, and returns agree. This is an execution optimization, not a model-based training shortcut.

## What "DAgger then SimbaV2" means

For each seed, the final DAgger actor and its observation-normalization statistics initialize the matching SimbaV2 seed. Everything else starts fresh:

- fresh two-critic SimbaV2 critic network;
- fresh target critic;
- fresh 100k replay buffer;
- fresh entropy-temperature parameter and optimizers;
- ordinary environment rewards only;
- no reference actions, reference loss, Q-filter, CQL, replay injection, or expert-action mixture during RL.

The online phase then exactly matches the standard SimbaV2 100k configuration: 100,000 environment steps, learning begins at step 1,000, batch 256, UTD 1, actor update every two steps, replay capacity 100,000, actor and critic learning rates linearly decaying from `1e-4` to `5e-5`, distributional twin critics with 51 bins, SimbaV2 observation normalization, reward scaling, and weight projection. Seeds 0-4 all end with a `run_complete` event at step 100,000.

Thus the combination tests whether a DAgger actor is a useful initialization for otherwise normal SimbaV2. It does **not** test an equal-total-interaction hybrid.

## Evaluation definition

Every condition is evaluated deterministically from the same 61 angle values and 41 angular velocities in `[-1, 1]`, giving 2,501 initial-state cells per seed and 12,505 rollouts per five-seed condition. Every rollout lasts 200 steps.

- Near-reference success means policy return is at least `max(DP return, controller return) - 5` for that exact initial state.
- A state is near upright when `cos(theta) >= 0.95` and `abs(theta_dot) <= 1`.
- Task success requires at least 80% of steps near upright and no consecutive not-near-upright streak longer than 50 steps.
- "Near down" below means `abs(theta) >= 150 degrees` within the evaluated reset support.

## Per-seed results and uncertainty

| Method | Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | 95% seed CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean DAgger | Near-reference | 0.8125 | 0.8365 | 0.7989 | 0.7929 | 0.8057 | 0.8093 | [0.7883, 0.8302] |
| Clean DAgger | Task success | 0.8645 | 0.8489 | 0.8673 | 0.8701 | 0.8709 | 0.8643 | [0.8531, 0.8755] |
| DAgger then SimbaV2 | Near-reference | 0.8249 | 0.9400 | 0.9232 | 0.9204 | 0.7953 | 0.8808 | [0.7991, 0.9625] |
| DAgger then SimbaV2 | Task success | 0.8984 | 0.9132 | 0.9036 | 0.9112 | 0.8681 | 0.8989 | [0.8763, 0.9216] |
| Standard SimbaV2 | Near-reference | 0.9148 | 0.9512 | 0.9088 | 0.9096 | 0.9072 | 0.9184 | [0.8953, 0.9414] |
| Standard SimbaV2 | Task success | 0.8992 | 0.9308 | 0.9132 | 0.9036 | 0.9260 | 0.9146 | [0.8976, 0.9316] |

The combination has much larger seed variance than standard SimbaV2. Its seed 1 is strong, but seed 4 drops to 0.7953 near-reference success. Reporting only the strongest combination seed would reverse the aggregate conclusion and would be invalid.

![Per-seed primary metrics](per_seed_primary_metrics.png)

## Near-down results

| Method | Near-reference, `abs(theta) >= 150 deg` | Task success, `abs(theta) >= 150 deg` |
|---|---:|---:|
| Clean DAgger 100k | 0.4812 | 0.4169 |
| DAgger 100k then SimbaV2 100k | 0.7086 | 0.6727 |
| Standard SimbaV2 100k | **0.7339** | **0.6945** |

The RL phase closes most of the clean DAgger near-down gap, but standard SimbaV2 remains better.

## Heatmaps: near-reference success

| Clean DAgger 100k | DAgger 100k then SimbaV2 100k | Standard SimbaV2 100k |
|---|---|---|
| ![Clean DAgger near-reference heatmap](../canonical_reference_dagger_100k_n5_20260716/relative/near_best_known_return_eps_map.png) | ![DAgger then SimbaV2 near-reference heatmap](../canonical_reference_dagger_then_simbav2_100k_n5_20260716/relative/near_best_known_return_eps_map.png) | ![Standard SimbaV2 near-reference heatmap](../week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/near_best_known_return_eps_map.png) |

## Heatmaps: task success

| Clean DAgger 100k | DAgger 100k then SimbaV2 100k | Standard SimbaV2 100k |
|---|---|---|
| ![Clean DAgger task heatmap](../canonical_reference_dagger_100k_n5_20260716/relative/task_success_map.png) | ![DAgger then SimbaV2 task heatmap](../canonical_reference_dagger_then_simbav2_100k_n5_20260716/relative/task_success_map.png) | ![Standard SimbaV2 task heatmap](../week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/task_success_map.png) |

## Why this does not reproduce the old 0.9928 "DAgger" result

The older result was not a clean 100k trajectory-only DAgger run. It started from a broad, balanced 240,000-state static distillation dataset and then added only two 10,000-state actor-rollout aggregation rounds. It therefore used 260,000 reference labels and three seeds. The one-seed static distillation comparison used 400,000 sampled labels.

| Older unmatched condition | Seeds | Reference labels per seed | Near-reference | Task success |
|---|---:|---:|---:|---:|
| Balanced static distillation | 1 | 400,000 | 0.9872 | 0.9444 |
| Static-240k initialization plus two DAgger rounds | 3 | 260,000 | 0.9928 | 0.9412 |

Those results show that broad static reference-state coverage plus a small policy-state correction can work extremely well. They do not show that clean DAgger from 100k trajectory states beats five-seed 100k SimbaV2. The new experiment answers that narrower question, and its answer is no.

## Rejected and excluded runs

The interrupted 25k-label pilot and the prematurely launched GPU continuations are not used anywhere in these tables, plots, or summaries. The accepted artifacts contain exactly five complete clean-DAgger checkpoints and five complete CPU SimbaV2 continuations.

## Artifact locations

- Clean DAgger runs: `runs/canonical_reference_dagger_100k_n5_20260716/seed{0,1,2,3,4}`
- Clean DAgger evaluation: `reports/canonical_reference_dagger_100k_n5_20260716`
- DAgger then SimbaV2 runs: `runs/canonical_reference_dagger_then_simbav2_100k_n5_20260716/seed{0,1,2,3,4}`
- DAgger then SimbaV2 evaluation: `reports/canonical_reference_dagger_then_simbav2_100k_n5_20260716`
- Existing standard SimbaV2 evaluation: `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt`
- Machine-readable comparison: `reports/canonical_dagger_matched_comparison_20260716/comparison_metrics.json`
