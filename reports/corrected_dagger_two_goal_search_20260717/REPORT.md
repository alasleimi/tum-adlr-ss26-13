# Correct DAgger Evaluation and Two-Goal Pendulum Search

- Final update: 2026-07-18
- Environment: `Pendulum-v1`
- Evaluation horizon: 200 deterministic steps
- Evaluation grid: exactly 61 angles in `[-pi, pi)` by 41 angular velocities in `[-1, 1]`, or 2,501 initial states
- Training/follow-up seeds: `0, 1, 2, 3, 4`, unless a row explicitly says otherwise
- Total final trials for a five-seed method: `2,501 x 5 = 12,505`

## Exact outcome

Goal 1 is achieved on its near-reference objective. Goal 2 is achieved on both declared primary metrics, near-reference and task success. Neither winner dominates every secondary metric.

| Goal | Previous target | New result | Status |
|---|---:|---:|---|
| Reference-assisted: near-reference success | legacy static + DAgger `0.992803` | clean explicit five-seed follow-up `0.994482`; nonzero RL-critic overlay `0.994642` | achieved |
| Pure RL: near-reference success | 41-action clipped-Q filter `0.929148` | initial-state Q-filter router `0.933547` | achieved |
| Pure RL: task success | 41-action clipped-Q filter `0.926989` | initial-state Q-filter router `0.928029` | achieved |

Important qualifications:

- The clean reference result is the more substantial Goal 1 result. It improves the old frontier without using RL at inference.
- The RL+supervised overlay adds only two near-reference successes over clean DAgger, changes just 12 actions in 2,501,000 decisions, and loses two task-success trials. It is a technical primary-metric win, not evidence of a broad hybrid improvement.
- The pure-RL router improves both primary metrics and both near-down metrics, but its strict `return >= reference` rate is `0.114914`, below the previous `0.116593`.

## Metrics, without shorthand

For each initial state and training seed, the policy runs for 200 steps.

- **Near-reference success** is one if the policy return is at least `max(DP return, controller return) - 5` for the same initial state.
- **Beats-reference success** is one if the policy return is at least `max(DP return, controller return)` with no epsilon.
- A step is **near upright** when `cos(theta) >= 0.95` and `abs(theta_dot) <= 1`.
- **Task success** is one if at least 80% of the 200 steps are near upright and no run of consecutive non-near-upright steps exceeds 50.
- **Near down** means `abs(initial theta) >= 150 degrees`. This region contains 451 grid cells and 2,255 five-seed trials.

All rates in the main result tables are trial means over the full seed-by-state grid. The heatmap value in a cell is the success fraction across the five seeds at that initial state.

## What the DAgger paper says

The implementation was checked against Ross, Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* ([AISTATS/PMLR paper](https://proceedings.mlr.press/v15/ross11a.html)). Algorithm 3 uses this loop at iteration `i`:

1. Execute `pi_i = beta_i * expert + (1 - beta_i) * learned_policy` to determine which states are visited.
2. Query the expert action for every visited state.
3. Add the new state/expert-action pairs to the cumulative dataset.
4. Train the next learned policy on the full cumulative dataset.
5. Select a policy using validation performance; the final iterate is not automatically guaranteed to be the returned policy.

The paper also gives the practical schedule `beta=1` initially and `beta=0` afterward. With `beta=0`, the learned policy supplies the action passed to `env.step()`. The reference labels the current state but does not control the transition.

The corrected collector follows that ordering exactly:

```text
state_t = current observation
action_t = learned_policy(state_t)
label_t = best_reference_action(state_t)
save(state_t, label_t)
state_t+1 = env.step(action_t)
```

The implementation saves `state_t` before `env.step`. Therefore the policy action, not the reference label, determines `state_t+1`.

## Corrected clean 100k DAgger comparison

This experiment answered the architecture question cleanly: ordinary DAgger and DAgger with a SimbaV2 actor backbone used the same data schedule and losses. No RL critic, RL replay, or reward loss was used.

| Item | Exact value per seed |
|---|---:|
| Initial expert episodes | 100 |
| Initial expert transitions and labels | 20,000 |
| Initial supervised epochs | 40 |
| Learner-only DAgger rounds | 4 |
| Learner episodes per round | 100 |
| Learner transitions and new labels per round | 20,000 |
| Supervised epochs after each round | 20 |
| Final cumulative labels | 100,000 |
| Collection environment transitions | 100,000 |
| Separate validation labels | 30,000 |
| Minibatch size | 1,024 |
| Learning rate | `3e-4` |
| Actor optimizer steps | 6,320 |
| Selection | lowest reference-action MAE on the independent 30k validation set |

An optimizer step means one minibatch update to the model weights. It is not an environment step.

Architectures:

- **Plain DAgger:** two ReLU hidden layers of width 256, followed by the continuous SAC-style bounded actor heads.
- **DAgger + SimbaV2 backbone:** actor width 32 with one residual block, observation normalization, input shift, feature normalization, bias-free HyperDense layers, and Simba weight projection.

| Method | Seeds | Near reference | Task success | Beats reference | Near-down near reference | Near-down task |
|---|---:|---:|---:|---:|---:|---:|
| Plain MLP DAgger, correct paper schedule, 100k | 5 | `0.836705` | `0.860376` | `0.100200` | `0.310865` | `0.264302` |
| SimbaV2-backbone DAgger, correct paper schedule, 100k | 5 | `0.833427` | `0.872851` | `0.062855` | `0.410643` | `0.359645` |
| Normal SimbaV2 pure RL, 100k | 5 | `0.918353` | `0.914594` | `0.085246` | `0.733925` | `0.694457` |
| Normal broad static distillation, 400k | 1 | `0.987205` | `0.944422` | `0.115154` | not reported | not reported |
| Legacy static + two-round DAgger | 3 follow-ups | `0.992803` | `0.941224` | `0.103692` | `0.977827` | `0.704360` |

The SimbaV2 backbone does not make the 100k DAgger actor uniformly better. It raises task success by `0.012475` and helps near-down starts, but lowers near-reference success by `0.003279` and strict beats-reference success by `0.037345`.

The old badly performing DAgger-like runs and these corrected 100k runs are not comparable to the legacy `0.992803` result by budget. The legacy actor already inherited 400k static reference labels before collecting learner states. The corrected clean runs began with only 20k expert-trajectory labels and had 100k labels total.

### Corrected 100k DAgger heatmaps

| Plain MLP DAgger | SimbaV2-backbone DAgger |
|---|---|
| ![Plain DAgger near-reference heatmap](plots/dagger_plain_near_reference.png) | ![SimbaV2 DAgger near-reference heatmap](plots/dagger_simba_near_reference.png) |
| ![Plain DAgger task-success heatmap](plots/dagger_plain_task_success.png) | ![SimbaV2 DAgger task-success heatmap](plots/dagger_simba_task_success.png) |

## Goal 1: reference-assisted winner

### A. Clean explicit five-seed DAgger follow-up

The previous three-seed frontier had incomplete sampler metadata. A mixed set of old and new seeds was therefore not promoted. Seeds 0-4 were instead run or rerun with one explicit recipe.

Each follow-up seed used:

- the same seed-0 SimbaV2 actor checkpoint previously distilled from 400,000 broad static reference labels;
- a fresh 240,000-state static pool labeled by the better action from DP and controller;
- two learner-only DAgger rounds;
- 50 deterministic Pendulum episodes per round;
- 200 steps per episode, so 10,000 learner-controlled transitions and 10,000 new reference labels per round;
- `beta=0` in both follow-up rounds;
- five full cumulative-dataset epochs after each round;
- minibatch size 1,024 and learning rate `3e-5`;
- the SimbaV2 actor with width 32 and one residual block;
- last-iterate selection, fixed before the 61 x 41 evaluation.

The first follow-up round trains on 250,000 pairs and the second on 260,000 pairs. That gives 2,495 new actor optimizer steps. Per seed, the follow-up itself queries 260,000 labels and uses 20,000 learner-controlled environment transitions. Including inherited pretraining, the actor has approximately 660,000 privileged reference labels in its history.

All five follow-ups start from the same 400k-label seed-0 checkpoint. They have seed-specific static pools, learner trajectories, and optimization, but they are not five independent end-to-end initializations. The result must be called a five-seed follow-up, not five independent pretraining runs.

Results:

| Method | Seeds | Near reference | Task success | Beats reference | Near-down near reference | Near-down task |
|---|---:|---:|---:|---:|---:|---:|
| Legacy static + DAgger target | 3 follow-ups | `0.992803` | `0.941224` | `0.103692` | `0.977827` | `0.704360` |
| **Explicit clean follow-up** | **5 follow-ups** | **`0.994482`** | `0.939304` | `0.092443` | **`0.981375`** | `0.704656` |

The clean follow-up beats the old primary frontier by `0.001679`. It does not beat the old task-success or strict beats-reference rates. Its near-reference result is 12,436 successes out of 12,505 trials.

Per-seed near-reference rates are `0.994802, 0.994802, 0.994402, 0.994402, 0.994002`. Per-seed task-success rates are `0.938425, 0.939224, 0.940424, 0.939224, 0.939224`.

![Explicit five-seed follow-up near-reference heatmap](plots/reference_explicit5_near_reference.png)

![Explicit five-seed follow-up task-success heatmap](plots/reference_explicit5_task_success.png)

### B. RL + supervised cross-pollination audit

This audit tested whether the pure-RL critics could safely choose among four fixed candidate actions at every state:

1. the seed-matched clean DAgger actor;
2. the shared broad 400k supervised actor;
3. the shared 450k hard-region supervised actor;
4. the seed-matched pure-RL actor.

For a state `s`, both pure-RL critics score all four actions. The candidate with the largest clipped value `min(Q1, Q2)` is identified. In the selected **unanimous-advantage** rule, that candidate replaces the DAgger action only when each critic separately gives it more than a `0.2` advantage over the DAgger action:

```text
candidate = argmax_a min(Q1(s,a), Q2(s,a))
switch only if min_i [Qi(s,candidate) - Qi(s,dagger)] > 0.2
```

No model weights are updated in this overlay. The RL contribution is the seed-matched critic gate and, when selected, the pure-RL actor action.

Selection used a disjoint midpoint grid with 17 angles by 11 velocities, or 187 states per seed. The screened margins were `0.02, 0.05, 0.1, 0.2, 0.5, 1, 2`, plus an effectively infinite zero-switch control. Both clipped-value and unanimous-advantage rules were screened. The ordering was near-reference success, task success, then mean environment return.

The unconstrained screen selected a zero-switch setting, proving that the safest held-out answer was simply to retain DAgger. A separately declared qualification audit excluded variants with zero held-out switches. It selected margin `0.2` with unanimous advantage. That setting switched once in 187,000 held-out decisions.

On the final grid it switched only 12 of 2,501,000 actions (`0.000480%`):

- seed 0: 5 switches, all to the hard supervised actor;
- seed 1: 5 switches, all to the hard supervised actor;
- seed 2: 2 switches, both to the pure-RL actor;
- seeds 3 and 4: 0 switches;
- broad static actor: 0 selected actions.

| Method | Near reference | Task success | Beats reference | Near-down near reference | Near-down task |
|---|---:|---:|---:|---:|---:|
| Clean explicit follow-up | `0.994482` (12,436/12,505) | `0.939304` (11,746/12,505) | `0.092443` (1,156/12,505) | `0.981375` | `0.704656` |
| **Nonzero RL-critic overlay** | **`0.994642` (12,438/12,505)** | `0.939144` (11,744/12,505) | `0.091883` (1,149/12,505) | **`0.982262`** | `0.704656` |

The overlay beats the legacy primary frontier by `0.001839` and the clean actor by exactly two near-reference trials. It also loses two task-success trials and seven strict beats-reference trials. Because only 12 actions change and the seed confidence intervals overlap, this should not be described as a robust RL improvement. Goal 1 is already achieved by the cleaner supervised DAgger follow-up; the overlay is evidence that conservative cross-pollination can alter a few boundary cases, not that RL broadly solved the remaining errors.

![RL-supervised overlay near-reference heatmap](plots/reference_hybrid_near_reference.png)

![RL-supervised overlay task-success heatmap](plots/reference_hybrid_task_success.png)

### Reference-search failures that were rejected

| Candidate | Seeds | Near reference | Task success | Decision |
|---|---:|---:|---:|---|
| Initial-state return gate, first version | 5 | `0.99264` | `0.93699` | lower than target |
| Task-aware initial-state gate | 5 | `0.99072` | `0.94058` | lower near-reference rate |
| Earlier low-margin RL-critic specialist filter | 5 | `0.97601` | `0.93083` | unstable over-switching |
| Dense four-specialist gate | 5 | `0.98856` | `0.93970` | failed to generalize |
| Static/hard threshold router | 1 | `0.98960` | `0.94402` | near-reference below target |
| Larger SimbaV2 64 x 2 reset-support pilot | 1 | `0.99200` | `0.94002` | below target; other seeds not run |
| Held-out reference-primary specialist gate | 5 | `0.98792` | `0.93914` | below target |

The earlier filters used margins that allowed many critic-driven replacements. Their lower final rates are why the accepted overlay uses a much more conservative margin and unanimous critic agreement.

## Goal 2: strictly pure RL winner

### Training method

The five base policies are SimbaV2 FastSACN agents trained for 50,000 environment steps. They use only Pendulum states, actions, rewards, and next states. Every reference-related configuration field is disabled: no DP/controller action is executed, no reference transition is put in replay, no distilled checkpoint initializes the actor, and no behavior-cloning or reference loss is used.

Exact components relevant to the winner:

- **SimbaV2 actor:** width 32, one residual block.
- **SimbaV2 critics:** two critics, width 64, two residual blocks, distributional 51-bin outputs.
- **Shared SimbaV2 machinery:** observation normalization, input shift, feature normalization, reward scaling, and weight projection are active. Thus this is still a SimbaV2 network and training stack.
- **FastSACN8 targets:** replay supplies trajectories up to eight steps. `fast_last` trains the first and eighth supported horizons, with horizon decay parameter `lambda=0.5`.
- **UTD 2:** after each environment step, the code performs two replay-sampled critic update calls. The actor policy frequency is 2, so the actor weights are updated on every second critic update call, approximately once per environment step after learning starts.
- **Replay:** capacity 100,000, minibatch 256, learning starts after 1,000 environment steps.
- **Optimization:** actor and critic learning rates decay from `1e-4` to `5e-5`; discount `0.99`; target-network coefficient `tau=0.005`.
- **Hard-region replay:** starting at step 10,000, 2% of sampled replay is drawn from stored transitions whose state has absolute angle 120-135 degrees and velocity within `[-1,1]`; that fraction decays over 20,000 steps to 0.1%.
- **No hard reset:** environment reset probability is zero. Hard-region replay reweights already collected pure-RL transitions; it does not insert reference data.

### Q search at inference

The deterministic actor proposes one torque. The Q-search policy also evaluates 41 equally spaced torques from `-2` to `2`. For each torque it computes `min(Q1,Q2)` and finds the largest clipped-Q candidate.

The **clipped-value** component uses the searched action only if its clipped-Q value exceeds the actor action's clipped-Q value by more than `0.005`.

The **unanimous-advantage** component first chooses the same clipped-Q candidate, but uses it only if both critics separately prefer it to the actor action by more than `0.005`.

Q search changes only action selection at evaluation time. It performs no gradient update and changes no weights.

### Initial-state router

The new method chooses a component once at the initial state and keeps it for all 200 steps:

```text
if abs(initial_angle_degrees) >= 150:
    use unanimous-advantage Q filtering for the episode
else:
    use clipped-value Q filtering for the episode
```

The threshold was selected without DP/controller information. A disjoint 17 x 11 midpoint grid compared thresholds `120, 135, 150, 165`, and never-route by task success and mean environment return. Thresholds 150 and 165 tied on that coarse held-out grid; 150 retained the pre-existing near-down boundary. The router threshold did not use final-grid returns; the final router rollout table was composed after that rule was fixed.

On the final grid, 2,255 of 12,505 trajectories (`18.03%`) use the unanimous component and 10,250 use the clipped-value component.

| Pure-RL policy | Seeds | Near reference | Task success | Beats reference | Near-down near reference | Near-down task |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic actor only | 5 | `0.90284` | `0.91084` | `0.11259` | `0.71441` | `0.67761` |
| Previous 41-action clipped-Q best | 5 | `0.929148` | `0.926989` | **`0.116593`** | `0.741463` | `0.677162` |
| 41-action unanimous-Q component | 5 | `0.927309` | `0.925150` | **`0.116593`** | `0.765854` | `0.682927` |
| **Initial-state router** | **5** | **`0.933547`** | **`0.928029`** | `0.114914` | **`0.765854`** | **`0.682927`** |

Router changes versus the previous clipped-Q best:

- near-reference: `+0.004398`;
- task success: `+0.001040`;
- near-down near-reference: `+0.024390`;
- near-down task success: `+0.005765`;
- strict beats-reference: `-0.001679`.

The router produces 11,674 near-reference successes and 11,605 task successes out of 12,505 trials. Per-seed near-reference rates are `0.927629, 0.954418, 0.922431, 0.931228, 0.932027`; task-success rates are `0.926829, 0.932027, 0.920032, 0.928429, 0.932827`.

An 81-action Q-search candidate was also selected on held-out data but failed the final grid (`0.922191` near reference, `0.926829` task success), so it was rejected.

![Pure-RL router near-reference heatmap](plots/pure_rl_router_near_reference.png)

![Pure-RL router task-success heatmap](plots/pure_rl_router_task_success.png)

## Final interpretation

- Correct paper-style DAgger evaluation is complete.
- DAgger with SimbaV2 as the actor backbone is complete and contains no hidden 50/50 RL mixture.
- Goal 1 is achieved on the declared near-reference metric by the explicit clean five-seed follow-up. The RL-critic overlay raises that metric by two more trials but is too small and mixed in its secondary effects to call a robust hybrid improvement.
- Goal 2 is achieved on near-reference and task-success metrics by a strictly pure-RL initial-state router. It is not a strict winner on every metric because the old clipped-Q method still has the higher beats-reference rate.
- Normal broad static distillation remains the best reported task-success rate (`0.944422`), but it is a one-seed result and does not have the best near-reference rate.

## Local reproduction artifacts

- Corrected 100k DAgger runs: `runs/paper_dagger_plainmlp_100k_5seed_20260717` and `runs/paper_dagger_simbav2_100k_5seed_20260717`.
- Explicit five-seed reference follow-ups: `runs/reference_specialist_dagger_extension_20260717/seed0` through `seed4`.
- Explicit follow-up evaluation: `reports/reference_specialist_dagger_explicit_5seed_20260718`.
- RL-overlay screen and evaluation: `reports/reference_assisted_nonzero_rl_overlay_5seed_20260718`.
- Pure-RL router evaluation: `reports/pure_rl_initial_state_qfilter_router_5seed_20260717`.
- Pushed pure-RL rollout router: `scripts/build_pendulum_initial_state_router.py`.

Large checkpoints and raw rollout CSVs are not duplicated in this report directory. The report commit contains the exact aggregate report and the requested near-reference and task-success heatmaps.

Machine-readable audit files copied into this report are:

- [`reference_clean_relative_summary.json`](data/reference_clean_relative_summary.json)
- [`reference_hybrid_heldout_screen.json`](data/reference_hybrid_heldout_screen.json)
- [`reference_hybrid_summary.json`](data/reference_hybrid_summary.json)
- [`reference_hybrid_relative_summary.json`](data/reference_hybrid_relative_summary.json)
- [`pure_rl_router_summary.json`](data/pure_rl_router_summary.json)
- [`pure_rl_router_relative_summary.json`](data/pure_rl_router_relative_summary.json)
