# Corrected DAgger Evaluation and Two-Goal Search

- Date: 2026-07-17
- Environment: `Pendulum-v1`
- Final evaluation: deterministic 200-step rollout from every point on the exact 61 × 41 grid, with 61 angles in `[-π, π)` and 41 velocities in `[-1, 1]`
- Primary metrics: task success and return within 5 of `max(DP, controller)`

## Outcome

There are three separate conclusions.

1. The DAgger evaluation is now corrected. “DAgger + SimbaV2” means DAgger with a SimbaV2 actor backbone. It does **not** mean 50% imitation followed by 50% online RL.
2. The new pure-RL initial-state router improves the previous pure-RL best on both primary metrics: near-reference rises from `0.92915` to `0.93355`, and task success rises from `0.92699` to `0.92803`. It does not dominate the old method on the stricter “beats the reference” metric, which falls from `0.11659` to `0.11491`.
3. The reference-assisted search did **not** beat the legacy `0.99280` near-reference frontier. Goal 1 therefore remains open. The best new capacity pilot reached `0.99200` on one seed, and the selected five-seed specialist gate reached `0.98792`.

## What the DAgger paper actually specifies

The implementation was checked against Ross, Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* ([AISTATS/PMLR paper and PDF](https://proceedings.mlr.press/v15/ross11a.html)). Algorithm 3 does the following at iteration `i`:

1. Execute the mixture policy `πᵢ = βᵢπ* + (1−βᵢ)π̂ᵢ` to determine the visited states.
2. Ask the expert `π*` for an action at every visited state.
3. Add all new state/expert-action pairs to the cumulative dataset.
4. Retrain the next learned policy on the full cumulative dataset.
5. Return the policy with the best validation performance, rather than automatically returning the final iterate.

The paper explicitly describes the practical schedule `βᵢ = 1` for the first iteration and `βᵢ = 0` thereafter. In plain language: collect the initial trajectories from the expert; after that, let the learner choose every action passed to `env.step()`, while the expert only supplies labels for the learner-visited states.

Our corrected collector saves the current state **before** `env.step(action)`. The learned actor chooses `action`; that action determines the next visited state. The label for the saved current state is produced by the reference `max(DP, controller)`. The reference action is not executed after the initial expert batch.

## Exact corrected 100k DAgger protocol

The plain-MLP and SimbaV2-backbone conditions differ only in actor architecture.

| Item | Exact value per seed |
|---|---:|
| Training seeds | `0, 1, 2, 3, 4` |
| Initial expert episodes | `100` |
| Initial expert transitions/labels | `20,000` |
| Initial supervised epochs | `40` |
| Learner-only DAgger rounds | `4` |
| Episodes per learner round | `100` |
| Transitions/labels per learner round | `20,000` |
| Supervised epochs per learner round | `20` |
| Final cumulative labels | `100,000` |
| Environment transitions used for collection | `100,000` |
| Reference-label queries used for training | `100,000` |
| Reference queries used for model selection only | `30,000` |
| Batch size | `1,024` |
| Learning rate | `3e-4` |
| Actor optimizer steps | `6,320` |
| Selection | lowest action MAE on the independent 30k-state validation set |
| Final evaluation | exact 61 × 41 grid; no test-grid selection |

“6,320 optimizer steps” means the optimizer changed the actor weights 6,320 times using minibatches. It does not mean 6,320 environment steps.

The plain actor is the CleanRL continuous SAC actor: two ReLU layers of width 256 followed by tanh-bounded mean and log-standard-deviation heads. The SimbaV2 actor uses hidden width 32, one residual block, observation normalization, input shift, feature normalization, bias-free HyperDense layers, and Simba weight projection. There is no critic and no RL loss in either corrected DAgger condition.

## Corrected DAgger results

| Method | Seeds | Near `max(DP, controller) − 5` | Task success | Beats `max(DP, controller)` | Near-down near-reference | Near-down task |
|---|---:|---:|---:|---:|---:|---:|
| Plain MLP DAgger, paper schedule, 100k | 5 | `0.83671` | `0.86038` | `0.10020` | `0.31086` | `0.26430` |
| SimbaV2-backbone DAgger, paper schedule, 100k | 5 | `0.83343` | **`0.87285`** | `0.06285` | **`0.41064`** | **`0.35965`** |
| Normal SimbaV2 RL, 100k | 5 | `0.91835` | `0.91459` | `0.08525` | `0.73392` | `0.69446` |
| Normal broad static distillation, 400k | 1 | `0.98721` | **`0.94442`** | `0.11515` | — | — |
| Legacy static + two-round DAgger | 3 | **`0.99280`** | `0.94122` | `0.10369` | **`0.97783`** | `0.70436` |

Per-seed corrected near-reference rates:

- Plain MLP: `0.86605, 0.81807, 0.82167, 0.84446, 0.83327`.
- SimbaV2 actor: `0.85006, 0.79528, 0.83087, 0.88964, 0.80128`.

Per-seed corrected task rates:

- Plain MLP: `0.88645, 0.85166, 0.85286, 0.86405, 0.84686`.
- SimbaV2 actor: `0.87205, 0.84806, 0.88165, 0.90164, 0.86086`.

The correct conclusion is not that the Simba backbone universally wins. It raises mean task success by `0.01248` and improves both near-down metrics, but lowers overall near-reference by `0.00328` and the strict beats-reference rate by `0.03735`.

The legacy `0.99280` result is not a clean 100k DAgger comparison. Each seed starts from the same one-seed actor trained on 400k static reference labels. The follow-up stage generates a 240k static reference pool, collects two learner-only batches of 10k states, and performs ten additional epochs. Thus the final actor inherits training from approximately 660k privileged labels: 400k from initialization plus 260k queried during the follow-up. Only three follow-up seeds exist, the initialization is shared, last-epoch selection was used, and the old run did not persist the static sampler mixture flags. Those missing flags must not be invented.

## Why the corrected 100k DAgger is worse than the legacy run

The result is explained by coverage and budget, not by a different definition of DAgger.

- Corrected DAgger receives only 20k broad expert-trajectory states before it must collect learner states.
- The legacy policy inherits a broad 400k-state static fit before collecting any learner states.
- The legacy follow-up then trains on another 240k static pool plus 20k learner-visited states.
- The corrected conditions have 100k privileged labels total; the legacy pipeline has roughly 660k across both stages.

The backbone affects stability, especially near downward starts, but it does not replace broad state coverage.

## Corrected DAgger heatmaps

### Plain MLP

![Plain DAgger near-reference heatmap](plots/dagger_plain_near_reference.png)

![Plain DAgger task-success heatmap](plots/dagger_plain_task_success.png)

### SimbaV2 actor backbone

![SimbaV2-backbone DAgger near-reference heatmap](plots/dagger_simba_near_reference.png)

![SimbaV2-backbone DAgger task-success heatmap](plots/dagger_simba_task_success.png)

## Goal 1: reference-assisted search

Target to beat: legacy static + DAgger near-reference `0.99280` over three follow-up seeds. The best separate one-seed task result remains broad static distillation at `0.94442`.

| Candidate | Seeds | Near-reference | Task success | Decision |
|---|---:|---:|---:|---|
| Legacy static + DAgger frontier | 3 | `0.99280` | `0.94122` | Target |
| Initial-state return gate, first version | 5 | `0.99264` | `0.93699` | Reject: lower on both target metrics |
| Task-aware initial-state gate | 5 | `0.99072` | `0.94058` | Reject |
| RL-critic filter over supervised specialist actions | 5 | `0.97601` | `0.93083` | Reject |
| Dense four-specialist gate | 5 | `0.98856` | `0.93970` | Reject |
| Static/hard threshold router | 1 | `0.98960` | `0.94402` | Reject: task essentially tied; near-reference lower |
| Larger SimbaV2 64×2 reset-support pilot | 1 | `0.99200` | `0.94002` | Reject; remaining four seeds not run |
| Held-out reference-primary DAgger/static/hard gate | 5 | `0.98792` | `0.93914` | Reject |

The 64×2 pilot is the cleanest training intervention. It used 400k explicit static labels—60% full reset support with velocity in `[-1,1]`, 20% near-upright, and 20% broad states with velocity in `[-8,8]`—plus three learner-only DAgger batches of 10k states. Total privileged labels were 430k, total collection transitions were 30k, and the actor received 43,600 optimizer steps. It reduced validation action MAE to `0.05572`, but did not cross the final return frontier.

The selector experiments demonstrate oracle complementarity but failed to generalize at the precision required by this grid. They must not be presented as wins.

### Current reference frontier

![Legacy reference DAgger near-reference heatmap](plots/legacy_reference_dagger_near_reference.png)

![Legacy reference DAgger task-success heatmap](plots/legacy_reference_dagger_task_success.png)

### Best new capacity pilot

![Large reference pilot near-reference heatmap](plots/large_reference_pilot_near_reference.png)

![Large reference pilot task-success heatmap](plots/large_reference_pilot_task_success.png)

## Goal 2: pure RL

### Base training method

The five base checkpoints are FastSACN8 SimbaV2 agents trained for 50k environment steps with UTD 2. They use only environment transitions and rewards. They use no DP actions, controller actions, reference replay, distilled initialization, behavior cloning, or supervised reference loss.

The existing inference policy evaluates 41 evenly spaced actions in `[-2,2]`. It scores each action with clipped double Q, `min(Q₁,Q₂)`. It uses the searched action only when

`min(Q₁(s,a_search), Q₂(s,a_search)) − min(Q₁(s,a_actor), Q₂(s,a_actor)) > 0.005`.

Otherwise it keeps the deterministic actor action.

The unanimous variant uses the same 41 candidates but requires both critics separately to prefer the searched action by more than `0.005`.

### New initial-state router

The new method chooses one component once, at the initial state, and uses that component for the entire episode:

```text
if abs(initial_angle_degrees) >= 150:
    use unanimous-advantage Q filtering for all 200 steps
else:
    use clipped-double-Q filtering for all 200 steps
```

The threshold was selected without reference information. On a disjoint 17 × 11 midpoint grid—187 states per seed, 935 trajectories total—thresholds `120, 135, 150, 165`, and “never route” were compared by task success and mean environment return. Thresholds 150 and 165 tied on the coarse validation grid; 150 was retained because it is the pre-existing near-down analysis boundary. No DP/controller action or return entered this selection.

On the final grid, 2,255 of 12,505 trajectories (`18.03%`) use the unanimous component, and 10,250 use the clipped-Q component.

| Pure-RL policy | Seeds | Near-reference | Task success | Beats reference | Near-down near-reference | Near-down task |
|---|---:|---:|---:|---:|---:|---:|
| Actor only | 5 | `0.90284` | `0.91084` | `0.11259` | `0.71441` | `0.67761` |
| 41-action clipped Q filter | 5 | `0.92915` | `0.92699` | **`0.11659`** | `0.74146` | `0.67716` |
| 41-action unanimous Q filter | 5 | `0.92731` | `0.92515` | `0.11659` | `0.76585` | `0.68293` |
| **Initial-state router** | **5** | **`0.93355`** | **`0.92803`** | `0.11491` | **`0.76585`** | **`0.68293`** |

Router deltas versus the previous clipped-Q best:

- Near-reference: `+0.00440`.
- Task success: `+0.00104`.
- Near-down near-reference: `+0.02439`.
- Near-down task: `+0.00577`.
- Beats-reference: `−0.00168`.

Per-seed router near-reference rates are `0.92763, 0.95442, 0.92243, 0.93123, 0.93203`. Per-seed task rates are `0.92683, 0.93203, 0.92003, 0.92843, 0.93283`.

Therefore Goal 2 is achieved on both declared primary metrics, but the router is not a strict improvement on every reported metric.

### Pure-RL winner heatmaps

![Pure-RL router near-reference heatmap](plots/pure_rl_router_near_reference.png)

![Pure-RL router task-success heatmap](plots/pure_rl_router_task_success.png)

## Reproduction artifacts

- The pushed pure-RL routing implementation is `scripts/build_pendulum_initial_state_router.py`.
- Local raw corrected runs are `runs/paper_dagger_plainmlp_100k_5seed_20260717` and `runs/paper_dagger_simbav2_100k_5seed_20260717`.
- Local raw corrected reports are under `reports/paper_dagger_backbone_100k_5seed_20260717`.
- Local raw pure-RL router outputs are under `reports/pure_rl_initial_state_qfilter_router_5seed_20260717`.

The large raw checkpoints, per-rollout CSV files, and rejected-candidate directories are intentionally not duplicated in this report commit. The exact aggregate values and the relevant heatmaps are included here.

## Final status

- Correct DAgger evaluation: **complete**.
- DAgger with SimbaV2 as the actor backbone: **complete**.
- Goal 1, beat the legacy reference-assisted near-reference frontier: **not achieved**.
- Goal 2, beat the previous pure-RL best on near-reference and task success: **achieved**, with the strict beats-reference caveat stated above.
