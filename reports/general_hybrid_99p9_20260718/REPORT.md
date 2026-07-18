# Project 15: qualifying 99.9% supervised + RL recipe

Date: 2026-07-18

Environment: `Pendulum-v1`

Final method: **G6C priority DAgger + conservative RL target shifts + fixed local critic Q-search**

## Result

The frozen recipe reached the goal on the one-shot authoritative evaluation:

| Metric | Exact result | Rate |
|---|---:|---:|
| Near reference, `return >= max(DP, controller) - 5` | **12,500 / 12,505** | **99.960016%** |
| Task success | 11,742 / 12,505 | 93.898441% |
| Literal strict reference wins, `return > max(DP, controller)` | **1,791 / 12,505** | **14.322271%** |
| Mean return | — | `-138.562192` |

The declared 99.9% threshold was 12,493 successes. The result exceeds it by
seven successes and has five failures. The pooled Wilson 95% interval for the
near-reference rate is `[99.906427%, 99.982920%]`; this is descriptive because
the grid is deterministic, not an IID population sample.

This is a qualifying result, not the earlier 99.96% diagnostic. This recipe has:

- no hand-written failure-region sampler;
- no action multiplier or post-training calibration;
- no per-seed hyperparameters or per-state policy mixture;
- no initial-state router;
- one shared critic and one fixed Q-search rule for all five actor seeds;
- no DP or controller query at inference;
- no checkpoint or hyperparameter selection on the authoritative grid.

The claim is exact for the five trained actors on the stated 61×41 grid. It is
not a claim that every future seed or every continuous initial state succeeds at
99.9%.

![Comparison of exact final-grid rates](authoritative/comparison_rates.png)

## What each metric means

Every rollout lasts 200 deterministic environment steps. There are 61 initial
angles in `[-pi, pi)` and 41 initial angular velocities in `[-1, 1]`, giving
2,501 initial-state cells. Five actor seeds give `2,501 × 5 = 12,505` trials.

For each initial state, the evaluation has a DP return and an energy-controller
return. The reference return is their maximum.

- **Near reference:** policy return is at least reference return minus 5.
- **Strict reference win:** policy return is literally greater than the
  reference return. Equality does not count.
- **Near upright at one step:** `cos(theta) >= 0.95` and
  `abs(theta_dot) <= 1`.
- **Task success:** at least 80% of the 200 steps are near upright and the
  rollout never has more than 50 consecutive non-near-upright steps.

Near-reference success and task success are different objectives. Return is a
dense sum of angle, velocity, and torque costs; task success applies two hard
thresholds to the trajectory. A policy can improve return while crossing a task
threshold in the wrong direction. That happened here: G6C adds 62
near-reference successes over the previous supervised + RL frontier but has two
fewer task-success trials.

## Final heatmaps over initial states

Each heatmap cell is the fraction of the five actor seeds that succeed from that
exact initial state.

![Near-reference success by initial state](authoritative/relative/near_best_known_return_eps_map.png)

The only non-perfect cells are at the far-left near-down boundary. All 2,501
cells succeed for at least one seed; 99.880048% of cells succeed for all five
seeds.

![Task success by initial state](authoritative/grid/task_success_rate_map.png)

Task failures are concentrated near the angle wrap/downward boundary, with a
few isolated threshold failures elsewhere. The all-five-seed task-success cell
fraction is 93.522591%.

## The five near-reference failures

`gap` below is `reference return - policy return`, so near-reference fails when
`gap > 5`.

| Seed | Initial angle (deg) | Initial velocity | Policy return | Reference return | Gap |
|---:|---:|---:|---:|---:|---:|
| 0 | -174.098361 | -0.10 | -365.207030 | -246.363752 | 118.843278 |
| 0 | -180.000000 | 0.00 | -311.570140 | -304.326295 | 7.243845 |
| 1 | -174.098361 | -0.05 | -375.187102 | -247.642521 | 127.544582 |
| 3 | -180.000000 | 0.00 | -313.618431 | -304.326295 | 9.292136 |
| 4 | -180.000000 | 0.00 | -313.873060 | -304.326295 | 9.546765 |

Seed 2 has no near-reference failure on the authoritative grid.

## Per-seed authoritative results

The selected round is the stage-2 DAgger round whose checkpoint was frozen by
the two development validation sets. Training still executed all ten rounds so
the allocated budget was identical for all seeds; later rounds were not copied
into an earlier selected checkpoint.

| Actor seed | Selected stage-2 round | Near reference | Task success | Strict wins | Mean return |
|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 2,499/2,501 (99.920032%) | 2,352/2,501 (94.042383%) | 334/2,501 (13.354658%) | -138.529912 |
| 1 | 1 | 2,500/2,501 (99.960016%) | 2,348/2,501 (93.882447%) | 267/2,501 (10.675730%) | -138.662203 |
| 2 | 2 | 2,501/2,501 (100.000000%) | 2,349/2,501 (93.922431%) | 375/2,501 (14.994002%) | -138.553635 |
| 3 | 7 | 2,500/2,501 (99.960016%) | 2,349/2,501 (93.922431%) | 378/2,501 (15.113954%) | -138.525696 |
| 4 | 2 | 2,500/2,501 (99.960016%) | 2,344/2,501 (93.722511%) | 437/2,501 (17.473011%) | -138.539512 |

## The recipe, component by component

The deployed object has two learned parts: one supervised actor for that actor
seed and one pure-RL critic ensemble shared by all actor seeds. The pure-RL
critic is used twice: first to make tiny supervised-target changes during
stage-2 training, and later to perform a local Q-search around the actor action.

### Component 1: the best-reference label

At a state with `h` steps left, the reference computes:

1. the finite-horizon DP value and its greedy action for `h` remaining steps;
2. the return from executing the energy controller for `h` remaining steps;
3. the DP action if the DP value is at least the controller return, otherwise
   the controller action.

Thus “best reference” does not average two actions. It selects one action using
the larger predicted remaining-horizon return.

### Component 2: independently trained SimbaV2 supervised actor

Each actor seed starts from its own supervised training run. Seed 0 reuses the
existing run that used the exact frozen command; seeds 1-4 were independently
recreated.

The actor is a SimbaV2 actor with width 64 and two residual blocks. It retains
SimbaV2 observation normalization, input shift, feature normalization, HyperDense
layers, and weight projection. Therefore the final actor still uses SimbaV2
network components even though its loss is supervised.

Initializer data and optimization per seed:

| Item | Exact value |
|---|---:|
| Static reference states | 400,000 |
| Static state mixture | 60% reset support (`theta in [-pi,pi)`, velocity `[-1,1]`); 40% broad (velocity `[-8,8]`) |
| Initial epochs | 80 |
| Initial batch size / learning rate | 1,024 / `3e-4` |
| Initializer DAgger rounds | 3 |
| Episodes per initializer round | 50 |
| New learner-visited labels per round | 10,000 |
| Retraining epochs per initializer round | 10 |
| Final initializer labels | 430,000 |
| Learner-controlled initializer transitions | 30,000 |
| Initializer actor weight updates | 43,600 |

All three initializer DAgger rounds use `beta=0`: the learner acts and the
reference only labels.

### Component 3: clean pure-RL critic

The shared critic is seed 1 from the clean FastSACN8 UTD-2, 50,000-step run. It
was trained only from Pendulum transitions and rewards. Reference guidance,
reference replay, model replay, hard resets, and hard-region replay are all zero.

Its relevant SimbaV2/SAC components are:

- two distributional critics, width 64 with two residual blocks and 51 bins;
- a width-32, one-block SimbaV2 actor during RL training, although that RL actor
  is not the action proposer in the final hybrid;
- observation normalization, input shift, feature normalization, reward
  scaling, and weight projection;
- replay capacity 100,000, batch size 256, and learning start at step 1,000;
- FastSACN8 with horizons 1 and 8 trained by `fast_last`, horizon decay
  `lambda=0.5`;
- UTD 2: two critic minibatch weight updates per environment step; policy
  frequency 2 updates the RL actor once per two critic updates;
- actor and critic learning rates linearly decay from `1e-4` to `5e-5`;
- discount `0.99` and target-network coefficient `0.005`.

The critic seed was selected before confirmation and authoritative evaluation.
Clean critic seeds were checked in ascending order against development actors
0-2 on the two off-grid validation sets. Critic 0 failed at least one actor;
critic 1 was the first critic that passed every development actor. Critics 3-4
were not screened. The same critic 1 is then fixed for all five final actors.
This is global development-set model selection, not a different critic chosen
for each actor or initial state.

### Component 4: coordinate-free priority DAgger

Stage 2 adds ten new DAgger rounds. The start-state rule is fixed and has no
angle bands:

1. Uniformly sample 4,000 candidate starts over `theta in [-pi,pi)` and
   velocity `[-1,1]`.
2. From each candidate, roll out the current raw supervised actor for 200 steps.
3. Also compute the better return of DP and the controller from that start.
4. Rank candidates by `reference return - actor return`.
5. Choose the 90 largest-regret candidates plus 10 uniformly chosen remaining
   candidates.
6. Execute 100 learner-only trajectories from those starts. At time `t`, save
   the current state, let the actor action determine `state_(t+1)`, and ask the
   best reference for the label with `200-t` steps remaining.
7. Add all 20,000 state/label pairs to the cumulative dataset.
8. Train the actor for three full epochs on the cumulative dataset.

This is DAgger in the important paper sense: the learned actor determines the
visited-state distribution. It differs from vanilla DAgger only in the automatic
choice of episode starts and the conservative RL adjustment described next.
There are no expert-executed recovery trajectories in the accepted recipe.

### Component 5: conservative RL changes to supervised targets

The stage-2 aggregate begins with 240,000 uniformly sampled reset-support states
and adds 20,000 learner-visited states per round. Every state first receives the
best-reference action `a_ref`.

For each label, the fixed pure-RL critic evaluates 41 torques over the complete
`[-2,2]` action range and identifies the action `a_Q` with maximum
`min(Q1,Q2)`. The label changes only if both critics prefer `a_Q` to `a_ref` by
more than `0.01`:

```text
if min(Q1(s,a_Q)-Q1(s,a_ref), Q2(s,a_Q)-Q2(s,a_ref)) > 0.01:
    target = a_ref + clip(0.005 * (a_Q - a_ref), -0.02, 0.02)
else:
    target = a_ref
```

The maximum possible label shift is 0.02 torque units. All examples have weight
1; there is no special sample weighting in G6C. Actor optimization uses batch
size 1,024, learning rate `1e-5`, and three epochs after each of ten rounds.

### Component 6: checkpoint selection

Every third epoch—equivalently after each round—the current actor plus the fixed
deployment Q-search is evaluated on two predeclared sets:

- a 47×31 midpoint grid containing 1,457 points, deliberately offset from the
  authoritative 61×41 grid;
- 5,001 continuously uniform reset-support points from a fixed seed.

The lexicographic selection key is midpoint near-reference rate, continuous
near-reference rate, midpoint task success, continuous mean return, then
midpoint mean return. These validation sets are allowed to use the reference
because reference supervision is available in this project setting. The
authoritative 61×41 grid is not used for selection.

### Component 7: fixed local Q-search at inference

At every environment step:

1. The supervised actor proposes `a_actor`.
2. Form five actions: `a_actor + [-0.10, -0.05, 0, 0.05, 0.10]`, clipped to
   `[-2,2]`.
3. Both pure-RL critics score all five actions.
4. Choose the candidate with largest `min(Q1,Q2)`.
5. Use that candidate only if each critic separately assigns it strictly higher
   Q than `a_actor`; otherwise use `a_actor`.
6. Pass the resulting single action to `env.step()`.

There is no reference call, gradient update, actor mixture, or state router at
inference. On the authoritative evaluation, Q-search replaced the actor action
420,771 times in 2,501,000 decisions, or 16.824110%. The per-seed switch rates
range from 14.857857% to 17.750100%; an accepted replacement moves the torque by
at most 0.10.

## Exact training and selection budget

The full allocated budget per actor seed is:

| Budget category | Initializer | G6C stage 2 | Full allocated total |
|---|---:|---:|---:|
| Direct supervised/reference labels | 430,000 | 440,000 | 870,000 |
| Learner-controlled DAgger transitions | 30,000 | 200,000 | 230,000 |
| Actor weight updates | 43,600 | 10,269 | 53,869 |
| Uniform candidate starts scored | 0 | 40,000 | 40,000 |
| Actor candidate-scoring transitions | 0 | 8,000,000 | 8,000,000 |
| DP/controller candidate rollouts | 0 | 80,000 | 80,000 |

The 80,000 candidate reference rollouts are 40,000 DP plus 40,000 controller
rollouts. At 200 steps each, that is 16,000,000 reference-model transitions.
These rollout-compute costs are reported separately from label count so they are
not hidden inside “440k examples.”

Because checkpoint selection can return an earlier round, the selected stage-2
checkpoint incorporated these direct labels: seed 0, 420,000; seed 1, 260,000;
seed 2, 280,000; seed 3, 380,000; seed 4, 280,000. Adding the 430,000-label
initializer gives selected-checkpoint histories of 850,000, 690,000, 710,000,
810,000, and 710,000 labels respectively. The complete ten-round run was still
executed for every seed.

The shared pure-RL critic has a separate 50,000-environment-step training
budget. It is trained once and reused by all five actor seeds.

## Comparison with existing methods

All five-seed rows use the same authoritative grid. Normal broad distillation is
the available single-seed result, so its apparently higher task-success rate is
not a five-seed estimate. “Strict wins” in this table was recomputed directly
from rollout returns using literal `>` for every row.

| Method | Seeds | Near reference | Task success | Strict wins | Mean return |
|---|---:|---:|---:|---:|---:|
| Normal broad distillation, 400k | 1 | 98.720512% | **94.442223%** | 11.515394% | -139.575942 |
| Correct plain DAgger, 100k | 5 | 83.670532% | 86.037585% | 10.019992% | -153.490464 |
| Correct DAgger with SimbaV2 actor, 100k | 5 | 83.342663% | 87.285086% | 6.285486% | -150.971476 |
| Normal SimbaV2 pure RL, 100k | 5 | 91.835266% | 91.459416% | 8.524590% | -141.638557 |
| Best pure RL router | 5 | 93.354658% | 92.802879% | 11.491403% | -140.897560 |
| Previous supervised + RL frontier | 5 | 99.464214% | **93.914434%** | 9.188325% | -139.016769 |
| **Frozen G6C, new** | **5** | **99.960016%** | 93.898441% | **14.322271%** | **-138.562192** |

Compared with the previous supervised + RL frontier, G6C changes:

- near-reference success: `+62` trials, `+0.495802` percentage points;
- task success: `-2` trials, `-0.015994` percentage points;
- strict wins: `+642` trials, `+5.133946` percentage points;
- mean return: `+0.454578` (less negative is better).

Compared with the best pure-RL router, G6C changes:

- near-reference success: `+826` trials, `+6.605358` percentage points;
- task success: `+137` trials, `+1.095562` percentage points;
- strict wins: `+354` trials, `+2.830867` percentage points;
- mean return: `+2.335368`.

## What the best pure-RL comparator actually is

The best current pure-RL result is not the rejected 81-action Q-search run. The
81-action run obtained 92.219112% near-reference and 92.682927% task success.
The best pure-RL result is the initial-state router at 93.354658% and 92.802879%.

Its five base agents use only environment transitions and rewards. They use the
SimbaV2/FastSACN8/UTD-2 stack, 50,000 environment steps, and a replay sampler
that starts at step 10,000 with 2% of each minibatch drawn from already-collected
states at 120–135 degrees and decays that share to 0.1% over 20,000 steps. It
does not add reference transitions or change environment resets.

At inference, it evaluates 41 torques from `-2` to `2`. At initial angles below
150 degrees in absolute value it uses a clipped-min-Q improvement test; at
initial angles of at least 150 degrees it requires unanimous per-critic
improvement. That choice is made once from the initial state and held for the
whole episode. This is why it is still pure RL but is a router rather than one
uniform inference rule. G6C does not inherit that router or its hand-written
angle threshold. It reuses the cleaner FastSACN8/UTD-2 critic learning recipe
and uses one local unanimous Q-search rule everywhere.

## Development history and rejected variants

No row in this table queried the authoritative 61×41 grid. All numbers are the
two development validation rates: 1,457 midpoint points and 5,001 continuous
uniform points.

| Variant | Change | Midpoint near reference | Continuous near reference | Decision |
|---|---|---:|---:|---|
| G1 | 5 priority rounds | 99.725463% raw; 99.862732% with Q-search | 99.700060% raw; 99.860028% with Q-search | below gate |
| G2Q seeds 0/1 | 10 rounds; Q-search-aware selection | 99.931366%; 100% | 99.920016%; 100% | pass those seeds |
| G2Q seed 2 | same-seed critic | 99.725463% | 99.700060% | reject frozen same-seed critic rule |
| G3D seed 2 | disagreement weight 4 | 99.725463% | 99.640072% | reject |
| G4E seed 2 | 200k extra expert-executed recovery labels | 99.725463% | 99.660068% | reject |
| G5X seed 2 | seed-0 initializer with seed-2 critic | 99.725463% | 99.720056% | initializer was not the cause |
| G6C seed 2 | one global critic | 100% | 99.900020% | pass and freeze |

After G6C was frozen, development actors 0-2 and untouched confirmation actors
3-4 all passed both gates. Only then was the authoritative evaluator run once.

## Exact artifacts and reproduction

Authoritative artifacts:

- machine summary: [`authoritative/authoritative_summary.json`](authoritative/authoritative_summary.json)
- all 12,505 enriched rollouts: [`authoritative/relative/relative_rollouts.csv`](authoritative/relative/relative_rollouts.csv)
- per-cell relative metrics: [`authoritative/relative/relative_cell_summary.csv`](authoritative/relative/relative_cell_summary.csv)
- baseline audit table: [`baseline_comparison.csv`](baseline_comparison.csv)
- complete pre-registration and pilot ledger: [`experiment_ledger.md`](experiment_ledger.md)

Evaluation code:

- `scripts/evaluate_general_hybrid_qsearch_authoritative.py`
- `src/last_nine_rl/hybrid_qsearch.py`
- `scripts/train_pendulum_qregularized_dagger.py`

Exact actor runs, in seed order:

1. `runs/general_hybrid_g6c_seed0_20260718/seed0`
2. `runs/general_hybrid_frozen_g2q_5seed_20260718/seed1`
3. `runs/general_hybrid_g6c_seed2_20260718/seed2`
4. `runs/general_hybrid_g6c_seed3_20260718/seed3`
5. `runs/general_hybrid_g6c_seed4_20260718/seed4`

Shared critic run:

`runs/simbav2_fastsacn8_lam05_utd2_50k_20260704/seed1`

Frozen authoritative command:

```powershell
python scripts/evaluate_general_hybrid_qsearch_authoritative.py `
  --actor-runs `
    runs/general_hybrid_g6c_seed0_20260718/seed0 `
    runs/general_hybrid_frozen_g2q_5seed_20260718/seed1 `
    runs/general_hybrid_g6c_seed2_20260718/seed2 `
    runs/general_hybrid_g6c_seed3_20260718/seed3 `
    runs/general_hybrid_g6c_seed4_20260718/seed4 `
  --critic-run runs/simbav2_fastsacn8_lam05_utd2_50k_20260704/seed1 `
  --out reports/general_hybrid_99p9_20260718/authoritative `
  --device cuda --num-actions 5 --search-radius 0.10 --margin 0
```

The checkpoints are intentionally not duplicated inside the report directory.
The paths above are the exact local training artifacts used for the audit.
