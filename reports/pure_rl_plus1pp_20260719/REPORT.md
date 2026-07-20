# Pure-RL +1 Percentage-Point Search

Date: 2026-07-20

## Final outcome

The official five-seed, 100,000-environment-step SimbaV2 baseline is
`11,484 / 12,505 = 91.835266%` near-reference success and
`11,437 / 12,505 = 91.459416%` task success.

The strongest eligible <=100k result established in this search is the same
clean SimbaV2 checkpoint set with one fixed reflection-consistent inference
rule:

- evaluate the deterministic actor on state `s` and on the exact reflected
  state `mirror(s)`;
- use `a_sym = 0.5 * [pi(s) - pi(mirror(s))]` as the actor fallback;
- evaluate 41 torques uniformly spaced from `-2` to `2`;
- select the torque with the largest `min(Q1, Q2)`;
- execute it only if **each** online critic rates it more than `0.005` above
  `a_sym`; otherwise execute `a_sym`.

That rule scores **`11,832 / 12,505 = 94.618153%` near reference**,
`11,567 / 12,505 = 92.499000%` task success,
`2,303 / 12,505 = 18.416633%` literal strict wins, and mean return
`-140.620617`. It exceeds the preregistered
`11,800 / 12,505 = 94.362255%` target by 32 successes. It improves the prior
eligible leader by 93 near successes (`+0.743703` percentage points) and the
original reported `11,674 / 12,505` frontier by 158 successes
(`+1.263495` points).

The clean compact FastSACN8/UTD2 continuation completed at exactly 100k steps,
but failed its preregistered seed-0 gate. It was therefore rejected without
training seeds 1--4; no interim checkpoint was promoted or selected. A second
new single-model curriculum--FastSACN8 through 50k, then one-step SAC through
100k--also failed its final seed-0 gate on task success. Symmetry-consistent
Q-search improved mean and failure-tail return but failed its untouched-seed
task-success control gate. A new symmetry-augmented SimbaV2 seed-0 training run
was also rejected by its frozen seed-0 gate. The successful method is family U,
the reflection-averaged actor fallback plus ordinary family-B Q-search.

## Qualifying result: family U

| Seed | Near reference | Task success | Strict `>` wins | Mean return |
|---:|---:|---:|---:|---:|
| 0 | 2,350 / 2,501 (93.962415%) | 2,313 / 2,501 (92.483007%) | 527 / 2,501 | -140.616943 |
| 1 | 2,405 / 2,501 (96.161535%) | 2,328 / 2,501 (93.082767%) | 434 / 2,501 | -140.628050 |
| 2 | 2,334 / 2,501 (93.322671%) | 2,305 / 2,501 (92.163135%) | 316 / 2,501 | -140.995138 |
| 3 | 2,336 / 2,501 (93.402639%) | 2,285 / 2,501 (91.363455%) | 437 / 2,501 | -140.858497 |
| 4 | 2,407 / 2,501 (96.241503%) | 2,336 / 2,501 (93.402639%) | 589 / 2,501 | -140.004456 |
| **All** | **11,832 / 12,505 (94.618153%)** | **11,567 / 12,505 (92.499000%)** | **2,303 / 12,505 (18.416633%)** | **-140.620617** |

There are 673 near-reference failures and 938 task failures. Every row is
preserved in [the exact near-reference failure table](family_u_remaining_near_reference_failures.csv)
and [the exact task-failure table](family_u_remaining_task_failures.csv), with
seed, angle, angular velocity, policy return, best reference return, return
gap, trajectory metrics, and checkpoint path. The 12 largest near-reference
shortfalls are shown below; the complete CSV, not this excerpt, is the
authoritative failure table.

| Seed | Angle (deg) | Angular velocity | Policy return | Best reference | Shortfall |
|---:|---:|---:|---:|---:|---:|
| 0 | -126.885246 | -0.85 | -248.982250 | -118.448283 | 130.533967 |
| 0 | 126.885246 | 0.85 | -248.982249 | -118.448409 | 130.533839 |
| 1 | -126.885246 | 0.90 | -261.119962 | -135.138314 | 125.981648 |
| 4 | 126.885246 | 0.85 | -244.259796 | -118.448409 | 125.811386 |
| 1 | -126.885246 | -0.85 | -243.808825 | -118.448283 | 125.360542 |
| 4 | -126.885246 | -0.90 | -240.966276 | -115.880858 | 125.085418 |
| 2 | 174.098361 | 0.25 | -364.827889 | -240.177812 | 124.650077 |
| 4 | -126.885246 | 0.90 | -259.587361 | -135.138314 | 124.449047 |
| 2 | 174.098361 | 0.10 | -370.215635 | -246.363752 | 123.851883 |
| 0 | 126.885246 | -0.90 | -258.850805 | -135.138547 | 123.712258 |
| 3 | -126.885246 | 0.90 | -258.197244 | -135.138314 | 123.058930 |
| 3 | 126.885246 | -0.90 | -258.197244 | -135.138547 | 123.058696 |

## Exact evaluation definitions

Every five-seed row contains 2,501 deterministic initial states per seed: a
61-angle by 41-angular-velocity grid over the normal Pendulum reset support,
for `5 * 2,501 = 12,505` rollouts. Each rollout lasts 200 environment steps.

- **Near reference:** policy return is at least
  `max(DP return, energy-controller return) - 5` on the same initial state.
- **Task success:** the existing trajectory stability criterion, independent
  of reference return.
- **Literal strict win:** policy return is strictly greater than
  `max(DP return, energy-controller return)`. Equality is not a win.

The DP/controller is not used during pure-RL training, checkpoint selection,
development-state evaluation, or inference. It is loaded only after a rule is
frozen, to score the authoritative grid.

## Corrected pure-RL inventory

| Method | Steps per seed | Near reference | Task success | Status |
|---|---:|---:|---:|---|
| Plain SAC, original Week-3 set | 100k | 83.0468% | 81.1835% | eligible |
| Plain SAC, May-9 set | 100k | 88.9004% | 85.9336% | eligible |
| SimbaV2 official five-seed set | 100k | 91.8353% | 91.4594% | eligible baseline |
| Recovered alternate SimbaV2 replicas | 100k | 92.1711% | 91.5634% | diagnostic post-hoc replica selection |
| SimbaV2 + SACn16 | 100k | 90.4278% | 89.8041% | eligible |
| SimbaV2 + FastSACN8, UTD1 | 100k | 88.4206% | 90.8517% | eligible |
| Clean FastSACN8 UTD2 actor | 50k | 88.8685% | 91.2115% | eligible |
| Clean FastSACN8 UTD2 + clipped Q-search | 50k | 93.3707% | 92.8429% | eligible |
| Clean FastSACN8 UTD2 + unanimous Q-search | 50k | 93.2187% | **92.9148%** | eligible; best clean task rate |
| Compact FastSACN8 UTD2 continuation | 100k | no authority query | no authority query | rejected by reference-free seed-0 gate |
| FastSACN8-to-one-step curriculum | 100k | no authority query | no authority query | rejected by reference-free seed-0 gate |
| Symmetry-augmented SimbaV2 training | 100k | no authority query | no authority query | rejected by reference-free seed-0 gate |
| Clean SimbaV2 + online-unanimous Q-search | 100k | 93.8745% | 92.2831% | eligible; previous near leader |
| Clean SimbaV2 + reflected actor + unanimous Q-search | 100k | **94.6182%** | **92.4990%** | eligible; target achieved |
| Clean SimbaV2 + disagreement-penalized Q-search | 100k | 92.6509% | 92.4030% | eligible; rejected for near objective |
| L2-feature SAC, five new seeds | 100k | 91.4194% | 88.8285% | eligible; rejected replication |
| Targeted-replay FastSACN8 + Q-search | 50k | 92.9148% | 92.6989% | ineligible manual replay region |
| Targeted-replay FastSACN8 + initial-state router | 50k | 93.3547% | 92.8029% | ineligible manual replay region and router |
| Plain SAC UTD2 | 250k | 95.8577% | 88.9164% | eligible long-budget |
| Plain SAC UTD1 | 500k | 96.3215% | 88.6126% | eligible long-budget |

The pre-search inventory was rechecked by parsing all 351
`relative_criterion_summary.csv` files under `reports/` and sorting every
five-seed summary with exactly 12,505 rollouts. Before family U was generated,
no additional reference-free, five-seed, <=100k result above 93.8745% was
found. Higher historical rows were either reference-assisted/hybrid recipes
or the 250k and 500k plain-SAC runs shown separately above.

The 500k plain-SAC score is real. It is not the matched 100k comparison: it
uses 2.5 million environment steps over five seeds, versus 500,000 total for a
five-seed 100k recipe. It also has much lower task success than the compact
Q-search recipes.

The archived `94.8820%` near-reference number for
`simba_full_official_opt_nois_fastsacn8_soft_lam05` is also real, but it is a
**single-seed** result: seed 0 scored `2,373 / 2,501`, with 92.1631% task
success. Its `relative_summary.json` records `actual_seeds: [0]`,
`num_training_seeds: 1`, and 2,501 total rollouts. It must not be quoted as a
five-seed score. The matched five-seed actor result for that 100k FastSACN8
UTD1 family is the 88.4206% near / 90.8517% task row in the table above.

## What was wrong in the previous FastSACN report

The July-10 report said its `92.9148%` near / `92.6989%` task row used the
uniform clean FastSACN8 checkpoints. A source-path and exact-score audit showed
that all five reported per-seed rows instead match the later `hard02to001`
checkpoints. Those checkpoints force a decaying replay fraction from a manually
specified 120--135-degree initial-angle band. They are pure in the narrow sense
of using no teacher labels, but they violate this goal's general-recipe rule.

The original fixed clipped-Q rule was therefore rerun on the actual clean
FastSACN checkpoints. Its corrected score is `93.3707%` near and `92.8429%`
task. The clean family was important; the old report attached the wrong source
paths and numbers to it.

## Reference-free development protocol

Candidate parameters were selected without DP/controller values. Development
rollouts start from continuous off-grid reset states with angle uniform in
`[-pi, pi]` and angular velocity uniform in `[-1, 1]`.

1. Phase A uses one development seed and 2,048 states.
2. Phase B uses seeds 0--2 and 4,096 states per seed.
3. A candidate survives only if mean-return and bottom-10%-return changes are
   nonnegative on every development seed.
4. Selection maximizes worst-seed mean-return change, then pooled mean-return
   change; task success and fewer candidate actions are tie-breakers.
5. Phase C checks the frozen candidate on untouched seeds 3--4 with 4,096
   states each. It can accept or reject, not retune.
6. Only after confirmation is the reference loaded for one authoritative-grid
   evaluation of that frozen family.

No candidate uses a state router, initial-angle threshold, manual failure band,
reference replay, reference action, reward shaping, or per-seed parameter.

## Completed candidate families

| Family | Frozen rule | Near | Task | Strict | Decision |
|---|---|---:|---:|---:|---|
| B | online-min proposal; online unanimity; margin .005 | 93.8745% | 92.2831% | 8.8445% | previous leader |
| D | online-min proposal; online+target veto; margin .01 | 92.6829% | 92.4430% | 8.9804% | reject |
| E | four-critic min proposal and unanimity; margin .005 | 93.6825% | 92.4510% | 8.8844% | reject |
| F | fixed fractional step toward family-B action | no new authority | no new authority | no new authority | fraction 1.0 dominated fractions .25/.5/.75 |
| G | online-mean proposal; online unanimity; margin .005 | 93.7865% | 92.2351% | 8.8285% | reject |
| H | clean FastSACN; online-min proposal; unanimity; margin .005 | 93.2187% | **92.9148%** | 7.2371% | reject for near objective |
| L | target-min proposal; all-critic unanimity; margin .01 | 92.6429% | 92.4510% | 8.9884% | reject for near objective |
| M | disagreement-penalized proposal; online unanimity; margin .01 | 92.6509% | 92.4030% | **9.0844%** | reject for near objective |
| N | task-selected global margin .01 | no authority | no authority | no authority | failed untouched-seed control gate |
| O | actor-centered action-distance trust region | no authority | no authority | no authority | failed three-seed return gate |
| P | interpolated proposal `mean - .25*disagreement` | 93.8585% | 92.2751% | 8.8844% | two near successes below leader |
| Q | disagreement-increase acceptance penalty | no authority | no authority | no authority | failed untouched-seed control gate |
| S | reflection-averaged critics with family-B Q-search | no authority | no authority | no authority | failed untouched-seed task control gate |
| U | reflection-averaged actor fallback with family-B Q-search | **94.6182%** | **92.4990%** | **18.4166%** | target achieved; current leader |

The result is not monotonic in off-grid return improvement. Family D improved
development mean and tail returns on all five seeds and improved authoritative
task success, but reduced categorical near-reference success. This is why the
report preserves both continuous return screens and the final grid criteria.

## Current leader: component-by-component

The network is the standard project SimbaV2 actor-critic checkpoint, not a new
mixture model. It retains the reduced-dimension SimbaV2 actor, two categorical
SimbaV2 critics, reward scaling, observation normalization, weight projection,
and the SAC training state. Each seed has exactly one actor and its own two
critics.

At each environment step:

1. The deterministic actor computes `pi(s)`.
2. Form the exact reflected observation
   `mirror(s)=[cos(theta),-sin(theta),-theta_dot]` and compute
   `pi(mirror(s))` with the same actor weights.
3. Map the reflected torque back and average:
   `a_sym = 0.5 * [pi(s) - pi(mirror(s))]`.
4. Pair the original observation with 41 torques from `-2.0` through `2.0`.
5. Both learned online critics score every torque and the selector takes
   `min(Q1,Q2)` for each.
6. Choose the largest clipped value as `a_search`.
7. Compute `Q1(a_search)-Q1(a_sym)` and `Q2(a_search)-Q2(a_sym)` separately.
8. Execute `a_search` only when both differences are strictly greater than
   `0.005`; otherwise execute `a_sym`.
9. Pass that one torque to `env.step()`.

There is no angle test, episode router, policy mixture, reference query, or
online weight update at inference. The reflection average evaluates the same
actor twice; it does not combine separately trained policies. Inference adds
one actor forward pass to family B and keeps the same 41-candidate, two-critic
Q-search. It is one uniform pure-RL policy-improvement operator.

### Exact training and inference budget

Family U does not add a second training stage. It uses the official clean
SimbaV2 final checkpoints, each trained from scratch for exactly 100,000
environment steps with:

- one residual actor block of width 32;
- two residual categorical critics, each with two width-64 blocks and 51 bins
  over `[-5,5]`;
- observation normalization, input shift, feature normalization, reward
  scaling, and weight projection;
- replay capacity 100,000, batch 256, learning starts at step 1,000;
- one-step SAC, critic UTD1, `gamma=0.99`, `tau=0.005`;
- policy frequency 2 with the CleanRL two-update compensation;
- actor and critic learning rates `1e-4 -> 5e-5`;
- initial entropy temperature `0.01`, target-entropy scale `-0.5`;
- uniform replay and no reference data, SACn, hard replay/reset, model replay,
  shaping, router, or mixture.

Across five seeds the budget is 500,000 environment transitions. Family U
adds zero environment steps, reference queries, or gradient updates. At each
inference step it performs two actor forward passes and 82 critic
state-action scores (41 actions times two critics), batched in the
implementation. Family B used one actor pass and the same 82 critic scores.

### Reproduction commands

The exact per-run configurations are stored in each checkpoint directory's
`config.json`. Seeds 0--2 are under
`runs/week3_simbav2_scale_100k_20260526/simba_full_official_opt/`; seeds 3--4
are under
`runs/week3_100k_component_ablation_20260527/simba_full_official_opt/`.
Reload and evaluate them with:

```powershell
python -m last_nine_rl.pendulum_grid `
  --runs `
    runs/week3_simbav2_scale_100k_20260526/simba_full_official_opt/seed0 `
    runs/week3_simbav2_scale_100k_20260526/simba_full_official_opt/seed1 `
    runs/week3_simbav2_scale_100k_20260526/simba_full_official_opt/seed2 `
    runs/week3_100k_component_ablation_20260527/simba_full_official_opt/seed3 `
    runs/week3_100k_component_ablation_20260527/simba_full_official_opt/seed4 `
  --out reports/pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_grid `
  --theta-bins 61 --velocity-bins 41 --velocity-limit 1 `
  --device cuda --checkpoint final.pt --action-selection critic_search `
  --critic-search-num-actions 41 --critic-search-margin 0.005 `
  --critic-search-filter-mode symmetric_actor_unanimous_advantage `
  --critic-search-blend-fraction 1

python -m last_nine_rl.pendulum_relative `
  --sac-rollouts reports/pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_grid/pendulum_grid_rollouts.csv `
  --dp-grid reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv `
  --controller-grid reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_grid.csv `
  --out reports/pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative `
  --condition-label exact --epsilon-return 5
```

## Rejected training recipe: critic-UTD2 / actor-UTD1

This candidate trained a new SimbaV2 agent for 100,000 environment steps. It
preserved the official SimbaV2 architecture and changed the update
schedule:

- two critic optimizer updates per environment step (`critic UTD = 2`);
- one actor/temperature optimizer update per environment step (`actor UTD = 1`);
- policy trigger frequency 2 with one actor update per trigger, correcting the
  legacy compensated behavior that otherwise performs two actor updates per
  environment step at UTD2;
- batch 256, replay capacity 100,000, learning starts 1,000;
- `gamma=0.99`, initial actor/critic LR `1e-4`, final LR `5e-5`;
- initial alpha `0.01`, target-entropy scale `-0.5`;
- one-step targets; no SACn/FastSACN target;
- the official 1x32 actor and 2x64 categorical critics with 51 bins in
  `[-5, 5]`, reward scaling, observation normalization, and weight projection;
- zero reference replay/loss, hard replay/reset, model replay, shaping,
  symmetry augmentation, router, or mixture.

The final seed-0 checkpoint completed with 100,000 environment steps, 198,000
critic optimizer updates, zero training collapses, and zero reference-guidance
or targeted-replay use. On the preregistered 4,096-state off-grid gate, the
candidate plus frozen Q-search was worse than official Simba plus the same
Q-search in mean return (-138.0434 versus -137.5661) and task success
(92.6758% versus 92.8955%), although its bottom-10% conditional mean was
better (-261.9407 versus -262.5138). It therefore failed the gate. Seeds 1--4
were not trained, and no post-gate setting was changed.

## Rejected training recipe: compact FastSACN8/UTD2 continued to 100k

This was a newly trained model, not a 50k/50k mixture and not checkpoint
routing. The frozen recipe used the compact SimbaV2 actor (one 32-unit block),
two categorical SimbaV2 critics (two 64-unit blocks, 51 bins in `[-5, 5]`),
reward scaling, observation normalization, weight projection, uniform replay,
FastSACN with eight-step `fast_last` targets and horizon lambda `0.5`, critic
UTD2, and the legacy compensated two actor updates per environment step. It
used no reference data, hard-state replay, model replay, reward shaping,
router, or policy mixture.

Seed 0 completed exactly 100,000 environment steps and 198,000 critic updates.
Only the final checkpoint was evaluated. On 4,096 preregistered continuous
off-grid states (RNG `19079100`), the actor plus the already frozen 41-action,
margin-0.005 unanimous Q-search rule produced:

| Seed-0 policy | Mean return | Bottom-10% mean | Task success |
|---|---:|---:|---:|
| New compact FastSACN8 100k + Q-search | **-139.3469** | **-258.5620** | 92.5537% |
| Official SimbaV2 100k + Q-search | -140.2712 | -264.3653 | 91.7480% |
| Clean compact FastSACN8 50k + Q-search | -140.3076 | -266.2666 | **92.9932%** |

The new model improved mean and bottom-tail return over both baselines and
improved task success over official SimbaV2 by `0.8057` percentage points.
It nevertheless trailed the matched clean 50k FastSACN baseline by `0.4395`
task-success points, violating the frozen gate that required nonnegative
changes in all three metrics against both baselines. Seeds 1--4 were therefore
not trained and the reference grid stayed sealed for this family.

## Rejected training recipe: FastSACN8-to-one-step curriculum

This was one continuously trained compact SimbaV2 model, not two checkpoints
or a 50/50 inference mixture. Steps 1--50,000 used FastSACN8 `fast_last`
targets with horizon lambda `0.5`; steps 50,001--100,000 used ordinary
one-step SAC targets on the same actor, critics, optimizers, and replay. The
recipe used critic UTD2 and the legacy compensated two actor updates per
environment step. The event log verifies that SACn fields are present through
step 50,000 and absent from the first post-boundary optimizer event at step
51,000.

The mandatory final seed-0 checkpoint was evaluated on 4,096 fresh continuous
states (RNG `19087100`) with the frozen family-B Q-search:

| Seed-0 policy | Mean return | Bottom-10% mean | Task success |
|---|---:|---:|---:|
| FastSACN8-to-one-step 100k + Q-search | **-138.4016** | **-256.5138** | 92.1631% |
| Official SimbaV2 100k + Q-search | -138.8924 | -261.8453 | 92.7979% |
| Clean compact FastSACN8 50k + Q-search | -138.9479 | -262.4541 | **93.9209%** |

The curriculum improves mean and failure-tail return but loses `0.6348` task
points to official SimbaV2 and `1.7578` points to clean 50k FastSACN. It fails
the all-metrics gate, so seeds 1--4 were not trained and no intermediate
checkpoint was substituted.

## Rejected inference recipe: reflection-averaged Q-search

The symmetry candidate averages each learned critic with the same critic at
the exact mirrored state/action:

```text
Q_i_sym(s,a) = 0.5 * [Q_i(s,a) + Q_i(mirror(s),-a)]
```

It then applies the unchanged 41-action, margin-0.005 unanimous Q-search. This
uses one checkpoint and a uniform rule, with no reference or state router.
Reference-free development selected it over actor-only and combined
actor/critic symmetrization by worst-seed mean-return improvement.

On untouched seeds 3--4 it improved mean return by `+1.0265` and `+1.6194`
and bottom-tail return by `+6.1206` and `+5.5506` versus the actors. However,
pooled task success was `92.6880%`, below ordinary Q-search at `92.9077%`.
It therefore failed before authority evaluation. This is another measured
case where a stronger continuous-return statistic did not improve categorical
task success.

## Rejected training recipe: symmetry-augmented SimbaV2

Candidate T trained a new official-size SimbaV2 model for exactly 100,000
environment steps. Each 256-transition replay minibatch was paired with its
exact reflected transitions inside the optimizer call, producing an effective
batch of 512. Environment steps, replay draws, optimizer triggers, networks,
and all ordinary one-step SAC hyperparameters otherwise matched the official
recipe. The run recorded 99,000 critic-update triggers and zero reference,
hard-state, model-replay, router, or mixture use.

On the frozen 4,096-state seed-0 gate with ordinary family-B Q-search, the new
model had mean return `-140.620932`, bottom-10% mean `-267.599865`, and task
success `92.602539%`. Official SimbaV2 scored `-140.389432`, `-267.117296`,
and `92.138672%`; clean 50k FastSACN scored `-140.328000`, `-267.418496`, and
`93.261719%`. The new model improved task success over official SimbaV2 but
lost all three metrics to clean FastSACN and both return metrics to official
SimbaV2. Seeds 1--4 were not trained and no authority query was made.

## Rejected training recipe: L2-feature SAC

The archive contained one 100k L2-feature SAC seed at 97.1212% near-reference
success. That number was not treated as five-seed evidence. The archived
implementation was recovered exactly: after each 256-unit ReLU layer in both
the actor and each scalar critic, it applies
`normalize(x, p=2, dim=1) * sqrt(256)`. Deterministic rollouts from the current
loader reproduce the archived rollout returns exactly.

Five new seeds were then trained from scratch with one frozen recipe and final
100k checkpoints: plain twin-critic SAC, two 256-unit layers, uniform 100,000
transition replay, batch 256, learning starts 5,000, `gamma=0.99`, `tau=0.005`,
actor LR `3e-4`, critic LR `1e-3`, automatic entropy temperature initialized
at 1.0, UTD1, and policy frequency 2 with the CleanRL two-update compensation.
Every seed records 100,000 environment steps and 95,000 critic, actor, and
temperature optimizer updates. The final policy is the deterministic actor;
Q-search was excluded before training because it worsened all three frozen
reference-free gate metrics on the archived seed.

The pooled authoritative result is:

| Seed | Near reference | Task success | Strict wins | Mean return |
|---:|---:|---:|---:|---:|
| 0 | 2,335 / 2,501 | 2,201 | 51 | -140.7373 |
| 1 | 2,289 / 2,501 | 2,193 | 180 | -142.3114 |
| 2 | 2,294 / 2,501 | 2,246 | 75 | -141.4847 |
| 3 | 2,267 / 2,501 | 2,243 | 61 | -141.5744 |
| 4 | 2,247 / 2,501 | 2,225 | 76 | -142.3187 |
| **All** | **11,432 / 12,505 (91.419432%)** | **11,108 / 12,505 (88.828469%)** | **443 / 12,505 (3.542583%)** | **-141.6853** |

This clean replication fails the goal by 368 near successes and trails the
official SimbaV2 actor by 52. The old 97.1212% result was seed-specific, not a
robust recipe result. No checkpoint, seed, or inference adjustment was made
after the authority result.

## Rejected inference recipe: target-critic proposal

This candidate kept the official clean SimbaV2 actors and changed only the
fixed Q-search operator. The candidate torque maximized the minimum of the two
slowly updated target critics over the same 41 torques. It replaced the actor
only if both online critics and both target critics each rated that torque more
than `0.01` above the actor torque. The acceptance rule and margin were selected
by the documented off-grid protocol, with target-only, online-only, and
all-critic acceptance and four margins preregistered before results.

All 12 Phase-B variants had nonnegative mean and tail deltas on development
seeds 0--2. The frozen all-critic/margin-0.01 rule had a `+0.472977` worst-seed
mean delta, `+0.522365` pooled mean delta, and `+2.745147` worst-seed tail
delta. Untouched seeds 3 and 4 also passed, with mean deltas `+1.070179` and
`+0.775495`. Nevertheless, its one-shot grid result was only
`11,585 / 12,505 = 92.642943%` near reference,
`11,561 / 12,505 = 92.451020%` task success, and
`1,124 / 12,505 = 8.988405%` strict wins. It is 247 near successes below the
current leader and is also lower on task and strict wins. No rule was retuned
after this result.

## Rejected inference recipe: disagreement-penalized Q-search

This candidate also kept the official clean SimbaV2 actors. For each of 41
torques it computed
`mean(Q1,Q2) - 0.75 * abs(Q1-Q2)`, chose the largest score, and executed that
torque only when **each** online critic valued it more than `0.01` above the
actor torque. Otherwise it executed the deterministic actor torque. The
disagreement penalty, margin, and acceptance rule are one global algorithm;
there is no initial-state router or manual region.

Nine penalty/margin combinations were preregistered. The selected rule had
`+0.455270` worst-seed mean-return delta, `+0.571642` pooled mean delta, and
`+2.980289` worst-seed bottom-10% delta on development seeds 0--2. It then
passed untouched seeds 3 and 4 with mean deltas `+0.701999` and `+0.481371`
and tail deltas `+4.426725` and `+3.494356`. Only then was the reference
unsealed.

Its one-shot authority result was
`11,586 / 12,505 = 92.650940%` near reference,
`11,555 / 12,505 = 92.403039%` task success,
`1,136 / 12,505 = 9.084366%` literal strict wins, and mean return
`-140.900604`. It misses the formal target by 214 near successes and trails
the current leader by 246. The rule was not retuned after scoring.

## Rejected inference recipe: task-selected global margin

This family kept the leader's networks, 41-action min-Q proposal, and
two-critic unanimous acceptance. It selected one global margin from
`{0.0075, 0.0100, 0.0125, 0.0150}` using fresh continuous states; margin
`0.005` was a non-promotable control. The preregistered three-seed ordering
selected `0.0100`: worst-seed task improvement was `+0.268555` percentage
points and pooled task success was 92.936198%, versus 92.561849% for control.

On untouched seeds 3--4, however, candidate task success pooled to 92.785645%
versus 93.017578% for control. The margin therefore failed before any
reference query. No alternate margin was substituted.

## Rejected inference recipe: actor-centered trust region

This family kept the complete leader Q-search rule and rejected any accepted
proposal farther than a single global torque radius from the actor action.
Radii were `{0.10, 0.25, 0.50, 1.00, 2.00}`. Only radius 2.0 passed the first
return/tail screen. In the three-seed screen it was worse than uncapped
Q-search on every seed: mean-return changes were `-0.466406`, `-0.273615`, and
`-0.631174`; bottom-tail changes were `-2.342438`, `-2.587611`, and
`-4.357020`. It stopped before holdout/reference evaluation.

Although only about 0.19%--0.23% of decisions exceeded the radius, those rare
large deviations supplied much of Q-search's gain. A hard distance cap is not
part of the retained recipe.

## Rejected inference recipe: interpolated critic proposal

For each candidate action, this rule scored
`mean(Q1,Q2) - c * abs(Q1-Q2)`. The fixed candidates were
`c in {0.125, 0.250, 0.375}`, between the completed mean (`c=0`) and min
(`c=0.5`) controls. Fresh three-seed development selected `c=0.25` by the
largest worst-seed mean-return gain (`+0.625838`). It passed untouched seeds
3--4 with positive mean/tail gains and higher pooled task success than both
controls.

The one-shot authority result was nevertheless
`11,737 / 12,505 = 93.858457%` near reference,
`11,539 / 12,505 = 92.275090%` task success,
`1,111 / 12,505 = 8.884446%` literal strict wins, and mean return
`-140.709499`. It is 95 near successes below the current leader and 63 near
successes short of the formal target. The coefficient was not changed
afterward.

## Rejected inference recipe: disagreement-increase acceptance penalty

This rule kept the min-Q proposal and subtracted
`k * max(0, disagreement(proposal) - disagreement(actor))` from the unanimous
advantage before applying margin `0.005`. Coefficients
`{0.25, 0.5, 1.0, 2.0}` were screened with `k=0` as control. The predetermined
three-seed ordering selected `k=0.5` by worst-seed mean-return gain.

It failed untouched seeds 3--4: pooled task success was 93.041992%, versus
93.115234% for the unpenalized control. The family stopped without an
authority query or coefficient substitution.

## Heatmaps

### Current best near-reference map

![Reflected actor plus fixed unanimous Q-search: near-reference success](authority_simba100k_symmetric_actor_q41m005_unanimous_relative/near_best_known_return_eps_map.png)

### Current best task-success map

![Reflected actor plus fixed unanimous Q-search: task success](authority_simba100k_symmetric_actor_q41m005_unanimous_relative/task_success_map.png)

### Current best literal strict-win map

![Reflected actor plus fixed unanimous Q-search: literal strict wins](authority_simba100k_symmetric_actor_q41m005_unanimous_relative/beats_best_known_return_map.png)

### Clean FastSACN clipped-Q near-reference map

![Clean FastSACN plus clipped Q-search: near-reference success](audit_clean_fastsacn8_utd2_q41m005_clipped_relative/near_best_known_return_eps_map.png)

### Clean FastSACN clipped-Q task-success map

![Clean FastSACN plus clipped Q-search: task success](audit_clean_fastsacn8_utd2_q41m005_clipped_relative/task_success_map.png)

### Rejected L2-feature SAC near-reference map

![Five-seed L2-feature SAC: near-reference success](authority_l2_feature_norm_100k_5seed_relative/near_best_known_return_eps_map.png)

### Rejected L2-feature SAC task-success map

![Five-seed L2-feature SAC: task success](authority_l2_feature_norm_100k_5seed_relative/task_success_map.png)

### Rejected target-proposal Q-search near-reference map

![Target-proposal Q-search: near-reference success](authority_simba100k_targetproposal_q41m01_allunanimous_relative/near_best_known_return_eps_map.png)

### Rejected target-proposal Q-search task-success map

![Target-proposal Q-search: task success](authority_simba100k_targetproposal_q41m01_allunanimous_relative/task_success_map.png)

### Rejected disagreement-penalized Q-search near-reference map

![Disagreement-penalized Q-search: near-reference success](authority_simba100k_lcb025_q41m01_unanimous_relative/near_best_known_return_eps_map.png)

### Rejected disagreement-penalized Q-search task-success map

![Disagreement-penalized Q-search: task success](authority_simba100k_lcb025_q41m01_unanimous_relative/task_success_map.png)

### Rejected interpolated-proposal near-reference map

![Interpolated-proposal Q-search: near-reference success](authority_simba100k_mid025_q41m005_unanimous_relative/near_best_known_return_eps_map.png)

### Rejected interpolated-proposal task-success map

![Interpolated-proposal Q-search: task success](authority_simba100k_mid025_q41m005_unanimous_relative/task_success_map.png)

## Artifact index

- Full preregistration and experiment ledger: `experiment_ledger.md`
- Current leader grid and rollout CSVs:
  `authority_simba100k_symmetric_actor_q41m005_unanimous_grid/`
- Current leader reference-relative metrics and heatmaps:
  `authority_simba100k_symmetric_actor_q41m005_unanimous_relative/`
- Exact current-leader failures:
  `family_u_remaining_{near_reference,task}_failures.csv`
- Previous family-B leader:
  `authority_simba100k_q41m005unanimous_{grid,relative}/`
- Corrected clean FastSACN clipped-Q result:
  `audit_clean_fastsacn8_utd2_q41m005_clipped_{grid,relative}/`
- Clean FastSACN unanimous-Q result:
  `authority_clean_fastsacn8_utd2_q41m005_unanimous_{grid,relative}/`
- Target-veto result:
  `authority_simba100k_q41m01_online_target_unanimous_{grid,relative}/`
- Four-critic joint result:
  `authority_simba100k_q41m005_joint_online_target_unanimous_{grid,relative}/`
- Mean-proposal result:
  `authority_simba100k_meanproposal_q41m005_unanimous_{grid,relative}/`
- Five-seed L2-feature SAC replication:
  `authority_l2_feature_norm_100k_5seed_{grid,relative}/`
- Target-critic-proposal Q-search:
  `authority_simba100k_targetproposal_q41m01_allunanimous_{grid,relative}/`
- Disagreement-penalized Q-search:
  `authority_simba100k_lcb025_q41m01_unanimous_{grid,relative}/`
- Interpolated-proposal Q-search:
  `authority_simba100k_mid025_q41m005_unanimous_{grid,relative}/`
