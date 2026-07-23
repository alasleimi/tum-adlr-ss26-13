# Pure-RL gap: causal diagnostic memo

Date: 2026-07-22

## Bottom line

The current evidence does **not** support a single explanation such as dead
networks, an insufficient Bellman horizon, or a uniformly bad critic. The
best-supported explanation is a coupled failure:

1. Pure RL receives a much weaker training signal on rare recovery states than
   the supervised + DAgger pipeline and optimizes reward rather than direct
   reference actions.
2. Its critic usually orders nearby actions correctly, but its twin-critic
   uncertainty is larger than the local action signal and it has severe
   state-specific rank reversals near the downward/wrap boundary.
3. The actor becomes almost bang-bang on those hard states while entropy
   pressure collapses. This makes the action correction gradient through the
   tanh mean extremely small.
4. Shared actor-parameter updates are not aligned with real rollout return even
   when statewise local action ranks look good. The frozen critic objective is
   anti-aligned with return on all three measured parameter-space slices.
5. Reflection and conservative Q-search repair some errors, but raw critic
   asymmetry and rare large Q-ranking failures remain. Their average gains are
   carried by large recoveries and coexist with many small per-state harms.

The largest evidential caveat is causal identification: all five pure-RL TOP5
rows reuse one trained checkpoint family, while the hybrid differs in training
signal, state coverage, actor capacity, and deployment critic. Thus the
mechanisms below are strong fixed-checkpoint diagnostics and matched deployment
effects, not yet retraining-level causal estimates. The queued one-factor
ablations are necessary to separate them.

## 1. What gap actually needs explaining

The hybrid leader has 12,496/12,505 near-reference successes, 11,737 task
successes, 1,570 strict wins, and mean return -138.647924. The pure-RL leader
has 11,832/12,505, 11,567, 2,303, and -140.620617, respectively. Therefore:

| Outcome | Hybrid minus pure RL | Interpretation |
|---|---:|---|
| Near reference | +664 trials / +5.3099 pp | The dominant gap |
| Task success | +170 trials / +1.3595 pp | Much smaller than the reference gap |
| Mean return | +1.972693 | Hybrid has the better lower-tail average |
| Strict wins | -733 trials / -5.8617 pp | Pure RL beats the reference more often |

This is not simply “pure RL cannot solve Pendulum.” Pure RL has nearly the
same task-success rate and substantially more strict wins, but it is less
uniform and has a much heavier failure tail. Direct reference labels naturally
reduce that variance and optimize the primary evaluation criterion. The TOP5
training/deployment audit supporting this comparison is
[`TOP5_COMPONENT_MATRIX.md`](../systematic_100k_budget_best_20260722/programmatic_inventory/TOP5_COMPONENT_MATRIX.md).

Cell consistency makes the same point. For pure RL, only 90.164% of the 2,501
grid cells succeed for all five seeds, while 98.441% succeed for at least one
seed. Equivalently, 246 cells fail for at least one seed but only 39 fail for
all seeds. For the hybrid these numbers are 99.680% and 100%. Much of the pure
gap is therefore training-seed instability on states that are demonstrably
solvable by the same architecture family, not a universal unreachable basin.
The source is the pure
[`relative_summary.json`](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/relative_summary.json)
and the hybrid
[`relative_summary.json`](../systematic_100k_budget_best_20260722/hybrid_qsearch/relative/relative_summary.json).

## 2. Heat-map anatomy: one broad recovery deficit plus a small precision mode

The pure leader has 673 near-reference failures and 938 task failures. Their
angle distribution is:

| Region | Near failures | Share of near failures | Near-failure rate in region | Task failures | Share of task failures | Task-failure rate in region |
|---|---:|---:|---:|---:|---:|---:|
| abs(theta) < 60 deg | 44 | 6.54% | 1.07% | 128 | 13.65% | 3.12% |
| 60 <= abs(theta) < 120 deg | 0 | 0% | 0% | 0 | 0% | 0% |
| 120 <= abs(theta) < 150 deg | 140 | 20.80% | 6.83% | 94 | 10.02% | 4.59% |
| abs(theta) >= 150 deg | 489 | 72.66% | 21.69% | 716 | 76.33% | 31.75% |

Thus 629/673 = 93.46% of near-reference failures and 810/938 = 86.35% of
task failures occur at abs(theta) >= 120 degrees. The most difficult registered
subregion is abs(theta) >= 150 degrees and abs(theta_dot) <= 0.5: near-reference
success is 75.325% and task success is 67.186%, versus 94.618% and 92.499%
overall. The entire 60--120 degree band is perfect on both criteria.

Velocity does not reduce the near-reference errors to one narrow cell strip:
among the 673 failures, 200 have abs(velocity) <= .25, 150 are in (.25,.5],
164 in (.5,.75], and 159 in (.75,1]. This is a broad recovery-policy problem,
with its largest concentration at downward/low-speed states, plus a smaller
near-upright timing/precision problem.

The published failure shares are in
[`failure_region_diagnostics.csv`](../systematic_100k_budget_best_20260722/failure_region_diagnostics.csv).
The rates and velocity counts above are direct re-tabulations of the same
authoritative
[`relative_rollouts.csv`](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/relative_rollouts.csv);
they do not introduce another evaluation surface.

There is also residual left/right asymmetry even though the fallback actor is
reflection-projected: 425 of the 673 near failures are on negative-angle grid
rows and 248 on positive-angle rows. This is consistent with the fact that the
actor fallback is symmetrized but the online critics that decide Q-search
acceptance are not exactly symmetric.

## 3. Critic quality: decent typical ordering, unreliable tail and weak signal-to-uncertainty

The training-aligned five-seed diagnostic uses 99 hard states, 21 local actions
within +/-0.5, and a 200-step counterfactual rollout. Aggregate results are:

| Diagnostic | Five-seed mean | Range |
|---|---:|---:|
| Mean per-state Spearman | .6931 | .5825--.8217 |
| Pairwise action-order accuracy | .8437 | .7891--.9067 |
| Within-state centered Pearson | .3901 | .2453--.6833 |
| Exact Q/true local argmax agreement | .7758 | .6970--.8586 |
| Q-selected raw gain over actor | +.0102 | -.1042--+.1469 |
| Q selection harms actor | 18.99% | 11.11%--25.25% |
| Twin disagreement / median local Q range | 2.426 | 1.717--2.908 |
| Median predicted/true-scaled range ratio | 1.396 | 1.212--1.529 |

The average rank numbers are real evidence that the critic contains useful
local action information. They are not evidence that it supplies a safe actor
gradient. First, critic disagreement is 2.43 times the entire median local Q
range. Second, the per-state rank distribution is sharply heterogeneous: its
median is 1.0, but the mean p10 is -.510. Third, 41.7% of the candidate actions
are duplicates because the saturated actor is close to an action bound and
the +/-0.5 sweep is clipped. These easy monotone/duplicate states inflate the
typical rank, while a small set of near-down rank reversals determines the
tail. Mean Q-selected regret is .1917 raw return despite the near-zero mean
gain.

The actor action itself is never the exact true local argmax on this hard-state
probe, while the Q-selected action is the exact argmax 77.6% of the time. Q
selection improves 79.6% of states but harms 19.0%; the improvements are
usually tiny and the rarer mistakes cancel them. Examples in the full JSON
include exact or near-exact reversed rankings at theta = +/-180 and around
+/-168 degrees.

The complete source, including every state/action series and worst example, is
[`pure_rl_gap_diagnostics.json`](pure_rl_gap_diagnostics_training_aligned_v2/pure_rl_gap_diagnostics.json),
with the compact readout in
[`pure_rl_gap_diagnostics.md`](pure_rl_gap_diagnostics_training_aligned_v2/pure_rl_gap_diagnostics.md).

### The Bellman-target definition is not the main mismatch

A separate probe extends the horizon to 917 steps, where gamma^H =
9.942e-5, and evaluates the checkpoint policy under both deterministic
reward-only and stochastic soft returns in training units. Deterministic
Spearman is .7217; stochastic-soft Spearman is .7112 and pairwise accuracy is
.8418. The stochastic-soft long-versus-200-step absolute difference is only
4.836e-5 on average.

These values are close to the finite task probe, so a finite-episode versus
continuing-soft objective mismatch does not explain most of the gap. This does
not make the critic exact: centered calibration remains weak and the
training-aligned probe has only 16 states, 9 actions, and 8 Monte Carlo samples.
It rules down a specific explanation; it does not rule out critic
approximation, off-policy, or distribution errors.

## 4. Actor geometry and learning dynamics

On the hard-state probe, 93.13% of deterministic actor actions lie at or beyond
99.5% of an action bound. Median absolute pre-tanh mean logit is 4.581 and the
median normalized tanh derivative is 4.339e-4; 93.13% of probed states have a
normalized derivative below .01. This is not automatically pathological:
near-bang-bang torque is sensible during swing-up. It becomes a problem when a
state needs a small timing correction or the opposite bound, because a useful
critic action gradient is attenuated almost completely before it reaches the
mean logit.

For the two seeds with full 10k-spaced histories, alpha falls from about .0046
at 10k to 1.94e-5 and 2.46e-5 at 100k, while estimated policy entropy falls
from .94/.92 to -.42/-.47. Across all five final checkpoints alpha averages
1.96e-5 (range 1.43e-5--2.46e-5). Actor loss is already near its plateau by
50--70k (about .10), while Q loss continues to fall from roughly 4.14 at 50k
to 1.27 at 100k.

This pattern is consistent with, but does not prove, a feedback loop: entropy
pressure disappears, the actor saturates, and later critic maturation cannot
move the deterministic mean effectively. Alpha decay could be a consequence
rather than the original cause of saturation, and the actor/Q loss scales are
nonstationary. The correct causal tests are the alpha-floor/exploration
factorial, direct logit regularization, delayed actor updates, and logit-space
Q distillation. Curves are in
[`training_curves.csv`](pure_rl_gap_diagnostics_training_aligned_v2/plots/training_curves.csv)
and
[`training_curves.png`](pure_rl_gap_diagnostics_training_aligned_v2/plots/training_curves.png).

## 5. Why local action ranking and the actor-parameter landscape disagree

The seed-0 actor was perturbed on three independent 5x5 planes with radius 1%
of the full actor parameter norm. Frozen-Q objectives use 4,096 fixed states;
paired real rollouts use 256 fixed starts. Across the three planes:

| Direction seed | Q/return Spearman | Projected critic-loss / real-loss gradient cosine | Return regret of Q-selected cell |
|---:|---:|---:|---:|
| 22072611 | -.4408 | -.9365 | 1.2169 |
| 22072612 | -.8477 | -.7831 | 1.1874 |
| 22072613 | +.6154 | -.7274 | .0156 |
| Mean | -.2244 | -.8157 | .8066 |

The actual-return range across a slice is .488--1.264, while the frozen min-Q
range is only .000138--.000555. In two slices, the cell preferred by Q reduces
real return by .781 and .631 relative to the checkpoint. The checkpoint is
neither the best-Q nor best-return cell on any slice.

There is no contradiction with the action-space ranks:

- The action probe changes one action at one state, then holds the future actor
  fixed. A parameter update changes actions at thousands of states and changes
  all future actions.
- The local ranks are dominated by easy clipped/monotone cases, whereas the
  policy gradient sums small systematic errors across shared parameters.
- Twin disagreement exceeds the local signal, and the parameter-plane Q range
  is much smaller again. An optimizer can consistently exploit approximation
  error even when most pairwise ranks are correct.
- Saturated means make the parameter-to-action Jacobian highly anisotropic, so
  the few unsaturated states and shared-feature directions dominate the update.

This is the strongest direct evidence for why “just put more SAC loss into the
actor” can degrade the policy. The landscape is nevertheless one checkpoint,
three random planes, and one rollout set; it needs repetition after every
candidate intervention. The index is
[`parameter_landscape_index.md`](pure_seed0_actor_parameter_landscape/parameter_landscape_index.md)
and each underlying table/plot is under
[`pure_seed0_actor_parameter_landscape/`](pure_seed0_actor_parameter_landscape/).

## 6. Dormancy and effective rank: no dead-network diagnosis

No actor or critic layer is dormant at the registered relative-activation
threshold on the hard actor manifold, replay samples, or broad uniform
support. The often-quoted 4.77% minimum rank includes the critic input
embedder; its centered rank is structurally limited by the low-dimensional
state-action input and is not evidence of collapse.

For critic hidden layers, minimum effective-rank fraction is .153 on hard
actor states, .265 on replay samples, and .296 on broad uniform state-actions.
For the actor it is .074, .109, and .109. All units can be active while their
activations are strongly correlated, so zero dormancy and low effective rank
are mathematically compatible. The large increase under broader probes says
that much of the low hard-state rank is manifold/probe geometry, amplified by
the nearly constant saturated action, rather than globally dead features.

The defensible conclusion is: neuron death is ruled down; low-dimensional and
correlated representations specifically on the failure manifold remain a
possible contributor, not an established cause. Per-layer values for all
three distributions are in the training-aligned diagnostic JSON.

## 7. Symmetry and Q-search: helpful projections, not a complete cure

The raw actor violates Pendulum reflection equivariance by .243 action units
on average across seeds (range .130--.381), with rare errors close to the full
four-unit action span. Mean critic reflection error is .0342 Q units. Although
that absolute number looks small, it is on average 6.91 times the median local
Q range, so it can reverse a Q-search gate.

On the authoritative grid, adding the reflection-projected fallback to the
same 41-action unanimous Q-search fixes 126 near-reference trials and breaks
33, for +93 net; task is +27, strict wins +1,197, and mean return +.1108. This
is a matched deployment effect and is the clearest existing evidence that
symmetry is useful. It does not show that reflection actor alone is better.

Indeed, on the 99 hard diagnostic states the reflected actor alone is roughly
16 return points worse than the ordinary actor. Q-search then gains +16.829
relative to that reflected fallback but only +.697 relative to the ordinary
actor. It harms the reflected fallback on 45.45% of states, yet harms it by
more than five return points on only 3.84%. The mean gain is therefore carried
by a small number of large trajectory recoveries. Q-search switches on only
.352% of all trajectory decisions, but on 9.49% of first decisions; a single
early recovery action can change the whole rollout.

The broader authority comparison tells the same heavy-tail story. Adding
global unanimous Q-search to the ordinary actor fixes 374 near-reference
trials and breaks 119 (+255 net), raises task by 103 and mean return by .907,
but improves return on only 53.41% of trials and has near-zero median return
delta. In an unfiltered continuous probe on seeds 3 and 4, always taking the
global Q argmax degrades 75.98% and 79.30% of states and drops task success
from .920 to .834 and from .939 to .898. Conservative unanimity/margin gates
are doing essential work.

The matched authority counts are in
[`paired_ablation_diagnostics.csv`](../systematic_100k_budget_best_20260722/paired_ablation_diagnostics.csv).
The continuous symmetry variants are in
[`pure_family_symmetry_seeds34_4096.json`](pure_family_symmetry_seeds34_4096.json),
and the unfiltered control is
[`pure_unconditional_qsearch_seeds34_512_cpu.json`](pure_unconditional_qsearch_seeds34_512_cpu.json).

## 8. FastSACN horizon weights: the old “8-step” setting is nearly one-step

The corrected horizon diagnostic shows what `fast_last`, lambda=.5 actually
optimizes. Only horizon 1 and the selected endpoint are active. At horizon 8,
the endpoint has relative weight .5^7 = .0078125 and only .7752% of the nominal
loss share; 99.2248% belongs to the one-step endpoint. With density importance
weights, the observed horizon-8 share falls further to .1259% overall. In the
hard-down subgroup it is 9.87e-6 (0.000987%), and every hard-down endpoint
weight is below 1e-3.

Consequently, historical FastSACN8 lambda=.5 should not be credited as a
strong multi-step solution to the pure-RL recovery gap. Its behavior is much
closer to one-step TD with a tiny long-endpoint auxiliary. Lambda=1 and `all`
target controls are scientifically necessary before concluding that n-step
credit does or does not help. The corrected sources are
[`sacn_horizon_summary.csv`](sacn_horizon_corrected_none_lambda05_4096/sacn_horizon_summary.csv)
and the density-weighted
[`sacn_horizon_summary.csv`](sacn_horizon_corrected_density_lambda05_4096/sacn_horizon_summary.csv).
These probes use dedicated 100k FastSACN checkpoints rather than the P-A
leader checkpoints; the weight arithmetic is configuration-level evidence,
while their target values are run-specific.

## 9. Causal ranking and falsification tests

| Hypothesis | Current status | Decisive matched test | What would falsify it |
|---|---|---|---|
| Missing hard-state coverage/reference anchoring is primary | High plausibility: heat-map localization, seed instability, and large near-vs-task gap | Strict-budget reward-only failure curriculum versus clean controls at identical learning steps; record replay occupancy by region | No improvement in fresh hard-down/off-grid tail despite materially increased occupancy |
| Saturation plus vanishing mean gradient blocks repair | High plausibility, fixed-checkpoint evidence | Logit-L2 weights, alpha floor/exploration factorial, and log-prob Q distillation | Saturation/derivative changes substantially but hard-region return and parameter alignment do not |
| Approximate critic policy gradient is misaligned | Strong seed-0 diagnostic, not multi-seed causal proof | Delayed actor, deterministic-mean versus stochastic actor objective, min versus mean Q, REDQ; rerun parameter planes | Interventions improve the actor without improving held-out gradient/landscape alignment, or anti-alignment vanishes on repeat baseline seeds |
| Symmetry generalization is a material error source | Proven useful at deployment; training mechanism unresolved | Actor-only, critic-only, both, data augmentation, then Q-distill interaction | Exact learned symmetry reduces errors but not fresh-seed hard-region performance |
| True multi-step targets solve recovery credit | Untested by old lambda=.5 setting | Matched one-step versus SACn lambda=1 with identical UTD, then `fast_last` versus `all` | Properly weighted long targets change target contribution but not recovery/tail metrics |
| Network dormancy causes the gap | Currently weak / ruled down | Recheck only as a secondary diagnostic on winners | Already contradicted by zero dormancy across three probe distributions |
| 200-step versus continuing-soft target mismatch causes the gap | Currently weak / ruled down | No expensive ablation warranted before higher-priority tests | Already contradicted by similar finite/long ranks and negligible discounted tail |
| Actor capacity explains the hybrid advantage | Plausible confound, no current matched result | Pure 32x1 versus 64x2 from scratch with all else fixed | 64x2 leaves fresh-seed primary/tail unchanged |

For every promoted arm, the diagnostic bundle should include: region-stratified
fresh continuous rollouts; actor logits and tanh derivatives; alpha/entropy and
actor/Q curves; twin disagreement divided by local Q range; finite and
training-aligned action ranks; symmetry errors; replay/broad-support rank and
dormancy; and repeated parameter landscapes. A component should be called
causal only if its matched retraining ablation changes the predicted mediator
and the held-out outcome in the predicted direction.

## 10. Integrated interpretation

The mixed system's advantage is best understood as **robust reference-shaped
coverage**, not proof that its FastSACN critic or a separate training stage is
intrinsically superior. DAgger supplies dense, low-variance corrective actions
at visited failures and a much broader supervised state set; the 64x2 actor can
represent them without depending on a noisy critic gradient. Pure SAC instead
must infer those corrections through bootstrapping, then transmit a small and
sometimes wrong Q difference through a nearly saturated 32x1 actor. Its critic
is good enough for conservative, rare Q-search corrections but not reliable
enough to optimize the entire shared actor unchecked.

This reconciles all apparently conflicting evidence:

- **Good local Q rank but bad actor landscape:** one-state action ordering is
  easier than an aggregate shared-parameter improvement direction.
- **Zero dormancy but low rank:** units are active yet correlated; the hard
  policy manifold is low-dimensional, while rank rises on broad support.
- **Positive Q-search mean gain but frequent harm:** rare early corrections
  have large trajectory value, whereas most harms are small; comparator choice
  also makes gain versus the weak reflected actor look much larger than gain
  versus the ordinary actor.
- **Reflection helps authority but hurts alone on the hard probe:** reflection
  fixes rare equivariance errors, while Q-search veto/correction is needed to
  prevent its averaging from damaging other hard states.
- **FastSACN appears in the hybrid but does not explain its long-horizon edge:**
  lambda=.5 gives the eight-step endpoint less than one percent of the loss.

The scientifically highest-value pure-RL route is therefore not one more
inference formula. It is a matched training program that improves hard-state
coverage, preserves exploration long enough for the critic to mature, keeps
the actor mean trainable near the bounds, enforces symmetry in the learned
components, and only then tests properly weighted multi-step targets and
critic ensembles.

## 11. Queue changes implied by the diagnosis

The live pure-RL runner now contains the missing causal tests rather than only
high-level stage recipes:

- exact-dose early/late actor windows plus a half actor/temperature-dose arm;
- the missing REDQ4-by-actor-mean cell;
- clean corrected SACn `fast_last`, lambda=1 and `all`, lambda=1 comparisons at
  UTD1, with matched one-step/SACn UTD2 controls;
- PER with log-probability Q distillation, logit L2 with log-probability Q
  distillation, and critic symmetry with log-probability Q distillation;
- learning-step-matched 80k controls for curriculum-by-logit and
  curriculum-by-PER interactions; and
- corrected-SACn-by-logit and corrected-SACn-by-curriculum interactions, with
  a separate 80k clean corrected-SACn control for the latter; and
- actor-only and critic-only symmetry pilots before either combined pilot; and
- 10k actor64x2 and critic128x2 capacity pilots, each matched directly to the
  clean 10k control, before spending a full-run budget on backbone transfer.

The completed clean FastSACN8 `fast_last`, lambda=.5 UTD1 checkpoint is reused
as historical evidence rather than pointlessly retrained. It is not a substitute
for the new lambda=1 controls.

Crucially, these definitions are not a mandate to run an eighty-arm Cartesian
screen. Normal runner execution stops at a matched 10k pilot gate, a separate
switch runs only the 25k actor-timing/curriculum block, and each 100k
confirmation requires an explicit single-arm promotion. The preregistered
mediators, promotion rules, exact commands, and final metric order are in
[`PURE_RL_PROMOTION_PROTOCOL.md`](PURE_RL_PROMOTION_PROTOCOL.md). The generated
[`queued experiment audit`](queued_experiment_audit/REPORT.md) verifies all 85
pure-RL definitions, 18 pilot comparisons, 30 full-horizon causal comparisons,
the 14 pilot/SACn controls, strict budgets, and the 14,002/89,002 matched actor
optimizer-step counts.
