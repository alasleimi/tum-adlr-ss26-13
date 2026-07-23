# TOP5 component matrix: actual training recipes and deployment variants

This audit joins the new programmatic TOP5 inventory back to the stored run
configs, distillation summaries, checkpoint paths, authoritative summaries, and
rollout tables. The full flat matrix is in
[`top5_component_matrix.csv`](top5_component_matrix.csv); the normalized
families, evaluations, budget semantics, and queued-transfer map are in
[`top5_component_matrix.json`](top5_component_matrix.json).

## The two most important corrections

First, **the clean DAgger actor is a SimbaV2 model**. It is not an MLP or a
separate lightweight supervised network. It has width 64 and two residual
blocks, with SimbaV2 observation normalization, input shift, feature
normalization, HyperDense layers, and weight projection. The clean pure-RL
actor is the smaller SimbaV2 configuration: width 32 with one residual block.

Second, the ten TOP5 rows are not ten training recipes:

- The hybrid TOP5 contains **three trained actor families**. Ranks 1, 3, and 5
  use exactly the same H-A actor checkpoints. Rank 3 changes only the inference
  actor formula; rank 5 removes inference Q-search. Ranks 2 and 4 are separately
  trained matched ablations.
- All five pure-RL rows use **the same five P-A trained checkpoints**. Every
  difference is an inference-time actor/Q-search formula. Thus the pure-RL
  TOP5 is strong evidence about deployment operators, but contains only one
  training recipe.

This distinction matters for mix-and-match work: counting inference variants
as independent recipes would create false evidence for their shared training
components.

## Programmatic TOP5, with checkpoint identity exposed

### RL + supervised

| Rank | Near reference | Task | Strict | Mean return | Trained family | Deployment difference |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | **12,496/12,505** | 11,737 | 1,570 | **-138.647924** | H-A | ordinary actor + local 5-action FastSACN Q-search |
| 2 | **12,496/12,505** | 11,737 | 1,570 | -138.649753 | H-B | same deployment; RL target shifts disabled in training |
| 3 | 12,492/12,505 | **11,747** | 1,617 | -139.271479 | **H-A again** | reflection-averaged actor + same local search |
| 4 | 12,486/12,505 | 11,735 | 1,459 | -138.738541 | H-C | uniform-reset DAgger replaces automatic priority |
| 5 | 12,470/12,505 | 11,705 | **1,809** | -138.791249 | **H-A again** | actor only; no inference critic/Q-search |

H-A seed-0 checkpoint SHA-256 is
`F04B034AB459E4AE6EE3A85A5E381CAD835B50F93A20FF5888A7F7EB41BBBA82`.
Ranks 1, 3, and 5 name the same H-A paths
`runs/systematic_100k_budget_g6c_round1_qsearch_20260722/seed{0..4}/checkpoints/final.pt`;
there was no retraining between those rows.

### Pure RL

| Rank | Near reference | Task | Strict | Mean return | Trained family | Deployment difference |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | **11,832/12,505** | **11,567** | **2,303** | **-140.620617** | P-A | reflection actor; online min-Q proposal; two-critic unanimity |
| 2 | 11,739/12,505 | 11,540 | 1,106 | -140.731425 | **P-A again** | ordinary actor; online min-Q proposal; two-critic unanimity |
| 3 | 11,737/12,505 | 11,539 | 1,111 | -140.709499 | **P-A again** | proposal score `mean(Q)-.25*disagreement` |
| 4 | 11,728/12,505 | 11,534 | 1,104 | -140.733257 | **P-A again** | online mean-Q proposal; two-critic unanimity |
| 5 | 11,715/12,505 | 11,561 | 1,111 | -140.735329 | **P-A again** | online+target four-critic min proposal and unanimity |

The P-A checkpoint set is seeds 0--2 from
`runs/week3_simbav2_scale_100k_20260526/simba_full_official_opt/` and seeds
3--4 from `runs/week3_100k_component_ablation_20260527/simba_full_official_opt/`.
All five rows point to this same set. The seed-0 final checkpoint SHA-256 is
`FD89FDD7A27915447886647421370401F143BC6931EC6A0B1C884055AD15C1FA`.
The later recovered seed-3/4 substitutes are not used.

## What was actually trained

### Hybrid actor lineage

Every hybrid family begins with an independently trained 64x2 SimbaV2 actor:

1. Label 400,000 static states with the finite-horizon best reference. The
   state mixture is 60% reset support (`theta` over the full circle and
   velocity `[-1,1]`) and 40% broad velocity `[-8,8]`.
2. Train for 80 epochs, batch 1,024, learning rate `3e-4`.
3. Run three deterministic learner-only DAgger rounds. Each round has 50
   200-step episodes, hence 10,000 learner-controlled transitions and labels;
   `beta=0`, so the reference labels but never executes the recovery action.
4. Train ten more epochs after each initializer round. The final initializer
   contains 430,000 labels, 30,000 learning transitions, and 43,600 recorded
   actor minibatch updates.

The <=100k follow-up does **one**, not ten, additional DAgger round:

- It creates 240,000 fresh reset-support reference labels, adds 20,000
  learner-visited labels, and trains three epochs at `1e-5` (762 minibatch
  updates). The follow-up dataset is 260,000 examples; it does not simply call
  the 20,000 trajectory steps a "20k supervised update."
- H-A and H-B automatically sample 4,000 candidate episode starts, roll the
  current actor for 200 steps from every candidate, rank starts by
  `reference_return - actor_return`, then execute DAgger from the 90 largest
  regrets plus 10 uniformly chosen remaining starts. This is 800,000
  discovery rollout steps per seed, recorded separately from the 20,000
  selected learning transitions.
- H-C runs 100 standard-reset learner trajectories instead and has no
  discovery rollout ledger.

Checkpoint selection has two different dependencies. The 430k-label
initializer checkpoint is selected by fixed-set `eval_action_mae`, so that
selection does **not** depend on the 50k critic. The H-A/H-B/H-C follow-up
summaries all have `inference_qsearch.enabled=true` and select by
`near_reference_eps > targeted_near_reference_eps > task_success >
targeted_mean_return > mean_return`; those validation rollouts use the fixed
50k critic/Q-search. All five seeds selected epoch 3, but the selection rule
still depends on that critic. Consequently, hybrid rank 5 is an actor-only
deployment ablation of a checkpoint selected for Q-search deployment, not a
separately actor-only-selected checkpoint.

### What FastSACN did in the historical hybrid

The hybrid critic is one fixed, separately trained clean pure-RL critic run:

`runs/simbav2_fastsacn8_lam05_utd2_50k_20260704/seed1`

Its SHA-256 is
`EB60CCBB1E579E7E729CA2178460BEC4DD4DFC6539116228EDD9A53175660742`.
It contains two width-64, two-block, 51-bin distributional SimbaV2 critics.
The critic was trained for 50,000 clean Pendulum transitions with uniform
replay (capacity 100,000, batch 256, learning starts 1,000), FastSACN8
`fast_last`, horizon lambda `0.5`, and critic UTD2. It used no reference replay,
hard reset/replay, model replay, or shaping. Its own 32x1 actor is not the
hybrid action proposer.

Crucially, **DAgger BC and the FastSACN actor objective were not mixed in one
historical loss**. FastSACN affected H-A/H-C in two indirect ways:

- During label construction, 41 global torques are scored. If both critics
  prefer the min-Q maximizer to the reference by more than `0.01`, the
  supervised label moves only `0.005` of the way toward it, clipped to a
  maximum shift of `0.02` torque.
- At inference, the critic searches five actions within +/-0.10 of the actor
  proposal and accepts the min-Q maximizer only when both critics prefer it.

H-B disables the first mechanism (`rl_blend=0`) but retains the second. It ties
H-A on near, task, and strict counts and differs in mean return by only
`+0.001830` in H-A's favor. Therefore the old authority evidence does **not**
show that tiny RL-shifted labels helped. A simultaneous BC+FastSACN loss needs
to be tested directly, which is now represented in the queued joint arms.

### Pure-RL training lineage

P-A is clean one-step SAC from scratch for 100,000 transitions per seed:

- 32x1 SimbaV2 actor;
- two 64x2 categorical SimbaV2 critics with 51 bins on `[-5,5]`;
- observation normalization, input shift, feature normalization, reward
  scaling, and weight projection;
- uniform replay capacity 100,000, batch 256, learning starts 1,000;
- one-step targets, critic UTD1, `gamma=.99`, `tau=.005`;
- actor and critic learning rates `1e-4 -> 5e-5`, alpha initialized at `.01`,
  target-entropy scale `-.5`;
- no FastSACN, reference data, DAgger, explicit uniform exploration, hard
  replay/reset, model replay, or shaping.

This means FastSACN is a real cross-category candidate, not a component already
shared by the pure TOP5.

## Deployment formulas, without hidden routing

All ten rows have `hardcoded_inference_router=false` and
`reference_at_inference=false`.

The hybrid local search evaluates
`a_actor + {-0.10,-0.05,0,+0.05,+0.10}` and uses margin zero. The pure search
evaluates 41 fixed global torques over `[-2,2]` and normally requires both
online critics to prefer the replacement by more than `.005`. Both are uniform
learned-Q operators at every state, which is within the user's allowed Q-search
rule. The hybrid does deploy an actor plus an independently trained critic, but
not a router, reference query, or policy mixture.

Reflection is also not a checkpoint mixture. It evaluates the same actor at a
state and its Pendulum mirror:

```text
a_sym(s) = 0.5 * (pi(s) - pi(mirror(s)))
```

Pure rank 5 is the only TOP5 row whose candidate and veto use target critics as
well as online critics. Pure rank 3 changes only proposal ranking to
`mean(Q1,Q2) - .25*abs(Q1-Q2)`; pure rank 4 uses mean Q. None retrains a model.

## Matched evidence from the TOP5

- **Hybrid local Q-search on exactly the same H-A actors:** rank 1 versus rank
  5 gives `+26` near-reference and `+32` task successes, but `-239` strict
  wins and `+0.143325` mean return. It helps the primary and task objectives
  while trading away literal strict wins.
- **Hybrid reflection on exactly the same H-A actors/search:** rank 3 versus
  rank 1 gives `-4` near, `+10` task, `+47` strict, and `-0.623556` mean
  return. Reflection is not automatically beneficial for the supervised actor.
- **Tiny critic target shift:** H-A versus H-B gives exactly zero difference
  in all three counts. The mean-return difference is too small to justify this
  component on current evidence.
- **Automatic priority versus uniform DAgger:** H-A gives `+10` near, `+2`
  task, `+111` strict, and `+0.090617` mean return relative to H-C. However,
  the near-count gain is `(seed0 +10, seeds1--4 +0)`. This is a hypothesis for
  replication, not robust five-seed proof, and it costs 800,000 discovery
  rollout steps per seed.
- **Pure reflection on exactly the same P-A checkpoints/search:** rank 1 versus
  rank 2 gives `+93` near, `+27` task, `+1,197` strict, and `+0.110808` mean
  return. This is the clearest cross-category structural signal.
- **Pure proposal variants on the same checkpoints:** interpolated scoring is
  `-2` near/`-1` task/`+5` strict versus online min-Q; mean proposal is
  `-11`/`-6`/`-2`; four-critic joint search is `-24` near but `+21` task and
  `+5` strict. These are deployment tradeoffs, not evidence about training.

## Cross-category transfers and current queue coverage

| Transfer | Why it is scientifically matched | Queue coverage |
| --- | --- | --- |
| Same-update reference BC + FastSACN actor loss | Directly tests the user's question; contrasts with the null tiny-label-shift ablation | **Present:** joint `h*` arms include BC-only, critic-only, SAC-only, one-step/FastSACN, target gates, gradient balancing, PCGrad, deterministic-mean objectives, lambda 1, and `all` targets; `l0`--`l5` extend selected controls to 30k |
| Hybrid 64x2 actor capacity -> pure RL | Separates architecture from supervised initialization | **Present:** `p7_large_actor64x2_100k` |
| FastSACN -> pure RL | The useful hybrid critic used it; P-A did not | **Present with matched confound controls:** `p6`, `p20`--`p25`, `p34`, `p35` vary one-step/FastSACN, critic UTD, actor UTD, lambda, target mode, and density weighting |
| Pure reflection -> learned symmetry | Pure authority gain is large; hybrid reflection tradeoff shows category dependence | **Present:** hybrid `h10`; pure `p14`, `p15`, `p28`, `p29`, `p33` isolate actor, critic, augmentation, and Q-distill interactions |
| Q-search -> actor distillation | Converts inference improvement into a retrained single actor; log-probability targets address saturated tanh actors | **Partial:** pure `p3`, `p13`, `p33` exist. An exact hybrid 64x2 BC + 41-action log-prob Q-distill matched arm is still missing |
| Automatic priority curriculum -> pure RL | Legal training-time discovery; may cover rare failures without an inference range router | **Not exact:** PER `p2/p4/p5` and uniform model-rollout `h8/h9` are related, but neither reproduces candidate-start scoring plus selected on-policy trajectories |
| Min/mean and online/target critic logic inside training | Pure ranks 2--5 expose proposal/gate tradeoffs | **Partial:** pure `p18/p19/p30/p31` and hybrid target gates exist; a matched hybrid min-Q versus mean-Q actor-loss arm is still missing |
| Exploration and alpha floor | P-A has no explicit uniform exploration; joint hybrid does | **Present:** hybrid `h6/h7`; pure `p16/p26/p27` separate the two factors |
| Best alternative hybrid actor initializer | H-B ties H-A without target shift; H-C avoids discovery cost | **Present:** `priority_no_shift_actor_plus_bc5k` and `uniform_dagger_actor_plus_bc5k` |

The queue mapping is based on
`scripts/run_joint_gradient_balance_screen_20260722.ps1`,
`scripts/run_joint_long_horizon_screen_20260722.ps1`,
`scripts/run_joint_top_actor_transfer_screen_20260722.ps1`, and
`scripts/run_pure_rl_improvement_screen_20260722.ps1`. "Present" means the arm
is defined in the current research queue; it is not a claim that authority
evaluation has completed.

### Recommended nonredundant core

The queue is intentionally broad, but the clean causal core should be read as
these matched chains rather than as dozens of unrelated recipes:

1. **Hybrid loss chain:** actor/BC-only control -> FastSACN critic-only ->
   unfiltered joint loss -> add target gate -> add PCGrad -> add `.005` gate
   margin -> change stochastic SAC actor loss to deterministic-mean -> change
   final BC weight -> change lambda `.5` to `1` -> change `fast_last` to `all`.
   The defined `h_critic_only`, `h2d`, `h3d`, `h3f`, `h3i`, `h3j`, `h3k`,
   `h12`, and `h14` arms cover this nearly one-factor-at-a-time sequence.
   Promote only the short-screen Pareto points to `l*` 30k runs.
2. **Pure replay/Q-distill 2x2:** clean control, PER only (`p2`), Q-distill
   only (`p3` or saturated-action-safe `p13`), and PER+Q-distill (`p4`).
3. **Pure symmetry decomposition:** actor only (`p14`), critic only (`p15`),
   both (`p28`), augmentation only (`p29`), then Q-distill+actor symmetry
   (`p33`). This is the minimum set that explains whether the pure reflection
   gain belongs in the actor, critic, data, or interaction.
4. **FastSACN factorial:** one-step UTD2/actor-UTD1 (`p21`) versus FastSACN
   with the same update rates (`p22`); lambda `.5` versus `1` (`p22/p23`);
   then UTD2 versus UTD1 at each lambda (`p22/p24`, `p23/p25`). `p20` is the
   explicit legacy actor-UTD2 confound control, not another proposed winner.
5. **Exploration 2x2:** alpha floor only (`p26`), uniform exploration only
   (`p27`), and both (`p16`) against clean control.
6. **Capacity transfer:** `p7` versus the exact clean pure control. Supervised
   weights must not initialize this arm or it ceases to be pure RL.

Two queue adjustments remain scientifically important. The hybrid initializer
transfer script currently gives H-B/H-C only another BC stage; after choosing a
joint-loss winner, that exact joint recipe should be rerun from H-A, H-B, and
H-C. Also add the missing hybrid BC+41-action log-prob Q-distillation arm; the
current joint SAC objective is related but is not that matched transfer.

## Budget ledger interpretation

The leaderboard cap is the recovered **per-seed deployed learning lineage**:

| Family | Initializer DAgger transitions | Follow-up DAgger | RL transitions | Learning total | Discovery | Labels | Recorded optimization iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H-A / H-B | 30,000 | 20,000 | 50,000 | **100,000** | 800,000 | 690,000 | 142,362 |
| H-C | 30,000 | 20,000 | 50,000 | **100,000** | 0 | 690,000 | 142,362 |
| P-A | 0 | 0 | 100,000 | **100,000** | 0 | 0 | 99,000 |

Each authority row additionally contains 12,505 evaluation rollouts of length
200, or 2,501,000 evaluation steps total. Static labels, automatic discovery,
oracle calls, optimizer iterations, and evaluation calls are intentionally not
hidden inside the transition count. The shared hybrid critic was physically
trained once and reused, but its 50k lineage is charged to every deployed
hybrid seed pipeline.

The TOP5 dependency ledgers themselves are recoverable: 30k initializer
DAgger + 20k follow-up DAgger + 50k FastSACN for each hybrid pipeline, and one
100k run for P-A. Historical continuations outside this TOP5 remain ambiguous
when their metadata does not say whether an inherited checkpoint contributed
only actor state or both actor and critic state; a nominal extra 5k cannot be
called <=100k until that dependency is resolved. Likewise, the 800k discovery
field counts actor candidate-rollout environment steps, while reference-label
calls and DP/controller scoring work are separate ledgers. Finally, the
142,362/99,000 optimization numbers are recorded iteration ledgers, not the
sum of every low-level actor, critic, and temperature `optimizer.step()` call.

## Evidence paths and limits

The ranking inputs are
[`hybrid_top5_programmatic.csv`](hybrid_top5_programmatic.csv) and
[`pure_rl_top5_programmatic.csv`](pure_rl_top5_programmatic.csv). Hybrid
deployment protocols and exact actor/critic paths are stored in the four
`authoritative_summary.json` files under `../hybrid_*` and
`../ablation_*_qsearch`. The actor-only source is
`../ablation_final_actor_only_relative/relative_summary.json`. Pure deployment
formulas and preregistered selection histories are documented in
`reports/pure_rl_plus1pp_20260719/experiment_ledger.md`, while the authoritative
row-level evidence is in each source rollout path listed in the CSV/JSON.

This matrix is exhaustive for the standardized programmatic inventory, not for
legacy aggregate-only reports that lack recoverable row-level evidence. It
also does not repair repeated use of the 61x41 grid as a development surface;
fresh continuous and untouched-seed gates remain necessary before final
authority claims.
