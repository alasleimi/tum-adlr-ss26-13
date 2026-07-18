# General supervised + RL 99.9% experiment ledger

Date: 2026-07-18

## Qualification rules fixed before pilots

- No hand-written angle/velocity failure bands in training, validation, or inference.
- No post-training action gain or other scalar calibration.
- No per-seed hyperparameters or final-grid checkpoint selection.
- No reference, DP, or controller query at inference.
- One identical recipe for independently trained seeds 0-4.
- A fixed global critic action search is allowed; a state-dependent hand-written router is not.
- Primary success is at least 12,493/12,505 returns within 5 of max(DP, controller).
- Strict wins use literal `policy_return > max(DP_return, controller_return)`.

## Eligible ingredients

- Static supervised states: mixtures of the full observation domain and the complete ordinary reset-support distribution only.
- DAgger initial states: ordinary Pendulum reset distribution only.
- DAgger transitions: learner action is always passed to `env.step()`; the best reference labels the saved learner-visited state at the correct remaining horizon.
- RL source: standard-reset FastSACN8 UTD-2 runs with zero hard resets and zero hard replay.
- RL influence: a fixed critic-advantage rule may weight or conservatively shift reference targets.
- Development selection: midpoint grids disjoint from the authoritative grid plus a fixed continuous uniform-reset holdout.

## Ineligible historical results

- `large_simba_gain1005_*`: post-training gain calibration and manually targeted data; diagnostic upper bound only.
- `large_simba_targeted_failuremix_qw_*`: manually targeted failure neighborhoods.
- `large_simba_corrected_dagger100k_*`: the follow-up support set includes `hard120` and `near_down` coordinate bands.

## Pilot G1: generic critic-regularized standard-reset DAgger

Status: completed; not promoted.

- Supervised initialization for the seed-0 pilot: the pre-targeting 64x2 `reset60` actor.
- Final five-seed recipe, if promoted: recreate that initialization independently for every seed.
- Fixed RL critic source: corresponding standard-reset FastSACN8 UTD-2 seed.
- Static support during the pilot: uniform over the complete Pendulum reset support.
- DAgger collection: standard reset, learner-only.
- Validation: disjoint midpoint grid plus continuous uniform-reset holdout.
- Promotion depends only on those validation sets, never individual authoritative-grid cells.

Command:

```powershell
python scripts/train_pendulum_qregularized_dagger.py `
  --dagger-run runs/reference_assisted_large_simba_reset60_dagger3_seed0_20260717/seed0 `
  --rl-run runs/simbav2_fastsacn8_lam05_utd2_50k_20260703/seed0 `
  --run-dir runs/general_hybrid_priority_g1_seed0_20260718/seed0 `
  --device cuda --seed 0 `
  --static-size 240000 `
  --broad-fraction 0 --reset-support-fraction 1 `
  --hard120-fraction 0 --near-down-fraction 0 `
  --broad-velocity-limit 1 --reset-velocity-limit 1 `
  --dagger-rounds 5 --dagger-episodes 100 `
  --dagger-initial-mode priority_uniform `
  --priority-candidate-multiplier 20 --priority-fraction 0.75 `
  --targeted-validation-size 5001 --targeted-validation-mode reset_uniform `
  --validation-theta-bins 47 --validation-velocity-bins 31 `
  --validation-every-epochs 3 --epochs-per-round 3 `
  --batch-size 1024 --lr 1e-5 `
  --num-actions 41 --rl-margin 0.01 `
  --rl-blend 0.005 --max-target-shift 0.02 `
  --selected-weight 1 --trainable-actor all
```

Declared pilot budget: 240,000 uniform static labels, 100,000 learner DAgger
transitions, 10,000 uniformly proposed initial-state candidates, and 15 actor
epochs. Candidate scoring uses 20,000 DP/controller rollout evaluations.

Observed result (selected checkpoint: epoch 15, after round 5):

- 47x31 midpoint grid: near-reference `0.9972546328`, task success
  `0.9437199725`, literal strict wins `0.1365820178`, mean return
  `-137.9684510`.
- 5,001-point continuous uniform-reset holdout: near-reference `0.9970005999`,
  task success `0.9438112378`, literal strict wins `0.1443711258`, mean return
  `-137.1987287`.
- Initial source on those sets: `0.9931365820` and `0.9922015597`
  near-reference, respectively.

Conclusion: the coordinate-free rollout-regret sampler materially improved
generalization, but both held-out near-reference rates remain below the frozen
`0.999` promotion threshold. The authoritative 61x41 grid was not queried.

## G1 fixed critic Q-search screen

The only inference-time RL component screened was a single global rule: search
five evenly spaced actions centered on the supervised actor action and replace
the actor action only when every clean pure-RL critic assigns positive advantage.
No reference is queried, and no angle/velocity condition is used at inference.

Held-out near-reference results by fixed search radius:

| Radius | Midpoint 47x31 | Continuous 5,001 | Decision switch rate |
|---:|---:|---:|---:|
| 0.02 | 0.9972546328 | 0.9976004799 | 0.1410970889 |
| 0.05 | 0.9979409746 | 0.9982003599 | 0.1475232270 |
| 0.10 | 0.9986273164 | 0.9986002799 | 0.5184066274 |
| 0.15 | 0.9986273164 | 0.9986002799 | 0.5171639827 |

Radius `0.10` was the best validation point, but it still missed the `0.999`
gate on both sets. Full-range Q-search was rejected after a disjoint small smoke
set showed that its rare switches changed actions by about 3.5-3.7 and reduced
near-reference success. The authoritative grid was not queried.

## Pilot G2: higher-coverage automatic-regret DAgger

Status: running.

G2 keeps every G1 learning hyperparameter fixed and changes only the declared
data budget: ten rounds instead of five, 4,000 uniformly proposed starts per
round instead of 2,000, and 90 priority-ranked starts plus 10 uniformly retained
starts per round. Selection remains entirely coordinate-free.

```powershell
python scripts/train_pendulum_qregularized_dagger.py `
  --dagger-run runs/reference_assisted_large_simba_reset60_dagger3_seed0_20260717/seed0 `
  --rl-run runs/simbav2_fastsacn8_lam05_utd2_50k_20260703/seed0 `
  --run-dir runs/general_hybrid_priority_g2_seed0_20260718/seed0 `
  --device cuda --seed 0 `
  --static-size 240000 `
  --broad-fraction 0 --reset-support-fraction 1 `
  --hard120-fraction 0 --near-down-fraction 0 `
  --broad-velocity-limit 1 --reset-velocity-limit 1 `
  --dagger-rounds 10 --dagger-episodes 100 `
  --dagger-initial-mode priority_uniform `
  --priority-candidate-multiplier 40 --priority-fraction 0.90 `
  --targeted-validation-size 5001 --targeted-validation-mode reset_uniform `
  --validation-theta-bins 47 --validation-velocity-bins 31 `
  --validation-every-epochs 3 --epochs-per-round 3 `
  --batch-size 1024 --lr 1e-5 `
  --num-actions 41 --rl-margin 0.01 `
  --rl-blend 0.005 --max-target-shift 0.02 `
  --selected-weight 1 --trainable-actor all
```

Declared pilot budget: 240,000 generic static labels, 200,000 learner DAgger
transitions, 40,000 uniformly proposed initial-state candidates, 80,000
DP/controller candidate-scoring rollouts, and 30 actor epochs.

Observed G2 raw-actor result (selected epoch 27): midpoint near-reference
`0.9979409746`; continuous near-reference `0.9982003599`. With the already
selected radius-0.10 fixed local Q-search, the same checkpoint produced
`0.9986273164` midpoint (2 misses) and `0.9992001600` continuous (4 misses).
Only the continuous holdout passed the `0.999` gate.

## Pilot G2Q: select the checkpoint using the deployed Q-search policy

Status: running.

G2Q repeats G2's deterministic training recipe and budget. The only change is
that each held-out checkpoint evaluation uses the exact inference policy:
five actions at actor action plus `{-0.10, -0.05, 0, 0.05, 0.10}`, clipped to
the action bounds, with the replacement chosen only for positive advantage under
every critic. This corrects G2's mismatch between raw-actor checkpoint selection
and Q-search deployment. The authoritative grid remains untouched.

Observed result: G2Q selected epoch 9 (round 3) and passed both promotion
holdouts: midpoint near-reference `0.9993136582` (1 miss of 1,457) and
continuous near-reference `0.9992001600` (4 misses of 5,001). Task success was
`0.9437199725` and `0.9442111578`; literal strict wins were `0.1338366507` and
`0.1397720456`, respectively.

## Frozen five-seed recipe

Status: training seeds 0-4. No further hyperparameter changes are allowed.

Each supervised initializer is trained independently with its corresponding seed:

```powershell
python -m last_nine_rl.distill_reference `
  --run-dir runs/general_hybrid_frozen_initializer_5seed_20260718/seedN `
  --policy best --seed N --device cuda `
  --actor-backbone simba_v2 --actor-hidden-dim 64 --actor-blocks 2 `
  --dataset-size 400000 --eval-dataset-size 50000 `
  --batch-size 1024 --epochs 80 --lr 3e-4 `
  --velocity-limit 8 `
  --reset-support-fraction 0.6 --reset-support-velocity-limit 1 `
  --near-down-fraction 0 --near-upright-fraction 0 `
  --selection-metric eval_action_mae `
  --dagger-iterations 3 --dagger-episodes-per-iteration 50 `
  --dagger-train-epochs-per-iteration 10 `
  --dagger-rollout-mode deterministic `
  --rollout-backend vectorized_pendulum `
  --dagger-expert-beta-start 0 --dagger-expert-beta-final 0
```

The existing seed-0 initializer is the independently trained seed-0 realization
of this exact recipe; seeds 1-4 are recreated under the frozen command. Each is
then paired with its same-seed clean FastSACN8 UTD-2 critic and run through the
exact G2Q command above. Training completes all ten rounds; checkpoint selection
uses only the two held-out validation sets and the fixed radius-0.10 Q-search.

Observed multi-seed validation:

| Seed | Selected round | Midpoint near-reference | Continuous near-reference | Gate |
|---:|---:|---:|---:|:---:|
| 0 | 3 | 0.9993136582 | 0.9992001600 | pass |
| 1 | 1 | 1.0000000000 | 1.0000000000 | pass |
| 2 | 10 | 0.9972546328 | 0.9970005999 | fail |

The recipe is therefore rejected before seeds 3-4 or the authoritative grid.
Seed-2 diagnostics showed the raw selected actor itself was limited to
`0.9979409746` midpoint and `0.9972005599` continuous. Wider Q-search radii
reduced success, so the failure is assigned to insufficient training correction,
not inference search.

## Pilot G3D: automatic disagreement-weighted DAgger

Status: running on development seed 2.

G3D changes only how the already eligible aggregate is sampled by the supervised
loss. Every DAgger state whose current actor action differs from its reference
target by more than `0.05` receives weight `4`; other DAgger states receive
weight `1`. Static states selected by the fixed pure-RL critic rule also receive
weight `4`. State collection, budgets, Q-search, and all coordinate-free rules
remain identical to G2Q. This is a fixed data-dependent rule, not a hand-written
state region.

Observed G3D result: midpoint `0.9972546328`, continuous `0.9964007199`.
The disagreement weighting is rejected.

## Pilot G4E: paired automatic recovery demonstrations

Status: running on development seed 2.

G4E restores unit sample weights. In every round it keeps all 100 learner-only
DAgger trajectories and additionally executes 100 best-reference trajectories
from those exact same automatically regret-selected starts. Thus each round adds
20,000 learner-visited labels plus 20,000 expert recovery labels. No start is
selected by coordinates. The purpose is to cover the recovery branch that the
learner may never visit after an early basin-selection error.

Declared G4E budget: 240,000 generic static labels, 200,000 learner DAgger
transitions, 200,000 paired expert demonstration transitions, 40,000 uniform
candidate starts, 80,000 candidate reference rollouts, and 30 actor epochs.

Observed G4E result: midpoint `0.9972546328`, continuous `0.9966006799`.
Paired demonstrations are rejected.

## Diagnostic G5X: separate initializer variance from critic/data variance

Status: running; explicitly ineligible as a final seed-2 model.

G5X runs the original G2Q recipe with the successful seed-0 supervised
initializer but seed-2 critic, DAgger RNG, and validation RNG. If it succeeds,
the remaining problem is supervised-initializer variance and an automatic
multi-restart recipe is justified. If it fails, the seed-2 critic or downstream
data path is the more likely bottleneck. This cross-seed actor is diagnostic only
and cannot enter the final five-seed result.

Observed G5X result: midpoint `0.9972546328`, continuous `0.9972005599`.
Changing the supervised initializer did not solve seed 2.

Fixed-actor critic swap diagnosis on the G2Q seed-2 actor:

| Q-search critic | Midpoint | Continuous |
|---|---:|---:|
| pure-RL seed 2 | 0.9972546328 | 0.9970005999 |
| pure-RL seed 0 | 0.9986273164 | 0.9988002400 |
| pure-RL seed 1 | 1.0000000000 | 0.9994001200 |

The seed-1 critic also passes with the seed-0 actor (`0.9993136582` midpoint,
`0.9998000400` continuous) and its own seed-1 actor (`1.0`, `1.0`). A global,
pre-deployed critic is therefore justified. The selection rule is fixed as:
evaluate clean pure-RL critics in ascending seed order and choose the first that
passes both holdouts for every development actor (0, 1, 2). Critic 0 fails this
rule; critic 1 is the first pass. Critics 3-4 are not screened.

## Pilot G6C: one globally selected pure-RL critic

Status: running on development seed 2.

G6C restores the original G2Q data and unit-weight loss. The clean FastSACN8
UTD-2 seed-1 critic is now used for both conservative training target shifts and
the fixed local Q-search for every supervised actor seed. This eliminates the
high variance from seed-matched critic quality while preserving independently
trained supervised actors. All other hyperparameters and budgets are G2Q.

Observed development seed-2 result: selected round 2, midpoint
`1.0000000000`, continuous `0.9990002000` (exactly 5 misses). G6C passes the
seed-2 gate. Seed 0 is now being retrained under the identical common-critic rule;
seed 1 already used critic 1 for both training and inference.

Final development-set G6C validation:

| Actor seed | Selected round | Midpoint | Continuous | Gate |
|---:|---:|---:|---:|:---:|
| 0 | 9 | 1.0000000000 | 0.9996000800 | pass |
| 1 | 1 | 1.0000000000 | 1.0000000000 | pass |
| 2 | 2 | 1.0000000000 | 0.9990002000 | pass |

G6C and the seed-1 global critic are now frozen. Seeds 3-4 are untouched
confirmation actors: their results may accept or reject the recipe but may not
change it. The authoritative 61x41 grid remains unqueried.

Untouched confirmation results:

| Actor seed | Selected round | Midpoint | Continuous | Gate |
|---:|---:|---:|---:|:---:|
| 3 | 7 | 1.0000000000 | 0.9996000800 | pass |
| 4 | 2 | 1.0000000000 | 0.9998000400 | pass |

## Pre-registration: one-shot authoritative audit

Status: frozen before execution on the authoritative grid.

The exact actors are, in seed order:

1. `runs/general_hybrid_g6c_seed0_20260718/seed0/checkpoints/final.pt`
2. `runs/general_hybrid_frozen_g2q_5seed_20260718/seed1/checkpoints/final.pt`
3. `runs/general_hybrid_g6c_seed2_20260718/seed2/checkpoints/final.pt`
4. `runs/general_hybrid_g6c_seed3_20260718/seed3/checkpoints/final.pt`
5. `runs/general_hybrid_g6c_seed4_20260718/seed4/checkpoints/final.pt`

Every actor uses the single preselected critic
`runs/simbav2_fastsacn8_lam05_utd2_50k_20260704/seed1/checkpoints/final.pt`.
The inference rule is fixed to five local actions spanning actor action plus or
minus `0.10`, with replacement only when every critic estimates strictly positive
advantage (`margin=0`). There is no checkpoint, critic, seed, margin, radius, or
action selection on the authoritative data. The result is accepted exactly when
near-reference success is at least 12,493 of 12,505 rollouts. Regardless of the
result, the recipe will not be changed after this audit.

Frozen command:

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

Observed one-shot result: `12,500 / 12,505 = 0.9996001599`
near-reference successes. This exceeds the pre-registered 12,493-success gate.
Task success is `11,742 / 12,505 = 0.9389844062`; literal strict
`return > max(DP, controller)` wins are `1,791 / 12,505 = 0.1432227109`.
No post-audit training, selection, calibration, routing, or hyperparameter change
was performed.
