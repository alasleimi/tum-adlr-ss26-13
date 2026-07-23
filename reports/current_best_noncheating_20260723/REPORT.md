# Best non-cheating methods at 100k steps or less

Date: 2026-07-23

This report uses completed checkpoints and stored five-seed rollout tables. It does not create a policy mixture, retrain a full model, or select a method on a newly inspected heat map. The only new model evaluation was the actor-only deployment of the already frozen no-RL-shift supervised checkpoints. All other analyses are read-only re-tabulations, paired comparisons, or stored diagnostics.

## Executive decision

### A. Best RL + supervised method

The scientifically preferred winner is:

**Automatic-priority no-RL-shift DAgger, followed by local FastSACN Q-search**

It achieves:

| Metric | Count | Rate |
| --- | ---: | ---: |
| Near reference | 12,496 / 12,505 | **99.928029%** |
| Task success | 11,737 / 12,505 | **93.858457%** |
| Strictly beats reference | 1,570 / 12,505 | **12.554978%** |
| Mean return |  | **-138.649753** |

This method ties the RL-shifted version on all three requested classifications. The shifted version improves mean return by only 0.001830. Removing the shift is therefore the cleaner scientific choice.

Under the conventional learning-transition ledger, the method has a 100k deployed learning lineage:

- 30k learner-controlled DAgger transitions in the supervised initializer
- 20k learner-controlled DAgger transitions in the automatic-priority follow-up
- 50k reward-only transitions for the separately trained FastSACN critic

Automatic priority also evaluates 4,000 candidate starts for 200 steps each, which is 800k discovery rollout steps per seed. These rollouts select training starts but are not inserted into a replay buffer or DAgger dataset. If the 100k limit is intended to include every simulator step that influences data acquisition, this method is a 900k procedure and is not eligible. Under that stricter ledger, the winner is:

**Uniform-start no-shift DAgger plus local FastSACN Q-search**

It uses 100k total learning and discovery transitions and obtains 12,486 near-reference successes, 11,735 task successes, 1,459 strict wins, and mean return -138.738541.

### B. Best pure-RL method by the stated primary metric

The winner is:

**SimbaV2 SAC at 100k reward-only steps, with a reflection-projected actor and unanimous 41-action Q-search**

It achieves:

| Metric | Count | Rate |
| --- | ---: | ---: |
| Near reference | 11,832 / 12,505 | **94.618153%** |
| Task success | 11,567 / 12,505 | **92.499000%** |
| Strictly beats reference | 2,303 / 12,505 | **18.416633%** |
| Mean return |  | **-140.620617** |

This is genuinely pure RL. Its five actors and ten critics are trained from reward only. It uses no reference labels, DAgger data, reference-derived priority, reward shaping, hard angle curriculum, or reference query at inference.

### Important FastSACN clarification

FastSACN is the **pure-RL task-success leader**, but it is not the near-reference leader:

| Pure-RL deployment | Near reference | Task success | Strict wins | Mean return |
| --- | ---: | ---: | ---: | ---: |
| Selected SimbaV2 one-step SAC + reflection + Q-search | **11,832, 94.6182%** | 11,567, 92.4990% | **2,303, 18.4166%** | **-140.6206** |
| FastSACN8 UTD2 50k + unanimous Q-search | 11,657, 93.2187% | **11,619, 92.9148%** | 905, 7.2371% | -140.8856 |

Relative to FastSACN, the selected method has:

- 175 more near-reference successes
- 52 fewer task successes
- 1,398 more strict wins
- 0.264949 better mean return

Because the requested ordering was reference success mainly, then task success and strict wins, the selected winner is still the one-step SimbaV2 checkpoint family. If task success alone were primary, FastSACN would be the correct winner.

The report initially left this FastSACN Pareto point outside the displayed main scorecard because it ranked below the near-reference top five. That omission made the decision look less careful than it was and has now been corrected.

A reflected FastSACN deployment was also screened on reference-free continuous dev and holdout states. It was not promoted to the authoritative grid because its holdout evidence was mixed. On holdout seed 3, reflection changed task success from 0.943115 to 0.941895 and slightly worsened bottom-decile conditional mean return, although it improved mean return and helped seed 4. Therefore there is no completed standardized five-seed authority result establishing reflected FastSACN as the overall winner.

The exact matched transfer is still untested. We have not trained the same 100k-step, UTD1 SimbaV2 recipe while changing only its one-step target to a properly weighted FastSACN target, then applied the same reflection and unanimous global Q-search. The completed FastSACN comparisons change training length, update ratio, target weighting, or deployment at the same time. Consequently, the current result identifies the best evaluated pure-RL pipeline. It does not establish that one-step SAC is intrinsically better than FastSACN under a controlled comparison.

### Legality verdict

| Requirement | Mixed winner | Pure-RL winner |
| --- | --- | --- |
| Hardcoded angle range at inference | No | No |
| Hardcoded angle range in training | No | No |
| Reference available at inference | No | No |
| Router over angle bands | No | No |
| Mixture of trained policy checkpoints | No | No |
| Q-search | Yes, local 5-action search | Yes, global 41-action search |
| Automatic performance-based data discovery | Yes | No |
| Uniform rule applied at every state | Yes | Yes |

The hybrid uses a policy actor and a value critic, but that is not a mixture of policy models. The pure reflection rule calls the same actor at a state and its mathematical mirror. It is also not a model mixture.

## What was searched

The repository audit found 67 matching result summaries and 59 unique standardized five-seed evaluations after row-table deduplication. Forty-two had a recoverable learning lineage at or below 100k under the conventional ledger and passed the deployment rules.

The current user definition is stricter than an earlier audit rule: manually selected angle bands during training are now disallowed. The two selected winners still pass because neither uses them. Full-circle uniform support is not a selected angle band. Automatic regret-based start selection also passes because it is based on measured policy performance, not coordinates.

Two historical 99.96% results scored 12,500 / 12,505, but neither is the current answer:

- The gain-calibrated targeted DAgger pipeline has a 300k selected actor-plus-critic lineage and 380k actually executed component steps. It also used hand-selected coordinate neighborhoods during training, which now violates the stated rule.
- G6C used automatic performance priority and had legal inference, but its training lineage was 280k before adding 8,000,000 discovery rollout steps per actor seed.

The issue with 99.96% was therefore not an inference-time reference leak. It was budget, and for the gain-calibrated recipe also the new prohibition on hand-selected training regions.

## Evaluation protocol and exact metric definitions

Each trained seed is evaluated deterministically on the same 61 angle by 41 angular-velocity grid:

\[
\theta \in [-\pi,\pi], \qquad \dot\theta \in [-1,1].
\]

That is 2,501 initial states per seed and 12,505 seed-state trials across five independently trained actor seeds. Every rollout lasts 200 steps.

Let the two stored reference returns be \(R_{\mathrm{DP}}(s)\) and \(R_{\mathrm{ctrl}}(s)\). The scoring reference is:

\[
R^\star(s)=\max\{R_{\mathrm{DP}}(s),R_{\mathrm{ctrl}}(s)\}.
\]

The requested metrics are:

\[
\text{near-reference}(s)=
\mathbb{1}\left[R_\pi(s)\ge R^\star(s)-5\right],
\]

\[
\text{strict-win}(s)=
\mathbb{1}\left[R_\pi(s)>R^\star(s)\right].
\]

A rollout step is near upright when:

\[
\cos\theta_t\ge 0.95
\quad\text{and}\quad
|\dot\theta_t|\le 1.
\]

Task success requires both:

\[
\frac{1}{200}\sum_{t=1}^{200}\mathbb{1}[\text{near upright at }t]\ge 0.8,
\]

and the longest consecutive not-near-upright streak is at most 50 steps.

The reference is loaded only after the policy and inference rule are frozen, and only to score the stored rollouts. Neither winning deployment calls it.

The 12,505 trials are correlated because neighboring grid cells are similar. The counts are exact descriptions of this grid, not 12,505 independent samples from a population. The most defensible comparisons are paired seed-state changes and consistency over five independently trained actor seeds.

## Main scorecard

| Method | Near reference | Task success | Strict wins | Mean return |
| --- | ---: | ---: | ---: | ---: |
| **Mixed winner, no shift + local Q-search** | **12,496, 99.9280%** | **11,737, 93.8585%** | 1,570, 12.5550% | **-138.6498** |
| Uniform-start mixed control | 12,486, 99.8481% | 11,735, 93.8425% | 1,459, 11.6673% | -138.7385 |
| Supervised actor deployment | 12,470, 99.7201% | 11,705, 93.6026% | 1,813, 14.4982% | -138.7912 |
| **Pure-RL winner** | **11,832, 94.6182%** | **11,567, 92.4990%** | **2,303, 18.4166%** | **-140.6206** |
| FastSACN8 UTD2 50k + unanimous Q-search | 11,657, 93.2187% | **11,619, 92.9148%** | 905, 7.2371% | -140.8856 |
| Plain SimbaV2 SAC at 100k | 11,484, 91.8353% | 11,437, 91.4594% | 1,066, 8.5246% | -141.6386 |
| Stronger clean DAgger 100k replication | 10,571, 84.5342% | 11,007, 88.0208% | 799, 6.3894% | -148.4159 |

![Main five-seed scorecard](figures/01_main_scorecard.png)

The supervised actor row is the newly evaluated no-RL-shift actor with the critic removed at inference. Its actor training contains only reference labels, with no RL target shift. It is best interpreted as a pure-supervised deployment ablation, not a completely independent pure-supervised selection experiment, because the final epoch was selected by a validation protocol that included the fixed Q-search critic. The clean DAgger 100k row is the independent comparator without that selection dependency.

### Comparison to DAgger backbone ablations

| Separate 100k DAgger experiment | Near | Task | Strict | Mean return |
| --- | ---: | ---: | ---: | ---: |
| Paper DAgger, plain MLP | 10,463 | 10,759 | 1,253 | -153.4905 |
| Paper DAgger, SimbaV2 backbone | 10,422 | 10,915 | 786 | -150.9715 |

The winning clean DAgger lineage already uses a SimbaV2 actor, specifically a width-64 actor with two residual blocks. The matched paper DAgger experiment shows that merely changing the backbone from a plain MLP to SimbaV2 did not improve near-reference count. The data mixture, iterative learner-state collection, optimization schedule, automatic priority, and inference critic matter more than the name of the backbone alone.

### Seedwise robustness

![Seedwise success](figures/02_seedwise_success.png)

The mixed winner is not being carried by one unusually good pooled seed. Its near-reference failures are extremely sparse. Pure RL varies much more over training seeds, which becomes clearer when success is aggregated by grid cell:

![Cross-seed cell consistency](figures/08_cross_seed_cell_consistency.png)

For near-reference success:

- Mixed winner: 99.680% of grid cells succeed for all five seeds; 100% succeed for at least one seed.
- Pure-RL winner: 90.164% succeed for all five seeds; 98.441% succeed for at least one seed.
- Pure RL has 246 cells that fail for at least one seed, but only 39 that fail for all five.

This is important. Most of the pure-RL gap is not an impossible region. It is seed-dependent unreliability on states that another pure-RL seed can solve.

### Near-reference reliability and strict wins are different objectives

![Reliability versus strict wins](figures/07_reliability_vs_strict_tradeoff.png)

The pure-RL winner has 733 more literal strict wins than the mixed winner, but 664 fewer near-reference successes. Direct reference labels make the mixed policy conservative and consistent around \(R^\star\). Reward-only SAC sometimes finds trajectories that beat the stored references, but it also has a substantially heavier failure tail.

![Return gap ECDF](figures/04_return_gap_ecdf.png)

The two thresholds in the distribution plot are:

- \(-5\): near reference
- \(0\): strictly beats reference

The strict metric should not be interpreted as a monotone improvement over near-reference reliability. It is a different operating point.

## Analysis of the top five in each category

### RL + supervised top five

| Rank | Method | Near | Task | Strict | Mean return | Trained family |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | RL-shifted automatic-priority actor + local Q-search | 12,496 | 11,737 | 1,570 | -138.647924 | H-A |
| 2 | **No-shift automatic-priority actor + local Q-search** | **12,496** | **11,737** | **1,570** | -138.649753 | H-B |
| 3 | Reflected H-A actor + the same local Q-search | 12,492 | 11,747 | 1,617 | -139.271479 | H-A again |
| 4 | Uniform-start actor + local Q-search | 12,486 | 11,735 | 1,459 | -138.738541 | H-C |
| 5 | H-A actor only | 12,470 | 11,705 | 1,809 | -138.791249 | H-A again |

These are not five independent training recipes. Ranks 1, 3, and 5 use the same H-A actor checkpoints. Rank 3 changes the actor formula at inference; rank 5 removes Q-search. H-B and H-C are separately trained matched ablations.

The no-shift H-B row is preferred over rank 1 because:

- near count is identical
- task count is identical
- strict count is identical
- the only difference is 0.001830 mean return
- the target-shift component is therefore unsupported

The reflected hybrid at rank 3 has the highest task and strict counts in this top five, but it loses four near-reference classifications and 0.621726 mean return relative to the preferred H-B winner. It is not the correct choice when near reference is primary.

### Pure-RL top five

| Rank | Inference rule on the same P-A checkpoints | Near | Task | Strict | Mean return |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **Reflected actor + online min-Q proposal + unanimous margin 0.005** | **11,832** | **11,567** | **2,303** | **-140.620617** |
| 2 | Ordinary actor + online min-Q proposal + unanimous margin 0.005 | 11,739 | 11,540 | 1,106 | -140.731425 |
| 3 | Ordinary actor + mean-Q minus 0.25 disagreement proposal | 11,737 | 11,539 | 1,111 | -140.709499 |
| 4 | Ordinary actor + mean-Q proposal | 11,728 | 11,534 | 1,104 | -140.733257 |
| 5 | Ordinary actor + online-and-target four-critic proposal | 11,715 | 11,561 | 1,111 | -140.735329 |

All five rows reuse the same five trained pure-RL checkpoints. They are evidence about inference operators, not five training recipes. The main result from the top five is that reflection projection is the decisive deployment improvement. Small changes to critic aggregation do not come close.

The best FastSACN row sits below this near-reference top five at 11,657 near successes, but above every displayed pure-RL row on task success with 11,619. It is a real Pareto alternative, not the primary-metric winner.

## Best mixed method in simple language

The mixed winner has five understandable parts:

1. **Learn the reference broadly.** A SimbaV2 actor learns reference actions on full-circle states, including both the ordinary reset velocity range and a broad velocity range.
2. **Let the learner visit its own mistakes.** DAgger runs the actor, records the states it actually reaches, and asks the reference what action should have been used there.
3. **Find hard starts automatically.** The actor is tested from many uniformly sampled starts. Starts are ranked by how much worse the actor performs than the reference. The method trains on the worst measured starts plus a small uniform fraction. No angle band is written by hand.
4. **Keep the supervised target clean.** The winning actor does not shift labels toward the critic. The existing tiny RL shifts did not change any requested classification.
5. **Make a small critic correction at inference.** The actor proposes an action. A reward-trained critic checks that action and four nearby actions. It replaces the actor action only when both critics agree that the replacement is better.

The actor provides stable global behavior. The critic is used as a cautious local corrector, not as a router and not as a second policy.

## Mixed method, exact training recipe

### Actor architecture

The actor is a width-64, two-residual-block SimbaV2 network. It uses:

- observation normalization
- input shift
- feature normalization
- HyperDense layers
- weight projection

This directly answers the architecture question: the clean DAgger actor used by the winning line is a SimbaV2 actor, not a plain MLP.

### Supervised initializer

The initial data contains 400,000 reference-labeled states:

- 60% from reset support, with full-circle angle and \(\dot\theta\in[-1,1]\)
- 40% from broad support, with full-circle angle and \(\dot\theta\in[-8,8]\)

The actor is trained for 80 epochs, batch size 1,024, at learning rate \(3\times10^{-4}\).

It then performs three learner-only DAgger rounds. Each round has 50 episodes of 200 steps, giving 10,000 learner transitions per round. The mixing coefficient is \(\beta=0\), so the learner always executes and the reference only supplies labels.

The initializer therefore has:

- 430,000 labels
- 30,000 learner-controlled transitions
- 43,600 recorded actor minibatch updates

### Automatic-priority follow-up

The follow-up creates 240,000 fresh reset-support labels. It also:

1. Samples 4,000 candidate starts uniformly from reset support.
2. Rolls the current actor for 200 steps from each start.
3. Computes the regret score

\[
\rho(s_0)=R^\star(s_0)-R_{\pi}(s_0).
\]

4. Selects the 90 largest-regret starts and 10 uniformly chosen remaining starts.
5. Runs one 200-step learner-only DAgger trajectory from each selected start.

This produces 20,000 new learner-state labels. The 240,000 static labels and 20,000 trajectory labels form a 260,000-example follow-up dataset. Training is three epochs at \(10^{-5}\), which is 762 recorded updates.

The selected no-shift condition uses:

\[
\texttt{rl\_blend}=0.
\]

No critic change is applied to the supervised target.

### Supervised objective

Let \(a^\star(s)\) be the reference action and let the action limit be 2. The normalized behavior-cloning objective is:

\[
\mathcal{L}_{\mathrm{BC}}(\theta)
=
\frac{1}{N}\sum_{(s,a^\star)\in\mathcal D}
\left(
\frac{\pi_\theta(s)-a^\star}{2}
\right)^2.
\]

After DAgger round \(k\):

\[
\mathcal D_{k+1}
=
\mathcal D_k
\cup
\{(s_t,\pi^\star(s_t))\}_{t=1}^{T}.
\]

The important point is that the learner controls the trajectory. This targets compounding errors that static behavior cloning never sees.

### FastSACN critic

The inference critic is the fixed checkpoint:

`runs/simbav2_fastsacn8_lam05_utd2_50k_20260704/seed1`

It contains two width-64, two-block, 51-bin categorical SimbaV2 critics trained for 50k clean reward-only transitions with:

- uniform replay, capacity 100k
- batch size 256
- learning starts at 1k
- critic UTD 2
- no reference replay
- no hard replay or hard reset
- no model replay
- no reward shaping

For an \(h\)-step target:

\[
G_h
=
\sum_{j=0}^{h-1}\gamma^j r_{t+j}
+
\gamma^h
\left[
\min_i Q_{\bar\phi_i}(s_{t+h},a')
-\alpha\log\pi(a'|s_{t+h})
\right].
\]

The categorical critic projects this target onto 51 fixed atoms in \([-5,5]\) and minimizes cross-entropy to the projected target distribution.

The stored `fast_last`, \(\lambda=0.5\) configuration activates the 1-step and 8-step endpoints. The nominal 8-step weight is:

\[
\lambda^7=0.5^7=0.0078125.
\]

After normalization, only 0.7752% of the nominal loss belongs to the long endpoint. The historical critic should therefore be viewed as mostly a one-step critic with a very small long-horizon auxiliary, not as strong evidence that eight-step credit assignment solved the problem.

### Local inference Q-search

The actor proposes:

\[
a_0=\pi_\theta(s).
\]

The five candidate actions are:

\[
\mathcal A_{\mathrm{local}}(s)
=
\operatorname{clip}
\left(
a_0+\{-0.10,-0.05,0,0.05,0.10\},
-2,2
\right).
\]

The proposal is:

\[
\hat a
=
\arg\max_{a\in\mathcal A_{\mathrm{local}}(s)}
\min\{Q_1(s,a),Q_2(s,a)\}.
\]

It is accepted only if both critics prefer it:

\[
Q_1(s,\hat a)>Q_1(s,a_0)
\quad\text{and}\quad
Q_2(s,\hat a)>Q_2(s,a_0).
\]

Otherwise the actor action is executed. The same rule is used at every state. There is no coordinate test and no reference query.

## Best pure-RL method in simple language

The pure method has three parts:

1. **Train one SAC actor and two SAC critics for 100k reward-only steps.**
2. **Force the actor to respect the environment's left-right symmetry.** The same actor is queried at the real state and the mirrored state. Their antisymmetric combination removes part of the learned left-right error.
3. **Search the full action range with the learned critics.** Forty-one torques from \(-2\) to \(2\) are checked. A new action is used only when both critics say it beats the reflected actor by more than 0.005.

There are no labels, no reference, no hand-selected hard states, and no second policy model.

## Pure-RL method, exact math

### Training configuration

Each seed is clean one-step SAC from scratch for 100,000 reward-only transitions:

- actor: width-32, one-block SimbaV2
- critics: two width-64, two-block categorical SimbaV2 networks
- 51 critic bins on \([-5,5]\)
- uniform replay capacity 100k
- batch size 256
- learning starts at 1k
- critic UTD 1
- \(\gamma=0.99\)
- target update \(\tau=0.005\)
- actor and critic learning rates \(10^{-4}\) to \(5\times10^{-5}\)
- initial temperature \(\alpha=0.01\)
- target-entropy scale \(-0.5\)

There is no FastSACN in this training recipe.

With \(a'\sim\pi_\theta(\cdot|s')\), the one-step target is:

\[
y
=
r
+
\gamma(1-d)
\left[
\min_i Q_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'|s')
\right].
\]

The actor minimizes:

\[
\mathcal L_{\mathrm{SAC}}(\theta)
=
\mathbb E_{s\sim\mathcal B,\;a\sim\pi_\theta}
\left[
\alpha\log\pi_\theta(a|s)
-\min_i Q_{\phi_i}(s,a)
\right].
\]

The bounded action is:

\[
a=2\tanh u_\theta(s,\epsilon).
\]

### Reflection projection

For observation:

\[
s=(\cos\theta,\sin\theta,\dot\theta),
\]

the mirrored state is:

\[
M(s)=(\cos\theta,-\sin\theta,-\dot\theta).
\]

The deterministic reflected actor is:

\[
a_{\mathrm{sym}}(s)
=
\frac{1}{2}
\left[
\pi_\theta(s)-\pi_\theta(M(s))
\right].
\]

This is a projection toward the physical equivariance:

\[
\pi(M(s))=-\pi(s).
\]

### Global unanimous Q-search

The candidates are:

\[
a_j=-2+\frac{4j}{40},
\qquad j=0,\ldots,40.
\]

The critic proposal is:

\[
\hat a
=
\arg\max_{a_j}
\min\{Q_1(s,a_j),Q_2(s,a_j)\}.
\]

The method executes \(\hat a\) only if:

\[
Q_i(s,\hat a)-Q_i(s,a_{\mathrm{sym}})>0.005
\quad\text{for both }i\in\{1,2\}.
\]

Otherwise it executes \(a_{\mathrm{sym}}\). There is no trust-region cap and no angle-conditioned rule.

## Most important completed ablations

The following comparisons are paired on the same five seeds and the same 12,505 grid trials:

| Component added or changed | Near fixed | Near broken | Near net | Task net | Strict net | Mean return change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k automatic-priority DAgger training stage | 30 | 1 | **+29** | +7 | +442 | +0.133650 |
| Local FastSACN Q-search | 26 | 0 | **+26** | +32 | -239 | +0.143325 |
| Automatic priority instead of uniform starts | 10 | 0 | **+10** | +2 | +111 | +0.090617 |
| Tiny RL label shifts instead of no shift | 0 | 0 | **0** | 0 | 0 | +0.001830 |
| Pure reflection fallback added to global Q-search | 126 | 33 | **+93** | +27 | +1,197 | +0.110808 |
| Pure global unanimous Q-search added to ordinary actor | 374 | 119 | **+255** | +103 | +40 | +0.907132 |

![Paired component ablations](figures/05_paired_component_ablation.png)

### What each ablation establishes

**The extra 20k supervised stage helps.** With the same local Q-search, it fixes 30 near-reference trials and breaks one. It also adds 442 strict wins. This is direct evidence that learner-state supervision is doing real work beyond the initializer.

**Local Q-search is a reliability correction, not a strict-win optimizer.** It fixes 26 near-reference classifications and breaks none, while adding 32 task successes. It loses 239 strict wins. A policy can move slightly closer to the reference and cross from positive to small negative signed gap while still improving reliability.

**Automatic priority has positive but weak across-seed evidence.** It adds 10 near successes relative to uniform starts, but all 10 occur in seed 0. Seeds 1 through 4 have zero near-count change. It costs 800k discovery rollout steps. It is retained under the conventional ledger because the pooled result is best, but the uniform-start control is the more conservative result under a strict simulator-call budget.

**Tiny critic-shifted labels do not help.** Zero near, task, or strict classifications change. This component should be removed.

**Reflection is category dependent.** It is very strong for pure RL, with +93 near and +1,197 strict successes, but applying reflection to the mixed actor loses four near successes and 0.623556 mean return. A useful pure-RL idea cannot be assumed to transfer unchanged to a supervised actor that already learned reference symmetry.

**Global Q-search is essential for the pure method.** It nets +255 near and +103 task successes over the plain actor, but it also breaks 119 near successes. The unanimity and margin gate are essential. Unconditional global argmax controls perform much worse.

### Paired return changes reveal heavy tails

![Paired return delta distributions](figures/06_paired_return_delta_distributions.png)

For the preferred no-shift actor, local Q-search has:

- mean paired return change \(+0.1414\)
- median change \(-0.0126\)
- positive change on only 29.4% of trials
- net +26 near and +32 task classifications

Its positive mean is carried by rare, important recovery fixes. Small return losses are common but usually do not change the classifications.

The pure reflection-plus-Q-search deployment versus the plain Simba actor has:

- mean paired return change \(+1.0179\)
- median change \(+0.0834\)
- positive change on 71.9% of trials

Matched decomposition shows that global Q-search and reflection both contribute, with reflection supplying most of the strict-win increase.

## Why DAgger and FastSACN were historically separate

The short answer is: they do not have to be separate, but the completed evidence says a naive sum of the losses is unsafe.

A simultaneous objective would be:

\[
\mathcal L_{\mathrm{joint}}(\theta)
=
w_{\mathrm{BC}}\mathcal L_{\mathrm{BC}}(\theta)
+
w_{\mathrm{RL}}\mathcal L_{\mathrm{SAC}}(\theta).
\]

The two gradients are:

\[
g_{\mathrm{BC}}=\nabla_\theta\mathcal L_{\mathrm{BC}},
\qquad
g_{\mathrm{RL}}=\nabla_\theta\mathcal L_{\mathrm{SAC}}.
\]

Their cosine is:

\[
c
=
\frac{g_{\mathrm{BC}}^\top g_{\mathrm{RL}}}
{\|g_{\mathrm{BC}}\|\,\|g_{\mathrm{RL}}\|}.
\]

When \(c<0\), the update directions conflict. The short joint screens repeatedly measured negative average cosine, for example \(-0.1106\) for BC plus one-step SAC and \(-0.1517\) for the unfiltered gradient-balanced FastSACN arm. The critic objective can move shared actor parameters in a direction that reduces frozen critic loss while harming real rollout return.

The historical winner avoids this conflict by dividing responsibilities:

- supervised DAgger builds a stable actor
- reward-trained FastSACN builds a critic
- the actor does not receive a critic gradient in the preferred H-B recipe
- the critic makes only a small, unanimity-gated action correction at inference

This separation is not ideology. It is what the current ablations support.

### Direct short-screen evidence for mixing the loss

The completed 5k seed-0 screen uses ten fixed evaluation episodes. It is useful mechanism evidence, but it is not a five-seed authority ranking.

| Arm | Final mean return | Worst return | Mean near-upright fraction | Task successes |
| --- | ---: | ---: | ---: | ---: |
| BC only | -170.435 | -270.871 | 0.8470 | 6 / 10 |
| Gated deterministic joint loss + PCGrad | -170.545 | -270.880 | 0.8470 | 6 / 10 |
| **All-horizon lambda-1 joint loss** | **-168.889** | -271.131 | **0.8555** | **7 / 10** |
| Unbounded global-Q log-prob distillation | -205.166 | -409.158 | 0.8330 | 7 / 10 |
| The same distillation at lower actor LR | -252.857 | -405.915 | 0.7495 | 5 / 10 |
| Q-filtered replay BC | -170.471 | -270.895 | 0.8470 | 6 / 10 |

![Joint loss short screen](figures/11_joint_loss_short_screen.png)

The all-horizon lambda-1 joint arm is the only promising direct mix in this completed short screen. Relative to the matched gated arm, it adds one task success, 0.0085 mean near-upright fraction, and 1.656 mean return. The task difference is one threshold crossing on ten episodes, so it is not enough to replace the five-seed authority winner.

The unbounded log-probability Q-distillation arms explain why simply adding a stronger Q term can fail. The initial raw Q-distillation loss was about 12,255. With a 0.05 multiplier its contribution was about 612, while the BC loss was about 0.0146. The update was dominated by the critic target. Lowering the policy learning rate preserved a pathological narrow policy distribution for longer and made the collapse worse.

The Q-filtered replay-BC arm was nearly a null intervention. Its active mask averaged only 1.302% because the 0.005 gate margin was large relative to the critic's local signal. It therefore behaved like anchor-heavy BC and produced the same classifications as the BC control.

The correct conclusion is:

1. Mixing BC and RL inside one loss is scientifically reasonable.
2. The scales, gate, critic maturity, and gradient conflict must be controlled.
3. Existing small screens do not beat the frozen mixed winner.
4. The all-horizon lambda-1 joint arm deserves multi-seed confirmation before promotion.

Stored training diagnostics:

![Joint loss curves](../systematic_joint_followup_20260722/joint_loss_training_diagnostics_seed0/loss_curves.png)

![Joint gradient and gate curves](../systematic_joint_followup_20260722/joint_loss_training_diagnostics_seed0/gradient_gate_curves.png)

## Specialized hybrid ablations

These completed experiments explain mechanisms but do not have authority-grid evidence.

### Saturation-aware supervised loss

A signed-logit hinge for saturated reference targets:

- reduces saturated replay MAE from 0.117667 to 0.093240, a 20.8% improvement
- reduces saturated anchor MAE from 0.033055 to 0.023666, a 28.4% improvement
- worsens unsaturated replay MAE by 42.3%
- worsens unsaturated anchor MAE by 28.7%
- increases prediction saturation by 21.85 percentage points
- reduces mean tanh derivative by 19.4%

It improves two of three fixed hard probes, but all three remain task failures. It reallocates accuracy toward saturated targets rather than uniformly improving the actor.

![Saturation-aware training curves](../systematic_joint_followup_20260722/hybrid_specialized_training_diagnostics/saturation_training_curves.png)

### Failure mining

Failure mining alone:

- improves actual-failure MAE by 7.1%
- worsens all-trajectory MAE by 2.1%
- worsens nonpriority MAE by 3.2%
- worsens anchor MAE by 1.4%

Combining mining with saturation-aware loss:

- improves actual-failure MAE by 14.2%
- improves saturated actual-failure MAE by 54.9%
- worsens broad trajectory MAE by 3.9%
- worsens anchor MAE by 2.7%

This is useful local synergy, but without rollout evaluation it cannot be called a better policy.

![Failure-mining factorial curves](../systematic_joint_followup_20260722/hybrid_specialized_training_diagnostics/failure_factorial_training_curves.png)

### Reference anchor ratio

A 10% reference-anchor component reduces fixed-anchor MAE by 56.6% and raises the ten-episode mean return by 5.963, but task success falls from 7 / 10 to 6 / 10. Seven paired episodes change by less than 0.34 return, while one failure improves by 87.683. The mean is tail dominated and is not robust evidence for promotion.

![Anchor-ratio curves](../systematic_joint_followup_20260722/hybrid_specialized_training_diagnostics/anchor_ratio_training_curves.png)

## Performance diagnostics

### Heat maps

Mixed winner near-reference map:

![Mixed near-reference map](../systematic_100k_budget_best_20260722/ablation_no_rl_shift_qsearch/relative/near_best_known_return_eps_map.png)

Mixed winner task-success map:

![Mixed task-success map](../systematic_100k_budget_best_20260722/ablation_no_rl_shift_qsearch/grid/task_success_rate_map.png)

Pure-RL winner near-reference map:

![Pure RL near-reference map](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/near_best_known_return_eps_map.png)

Pure-RL winner task-success map:

![Pure RL task-success map](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/task_success_map.png)

The mixed near-reference errors are nine isolated boundary failures. Pure RL shows a broad recovery deficit near the downward and wrap boundary.

### Failure rates by angle region

![Failures by angle region](figures/03_failure_by_angle_region.png)

For the mixed winner:

- all 9 near-reference failures occur at \(|\theta|\ge 120^\circ\)
- 5 occur at \(|\theta|\ge 150^\circ\)
- 666 of 768 task failures occur at \(|\theta|\ge 150^\circ\)

For pure RL:

- 673 near-reference failures total
- 629 / 673, or 93.46%, occur at \(|\theta|\ge 120^\circ\)
- 489 / 673, or 72.66%, occur at \(|\theta|\ge 150^\circ\)
- 716 / 938 task failures occur at \(|\theta|\ge 150^\circ\)
- the complete \(60^\circ\) to \(120^\circ\) band is perfect on near and task metrics

The pure near-down low-speed region, \(|\theta|\ge150^\circ\) and \(|\dot\theta|\le0.5\), has only 75.325% near-reference success and 67.186% task success.

## Why pure RL does not reach the mixed method

The gap is:

| Outcome | Mixed minus pure RL |
| --- | ---: |
| Near reference | **+664 trials, +5.3099 percentage points** |
| Task success | **+170 trials, +1.3595 percentage points** |
| Mean return | **+1.970864** |
| Strict wins | **-733 trials, -5.8617 percentage points** |

![Mixed versus pure gap](figures/09_mixed_vs_pure_gap.png)

The evidence supports a coupled explanation, not one single defect.

### 1. Training signal and state coverage

DAgger receives the desired action at the states its actor actually reaches. Automatic priority further concentrates labels on measured high-regret trajectories. Pure SAC receives only scalar reward and uniform replay. Rare early swing-up decisions that determine the entire trajectory occupy a small fraction of replay.

This directly explains why the reference metric gap is much larger than the task gap. Pure RL usually completes the task, but it is less consistent at matching the stored reference return.

### 2. The critic is useful on average but unreliable in the tail

On five seeds, 99 hard states, and 21 local actions:

| Diagnostic | Mean |
| --- | ---: |
| Per-state critic Spearman rank correlation | 0.6931 |
| Pairwise action-order accuracy | 0.8437 |
| Centered within-state Pearson correlation | 0.3901 |
| Q-selected raw gain over actor | +0.0102 |
| Q-selection harms actor | 18.99% |
| Twin disagreement divided by local Q range | 2.4264 |

The critic contains real local action information. The problem is that twin disagreement is 2.43 times the median local action signal. Typical ranks look reasonable, but a small set of severe rank reversals controls the recovery tail.

![Pure RL diagnostic flags](figures/10_pure_rl_diagnostic_flags.png)

![Five-seed hard-state diagnostic summary](../systematic_joint_followup_20260722/pure_rl_gap_diagnostics_training_aligned_v2/plots/diagnostic_summary.png)

![Critic counterfactual failures](../systematic_joint_followup_20260722/pure_rl_gap_diagnostics_training_aligned_v2/plots/critic_action_counterfactual_failures.png)

### 3. The actor saturates before the critic fully matures

On hard states:

- 93.13% of deterministic actor actions are at or beyond 99.5% of an action bound
- median absolute pre-tanh logit is 4.581
- median normalized tanh derivative is \(4.339\times10^{-4}\)
- final entropy temperature averages \(1.96\times10^{-5}\)

For action \(a=2\tanh u\):

\[
\frac{\partial a}{\partial u}
=
2(1-\tanh^2 u).
\]

At a large \(|u|\), this derivative is nearly zero. Bang-bang torque can be appropriate for swing-up, but a useful late critic gradient cannot easily make a small timing correction through the saturated mean.

Actor loss is close to its plateau by roughly 50k to 70k, while Q loss continues improving through 100k. This supports a feedback loop in which entropy disappears, the actor saturates, and later critic maturation cannot repair the deterministic action.

![Pure-RL training curves](../systematic_joint_followup_20260722/pure_rl_gap_diagnostics_training_aligned_v2/plots/training_curves.png)

### 4. Action-space ranking does not imply a safe parameter-space update

The seed-0 actor was perturbed on three independent 5 by 5 parameter planes with radius 1% of the actor parameter norm. Frozen-critic loss and paired real rollout return were evaluated at every plane cell.

| Direction seed | Q versus return Spearman | Critic-loss versus real-loss gradient cosine | Return regret of Q-selected cell |
| ---: | ---: | ---: | ---: |
| 22072611 | -0.4408 | -0.9365 | 1.2169 |
| 22072612 | -0.8477 | -0.7831 | 1.1874 |
| 22072613 | +0.6154 | -0.7274 | 0.0156 |
| Mean | **-0.2244** | **-0.8157** | **0.8066** |

The critic-loss gradient is anti-aligned with real return loss on all three measured slices. This is the strongest direct evidence for why adding more SAC actor loss can make the supervised actor worse.

An action counterfactual changes one action at one state and then follows the frozen policy. A parameter update changes actions at many states and every future time step. Shared features allow small systematic critic errors to add coherently.

Example parameter-plane diagnostics:

![Parameter landscape Q versus rollout, direction 22072611](../systematic_joint_followup_20260722/pure_seed0_actor_parameter_landscape/simba_full_official_opt__seed0__15c8edcb/direction_seed_22072611/q_vs_rollout_delta.png)

![Parameter landscape mean return, direction 22072611](../systematic_joint_followup_20260722/pure_seed0_actor_parameter_landscape/simba_full_official_opt__seed0__15c8edcb/direction_seed_22072611/mean_return_landscape.png)

![Parameter landscape frozen critic loss, direction 22072611](../systematic_joint_followup_20260722/pure_seed0_actor_parameter_landscape/simba_full_official_opt__seed0__15c8edcb/direction_seed_22072611/frozen_critic_loss_landscape.png)

### 5. Reflection helps, but critic asymmetry remains

The raw actor reflection error averages 0.2432 action units. The critic reflection error averages only 0.0342 Q units, but that is 6.91 times the median local Q range. It is large enough to reverse a search gate.

Adding the reflected fallback to the same pure Q-search:

- fixes 126 near-reference trials
- breaks 33
- nets +93 near
- adds +27 task
- adds +1,197 strict wins

Despite actor reflection projection, 425 of the remaining 673 near failures occur at negative angles and 248 at positive angles. The online critics that accept or reject the Q proposal are not exactly symmetric.

### 6. Dormancy is not the explanation

No actor or critic layer is dormant at the registered relative-activation threshold on:

- hard actor-manifold states
- replay samples
- broad uniform state-action support

Critic hidden-layer minimum effective-rank fractions are:

- 0.153 on hard actor states
- 0.265 on replay samples
- 0.296 on broad support

The rank increases on broader probes. Much of the low hard-state rank is therefore manifold geometry and nearly constant saturated actions, not dead neurons. Dormancy is ruled down as a primary cause.

### 7. Bellman horizon mismatch is not the main cause

A long-horizon probe extends to 917 steps, where:

\[
\gamma^{917}\approx9.94\times10^{-5}.
\]

The deterministic critic Spearman is 0.7217, the stochastic-soft Spearman is 0.7112, and the long-versus-200-step absolute tail difference is only \(4.836\times10^{-5}\) in training units. The 200-step versus continuing-soft objective mismatch does not explain most of the gap.

The historical FastSACN lambda-0.5 target also places less than 1% nominal weight on its 8-step endpoint, so it does not test whether a properly weighted long target can help.

## Best-supported causal picture

The mixed method wins near-reference reliability because:

1. Reference labels provide a low-variance action target.
2. Learner-only DAgger covers states induced by the current policy.
3. Automatic priority adds labels at measured high-regret starts.
4. The width-64 supervised actor remains stable over the broad state distribution.
5. Local Q-search makes small, conservative recovery corrections without exposing the actor parameters to a misaligned critic gradient.

Pure RL remains behind because:

1. Uniform reward-only replay underrepresents rare trajectory-defining recovery states.
2. The critic has good typical ordering but poor signal-to-uncertainty on the hard tail.
3. Entropy temperature collapses and the actor saturates early.
4. The actor's parameter-space critic objective can be anti-aligned with real rollout return.
5. Reflection and Q-search repair many errors, but critic asymmetry and tail rank reversals remain.

This picture explains all four headline facts at once:

- pure RL nearly matches task success
- pure RL has more strict wins
- pure RL has many more near-reference failures
- the mixed method has much better cross-seed uniformity

## Scientifically necessary next confirmations

The completed evidence is enough to identify the current winners. The following are the smallest nonredundant confirmations needed to improve them without angle cheating:

1. **Confirm the all-horizon lambda-1 joint loss over multiple actor and critic seeds.** Keep the no-shift H-B actor, critic UTD, data, and 100k ledger fixed. Compare BC-only, gated joint, and all-horizon lambda-1 joint one factor at a time.
2. **Pure-RL automatic failure curriculum.** Sample candidate starts uniformly, rank only by the policy's own reward or critic-calibrated performance, and train from selected starts while preserving exactly 100k reward-bearing learning steps. Do not use reference regret or angle bands.
3. **Pure symmetry factorial.** Compare actor-equivariance regularization, critic-equivariance regularization, both, and data augmentation against the same plain Simba checkpoint recipe.
4. **Saturation and exploration factorial.** Test alpha floor only, uniform exploration only, both, and neither. Record replay occupancy, tanh derivative, hard-region return, and parameter-landscape alignment.
5. **True FastSACN test.** Compare one-step against lambda 1 with identical critic UTD and actor UTD. Compare `fast_last` against all-horizon targets. The existing lambda-0.5 result is not a strong long-horizon ablation.
6. **Pure actor capacity.** Train width-32 by one block versus width-64 by two blocks from scratch with reward only. Do not initialize from supervised weights, because that would no longer be pure RL.
7. **Repeat the actor-parameter landscape.** Any proposed pure-RL winner should improve both hard-region performance and critic-versus-return alignment on several seeds. Better mean return without better alignment is likely another tail accident.

For every promoted method, report:

- near reference, task success, and strict wins
- all-five-seed and any-seed grid-cell success
- angle and velocity failure regions
- paired fixed and broken cells versus its parent
- critic rank, disagreement-to-signal, Q harm rate
- actor saturation, entropy temperature, and tanh derivative
- dormancy and effective rank
- parameter-space critic-versus-return alignment

## Scientific limitations

- The heat-map grid has been used repeatedly during development. It is now a descriptive authority surface, not a pristine test set.
- Neighboring cells are correlated, so trial-level confidence intervals exaggerate independent evidence.
- The hybrid uses one globally selected fixed critic for all five actor seeds. The five rows are independent actor replications, not five independent full-pipeline replications.
- The actor-only supervised row depends on Q-search-aware checkpoint selection even though its training targets and deployed inference are supervised-only.
- The direct joint-loss screens use one training seed and ten fixed evaluation episodes. They cannot overrule a completed five-seed grid result.
- The parameter landscape uses one checkpoint and three random planes. It is strong mechanistic evidence, but still needs repetition across seeds and interventions.

## Reproducible artifacts

Main generated tables:

- [Main scorecard](main_scorecard.csv)
- [Seedwise rates](seedwise_rates.csv)
- [Regional failure rates](regional_failure_rates.csv)
- [Cross-seed cell consistency](cross_seed_cell_consistency.csv)
- [Paired ablation diagnostics](paired_ablation_diagnostics.csv)
- [Paired return-delta summary](paired_return_delta_summary.csv)
- [Mixed-versus-pure gap](mixed_vs_pure_gap.csv)
- [Pure-RL diagnostic key metrics](pure_rl_diagnostic_key_metrics.csv)
- [Joint-loss short-screen curves](joint_loss_short_screen_curves.csv)
- [Build manifest](build_manifest.json)

Primary source evaluations:

- [Mixed winner rollout rows](../systematic_100k_budget_best_20260722/ablation_no_rl_shift_qsearch/relative/relative_rollouts.csv)
- [Pure supervised actor rollout rows](pure_supervised_no_rl_shift_actor_relative/relative_rollouts.csv)
- [Pure-RL winner rollout rows](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/relative_rollouts.csv)
- [FastSACN task-leader rollout rows](../pure_rl_plus1pp_20260719/authority_clean_fastsacn8_utd2_q41m005_unanimous_relative/relative_rollouts.csv)
- [Plain SimbaV2 rollout rows](../week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_rollouts.csv)
- [Clean DAgger rollout rows](../canonical_reference_dagger_100k_5seed_20260716/relative/relative_rollouts.csv)

Supporting audits:

- [Systematic 100k inventory report](../systematic_100k_budget_best_20260722/REPORT.md)
- [Top-five component matrix](../systematic_100k_budget_best_20260722/programmatic_inventory/TOP5_COMPONENT_MATRIX.md)
- [Pure-RL causal gap memo](../systematic_joint_followup_20260722/PURE_RL_GAP_CAUSAL_MEMO.md)
- [Training-aligned pure-RL diagnostics](../systematic_joint_followup_20260722/pure_rl_gap_diagnostics_training_aligned_v2/pure_rl_gap_diagnostics.md)
- [Hybrid specialized-training diagnostics](../systematic_joint_followup_20260722/hybrid_specialized_training_diagnostics/REPORT.md)
- [Joint-loss diagnostics](../systematic_joint_followup_20260722/joint_loss_training_diagnostics_seed0/REPORT.md)
- [Parameter-landscape index](../systematic_joint_followup_20260722/pure_seed0_actor_parameter_landscape/parameter_landscape_index.md)

Plot and table builder:

- [build_current_best_noncheating_report_20260723.py](../../scripts/build_current_best_noncheating_report_20260723.py)
