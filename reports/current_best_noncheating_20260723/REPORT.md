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



The exact matched transfer is still untested. We have not trained the same 100k-step, UTD1 SimbaV2 recipe while changing only its one-step target to a properly weighted FastSACN target, then applied the same reflection and unanimous global Q-search. The completed FastSACN comparisons change training length, update ratio, target weighting, or deployment at the same time. Consequently, the current result identifies the best evaluated pure-RL pipeline. It does not establish that one-step SAC is intrinsically better than FastSACN under a controlled comparison.



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



### The actor saturates before the critic fully matures

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


### Reflection helps, but critic asymmetry remains

The raw actor reflection error averages 0.2432 action units. The critic reflection error averages only 0.0342 Q units, but that is 6.91 times the median local Q range. It is large enough to reverse a search gate.

Adding the reflected fallback to the same pure Q-search:

- fixes 126 near-reference trials
- breaks 33
- nets +93 near
- adds +27 task
- adds +1,197 strict wins

Despite actor reflection projection, 425 of the remaining 673 near failures occur at negative angles and 248 at positive angles. The online critics that accept or reject the Q proposal are not exactly symmetric.






