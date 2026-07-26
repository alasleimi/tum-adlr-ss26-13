# Chasing the Nines: 8-minute presentation path

Target speaking time: 7 minutes 40 seconds to 8 minutes. The remaining margin allows for starting the videos and moving between poster regions.

## 0:00 to 0:45 | Question

Our project asks a reliability question. Can a policy succeed not only on average, but across difficult initial states and independent training seeds?

We call this “chasing the nines.” Moving from 96% to 99.9% success changes the problem. Average return is no longer enough because a small, seed-dependent failure tail dominates the result.

We studied Pendulum because it gives us a controlled setting where every initial angle and angular velocity can be evaluated systematically. The broader question is relevant whenever an RL policy must recover reliably rather than perform well on a typical episode.

## 0:45 to 1:30 | Evaluation

Point to panel 01.

We evaluate five independently trained actors on the same grid of 2,501 starts. That produces 12,505 seed-state trials for each recipe.

Our main metric is near-reference success. A trial succeeds when its return is within five points of a strong reference controller. We also report task success, which asks whether the pendulum stays upright and slow for most of the episode, and strict wins, where the learned policy obtains a higher return than the reference.

The grid matters because it shows whether failures form a stable region or appear only for particular seeds.

## 1:30 to 2:35 | The two selected recipes

Point to the method boxes in panel 01.

The mixed recipe has three stages. First, DAgger labels states visited by the learner, so the actor learns corrections on its own state distribution. Second, an automatic follow-up stage revisits starts where measured performance is weak. It does not use a hand-coded angle range. Third, a reward-trained critic performs conservative local action search.

The selected pure-RL recipe uses 100,000 reward transitions. It combines a SimbaV2 network, FastSACN8 critic training, physical symmetry, and twin-critic action search.

FastSACN8 is our simplified multi-step critic variant. Published SACn averages losses from multiple return horizons and adds importance weighting and a lower-variance entropy estimate. Our variant keeps only one-step and eight-step categorical critic targets. It also performs two critic updates per transition.

The eight-step term has an explicit normalized coefficient of 0.775% in the selected implementation. That number is not a measured gradient share and does not show that the term is negligible. A small coefficient can still have a cumulative effect through repeated nonlinear updates and shared features. Our current experiments also couple the target change with twice as many critic updates, so they do not isolate those two effects.

## 2:35 to 3:25 | Main result

Point to the percentages and then panel 02.

The mixed recipe reaches 99.928% near-reference success, with nine failures out of 12,505 trials. The selected pure-RL recipe reaches 96.026%, with 497 failures.

The heat maps reveal more than the aggregate rates. All five mixed actors succeed on 2,493 of 2,501 starts. For pure RL, all five succeed on 2,293 starts, and another 195 starts depend on which seed was trained.

This is the reliability gap we want to explain. Supervision almost removes both the state-space tail and the seed-to-seed variation.

## 3:25 to 4:25 | Closed-loop recovery

Play `01_same_hard_start_learning_gap.mp4`, then point to panel 03.

This video begins every policy from the same hard state. By step 64, the reference and mixed policy have recovered to the upright region. The raw pure-RL actor is still drifting and finishes 103.7 return points lower.

This example illustrates a closed-loop issue. A corrective action changes the next state. The policy must then choose the right next action from that new state, and continue doing so for several decisions. DAgger directly labels these learner-visited recovery states.

## 4:25 to 5:20 | One action versus a sequence

Point to panel 04.

We tested this idea on 1,267 failures of the selected raw pure-RL actor. We temporarily apply a state-matched reference correction for a fixed number of steps and then return control to the same actor.

One corrected action repairs only 19.1% of failures. Eight steps repair 47.7%, 16 repair 91.3%, and 32 repair 99.0%. If the same actions are shuffled across states, the repair rate collapses.

The result is not simply that the reference uses stronger torque. The ordering and state matching matter. Failed rollouts require a coherent corrective segment.

## 5:20 to 6:35 | What the critic knows and what the actor executes

Point to panel 05.

Several diagnostics narrow the explanation.

First, small critic-guided action changes improve the realized return more often after multi-step critic training. Reflection and twin-Q search together repair 770 seed-state classifications on the authority grid. This means the critics contain useful local action information that the raw actor does not consistently express.

Second, 98.9% of diagnosed failures already have a reference-like action among nearby replay transitions. Broad missing experience therefore does not explain most failures.

Third, the selected raw actor reaches the torque limit on 70.3% of matched diagnostic states, compared with 59.1% for the matched one-step actor. The stronger critic does not automatically produce a better raw actor.

Completed SimbaV2 runs also show no dormant units under our registered threshold. This does not eliminate every plasticity explanation, but it weakens a simple dead-feature account.

## 6:35 to 7:25 | Mechanistic account

Point to the bottom conclusion.

The evidence supports a specific account.

Multi-step targets can improve critic-side ranking because they connect an action with consequences several transitions later. Critic search can exploit that ranking at the current state.

The actor still has to produce a full closed-loop recovery. Learner-state supervision teaches what to do at each state created by the actor’s earlier decisions. That is why supervision reduces the failure tail much more strongly than one-step critic-guided repair.

This account also explains why strict wins and reliability have different orderings. Pure RL can outperform the reference on many easy or favorable trials while still failing badly on a smaller set of difficult recoveries.

## 7:25 to 8:00 | Conclusion and next experiment

The main result is nine failures for RL plus supervision versus 497 for the strongest five-seed pure-RL recipe we found.

The main scientific result is that the pure-RL gap is not well described as a total absence of useful actions. The actions exist in replay, and the critics often rank better immediate actions. The larger problem is turning those local preferences into a stable sequence under closed-loop state changes.

The cleanest next experiment is a factorial comparison of one-step versus eight-step targets and one versus two critic updates, followed by an actor objective that distills critic-selected corrective sequences while controlling torque saturation.

If there is time for a second demonstration, play `02_pure_rl_qsearch_repairs_failure.mp4` to show one failure repaired by reflection and twin-Q search.
