# Chasing the Nines: 8-minute presentation path

Target speaking time: 7 minutes 40 seconds. The remaining 20 seconds allow for starting the video and moving between poster regions.

## 0:00 to 0:45 | Question and result

Point to the dark headline.

Our question is whether one learned policy can succeed across difficult initial states and independent training seeds. We call this “chasing the nines.” Moving from 96% to 99.9% reliability changes the problem because a small failure tail dominates the conclusion.

On the same 12,505 seed-state trials, our best RL plus supervised recipe has 9 near-reference failures. Our best five-seed pure-RL recipe has 497. Pure RL wins strictly against the reference more often, so reliability and return dominance are different objectives.

## 0:45 to 1:35 | Evaluation

Point to panel 01 and its fixed-evaluation box.

We train five independent actors and evaluate every actor on 2,501 starts from a 61 by 41 grid. Each rollout lasts 200 steps.

Near-reference success means that return is within five points of the better reference controller. Task success asks whether the pendulum stays upright and slow for at least 80 percent of the rollout. A strict win means that the learned policy beats both reference controllers.

The fixed grid exposes two sources of unreliability: hard states and disagreement between training seeds.

## 1:35 to 2:45 | The two selected routes

Point to the two colored method boxes in panel 02.

The mixed route has three training stages. DAgger asks the reference controller to label states actually visited by the learner. Priority refit automatically samples the starts with the largest measured return deficits. Local Q-search then scores the actor torque and four nearby torques with both learned critics.

The pure-RL route uses 100,000 reward transitions. Its actor and critics use SimbaV2 networks. FastSACN8 is our reduced SACn-inspired critic objective that combines one-step and eight-step targets. The selected recipe also uses symmetry reflection and a 41-action twin-critic search.

Reference labels and discovery rollouts are used only while training the mixed route. At deployment, each route is one actor/critic pipeline. It receives the current state and does not query the reference.

## 2:45 to 3:35 | What each component contributes

Point to the matched controls and pure-RL path in panel 02.

The top row contains three separate five-seed controls for the mixed route. Learner-state refitting reduces 38 failures to 9. Priority sampling reduces 19 to 9 relative to uniform starts. Local Q-search reduces 35 to 9 relative to the same actor. These comparisons share the final method, but they are separate controls rather than a temporal chain.

The bottom row is a sequential deployment path. The raw FastSACN8 actor has 1,267 failures. Reflection removes 312, then 41-action Q-search removes another 458, leaving 497. The headline reports task success and strict wins as separate objectives, including the reliability versus strict-win tradeoff.

## 3:35 to 4:15 | Failure geography and video

Point to the stacked heatmaps in panel 01, then play `01_same_hard_start_learning_gap.mp4`.

All five mixed actors pass 2,493 of the 2,501 starts. All five pure-RL actors pass 2,293 starts, and another 195 starts depend on which seed was trained.

The video begins both policies from the same difficult state. The mixed policy recovers, while the raw pure-RL actor drifts away. The heatmap and video show the same phenomenon at different scales: the gap is concentrated in a small but reproducible recovery tail.

## 4:15 to 5:10 | Baselines and one difficult recovery

Point to the comparator row in panel 01, then to the three pendulum snapshots in panel 03.

The supervised actor reaches 99.72 percent near-reference success on one seed. Plain five-seed SimbaV2 reaches 91.84 percent, and clean DAgger reaches 84.53 percent. The selected mixed route therefore does more than imitate a fixed dataset: it labels learner states, revisits measured deficits, and then uses learned critics.

The snapshots show one post hoc difficult start at step 81. The diagnostic reference and mixed actor are upright, while the raw pure-RL actor has drifted. Their final returns are −280.9, −280.6, and −384.3. This example is not an aggregate estimate; it makes the failure mode visible before we inspect the networks.

The aggregate heatmap and this trajectory point to the same question: why can the pure-RL critics identify useful actions while the raw actor still fails to execute them?

## 5:10 to 6:35 | White-box network test

Point to the four large white-box probes in panel 03.

This is a matched network intervention. Architecture, five seeds, 100,000 transitions, and update ratio are fixed. Only the critic target changes. SAC uses one-step targets, Fast denotes FastSACN8 and averages one-step and eight-step targets, and N8 denotes SACN8 and uses the eight-step target.

Panel A shows that the multistep variants push actor outputs to the torque bound on about 87 percent of tested states, compared with 59 percent for one-step SAC. Panel B shows that the median derivative through the tanh output falls from 0.120 to 0.039.

Panels C and D test 512 fixed failures per seed. The critics often point toward or rank a higher-return controller action, but the actor is already near its torque bound. After projection and clipping, only about 7 percent of the proposed local critic step survives for Fast and N8.

This is a white-box actor bottleneck. Multi-step targets can improve critic direction while simultaneously making the actor output harder to change. Scoring 41 torques bypasses that local projection and raises the five-seed fixed-grid near-reference rate from 92.363 percent after reflection to 96.026 percent.

## 6:35 to 7:25 | Why the losses were not simply mixed

Point to the joint-loss pilot at the bottom of panel 02.

We also tested simultaneous behavior-cloning and SACN8 actor updates. On one 25,000-step pilot, joint training reaches 2,443 near-reference starts versus 1,757 for reward-only SACN8.

The gradient diagnostic explains why one fixed coefficient is not a complete solution. The weighted cloning gradient is 37.2 times larger than the SAC gradient at the median. Their cosine ranges from strongly opposed to strongly aligned during training. A single scalar coefficient therefore changes both the effective scale and the amount of interference over time.

The next controlled test is normalized or projected gradient mixing, not an assumption that a small nominal coefficient is negligible.

## 7:25 to 8:00 | Conclusion

Point to the footer.

The main empirical result is 9 failures for RL plus supervision versus 497 for the strongest five-seed pure-RL recipe.

The diagnostic result is more specific. Pure-RL critics often contain useful local action information, but multistep training pushes the actor toward saturated outputs where only a small fraction of that local update survives. Supervision nearly removes the state and seed failure tail, while critic search partially recovers pure RL by acting directly in action space.

If time permits after the talk, play `02_pure_rl_qsearch_repairs_failure.mp4` to show a failure repaired by reflection and twin-Q search.
