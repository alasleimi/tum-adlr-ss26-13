# Speaker B Sample Speech

This is a simpler spoken version for slides 6-11. It is written to be read aloud in about 4 minutes 10 seconds.

## Slide 6: First Result, 3:50-4:35

Now I will show the main result.

Each dot here is one training seed, and the diamond is the mean over seeds. The error bars are seed-level confidence intervals, so we are not pretending that every grid cell is an independent experiment.

The headline result is that full SimbaV2 is more reliable at 100k steps. On the return-reference metric, SAC reaches 79.2 percent, while full SimbaV2 reaches 92.5 percent. On the stricter task-stability metric, SAC is 75.8 percent and SimbaV2 is 91.4 percent.

The large SAC error bar is important. SAC has one very bad seed. So the story is not just that SimbaV2 has a slightly higher mean. The story is that SimbaV2 is much more consistent across seeds.

Short version if time is tight:

SimbaV2 improves both reference success and task-stability, and it does so with much smaller seed-to-seed variation.

## Slide 7: Exploration vs Optimization, 4:35-5:25

The next question is why SAC fails.

One possible explanation is exploration. Maybe SAC simply does not visit useful states often enough. But the replay data does not support that as the whole explanation. SAC and SimbaV2 both have about 82.6 to 82.7 percent replay coverage near upright states.

The stronger difference is in the critic representation. Dormant units are units in the critic network that almost never activate. Lower dormancy is better. Effective rank is a rough measure of how many independent directions the critic features use. Higher rank is better.

SAC has many dormant critic units and low effective rank. SimbaV2 has zero measured Q1 dormancy here and much higher rank. So our diagnosis is that this is not only an exploration failure. It is also an optimization and plasticity problem in the value function.

Short version if time is tight:

Replay coverage is almost tied, but critic health is very different. That points toward optimization, not pure exploration.

## Slide 8: SAC Norm Diagnostics, 5:25-5:55

This slide gives one more mechanism check for that diagnosis.

In a SAC diagnostic run, the critic parameter norm grows from about 38 to about 348. The actor also grows, but much less. The critic hidden feature norm grows even more dramatically: one Q-layer feature norm goes from about 606 to almost 4856, with a peak above 5400.

This matters because two of SimbaV2's main changes directly target scale drift. Feature normalization keeps hidden activations under control, and weight normalization keeps parameter scale under control.

So I would not treat this slide as a standalone result. I would treat it as evidence that the SimbaV2 mechanisms are addressing a real pathology we see in SAC.

Short version if time is tight:

SAC shows critic scale drift, and SimbaV2 has mechanisms that directly target feature and weight scale.

## Slide 9: What Did Not Solve It?, 5:55-6:30

We also tested whether the answer is simply more SAC compute.

The careful answer is mixed. More compute does help SAC on the return-reference metric. SAC at 500k reaches 96.3 percent reference success, which is even higher than full SimbaV2 at 100k.

But it still does not solve the behavioral reliability problem. On task-stability, SAC 500k is 88.6 percent, while full SimbaV2 100k is 91.4 percent. The gap is larger on near-down starts: 58.4 percent for SAC 500k versus 70.6 percent for SimbaV2 100k.

The critic-health signal also gets worse. SAC 500k has 77.3 percent dormant Q1 units. So more compute improves return matching, but it does not fix the underlying reliability issue cleanly.

Short version if time is tight:

More SAC compute improves return matching, but task reliability and critic health still lag behind SimbaV2.

## Slide 10: What Else Did Not Solve It?, 6:30-7:10

We also tested a more direct data-distribution idea.

Hard reset means that 20 percent of training episodes start from a difficult large-angle region. Hard replay means that 20 percent of replay batches are sampled from transitions in that same hard-start region.

The result is not a clean win. Hard reset slightly improves task-stability at 50k, from 89.8 to 90.3 percent, but reference success drops from 81.6 to 73.6 percent. Hard replay is worse on both main metrics: 89.0 percent task-stability and 72.0 percent reference success.

So the missing ingredient is not just "show the agent hard states more often." That is too blunt. It can help some hard starts while hurting value accuracy elsewhere.

Short version if time is tight:

Hard-state training pressure alone is too blunt. It does not improve the full reliability frontier.

## Slide 11: Next Experiments, 7:10-8:00

The next steps are about separating mechanisms instead of treating SimbaV2 as one black-box change.

First, we will ablate the SimbaV2 components at 100k. That means removing one thing at a time: feature normalization, weight projection, the distributional critic, reward scaling, and the official SAC recipe. This should tell us which parts are actually driving the reliability gain.

Second, we will try other reliability ideas on Pendulum, especially ReDo, Sample Weight Decay, Fisher-guided selective forgetting, and regret-weighted auxiliary losses. The goal is to push the success rate closer to 0.99, not just improve the mean.

Third, after Pendulum is stable, we move the same protocol to CartPole swing-up. That will test whether this diagnosis transfers to a task where exploration is harder.

Short version if time is tight:

Next we separate the SimbaV2 components, try other plasticity methods, and then move the protocol to CartPole swing-up.

## One-Sentence Ending

The main takeaway is that SimbaV2 improves reliability on Pendulum not because replay coverage is obviously better, but because the critic representation appears healthier and more stable.

## If Asked About The Two Extra Seeds

Two more SAC and two more SimbaV2 seeds have finished training and look sane on fixed evaluation, but they are not yet in the headline slide because they still need the exact-grid DP/controller-relative evaluation.
