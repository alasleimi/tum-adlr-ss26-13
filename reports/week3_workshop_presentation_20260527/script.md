# Project 15 Week 3 Workshop Script

This script is written for an 8 minute talk. Speaker A covers slides 1-5. Speaker B covers slides 6-11. Backup slides are for questions.

## Slide 1, Speaker A, 0:00-0:40
Our project is about reliability in deep reinforcement learning. Pendulum is not supposed to be a hard benchmark, so average return is not the interesting part. The interesting part is the last set of initial states where a neural SAC policy still fails to swing up and stabilize.

The GIF is not random. It is the same initial state from the exact evaluation grid: initial angle is minus 174.1 degrees and angular velocity is minus 1.0. SAC seed 0 fails there, while full SimbaV2 seed 0 succeeds. The headline success metric is within 5 return points of max(DP, controller): SAC 100k is at 83.0% with +/- 7.8 pp SE, while full SimbaV2 100k is at 91.8% with +/- 0.8 pp SE. The stricter all-grid task-stability check is 81.2% with +/- 8.9 pp SE versus 91.5% with +/- 0.6 pp SE. The wider 95 percent intervals are on slide 6.

## Slide 2, Speaker A, 0:40-1:35
The most important thing is the success definition. We first tried an intuitive task-stability definition: the policy should swing up, stay near upright for at least 80 percent of the episode, and not lose the upright region for more than 50 consecutive steps.

That is useful, but it is not a fair universal success definition because our references do not satisfy it everywhere. DP or the hand controller satisfies that task-stability predicate on 2336 out of 2501 cells, which is 93.4%. The remaining cells are not certified by either reference under that behavioral predicate.

So the headline success metric is state-conditioned return matching: the policy return must be within 5 of max(DP, controller) from the same start. Most of the time DP is the better reference: 2403 out of 2501 cells. The controller is better on 98 cells, so using the max matters.

## Slide 3, Speaker A, 1:35-2:15
The evaluation grid has 61 angle bins and 41 angular-velocity bins, so 2501 reset-support states. For the 100k comparison we now have 5 training seeds, so 12505 deterministic grid rollouts per policy.

In the maps, each cell is the fraction of seeds that satisfy the plotted criterion from that exact state. With five seeds, that means 0 percent, 20 percent, 40 percent, 60 percent, 80 percent, 100 percent. Seeds 3 and 4 are follow-up runs with diagnostics every 10k steps rather than 100k steps, but the training recipe and final evaluation are otherwise the same. The references are finite-horizon DP and an energy-swing-up plus PD hand controller. The diagnostics track replay coverage and critic representation health.

## Slide 4, Speaker A, 2:15-3:05
For listeners who have not read SimbaV2, the paper is not just "a bigger SAC network." It makes four concrete changes to SAC.

First, hyperspherical feature normalization means L2-normalizing hidden feature vectors so their scale does not drift. Second, hyperspherical weight normalization means removing weight decay and projecting selected weights back to unit norm after each update. Third, the critic becomes distributional: it predicts binned return probabilities, and reward scaling keeps the target variance in a stable range. Fourth, the paper uses a different SAC recipe from CleanRL: smaller learning rates, lower initial entropy weight, different target entropy scale, smaller Simba networks, and the distributional critic.

The table gives the exact side-by-side settings from our 100k runs. The important scientific point is that we should not claim which component drives the improvement yet. The short-budget ablations were useful for debugging, but not representative enough for the talk.

## Slide 5, Speaker A, 3:05-3:50
These are the raw maps. The left column is task-stability success. The middle column is the headline metric: within 5 return points of max(DP, controller). The right column is the continuous version of that same reference comparison: how many return points the policy is below max(DP, controller). The first row is SAC 100k and the second row is full SimbaV2 100k. Unclipped cell-mean shortfall is max 122.4, mean 4.6 plus or minus 8.2 for SAC, and max 113.8, mean 3.2 plus or minus 9.3 for SimbaV2.

The right column uses a color cap at 20 return points. That is not changing the numbers; it only saturates the color scale. We do it because a few cells are more than 100 return points below the reference, and a full-range color scale would make all ordinary near-boundary differences look the same.

## Slide 6, Speaker B, 3:50-4:35
Here is the seed-level main result. The dots are training seeds and the diamonds are means. The intervals are 95 percent t-intervals over seeds, so they are still conservative at five seeds.

Reference success improves from 83.0% +/- 21.6 pp for SAC to 91.8% +/- 2.3 pp for full SimbaV2. The stricter all-grid task-stability check improves from 81.2% +/- 24.7 pp to 91.5% +/- 1.7 pp. On known-feasible cells, task-stability improves from 86.6% +/- 26.2 pp to 97.2% +/- 1.3 pp. SAC has one very bad seed, which is why its interval is huge. We should present that uncertainty explicitly.

## Slide 7, Speaker B, 4:35-5:25
This is the exploration versus optimization diagnosis. If SAC failed only because it never saw useful states, replay near-upright coverage should separate. It does not: SAC is at 82.6% and full SimbaV2 is at 82.4%.

The separation is critic health. Dormant units are critic units that barely activate, so lower is better. Effective rank is a proxy for how diverse the critic features are, so higher is better. SAC has high dormancy and low rank, while SimbaV2 has zero measured Q1 dormancy and much higher rank. That points to optimization, plasticity, and value estimation, not pure exploration.

## Slide 8, Speaker B, 5:25-5:55
This slide adds the norm-diagnostics evidence. It is not a new main result, because it is one SAC diagnostic run, not a seeded comparison. But it explains why the SimbaV2 changes are plausible.

In this run, the critic parameter norm grows from 38.4 to 348.5, while the actor norm only grows from 14.4 to 67.1. The critic hidden feature scale also inflates: Q1 fc2 grows from about 606 to 4856, with a peak around 5416. This is exactly the kind of scale drift that SimbaV2 tries to remove with hyperspherical feature normalization and hyperspherical weight normalization.

The caveat is important: this supports the optimization diagnosis and motivates the 100k component ablations. Component attribution is exactly what those ablations are for.

## Slide 9, Speaker B, 5:55-6:30
More SAC compute is a useful negative result, but we need to state the metric carefully. SAC 500k is not worse on reference success: it reaches 96.3%, compared with 91.8% for full SimbaV2 100k.

It is still worse on the behavioral reliability metrics. Exact-grid task-stability is 88.6% for SAC 500k versus 91.5% for full SimbaV2 100k, so SAC is 2.8 percentage points behind. On near-down starts the gap is 58.4% versus 69.4%, so SAC is 11.1 percentage points behind. The critic-health signal also worsens: SAC 500k has 77.3 percent dormant Q1 units, while full SimbaV2 has 0.0 percent in this diagnostic.

## Slide 10, Speaker B, 6:30-7:10
Hard reset and hard replay test the data-distribution hypothesis more directly. Hard reset p=0.2 means 20 percent of episode resets are forced into a large-angle band: absolute theta between 120 and 135 degrees, with absolute angular velocity at most 1. Hard replay p=0.2 means 20 percent of each replay minibatch is sampled from transitions in that same hard-start band.

The graph is now a grouped bar chart because these are categorical interventions, not a progression. Ordinary full SimbaV2 50k has task-stability 89.8 percent and reference success 81.6 percent. Hard reset moves task-stability slightly to 90.3 percent, but reference success drops to 73.6 percent. Hard replay is worse on both headline metrics, at 89.0 percent task-stability and 72.0 percent reference success. Near-down task success is also worse for hard replay.

So just showing hard states more often is too blunt. The next intervention has to preserve value accuracy on the rest of the state space.

## Slide 11, Speaker B, 7:10-8:00
The next step is to move the component claims to representative budget. We should ablate away SimbaV2 components one at a time at 100k: feature normalization, weight projection, distributional critic, reward scaling, and the official SAC recipe. That tells us what actually drives the reliability improvement.

In parallel, we should push Pendulum toward at least 0.99 reference success using other reliability ideas from the proposal: ReDo, Sample Weight Decay, Fisher-guided selective forgetting, and regret-weighted auxiliary losses. After the Pendulum frontier is stable, we move the same protocol to CartPole-Swingup, where exploration is a more serious part of the problem.

## Backup / Q&A
Use the seed-0-excluded raw maps if asked how much of the map story is driven by the outlier SAC seed. Use the DP slide if asked whether DP is optimal. It is approximate finite-horizon DP, not a proof. Use the controller slide if asked what max(DP, controller) means. Use the bad-seed slide if asked why SAC has a huge confidence interval.

GIF status: Policy GIF: exact-grid contrast where SAC seed0 fails and full SimbaV2 seed0 succeeds (theta=-174.1 deg, theta_dot=-1.00, return gap=+18.2).
