# Project 15 Week 3 Workshop Script

This script is written for an 8 minute talk. Speaker A covers slides 1-5. Speaker B covers slides 6-10. Backup slides are for questions.

## Slide 1, Speaker A, 0:00-0:40
Our project is about reliability in deep reinforcement learning. Pendulum is not supposed to be a hard benchmark, so average return is not the interesting part. The interesting part is the last set of initial states where a neural SAC policy still fails to swing up and stabilize.

The GIF is not random. It is the same initial state from the exact evaluation grid: initial angle is minus 174.1 degrees and angular velocity is minus 1.0. SAC seed 0 fails there, while full SimbaV2 seed 0 succeeds. The headline success metric is within 5 return points of max(DP, controller): SAC 100k is at 79.2%, while full SimbaV2 100k is at 92.5%. The stricter all-grid task-stability check is 75.8% versus 91.4%. Seed-level intervals are on slide 6, not on the opening slide.

## Slide 2, Speaker A, 0:40-1:35
The most important thing is the success definition. We first tried an intuitive task-stability definition: the policy should swing up, stay near upright for at least 80 percent of the episode, and not lose the upright region for more than 50 consecutive steps.

That is useful, but it is not a fair universal success definition because our references do not satisfy it everywhere. DP or the hand controller satisfies that task-stability predicate on 2336 out of 2501 cells, which is 93.4%. The remaining cells are not certified by either reference under that behavioral predicate.

So the headline success metric is state-conditioned return matching: the policy return must be within 5 of max(DP, controller) from the same start. Most of the time DP is the better reference: 2403 out of 2501 cells. The controller is better on 98 cells, so using the max matters.

## Slide 3, Speaker A, 1:35-2:15
The evaluation grid has 61 angle bins and 41 angular-velocity bins, so 2501 reset-support states. For the 100k comparison we have three training seeds, so 7503 deterministic grid rollouts.

In the maps, each cell is the fraction of seeds that satisfy the plotted criterion from that exact state. With three seeds, that means zero, one third, two thirds, or all seeds. The references are finite-horizon DP and an energy-swing-up plus PD hand controller. The diagnostics track replay coverage and critic representation health.

## Slide 4, Speaker A, 2:15-3:05
For listeners who have not read SimbaV2, the paper is not just "a bigger SAC network." It makes four concrete changes to SAC.

First, hyperspherical feature normalization means L2-normalizing hidden feature vectors so their scale does not drift. Second, hyperspherical weight normalization means removing weight decay and projecting selected weights back to unit norm after each update. Third, the critic becomes distributional: it predicts binned return probabilities, and reward scaling keeps the target variance in a stable range. Fourth, the paper uses a different SAC recipe from CleanRL: smaller learning rates, lower initial entropy weight, different target entropy scale, smaller Simba networks, and the distributional critic.

The table gives the exact side-by-side settings from our 100k runs. The important scientific point is that we should not claim which component drives the improvement yet. The short-budget ablations were useful for debugging, but not representative enough for the talk.

## Slide 5, Speaker A, 3:05-3:50
These are the raw maps. The left column is task-stability success. The middle column is the headline metric: within 5 return points of max(DP, controller). The right column is the continuous version of that same reference comparison: how many return points the policy is below max(DP, controller). The first row is SAC 100k and the second row is full SimbaV2 100k.

The right column uses a color cap at 20 return points. That is not changing the numbers; it only saturates the color scale. We do it because a few cells are more than 100 return points below the reference, and a full-range color scale would make all ordinary near-boundary differences look the same.

## Slide 6, Speaker B, 3:50-4:35
Here is the seed-level main result. The dots are training seeds and the diamonds are means. The intervals are 95 percent t-intervals over seeds, so they are deliberately conservative with only three seeds.

Reference success improves from 79.2% +/- 57.6 pp for SAC to 92.5% +/- 5.7 pp for full SimbaV2. The stricter all-grid task-stability check improves from 75.8% +/- 64.6 pp to 91.4% +/- 3.9 pp. On known-feasible cells, task-stability improves from 80.9% +/- 68.6 pp to 97.3% +/- 3.3 pp. SAC has one very bad seed, which is why its interval is huge. We should present that uncertainty explicitly.

## Slide 7, Speaker B, 4:35-5:25
This is the exploration versus optimization diagnosis. If SAC failed only because it never saw useful states, replay near-upright coverage should separate. It does not: SAC and full SimbaV2 are both about 82.6 to 82.7 percent.

The separation is critic health. Dormant units are critic units that barely activate, so lower is better. Effective rank is a proxy for how diverse the critic features are, so higher is better. SAC has high dormancy and low rank, while SimbaV2 has zero measured Q1 dormancy and much higher rank. That points to optimization, plasticity, and value estimation, not pure exploration.

## Slide 8, Speaker B, 5:25-6:05
More SAC compute is a useful negative result, but we need to state the metric carefully. SAC 500k is not worse on reference success: it reaches 96.3 percent, compared with 92.5 percent for full SimbaV2 100k.

It is still worse on the behavioral reliability metrics. Exact-grid task-stability is 88.6 percent for SAC 500k versus 91.4 percent for full SimbaV2 100k, so SAC is 2.8 percentage points behind. On near-down starts the gap is much larger: 58.4 percent versus 70.6 percent, so SAC is 12.2 percentage points behind. The critic-health signal also worsens: SAC 500k has 77.3 percent dormant Q1 units, while full SimbaV2 has 0.0 percent in this diagnostic.

## Slide 9, Speaker B, 6:05-6:45
Hard reset and hard replay test the data-distribution hypothesis more directly. Hard reset p=0.2 means 20 percent of episode resets are forced into a large-angle band: absolute theta between 120 and 135 degrees, with absolute angular velocity at most 1. Hard replay p=0.2 means 20 percent of each replay minibatch is sampled from transitions in that same hard-start band.

The graph is now a grouped bar chart because these are categorical interventions, not a progression. Ordinary full SimbaV2 50k has task-stability 89.8 percent and reference success 81.6 percent. Hard reset moves task-stability slightly to 90.3 percent, but reference success drops to 73.6 percent. Hard replay is worse on both headline metrics, at 89.0 percent task-stability and 72.0 percent reference success. Near-down task success is also worse for hard replay.

So just showing hard states more often is too blunt. The next intervention has to preserve value accuracy on the rest of the state space.

## Slide 10, Speaker B, 6:45-8:00
The next step is to move the component claims to representative budget. We should ablate away SimbaV2 components one at a time at 100k: feature normalization, weight projection, distributional critic, reward scaling, and the official SAC recipe. That tells us what actually drives the reliability improvement.

In parallel, we should push Pendulum toward at least 0.99 reference success using other reliability ideas from the proposal: ReDo, Sample Weight Decay, Fisher-guided selective forgetting, and regret-weighted auxiliary losses. After the Pendulum frontier is stable, we move the same protocol to CartPole-Swingup, where exploration is a more serious part of the problem.

## Backup / Q&A
Use the DP slide if asked whether DP is optimal. It is approximate finite-horizon DP, not a proof. Use the controller slide if asked what max(DP, controller) means. Use the bad-seed slide if asked why SAC has a huge confidence interval.

GIF status: Policy GIF: exact-grid contrast where SAC seed0 fails and full SimbaV2 seed0 succeeds (theta=-174.1 deg, theta_dot=-1.00, return gap=+18.2).
