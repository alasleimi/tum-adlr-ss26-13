# Proposal Alignment and Plot Plan

The proposal objective is not "train SAC on Pendulum." It is to explain and reduce the gap between high average return and near-perfect task success. Week 1/2 therefore needs evidence about tails, hard initial states, replay coverage, and optimization pathologies.

## What Went Wrong In The First Pass

The 25k and 100k runs were useful diagnostics, but they were underpowered for the proposal's reliability question.

- Pendulum 100k produced a reasonable average-return policy, not a reliable policy.
- The fixed eval suite exposed a consistent hard-start failure pattern: starts near the downward position with small or opposing angular velocity stay below the `-200` return threshold.
- The compute budget was too small for a scale claim. The proposal explicitly says to vary interaction budget, critic width/depth, and update-to-data ratio.
- The current CleanRL baseline has fixed `256 x 256` networks, so width/depth sweeps require a separate architecture-configurable variant rather than hidden config knobs.
- Return thresholds still need oracle or near-oracle calibration. The hand-coded Pendulum controller is a sanity reference, not proof of optimality.

## Required Graphs

Every main result should include these plots:

- Cross-seed learning curves: mean return, worst return, return success, strict success.
- Reliability nines curves: empirical and Wilson-lower-bound `-log10(failure rate)` for return success and strict success.
- Threshold ladder: fraction of eval episodes above each return threshold over training.
- Final eval heatmaps: training seed by fixed eval seed for return and strict success, with eval seeds sorted by difficulty.
- Initial-state map for Pendulum: initial `theta, theta_dot` colored by final mean return and strict success rate.
- Replay coverage vs evaluation success: replay near-upright transition fraction against strict eval success.
- Optimization health curves: critic/actor losses, `alpha`, parameter norms, gradient norms, and actual optimizer update norm ratios.
- Representation health curves: dormant fractions and effective-rank fractions for actor and critics.
- Final return distribution: per-episode returns, p05/p10/worst, and seed-level bootstrap intervals.
- Scale curves: budget, update-to-data ratio, width, and depth versus worst-seed return and strict success.

The new `python -m last_nine_rl.compare` command generates the first six aggregate plots from existing run telemetry.

## Compute Plan

Minimum proposal-aligned baseline:

- Pendulum: budgets `25k, 100k, 500k, 1M`; at least 10 actual seeds; final post-hoc eval on at least 100 fixed eval episodes per seed.
- DMC CartPole Swingup: budgets `100k, 500k, 1M`; at least 5-10 actual seeds; final post-hoc eval on at least 100 fixed eval episodes per seed.
- Update-to-data sweep: at least UTD `1, 2, 4` at `100k` and `500k`.
- Architecture sweep after the baseline runner supports it: critic width `256, 512, 1024`; depth `2, 3, 4`.

Reliability claims need more:

- At least 30 actual seeds.
- At least 100 final eval episodes per seed.
- Report seed-unit confidence intervals and fixed-eval-seed difficulty, not just pooled episode Wilson intervals.

## Immediate Technical Gaps

- Add checkpoint loading and post-hoc evaluation so large final eval sets do not slow every training checkpoint.
- Add an architecture-configurable SAC variant before claiming width/depth scaling.
- Add oracle or near-oracle Pendulum calibration, preferably numerical dynamic programming, MPC, or another model-based controller over the real Gymnasium dynamics.
- Add video or rollout rendering for the hardest eval seeds.
- Add a scale-sweep report that compares run groups by budget/UTD/architecture.
