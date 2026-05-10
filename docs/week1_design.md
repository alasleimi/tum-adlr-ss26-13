# Week 1 Design Document: SAC Reliability and Replay Inspection

## Scope

The proposal's first concrete milestone is "SAC metrics and replay inspection." Week 1 therefore implements a controlled baseline rather than a partial SimbaV2 implementation. The deliverable is a reproducible SAC training pipeline for:

- `Pendulum-v1` through Gymnasium.
- `dm_control/cartpole-swingup-v0` through Shimmy's Gymnasium compatibility layer.

The Week 1 code is intentionally instrumented for the later SimbaV2 claim: if replay contains useful near-upright transitions but evaluation still collapses, the failure is less likely to be pure exploration and more likely to involve optimization, critic instability, or plasticity.

## Research Questions

1. Does standard SAC reach high mean return on tasks with known solvable controllers?
2. Does high mean return hide low worst-seed performance, collapse events, or poor strict-threshold success?
3. Does replay contain near-upright behavior before the policy is reliable?
4. Are optimization diagnostics already abnormal before or during reliability failures?

## Environments

`Pendulum-v1` is the dense-reward sanity check. Exploration is comparatively easy because upright states are reachable and rewarded continuously. A failure after replay contains many near-upright states is strong evidence against "not enough exploration" as the only explanation.

`dm_control/cartpole-swingup-v0` is the DeepMind Control swing-up task. Shimmy exposes it as a Gymnasium environment. Observations are flattened before reaching SAC. For DMC cartpole, Shimmy provides `position[cart_x, pole_cos, pole_sin]` and `velocity[cart_v, pole_v]`; the near-upright detector therefore uses `pole_cos >= threshold` and small pole velocity.

## Baseline Algorithm

The Week 1 algorithm is CleanRL's continuous-action SAC, vendored from CleanRL commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`:

- Squashed Gaussian actor from `cleanrl/sac_continuous_action.py`.
- Twin Q critics and Polyak target networks from `cleanrl/sac_continuous_action.py`.
- Off-policy replay with uniform sampling.
- Time-limit truncations are not treated as terminal for bootstrapping.
- Deterministic evaluation uses the actor mean passed through `tanh`.

No SimbaV2 modification is active in Week 1. This matters academically because the baseline telemetry defines the counterfactual for later normalization, projection, distributional critic, and plasticity interventions. Local code only wraps the copied CleanRL implementation with task-specific environment flattening, reliability evaluation, replay inspection, diagnostics, and checkpointing.

## Outcome Metrics

Training is not judged by mean return alone. Each run logs:

- Episode return and episode length.
- Evaluation mean, median, standard deviation, best, and worst return.
- Evaluation return percentiles: p05, p10, p25, p75, p90, p95.
- Evaluation success rate: fraction of evaluation episodes with return at or above the configured success threshold.
- Strict success rate: return success plus near-upright fraction and maximum not-near-upright streak. Return-only success remains the primary comparable metric; strict success is the conservative "nines" metric.
- Wilson 95% confidence interval for evaluation success rate.
- Collapse rate: fraction of evaluation episodes at or below the configured collapse threshold.
- Per-episode evaluation table with fixed seed, return, length, near-upright fraction, minimum step reward, and longest not-near-upright streak.
- Per-episode return-success, stability-success, streak-success, strict-success, and collapse labels.
- Strict-threshold fractions, e.g. fraction of eval episodes above `-250, -200, -150, -100` for Pendulum or `800, 850, 900, 950` for DMC CartPole Swingup.
- Worst-seed return and fraction of seeds reaching strict thresholds are computed by `python -m last_nine_rl.aggregate`.
- Evaluation is logged at step 0 and at the final environment step regardless of interval alignment, so aggregate reports never use a stale last checkpoint.

Default thresholds are deliberately conservative and configurable. They are operational criteria for Week 1 curves, not final claims of perfect control.

## Reliability Criteria

A later "n nines" claim should not be made from a handful of episodes. Recommended reporting rules:

- Development check: at least 10 seeds and 10 evaluation episodes per seed.
- Reliability claim: at least 30 seeds and 100 evaluation episodes per seed.
- Every checkpoint for a given experiment condition must use the same fixed evaluation seed list. This prevents learning curves from changing only because the evaluation initial states changed.
- Run directories are immutable by default. A rerun into the same directory must use explicit overwrite semantics so telemetry cannot silently mix runs.
- Same-seed repeats are useful only as nondeterminism audits. Statistical comparisons should use independent actual seeds; the sweep manifest records both the base seed and the actual seed so this distinction is visible.
- Aggregation uses actual seed as the statistical unit. If several run directories share an actual seed, the summary reports them as duplicates and keeps raw run-level results for inspection.
- Report confidence intervals for success rates, preferably Wilson or Clopper-Pearson intervals for binomial success and bootstrap intervals over seeds for return statistics.
- Always report worst seed and collapse frequency alongside mean return.
- A seed is considered collapsed if any evaluation phase after first success has collapse rate above zero or if final worst episode return is below the collapse threshold.
- A method only "improves reliability" if it improves worst-seed and collapse metrics without degrading median performance beyond the bootstrap confidence interval.

## Replay Inspection

Replay summaries are logged periodically:

- Buffer size and fill fraction.
- Reward mean, standard deviation, min, and max.
- Done fraction.
- Near-upright observation fraction.
- Near-upright next-observation fraction.
- Any-transition near-upright fraction.
- Transition age mean and max.
- Sample count mean and max.
- Mean and max action magnitude.
- Action saturation fraction.

The key diagnostic comparison is:

- Replay near-upright fraction high, evaluation success low: likely optimization, value estimation, policy extraction, or plasticity problem.
- Replay near-upright fraction near zero, evaluation success low: likely exploration, curriculum, or action-sampling problem.

## Pendulum Reference Controller

Week 1 includes an energy-shaping plus local-PD reference controller for `Pendulum-v1`. It is not claimed to be the exact optimal finite-horizon policy. Its role is threshold calibration: if the reference controller cannot satisfy a candidate success threshold reliably under Gymnasium torque limits and 200-step episodes, that threshold is probably too strict for a Week 1 "nines" claim.

The controller uses Gymnasium's convention where `theta = 0` is upright:

- Energy: `E = 0.5 * theta_dot^2 + a * cos(theta)`, with `a = 3g/(2l)`.
- Desired energy: `E* = a`.
- Swing-up torque: `u = -k * (E - E*) * theta_dot`, clipped to the environment torque limit.
- Near upright, it switches to `u = -Kp * theta - Kd * theta_dot`.

Run it with `python -m last_nine_rl.reference --config configs/week1_pendulum.json --episodes 100`.

## Optimization Telemetry

SAC update telemetry logs:

- Q1, Q2, and combined Q loss.
- Actor loss.
- Entropy temperature `alpha` and alpha loss.
- Policy log-probability mean and entropy estimate.
- Q value mean and target Q mean.
- Actor and critic parameter norms.
- Actor and critic gradient norms.
- Actual optimizer update scale: `norm(parameters_after_step - parameters_before_step) / norm(parameters_before_step)`.

Update telemetry is accumulated between logging intervals. The raw metric name stores the latest value in the window, and `_mean`, `_min`, and `_max` variants summarize all optimizer updates in that window. This matters when update-to-data ratio is greater than one.

Diagnostics additionally log hidden-layer health:

- Mean and max absolute activation per actor and critic layer.
- Dormant fraction: units whose mean absolute activation is below a relative threshold times the layer mean.
- Effective-rank fraction from singular values of batch activations.

These metrics are chosen because the proposal's SimbaV2 hypothesis is about feature norm growth, parameter norm growth, unstable effective learning rates, feature rank loss, dormant units, and critic instability. For Adam, actual optimizer update scale is the learning-rate diagnostic because it measures the parameter movement after Adam's moment normalization.

## Sizes to Vary

Scale is part of the scientific question, so Week 1 records the variables that later sweeps should vary:

- Interaction budget: `25k, 100k, 500k, 1M` environment steps.
- Critic and actor width: `128, 256, 512, 1024`.
- Network depth: `2, 3, 4` hidden layers.
- Update-to-data ratio: `1, 2, 4, 8`.
- Batch size: `256, 512`.
- Replay capacity: at least the interaction budget for small tasks, plus smaller-capacity stress tests later.

Only one scale axis should change at a time in early sweeps. Full factorial grids are expensive and can blur the causal interpretation. The copied CleanRL baseline is fixed at two hidden layers of width 256; width/depth sweeps should be implemented as a separate, explicitly named CleanRL-derived variant rather than hidden behind baseline config fields.

Update-to-data ratio is interpreted in gradient-update units: increasing UTD adds additional sampled updates per environment step, while actor and target update frequencies are applied to the one-indexed gradient update counter. This avoids changing actor/target cadence only on environment steps divisible by the policy frequency.

## Success Criteria for Week 1

Week 1 is complete when:

1. SAC can be launched on Pendulum and DMC CartPole Swingup from checked-in configs.
2. Runs produce a config snapshot, scalar metrics, structured event logs, replay summaries, diagnostics, and final checkpoint.
3. Evaluation logs include mean return, success rate, worst return, collapse rate, and strict-threshold fractions.
4. Aggregation across run directories reports worst-seed and threshold-crossing statistics.
5. Unit tests cover environment creation, replay inspection, action bounds, and one SAC update.

## Files

- `src/last_nine_rl/train.py`: training CLI and run loop.
- `src/cleanrl/sac_continuous_action.py`: copied CleanRL SAC source.
- `src/cleanrl_utils/buffers.py`: copied CleanRL replay buffer dependency.
- `src/last_nine_rl/sac.py`: telemetry wrapper around the copied CleanRL actor, critics, and update logic.
- `src/last_nine_rl/replay.py`: CleanRL replay buffer subclass plus replay-summary helpers.
- `src/last_nine_rl/sweep.py`: sequential multi-seed/scale launcher with a manifest.
- `src/last_nine_rl/visualize.py`: post-training PNG/HTML report generator.
- `src/last_nine_rl/envs.py`: Gymnasium and DMC environment factory plus near-upright detection.
- `src/last_nine_rl/replay.py`: replay storage and replay summary metrics.
- `src/last_nine_rl/telemetry.py`: `events.jsonl`, `metrics.csv`, and config snapshots.
- `src/last_nine_rl/aggregate.py`: multi-seed reliability aggregation.
- `configs/week1_*.json`: runnable baseline configs.
- `configs/week1_scale_grid.json`: recommended scale variables.
