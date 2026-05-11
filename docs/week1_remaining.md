# Week 1 Remaining Work

This checklist is the current boundary between the Week 1 baseline milestone and the later SimbaV2 intervention work.

## Already In Place

- CleanRL SAC baseline with fixed configs for `Pendulum-v1` and DMC CartPole Swingup.
- TensorBoard, CSV, JSONL, checkpoint, replay, optimizer, activation, and effective-rank telemetry.
- Pendulum 100k and 500k UTD1 five-seed datasets.
- Pendulum post-hoc 1000-episode evals, reset-support initial-condition maps, DP calibration, energy-controller baseline, and relative success metrics.
- Regret-map convention fixed: `regret` now means nonnegative shortfall, with signed return gaps plotted separately.

## Still Needed For A Strong Week 1 Package

1. Finish and summarize the active scale runs.
   - Complete the currently running `pendulum_250k_utd2_buffer500k` condition or explicitly mark it as incomplete. As of this pass, seeds 0 and 1 reached `250000` steps and seed 2 appears stopped around `76000` steps, so the condition is not report-ready.
   - If the planned `500k UTD2` condition is still required, run it under the same post-hoc and grid pipeline.
   - Keep UTD comparisons separated from the CleanRL-equivalent UTD1 baseline.

2. Decide the DMC CartPole Swingup scope.
   - The existing DMC result is a diagnostic, not a full reliability dataset.
   - For a Week 1 claim, run at least 5 seeds with larger post-hoc eval coverage, or label DMC as infrastructure-only.
   - Add DMC-specific task-state success plots; return threshold alone was misleading there too.

3. Tighten the reliability statistics.
   - Five seeds are enough for debugging, not for a "nines" claim.
   - Development reliability should move toward 10 seeds; any final reliability claim needs substantially more seeds and fixed eval episodes.
   - Report seed-level intervals first, pooled Wilson intervals second.

4. Finish the human-review artifacts.
   - Ask humans to inspect TensorBoard traces, initial-state maps, and representative failed rollouts.
   - Add short notes for any visually obvious failure mode: wrong swing direction, late stabilization, action saturation, or critic/actor instability.

5. Keep DP and controller caveats explicit.
   - DP is an approximate finite-horizon calibration, not an oracle proof.
   - The controller is a reference baseline, not an optimal policy.
   - Main Pendulum success should remain task-only and reference-relative; `return >= -200` should remain diagnostic-only.

## SimbaV2 Readiness Gate

Start SimbaV2 implementation only after the baseline comparison surface is stable:

- Same fixed eval seeds and initial-state grids are reusable for SAC and SimbaV2.
- Metrics distinguish task success, diagnostic threshold passes, reference-relative success, nonnegative shortfall, and signed gaps.
- The active scale-run outputs are either completed and summarized or clearly excluded.

The first SimbaV2 code step should be an explicitly named architecture variant, not a hidden change to the CleanRL baseline. The [ICML 2025 SimbaV2 paper](https://proceedings.mlr.press/v267/lee25u.html) frames the method around hyperspherical normalization, distributional value estimation, and reward scaling; those should be introduced as separate ablations so the project can attribute any reliability change.
