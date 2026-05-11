# Experiment Log

## 2026-05-11 Completed 500k UTD1 And Relative Success Metrics

Status: completed.

Purpose:
- Update the 500k UTD1 condition now that all five seeds completed.
- Add task-only, DP-relative, controller-relative, and best-known-baseline-relative success metrics for both 100k and 500k.
- Reframe `return >= -200` as a legacy diagnostic threshold, not the main Pendulum success definition.
- Fix the regret-map convention: `regret` now means nonnegative shortfall `max(0, reference_return - SAC_return)`, while signed return gaps are stored and plotted separately.

Completed 500k UTD1 artifacts:
- Runs: `runs/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k`, seeds 0-4.
- Aggregate: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/aggregate.json`.
- Post-hoc eval: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/posthoc_1000eps/posthoc_eval_summary.json`.
- Reset-support grid: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41`.

500k post-hoc diagnostics:
- Mean seed mean return: -139.3497.
- Mean seed fixed-threshold diagnostic: 0.7028.
- Mean seed strict-threshold diagnostic: 0.6802.
- Pooled fixed-threshold diagnostic: 3514/5000 = 0.7028, Wilson 95% [0.6900, 0.7153].
- Pooled strict-threshold diagnostic: 3401/5000 = 0.6802, Wilson 95% [0.6671, 0.6930].

Relative-success commands:
- 100k: `python -m last_nine_rl.pendulum_relative --condition-label "Pendulum SAC 100k" --sac-rollouts reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_rollouts.csv --dp-grid reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv --controller-grid reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_grid.csv --out reports/pendulum_investigation_20260509/relative_success_100k --epsilon-return 5.0`
- 500k: `python -m last_nine_rl.pendulum_relative --condition-label "Pendulum SAC 500k UTD1" --sac-rollouts reports/pendulum_investigation_20260509/pendulum_500k_utd1_buffer500k/grid_reset_support_61x41/pendulum_grid_rollouts.csv --dp-grid reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv --controller-grid reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_grid.csv --out reports/pendulum_investigation_20260509/relative_success_500k_utd1 --epsilon-return 5.0`

Key paired 500k minus 100k results over the reset-support grid:
- Fixed-threshold diagnostic: +0.0009, 95% CI [-0.0007, +0.0024], uncorrected paired p = 0.1894.
- Task-only success: +0.0268, 95% CI [-0.0873, +0.1409], uncorrected paired p = 0.5499.
- Strict-threshold diagnostic: +0.0056, 95% CI [-0.0192, +0.0304], uncorrected paired p = 0.5645.
- Beats DP: +0.0500, 95% CI [+0.0090, +0.0909], uncorrected paired p = 0.0275.
- Near DP within 5 return points: +0.0741, 95% CI [-0.0815, +0.2296], uncorrected paired p = 0.2567.
- Beats best known `max(DP, controller)`: +0.0509, 95% CI [+0.0087, +0.0931], uncorrected paired p = 0.0287.
- Near best known within 5 return points: +0.0742, 95% CI [-0.0812, +0.2296], uncorrected paired p = 0.2555.

Interpretation:
- The fixed-threshold diagnostic is essentially unchanged from 100k to 500k.
- 500k is closer to the DP/controller references, especially on exact beats-DP and beats-best-known metrics.
- The robust near-DP and near-best-known metrics are high for both conditions; their paired intervals are wide because the 100k seed-level rates have one low outlier.
- The regret-map verification found no state-grid join mismatch: DP, controller, 100k SAC, and 500k SAC all share the same 2501 reset-support cells.
- Detailed writeups: `docs/pendulum_500k_results.md`, `docs/pendulum_relative_success_results.md`, and `docs/pendulum_models_and_success_criteria.md`.

## 2026-05-10 Pendulum DP Feasibility Calibration

Status: completed.

Purpose:
- Check whether the fixed Pendulum success threshold `return >= -200` is feasible from the hard initial states found in the 100k checkpoint map.
- Join a finite-horizon dynamic-programming calibration to the existing 100k SAC checkpoint grid without retraining.

Command:
- `python -m last_nine_rl.pendulum_dp --out reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81 --sac-grid reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_summary.csv --horizon 200 --theta-bins 241 --velocity-bins 161 --action-bins 81 --eval-theta-bins 61 --eval-velocity-bins 41 --eval-velocity-limit 1.0 --save-solution`

Sensitivity check:
- `python -m last_nine_rl.pendulum_dp --out reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_361x241x101_check --sac-grid reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/pendulum_grid_summary.csv --horizon 200 --theta-bins 361 --velocity-bins 241 --action-bins 101 --eval-theta-bins 61 --eval-velocity-bins 41 --eval-velocity-limit 1.0`

Primary results:
- DP return-feasible cells: 1736/2501 = 0.6941.
- DP strict-feasible cells: 1734/2501 = 0.6933.
- SAC 100k return-success cell fraction on the same grid: 0.6918.
- SAC 100k strict-success cell fraction on the same grid: 0.6692.
- SAC failure rate among DP return-feasible cells: 0.0033.
- SAC strict-failure rate among DP strict-feasible cells: 0.0348.
- Region `|theta| >= 150 degrees`: DP return-feasible cells 0/451, DP mean return -240.94.
- Region `|theta| >= 150 degrees` and `|theta_dot| <= 0.5`: DP return-feasible cells 0/231, DP mean return -241.92.

Sensitivity result:
- The finer 361 by 241 state grid with 101 actions produced the same feasible-cell counts: 1736 return-feasible cells, 1734 strict-feasible cells, and zero near-down feasible cells.

Interpretation:
- The original near-down SAC failure map should not be read as pure RL failure under the fixed `-200` threshold. Approximate DP also fails that region under the same threshold.
- The better Week 1 conclusion is that SAC nearly matches the DP return-feasible mask, while a small strict-stabilization gap remains on DP-strict-feasible cells.
- Detailed writeup: `docs/pendulum_dp_calibration.md`.

## 2026-05-09 Pendulum Reliability Investigation

Status: running.

Purpose:
- Check whether the low Pendulum reliability is due to a CleanRL configuration or implementation bug.
- Map success by exact initial condition, especially over Gymnasium's reset support.
- Launch a longer GPU sweep that separates environment-interaction budget from optimizer-update budget.

Code/infrastructure changes:
- Added checkpoint loading and large post-hoc fixed-seed evaluation: `python -m last_nine_rl.posthoc_eval`.
- Added exact Pendulum initial-condition grid evaluation: `python -m last_nine_rl.pendulum_grid`.
- Added train CLI overrides for `--buffer-size`, `--batch-size`, and `--updates-per-step`.
- Added regression tests for checkpoint loading and grid cell aggregation.
- Wrote `docs/cleanrl_audit.md`.

Verification:
- `python -m pytest` passed: 18 tests.

CleanRL audit result:
- No obvious SAC-loop, terminal/truncation, alpha-update, actor/critic architecture, or replay-buffer bug explains the low reliability.
- The 100k baseline is CleanRL SAC plus telemetry. The only exact-equivalence caveat is that UTD variants use optimizer-step actor gating and should be labeled scale variants, not exact CleanRL.
- See `docs/cleanrl_audit.md`.

Post-hoc 100k Pendulum evaluation:
- Runs: `runs/week1_real_gpu_20260509/pendulum_100k`, seeds 0-4.
- Eval: 1000 fixed deterministic eval seeds per training seed, seed base 200000.
- Mean seed mean return: -140.6066.
- Mean seed return success: 0.7012.
- Mean seed strict success: 0.6734.
- Pooled return success: 3506/5000 = 0.7012.
- Pooled strict success: 3367/5000 = 0.6734.
- Report: `reports/pendulum_investigation_20260509/posthoc_100k_1000eps/posthoc_eval_summary.json`.

Initial-condition maps:
- Full Pendulum state map, `theta_bins=61`, `theta_dot_bins=41`, velocity range `[-8, 8]`:
  - Mean cell return success: 0.9226.
  - Mean cell strict success: 0.9056.
  - Report: `reports/pendulum_investigation_20260509/pendulum_grid_100k_61x41/index.html`.
- Gymnasium reset-support map, `theta_bins=61`, `theta_dot_bins=41`, velocity range `[-1, 1]`:
  - Mean cell return success: 0.6918.
  - Mean cell strict success: 0.6692.
  - Cells with all five training seeds strict-successful: 0.6385.
  - Region `|theta| >= 150 degrees`: mean return success 0.0 and mean strict success 0.0 across 451 cells.
  - Region `|theta| >= 150 degrees` and `|theta_dot| <= 0.5`: mean return success 0.0 and mean strict success 0.0 across 231 cells.
  - Region `60 <= |theta| <= 120 degrees`: mean return success 1.0 and mean strict success 1.0 across 820 cells.
  - Report: `reports/pendulum_investigation_20260509/pendulum_grid_100k_reset_support_61x41/index.html`.
- Interpretation: the reliability gap is concentrated around downward starts with low or opposing angular velocity. The full-state map looks better because high angular velocities often make swing-up easier; the reset-support map is the relevant one for Gymnasium evaluation.

Longer GPU sweep:
- Script: `reports/pendulum_investigation_20260509/run_pendulum_investigation.ps1`.
- Run root: `runs/pendulum_investigation_20260509`.
- Background runner PID: 63332, started 2026-05-09 18:33:36 local time. Runner metadata: `reports/pendulum_investigation_20260509/logs/runner_process.json`.
- The first runner launch failed because PowerShell split the `Project 15` path; it was relaunched with an explicitly quoted script path.
- Planned conditions:
  - `pendulum_500k_utd1_buffer500k`, seeds 0-4.
  - `pendulum_250k_utd2_buffer500k`, seeds 0-4. This roughly matches the optimizer-update count of 500k UTD1 with half the environment interaction.
  - `pendulum_500k_utd2_buffer500k`, seeds 0-2. This checks whether more updates on the larger interaction budget reduce the hard-start failure mode.
- Each training run uses CUDA, checkpointing, TensorBoard, 20 fixed eval episodes at 100k intervals, replay diagnostics every 25k steps, and post-hoc 1000-episode evaluation after each condition completes.

Partial result, 2026-05-10 13:16 local time:
- Completed runs: `pendulum_500k_utd1_buffer500k` seeds 0, 1, and 2.
- Still running: `pendulum_500k_utd1_buffer500k` seed 3, checked at step 187800/500000.
- Built-in final 20-episode eval for completed seeds:
  - seed 0: mean return -159.0613, return success 0.55, strict success 0.55.
  - seed 1: mean return -157.7837, return success 0.55, strict success 0.55.
  - seed 2: mean return -161.2696, return success 0.55, strict success 0.55.
- Post-hoc eval on completed 500k UTD1 seeds 0-2, 1000 fixed eval episodes each:
  - Mean seed mean return: -139.6041.
  - Mean seed return success: 0.7023.
  - Mean seed strict success: 0.6773.
  - Pooled return success: 2107/3000 = 0.7023.
  - Pooled strict success: 2032/3000 = 0.6773.
  - Report: `reports/pendulum_investigation_20260509/pendulum_500k_utd1_partial_seed0_2/posthoc_1000eps/posthoc_eval_summary.json`.
- Interpretation: so far, increasing the CleanRL baseline from 100k to 500k environment steps with UTD1 does not materially improve Pendulum reliability. The comparable 100k five-seed post-hoc result was return success 0.7012 and strict success 0.6734.


## 2026-05-09 Week 1 Debug and Extended Data Pass

Status: completed.

Purpose:
- Debug why the 2026-05-08 Week 1 dataset is not enough for the milestone.
- Separate implementation failures from insufficient interaction budget and insufficient reliability evidence.
- Collect a full-budget Pendulum SAC dataset with larger final evaluation coverage.

Debug findings before launch:
- The 25k Pendulum SAC runs were not failing because of a missing Gymnasium/DMC wrapper or a logging-only metric bug. All five actual seeds showed the same hard evaluation initial states failing, while easy initial states were solved. That means the policy is not reliable across starts at 25k steps.
- The existing fixed evaluation seed set is useful for paired comparisons across training seeds, but pooled episode Wilson intervals should be interpreted cautiously because the same initial states are reused across seeds.
- The 25k DMC CartPole Swingup run is only a checked pipeline run. Its final success rate was 0/10 and it should not be treated as a solved baseline.
- A local benchmark showed CUDA is faster for this code path despite single-environment overhead: 5k Pendulum steps took 225.08 s on CPU and 136.23 s on CUDA.
- The GPU was already shared with desktop applications before launch, including a game process, so wall-clock timing may be noisy.

Run roots:
- Runs: `runs/week1_real_gpu_20260509`
- Reports: `reports/week1_real_gpu_20260509`

Planned runs:
- Pendulum SAC full-budget baseline: actual seeds 0-4, 100000 environment steps, 50 fixed eval episodes at steps 0, 25000, 50000, 75000, and 100000, CUDA device.
- After Pendulum aggregation, run additional DMC CartPole Swingup diagnostics if wall-clock time and GPU headroom allow. The prepared diagnostic plan is seeds 0-2, 100000 environment steps, 5 fixed eval episodes at 0/25k/50k/75k/100k.

Commands:
- Pendulum 100k seeds 0-4 via `reports/week1_real_gpu_20260509/logs/run_pendulum_100k.ps1`.
- Optional DMC CartPole 100k seeds 0-2 via `reports/week1_real_gpu_20260509/logs/run_dmc_cartpole_100k.ps1`.

Instrumentation changes during run:
- Added aggregate-level fixed evaluation seed difficulty summaries. These identify hard eval seeds by mean return and strict success rate across actual training seeds, so the report can distinguish concentrated hard-start failures from broad random unreliability.
- Covered the new aggregate fields with `tests/test_aggregate.py`.

Intermediate analysis:
- Re-aggregated the 2026-05-08 Pendulum 25k dataset with fixed-eval-seed difficulty fields at `reports/week1_real_gpu_20260509/pendulum_25k_reaggregate_with_eval_seed_difficulty.json`.
- The 25k policy has systematic hard-start failures: the hardest eval seed averaged about -259.64 return across five actual seeds, and the bottom eval seeds had 0/5 strict success. This supports collecting the 100k run rather than treating the 25k result as sufficient.

Pendulum SAC 100k GPU baseline:
- Runs: 5 actual seeds, seeds 0-4.
- Eval: 50 fixed eval episodes at steps 0, 25k, 50k, 75k, and 100k.
- All five runs completed with `config.json`, `events.jsonl`, `metrics.csv`, `eval_episodes.csv`, TensorBoard event files, and `checkpoints/final.pt`.
- Final mean seed mean return: -159.4748.
- Final worst seed mean return: -161.2780.
- Final worst evaluated episode return: -297.9396.
- Final pooled return success: 150/250 = 0.60, Wilson 95% [0.5382, 0.6588].
- Final pooled strict success: 147/250 = 0.588, Wilson 95% [0.5261, 0.6472].
- Fixed eval seed difficulty: 50 unique eval seeds; several eval seeds have 0/5 success across actual seeds. Hardest final eval seed was 100043 with mean return -287.7891 and 0/5 strict success.
- Interpretation: extending Pendulum from 25k to 100k does not solve the hard-start reliability issue. The baseline converges to a stable partial solution around 60% return success on this fixed eval suite.
- Aggregate: `reports/week1_real_gpu_20260509/pendulum_100k_aggregate.json`.
- HTML report: `reports/week1_real_gpu_20260509/pendulum_100k_html/index.html`.

DMC CartPole Swingup 100k GPU diagnostic:
- Runs: 3 actual seeds, seeds 0-2.
- Eval: 5 fixed eval episodes at steps 0, 25k, 50k, 75k, and 100k.
- All three runs completed with `config.json`, `events.jsonl`, `metrics.csv`, `eval_episodes.csv`, TensorBoard event files, and `checkpoints/final.pt`.
- Final mean seed mean return: 855.2763.
- Final worst seed mean return: 849.4121.
- Final pooled return success at threshold 850: 10/15 = 0.6667, Wilson 95% [0.4171, 0.8482].
- Final pooled strict success: 0/15 = 0.0, Wilson 95% upper bound 0.2039.
- Seed 0 ended just below the return threshold with mean return 849.4121. Seeds 1 and 2 exceeded the return threshold, but all seeds failed strict success because the maximum not-near-upright streak stayed around 172-190 steps.
- Interpretation: the DMC pipeline and learning signal are healthy, but the current 100k diagnostic does not establish reliable swingup under the strict Week 1 criteria. Threshold ladders and strict stability metrics are necessary here because return success and sustained stability disagree.
- Aggregate: `reports/week1_real_gpu_20260509/dmc_cartpole_100k_aggregate.json`.
- HTML report: `reports/week1_real_gpu_20260509/dmc_cartpole_100k_html/index.html`.

Verification:
- `python -m pytest` passed: 16 tests.
- TensorBoard started for local inspection at `http://127.0.0.1:6006` with logdir `runs`; process metadata is in `reports/week1_real_gpu_20260509/logs/tensorboard.json`.

Proposal-alignment follow-up:
- Added `docs/proposal_alignment.md` to clarify that the project target is the last-nine reliability gap, not only mean-return SAC training.
- Added `python -m last_nine_rl.compare` for cross-seed proposal-relevant plots.
- Generated comparison reports:
  - `reports/week1_real_gpu_20260509/pendulum_100k_compare/index.html`
  - `reports/week1_real_gpu_20260509/dmc_cartpole_100k_compare/index.html`

## 2026-05-08 Week 1 GPU Data Collection

Status: completed.

Purpose:
- Collect real Week 1 data for the CleanRL SAC baseline rather than only smoke tests.
- Keep raw run directories, TensorBoard data, checkpoints, per-episode evaluation CSVs, aggregates, and HTML plots.
- Calibrate Pendulum success thresholds with the reference energy-shaping controller.

Hardware/software:
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB.
- PyTorch: 2.5.1+cu121.
- CUDA visible to PyTorch: yes.
- Shimmy installed: yes.
- dm_control installed: yes.
- TensorBoard installed: yes.

Run roots:
- Runs: `runs/week1_real_gpu_20260508`
- Reports: `reports/week1_real_gpu_20260508`

Planned runs:
- Pendulum reference controller: 100 eval episodes, seed base 100000.
- Pendulum SAC baseline: seeds 0-4, 25000 environment steps, 20 eval episodes per checkpoint, CUDA device.
- DMC CartPole Swingup SAC baseline: run after Pendulum if wall-clock time remains reasonable; at minimum produce a GPU-backed checked run and mark any incomplete long runs explicitly.

Notes:
- `success_rate` is return-threshold success.
- `strict_success_rate` is return success plus near-upright fraction plus maximum not-near-upright streak.
- Aggregation uses actual seed as the statistical unit.

Execution notes:
- A first 100000-step Pendulum run was started for seed 0, but measured runtime implied several hours for 5 seeds on the single-environment loop. It was stopped at about 17k steps and moved to `runs/week1_real_gpu_20260508/partial_100k_seed0_aborted`.
- The completed Week 1 dataset is therefore scoped to the 25k interaction-budget point plus reference calibration, with any DMC run logged separately.

Commands:
- `python -m last_nine_rl.reference --config configs/week1_pendulum.json --episodes 100 --seed-base 100000 --out reports/week1_real_gpu_20260508/pendulum_reference.json`
- Pendulum 25k seeds 0-4 via `reports/week1_real_gpu_20260508/logs/run_pendulum.ps1`.
- `python -m last_nine_rl.aggregate --runs runs/week1_real_gpu_20260508/pendulum_25k --thresholds -250 -200 -150 -100`
- `python -m last_nine_rl.visualize --runs runs/week1_real_gpu_20260508/pendulum_25k --out reports/week1_real_gpu_20260508/pendulum_25k_html`
- DMC CartPole 25k seed 0 via `reports/week1_real_gpu_20260508/logs/run_dmc_cartpole.ps1`.
- `python -m last_nine_rl.aggregate --runs runs/week1_real_gpu_20260508/dmc_cartpole_25k --thresholds 800 850 900 950`
- `python -m last_nine_rl.visualize --runs runs/week1_real_gpu_20260508/dmc_cartpole_25k --out reports/week1_real_gpu_20260508/dmc_cartpole_25k_html`

Pendulum reference calibration:
- Episodes: 100 fixed eval seeds, seed base 100000.
- Mean return: -148.7746.
- Worst return: -355.4701.
- Return/strict success rate at current thresholds: 0.68.
- Wilson 95% success interval: [0.5834, 0.7633].
- Interpretation: current Pendulum threshold `return >= -200` is not an oracle-perfect criterion even for the hand-designed controller. It is useful as a strict operational threshold, but later claims should calibrate or report a threshold ladder.

Pendulum SAC 25k GPU baseline:
- Runs: 5 actual seeds, seeds 0-4.
- Eval: 20 fixed eval episodes per checkpoint, final step 25000.
- Final mean seed mean return: -158.8715.
- Final worst seed mean return: -162.1293.
- Final worst evaluated episode return: -276.0048.
- Final pooled return success: 55/100 = 0.55, Wilson 95% [0.4524, 0.6439].
- Final pooled strict success: 55/100 = 0.55, Wilson 95% [0.4524, 0.6439].
- Final collapse frequency: 0.0.
- Final mean near-upright fraction: 0.84165.
- Conservative nines from Wilson lower bound: about 0.18 nines, so this is not a reliability result yet.
- Aggregate: `reports/week1_real_gpu_20260508/pendulum_25k_aggregate.json`.
- HTML report: `reports/week1_real_gpu_20260508/pendulum_25k_html/index.html`.

DMC CartPole Swingup 25k GPU checked run:
- Runs: 1 actual seed, seed 0.
- Eval: 10 fixed eval episodes per checkpoint, final step 25000.
- Final mean return: 177.7461.
- Final success and strict success: 0/10.
- Final collapse rate: 1.0 under current `return <= 200` collapse threshold.
- Final near-upright fraction: 0.0114.
- Replay near-upright-any-transition fraction at 20000 steps: 0.0024.
- Interpretation: the DMC pipeline works through Shimmy/Gymnasium and produces the right diagnostics, but 25k steps is not enough to solve or meaningfully assess reliability.
- Aggregate: `reports/week1_real_gpu_20260508/dmc_cartpole_25k_aggregate.json`.
- HTML report: `reports/week1_real_gpu_20260508/dmc_cartpole_25k_html/index.html`.

Artifact check:
- Pendulum 25k: 5/5 runs completed; each has `config.json`, `events.jsonl`, `metrics.csv`, `eval_episodes.csv`, TensorBoard event file, and `checkpoints/final.pt`.
- DMC CartPole 25k: 1/1 run completed with the same artifact set.
