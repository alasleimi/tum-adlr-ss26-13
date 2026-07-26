# Presentation videos

These two H.264 videos are designed for playback on the presentation laptop. Both are 1920 × 1080, 20 frames per second, and 12 seconds long. Each video shows 200 deterministic Pendulum-v1 control steps at their true simulation timing, followed by a two-second result hold.

## Primary video: same hard start, learning gap

File: `01_same_hard_start_learning_gap.mp4`

This video compares three trajectories from exactly the same initial state:

- A dynamic-programming reference, shown for diagnosis only.
- The seed-0 raw actor from the final pure-RL P7 lineage.
- The seed-0 raw actor from the RL + supervised lineage.

The initial state is θ = 3.038589615767177 radians (174.098°) and angular velocity = −0.25. The final returns exactly reproduce `reports/plan2507_qualitative_trajectory_20260725.json`:

- Diagnostic reference: −280.8962277
- Pure-RL raw actor: −384.2624192
- RL + supervised actor: −280.5805487

The state was selected post hoc because the pure-RL raw actor fails near-reference while the mixed actor succeeds. It is a qualitative illustration, not an additional aggregate estimate. The learned policies never query the reference.

Checkpoint provenance:

- Pure RL: `runs/plan2307_completion_20260723/pure_target_architecture_matrix/p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k/seed0/checkpoints/final.pt`
- RL + supervised: `runs/systematic_100k_ablation_no_rl_shift_20260722/seed0/checkpoints/final.pt`
- Diagnostic DP solution: `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_solution.npz`

## Secondary video: one Q-search decision repairs a failure

File: `02_pure_rl_qsearch_repairs_failure.mp4`

This video holds the seed-4 P7 checkpoint and initial state fixed. It compares:

- The reflection-averaged actor by itself.
- The final deployed pure-RL policy, which adds a global 41-action Q search with a 0.005 unanimous-critic margin.

The initial state is θ = −3.038589615767177 radians (−174.098°) and angular velocity = −0.9. Q search accepts a different action only at step 13. That single early choice sends the deterministic system onto a different trajectory:

- Reflection only: return −345.1351787, upright-and-slow for 63.5% of steps, longest miss streak 73, task failure.
- Full deployment: return −220.0193201, upright-and-slow for 80.5% of steps, longest miss streak 39, task success.

The replay matches the two authoritative evaluation rows in:

- `reports/plan2507_p7_reflection_authority_20260725/grid/pendulum_grid_rollouts.csv`
- `reports/plan2507_p7_authority_20260725/grid/pendulum_grid_rollouts.csv`

The full deployment tests 41 fixed torques across the environment action range [−2, 2]. It switches only when every learned online critic predicts an improvement greater than 0.005.

This state was selected post hoc from 250 audited grid seed-state cells where reflection alone failed task success and full deployment passed. It is an illustrative case, not an additional aggregate estimate.

Checkpoint provenance:

- `runs/plan2307_completion_20260723/pure_target_architecture_matrix/p7_simba_fastsacn8_lambda0p5_utd2_actorutd1_100k/seed4/checkpoints/final.pt`

## Verification and reproduction

`manifest.json` records media metadata, SHA-256 hashes, exact source rows, checkpoint hashes, deployment rules, and replay errors. The scalar video replay and batched evaluation agree exactly on near-upright fraction, longest miss streak, and task-success status. Return differences are below 0.00013, caused by floating-point accumulation differences between scalar and batched network evaluation.

Both final MP4 files passed a complete frame-by-frame FFmpeg decode check after generation.

Rebuild both videos without training:

```powershell
$env:PYTHONPATH = "src;."
python scripts\build_plan2507_presentation_videos.py
```

Recommended playback order:

1. Use the learning-gap video while explaining what supervision contributes to the actor.
2. Use the Q-search video while explaining why a critic-based deployment rule can change reliability through a single early action.
