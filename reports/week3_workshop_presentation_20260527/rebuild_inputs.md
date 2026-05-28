# Week 3 Workshop Presentation Rebuild Inputs

The compiled deck is `index.html`. To regenerate it, run:

```powershell
python scripts/build_workshop_presentation_20260527.py
```

The generator reads these report artifacts:

- `reports/week3_simbav2_scale_100k_n5_20260527/relative_frontier_n5.csv`
- `reports/week3_reliability_frontier_20260526/reliability_frontier.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/key_posthoc_results_n5.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/key_diagnostic_results_n5.csv`
- `extracted_telemetry.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/sac/relative_rollouts.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/sac/relative_cell_summary.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_rollouts.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/relative_success/simba_full_official_opt/relative_cell_summary.csv`
- `reports/week3_simbav2_scale_100k_n5_20260527/replay_diagnostics/replay_diagnostics_snapshot.csv`
- `reports/pendulum_investigation_20260509/pendulum_dp_100k_reset_support_241x161x81/pendulum_dp_grid.csv`
- `reports/pendulum_investigation_20260509/pendulum_controller_reset_support_61x41/controller_grid.csv`

The GIF can be rebuilt from local checkpoints when available. If checkpoints are absent, the script falls back to an illustrative GIF; the compiled policy-contrast GIF is already committed under `figures/`.
