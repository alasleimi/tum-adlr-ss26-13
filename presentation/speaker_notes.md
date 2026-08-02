# Project 15 Week 3 Workshop Speaker Notes

Format: 8 minutes talk plus 5 minutes questions. Speaker A covers slides 1-5. Speaker B covers slides 6-11. Slides 12-18 are backup.

## Story Arc
1. Pendulum is the controlled testbed: high average return is not enough; we care about the last failing starts.
2. Define success carefully. We first tried an intuitive task-stability definition, found that it was not feasible/certified everywhere, then switched the headline metric to same-start max(DP, controller) within epsilon.
3. Explain SimbaV2 in plain language through the four paper changes, without claiming yet which component drives the gain.
4. Show raw maps first, then seed-level statistics.
5. Diagnose exploration versus optimization using replay and critic-health trackers.
6. Add SAC norm diagnostics as a mechanism check, with component attribution left to the 100k ablations.
7. Close with negative results and concrete next steps.

## Key Numbers To Say Correctly
- Known-feasible cells: 2336/2501 = 93.4%. The remaining 165 cells are uncertified, not proven impossible.
- DP has the higher reference return on 2403/2501 cells; the hand controller is higher on 98/2501 cells.
- Current 100k comparison: 5 training seeds, 12505 exact-grid rollouts per policy.
- Protocol caveat: seeds 3-4 are follow-up runs with 10k-step diagnostics instead of 100k-step diagnostics; training recipe and final evaluation are otherwise the same.
- SAC 100k all-grid task: 81.2% +/- 24.7 pp.
- Full SimbaV2 100k all-grid task: 91.5% +/- 1.7 pp.
- SAC 100k known-feasible task: 86.6% +/- 26.2 pp.
- Full SimbaV2 100k known-feasible task: 97.2% +/- 1.3 pp.
- SAC 100k reference success: 83.0% +/- 21.6 pp.
- Full SimbaV2 100k reference success: 91.8% +/- 2.3 pp.
- Opening-card standard errors: SAC reference +/- 7.8 pp SE, SimbaV2 reference +/- 0.8 pp SE, SAC task +/- 8.9 pp SE, SimbaV2 task +/- 0.6 pp SE.
- Unclipped cell-mean shortfall to max(DP, controller): SAC max 122.4, mean 4.6 +/- 8.2; SimbaV2 max 113.8, mean 3.2 +/- 9.3.
- SAC norm diagnostic: Q parameter norm 38.4 -> 348.5; Q1 fc2 feature norm 606 -> 4856 with peak 5416. This is one diagnostic run, not a seeded claim.

## Slide Timing
- Slide 1: 0:00-0:40. Goal and GIF.
- Slide 2: 0:40-1:35. Definitions and feasibility caveat.
- Slide 3: 1:35-2:15. Evaluation protocol.
- Slide 4: 2:15-3:05. SimbaV2 changes and official-recipe comparison.
- Slide 5: 3:05-3:50. Raw maps.
- Slide 6: 3:50-4:35. Main seed-level result.
- Slide 7: 4:35-5:25. Exploration versus optimization.
- Slide 8: 5:25-5:55. SAC norm diagnostics.
- Slide 9: 5:55-6:30. More compute negative result.
- Slide 10: 6:30-7:10. Hard reset/replay negative result.
- Slide 11: 7:10-8:00. Next steps.

## Q&A Backup
- Pendulum convention: theta is 0 upright and +/-180 degrees downward; observation is [cos(theta), sin(theta), theta_dot].
- Per-seed shortfall maps show return points below max(DP, controller), clipped at 20, one panel per seed.
- The seed-0-excluded raw maps are a sensitivity check; the main result keeps all seeds.
- DP uses the known Pendulum dynamics and reward on a discretized grid; it is an approximate reference, not a proof of optimality.
- The hand controller is energy shaping plus local PD, following Astrom-Furuta style swing-up; it is a witness and return reference, not an oracle.
- SAC 100k per-seed spread: seed 0: 52.4% ref / 45.8% task; seed 1: 93.0% ref / 92.2% task; seed 2: 92.2% ref / 89.3% task; seed 3: 85.0% ref / 87.2% task; seed 4: 92.6% ref / 91.4% task. Replay coverage is tied, so seed 0 is not simply "did not see upright states."
- Seed is the statistical unit. Cell-level pooling is for maps; seed-level intervals are used for claims.

GIF status: Policy GIF: exact-grid contrast where SAC seed0 fails and full SimbaV2 seed0 succeeds (theta=-174.1 deg, theta_dot=-1.00, return gap=+18.2).
