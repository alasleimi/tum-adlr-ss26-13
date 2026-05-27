# Project 15 Week 3 Workshop Speaker Notes

Format: 8 minutes talk plus 5 minutes questions. Speaker A covers slides 1-5. Speaker B covers slides 6-11. Slides 12-14 are backup.

## Story Arc
1. Pendulum is the controlled testbed: high average return is not enough; we care about the last failing starts.
2. Define success carefully. We first tried an intuitive task-stability definition, found that it was not feasible/certified everywhere, then switched the headline metric to same-start max(DP, controller) within epsilon.
3. Explain SimbaV2 in plain language through the four paper changes, without claiming yet which component drives the gain.
4. Show raw maps first, then seed-level statistics.
5. Diagnose exploration versus optimization using replay and critic-health trackers.
6. Add SAC norm diagnostics as a mechanism check, without treating them as seeded proof.
7. Close with negative results and concrete next steps.

## Key Numbers To Say Correctly
- Known-feasible cells: 2336/2501 = 93.4%. The remaining 165 cells are uncertified, not proven impossible.
- DP has the higher reference return on 2403/2501 cells; the hand controller is higher on 98/2501 cells.
- SAC 100k all-grid task: 75.8% +/- 64.6 pp.
- Full SimbaV2 100k all-grid task: 91.4% +/- 3.9 pp.
- SAC 100k known-feasible task: 80.9% +/- 68.6 pp.
- Full SimbaV2 100k known-feasible task: 97.3% +/- 3.3 pp.
- SAC 100k reference success: 79.2% +/- 57.6 pp.
- Full SimbaV2 100k reference success: 92.5% +/- 5.7 pp.
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
- DP uses the known Pendulum dynamics and reward on a discretized grid; it is an approximate reference, not a proof of optimality.
- The hand controller is energy shaping plus local PD, following Astrom-Furuta style swing-up; it is a witness and return reference, not an oracle.
- SAC 100k seed0 is a real bad seed: 52.4% reference success versus 93.0%/92.2% for seeds 1/2, with replay coverage tied.
- Seed is the statistical unit. Cell-level pooling is for maps; seed-level intervals are used for claims.

GIF status: Policy GIF: exact-grid contrast where SAC seed0 fails and full SimbaV2 seed0 succeeds (theta=-174.1 deg, theta_dot=-1.00, return gap=+18.2).
