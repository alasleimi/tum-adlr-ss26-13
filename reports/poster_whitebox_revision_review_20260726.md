# Poster white-box revision review

Exact reviewed poster before the final wording fixes:

- PDF-only rubric score: 23.5/30
- Template gate: pass
- No clipping, occlusion, or text outside the page
- A0 landscape and minimum-type requirements: pass

Changes made after the review:

- Replaced the causal headline with a composite-pipeline comparison.
- Added the 61 by 41 grid and 200-step rollout protocol.
- Renamed the mixed stages to learner-state DAgger and priority follow-up.
- Defined actor bound occupancy and actor-output sensitivity.
- Removed the undefined dormancy sentence.
- Replaced tiny raster legends with full-size HTML definitions.
- Increased plot titles, ticks, axis labels, and value labels.
- Rephrased the shuffled-prefix result as joint evidence about action order and the induced state trajectory.
- Stated that deployed policies use learned actors and critics without a reference query.
- Removed the small single-start montage from the poster and retained it in the report and selected video.
- Replaced the raster policy-family scorecard with a native full-size table.
- Shortened central diagnostic labels and added export padding.
- Enlarged and simplified the joint-gradient plot labels.

Blind preference results and the subsequent final-candidate hash change are
recorded in `reports/plan2507_final_blind_gate.json`.

## Exact-PDF production gate

The first exact-PDF gate on the rebuilt poster scored 24/30 and failed because
the joint-gradient raster labels were below the template minimum. The reviewer
also requested explicit model identities for the prefix and joint-loss
diagnostics.

The corrected exact PDF then scored 29/30 and passed every blocking gate:

- A0 landscape size and official header: pass
- Minimum type, including rasterized plot labels: pass
- Clipping, occlusion, and safe margins: pass
- Numerical consistency: pass
- Scientific question, methods, evidence, standalone clarity, and production:
  5/5 each

The sole 29/30 comment was that adjacent subplot titles could read as one
phrase. They were replaced with the separated labels `A Norm ratio` and
`B Cosine`. The reviewer then scored the exact-current PDF 30/30, with no
blocking compliance defect, clipping, overlap, minimum-type violation, or
visible production issue.
