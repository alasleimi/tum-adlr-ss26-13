# Chasing the Nines deliverables

This directory is the final delivery package for the 23 July 2026 production plan. It tells a reproducible story about improving deep reinforcement-learning reliability across both initial states and independent actor seeds.

## Final artifacts

- Report PDF: `report/report.pdf`
- Report source: `report/report.tex`
- A0 landscape poster PDF: `poster/poster.pdf`
- Poster source: `poster/poster.html`
- Full-resolution poster preview: `poster/poster_full.png`
- Verified scorecard: `verified_scorecard.csv`
- Verified aggregate metrics: `verified_metrics.json`
- Verified cell consistency counts: `verified_cell_consistency.csv`
- Claim and evidence registry: `evidence_registry.csv`
- Joint-loss pilot summary: `joint_loss_pilot_summary.json`
- Automated delivery validation: `final_validation.json`
- Final figures: `figures/`

The report has nine A4 pages and more than four full pages of main content. The poster is one true A0 landscape page. All fonts are embedded in both PDFs.

## Headline result

Under the registered clean evaluation rules, the selected mixed RL and supervised pipeline obtains 12,496 near-reference successes from 12,505 seed-state trials, or 99.928%. The selected pure-RL pipeline obtains 11,832, or 94.618%. The mixed pipeline uses more total data and simulator interactions, so this is a comparison of selected completed pipelines rather than a causal estimate of the value of supervision.

Neither pipeline uses hardcoded angle ranges, reference access at deployment, or checkpoint mixtures. Critic-guided action search is permitted because it uses only learned reward-based value estimates.

## Evidence hierarchy

The standardized identity ledger is:

`reports/two_repo_forensic_inventory_20260723/unique_standardized_evaluations.csv`

It contains 402 unique standardized evaluations from Project 15 and the neighboring SAC-n experiment repository. Every central report and poster claim is linked to recoverable evidence in `evidence_registry.csv`.

- Tier A claims are regenerated from raw five-seed standardized rollout tables or paired seed-state tables.
- Tier B claims use recoverable multi-seed diagnostic artifacts with targeted sampling that differs from the main grid.
- Tier C claims are exploratory one-seed or short-screen evidence and are labeled as hypotheses or pilot results.

Regenerate the final scorecard, consistency tables, and figures with:

`python scripts/build_final_delivery_figures_20260723.py`

Regenerate the joint-loss gradient diagnostic and seed-0 comparison with:

`python scripts/build_joint_loss_pilot_diagnostic_20260723.py`

## Evaluation definitions

The authoritative benchmark evaluates five actor seeds on 2,501 initial states each, for 12,505 seed-state trials.

- Near reference: the policy return is within 5 return units of the stored reference.
- Task success: return is at least -150.
- Strict win: the policy return is strictly greater than the reference return.

The reference is loaded only for offline scoring after each policy has been frozen.

## Review record

The package includes at least three independent, fresh-context report reviews and three equivalent poster reviews in `reviews/`. Each reviewer saw only the artifact under review and its grading instructions, without old feedback or source files.

The final poster also passed the registered blind stopping test on the same final version:

- Against `Geometry-Supervised_3DGS_A0_Poster_Final.pdf`, the counterbalanced anonymous judge preferred this poster 94 to 91. See `blind_tests/geometry_round6/verdict.md`.
- Against `NEED-A0-print-poster.pdf`, the anonymous judge preferred this poster 95 to 89. See `blind_tests/need_round5/verdict.md`.

Earlier blind rounds, including losses, are retained to make the judge sensitivity and iteration history auditable.

The validation script checks page sizes, font embedding, style constraints, claim counts, review counts, and pixel identity between the current poster and both final winning poster copies:

`python scripts/validate_final_delivery_20260723.py`

## Scope of the completed joint-loss pilot

The completed simultaneous behavior-cloning and SAC experiment is a 25,000-step, seed-0 diagnostic pilot. Its joint actor obtains 2,443 of 2,501 near-reference successes, or 97.681%, compared with 1,757, or 70.252%, for its matched reward-only control. The median weighted behavior-cloning gradient norm is 37.19 times the weighted SAC gradient norm. Individual minibatch gradient cosines range from -0.943 to 0.935, and no logged layer is dormant at the registered threshold.

This result does not establish that simultaneous training is better than behavior cloning alone or staged training. The missing controls were not completed. The pilot also uses static reference anchors and equally weighted SACn8 targets, so the full learner-state DAgger plus properly weighted FastSACN8 experiment is still required.
