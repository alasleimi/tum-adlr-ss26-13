# Chasing the Nines deliverables

This directory contains the report, A0 poster, figures, evidence registry, and validation artifacts for the project on improving reinforcement-learning reliability across initial states and independent training seeds.

## Current artifact contract

- `report/report.pdf` contains exactly four pages of main scientific content.
- References begins on page 5.
- References and the annex do not count toward the four-page limit.
- `poster/poster.pdf` is one A0 landscape page.
- Every displayed scorecard row uses five trained actor seeds and 12,505 seed-state trials.
- Selected routes use at most 100,000 learning transitions. Discovery interactions and reference queries are reported separately.
- Legal deployment cannot use hardcoded angle ranges, reference access, or actor mixtures. Learned critic-guided action search is allowed.

The final project decision retains the strongest established mixed result in the mainline and discloses its supervised-sampler limitation. Its initializer oversampled near-upright states within 35 degrees, while the subsequent automatic-priority stage selects starts from measured performance. Deployment uses neither an angle rule nor the reference. Earlier frozen variants are retained only as comparison artifacts.

## Current legal leaders

The completed pure-RL leader is SimbaV2 SAC trained for 100,000 transitions with global reflection and unanimous 41-action critic search at deployment. Across 12,505 seed-state trials it records:

- near reference: 11,832, or 94.618%
- task success: 11,567, or 92.499%
- strict wins over the reference: 2,303, or 18.417%
- cells solved by all five actor seeds: 2,255 of 2,501, or 90.164%

The completed mixed leader is a supervised SimbaV2 actor with 30,000 learner-controlled DAgger transitions, a 20,000-transition automatically prioritized follow-up, and a shared 50,000-transition FastSACN8 critic for conservative local Q-search. The reference is absent at deployment. It records:

- near reference: 12,496, or 99.928%
- task success: 11,737, or 93.858%
- strict wins over the reference: 1,570, or 12.555%
- cells solved by all five actor seeds: 2,493 of 2,501, or 99.680%

These are the accepted category leaders. The active matched temporal-target run may update the pure-RL interpretation only after all five seeds and the locked evaluation complete.

## Provenance footnote

The accepted 99.928% mixed lineage uses an initializer with 60% reset-support states, 20% broad full-angle states, and 20% near-upright states within 35 degrees. The final project decision does not disqualify this established supervised result. The sampler is disclosed because it limits the strength of claims about automatic training-state selection. The deployed actor and Q-search use no angle range and do not query the reference.

The stored 99.960% rows also do not qualify. One uses a 260,000-transition selected lineage plus 8,000,000 automatic-discovery interactions. Another uses a 300,000-transition selected lineage and manually specified training regions. The first exceeds the project budget. The second exceeds the budget and violates the no-manual-region rule.

The recovered commands and hashes are recorded in:

- `reports/two_repo_forensic_inventory_20260723/legacy_angle_targeted_initializer_provenance.json`
- `reports/two_repo_forensic_inventory_20260723/balanced_angle_targeted_initializer_provenance.json`

## Forensic inventory

The current pre-final ledger is:

`reports/two_repo_forensic_inventory_20260723/unique_standardized_evaluations.csv`

It contains 404 deduplicated standardized evaluations from Project 15 and the neighboring Project 15-sac-n-experiments repository:

- 255 pure RL
- 135 RL plus supervised
- 14 supervised-only
- 132 clean, 268 disqualified, and 4 with unresolved legality
- 344 with fewer than five seeds
- 34 eligible clean five-seed evaluations at or below 100,000 transitions: 23 pure RL, 7 RL plus supervised, and 4 supervised-only

Training was active during this crawl, so a quiescent terminal refresh is still required.

## Evaluation definitions

The authority benchmark evaluates five independently trained actor seeds on the same 2,501 initial states, for 12,505 trials.

- Near reference means policy return is within 5 return units of the stored reference.
- Task success requires the pendulum to satisfy \(\cos\theta \geq 0.95\) and \(|\dot{\theta}| \leq 1\) for at least 80% of 200 steps, with no not-near-upright streak longer than 50 steps.
- Strict win means policy return is strictly greater than the reference return.

Reference actions and returns may be used during supervised training or frozen diagnostics when declared. They cannot be used at deployment. Diagnostic interventions never participate in model selection.

## Evidence and regeneration

The principal machine-readable artifacts are:

- `verified_scorecard.csv`
- `verified_metrics.json`
- `verified_cell_consistency.csv`
- `evidence_registry.csv`
- `final_validation.json`
- `figures/`

Regenerate report and poster figures with:

`python scripts/build_final_delivery_figures_20260723.py`

Run the final delivery checks with:

`python scripts/validate_midnight_delivery_20260725.py`

The final validator checks the four-page scientific body, References on page 5, font embedding, A0 poster geometry, accepted pooled and seedwise values, evidence hashes, sequential review records, and blind-comparison identities.

## Terminal work status

The report and poster completed three fresh sequential review and revision passes. The report beat the frozen prior report in a blind comparison, and the poster beat both named poster references in separate blind comparisons. The bounded GPU queue is complete.

The five-seed 100k FastSACN8 plus critic-UTD2 combination improves locked off-grid near-reference reliability from 94.489% to 95.909% and strict wins from 17.792% to 21.930%, but loses 25 of 27,765 task-success trials. It therefore fails the registered no-regression gate, and the authority grid was not queried. The accepted 94.618% authority-grid pure-RL result stays in the mainline. The P8 actor-UTD2 seed-zero screen also fails its gate and receives no multi-seed promotion.

The terminal diagnostic shows that the P7 ordinary actor is less reliable while its critic-guided deployment is more reliable. Post-training actor saturation increases and action sensitivity decreases in all five paired seeds. This evidence supports stronger deployment repair coupled to weaker raw-actor geometry, without isolating the target from critic update dose.
