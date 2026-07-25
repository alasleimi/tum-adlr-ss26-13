# Chasing the Nines: final delivery

This directory is the clean midnight delivery package for Project 15.

## Primary artifacts

- `Chasing_the_Nines_Report.pdf`: four pages of main scientific content, followed by references and a diagnostic annex
- `Chasing_the_Nines_A0_Poster.pdf`: print-ready landscape A0 poster
- `Chasing_the_Nines_A0_Poster.png`: raster preview of the poster
- `plan2307.md`: completed execution and acceptance plan
- `verified_scorecard.csv`: audited seven-method scorecard
- `evidence_registry.csv`: claim-level evidence and limitation ledger
- `final_validation_20260725.json`: machine-readable terminal validation record

## Accepted mainline results

The accepted RL plus supervised route records 12,496/12,505 near-reference trials, 99.928%, with 11,737 task successes and 1,570 strict reference wins. Its deployed actor and Q-search use no angle router and do not query the reference. The initializer's disclosed near-upright oversampling stays as a scope footnote.

The accepted pure-RL route records 11,832/12,505 near-reference trials, 94.618%, with 11,567 task successes and 2,303 strict reference wins. It uses five 100k SimbaV2 SAC actors, reflection, and unanimous 41-action Q-search.

## Terminal FastSACN8 result

The five-seed 100k FastSACN8 plus critic-UTD2 combination improves locked off-grid near-reference reliability from 94.489% to 95.909% and strict wins from 17.792% to 21.930%, but loses 25 of 27,765 task-success trials. It fails the registered no-regression gate, so the authority grid was not queried and the established pure-RL mainline is unchanged.

The ordinary FastSACN8-combination actor is less reliable than the one-step actor, while the identical critic-guided deployment is more reliable. Actor saturation increases and action sensitivity decreases in all five paired seeds. This supports stronger critic repair coupled to weaker raw-actor geometry, without isolating the temporal target from critic update dose.

The actor-UTD2 P8 seed-zero screen also fails its promotion gate and receives no four-seed extension.

## Acceptance record

- Terminal validator: PASS
- Report blind comparison: 28.5/30 versus 27.0
- Poster versus NEED: 28/30 versus 16/30
- Poster versus 3DPRAC: 26/30 versus 14/30
- Report main content: exactly four pages
- Poster: one landscape A0 page with embedded fonts and no detected overflow

The `source` directory contains portable report and poster sources with their figures. The `terminal` directory contains locked evaluation JSON files and the promotion decision. The `reviews` directory contains all six sequential reviews and the three terminal blind verdicts. The `reproducibility` directory contains the evaluation, diagnostic, rendering, and validation scripts used for the terminal package.
