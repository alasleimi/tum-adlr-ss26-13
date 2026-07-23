# Complete two-worktree result inventory

Date: 2026-07-23

Repositories:

- `C:\Users\Ala\Desktop\Project 15`
- `C:\Users\Ala\Desktop\Project 15-sac-n-experiments`

## Executive conclusion

The neighboring SAC-N folder does not contain a unique experiment or a hidden winner. It is an older worktree whose result-bearing files are all present in Project 15. Within the audited result scope, 567 neighboring files are exact same-path mirrors, 30 are older or revised same-path files, and zero are neighbor-only.

The old five-seed inventory was materially incomplete. It ignored 347 of the 409 primary relative-grid summaries because they had fewer than five actor seeds. It also omitted custom trainers that use `run_manifest.json` and `training_summary.json`, copied checkpoint aliases, artifact-only runs, UTF-16/BOM report tables, and several alternative lineage fields.

The final census contains:

| Unit | Count | Meaning |
|---|---:|---|
| Candidate artifacts | 19,650 | Result, run, config, script, and documentation artifacts across both worktrees |
| Run-directory instances | 754 | 729 in Project 15 and 25 in the historical worktree |
| Physical training executions | 732 | Excludes 22 copied-checkpoint or SWA aliases |
| Primary completed physical executions | 630 | Completion supported by checkpoint and progress evidence |
| Primary incomplete or unresolved executions | 77 | Includes killed probes, partial screens, and one artifact-only unresolved run |
| Primary normalized run families | 438 | Path-normalized execution families |
| Literal config-parent groups | 451 | 29 have five configured actor seeds; 422 have fewer than five |
| Standardized evaluation wrappers | 422 | Relative summaries, authority wrappers, and worktree mirrors |
| Distinct standardized evaluations | 402 | Deduplicated by exact scored rollout-table identity |
| Five-seed standardized evaluations | 60 | Only 14.9% of the distinct standardized corpus |
| Fewer-than-five-seed standardized evaluations | 342 | 85.1% of the distinct standardized corpus |
| Result-bearing JSON files | 442 | 2,181 metric-bearing JSON nodes |
| Aggregate/report-only CSV rows | 843 | 778 after cross-worktree mirror reconciliation |

No scored heat-map result is missing. Forty-seven summaries contain stale paths to deleted temporary raw-grid folders, but every stable `relative_rollouts.csv` and every sibling grid needed to recover them is present.

## Rules used

The classification follows the requested strict definition.

A result is marked cheating if any deployed learned component's lineage uses:

1. a manually fixed angle or velocity region for training-data selection, resets, replay, model rollouts, anchors, shaping, or specialists;
2. a hardcoded state-range router at inference;
3. the reference policy at inference.

The following are allowed, but recorded separately:

- learned-critic Q-search;
- automatic reward-based failure mining;
- automatically learned regions or gates;
- global reflection symmetry;
- reference supervision during training;
- model mixtures.

Reference supervision makes a result supervised-only or RL plus supervised. It does not by itself make the result cheating. Reference-derived DP/controller potential shaping is also reference-assisted training, so it is not pure RL even if the shaping is global and non-cheating.

Pure RL requires reward-only training through the recursive actor and inference-critic lineage. Reference use after rollout solely to compute the evaluation score does not contaminate training or inference.

`<5 seeds` means fewer than five independently trained actor pipelines. Evaluation RNG seeds do not count. A result with five actors but one shared inference critic is reported as five actor seeds with a shared critic, not five independent critic replications.

## Evaluation census

Project 15 contains exactly 409 primary `relative_summary.json` files. Every one covers 2,501 initial-condition cells with epsilon 5.

| Actor seeds in stored evaluation | Summary count |
|---:|---:|
| 1 | 315 |
| 2 | 10 |
| 3 | 15 |
| 4 | 7 |
| 5 | 62 |

The 409 row tables reduce to 402 exact byte identities. The duplicates are report copies, authority wrappers, copied checkpoint views, or renamed specialist reports with identical root settings and outcomes. The six authority summaries and seven historical-worktree summaries are retained as aliases rather than counted as new experiments.

![Evaluations by seed count and legality](figures/01_evaluations_by_seed_and_legality.png)

Across the 402 distinct standardized evaluations:

| Classification | Count |
|---|---:|
| Pure RL | 255 |
| RL plus supervised | 133 |
| Supervised only | 14 |
| Clean under the strict rule | 161 |
| Cheating under the strict rule | 237 |
| Legality unresolved | 4 |

The four unresolved results are two reference-assisted RL overlays and two specialist Q-filter reports with no recoverable run metadata. They remain unknown rather than being silently labeled clean.

![All standardized results](figures/02_all_standardized_results_scatter.png)

## Best clean, non-mixture, at-most-100k results

This table requires at least five actor seeds, no manually fixed training range, no hardcoded range router, no reference at inference, no model mixture, and at most 100k learning-transition lineage. Q-search and global reflection are allowed.

| Category and role | Method | Steps | Near reference | Task success | Strictly beats reference |
|---|---|---:|---:|---:|---:|
| RL plus supervised, best near | Automatic-priority hybrid plus FastSACN local Q-search | 100k | 99.928% | 93.858% | 12.555% |
| RL plus supervised, best task among 100k finalists | Symmetric actor hybrid plus FastSACN local Q-search | 100k | 99.896% | 93.938% | 12.931% |
| RL plus supervised, under-budget ablation | Supervised initializer plus FastSACN critic Q-search | 80k | 99.696% | 93.802% | 9.020% |
| Supervised only, strong under-budget result | Reference DAgger specialist extension | 20k rollout transitions | 99.336% | 94.042% | 10.556% |
| Pure RL, best near and strong strict | SimbaV2 100k plus reflection and unanimous Q-search | 100k | 94.618% | 92.499% | 18.417% |
| Pure RL, best task | FastSACN8 UTD2 50k plus unanimous Q-search | 50k | 93.219% | 92.915% | 7.237% |
| Pure RL, best strict | SimbaV2 plus density SACn16 | 100k | 90.428% | 89.804% | 18.880% |
| Pure RL baseline | Full SimbaV2 | 100k | 91.835% | 91.459% | 8.525% |

The reference DAgger line has only 20k rollout transitions, but it also uses a large static labelled dataset. It is a valid strong result, not evidence that its total information cost is only 20k.

Clean canonical DAgger at 100k is much weaker: 84.534% near reference, 88.021% task, and 6.389% strict beats. The paper-style plain MLP DAgger reaches 83.671%, 86.038%, and 10.020%. Its SimbaV2-backbone counterpart reaches 83.343%, 87.285%, and 6.285%. A SimbaV2 backbone therefore helps task success in that pair, but it is not sufficient to produce the modern hybrid performance.

## The FastSACN question in pure RL

The best pure-RL answer depends on the target metric.

- FastSACN8 UTD2 at 50k plus allowed Q-search is the five-seed pure-RL task leader at 92.915%.
- SimbaV2 100k plus global reflection and Q-search is the pure-RL near-reference leader at 94.618% and nearly the strict leader at 18.417%.
- Density SACn16 at 100k has the highest strict-beats rate, 18.880%, but loses 1.407 percentage points of near-reference success and 1.655 points of task success versus plain Full SimbaV2.
- Plain Full SimbaV2 100k is 91.835% near and 91.459% task.

Thus the result is not simply "plain SimbaV2 plus inference tricks." The task-leading trained actor is FastSACN. The best near-reference deployment instead uses the older 100k SimbaV2 actor with allowed global symmetry and critic Q-search.

We did try a five-seed FastSACN 100k family at UTD1. It reaches 88.421% near, 90.852% task, and 11.323% strict, so continuing that exact recipe was not an improvement. A clean FastSACN8 UTD2 100k retry completed only for seed 0. Its off-grid task result is promising, but seeds 1 through 4 were never trained. That missing five-seed UTD2 test is the main unresolved pure-RL promotion.

## Why the 99.96% headlines do not all qualify

There are three different 99.96% stories:

| Result | Seeds | Steps | Status | Reason |
|---|---:|---:|---|---|
| h1 joint one-step plus local Q-search | 1 | 35k | Cheating | Recursive lineage uses reference anchors selected by a fixed velocity bound |
| Gain-1.005 large-Simba hybrid | 5 | 380k | Cheating and over budget | Manually targeted training states and 380k lineage |
| General G6C hybrid | 5 | 280k | Clean but over budget | No inference reference or hardcoded router, but 280k lineage |

The best clean five-seed result that actually satisfies the 100k cap is 99.928%, not 99.96%.

## Promising ideas at the wrong budget or replication count

The curated machine-readable list is in [`curated_promising_ideas.csv`](curated_promising_ideas.csv). The most actionable items are:

1. **FastSACN8 UTD2 50k to matched 100k, five seeds.** The current five-seed task leader is already strong at 50k. Only one UTD2 100k seed exists. This is the highest-priority pure-RL completion.

2. **Supervised initializer plus FastSACN critic Q-search at 80k.** It already reaches 99.696% near and 93.802% task with five seeds. The next scientific question is allocation, not merely adding 20k: actor supervision dose, critic steps, and Q-search need matched controls.

3. **Reference DAgger specialist extension at 20k rollout steps.** The five-seed result is 99.336% near and 94.042% task. It needs fresh-seed replication and explicit static-label/query accounting.

4. **Static-240k distillation plus 20k DAgger.** The three-seed result is 99.280% near and 94.122% task. Seeds 3 and 4 are missing.

5. **Legacy SAC 250k UTD2.** Versus legacy 100k, it gains 6.957 points near, 2.983 task, and 5.222 strict. It is over budget, but it argues for a capped higher-UTD experiment.

6. **Scalar-critic SimbaV2 UTD2 at 50k.** Against its matched UTD1 scalar control, near-reference improves by 8.477 points, task falls by 0.280, and strict beats falls by 4.412. This is a near-reference signal, not an across-metric win. The untested transfer is full distributional SimbaV2 UTD2 at 100k.

7. **Density SACn16.** Its +10.356-point strict gain versus Full SimbaV2 is real and five-seed, but primary metrics fall. It is a component signal rather than the best complete recipe.

8. **No-importance SACn8 stopped after 10k.** Seed 0 looked promising at 93.962% near and 92.043% task. The preserved three-seed aggregate falls to 87.232% and 89.298%. It should no longer be promoted from the seed-0 headline.

![Stored family budgets](figures/03_run_family_budget_distribution.png)

## What looked lost, and what actually happened

### Recovered or reconciled

- All 47 stale `.tmp-*` raw-grid links in the joint follow-up reports resolve to stable sibling grids. The scored relative tables were never lost.
- Fourteen custom training runs with `run_manifest.json` and `training_summary.json` but no `config.json` are now indexed.
- Nine artifact-only run directories are now indexed. Eight have enough event/checkpoint evidence to establish completion; one late-environment-reset run remains unresolved at 20.2k with no checkpoint.
- Twenty-two `checkpoint_eval_aliases` directories are not new executions. Seventeen are byte-exact intermediate checkpoint copies. Five are SWA actors over 30k, 40k, and 50k checkpoints with inferred 50k lineage and missing creation manifests.
- The five-seed Week3 SAC and Full SimbaV2 results stitch seeds 0 through 2 from one run root and seeds 3 through 4 from another. They are one five-seed result each, not two recipes.
- `runs/norms_checking` and `runs/week1_pendulum_sac/20260516_141143_seed0` contain the same exact model. They are aliases with different diagnostic telemetry.
- The stale document saying the 250k UTD2 run was partial is wrong. All five seeds completed and the five-seed grid is present.

### Incomplete or genuinely absent

- The primary ledger has 77 incomplete or unresolved physical executions.
- Seven named or configured five-seed families are incomplete.
- The historical `partial_100k_seed0_aborted` run reached about 18k metrics and 15k evaluation steps. No checkpoint survived.
- The original no-importance SACn8 seed-3 run stopped around 3.6k, a separate seed-3 retry reached 50k, and seed 4 is absent.
- There are 324 completed checkpoints with no standardized 61x41 evaluation. Many are smokes or screened-out arms, but the table makes every one explicit rather than dropping it.
- Twelve trainer `metrics.csv` files are empty or NUL-corrupted. Ten additional online-RL-style directories lack the standard `metrics.csv`, mostly because custom trainers wrote alternative metric files. Events, summaries, and checkpoints preserve partial evidence.
- A 205-row queue ledger is mostly a plan, not experiment evidence. At its snapshot, 145 fresh pure-RL definitions had no execution. Planned rows are not counted as results.

The detailed exception ledger is [`orphan_incomplete_and_missing_eval_audit.csv`](orphan_incomplete_and_missing_eval_audit.csv).

## Worktree reconciliation

The SAC-N worktree is historical provenance, not a second experiment store.

- All 419 neighboring report files exist at the same relative path in Project 15.
- All 152 neighboring run files are exact Project 15 subsets.
- Four JSON paths and 37 result-index paths missing inside the neighbor all resolve in Project 15.
- No checkpoint, result table, or learned model is unique to the neighbor.

The complete same-path disposition is [`repo_reconciliation.csv`](repo_reconciliation.csv).

## Machine-readable inventory

| File | Purpose |
|---|---|
| [`coverage_manifest.json`](coverage_manifest.json) | Snapshot timestamps, policy, counts, and output list |
| [`coverage_checks.csv`](coverage_checks.csv) | Reconciliation assertions; all final checks pass |
| [`artifact_manifest.csv`](artifact_manifest.csv) | Every candidate physical artifact, role, size, timestamp, hash where practical, and worktree relation |
| [`repo_reconciliation.csv`](repo_reconciliation.csv) | Same-path mirror, revision, primary-only, or neighbor-only status |
| [`run_instances.csv`](run_instances.csv) | Every run directory with seed, requested and observed steps, completion, lineage, legality, category, local evaluation, and file integrity |
| [`run_families.csv`](run_families.csv) | Normalized family seed coverage, completion, evaluation links, and feature tags |
| [`config_parent_groups.csv`](config_parent_groups.csv) | Literal config-parent seed groups and `<5 seeds` flag |
| [`configuration_signature_inventory.csv`](configuration_signature_inventory.csv) | Normalized config/manifest signatures, with conflation warnings when one signature spans several families |
| [`checkpoint_alias_audit.csv`](checkpoint_alias_audit.csv) | Exact copied checkpoints, inferred SWA constituents, and inherited lineage steps |
| [`evaluation_aliases.csv`](evaluation_aliases.csv) | All 422 standardized summary wrappers and their canonical result IDs |
| [`unique_standardized_evaluations.csv`](unique_standardized_evaluations.csv) | All 402 deduplicated heat-map results with metrics, seeds, steps, category, legality, deployment, and lineage |
| [`metric_json_inventory.csv`](metric_json_inventory.csv) | Schema-discovered JSON metric nodes, including noncanonical diagnostics |
| [`tabular_metric_rows.csv`](tabular_metric_rows.csv) | Aggregate/report-only result rows with cross-worktree alias IDs |
| [`promising_wrong_budget_or_underreplicated.csv`](promising_wrong_budget_or_underreplicated.csv) | Broad programmatic candidate screen |
| [`curated_promising_ideas.csv`](curated_promising_ideas.csv) | Scientifically reviewed shortlist with matched evidence and next action |
| [`orphan_incomplete_and_missing_eval_audit.csv`](orphan_incomplete_and_missing_eval_audit.csv) | Incomplete runs, missing evaluations, corrupted metrics, and unfinished five-seed families |

The crawler is [`scripts/inventory_all_results_two_repos_20260723.py`](../../scripts/inventory_all_results_two_repos_20260723.py).

## Reproducibility and limitation

The final validated scan ran from 08:43:59 to 08:45:14 UTC on 2026-07-23. No audited file changed during that final scan, and all coverage checks passed. A separate training process remained active after the scan, so this is a timestamped complete snapshot, not a promise that the workspace will never acquire another result. Rerun the crawler after the active campaign ends to append later artifacts.

Noncanonical JSON and aggregate CSV classifications are explicitly marked when they rely on schema/name heuristics. The authoritative cheating, purity, seed, and metric conclusions above use standardized results with resolved run lineage. Equality of metric triples alone is never used for deduplication.
