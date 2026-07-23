# Systematic <=100k Pendulum recipe audit

Date: 2026-07-22

## Decision

Under the stored learning-transition ledger, the best eligible **RL + supervised** result is a tie between the one-round automatic-priority DAgger recipes with and without tiny RL target shifts. Both score **12,496/12,505 = 99.928029% near reference**, **11,737/12,505 = 93.858457% task success**, and **1,570/12,505 = 12.554978% literal strict wins**. The shifted version has only a `+0.001830` mean-return edge; because none of the three requested classifications changes, the no-shift version is the scientifically simpler recipe.

The best eligible **pure RL** recipe remains clean 100k SimbaV2 plus reflection-averaged actor fallback and 41-action unanimous Q-search. It scores **11,832/12,505 = 94.618153% near reference**, **11,567/12,505 = 92.499000% task success**, and **2,303/12,505 = 18.416633% literal strict wins**.

Both rules are uniform over all states. Neither uses an initial-state router, a hand-written inference range, a reference query at inference, or multiple actors. Q-search is learned-critic action selection, which is explicitly allowed.

## Inventory scope

The TOP5 tables below now come from a repository crawl rather than the old six/seven-row shortlist. The crawler found **67 matching summary files** and **59 unique standardized five-seed evaluations** after row-table deduplication. **42** have a recoverable <=100k executed-learning lineage and satisfy the deployment rule. Full classification, dependency evidence, and every exclusion are under `programmatic_inventory/`; the former shortlist is retained only as `legacy_*_shortlist.csv` for reproducibility of the old report.

Manual coordinate ranges used only to collect training data are no longer an exclusion. They are reported in the `manual_training_region` column. Hardcoded inference thresholds, reference calls at inference, and multi-model specialist selection remain separate deployment exclusions. Legacy aggregate-only results without pooled rollout rows are preserved but unranked rather than silently discarded.

## Budget definition

There are two honest ledgers because automatic-priority discovery itself rolls the actor through the simulator:

| Ledger | Automatic-priority hybrid | Uniform-start hybrid |
| --- | ---: | ---: |
| Learning transitions inserted into the learned components | 30k initializer DAgger + 20k new DAgger + 50k critic = **100k** | 30k + 20k + 50k = **100k** |
| Learning transitions plus automatic candidate rollouts used to choose DAgger starts | 100k + (4,000 candidates x 200 steps) = **900k** | **100k** |

The main leaderboard uses the first, conventional learning-transition ledger. If `<=100k` means every simulator transition that influences training-data acquisition, the automatic-priority rows are not eligible and the uniform-start DAgger plus local Q-search row is the hybrid leader at **12,486/12,505** near reference. `component_interaction_budget.csv` and `component_plus_discovery_budget.csv` preserve both definitions.

Static sampled-state labels, optimizer updates, validation rollouts, and reference-oracle computations are excluded from both ledgers. Therefore the second ledger is specifically a component-plus-priority-discovery audit, not a claim to count every development or validation simulator call. Pure RL uses 100k reward-only environment steps per seed and has no automatic-priority discovery term.

## Current eligible RL + supervised top five

| Method | Near | Task | Strict | Mean return |
| --- | ---: | ---: | ---: | ---: |
| 100k-total automatic-priority RL-shifted DAgger + FastSACN local Q-search | 12496 | 11737 | 1570 | -138.647924 |
| Ablation: no_rl_shift + FastSACN local Q-search | 12496 | 11737 | 1570 | -138.649753 |
| 100k-total automatic-priority DAgger + reflection actor + FastSACN local Q-search | 12492 | 11747 | 1617 | -139.271479 |
| Ablation: uniform_dagger + FastSACN local Q-search | 12486 | 11735 | 1459 | -138.738541 |
| Ablation: final hybrid actor without inference Q-search | 12470 | 11705 | 1809 | -138.791249 |

## Current eligible pure-RL top five

| Method | Near | Task | Strict | Mean return |
| --- | ---: | ---: | ---: | ---: |
| SimbaV2 100k + reflection actor + 41-action unanimous Q-search (margin 0.005) | 11832 | 11567 | 2303 | -140.620617 |
| Clean SimbaV2 100k + frozen unanimous Q-search n41 margin0.005 | 11739 | 11540 | 1106 | -140.731425 |
| Pure RL SimbaV2 100k + q41 margin .005 mid c=.25 unanimous | 11737 | 11539 | 1111 | -140.709499 |
| Clean SimbaV2 100k + mean-proposal unanimous Q-search n41 margin0.005 | 11728 | 11534 | 1104 | -140.733257 |
| Clean SimbaV2 100k + joint online-target unanimous Q-search n41 margin0.005 | 11715 | 11561 | 1111 | -140.735329 |

## Historical audit exclusions

The highest historical rows were not silently compared as eligible. G6C and the gain-calibrated actors exceed the interaction budget; the overlay/gate families deploy several actors or a state router. Exact exclusions are in `excluded_historical_leaders.csv`.

### What was wrong with the historical 99.96% rows

Two different historical recipes scored 12,500/12,505. Neither hard-codes an inference range or uses a state router, so their problem is **budget**, not the user's definition of cheating.

- **Gain-calibrated targeted RL-weighted DAgger:** the selected deployed lineage contains 30k initializer transitions, the selected 20k corrected-DAgger round, and 200k targeted DAgger transitions, for 250k actor transitions. Its separately trained 50k critic influenced training weights, making the selected pipeline **300k**. Counting every corrected-DAgger round that was actually executed makes the experimental component total **380k**. Its hard-coded coordinate neighborhoods occur during training; inference is one actor with one globally fixed gain of 1.005.
- **G6C ten-round priority DAgger plus local Q-search:** the full recipe executes 30k initializer plus 200k stage-two actor transitions and a separate 50k critic, or **280k** before counting priority discovery. The selected actor/critic checkpoint lineages across seeds are **260k, 100k, 120k, 220k, and 120k**. Ten rounds of 4,000 candidate rollouts add **8,000,000** priority-discovery simulator steps per actor seed. Its inference rule is nevertheless one uniform local Q-search with no range router.

The exact machine-readable accounting is in `historical_99p96_budget_ledger.csv`.

## Hybrid recipe

Per seed, the actor starts from an independently trained width-64, two-block SimbaV2 supervised initializer with 400k broad/reset-support static labels and 30k learner-only DAgger transitions. The new stage samples 240k states uniformly over reset support, automatically draws 4,000 candidate initial states, ranks them by reference regret, chooses 90 high-regret plus 10 uniform starts, collects 20k learner-only DAgger transitions, and trains for three epochs. A clean 50k FastSACN8/UTD2 critic makes only tiny training target shifts (maximum 0.02), then supplies a uniform five-action local Q-search within +/-0.10 at inference with unanimous positive advantage.

## Component attribution

| Comparison | Near fixed | Near broken | Near net | Task net | Strict net | Mean return delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k automatic-priority training stage | 30 | 1 | 29 | 7 | 442 | 0.133650 |
| local FastSACN Q-search at inference | 26 | 0 | 26 | 32 | -239 | 0.143325 |
| automatic priority versus uniform DAgger starts | 10 | 0 | 10 | 2 | 111 | 0.090617 |
| tiny RL target shifts versus no shift | 0 | 0 | 0 | 0 | 0 | 0.001830 |
| pure-RL reflection fallback added to global Q-search | 126 | 33 | 93 | 27 | 1197 | 0.110808 |
| pure-RL global unanimous Q-search added to actor | 374 | 119 | 255 | 103 | 40 | 0.907132 |

- **Automatic priority is causal and useful.** Versus ordinary reset-start DAgger, it nets +10 near-reference trials, +2 task trials, +111 strict wins, and +0.090617 mean return. It discovers failures using rollout regret rather than coordinate bands.
- **The 20k supervised update is useful.** Relative to the inherited initializer with the same Q-search, it nets +29 near-reference trials, +7 task trials, +442 strict wins, and +0.133650 mean return.
- **Local Q-search is useful for the primary and task metrics but not strict wins.** It nets +26 near-reference and +32 task trials, while losing 239 strict wins. It is a conservative reliability correction, not a return-dominance optimizer.
- **Tiny RL training target shifts are not supported.** Removing them changes zero near/task/strict classifications; the mean-return effect is only +0.001830 for the shifted recipe. The simpler no-shift actor is scientifically preferable unless mean return is the tie-breaker.
- **Reflection is strongly beneficial for pure RL but not portable without qualification.** On pure RL it nets +93 near, +27 task, and +1,197 strict trials over Q-search alone. On the supervised hybrid it lost four near trials and worsened mean return, despite adding task/strict wins, so it was rejected.
- **FastSACN is useful as a short-budget critic.** The clean 50k FastSACN critic enables the hybrid's local search and has the best task count among the audited pure-RL Q-search candidates, but it ranks sixth on the primary near-reference metric; replacing the pure winner's 100k Simba checkpoints with 50k FastSACN plus reflection failed the reference-free tail/task gate.

## Heat-map diagnosis

The final hybrid has nine near-reference failures, all preserved in `hybrid_remaining_near_failures.csv`. Region counts for both winners are in `failure_region_diagnostics.csv`. Hybrid near-reference errors are isolated boundary failures rather than a broad missing basin; task failures remain concentrated near the downward/wrap boundary. Pure-RL residual near-reference failures are much broader and overwhelmingly occupy `abs(theta) >= 120 degrees`, explaining why Q-search and symmetry improve the frontier but do not close it.

![Hybrid near-reference heatmap](hybrid_qsearch/relative/near_best_known_return_eps_map.png)

![Hybrid task-success heatmap](hybrid_qsearch/grid/task_success_rate_map.png)

![Pure-RL near-reference heatmap](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/near_best_known_return_eps_map.png)

![Pure-RL task-success heatmap](../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/task_success_map.png)

## Scientific limits

The 61x41 grid is deterministic and highly correlated across neighboring states, so pooled Wilson intervals or trial-level p-values are not evidence for population generalization. The defensible evidence is five independently trained actor seeds, paired state-by-seed comparisons, disjoint midpoint/continuous validation before authoritative evaluation, and explicit reporting of every remaining failure. The shared hybrid critic is a globally selected component, so a future confirmation should retrain or pre-register critic selection on new critic seeds rather than treating these five actor seeds as five independent full-pipeline replications.

## Reproduction artifacts

- Hybrid checkpoints: `runs/systematic_100k_budget_g6c_round1_qsearch_20260722/seed0` through `seed4`.
- Hybrid authoritative result: `hybrid_qsearch/authoritative_summary.json`.
- Pure-RL authoritative result: `../pure_rl_plus1pp_20260719/authority_simba100k_symmetric_actor_q41m005_unanimous_relative/relative_summary.json`.
- Audited eligible-candidate inventories used to derive the top five: `hybrid_inventory.csv`, `pure_rl_inventory.csv`.
- Sorted top-five leaderboards: `hybrid_top5.csv`, `pure_rl_top5.csv`.
- Step accounting: `component_interaction_budget.csv`, `component_plus_discovery_budget.csv`, and `historical_99p96_budget_ledger.csv`.
- Paired diagnostics: `paired_ablation_diagnostics.csv`.
- Failure diagnostics: `failure_region_diagnostics.csv` and `hybrid_remaining_near_failures.csv`.
