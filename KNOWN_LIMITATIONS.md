# Provenance and interpretation notes

The final ShareLaTeX report is preserved byte-for-byte. The following audit
notes are recorded separately so that historical results remain reproducible
without silently strengthening their interpretation.

1. **Mixed seed lineage is heterogeneous.** Seed 0 descends from
   `reference_assisted_large_simba_reset60_dagger3_seed0_20260717`; seeds 1-4
   descend from `general_hybrid_frozen_initializer_5seed_20260718`. All inherit
   the recovered angle-targeted initializer and its nonuniform state sampling.
2. **The row described as "No learner-state labels" is more precisely "No 20k
   high-regret follow-up."** Its base actors already contain three learner-state
   DAgger rounds, totaling 30k learner-labeled transitions.
3. **The uniform-start mixed ablation is not a single-factor priority
   comparison.** Its target-shift setting also differs. A shifted-priority
   comparator is retained to make this visible.
4. **Figure 58 has a selector/wording mismatch.** Values 73.09%, 74.84%, and
   76.21% are gradient-sign agreement, while realized beneficial-step rates are
   approximately 72.23%, 73.36%, and 75.39%. The verifier reports both instead
   of relabeling either series.

These notes do not invalidate the stored rollout counts. They narrow which
causal statements can be made from the ablation labels and diagnostics.
