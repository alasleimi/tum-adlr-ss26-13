# Report Review 4

## Scope

This review considers only `report.pdf` and `REPORT_GRADING_TEMPLATE.md`. Page references below use the printed PDF page numbers. The complete nine-page PDF was visually inspected. Because no source files, artifacts, or repositories were consulted, this review assesses internal consistency, scientific presentation, and evidentiary scope rather than independently verifying the implementation or references.

## Score summary

| Category | Score |
|---|---:|
| Effective visual aids | 4.5/5 |
| Spelling and grammar | 5/5 |
| Introduction | 4.5/5 |
| Related work | 4/5 |
| Technical section and results | 4/5 |
| Analysis and discussion | 5/5 |
| **Total** | **27/30** |

## 1. Effective visual aids — 4.5/5

The visuals are unusually effective for a compact report. Figure 1 makes the actor-training/deployment split and the role of the separate critic immediately understandable (p. 3). Figure 2 gives a well-labelled three-metric scorecard and warns in its caption that the mixed rows do not represent five full-pipeline replications (p. 5). Figures 3 and 4 clearly distinguish pooled trial success from cross-seed cell consistency (p. 6). Figures 5–7 communicate distinct operating points, paired classification changes, and the exploratory status of the gradient diagnostic (pp. 7–8). Tables 1–3 have meaningful captions and replication-unit qualifications.

Minor limitations keep this below full credit. Figure 6 uses substantially different horizontal scales in its three panels, so bar lengths should not be compared across metrics without reading the axes and value labels (p. 7). Figure 4 uses a truncated 70–100% horizontal range; this is labelled and defensible for resolving the reliability gap, but the caption could state the truncation explicitly (p. 6). The audited scorecard and artifact-path text are small relative to the rest of the report (p. 9), though still legible.

## 2. Spelling and grammar — 5/5

The prose is clear, controlled, and technically consistent. Terms such as “near-reference success,” “task success,” “strict win,” “selected route,” and “replication unit” are defined and then used carefully (pp. 1–3). Captions generally say what the visual demonstrates without silently upgrading descriptive evidence into causal evidence. Mathematical notation and units are readable, and the report reads as a coherent scientific paper rather than project notes. I found no material spelling or grammatical defect in the rendered PDF.

## 3. Introduction — 4.5/5

The introduction motivates why average return is insufficient, separates reliability across states from reliability across seeds, states the practical comparison, and previews the principal result with exact counts (p. 1). The continuation in the right column clearly enumerates four contributions and explicitly limits the resource ledger and diagnostic claims (p. 1).

The opening research question—what prevents a policy from reaching successive “nines”—is broader than the final descriptive comparison of two selected pipelines. The report explains that the practical target changed, but a more operational hypothesis would align the introduction more tightly with the evidence: for example, whether the selected mixed pipeline reduces seed-dependent fixed-grid failures relative to the selected pure-RL pipeline under the stated route ledger. This is a framing refinement, not a missing motivation.

## 4. Related work — 4/5

The section is organized by technical relationship rather than as a paper list: SAC, DAgger, normalized residual networks, multi-step critics, reliable evaluation, demonstration-augmented RL, and critic-guided action optimization are connected directly to the project (pp. 1–2). It also distinguishes inherited components from the report’s deployment search and automatic start-selection choices.

Coverage is concise to the point of leaving some stated neighboring areas unsourced. The report itself names tail-sensitive objectives, hard-start curricula, equivariant actors/critics, uncertainty-gated selection, and search distillation as part of the broader design space (p. 2), but does not situate those areas with references. Adding a short, organized comparison to the most relevant work in those families would better establish which reliability ideas are adapted, omitted, or newly combined here. The eight listed references are relevant, but the literature synthesis is narrower than the method and discussion.

## 5. Technical section and results — 4/5

The report supplies strong protocol detail: observation/action spaces, reward and horizon, grid construction, seed identifiers, three outcome definitions, reference construction, deployment legality, learning and discovery ledgers, network sizes, replay/update settings, objectives, action-candidate sets, and selection rules (pp. 2–4). Results are reported as exact counts and percentages, and the key arithmetic is internally consistent: 12,496/12,505 = 99.928% and 11,832/12,505 = 94.618% (pp. 1, 4, and 9). The all-seed cell counts also reconcile: 2,255 all-seed successes + 207 variable cells + 39 all-seed failures = 2,501 pure-RL grid cells (pp. 4 and 6). Table 1 is particularly valuable because it exposes full-study resource and replication differences (p. 3).

The main deductions concern experimental identification rather than missing prose. The mixed headline uses five actor seeds but one retrospectively selected shared critic, while pure RL has five complete actor-critic runs (pp. 2–3). Method and checkpoint selection reused the standardized evaluation campaign, so the fixed grid is a descriptive benchmark rather than a fresh held-out generalization test (pp. 4 and 8). Full-study interactions, reference supervision, discovery cost, and optimizer updates are unmatched (pp. 2–3). The matched 100,000-step one-step versus properly weighted FastSACN8 comparison is absent (pp. 4 and 8). The report discloses all of these limitations responsibly, but they still limit the strength of the experimental result.

There is also one internal result-label ambiguity requiring correction. The mixed local Q-search is said to lose 239 strict wins relative to its “frozen no-shift actor” baseline (p. 5). Given the mixed winner’s 1,570 strict wins (pp. 4 and 9), that baseline would have 1,809 strict wins. The appendix row labelled “Supervised actor” reports 1,813 strict wins while matching the implied no-shift baseline on near-reference count (12,470) and task success (11,705) (p. 9). If these are different variants, the report needs to name and map them explicitly; if they are intended to be the same variant, the strict-win count or the stated −239 change is inconsistent.

## 6. Analysis and discussion — 5/5

This is the strongest section. Experiments are connected to claims, negative and conditional results are retained, and plausible mechanisms are separated from demonstrated effects. The report explains that the learner-state/priority comparison does not isolate its two ingredients and that the automatic-priority result has only one follow-up family (pp. 4–5). It reports both fixes and newly broken classifications for critic search and reflection (p. 5). The targeted 99-state diagnostic is treated as mechanism evidence rather than a population estimate (p. 6); saturation is explicitly called a hypothesis because its frequency is unmeasured (p. 7); and the one-seed joint-loss study is excluded from the main scorecard because it lacks replication and factorial controls (pp. 7–9).

The discussion covers shared-critic replication, unequal resource ledgers, missing matched baselines, targeted diagnostics, selection bias, and limited transfer beyond Pendulum (p. 8). The conclusion follows these qualifications and does not claim a causal effect of supervision. The evidence-tier annex is also a strong scientific practice (p. 9).

## Overall assessment

The report’s central claim is that, on the selected fixed-grid Pendulum benchmark, the selected clean mixed RL-plus-supervision pipeline has substantially higher near-reference reliability across initial states and actor seeds than the selected pure-RL pipeline.

The PDF supports that claim as a descriptive comparison of selected pipelines: the exact grid counts, cross-seed consistency measures, and paired deployment ablations are clearly presented. It does **not** support a causal claim that supervision itself produces the full gap, a claim of blind generalization, or a claim that five independently replicated mixed pipelines were compared with five pure-RL pipelines. The report generally states these boundaries correctly.

## Highest-priority corrections

1. **Reconcile the “Supervised actor” and no-shift baseline identities (Results/Table 2 and Annex/Table A.2, pp. 5 and 9).** Add a unique variant name or identifier to every scorecard and ablation baseline. Resolve the 1,813 strict wins in Table A.2 versus the 1,809 implied by 1,570 − (−239). This is the only apparent internal numerical/identity conflict and directly affects the technical-results score.

2. **Make the statistical unit visible in the headline result (Abstract, Results, and Figure 2, pp. 1 and 5).** Add per-actor-seed near-reference rates or a compact range/summary beside the pooled 12,505-trial total, while separately identifying the single shared selected critic. This would prevent readers from interpreting five actor seeds as five independent full-pipeline replications.

3. **Separate selection-grid performance from generalization more prominently (Abstract and Results, pp. 1 and 4).** The discussion discloses that the standardized grid campaign was used for method/checkpoint selection (p. 8); bring that qualification forward into the headline results or caption. A fresh held-out state grid or pre-registered selection split would be the substantive experimental remedy.

4. **Add the missing matched critic-target comparison (Technical results and Discussion, pp. 4 and 8).** The report itself identifies the absent 100,000-step UTD1 one-step versus properly weighted FastSACN8 comparison. Completing it would support claims about critic targets without the current differences in training budget and update ratio.

5. **Clarify mixed-data accounting and expand adjacent-work context (Method, p. 3; Related work, p. 2).** State explicitly how the 400,000 initial labels, 30,000 learner-state transitions, 240,000 reset-support labels, and 20,000 follow-up trajectory samples relate to the “resulting 260,000-example dataset” and to Table 1’s ledger. Cite and compare the tail-risk, automatic-curriculum, equivariance, uncertainty-gating, and distillation families already named in the report.

## Claim and evidence audit

- **Unequal-resource headline comparison (pp. 1–3):** The 99.928% versus 94.618% contrast compares selected pipelines, not matched full studies. Mixed and pure routes differ in full-study learning transitions, discovery interactions, reference queries, update counts, supervision, and replication unit. The report states this; the claim should remain descriptive.

- **Shared selected critic (pp. 2–3 and 5):** Five mixed actors share one critic seed selected retrospectively on an earlier standardized grid campaign. Uncertainty across actor seeds therefore does not quantify full mixed-pipeline variability. Figure 2’s caption and the discussion acknowledge this.

- **Evaluation-selection reuse (pp. 4 and 8):** Reflection/Q-search deployment rules and the selected methods were chosen using the completed standardized evaluation campaign. The displayed fixed-grid result is not a blind held-out generalization estimate.

- **Automatic-priority evidence (pp. 4–5):** The +10 near-reference, +2 task-success, and +111 strict-win changes come from one follow-up family over five actors. They are conditional evidence, not a replicated estimate of the priority mechanism. Table 2 labels this correctly.

- **Possible strict-win mismatch (pp. 5 and 9):** A −239 strict-win change from a 1,570-win mixed result implies 1,809 baseline wins, whereas the “Supervised actor” scorecard row gives 1,813. The PDF does not establish whether these labels denote different variants.

- **Reference interpretation (p. 2):** Near-reference success is closeness to the maximum of two stored reference solvers, not optimality or task completion. The report appropriately defines a separate task-success metric and notes that strict wins demonstrate reference fallibility.

- **Targeted critic diagnostic (p. 6):** The 0.693 median Spearman correlation, 0.844 ordering accuracy, 19.0% harmful-proposal rate, and 2.43× disagreement ratio are measured on 99 targeted hard states. They support a local mechanism account, not a population causal estimate.

- **Actor-saturation explanation (p. 7):** The derivative is mathematically relevant, but the report has not measured how often saturation blocks useful corrections. The text correctly labels this as a hypothesis.

- **Joint-loss pilot (pp. 7–9):** The 97.681% versus 70.252% result is a single-seed 25,000-step pilot with missing behavior-cloning-only and staged controls. It shows that the supervised term helps in that condition, but cannot identify simultaneous optimization as the cause or generalize across seeds.

- **Missing matched FastSACN8 test (pp. 4 and 8):** The 50,000-step UTD2 FastSACN8 family is not a matched test against the 100,000-step UTD1 one-step winner. The PDF correctly avoids claiming intrinsic superiority of the one-step target.
