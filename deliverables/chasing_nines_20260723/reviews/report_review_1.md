# Blind final-report review

## Scope note

This review uses only `REPORT_GRADING_TEMPLATE.md` and the rendered eight-page PDF. The supplied rubric is explicitly tailored to an RGB-only 3D Gaussian Splatting project, including geometry, depth, normals, exposure, pruning, and 10k/2k multi-scene experiments. The submitted PDF instead reports a fixed-grid Pendulum reinforcement-learning case study. I therefore credit the report where it satisfies the rubric's general scientific-writing criteria, but I also penalize the explicit project-specific requirements that it does not address. The resulting score should be read as a strict score against the supplied rubric, not as a grade under a hypothetical reinforcement-learning rubric.

## Score summary

| Category | Score |
|---|---:|
| Effective visual aids | 4.5/5 |
| Spelling and grammar | 4.5/5 |
| Introduction | 3.0/5 |
| Related work | 2.5/5 |
| Technical section and results | 3.0/5 |
| Analysis and discussion | 4.0/5 |
| **Total** | **21.5/30** |

### 1. Effective visual aids — 4.5/5

Figures 1–6 and Tables 1–2 are clean, legible, captioned, and consistently referenced (pp. 3, 5–7). Figure 1 gives a useful mechanism-level pipeline overview; Figures 3 and 4 make the state-by-seed reliability argument immediately visible; Figure 6 clearly separates gains and harms across three outcome definitions. Colors, legends, axes, units, denominators, and replication caveats are generally complete. The principal minor weakness is density: Figure 5 is small relative to the amount of surrounding analysis, and several plots use small labels at A4 page size (pp. 6–7). The visuals are internally consistent with the displayed numerical claims, although implementation agreement cannot be checked in this PDF-only review.

### 2. Spelling and grammar — 4.5/5

The writing is unusually polished, precise, and coherent throughout. Terms such as near-reference success, strict reference win, task success, selected-route ledger, actor seed, and replication unit are defined and then used consistently (pp. 1–3). Captions and mathematical notation are accurate and readable. Minor deductions are for occasional over-compressed sentences and noun stacks, especially in the abstract and budget discussion (pp. 1–2), plus awkward heading line breaks on pp. 4, 6, and 7. These do not materially impede understanding.

### 3. Introduction — 3.0/5

For the actual Pendulum topic, the introduction clearly motivates tail reliability, distinguishes reliability across states from reliability across seeds, poses a concrete research question, previews the main result, and avoids a causal overclaim (p. 1). However, it does not address the rubric's required RGB-only 3DGS limitation or any 3D-reconstruction contribution. Even within its own domain, the four stated contributions are mostly an audit, selection of completed pipelines, ablation, and diagnosis; the introduction could distinguish more sharply what method was newly implemented versus inherited from SAC, DAgger, SimbaV2, FastSACN8, reflection, and Q-search.

### 4. Related work — 2.5/5

The section is organized by technical relationship rather than as a paper-by-paper list, and it connects SAC, DAgger, SimbaV2, multi-step critics, reliable evaluation, demonstration augmentation, and critic-guided search to the reported system (pp. 1–2). It also identifies several untested neighboring ideas. Nevertheless, it is brief and does not clearly tabulate which elements are inherited, adapted, or newly proposed. It gives limited coverage of tail-sensitive RL objectives, reliability calibration, automatic curricula, equivariant control, and uncertainty-aware action correction. Most importantly under the supplied rubric, it contains none of the required 3DGS geometry, depth, normal, exposure, or pruning literature.

### 5. Technical section and results — 3.0/5

For the Pendulum study, the technical reporting is strong: the state representation, action bounds, reward, evaluation grid, success definitions, reference construction, training budgets, update counts, network sizes, objectives, non-gradient deployment searches, checkpoint choices, and ablation replication units are described in substantial detail (pp. 2–5). The main percentages agree with the displayed counts: 12,496/12,505 = 99.928%, 11,832/12,505 = 94.618%, 2,493/2,501 = 99.680%, and 2,255/2,501 = 90.164%. Figure 2 and Annex Table A.2 also agree to the shown precision (pp. 5 and 8).

The evidence is weakened by the admitted unequal full-study costs, reference-query counts, and replication structures (pp. 2–3); selection on the same stored standardized grid later used for the headline score (pp. 3–4); one retrospectively selected shared critic for all mixed actors (p. 3); and the absence of a fresh held-out evaluation or complete mixed-pipeline replications. Under the literal rubric, the renderer outputs, depth/normal/exposure/pruning mechanisms, multi-scene 10k result, and 2k component ablation are entirely absent.

### 6. Analysis and discussion — 4.0/5

The report does real analysis rather than merely restating results. It separates reliability from strict wins, connects cross-seed maps to the coverage hypothesis, reports harmful critic proposals, labels actor saturation and joint-loss arguments as hypotheses or pilot diagnostics, discusses negative and incomplete experiments, and repeatedly distinguishes descriptive evidence from causal or population-level claims (pp. 4–8). The limitations section is candid about correlated grid states, targeted diagnostics, selection bias, unmatched resources, one shared critic, and lack of transfer evidence (p. 8).

The remaining limitation is that the strongest explanation—better state coverage from supervision—has not been established by a fully replicated, equal-resource, held-out comparison. The broader significance is also restricted to a one-environment case study. The rubric's requested discussion of multi-scene 3D reconstruction, pose assumptions, initialization, image resolution, reference quality, and exposure behavior is not applicable to and not present in this report.

## Overall assessment

The central claim is: on the fixed 2,501-state Pendulum grid, the selected clean mixed supervision-plus-RL pipeline achieves substantially more consistent near-reference reliability across actor seeds than the selected pure-RL pipeline.

The PDF supports that claim as a descriptive comparison of the selected frozen pipelines: the counts, consistency maps, and paired deployment ablations are mutually coherent. It does **not** support a causal claim that supervision is intrinsically superior, a generalization claim beyond the audited grid and environment, or an equal-resource comparison of independently replicated end-to-end pipelines. The report usually acknowledges these boundaries correctly.

## Three most important remaining weaknesses

### 1. The report does not match the supplied project-specific rubric

**Pages:** 1–8.

The rubric requires an RGB-only 3DGS problem statement, geometry/depth/normal/exposure/pruning methods, multi-scene training evidence, and a 2k component ablation. The PDF is exclusively about Pendulum reinforcement learning. This directly caps the Introduction, Related Work, Technical Results, and Discussion scores no matter how well the RL study is written.

**Concrete revision:** If this is the intended rubric, the report must be replaced or comprehensively reframed around the required 3DGS project, including the requested method components and 10k/2k experiments. If the Pendulum report is the intended submission, attach and grade it with a domain-appropriate RL rubric; this is not a problem that a local wording edit can solve.

### 2. The headline comparison is selected, unequal-resource, and not held out

**Pages:** 2 (§3.2), 3 (Table 1 and §4.2), 4 (§4.4 and §5.1), and 8 (§8–9).

The mixed and pure pipelines differ in full-study learning transitions, discovery interactions, reference supervision, update counts, and replication structure. The mixed route shares one retrospectively selected critic, and the standardized grid was involved in selection before being reused for the final score. The report is commendably candid, but candor does not remove the confound.

**Concrete revision:** Add a preregistered evaluation on untouched initial states and at least one additional continuous-control environment. Retrain complete end-to-end mixed and pure pipelines with independently sampled actor and critic seeds. Report both an equal-total-interaction comparison and an equal-learning-update comparison, with reference-query cost shown separately. Keep the present audit as development evidence and reserve the held-out results for the headline claim.

### 3. Component attribution and novelty remain conditional

**Pages:** 1–2 (contributions and related work), 4–5 (§6 and Table 2), 6–7 (§7), and 8 (§8).

The abstract says the follow-up stage, conservative critic search, symmetry projection, and global critic search “each contribute,” but several comparisons do not isolate a single factor. The learner follow-up combines learner-state collection with automatic priority; automatic priority has only one follow-up family; the mixed critic is a single selected replication; and reflection/Q-search results are largely deployment ablations on existing checkpoints. The report also does not state crisply whether its novelty is a new algorithm, a benchmark protocol, a pipeline synthesis, or an empirical audit.

**Concrete revision:** Add a contribution table with columns for inherited, modified, and newly proposed elements. Run a crossed ablation over learner-state collection × priority selection × local Q-search, with independently trained critics and complete pipeline seeds. Report seed-level paired effects and uncertainty, not only pooled cell-count changes. Expand related work to compare directly with tail-sensitive objectives, hard-start curricula, equivariant policies/critics, and uncertainty-gated action selection, then narrow the novelty claim to what those controls establish.

## Claim and evidence audit

- **Internally matched headline numbers:** The main counts and percentages agree across the abstract, §5, Figures 2–4, and Annex Table A.2 (pp. 1, 4–6, 8).
- **Budget/replication qualification:** The PDF correctly states that the selected-route ledger does not equalize full-study cost and that the mixed result is not five independent full-pipeline replications (pp. 1–3, 5, 8). Any shorter reuse of the headline result should retain this qualification.
- **Potentially overstated component claim:** “Each contribute” in the abstract (p. 1) is stronger than the conditional and partly combined evidence in §6/Table 2 (pp. 4–5). Replace it with “the paired deployment and follow-up comparisons are consistent with contributions from…” until fully crossed replications are available.
- **Selection and held-out status:** The pure deployment rule and mixed critic were selected retrospectively, and the final grid is described as a benchmark comparison rather than blind generalization (pp. 3–4). Conclusions should remain explicitly benchmark-specific.
- **Metric definitions:** Near-reference success, strict reference win, task success, and mean return are defined (p. 2). The report appropriately warns that strict wins are not a stronger form of near-reference success (p. 4).
- **Statistical unit:** Neighboring grid states are acknowledged as correlated, and the report avoids treating 12,505 trials as independent population samples (p. 2). However, no seed-level uncertainty interval is reported, and one shared critic prevents full-pipeline variance estimation.
- **Diagrams and implementation:** No contradiction is visible between Figure 1, the prose method description, and the reported result tables (pp. 3–5). Agreement with implementation cannot be verified under this PDF-only review.
- **Conclusions:** The benchmark-specific conclusion follows from the displayed evidence. A causal claim about supervision, an intrinsic superiority claim about one-step SAC, or transfer beyond Pendulum would not; the report generally avoids those claims (pp. 4, 7–8).
