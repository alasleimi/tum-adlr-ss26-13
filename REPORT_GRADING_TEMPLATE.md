# Report Grading Template

This local template is based closely on the report-writing guidance and grading
criteria in `first-course-slide.pdf` from the 3D seminar. The original report
rubric is preserved: six categories worth 5 points each, for 30 points total.

For this practical-course project, interpret references to "the paper" as the
project's proposed method, implementation, and experimental evidence. Do not
reward work merely because it exists in the repository. Grade what the report
communicates and supports.

## Expected report structure

1. **Abstract:** One paragraph covering the problem, method, key results, and
   main takeaway.
2. **Introduction:** Motivate the problem and state the contribution clearly in
   the authors' own words.
3. **Related work:** Organize relevant methods by idea or family and explain how
   this project differs.
4. **Method:** Explain the approach clearly. Use diagrams or notation when they
   improve understanding.
5. **Experiments and results:** Explain training, datasets, baselines, metrics,
   and protocols. Present the key findings rather than every available number.
6. **Analysis and discussion:** Interpret the results, test whether ablations
   support the claims, discuss negative results, and identify limitations.
7. **References:** Cite the sources needed to establish context, prior work,
   datasets, and borrowed methods.

## Scoring scale

- **5:** Excellent. Complete, precise, easy to follow, and supported by evidence.
- **4:** Good. Substantively correct with minor omissions or presentation issues.
- **3:** Adequate. Understandable but incomplete, weakly prioritized, or uneven.
- **2:** Weak. Important information is unclear, unsupported, or poorly organized.
- **1:** Minimal. The category is barely addressed.
- **0:** Missing or fundamentally unusable.

Use intermediate half-points only when the evidence genuinely lies between two
anchors. Be strict. Do not raise a score because the project appears ambitious.

## Grading rubric

### 1. Effective visual aids: 5 points

Check whether figures, diagrams, plots, and tables:

- communicate a specific result or mechanism;
- remain legible at the intended page size;
- have complete labels, units, legends, and captions;
- are referenced and interpreted in the prose;
- distinguish baselines, proposed variants, targets, and evaluation data;
- avoid clipped text, hidden elements, misleading scales, and decorative clutter;
- agree with the method, code, and reported numbers.

Deduct heavily when a visual requires prior knowledge that the report has not
provided, or when its caption makes claims that cannot be seen in the visual.

Score: **__/5**

Evidence and comments:

### 2. Spelling and grammar: 5 points

Check whether the writing:

- is clear, concise, and grammatically correct;
- uses technical terms consistently and defines them before use;
- avoids vague antecedents, unexplained abbreviations, and non-sequiturs;
- uses accurate captions, headings, mathematical notation, and units;
- reads as a coherent scientific report rather than notes or repository documentation.

Score: **__/5**

Evidence and comments:

### 3. Introduction: 5 points

Check whether the introduction:

- explains why the problem matters;
- identifies the limitation of RGB-only 3DGS that motivates the work;
- states the research question or hypothesis;
- identifies the proposed contributions precisely;
- distinguishes the implemented contribution from existing work;
- previews the principal result without overclaiming.

Score: **__/5**

Evidence and comments:

### 4. Related work: 5 points

Check whether related work:

- covers the most relevant prior approaches;
- groups work by technical relationship rather than listing papers one by one;
- explains what is inherited, adapted, or newly proposed;
- compares the project with appropriate 3DGS geometry, depth, normal,
  exposure, and pruning methods;
- uses accurate, complete, and non-hallucinated references.

Score: **__/5**

Evidence and comments:

### 5. Technical section and results: 5 points

Check whether the report clearly explains:

- the representation and renderer outputs;
- depth, normal, exposure, and pruning mechanisms;
- the complete optimization objective and non-gradient operations;
- datasets, splits, initialization, training budgets, seeds, and baselines;
- metric definitions and whether training and held-out evidence are separated;
- the main 10k multi-scene result and the role of the 2k component ablation;
- controlled and robustness experiments without presenting them as equivalent
  forms of evidence;
- enough implementation and protocol detail for meaningful reproduction.

Verify that each numerical claim matches its table or figure. Flag confounds,
unmatched comparisons, unclear aggregation, missing variance, and selective
reporting.

Score: **__/5**

Evidence and comments:

### 6. Analysis and discussion: 5 points

This category should separate a strong report from a method description. Check
whether the report:

- explains what the results mean and why they occur;
- connects each experiment to a stated claim or research question;
- uses ablations to identify which components contribute;
- reports negative or conditional results, including exposure behavior;
- distinguishes demonstrated findings from plausible explanations;
- discusses dataset count, correlated trajectories, pose assumptions,
  initialization, resolution, training budget, and reference-quality limits;
- explains why the contribution matters in the broader 3D reconstruction context;
- ends with conclusions that follow directly from the evidence.

Score: **__/5**

Evidence and comments:

## Required grading output

### Score summary

| Category | Score |
|---|---:|
| Effective visual aids | __/5 |
| Spelling and grammar | __/5 |
| Introduction | __/5 |
| Related work | __/5 |
| Technical section and results | __/5 |
| Analysis and discussion | __/5 |
| **Total** | **__/30** |

### Overall assessment

State the report's central claim in one sentence. Then judge whether the report
actually supports it.

### Highest-priority corrections

List the three to five changes most likely to improve the grade. For each one,
identify the affected section and explain the grading consequence.

### Claim and evidence audit

Flag:

- unsupported or overstated claims;
- results quoted with the wrong budget, variant, scene set, or aggregation;
- diagrams that disagree with the implementation;
- metrics whose definitions are missing or misleading;
- comparisons with unequal data, initialization, training, or evaluation;
- conclusions that do not follow from the displayed evidence.

### Common mistakes to penalize

- Copying language from related papers without clear attribution.
- Describing the implementation without explaining the research contribution.
- Reporting many numbers without identifying the central findings.
- Ignoring limitations, failure cases, or negative results.
- Omitting the broader significance of the contribution.
- Treating seeds, views, trajectories, and physical environments as interchangeable
  statistical units.
- Presenting controlled-scene mechanism tests as real-scene generalization.
- Giving repository details that do not help the scientific argument.
