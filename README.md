# Chasing the Nines

This is the submission-focused branch of ADLR Project 15. It retains the final
report, poster v115, HTML presentation and PDF, plus only the code and immutable
evidence needed for experiments discussed in the report.

The versioned exploratory history is preserved on
`research/exploratory-archive-20260802`. That branch contains intermediate
reports, broad sweeps, temporary diagnostics, and abandoned trials. A separate
pre-operation filesystem snapshot preserves the original untracked working
files and large local trove. Use `main` as the grading entry point and the
research archive only when reconstructing work outside the final report.
Backup names, verification counts, and recovery instructions are recorded in
`RESEARCH_ARCHIVE.md`.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/last_nine_rl/` | Training, checkpoint, evaluation, DAgger, SACn, and Q-search code |
| `src/last_nine_repro/` | Evidence verification and report/poster figure reproduction |
| `experiments/` | Report-scoped recipes, implementations, and frozen protocols |
| `artifacts/report_reproduction/` | Retained checkpoints, configs, summaries, and selected replay archives |
| `data/reference/` | Frozen DP and hand-controller reference data |
| `data/report/` | Common-grid rollouts, diagnostics, claims, and provenance |
| `report/` | Final ShareLaTeX bundle, self-contained source, and PDF |
| `poster/` | Poster v115 source, assets, PNG, and PDF; no older versions |
| `presentation/` | HTML presentation, local media, notes, and print-ready PDF |

## Fresh-clone evaluator

Git LFS is required because checkpoints, arrays, images, PDFs, and large CSVs
are LFS objects. From a fresh clone:

```bash
git lfs install
git lfs pull
uv python install 3.11
uv sync --frozen --extra test
uv run --frozen last-nine evaluate
uv run --frozen pytest
```

`.python-version` selects Python 3.11, and `uv.lock` pins the tested submission
environment. The lockfile is not a recovered freeze of the environment that
originally trained every checkpoint, so the README deliberately does not
duplicate or overclaim historical package versions.

`last-nine evaluate` is the one-command grading path. It:

1. verifies manifest SHA-256 hashes and byte sizes;
2. validates rollout schemas, the shared 61 x 41 x 5 grid, claims, metric
   semantics, and provenance qualifications;
3. rebuilds all nine report figures and five numerical poster panels from
   preserved evidence; and
4. writes metrics and informational image comparisons under
   `.build/reproduction/`.

Nothing in the canonical `report/` or `poster/` directories is overwritten.
The numerical checks are gating. Image byte equality is reported but is not a
correctness requirement because fonts, Matplotlib, and operating systems can
change rasterization without changing the data.

Useful narrower commands are:

```bash
uv run --frozen last-nine verify --require-manifest
uv run --frozen last-nine reproduce --target report
uv run --frozen last-nine reproduce --target poster
```

The poster target stages a complete disposable HTML tree at
`.build/reproduction/poster/`. Five panels are independently regenerated. The
pendulum illustration is a manifest-protected editorial design asset, not
experiment evidence, and is therefore copied rather than falsely presented as
a numerical reproduction.

## Render the deliverables

The figure evaluator does not require a browser or TeX installation. Those
tools are optional for final-document rendering:

```bash
# Report: writes only below .build/report/
latexmk -cd '-outdir=../../.build/report' -pdf \
  -interaction=nonstopmode -halt-on-error report/source/main.tex

# Browser renderers (require Microsoft Edge; pass --edge PATH if undiscovered)
uv sync --frozen --extra test --extra presentation
uv run --frozen python scripts/render_a0_poster.py \
  --html .build/reproduction/poster/poster_visual_v115.html \
  --pdf .build/reproduction/poster/poster_visual_v115.pdf \
  --png .build/reproduction/poster/poster_visual_v115.png
uv run --frozen python scripts/render_presentation_pdf.py
```

Both browser renderers default to `.build/`; ordinary use cannot replace the
checked-in poster or presentation PDFs.

## Same-seed experiment replication

The report's actor rows aggregate exactly five actor seeds: `0 1 2 3 4`.
The pure actors are independently trained; the mixed actors share the retained
critic documented in the report. Run the selected reward-only family with the
report-scoped wrapper:

```bash
uv run --frozen python experiments/run_pure_report_matrix.py \
  --family pure_selected --seeds 0 1 2 3 4 --device auto
```

Use `--dry-run` first to inspect every resolved command. The mixed route is
staged; for example, seed 0 can be rebuilt and followed by its selected refit:

```bash
uv run --frozen python experiments/run_mixed_report_pipeline.py \
  initializer --seed 0 --device auto
uv run --frozen python experiments/run_mixed_report_pipeline.py \
  selected --seed 0 --device auto \
  --initializer-run runs/report_reproduction/mixed_initializer/seed0
```

Repeat the mixed commands for seeds 1 through 4 to rebuild the selected mixed
actor family conditional on that retained shared critic. Outputs go under
ignored `runs/` paths and existing completed runs are never overwritten unless
`--overwrite` is explicit.

The pure trainer and mixed initializer seed Python, NumPy, PyTorch, and CUDA
RNGs. The historical mixed follow-up uses explicit seeded NumPy generators but
is not presented as fully deterministic. Same-seed replication is not
byte-identical across different hardware, CUDA/cuDNN, or PyTorch versions.
Most retained configs were trained for CUDA; `--device auto` also runs on CPU
when CUDA is unavailable, but report-scale 100k-transition studies will be much
slower. Exact report reproduction comes from the immutable hashed evidence,
not from promising bitwise-identical retraining.

See `experiments/README.md` for all retained report families and
`KNOWN_LIMITATIONS.md` before interpreting mixed ablations or target-family
diagnostics.

For a quick deterministic Gym-reset smoke evaluation of any checkpoint:

```bash
uv run --frozen last-nine-eval \
  artifacts/report_reproduction/models/pure_selected/seed0 \
  --deployment pure-qsearch --episodes 2 --device auto
```

Use `--deployment mixed-qsearch` for a selected mixed actor (the retained shared
critic is the default) or `actor` for its raw deterministic actor. This command
is deliberately labelled a smoke evaluation; `last-nine evaluate` remains the
report-grade frozen common-grid check.
