# Chasing the Nines

This is the submission-focused branch of ADLR Project 15. It retains the final
report, poster v115, the HTML presentation and its PDF rendering, and only the
code and immutable evidence needed for experiments discussed in the report.

The complete exploratory history is preserved on
`research/exploratory-archive-20260802`. That branch contains intermediate
reports, abandoned trials, broad sweeps, temporary diagnostics, and the full
research process. Do not use it as the grading entry point, but do use it when
mining old results or reconstructing work outside the final report.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/last_nine_rl/` | Training, checkpoint, evaluation, DAgger, SACn, and Q-search implementation |
| `src/last_nine_repro/` | Report-evidence validation and figure reproduction CLI |
| `experiments/` | Report-scoped experiment recipes, implementations, and frozen protocols |
| `artifacts/report_reproduction/` | Final checkpoints, run configs, summaries, and selected replay archives |
| `data/reference/` | Frozen DP and hand-controller reference data |
| `data/report/` | Common-grid rollouts, diagnostics, claims, and provenance metadata |
| `report/` | Final ShareLaTeX bundle, self-contained source, and compiled PDF |
| `poster/` | Poster v115 source, assets, PNG, and PDF; no older poster versions |
| `presentation/` | HTML presentation, local media, notes, and print-ready PDF |

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are recommended. Git LFS must
be installed before cloning because checkpoints, arrays, images, PDFs, and the
large rollout tables are LFS objects.

```bash
git lfs install
git lfs pull
uv sync --extra test --extra presentation
```

The core test suite is intentionally CPU-safe:

```bash
uv run pytest
```

## Verify the reported evidence

The quickest grading check validates file hashes, rollout schemas, the shared
61 x 41 x 5 evaluation grid, metric semantics, exact report counts, and known
provenance qualifications:

```bash
uv run last-nine verify
uv run python scripts/build_artifact_manifest.py --check
```

Generate report figures into a disposable directory after verification:

```bash
uv run last-nine reproduce --output .build/report-figures
```

The committed figures in `report/source/figures/` are the authoritative
ShareLaTeX inputs. The disposable render is for an independent comparison.

## Run the report experiments

The selected pure-RL recipe is ordinary reward-only training and can be run
from a retained seed config while changing only the seed and output directory:

```bash
uv run last-nine-train \
  --config artifacts/report_reproduction/models/pure_selected/seed0/config.json \
  --seed 0 \
  --run-dir runs/pure_fastsacn8/seed0
```

The mixed route is staged: reference pre-training, learner-state DAgger, the
high-regret follow-up, then frozen-critic deployment search. Exact report
commands and input/output roles are documented in `experiments/README.md`;
the implementation is retained in `experiments/implementations/` rather than
duplicated in notebooks or one-off report builders.

Training 100k-transition multi-seed studies is computationally expensive.
Evidence verification and figure reproduction use the retained outputs and do
not retrain models.

## Build the deliverables

```bash
# Report (requires a TeX installation; writes only under .build/)
latexmk -cd '-outdir=../../.build/report' -pdf \
  -interaction=nonstopmode -halt-on-error report/source/main.tex

# Presentation (requires the presentation extra and Edge/Chromium)
uv run python scripts/render_presentation_pdf.py

# Poster v115
uv run python scripts/render_a0_poster.py
```

See `KNOWN_LIMITATIONS.md` before interpreting the mixed ablations or the
target-family diagnostic figure. These qualifications preserve the historical
record; the final ShareLaTeX report itself is retained unchanged.
