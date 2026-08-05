# Reproduce the report figures using our existing data

From the repository root, download the Git LFS data, create the Python
environment, and redraw the nine figures used by the report:

```bash
git lfs install
git lfs pull
uv python install 3.11
uv sync --frozen
uv run --frozen last-nine reproduce --target report
```

This command checks that the downloaded data is complete, then uses it to
recreate the nine report figures under `.build/reproduction/report/figures/`.
It also saves the calculated numbers and a comparison with the report figures.

## Reproduce the data needed for the figures from existing checkpoints

Run one command:

```bash
uv run --frozen python experiments/reproduce_report_data.py --device auto
```

It rebuilds the rollout tables and diagnostics from the retained checkpoints
and replay archives, then redraws all nine figures. The data is written to
`.build/report-data/` and the figures to `.build/report-data/figures/`.

## Reproduce the checkpoints from the seeds

Rebuild all report checkpoint families with:

```bash
uv run --frozen python experiments/reproduce_report_checkpoints.py --device auto
```

The rebuilt model tree is `runs/report_reproduction/models/`. To rebuild the
figure data and figures from those new checkpoints, run:

```bash
uv run --frozen python experiments/reproduce_report_data.py \
  --models-root runs/report_reproduction/models --device auto
```

## Exploratory history

The versioned exploratory history is preserved on
`research/exploratory-archive-20260802`. That branch contains intermediate
reports, broad sweeps, temporary diagnostics, abandoned trials, and essentially
all the work completed during the semester. The `main` branch provides a
focused, easy way to reproduce what was included in the report.

## Detailed fresh-clone setup

Install Git, Git LFS, and [uv](https://docs.astral.sh/uv/), then run:

```bash
git lfs install
git clone https://github.com/alasleimi/tum-adlr-ss26-13.git
cd tum-adlr-ss26-13
git lfs pull
uv python install 3.11
uv sync --frozen --extra test
uv run --frozen last-nine verify --require-manifest
uv run --frozen pytest
```

### Repository map

| Path | Purpose |
| --- | --- |
| `src/last_nine_rl/` | Training, checkpoint, evaluation, DAgger, SACn, and Q-search code |
| `src/last_nine_repro/` | Evidence verification and report/poster figure reproduction |
| `experiments/` | Report-scoped runners, implementations, and frozen protocols |
| `artifacts/report_reproduction/` | Retained checkpoints, configs, summaries, and selected replay archives |
| `data/reference/` | Frozen DP and hand-controller reference data |
| `data/report/` | Manifest-protected rollouts, diagnostics, claims, and provenance |
| `report/` | Final ShareLaTeX bundle, self-contained source, and PDF |
| `poster/` | Poster v115 source, assets, PNG, and PDF |
| `presentation/` | HTML presentation, local media, notes, and print-ready PDF |
| `submit/` | Submission-ready report and presentation PDFs |
| `scripts/` | Manifest, evidence-extraction, and rendering utilities |
| `tests/` | Unit, smoke, and reproduction tests |
