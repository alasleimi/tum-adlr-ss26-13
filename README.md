# Chasing the Nines

This is the submission-focused branch of ADLR Project 15. It retains the final
report, poster v115, HTML presentation and PDF, plus only the code and immutable
evidence needed for experiments discussed in the report.

The versioned exploratory history is preserved on
`research/exploratory-archive-20260802`. That branch contains intermediate
reports, broad sweeps, temporary diagnostics, and abandoned trials.



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



`last-nine evaluate`

1. verifies manifest SHA-256 hashes and byte sizes;
2. validates rollout schemas, the shared 61 x 41 x 5 grid, claims, metric
   semantics, and provenance qualifications;
3. rebuilds all nine report figures and five numerical poster panels from
   preserved evidence; and
4. writes metrics and informational image comparisons under
   `.build/reproduction/`.




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


For a quick deterministic Gym-reset smoke evaluation of any checkpoint:

```bash
uv run --frozen last-nine-eval \
  artifacts/report_reproduction/models/pure_selected/seed0 \
  --deployment pure-qsearch --episodes 2 --device auto
```

Use `--deployment mixed-qsearch` for a selected mixed actor (the retained shared
critic is the default) or `actor` for its raw deterministic actor.
