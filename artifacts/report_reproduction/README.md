# Retained experiment artifacts

This allowlist contains only checkpoints needed for final-report methods and
diagnostics. Every model directory is run-like:

```text
models/<family>/seedN/
  config.json
  checkpoints/final.pt
  training_summary.json  # where the historical run produced one
```

Families are named by their role in the report: selected mixed and pure
pipelines, mixed controls, P0/P1/P2 target diagnostics, standard-MLP target
checks, canonical DAgger, and the objective-share replay comparison.

The six replay archives are under `replay/objective_share/`. `PROVENANCE.tsv`
binds each clean path to its original research path, byte size, and SHA-256.
The repository-wide `data/report/manifest.json` independently hashes these
copies together with all curated evidence and deliverables.
