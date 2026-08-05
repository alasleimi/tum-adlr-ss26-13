# Report-scoped experiments

This directory contains the training, evaluation, and diagnostic entry points
used by the final ShareLaTeX report. The dated files in `implementations/`
retain the scientific implementations behind the short commands below.

## Complete reproduction commands

Rebuild all report checkpoint families from their recorded seeds:

```bash
uv run --frozen python experiments/reproduce_report_checkpoints.py --device auto
```

Rebuild the nine figures' input data from the retained checkpoints and draw the
figures:

```bash
uv run --frozen python experiments/reproduce_report_data.py --device auto
```

Use the newly trained checkpoints instead with:

```bash
uv run --frozen python experiments/reproduce_report_data.py \
  --models-root runs/report_reproduction/models --device auto
```

Both commands accept `--dry-run`. The checkpoint command resumes completed
families, and the data command writes its results under `.build/report-data/`.

## Focused runners

`run_pure_report_matrix.py` rebuilds selected reward-only families.
`run_mixed_report_pipeline.py` exposes the initializer, selected, uniform, and
priority-shifted stages individually. The complete checkpoint command invokes
these recipes in dependency order and supplies the rebuilt shared critic.

The JSON files in `protocols/` are frozen evaluation/diagnostic specifications,
and `implementations/` contains their reusable training and diagnostic code.
