# Report-scoped experiments

This directory contains only training and diagnostics used by the final
ShareLaTeX report. The two short runners are the intended entry points; the
dated files in `implementations/` retain the exact scientific implementation
behind the selected recipes and diagnostic panels.

## Reward-only experiments

Run the selected five-seed SimbaV2 FastSACN8 recipe:

```bash
uv run --frozen python experiments/run_pure_report_matrix.py \
  --family pure_selected --seeds 0 1 2 3 4 --device auto
```

The report's one-step, FastSACN8, SACN8, and standard-MLP comparisons are the
other accepted `--family` values. `--dry-run` prints the fully resolved commands
without starting compute. At most two workers are allowed because the original
training GPU had 6 GB of memory; one worker is the safe default.

## Mixed experiments

Rebuild the recovered initializer from scratch:

```bash
uv run --frozen python experiments/run_mixed_report_pipeline.py initializer \
  --seed 0 --device auto
```

Re-run the selected automatic-priority 20k follow-up from the retained
seed-matched initializer and shared critic:

```bash
uv run --frozen python experiments/run_mixed_report_pipeline.py selected \
  --seed 0 --device auto
```

Use `uniform` for the report's uniform-start control. That control also used a
0.005 critic target blend, while the selected priority run used zero target
blend; it is therefore not an isolated priority-only contrast. The runner keeps
this historical difference explicit.

The recovered initializer used a 400k static reference dataset, 80 epochs,
three learner-only DAgger rounds of 10k transitions, and an angle-targeted 20%
near-upright sampling component. The latter and the seed-lineage split are
recorded in `../KNOWN_LIMITATIONS.md` and the machine-readable provenance file.

## Evidence versus retraining

`data/report/rollouts/` contains the exact five-seed 61 x 41 evaluation tables
used for report counts. `data/report/diagnostics/` contains the raw inputs for
the report's mechanistic figures. `artifacts/report_reproduction/` contains only
the checkpoints/configs needed for the listed report families.

Use `uv run --frozen last-nine evaluate` to validate the frozen evidence and
rebuild the report/poster figures. Verification and figure rendering are
inexpensive; retraining all report families is not. The report aggregates seeds
0 through 4; a one-seed command is only an example. Same-seed training is not
guaranteed to be bitwise identical across hardware or CUDA/PyTorch versions.

The JSON files in `protocols/` are frozen evaluation/diagnostic specifications,
not a catalog of exploratory sweeps.

The pure trainer and mixed initializer seed Python, NumPy, PyTorch, and CUDA
RNGs. The retained historical mixed follow-up uses explicit seeded NumPy
generators; it is not claimed to be bitwise deterministic across platforms.
