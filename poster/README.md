# Poster v115

This directory intentionally contains only poster v115 and the assets it uses.
Older poster PDFs and their sources are preserved on the research branch, not
on clean `main`.

- `poster_visual_v115.html`: editable source
- `poster_visual_v115.pdf`: one-page A0 landscape delivery
- `poster_visual_v115.png`: high-resolution preview
- `assets/`: referenced numerical panels and editorial design assets

Rebuild the five numerical panels from preserved evidence into a disposable
poster tree:

```bash
uv run --frozen last-nine reproduce --target poster
```

Then optionally render that tree to A0 PDF/PNG (requires the presentation
extra and Microsoft Edge):

```bash
uv sync --frozen --extra presentation
uv run --frozen python scripts/render_a0_poster.py \
  --html .build/reproduction/poster/poster_visual_v115.html \
  --pdf .build/reproduction/poster/poster_visual_v115.pdf \
  --png .build/reproduction/poster/poster_visual_v115.png
```

The pendulum illustration is a preserved editorial asset, not numerical
experiment evidence. The renderer defaults to `.build` and does not overwrite
the checked-in delivery.

The recovery atlas uses compact derived trajectories for a fast routine build.
To independently re-extract those trajectories from the retained checkpoints:

```bash
uv run --frozen python scripts/extract_recovery_atlas_evidence.py --overwrite
```

That audit writes to `.build/recovery-atlas-evidence` by default and validates
its rollout returns against the retained common-grid tables.
