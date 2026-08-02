# Poster v115

This directory intentionally contains only poster v115 and the assets it uses.
Older poster PDFs and their sources are preserved on the research branch, not
on clean `main`.

- `poster_visual_v115.html`: editable source
- `poster_visual_v115.pdf`: one-page A0 landscape delivery
- `poster_visual_v115.png`: high-resolution preview
- `assets/`: the eight referenced report panels

Render and validate with:

```bash
uv run python scripts/render_a0_poster.py
```
