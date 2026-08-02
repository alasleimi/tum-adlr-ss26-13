# HTML presentation

`index.html` is the remotely tracked workshop presentation source. All image
and animation assets are local to this directory. The print stylesheet fixes
each slide to a 16:9 page with backgrounds enabled.

Generate the PDF with:

```bash
uv sync --extra presentation
uv run python scripts/render_presentation_pdf.py
```

The renderer waits for images, fonts, and MathJax, audits the slide count, and
writes `week3_workshop_presentation_20260527.pdf`. The expected result is 18
pages, each 960 x 540 PDF points.
