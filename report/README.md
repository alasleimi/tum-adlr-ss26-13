# Final report

- `Chasing_the_Nines_Report_FINAL_20260731.pdf` is the submitted nine-page A4
  document.
- `Chasing_the_Nines_ShareLaTeX_20260731.zip` is the authoritative upload
  bundle.
- `source/` is the extracted, self-contained bundle for local inspection and
  compilation. `source/main.tex` is the entry point.

Build with pdfLaTeX through latexmk:

```bash
latexmk -pdf -cd report/source/main.tex
```

The report text and committed figures are preserved unchanged. Recomputed
figures should be written elsewhere, for example:

```bash
uv run last-nine reproduce --output .build/report-figures
```

See `../KNOWN_LIMITATIONS.md` for provenance qualifications found during the
clean-branch audit.
