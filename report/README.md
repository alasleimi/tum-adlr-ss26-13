# Final report

- `Chasing_the_Nines_Report_FINAL_20260803.pdf` is the reviewed nine-page A4
  document.
- `Chasing_the_Nines_ShareLaTeX_20260803.zip` is the authoritative upload
  bundle.
- `source/` is the clean, self-contained upload folder. `source/main.tex` is
  the entry point and uses the single final body, `source/report_body.tex`.

Build with pdfLaTeX through latexmk:

```bash
latexmk -cd '-outdir=../../.build/report' -pdf \
  -interaction=nonstopmode -halt-on-error report/source/main.tex
```

The committed figures are preserved unchanged. Recomputed figures should be
written elsewhere, for example:

```bash
uv run --frozen last-nine reproduce --target report
```

See `../KNOWN_LIMITATIONS.md` for provenance qualifications found during the
clean-branch audit.
