from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright


A0_WIDTH_MM = 1189.0
A0_HEIGHT_MM = 841.0
MIN_FONT_MM = 11.1
PX_PER_MM = 96.0 / 25.4


def pdf_page_size(pdf: Path) -> tuple[float, float]:
    result = subprocess.run(
        ["pdfinfo", str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        result.stdout,
    )
    if not match:
        raise RuntimeError("Could not parse PDF page size")
    width_pt, height_pt = map(float, match.groups())
    return width_pt * 25.4 / 72.0, height_pt * 25.4 / 72.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    html = args.html.resolve()
    pdf = args.pdf.resolve()
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.exists():
        raise FileNotFoundError(edge)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(edge), headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1132})
        page.goto(html.as_uri(), wait_until="networkidle")
        dom = page.evaluate(
            """() => {
              const directText = (el) => [...el.childNodes].some(
                (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim()
              );
              const visible = [...document.querySelectorAll('body *')]
                .filter((el) => {
                  const s = getComputedStyle(el);
                  const r = el.getBoundingClientRect();
                  return s.display !== 'none' && s.visibility !== 'hidden' &&
                    r.width > 0 && r.height > 0 && directText(el);
                })
                .map((el) => {
                  const r = el.getBoundingClientRect();
                  return {
                    tag: el.tagName,
                    className: String(el.className || ''),
                    text: el.innerText.trim().slice(0, 160),
                    fontPx: parseFloat(getComputedStyle(el).fontSize),
                    rect: [r.left, r.top, r.right, r.bottom],
                  };
                });
              return {
                body: [document.body.scrollWidth, document.body.scrollHeight],
                imageFailures: [...document.images].filter(
                  (img) => !img.complete || img.naturalWidth === 0
                ).length,
                text: visible,
              };
            }"""
        )
        browser.close()

    min_font_px = MIN_FONT_MM * PX_PER_MM
    undersized = [
        item for item in dom["text"] if item["fontPx"] + 0.05 < min_font_px
    ]
    width_mm, height_mm = pdf_page_size(pdf)
    page_ok = (
        abs(width_mm - A0_WIDTH_MM) <= 1.0
        and abs(height_mm - A0_HEIGHT_MM) <= 1.0
    )
    payload = {
        "html": str(html),
        "pdf": str(pdf),
        "expected_page_mm": [A0_WIDTH_MM, A0_HEIGHT_MM],
        "actual_page_mm": [width_mm, height_mm],
        "page_ok": page_ok,
        "orientation": "landscape" if width_mm > height_mm else "portrait",
        "minimum_font_mm": MIN_FONT_MM,
        "minimum_font_px": min_font_px,
        "observed_minimum_font_px": min(item["fontPx"] for item in dom["text"]),
        "undersized_text": undersized,
        "image_failures": dom["imageFailures"],
        "dom_body_px": dom["body"],
    }
    payload["pass"] = (
        page_ok
        and payload["orientation"] == "landscape"
        and not undersized
        and payload["image_failures"] == 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
