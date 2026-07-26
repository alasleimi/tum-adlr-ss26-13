from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    html = args.html.resolve()
    pdf = args.pdf.resolve()
    png = args.png.resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)

    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.exists():
        raise FileNotFoundError(edge)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(edge),
            headless=True,
        )
        page = browser.new_page(
            viewport={"width": 1600, "height": 1132},
            device_scale_factor=1,
        )
        page.goto(html.as_uri(), wait_until="networkidle")
        audit = page.evaluate(
            """() => {
                const offenders = [...document.querySelectorAll('*')]
                  .filter((el) => !['HTML', 'BODY'].includes(el.tagName))
                  .filter((el) => el.clientWidth > 0 && el.clientHeight > 0)
                  .filter((el) =>
                    // Chromium's A0 scale can report a 3 to 6 px glyph-metric
                    // difference even when the line box and visible ink fit.
                    // Eight pixels is 2.1 mm on the 4494 px A0 audit canvas.
                    el.scrollWidth > el.clientWidth + 8 ||
                    el.scrollHeight > el.clientHeight + 8
                  )
                  .map((el) => ({
                    tag: el.tagName,
                    className: String(el.className || ''),
                    client: [el.clientWidth, el.clientHeight],
                    scroll: [el.scrollWidth, el.scrollHeight],
                  }));
                const pageWidth = document.body.scrollWidth;
                const pageHeight = document.body.scrollHeight;
                const outsidePage = [...document.querySelectorAll('body *')]
                  .filter((el) => {
                    const style = getComputedStyle(el);
                    if (
                      style.display === 'none' ||
                      style.visibility === 'hidden'
                    ) return false;
                    const rect = el.getBoundingClientRect();
                    return (
                      rect.left < -2 ||
                      rect.top < -2 ||
                      rect.right > pageWidth + 2 ||
                      rect.bottom > pageHeight + 2
                    );
                  })
                  .map((el) => {
                    const rect = el.getBoundingClientRect();
                    return {
                      tag: el.tagName,
                      className: String(el.className || ''),
                      rect: [rect.left, rect.top, rect.right, rect.bottom],
                    };
                  });
                return {
                  body: [pageWidth, pageHeight],
                  imagesIncomplete: [...document.images].filter(
                    (img) => !img.complete || img.naturalWidth === 0
                  ).length,
                  offenders,
                  outsidePage,
                };
            }"""
        )
        print(audit)
        if (
            audit["imagesIncomplete"]
            or audit["offenders"]
            or audit["outsidePage"]
        ):
            raise RuntimeError(
                "Poster DOM audit failed before PDF rendering: "
                f"{audit}"
            )
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf),
            width="1189mm",
            height="841mm",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        page.emulate_media(media="screen")
        page.screenshot(path=str(png), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
