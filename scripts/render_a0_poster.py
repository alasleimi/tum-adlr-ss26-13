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
                  .filter((el) => el.clientWidth > 0 && el.clientHeight > 0)
                  .filter((el) =>
                    el.scrollWidth > el.clientWidth + 2 ||
                    el.scrollHeight > el.clientHeight + 2
                  )
                  .map((el) => ({
                    tag: el.tagName,
                    className: String(el.className || ''),
                    client: [el.clientWidth, el.clientHeight],
                    scroll: [el.scrollWidth, el.scrollHeight],
                  }));
                return {
                  body: [document.body.scrollWidth, document.body.scrollHeight],
                  imagesIncomplete: [...document.images].filter((img) => !img.complete).length,
                  offenders,
                };
            }"""
        )
        print(audit)
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
