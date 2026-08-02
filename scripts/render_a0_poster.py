"""Render the canonical v115 HTML poster to an A0-landscape PDF.

The checked-in PNG is intentionally left untouched unless ``--png`` is passed.
Run from any directory with::

    python scripts/render_a0_poster.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = PROJECT_ROOT / "poster" / "poster_visual_v115.html"
DEFAULT_PDF = PROJECT_ROOT / "poster" / "poster_visual_v115.pdf"
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def resolve_edge(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Edge executable not found: {candidate}")
        return candidate
    discovered = shutil.which("msedge")
    if discovered:
        return Path(discovered).resolve()
    for candidate in EDGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Microsoft Edge was not found. Pass its path with --edge."
    )


def wait_for_page_assets(page: Page, timeout_ms: int) -> None:
    page.wait_for_load_state("load", timeout=timeout_ms)
    page.wait_for_function(
        "() => document.readyState === 'complete'", timeout=timeout_ms
    )
    page.evaluate(
        """async () => {
            if (document.fonts) await document.fonts.ready;
        }"""
    )
    page.wait_for_function(
        "() => !document.fonts || document.fonts.status === 'loaded'",
        timeout=timeout_ms,
    )
    page.wait_for_function(
        """() => [...document.images].every(
            (image) => image.complete && image.naturalWidth > 0
        )""",
        timeout=timeout_ms,
    )


def audit_poster(page: Page) -> dict[str, object]:
    """Catch missing assets and visible overflow before producing the PDF."""

    audit = page.evaluate(
        """() => {
            const pageWidth = document.body.scrollWidth;
            const pageHeight = document.body.scrollHeight;
            const offenders = [...document.querySelectorAll('*')]
                .filter((element) => !['HTML', 'BODY'].includes(element.tagName))
                .filter((element) =>
                    element.namespaceURI !== 'http://www.w3.org/1998/Math/MathML'
                )
                .filter((element) => element.clientWidth > 0 && element.clientHeight > 0)
                .filter((element) =>
                    element.scrollWidth > element.clientWidth + 12 ||
                    element.scrollHeight > element.clientHeight + 12
                )
                .map((element) => ({
                    tag: element.tagName,
                    className: String(element.className || ''),
                    client: [element.clientWidth, element.clientHeight],
                    scroll: [element.scrollWidth, element.scrollHeight],
                }));
            const outsidePage = [...document.querySelectorAll('body *')]
                .filter((element) => {
                    const style = getComputedStyle(element);
                    if (style.display === 'none' || style.visibility === 'hidden') {
                        return false;
                    }
                    const rect = element.getBoundingClientRect();
                    return (
                        rect.left < -2 || rect.top < -2 ||
                        rect.right > pageWidth + 2 || rect.bottom > pageHeight + 2
                    );
                })
                .map((element) => {
                    const rect = element.getBoundingClientRect();
                    return {
                        tag: element.tagName,
                        className: String(element.className || ''),
                        rect: [rect.left, rect.top, rect.right, rect.bottom],
                    };
                });
            return {
                body: [pageWidth, pageHeight],
                incompleteImages: [...document.images]
                    .filter((image) => !image.complete || image.naturalWidth === 0)
                    .map((image) => image.getAttribute('src')),
                offenders,
                outsidePage,
            };
        }"""
    )
    if audit["incompleteImages"] or audit["offenders"] or audit["outsidePage"]:
        raise RuntimeError(f"Poster DOM audit failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--png",
        type=Path,
        help="Optional screenshot path. Omit to preserve the checked-in PNG.",
    )
    parser.add_argument("--edge", type=Path)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    args = parser.parse_args()

    html = args.html.expanduser().resolve()
    pdf = args.pdf.expanduser().resolve()
    png = args.png.expanduser().resolve() if args.png else None
    edge = resolve_edge(args.edge)
    if not html.is_file():
        raise FileNotFoundError(f"Poster HTML not found: {html}")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if png:
        png.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(edge),
            headless=True,
        )
        try:
            page = browser.new_page(
                viewport={"width": 1600, "height": 1132},
                device_scale_factor=1,
            )
            page.goto(
                html.as_uri(),
                wait_until="networkidle",
                timeout=args.timeout_ms,
            )
            wait_for_page_assets(page, args.timeout_ms)
            audit = audit_poster(page)
            page.emulate_media(media="print")
            page.pdf(
                path=str(pdf),
                width="1189mm",
                height="841mm",
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            if png:
                page.emulate_media(media="screen")
                page.screenshot(path=str(png), full_page=True)
        finally:
            browser.close()

    print(
        json.dumps(
            {
                "html": str(html),
                "pdf": str(pdf),
                "png": str(png) if png else None,
                "edge": str(edge),
                "audit": audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
