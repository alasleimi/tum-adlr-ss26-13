"""Render the HTML workshop deck to a timing-stable 16:9 PDF.

Run from any directory with::

    python scripts/render_presentation_pdf.py

The HTML remains an interactive deck. For the static PDF, animated GIFs are
frozen on their first frame so the visual result does not depend on timing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = PROJECT_ROOT / "presentation" / "index.html"
DEFAULT_PDF = (
    PROJECT_ROOT
    / "presentation"
    / "week3_workshop_presentation_20260527.pdf"
)
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def resolve_edge(explicit: Path | None) -> Path:
    """Return an installed Edge executable or raise a useful error."""

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
    """Wait for document, web fonts, images, and MathJax to be complete."""

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

    broken_images = page.evaluate(
        """() => [...document.images]
            .filter((image) => !image.complete || image.naturalWidth === 0)
            .map((image) => image.getAttribute('src'))"""
    )
    if broken_images:
        raise RuntimeError(f"Images failed to load: {broken_images}")

    if page.locator("script[src*='mathjax']").count():
        page.wait_for_function(
            """() => Boolean(
                window.MathJax &&
                window.MathJax.startup &&
                window.MathJax.startup.promise
            )""",
            timeout=timeout_ms,
        )
        page.evaluate(
            """() => {
                window.__project15MathJaxReady = false;
                window.__project15MathJaxError = null;
                window.MathJax.startup.promise.then(
                    () => { window.__project15MathJaxReady = true; },
                    (error) => {
                        window.__project15MathJaxError = String(error);
                    }
                );
            }"""
        )
        page.wait_for_function(
            """() => (
                window.__project15MathJaxReady ||
                window.__project15MathJaxError
            )""",
            timeout=timeout_ms,
        )
        mathjax_error = page.evaluate(
            "() => window.__project15MathJaxError"
        )
        if mathjax_error:
            raise RuntimeError(f"MathJax failed: {mathjax_error}")
        mathjax_count = page.locator("mjx-container").count()
        if mathjax_count == 0:
            raise RuntimeError("MathJax loaded but produced no rendered equations")
        # MathJax injects its CHTML webfonts during startup.  The earlier
        # document.fonts wait can therefore finish before those font faces
        # even exist; wait a second time after typesetting.
        page.evaluate(
            """async () => {
                if (document.fonts) await document.fonts.ready;
            }"""
        )
        page.wait_for_function(
            "() => !document.fonts || document.fonts.status === 'loaded'",
            timeout=timeout_ms,
        )
        incomplete_math = page.evaluate(
            """() => [...document.querySelectorAll('mjx-container')]
                .map((container, index) => ({
                    index,
                    glyphs: container.querySelectorAll('mjx-c, mjx-char').length,
                    width: container.getBoundingClientRect().width,
                    height: container.getBoundingClientRect().height,
                }))
                // Equation slides are intentionally hidden in screen mode,
                // so geometry is checked later after print media is enabled.
                .filter((item) => item.glyphs === 0)"""
        )
        if incomplete_math:
            raise RuntimeError(
                f"MathJax output is incomplete after font loading: {incomplete_math}"
            )


def freeze_gifs(page: Page, html: Path, timeout_ms: int) -> list[str]:
    """Replace local GIFs with deterministic first-frame PNG data URLs."""

    gif_images = page.locator("img[src$='.gif']")
    frozen: list[str] = []
    if gif_images.count() == 0:
        return frozen

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Pillow is required to freeze animated GIFs for deterministic PDF output"
        ) from error

    for index in range(gif_images.count()):
        image = gif_images.nth(index)
        source = image.get_attribute("src")
        if not source:
            continue
        gif_path = (html.parent / source).resolve()
        if not gif_path.is_file():
            raise FileNotFoundError(f"GIF referenced by the deck is missing: {gif_path}")
        with Image.open(gif_path) as animation:
            animation.seek(0)
            frame = animation.convert("RGBA")
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG", optimize=True)
        data_url = "data:image/png;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
        image.evaluate("(node, replacement) => { node.src = replacement; }", data_url)
        frozen.append(source)

    page.wait_for_function(
        """() => [...document.images].every(
            (image) => image.complete && image.naturalWidth > 0
        )""",
        timeout=timeout_ms,
    )
    return frozen


def audit_print_layout(page: Page, timeout_ms: int) -> dict[str, object]:
    """Return print-layout dimensions and fail on clipped slide content."""

    page.emulate_media(media="print")
    # Equations live on slides that are hidden in screen mode.  Switching to
    # print makes those glyphs visible and only then triggers MathJax's font
    # downloads, so this wait must happen after emulate_media().
    page.evaluate(
        """async () => {
            if (document.fonts) await document.fonts.ready;
        }"""
    )
    page.wait_for_function(
        "() => !document.fonts || document.fonts.status === 'loaded'",
        timeout=timeout_ms,
    )
    audit = page.evaluate(
        """() => {
            const slides = [...document.querySelectorAll('.slide')];
            const geometry = slides.map((slide, index) => {
                const rect = slide.getBoundingClientRect();
                return {
                    slide: index + 1,
                    width: rect.width,
                    height: rect.height,
                    overflowX: Math.max(0, slide.scrollWidth - slide.clientWidth),
                    overflowY: Math.max(0, slide.scrollHeight - slide.clientHeight),
                };
            });
            return {
                slideCount: slides.length,
                geometry,
                mathGeometry: [...document.querySelectorAll('mjx-container')]
                    .map((container, index) => {
                        const rect = container.getBoundingClientRect();
                        return { index, width: rect.width, height: rect.height };
                    }),
                incompleteImages: [...document.images]
                    .filter((image) => !image.complete || image.naturalWidth === 0)
                    .map((image) => image.getAttribute('src')),
            };
        }"""
    )
    if audit["slideCount"] != 18:
        raise RuntimeError(f"Expected 18 slides, found {audit['slideCount']}")
    if audit["incompleteImages"]:
        raise RuntimeError(f"Incomplete images at print time: {audit}")
    hidden_math = [
        item
        for item in audit["mathGeometry"]
        if item["width"] <= 0 or item["height"] <= 0
    ]
    if hidden_math:
        raise RuntimeError(f"MathJax output is hidden in print layout: {hidden_math}")
    wrong_size = [
        item
        for item in audit["geometry"]
        if abs(item["width"] - 1280) > 0.1
        or abs(item["height"] - 720) > 0.1
    ]
    if wrong_size:
        raise RuntimeError(f"Unexpected print slide geometry: {wrong_size}")
    overflow = [
        item
        for item in audit["geometry"]
        if item["overflowX"] > 1 or item["overflowY"] > 1
    ]
    if overflow:
        raise RuntimeError(f"Slide content overflows its page: {overflow}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--edge", type=Path)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument(
        "--keep-animated-gifs",
        action="store_true",
        help="Capture GIFs at their current frame instead of freezing frame one.",
    )
    args = parser.parse_args()

    html = args.html.expanduser().resolve()
    pdf = args.pdf.expanduser().resolve()
    edge = resolve_edge(args.edge)
    if not html.is_file():
        raise FileNotFoundError(f"Presentation HTML not found: {html}")
    pdf.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(edge),
            headless=True,
        )
        try:
            page = browser.new_page(
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
            )
            page.goto(
                html.as_uri(),
                wait_until="networkidle",
                timeout=args.timeout_ms,
            )
            wait_for_page_assets(page, args.timeout_ms)
            frozen_gifs = (
                []
                if args.keep_animated_gifs
                else freeze_gifs(page, html, args.timeout_ms)
            )
            audit = audit_print_layout(page, args.timeout_ms)
            page.pdf(
                path=str(pdf),
                width="13.333333in",
                height="7.5in",
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
        finally:
            browser.close()

    print(
        json.dumps(
            {
                "html": str(html),
                "pdf": str(pdf),
                "edge": str(edge),
                "slides": audit["slideCount"],
                "frozen_gifs": frozen_gifs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
