#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

IMG_SELECTOR = "img.oneAllImg[data-page][src]"

# The "load more pages" link you mentioned
NEXT_LINK_SELECTOR = 'a[title="Zobrazit další strany"], a[title="Zobrazit další strany"] *'


@dataclass(frozen=True)
class FlyerPage:
    page: int
    url: str


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "flyer"


def default_out_name(flyer_url: str) -> str:
    p = urlparse(flyer_url)
    parts = [x for x in p.path.split("/") if x]
    last = parts[-1] if parts else "flyer"
    return f"{slugify(last)}.pdf"


def fetch_html(session: requests.Session, url: str, timeout: int) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_pages_and_next(html: str, base_url: str) -> Tuple[List[FlyerPage], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    # pages
    imgs = soup.select(IMG_SELECTOR)
    pages: List[FlyerPage] = []
    for img in imgs:
        try:
            page = int(img.get("data-page", "").strip())
        except ValueError:
            continue
        src = img.get("src")
        if not src:
            continue
        pages.append(FlyerPage(page=page, url=urljoin(base_url, src)))

    # next link ("Zobrazit další strany")
    next_a = soup.find("a", attrs={"title": "Zobrazit další strany"})
    next_url = urljoin(base_url, next_a["href"]) if next_a and next_a.get("href") else None

    return pages, next_url


def crawl_all_pages(session: requests.Session, start_url: str, timeout: int, quiet: bool) -> List[FlyerPage]:
    # Collect into dict to dedupe by page number
    by_page: Dict[int, str] = {}

    visited = set()
    url = start_url

    while url and url not in visited:
        visited.add(url)
        if not quiet:
            print(f"[html] {url}")

        html = fetch_html(session, url, timeout)
        pages, next_url = parse_pages_and_next(html, url)

        for p in pages:
            # Keep first seen (or replace; either is fine since they should be identical)
            by_page.setdefault(p.page, p.url)

        url = next_url

    result = [FlyerPage(page=k, url=v) for k, v in sorted(by_page.items(), key=lambda kv: kv[0])]
    return result


def download_pages(
    session: requests.Session,
    pages: List[FlyerPage],
    out_dir: Path,
    timeout: int,
    overwrite: bool,
    quiet: bool,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for fp in pages:
        ext = os.path.splitext(urlparse(fp.url).path)[1].lower() or ".jpg"
        fn = out_dir / f"page_{fp.page:03d}{ext}"

        if fn.exists() and not overwrite:
            files.append(fn)
            if not quiet:
                print(f"[skip] {fn.name}")
            continue

        if not quiet:
            print(f"[get ] page {fp.page:>3}: {fp.url}")

        r = session.get(fp.url, timeout=timeout)
        r.raise_for_status()
        fn.write_bytes(r.content)
        files.append(fn)

    return files


def build_pdf_img2pdf(image_files: List[Path], out_pdf: Path) -> None:
    import img2pdf  # type: ignore

    paths = [str(p) for p in image_files]
    out_pdf.write_bytes(img2pdf.convert(paths))


def build_pdf_pillow(image_files: List[Path], out_pdf: Path) -> None:
    from PIL import Image  # type: ignore

    images = []
    for p in image_files:
        im = Image.open(p)
        if im.mode != "RGB":
            im = im.convert("RGB")
        images.append(im)

    if not images:
        raise RuntimeError("No images to write to PDF.")

    images[0].save(str(out_pdf), save_all=True, append_images=images[1:])


def process_flyer(
    session: requests.Session,
    url: str,
    out_pdf: Path,
    out_dir: Path,
    timeout: int,
    overwrite: bool,
    engine: str,
    quiet: bool,
) -> int:
    try:
        pages = crawl_all_pages(session, url, timeout, quiet)
    except Exception as e:
        print(f"ERROR: failed while crawling flyer pages for {url}: {e}", file=sys.stderr)
        return 2

    if not pages:
        print(
            f"ERROR: no pages found for {url}. Site structure may have changed.\n"
            f"Expected images matching: {IMG_SELECTOR}",
            file=sys.stderr,
        )
        return 3

    if not quiet:
        print(f"Found {len(pages)} unique pages. Downloading into: {out_dir}")

    try:
        files = download_pages(
            session=session,
            pages=pages,
            out_dir=out_dir,
            timeout=timeout,
            overwrite=overwrite,
            quiet=quiet,
        )
    except Exception as e:
        print(f"ERROR: failed to download images for {url}: {e}", file=sys.stderr)
        return 4

    # Order is guaranteed by pages list; sort files defensively anyway:
    files_sorted = sorted(files, key=lambda p: p.name)

    try:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        if engine == "img2pdf":
            build_pdf_img2pdf(files_sorted, out_pdf)
        else:
            build_pdf_pillow(files_sorted, out_pdf)
    except ModuleNotFoundError as e:
        print(
            f"ERROR: missing dependency for engine '{engine}': {e}\n"
            "Install with:\n"
            "  python -m pip install img2pdf\n"
            "or\n"
            "  python -m pip install pillow",
            file=sys.stderr,
        )
        return 5
    except Exception as e:
        print(f"ERROR: failed to build PDF for {url}: {e}", file=sys.stderr)
        return 6

    try:
        shutil.rmtree(out_dir)
    except Exception as e:
        print(f"WARNING: wrote PDF, but failed to remove downloaded images directory {out_dir}: {e}", file=sys.stderr)

    if not quiet:
        print(f"OK: wrote {out_pdf} ({len(files_sorted)} pages)")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download akcneletaky.sk flyer (including hidden pages via 'Zobrazit další strany') and merge to PDF."
    )
    ap.add_argument("urls", nargs="+", help="One or more flyer URLs, e.g. https://lidl.akcneletaky.sk/lidl-letak-od-2-1_25428/")
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output PDF path for one URL, or output directory for multiple URLs (default: derived from each URL)",
    )
    ap.add_argument(
        "-d",
        "--dir",
        default=None,
        help="Directory for downloaded images with one URL (default: ./<outstem>_images)",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default: 60)")
    ap.add_argument("--ua", default=DEFAULT_UA, help="User-Agent string")
    ap.add_argument("--overwrite", action="store_true", help="Re-download images even if present")
    ap.add_argument("--engine", choices=["img2pdf", "pillow"], default="img2pdf",
                    help="PDF engine: img2pdf (best quality) or pillow (fallback). Default: img2pdf")
    ap.add_argument("--quiet", action="store_true", help="Less output")
    args = ap.parse_args()

    if len(args.urls) > 1 and args.dir:
        print("ERROR: --dir can only be used with a single URL.", file=sys.stderr)
        return 1

    out_arg = Path(args.out) if args.out else None
    if len(args.urls) > 1 and out_arg and out_arg.suffix.lower() == ".pdf":
        print("ERROR: --out must be a directory when using multiple URLs.", file=sys.stderr)
        return 1

    out_dir_base = out_arg if len(args.urls) > 1 and out_arg else None

    s = requests.Session()
    s.headers.update({"User-Agent": args.ua})

    exit_code = 0
    for url in args.urls:
        default_pdf = Path(default_out_name(url))
        if len(args.urls) == 1:
            out_pdf = out_arg if out_arg else default_pdf
            image_dir = Path(args.dir) if args.dir else Path(f"{out_pdf.stem}_images")
        else:
            out_pdf = (out_dir_base / default_pdf.name) if out_dir_base else default_pdf
            image_dir = out_pdf.with_name(f"{out_pdf.stem}_images")

        if not args.quiet and len(args.urls) > 1:
            print(f"\n=== {url} ===")

        result = process_flyer(
            session=s,
            url=url,
            out_pdf=out_pdf,
            out_dir=image_dir,
            timeout=args.timeout,
            overwrite=args.overwrite,
            engine=args.engine,
            quiet=args.quiet,
        )
        if result != 0:
            exit_code = result

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
