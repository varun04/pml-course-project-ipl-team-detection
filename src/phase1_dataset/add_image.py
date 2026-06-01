"""T1.2 — Download helper for raw image collection.

Given a source URL and a franchise code, this script:
  1. Downloads the image into dataset/raw/<NN_FRANCHISE>/.
  2. Names it <FRANCHISE>_<NNN>.<ext> using the next free index.
  3. Validates that the image is at least 800x600 native; if not, the file
     is moved to dataset/_rejected/<franchise>/ and NOT logged as collected.
  4. Appends a row to that franchise's _sources.csv on success.

Usage:
    # one-off
    python src/phase1_dataset/add_image.py 01_CSK https://example.com/a.jpg --note "post-match"

    # batch from a text file (one URL per line, blank lines / # comments ignored)
    python src/phase1_dataset/add_image.py 01_CSK --batch urls_csk.txt

    # override the downloader name shown in _sources.csv (defaults to $USER)
    python src/phase1_dataset/add_image.py 01_CSK <url> --downloader Varun

The franchise code is the directory name under dataset/raw/, e.g. 01_CSK, 06_MI,
00_no_team.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import cv2
import requests

MIN_W, MIN_H = 800, 600
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_EXT = ".jpg"
TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; pml-ipl-dataset-collector/1.0)"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
REJECT_DIR = REPO_ROOT / "dataset" / "_rejected"

# Franchise dir name -> short prefix used in file names
PREFIX = {
    "00_no_team": "NOTEAM",
    "01_CSK": "CSK",
    "02_DC": "DC",
    "03_GT": "GT",
    "04_KKR": "KKR",
    "05_LSG": "LSG",
    "06_MI": "MI",
    "07_PBKS": "PBKS",
    "08_RR": "RR",
    "09_RCB": "RCB",
    "10_SRH": "SRH",
}


def next_index(franchise_dir: Path, prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})\.")
    highest = 0
    for p in franchise_dir.iterdir():
        m = pat.match(p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def pick_extension(url: str, content_type: str | None) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }
        if ct in mapping:
            return mapping[ct]
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in VALID_EXT:
        return ".jpg" if suffix == ".jpeg" else suffix
    return DEFAULT_EXT


def download_one(url: str, franchise: str, downloader: str, note: str) -> tuple[bool, str]:
    if franchise not in PREFIX:
        return False, f"unknown franchise '{franchise}' (expected one of: {', '.join(sorted(PREFIX))})"
    franchise_dir = RAW_DIR / franchise
    if not franchise_dir.is_dir():
        return False, f"missing dir {franchise_dir}"

    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        return False, f"download failed: {e}"

    ext = pick_extension(url, resp.headers.get("Content-Type"))
    prefix = PREFIX[franchise]
    idx = next_index(franchise_dir, prefix)
    filename = f"{prefix}_{idx:03d}{ext}"
    dest = franchise_dir / filename

    try:
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    except OSError as e:
        return False, f"write failed: {e}"

    img = cv2.imread(str(dest))
    if img is None:
        dest.unlink(missing_ok=True)
        return False, "unreadable after download (not a valid image)"
    h, w = img.shape[:2]
    if w < MIN_W or h < MIN_H:
        reject_dir = REJECT_DIR / franchise
        reject_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dest), reject_dir / filename)
        return False, f"too small: {w}x{h} (need >={MIN_W}x{MIN_H}) — moved to _rejected"

    csv_path = franchise_dir / "_sources.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filename", "source_url", "date_downloaded", "downloader", "notes"])
        writer.writerow([filename, url, date.today().isoformat(), downloader, note])

    return True, f"saved {filename} ({w}x{h})"


def iter_batch(path: Path):
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("franchise", help="franchise dir name, e.g. 01_CSK, 06_MI, 00_no_team")
    ap.add_argument("url", nargs="?", help="image URL (omit when using --batch)")
    ap.add_argument("--batch", type=Path, help="text file with one URL per line")
    ap.add_argument("--note", default="", help="free-text note recorded in _sources.csv")
    ap.add_argument("--downloader", default=os.environ.get("USER", "unknown"))
    args = ap.parse_args()

    if not (args.url or args.batch):
        ap.error("provide a URL positional arg or --batch <file>")
    if args.url and args.batch:
        ap.error("provide either a URL or --batch, not both")

    urls = [args.url] if args.url else list(iter_batch(args.batch))
    if not urls:
        print("no urls to process", file=sys.stderr)
        return 1

    ok_count = 0
    fail_count = 0
    for url in urls:
        ok, msg = download_one(url, args.franchise, args.downloader, args.note)
        marker = "OK " if ok else "FAIL"
        print(f"[{marker}] {url}\n       {msg}")
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    print("-" * 60)
    print(f"saved: {ok_count}    failed/rejected: {fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
