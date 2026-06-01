"""T1.6 — Ingest Group 19's pre-labelled SRHvsDC dataset.

Source layout (on local disk):
  /Users/varuntomar/Downloads/Group_19/
      Group_19_TJ.csv                              # labels CSV
      SRHvsDC_labelled_only/SRHvsDC_image_*.jpg    # 158 images, all native 800x600

Target layout (inside the repo):
  dataset/raw/SRHvsDC/SRHvsDC_image_*.jpg          # copies of the 158 images
  dataset/raw/SRHvsDC/_sources.csv                 # provenance row per image
  dataset/labels.csv                               # normalized 8x8 grid labels for the
                                                     whole project (image, c01..c64)

The source CSV deviates from the project's required submission schema:
  - line 1 is a sentinel marker ("Group_19 v1")
  - real header on line 2: image, labelled_by, num_players, c01..c64
We strip the sentinel + the two non-required columns and write a clean
per-image labels CSV. Inconsistent labeller names ("Tushita_J" vs "Tushita J")
are recorded only as provenance in _sources.csv; they don't reach labels.csv.

Mismatches between the labels CSV and the image folder (rows without files,
files without rows) are reported but do NOT abort the run — they're logged
into dataset/labels_import_warnings.txt for review.

Usage:
    python src/phase1_dataset/import_group19.py             # dry-run report
    python src/phase1_dataset/import_group19.py --apply     # actually copy + write
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path("/Users/varuntomar/Downloads/Group_19")
SRC_IMG_DIR = SRC_DIR / "SRHvsDC_labelled_only"
SRC_CSV = SRC_DIR / "Group_19_TJ.csv"

MATCH_NAME = "SRHvsDC"
DEST_RAW_DIR = REPO_ROOT / "dataset" / "raw" / MATCH_NAME
DEST_LABELS = REPO_ROOT / "dataset" / "labels.csv"
DEST_WARNINGS = REPO_ROOT / "dataset" / "labels_import_warnings.txt"
DEST_SOURCES = DEST_RAW_DIR / "_sources.csv"

CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]
LABELS_HEADER = ["image", *CELL_COLS]
SOURCES_HEADER = ["filename", "source", "date_imported", "downloader", "notes"]


def read_friend_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(path.open()))
    if not rows:
        raise SystemExit(f"empty csv: {path}")
    # line 1 might be a sentinel like "Group_19 v1"
    if rows[0][0].lower().startswith("group_") and "image" not in rows[0]:
        sentinel = rows[0]
        header = rows[1]
        data = rows[2:]
    else:
        sentinel = []
        header = rows[0]
        data = rows[1:]
    expected = {"image", *CELL_COLS}
    if not expected.issubset(set(header)):
        missing = expected - set(header)
        raise SystemExit(f"header missing columns: {sorted(missing)}")
    return header, data


def normalize_rows(header: list[str], data: list[list[str]]) -> tuple[list[dict], list[str]]:
    """Return (normalized rows, list of warning strings)."""
    idx = {c: i for i, c in enumerate(header)}
    out: list[dict] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    for line_no, row in enumerate(data, start=3):
        if not row or not any(v.strip() for v in row):
            continue
        image = row[idx["image"]].strip()
        if not image:
            warnings.append(f"line {line_no}: blank image name — skipped")
            continue
        if image in seen:
            warnings.append(
                f"line {line_no}: duplicate image '{image}' (first seen at line {seen[image]}) — kept first, skipped this"
            )
            continue
        seen[image] = line_no
        cells = []
        bad_cell = False
        for col in CELL_COLS:
            v = row[idx[col]].strip()
            if v == "":
                v = "0"
            if not v.isdigit() or not 0 <= int(v) <= 10:
                warnings.append(f"line {line_no}: bad cell value '{v}' in {col} for {image} — replaced with 0")
                v = "0"
                bad_cell = True
            cells.append(v)
        rec = {"image": image, "cells": cells, "labelled_by": row[idx.get("labelled_by", -1)].strip() if "labelled_by" in idx else ""}
        out.append(rec)
    return out, warnings


def cross_check_with_images(records: list[dict], img_dir: Path) -> tuple[list[dict], list[str]]:
    """Drop label rows whose image file is missing; warn for orphan images."""
    warnings: list[str] = []
    available = {p.name for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
    labelled = {r["image"] for r in records}
    missing_files = labelled - available
    orphan_files = available - labelled
    kept = [r for r in records if r["image"] in available]
    for name in sorted(missing_files):
        warnings.append(f"label row references missing file: {name} — dropped from labels.csv")
    for name in sorted(orphan_files):
        warnings.append(f"image file has no label row: {name} — copied to raw/ but not in labels.csv")
    return kept, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually copy files and write CSVs (default: dry run)")
    ap.add_argument("--downloader", default="Varun", help="value recorded in _sources.csv")
    args = ap.parse_args()

    if not SRC_IMG_DIR.is_dir():
        print(f"missing source dir: {SRC_IMG_DIR}", file=sys.stderr)
        return 1
    if not SRC_CSV.is_file():
        print(f"missing source csv: {SRC_CSV}", file=sys.stderr)
        return 1

    header, data = read_friend_csv(SRC_CSV)
    records, csv_warnings = normalize_rows(header, data)
    records, file_warnings = cross_check_with_images(records, SRC_IMG_DIR)
    warnings = csv_warnings + file_warnings

    available_files = sorted(p for p in SRC_IMG_DIR.iterdir() if p.suffix.lower() == ".jpg")
    print(f"source images   : {len(available_files)}")
    print(f"label rows kept : {len(records)}")
    print(f"warnings        : {len(warnings)}")
    if warnings:
        for w in warnings[:8]:
            print(f"  - {w}")
        if len(warnings) > 8:
            print(f"  ... ({len(warnings) - 8} more, full list will be written to {DEST_WARNINGS.relative_to(REPO_ROOT)} on --apply)")

    if not args.apply:
        print("\n(dry run) re-run with --apply to copy files and write CSVs")
        return 0

    DEST_RAW_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in available_files:
        dest = DEST_RAW_DIR / p.name
        if dest.exists() and dest.stat().st_size == p.stat().st_size:
            continue
        shutil.copy2(p, dest)
        copied += 1

    with DEST_SOURCES.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SOURCES_HEADER)
        for p in available_files:
            note = f"Group 19 SRHvsDC IPL 2025; labelled_by varies (Tushita_J / Arijit)"
            w.writerow([p.name, "Group_19_TJ.csv (Tushita_J / Arijit)", date.today().isoformat(), args.downloader, note])

    with DEST_LABELS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABELS_HEADER)
        for r in records:
            w.writerow([r["image"], *r["cells"]])

    with DEST_WARNINGS.open("w") as f:
        for line in warnings:
            f.write(line + "\n")

    print(f"\ncopied images        : {copied} (skipped {len(available_files) - copied} already present)")
    print(f"wrote labels.csv     : {DEST_LABELS.relative_to(REPO_ROOT)} ({len(records)} rows)")
    print(f"wrote sources csv    : {DEST_SOURCES.relative_to(REPO_ROOT)}")
    print(f"wrote warnings log   : {DEST_WARNINGS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
