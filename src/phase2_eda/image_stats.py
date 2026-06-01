"""T2.1 — Per-image statistics.

Walks dataset/raw/<match>/ and for every image cross-referenced with
dataset/labels.csv produces one row in dataset/image_stats.csv with:

  image            file name
  match            match-name bucket (folder under dataset/raw/)
  width, height    actual pixel dimensions
  channels         3 = colour, 1 = grayscale
  sha256           content hash (catches duplicates across groups)
  bytes            file size on disk
  non_zero_cells   count of cells in [1..10]
  dominant_label   the most frequent non-zero label, or 0 if all-empty
  labels_present   sorted comma-separated list, e.g. "2,10"

This file is the spine of the EDA notebook — most plots filter it.

Usage:
    python src/phase2_eda/image_stats.py
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
LABELS = REPO_ROOT / "dataset" / "labels.csv"
OUT = REPO_ROOT / "dataset" / "image_stats.csv"

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]


def load_labels() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with LABELS.open() as f:
        for row in csv.DictReader(f):
            out[row["image"]] = [int(row[c]) for c in CELL_COLS]
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    labels = load_labels()
    rows: list[dict] = []
    skipped_no_label = 0
    skipped_unreadable = 0

    for match_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for img_path in sorted(match_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXT:
                continue
            if img_path.name not in labels:
                skipped_no_label += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_unreadable += 1
                continue
            h_, w_ = img.shape[:2]
            ch = 1 if img.ndim == 2 else img.shape[2]
            cells = labels[img_path.name]
            non_zero = sum(1 for v in cells if v != 0)
            present = sorted(set(v for v in cells if v != 0))
            counts = Counter(v for v in cells if v != 0)
            dominant = counts.most_common(1)[0][0] if counts else 0
            rows.append({
                "image": img_path.name,
                "match": match_dir.name,
                "width": w_,
                "height": h_,
                "channels": ch,
                "sha256": sha256_of(img_path),
                "bytes": img_path.stat().st_size,
                "non_zero_cells": non_zero,
                "dominant_label": dominant,
                "labels_present": ",".join(str(x) for x in present),
            })

    rows.sort(key=lambda r: r["image"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_off_size = sum(1 for r in rows if r["width"] != 800 or r["height"] != 600)
    sha_counts = Counter(r["sha256"] for r in rows)
    dup_hashes = {h for h, n in sha_counts.items() if n > 1}

    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"  rows                 : {len(rows)}")
    print(f"  skipped (no label)   : {skipped_no_label}")
    print(f"  skipped (unreadable) : {skipped_unreadable}")
    print(f"  not 800x600          : {n_off_size}")
    print(f"  duplicate sha256     : {len(dup_hashes)} hash(es) cover {sum(sha_counts[h] for h in dup_hashes)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
