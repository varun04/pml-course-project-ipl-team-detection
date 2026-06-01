"""T3.2 — Walk dataset/processed/, extract per-cell features, save .npy arrays.

For every (image, cell) pair where we have BOTH an image on disk AND a label
row in dataset/labels.csv, this produces:

    dataset/features/X.npy           (N_cells, 67)   float32  — feature vectors
    dataset/features/y.npy           (N_cells,)      int8     — labels 0..10
    dataset/features/image_idx.npy   (N_cells,)      int32    — index into image_names
    dataset/features/cell_idx.npy    (N_cells,)      int8     — 0..63
    dataset/features/image_names.npy (N_images,)     str      — filenames
    dataset/features/split.npy       (N_images,)     int8     — 0 = train, 1 = test
    dataset/features/meta.json                                — feature_names, sizes, etc.

Split is **image-level** with 90/10 ratio (seed=42). Per-cell records inherit
the split of their parent image. This prevents leaking cells from the same
image between train and test (which would inflate accuracy).

Usage:
    python src/phase3/build_dataset.py
    python src/phase3/build_dataset.py --limit 200   # for quick iteration
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC = REPO_ROOT / "dataset" / "processed"
LABELS = REPO_ROOT / "dataset" / "labels.csv"
OUT_DIR = REPO_ROOT / "dataset" / "features"

sys.path.insert(0, str(REPO_ROOT / "src"))
from phase3.extract import extract_image_features, FEATURE_NAMES, FEATURE_DIM, N_CELLS

CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]
TEST_FRAC = 0.10
SEED = 42


def load_label_lookup() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with LABELS.open() as f:
        for row in csv.DictReader(f):
            out[row["image"]] = [int(row[c]) for c in CELL_COLS]
    return out


def discover_images() -> list[Path]:
    out: list[Path] = []
    for sub in sorted(PROC.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        out.extend(sorted(p for p in sub.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="process only this many images (debug)")
    args = ap.parse_args()

    label_lookup = load_label_lookup()
    image_paths = discover_images()
    image_paths = [p for p in image_paths if p.name in label_lookup]
    if args.limit:
        image_paths = image_paths[: args.limit]

    n_images = len(image_paths)
    n_cells = n_images * N_CELLS
    print(f"images to process    : {n_images}")
    print(f"cells (rows)         : {n_cells:,}")
    print(f"feature dim          : {FEATURE_DIM}")
    print(f"memory for X         : ~{n_cells * FEATURE_DIM * 4 / 1024 / 1024:.1f} MB float32")

    X = np.empty((n_cells, FEATURE_DIM), dtype=np.float32)
    y = np.empty(n_cells, dtype=np.int8)
    image_idx = np.empty(n_cells, dtype=np.int32)
    cell_idx = np.empty(n_cells, dtype=np.int8)
    image_names: list[str] = []

    t0 = time.time()
    cursor = 0
    for img_id, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None or img.shape[:2] != (600, 800):
            print(f"  skip {path.name} (unreadable or wrong size)")
            continue
        feats = extract_image_features(img)
        lbls = np.array(label_lookup[path.name], dtype=np.int8)
        X[cursor:cursor + N_CELLS] = feats
        y[cursor:cursor + N_CELLS] = lbls
        image_idx[cursor:cursor + N_CELLS] = img_id
        cell_idx[cursor:cursor + N_CELLS] = np.arange(N_CELLS, dtype=np.int8)
        cursor += N_CELLS
        image_names.append(path.name)
        if (img_id + 1) % 250 == 0 or img_id == n_images - 1:
            elapsed = time.time() - t0
            rate = (img_id + 1) / elapsed
            eta = (n_images - img_id - 1) / rate if rate > 0 else 0
            print(f"  {img_id + 1}/{n_images}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left, {rate:.1f} img/s)")

    # trim arrays to actual cursor (handles skipped images)
    X = X[:cursor]; y = y[:cursor]; image_idx = image_idx[:cursor]; cell_idx = cell_idx[:cursor]
    n_images_kept = len(image_names)

    # image-level 90/10 split (deterministic)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(n_images_kept)
    n_test = int(round(n_images_kept * TEST_FRAC))
    test_image_ids = set(shuffled[:n_test].tolist())
    split = np.zeros(n_images_kept, dtype=np.int8)
    for i in test_image_ids:
        split[i] = 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "image_idx.npy", image_idx)
    np.save(OUT_DIR / "cell_idx.npy", cell_idx)
    np.save(OUT_DIR / "image_names.npy", np.array(image_names))
    np.save(OUT_DIR / "split.npy", split)

    meta = {
        "n_images": n_images_kept,
        "n_cells": int(cursor),
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "test_frac": TEST_FRAC,
        "seed": SEED,
        "n_train_images": int((split == 0).sum()),
        "n_test_images": int((split == 1).sum()),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    n_train_cells = int((split[image_idx] == 0).sum())
    n_test_cells = int((split[image_idx] == 1).sum())
    elapsed = time.time() - t0
    print()
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/  (X, y, image_idx, cell_idx, image_names, split, meta.json)")
    print(f"images kept   : {n_images_kept}  (train {meta['n_train_images']} / test {meta['n_test_images']})")
    print(f"cells         : {cursor:,}        (train {n_train_cells:,} / test {n_test_cells:,})")
    print(f"X shape       : {X.shape}")
    print(f"y class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"elapsed       : {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
