"""T1.4 — Produce 800x600 dataset.

Reads every image under dataset/raw/<franchise>/, centre-crops it to 4:3, and
downsizes to exactly 800x600. The processed file is written to
dataset/processed/<franchise>/ as JPEG. Originals are left untouched.

Native resolution < 800x600 is skipped (spec forbids upscaling). Run
validate_images.py first to move those out of the way.

Usage:
    python src/phase1_dataset/resize_images.py
    python src/phase1_dataset/resize_images.py --overwrite   # redo existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

TARGET_W, TARGET_H = 800, 600
TARGET_RATIO = TARGET_W / TARGET_H  # 4:3 == 1.333...
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
JPEG_QUALITY = 95

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
PROC_DIR = REPO_ROOT / "dataset" / "processed"


def centre_crop_to_4x3(img):
    """Crop to 4:3 around the centre. Returns the cropped view."""
    h, w = img.shape[:2]
    cur_ratio = w / h
    if abs(cur_ratio - TARGET_RATIO) < 1e-3:
        return img
    if cur_ratio > TARGET_RATIO:
        # too wide — trim left/right
        new_w = int(round(h * TARGET_RATIO))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    # too tall — trim top/bottom
    new_h = int(round(w / TARGET_RATIO))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def process_one(src: Path, dest: Path, overwrite: bool) -> str:
    if dest.exists() and not overwrite:
        return "skip_exists"
    img = cv2.imread(str(src))
    if img is None:
        return "unreadable"
    h, w = img.shape[:2]
    if w < TARGET_W or h < TARGET_H:
        return f"too_small_{w}x{h}"
    cropped = centre_crop_to_4x3(img)
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return "written" if ok else "write_failed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true", help="Re-process even if output already exists")
    args = ap.parse_args()

    if not RAW_DIR.exists():
        print(f"raw dir missing: {RAW_DIR}", file=sys.stderr)
        return 1

    counts: dict[str, int] = {"written": 0, "skip_exists": 0, "unreadable": 0, "write_failed": 0}
    per_franchise: dict[str, int] = {}

    for franchise_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for img_path in sorted(franchise_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXT:
                continue
            dest = PROC_DIR / franchise_dir.name / (img_path.stem + ".jpg")
            outcome = process_one(img_path, dest, args.overwrite)
            if outcome.startswith("too_small"):
                counts.setdefault("too_small", 0)
                counts["too_small"] += 1
            else:
                counts[outcome] = counts.get(outcome, 0) + 1
            if outcome in ("written", "skip_exists"):
                per_franchise[franchise_dir.name] = per_franchise.get(franchise_dir.name, 0) + 1

    print("=" * 60)
    print("Resize summary")
    for k, v in counts.items():
        print(f"  {k:<16} {v}")
    print("-" * 60)
    print(f"{'franchise':<14}{'in_processed':>14}")
    for franchise in sorted(per_franchise):
        print(f"{franchise:<14}{per_franchise[franchise]:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
