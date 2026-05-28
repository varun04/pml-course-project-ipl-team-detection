"""T1.3 — Validate raw images.

Walks dataset/raw/<franchise>/ and checks every image. The spec says only
images whose native resolution is at least 800x600 are usable (no upscaling).
Anything that fails is moved to dataset/_rejected/<franchise>/ with a reason
appended to dataset/_rejected/_rejection_log.csv so it can be reviewed.

Usage:
    python src/phase1_dataset/validate_images.py            # report only
    python src/phase1_dataset/validate_images.py --apply    # also move rejects
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

MIN_W, MIN_H = 800, 600
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
REJECT_DIR = REPO_ROOT / "dataset" / "_rejected"
LOG_PATH = REJECT_DIR / "_rejection_log.csv"


def check(path: Path) -> tuple[bool, str]:
    img = cv2.imread(str(path))
    if img is None:
        return False, "unreadable"
    h, w = img.shape[:2]
    if w < MIN_W or h < MIN_H:
        return False, f"too_small_{w}x{h}"
    return True, f"ok_{w}x{h}"


def iter_images(root: Path):
    for franchise_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for img_path in sorted(franchise_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXT:
                yield franchise_dir.name, img_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Move rejects into dataset/_rejected/")
    args = ap.parse_args()

    if not RAW_DIR.exists():
        print(f"raw dir missing: {RAW_DIR}", file=sys.stderr)
        return 1

    REJECT_DIR.mkdir(exist_ok=True)
    log_rows: list[list[str]] = []

    totals = {"checked": 0, "passed": 0, "rejected": 0}
    per_franchise: dict[str, dict[str, int]] = {}

    for franchise, img_path in iter_images(RAW_DIR):
        totals["checked"] += 1
        stats = per_franchise.setdefault(franchise, {"passed": 0, "rejected": 0})

        ok, reason = check(img_path)
        if ok:
            totals["passed"] += 1
            stats["passed"] += 1
        else:
            totals["rejected"] += 1
            stats["rejected"] += 1
            log_rows.append([
                datetime.now().isoformat(timespec="seconds"),
                franchise,
                img_path.name,
                reason,
                "moved" if args.apply else "report_only",
            ])
            if args.apply:
                dest_dir = REJECT_DIR / franchise
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), dest_dir / img_path.name)

    if log_rows:
        write_header = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "franchise", "file", "reason", "action"])
            writer.writerows(log_rows)

    print("=" * 60)
    print(f"Checked: {totals['checked']}    Passed: {totals['passed']}    Rejected: {totals['rejected']}")
    print(f"Mode:    {'APPLY (rejects moved)' if args.apply else 'REPORT ONLY'}")
    print("-" * 60)
    print(f"{'franchise':<14}{'passed':>10}{'rejected':>10}")
    print("-" * 60)
    for franchise in sorted(per_franchise):
        s = per_franchise[franchise]
        print(f"{franchise:<14}{s['passed']:>10}{s['rejected']:>10}")
    if log_rows:
        print(f"\nRejection log: {LOG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
