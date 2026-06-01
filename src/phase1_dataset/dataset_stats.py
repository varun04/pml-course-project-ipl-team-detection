"""T1.5 — Dataset statistics.

Reports two views of the dataset:

  1) Folder counts: how many images sit under dataset/raw/<X>/ and
     dataset/processed/<X>/. Useful for tracking download/resize progress.

  2) Per-franchise label coverage: from dataset/labels.csv, how many images
     contain at least one cell of each franchise (1=CSK..10=SRH). This is
     the count that actually has to clear the spec floor of >=100 per
     franchise, because a single image can be an "instance" of multiple
     teams (e.g. an SRHvsDC photo is an instance of both DC and SRH).

Usage:
    python src/phase1_dataset/dataset_stats.py
"""

from __future__ import annotations

import csv
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_PER_FRANCHISE = 100

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
PROC_DIR = REPO_ROOT / "dataset" / "processed"
LABELS_CSV = REPO_ROOT / "dataset" / "labels.csv"

FRANCHISE_NAMES = {
    0: "no_team",
    1: "CSK", 2: "DC", 3: "GT", 4: "KKR", 5: "LSG",
    6: "MI", 7: "PBKS", 8: "RR", 9: "RCB", 10: "SRH",
}


def count_images(d: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not d.exists():
        return out
    for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")):
        out[sub.name] = sum(1 for f in sub.iterdir() if f.suffix.lower() in VALID_EXT)
    return out


def images_on_disk(raw_dir: Path) -> set[str]:
    out: set[str] = set()
    if not raw_dir.exists():
        return out
    for sub in raw_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        for f in sub.iterdir():
            if f.suffix.lower() in VALID_EXT:
                out.add(f.name)
    return out


def per_franchise_image_counts(labels_path: Path, on_disk: set[str]) -> tuple[dict[int, dict[str, int]], int, int]:
    """For each franchise 0..10, count (a) all label rows and (b) only rows whose image is on disk."""
    counts = {k: {"labelled": 0, "usable": 0} for k in FRANCHISE_NAMES}
    total_labels = 0
    total_usable = 0
    if not labels_path.exists():
        return counts, total_labels, total_usable
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        cell_cols = [f"c{i:02d}" for i in range(1, 65)]
        for row in reader:
            total_labels += 1
            usable = row["image"] in on_disk
            if usable:
                total_usable += 1
            present = {int(row[c]) for c in cell_cols if row[c].strip().isdigit()}
            for f_id in FRANCHISE_NAMES:
                hit = (present == {0}) if f_id == 0 else (f_id in present)
                if hit:
                    counts[f_id]["labelled"] += 1
                    if usable:
                        counts[f_id]["usable"] += 1
    return counts, total_labels, total_usable


def main() -> int:
    raw = count_images(RAW_DIR)
    proc = count_images(PROC_DIR)
    folders = sorted(set(raw) | set(proc))

    print("=" * 60)
    print("Folder counts (download/resize progress)")
    print("-" * 60)
    print(f"{'folder':<16}{'raw':>8}{'processed':>14}")
    for f in folders:
        print(f"{f:<16}{raw.get(f, 0):>8}{proc.get(f, 0):>14}")
    print("-" * 60)
    print(f"{'TOTAL':<16}{sum(raw.values()):>8}{sum(proc.values()):>14}")

    on_disk = images_on_disk(RAW_DIR)
    counts, total_labels, total_usable = per_franchise_image_counts(LABELS_CSV, on_disk)
    print()
    print("=" * 70)
    print(f"Label coverage — labels.csv has {total_labels} rows; {total_usable} have an image on disk")
    print("'labelled' = row exists in labels.csv;  'usable' = label + image both available")
    print("'meets_min' tracks the *usable* count vs the spec's 100/franchise floor")
    print("class 0 = images with NO player cells at all (pure background)")
    print("-" * 70)
    print(f"{'franchise':<16}{'labelled':>10}{'usable':>10}{'meets_min':>22}")
    for fid, name in FRANCHISE_NAMES.items():
        c = counts[fid]
        meets = "yes" if c["usable"] >= MIN_PER_FRANCHISE else f"NO ({MIN_PER_FRANCHISE - c['usable']} short)"
        print(f"{fid:>3} {name:<12}{c['labelled']:>10}{c['usable']:>10}{meets:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
