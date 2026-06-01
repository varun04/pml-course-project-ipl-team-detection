"""T1.8 — Ingest a downloaded UI bundle (master.csv + images/).

The labelling tool's "Download dataset" button produces a folder like:
    <bundle>/
        master.csv
        images/
            <image1>.jpg
            ...

This script reads from such a folder and copies what's new into our local
dataset, mirroring what import_master.py does over the network but with no
S3 fetches at all. Use this when you've grabbed a bundle in your browser
and want to sync without hitting the API.

Behaviour:
- Updates dataset/labels.csv and dataset/labels_provenance.csv from the
  bundle's master.csv (replacing the existing files — append-only history is
  in dataset/_external/).
- Copies any image files from the bundle into dataset/raw/<match>/<file>,
  bucketed by the same TeamAvsTeamB filename prefix import_master.py uses.
- **Never deletes** images already in dataset/raw/. So if S3 has since 403'd
  on an image we previously cached, our local copy survives.
- Archives the bundle's master.csv to
  dataset/_external/class_master_<utc>_from_bundle.csv.

Usage:
    python src/phase1_dataset/import_bundle.py <bundle_dir>           # dry run
    python src/phase1_dataset/import_bundle.py <bundle_dir> --apply
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "dataset"
RAW = DATASET / "raw"
EXTERNAL = DATASET / "_external"
LABELS = DATASET / "labels.csv"
PROV = DATASET / "labels_provenance.csv"

CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]
MATCH_RE = re.compile(r"^([A-Za-z]{2,5}vs[A-Za-z]{2,5})_image_\d+\.")


def match_bucket(image: str) -> str:
    m = MATCH_RE.match(image)
    return m.group(1) if m else "_misc"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir", type=Path, help="folder containing master.csv and images/")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src_csv = args.bundle_dir / "master.csv"
    src_imgs = args.bundle_dir / "images"
    if not src_csv.is_file():
        print(f"missing {src_csv}", file=sys.stderr); return 1
    if not src_imgs.is_dir():
        print(f"missing {src_imgs}", file=sys.stderr); return 1

    rows = list(csv.DictReader(src_csv.open()))
    if not rows:
        print("bundle csv is empty"); return 1
    header = list(rows[0].keys())
    required = {"image", *CELL_COLS}
    missing = required - set(header)
    if missing:
        print(f"bundle csv missing columns: {sorted(missing)}", file=sys.stderr); return 1

    bundle_images = {p.name for p in src_imgs.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}}

    # existing local images
    local_images = set()
    for sub in RAW.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        for p in sub.iterdir():
            if p.suffix.lower() in {".jpg",".jpeg",".png"}:
                local_images.add(p.name)

    to_copy = bundle_images - local_images
    only_local = local_images - bundle_images
    overlap = bundle_images & local_images

    # existing labels.csv (for delta reporting)
    prev_label_count = 0
    if LABELS.exists():
        with LABELS.open() as f:
            prev_label_count = sum(1 for _ in csv.DictReader(f))

    print(f"bundle master.csv rows  : {len(rows)}")
    print(f"local labels.csv rows   : {prev_label_count}")
    print(f"bundle images           : {len(bundle_images)}")
    print(f"local images            : {len(local_images)}")
    print(f"images to copy from bundle : {len(to_copy)}")
    print(f"images only in local       : {len(only_local)}  (kept — never deleted)")
    print(f"images in both             : {len(overlap)}")

    new_by_match: dict[str, int] = {}
    for n in to_copy:
        m = match_bucket(n)
        new_by_match[m] = new_by_match.get(m, 0) + 1
    if new_by_match:
        print(f"new images by match:")
        for m, n in sorted(new_by_match.items(), key=lambda kv: -kv[1]):
            print(f"  {m}: {n}")

    if not args.apply:
        print("\n(dry run) re-run with --apply to copy files and refresh CSVs")
        return 0

    # archive the bundle CSV
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = EXTERNAL / f"class_master_{stamp}_from_bundle.csv"
    shutil.copy2(src_csv, archive_path)
    print(f"\narchived bundle csv    : {archive_path.relative_to(REPO_ROOT)}")

    # rewrite labels.csv (project schema: image + c01..c64 only)
    with LABELS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", *CELL_COLS])
        for r in rows:
            w.writerow([r["image"], *(r.get(c, "0") or "0" for c in CELL_COLS)])
    print(f"wrote labels.csv       : {LABELS.relative_to(REPO_ROOT)} ({len(rows)} rows)")

    # rewrite labels_provenance.csv
    with PROV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "submitted_by", "labelled_by", "num_players"])
        for r in rows:
            w.writerow([
                r.get("image", ""),
                r.get("submitted_by", "") or "",
                r.get("labelled_by", "") or "",
                r.get("num_players", "") or "",
            ])
    print(f"wrote provenance       : {PROV.relative_to(REPO_ROOT)}")

    # copy new images
    copied = 0
    for name in sorted(to_copy):
        src = src_imgs / name
        dest_dir = RAW / match_bucket(name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists():
            continue
        shutil.copy2(src, dest)
        copied += 1
    print(f"copied images          : {copied}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
