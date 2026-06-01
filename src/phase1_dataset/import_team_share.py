"""T1.9 — Ingest the team-share folder (Labelled_Images_With_CSV).

Some groups in the cohort never uploaded their images to S3, but shared the
labelled image files directly via a shared folder (Drive/Dropbox/etc.).
This script walks that folder and adopts any image whose filename matches
a row in dataset/labels.csv (the master) but for which we have no local
image file yet.

Layout it handles:
    <team_share>/
      *.csv                       loose per-group CSV backups (ignored — master is truth)
      group_X/                    folder with images and maybe a CSV
        ...                       arbitrarily nested
        *.jpg / *.png
      *.zip                       ZIP archives with the same structure

For each image found (either loose or inside a ZIP):
  - if its name already exists in dataset/raw/<any>/, skip
  - if its name is NOT in master labels.csv, log to dataset/team_share_orphans.txt
    (image with no label — we don't ingest it because we have no idea what
    franchise to attribute its cells to)
  - else copy / extract into dataset/raw/<match>/ using the import_master
    bucket convention

Never deletes anything. Never modifies labels.csv.

Usage:
    python src/phase1_dataset/import_team_share.py <team_share_dir>           # dry run
    python src/phase1_dataset/import_team_share.py <team_share_dir> --apply
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "dataset" / "raw"
LABELS = REPO_ROOT / "dataset" / "labels.csv"
ORPHANS = REPO_ROOT / "dataset" / "team_share_orphans.txt"

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MATCH_RE = re.compile(r"^([A-Za-z]{2,5}vs[A-Za-z]{2,5})_image_\d+\.")


def match_bucket(image: str) -> str:
    m = MATCH_RE.match(image)
    return m.group(1) if m else "_misc"


def current_local_images() -> set[str]:
    out: set[str] = set()
    for sub in RAW.iterdir():
        if sub.is_dir() and not sub.name.startswith("_"):
            for p in sub.iterdir():
                if p.suffix.lower() in VALID_EXT:
                    out.add(p.name)
    return out


def labelled_in_master() -> set[str]:
    with LABELS.open() as f:
        return {row["image"] for row in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("share_dir", type=Path)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.share_dir.is_dir():
        print(f"missing {args.share_dir}", file=sys.stderr)
        return 1

    have_locally = current_local_images()
    in_master = labelled_in_master()

    # plan: for every (source, image_name) we encounter, decide one of:
    #   "have"     already in raw/
    #   "orphan"   not in master labels
    #   "adopt"    novel and labelled
    by_match: Counter[str] = Counter()
    plan: list[tuple[str, str, str, Path | tuple[Path, str]]] = []
    # ('have'|'orphan'|'adopt', image_name, source_desc, src_loc)
    # src_loc is a Path for loose files, or (zip_path, member_name) for zip members

    # walk loose files first
    for p in args.share_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            name = p.name
            if name in have_locally:
                plan.append(("have", name, str(p.relative_to(args.share_dir)), p))
            elif name not in in_master:
                plan.append(("orphan", name, str(p.relative_to(args.share_dir)), p))
            else:
                plan.append(("adopt", name, str(p.relative_to(args.share_dir)), p))
                by_match[match_bucket(name)] += 1

    # walk zip files
    for zp in args.share_dir.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    name = Path(info.filename).name
                    if Path(name).suffix.lower() not in VALID_EXT:
                        continue
                    desc = f"{zp.name}::{info.filename}"
                    if name in have_locally:
                        plan.append(("have", name, desc, (zp, info.filename)))
                    elif name not in in_master:
                        plan.append(("orphan", name, desc, (zp, info.filename)))
                    else:
                        plan.append(("adopt", name, desc, (zp, info.filename)))
                        by_match[match_bucket(name)] += 1
        except zipfile.BadZipFile:
            print(f"bad zip skipped: {zp}", file=sys.stderr)

    counts = Counter(p[0] for p in plan)
    print(f"items scanned : {len(plan)}")
    print(f"  already have: {counts.get('have', 0)}")
    print(f"  orphan      : {counts.get('orphan', 0)}  (not in master labels — skipped)")
    print(f"  adopt       : {counts.get('adopt', 0)}")
    if by_match:
        print("by match (adoptable):")
        for m, n in by_match.most_common():
            print(f"  {m}: {n}")

    if not args.apply:
        print("\n(dry run) re-run with --apply to copy/extract adoptable images")
        return 0

    copied = 0
    seen_dest: set[Path] = set()
    orphan_lines: list[str] = []
    for action, name, desc, src in plan:
        if action == "orphan":
            orphan_lines.append(f"{desc}: {name}")
            continue
        if action != "adopt":
            continue
        dest_dir = RAW / match_bucket(name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest in seen_dest or dest.exists():
            continue  # avoid duplicate work if name appears twice across share
        seen_dest.add(dest)
        if isinstance(src, Path):
            shutil.copy2(src, dest)
        else:
            zp, member = src
            with zipfile.ZipFile(zp) as z, z.open(member) as src_f, dest.open("wb") as out_f:
                shutil.copyfileobj(src_f, out_f)
        copied += 1

    if orphan_lines:
        with ORPHANS.open("w") as f:
            f.write(f"# images in {args.share_dir} that have no row in master labels.csv\n")
            f.write(f"# total: {len(orphan_lines)}\n")
            for line in sorted(set(orphan_lines)):
                f.write(line + "\n")
        print(f"orphan log         : {ORPHANS.relative_to(REPO_ROOT)}")
    print(f"images copied      : {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
