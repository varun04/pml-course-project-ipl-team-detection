"""T1.7 — Pull and ingest the class-wide master labelled dataset.

Hits the cohort labelling tool's API (see reference_class_master_tool memory):

    GET https://3dixfexeg3.execute-api.ap-south-1.amazonaws.com/master

…then downloads every referenced image from the per-group S3 path:

    https://ipl2026-labeller-051024.s3.ap-south-1.amazonaws.com/images/<group>/<image>

Output layout:
    dataset/_external/class_master_<utc_iso>.csv          archived raw master pull
    dataset/raw/<match>/<image>.jpg                       downloaded images, bucketed
                                                            by match prefix (SRHvsDC,
                                                            PBKSvsCSK, ...). Falls back
                                                            to dataset/raw/_misc/ if the
                                                            filename doesn't match the
                                                            usual <TEAMvsTEAM>_image_*
                                                            pattern.
    dataset/labels.csv                                    image, c01..c64
                                                            (project submission schema —
                                                            'Train Or Test' is added in
                                                            a later phase, not here)
    dataset/labels_provenance.csv                         image, submitted_by,
                                                            labelled_by, num_players
                                                            (kept for our own auditing —
                                                            NOT for final submission)
    dataset/master_import_warnings.txt                    one-line warning per skipped
                                                            row / failed download

Behaviour:
- Idempotent. Already-downloaded images (matching size on disk) are skipped.
- Rows in master that have no `submitted_by` (orphan labels) are kept in
  labels.csv with no image download attempted.
- Rows whose image returns 403 / 404 from S3 are still kept in labels.csv;
  the failure is logged.
- Concurrency capped at 8 (matches the tool's own bundle downloader).

Usage:
    python src/phase1_dataset/import_master.py            # dry run (no writes)
    python src/phase1_dataset/import_master.py --apply    # do it
    python src/phase1_dataset/import_master.py --apply --concurrency 12
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

API_BASE = "https://3dixfexeg3.execute-api.ap-south-1.amazonaws.com"
S3_BASE = "https://ipl2026-labeller-051024.s3.ap-south-1.amazonaws.com/images"
MASTER_URL = f"{API_BASE}/master"
TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (pml-ipl-team-detection/group_18_varun)"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
EXTERNAL_DIR = DATASET_DIR / "_external"
LABELS_PATH = DATASET_DIR / "labels.csv"
PROV_PATH = DATASET_DIR / "labels_provenance.csv"
WARN_PATH = DATASET_DIR / "master_import_warnings.txt"

CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]
MATCH_RE = re.compile(r"^([A-Za-z]{2,5}vs[A-Za-z]{2,5})_image_\d+\.")


def match_bucket(image: str) -> str:
    m = MATCH_RE.match(image)
    return m.group(1) if m else "_misc"


def fetch_master(session: requests.Session) -> str:
    r = session.get(MASTER_URL, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def parse_master(csv_text: str) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    reader = csv.DictReader(csv_text.splitlines())
    header = reader.fieldnames or []
    required = {"image", *CELL_COLS}
    missing = required - set(header)
    if missing:
        raise SystemExit(f"master CSV missing columns: {sorted(missing)}")
    rows: list[dict] = []
    seen: dict[str, int] = {}
    for line_no, row in enumerate(reader, start=2):
        image = (row.get("image") or "").strip()
        if not image:
            warnings.append(f"line {line_no}: blank image column — skipped")
            continue
        if image in seen:
            warnings.append(
                f"line {line_no}: duplicate image '{image}' (first at line {seen[image]}) — kept first"
            )
            continue
        seen[image] = line_no
        cells = []
        for col in CELL_COLS:
            v = (row.get(col) or "").strip()
            if v == "":
                v = "0"
            if not v.isdigit() or not 0 <= int(v) <= 10:
                warnings.append(f"line {line_no}: bad cell '{v}' in {col} for {image} — coerced to 0")
                v = "0"
            cells.append(v)
        rows.append({
            "image": image,
            "cells": cells,
            "labelled_by": (row.get("labelled_by") or "").strip(),
            "num_players": (row.get("num_players") or "").strip(),
            "submitted_by": (row.get("submitted_by") or "").strip(),
        })
    return rows, warnings


def download_one(session: requests.Session, rec: dict, dest: Path) -> str:
    """Returns 'ok' / 'skip_exists' / error string."""
    if not rec["submitted_by"]:
        return "no_group"
    url = f"{S3_BASE}/{rec['submitted_by']}/{rec['image']}"
    if dest.exists() and dest.stat().st_size > 0:
        return "skip_exists"
    try:
        r = session.get(url, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        return f"net_error:{type(e).__name__}"
    if r.status_code != 200:
        return f"http_{r.status_code}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        return f"write_error:{e}"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually write files; default is dry run")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel image downloads")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"Fetching master CSV from {MASTER_URL} …")
    csv_text = fetch_master(session)
    rows, warnings = parse_master(csv_text)

    by_group: dict[str, int] = {}
    by_match: dict[str, int] = {}
    no_group = 0
    for r in rows:
        if r["submitted_by"]:
            by_group[r["submitted_by"]] = by_group.get(r["submitted_by"], 0) + 1
        else:
            no_group += 1
        by_match[match_bucket(r["image"])] = by_match.get(match_bucket(r["image"]), 0) + 1

    print(f"master rows            : {len(rows)}")
    print(f"with submitted_by      : {len(rows) - no_group}")
    print(f"without submitted_by   : {no_group} (label kept, no image will be fetched)")
    print(f"groups submitting      : {len(by_group)}")
    print(f"match buckets          : {len(by_match)}")
    print(f"parsing warnings       : {len(warnings)}")
    if warnings:
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... ({len(warnings) - 5} more)")

    if not args.apply:
        print("\n(dry run) re-run with --apply to write files and download images")
        return 0

    # archive raw master pull
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = EXTERNAL_DIR / f"class_master_{stamp}.csv"
    archive_path.write_text(csv_text)
    print(f"\narchived raw master  : {archive_path.relative_to(REPO_ROOT)}")

    # write labels.csv (image + c01..c64) and provenance (image + meta)
    with LABELS_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", *CELL_COLS])
        for r in rows:
            w.writerow([r["image"], *r["cells"]])
    print(f"wrote labels         : {LABELS_PATH.relative_to(REPO_ROOT)} ({len(rows)} rows)")

    with PROV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "submitted_by", "labelled_by", "num_players"])
        for r in rows:
            w.writerow([r["image"], r["submitted_by"], r["labelled_by"], r["num_players"]])
    print(f"wrote provenance     : {PROV_PATH.relative_to(REPO_ROOT)}")

    # download images
    tasks: list[tuple[dict, Path]] = []
    for r in rows:
        if not r["submitted_by"]:
            continue
        bucket = match_bucket(r["image"])
        dest = RAW_DIR / bucket / r["image"]
        tasks.append((r, dest))

    print(f"\ndownloading {len(tasks)} images with concurrency={args.concurrency} …")
    counts = {"ok": 0, "skip_exists": 0}
    fails: list[tuple[str, str]] = []

    def work(item):
        r, dest = item
        outcome = download_one(session, r, dest)
        return r["image"], outcome

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        done_n = 0
        for fut in as_completed(futures):
            name, outcome = fut.result()
            done_n += 1
            if outcome in counts:
                counts[outcome] += 1
            else:
                fails.append((name, outcome))
            if done_n % 200 == 0:
                print(f"  progress {done_n}/{len(tasks)}  ok={counts['ok']} skip={counts['skip_exists']} fail={len(fails)}")

    print()
    print(f"downloaded fresh     : {counts['ok']}")
    print(f"skipped (cached)     : {counts['skip_exists']}")
    print(f"failed               : {len(fails)}")
    if fails[:5]:
        for name, why in fails[:5]:
            print(f"  - {name}: {why}")
        if len(fails) > 5:
            print(f"  ... ({len(fails) - 5} more, full list in warnings file)")

    with WARN_PATH.open("w") as f:
        for w in warnings:
            f.write(f"parse: {w}\n")
        for name, why in fails:
            f.write(f"download: {name}: {why}\n")
    print(f"warnings log         : {WARN_PATH.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
