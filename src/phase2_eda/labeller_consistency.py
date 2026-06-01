"""T2.3 — Inter-labeller (inter-group) consistency check.

The class master CSV resolves filename collisions with "later submission
wins". So if two groups labelled the same image, only one survives in the
master. To measure disagreement, we have to pull each group's CSV directly
from GET /group/<name> and find overlaps.

Output:
  dataset/labeller_consistency.csv
    image, group_a, group_b, agreement_cells, disagreement_cells,
    agreement_pct, a_labels_present, b_labels_present
  one row per (image, group_pair). Sorted by lowest agreement first.

  dataset/labeller_consistency_summary.txt
    overall stats: how many images shared, mean agreement, per-pair counts.

Usage:
    python src/phase2_eda/labeller_consistency.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import quote

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "dataset"
OUT_PAIRS = DATASET / "labeller_consistency.csv"
OUT_SUMMARY = DATASET / "labeller_consistency_summary.txt"

API_BASE = "https://3dixfexeg3.execute-api.ap-south-1.amazonaws.com"
TIMEOUT = 30
CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]


def list_groups(session: requests.Session) -> list[str]:
    r = session.get(f"{API_BASE}/groups", timeout=TIMEOUT)
    r.raise_for_status()
    data = json.loads(r.text)
    return [g["group"] for g in data]


def fetch_group(session: requests.Session, group: str) -> dict[str, list[int]]:
    r = session.get(f"{API_BASE}/group/{quote(group)}", timeout=TIMEOUT)
    r.raise_for_status()
    text = r.text.strip()
    if not text:
        return {}
    reader = csv.DictReader(io.StringIO(text))
    out: dict[str, list[int]] = {}
    for row in reader:
        name = (row.get("image") or "").strip()
        if not name:
            continue
        try:
            cells = [int((row.get(c) or "0").strip() or 0) for c in CELL_COLS]
        except ValueError:
            continue
        out[name] = cells
    return out


def main() -> int:
    session = requests.Session()
    print("listing groups …")
    groups = list_groups(session)
    print(f"  {len(groups)} groups")

    per_group: dict[str, dict[str, list[int]]] = {}
    for g in groups:
        print(f"  fetching {g} …")
        per_group[g] = fetch_group(session, g)

    # invert: image -> {group: cells}
    by_image: dict[str, dict[str, list[int]]] = {}
    for g, m in per_group.items():
        for name, cells in m.items():
            by_image.setdefault(name, {})[g] = cells

    shared = {n: d for n, d in by_image.items() if len(d) >= 2}

    rows: list[dict] = []
    pair_counts: dict[tuple[str, str], int] = {}
    for name, by_g in shared.items():
        items = sorted(by_g.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ga, ca = items[i]
                gb, cb = items[j]
                agree = sum(1 for a, b in zip(ca, cb) if a == b)
                rows.append({
                    "image": name,
                    "group_a": ga,
                    "group_b": gb,
                    "agreement_cells": agree,
                    "disagreement_cells": 64 - agree,
                    "agreement_pct": round(agree / 64 * 100, 2),
                    "a_labels_present": ",".join(str(x) for x in sorted(set(v for v in ca if v != 0))),
                    "b_labels_present": ",".join(str(x) for x in sorted(set(v for v in cb if v != 0))),
                })
                pair_counts[(ga, gb)] = pair_counts.get((ga, gb), 0) + 1

    rows.sort(key=lambda r: r["agreement_pct"])
    DATASET.mkdir(parents=True, exist_ok=True)
    with OUT_PAIRS.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("image,group_a,group_b,agreement_cells,disagreement_cells,agreement_pct,a_labels_present,b_labels_present\n")

    n_shared = len(shared)
    mean_pct = (sum(r["agreement_pct"] for r in rows) / len(rows)) if rows else 0.0
    with OUT_SUMMARY.open("w") as f:
        f.write(f"groups queried        : {len(groups)}\n")
        f.write(f"images labelled by 2+ : {n_shared}\n")
        f.write(f"pair-image rows       : {len(rows)}\n")
        f.write(f"mean agreement (cells/64): {mean_pct:.2f}%\n\n")
        f.write("Per-pair overlap counts (only pairs with overlap):\n")
        for (a, b), n in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
            f.write(f"  {a} <-> {b}: {n} image(s)\n")

    print(f"\nwrote {OUT_PAIRS.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    print(f"images shared across 2+ groups: {n_shared}")
    print(f"mean cell-level agreement: {mean_pct:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
