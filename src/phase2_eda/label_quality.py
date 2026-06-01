"""T2.2 — Surface suspicious label patterns.

Reads dataset/labels.csv and dataset/labels_provenance.csv and writes
dataset/label_quality_flags.csv with one row per (image, flag) — i.e. an
image can appear multiple times if it triggers multiple flags.

Flags emitted:

  ISOLATED_CELL          A non-zero cell whose four 4-neighbours are all 0.
                         Likely a stray click. We log it; you decide if it's
                         legitimate (single small player at the edge) or noise.

  ALL_64_NONZERO         All 64 cells are non-zero — the labeller probably
                         painted the whole canvas instead of selective cells.

  MULTI_TEAM_NON_MATCH   The image's filename matches a known TeamAvsTeamB
                         pattern but cells contain a 3rd team's label.
                         Either a mislabel, or a third-team player legitimately
                         on the field (rare; flagged for review).

  SUSPICIOUS_NUM_PLAYERS num_players in the provenance disagrees grossly
                         with the cells: e.g. num_players >= 5 but fewer than
                         3 non-zero cells in the whole image.

Usage:
    python src/phase2_eda/label_quality.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = REPO_ROOT / "dataset" / "labels.csv"
PROV = REPO_ROOT / "dataset" / "labels_provenance.csv"
OUT = REPO_ROOT / "dataset" / "label_quality_flags.csv"

CELL_COLS = [f"c{i:02d}" for i in range(1, 65)]
TEAM_TO_IDX = {"CSK": 1, "DC": 2, "GT": 3, "KKR": 4, "LSG": 5,
               "MI": 6, "PBKS": 7, "RR": 8, "RCB": 9, "SRH": 10}
MATCH_RE = re.compile(r"^([A-Za-z]{2,5})vs([A-Za-z]{2,5})_image", re.IGNORECASE)


def neighbours_in_grid(idx: int) -> list[int]:
    """4-neighbours of cell idx (0..63) in the 8x8 grid, clipped to bounds."""
    r, c = divmod(idx, 8)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            out.append(nr * 8 + nc)
    return out


def load_csv_as_dict(path: Path, key: str) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row[key]] = row
    return out


def main() -> int:
    labels = load_csv_as_dict(LABELS, "image")
    prov = load_csv_as_dict(PROV, "image") if PROV.exists() else {}

    flags: list[dict] = []

    for name, row in labels.items():
        cells = [int(row[c]) for c in CELL_COLS]

        # ALL_64_NONZERO
        if all(v != 0 for v in cells):
            flags.append({"image": name, "flag": "ALL_64_NONZERO", "detail": "every cell is non-zero"})

        # ISOLATED_CELL
        for i, v in enumerate(cells):
            if v == 0:
                continue
            if all(cells[j] == 0 for j in neighbours_in_grid(i)):
                flags.append({
                    "image": name,
                    "flag": "ISOLATED_CELL",
                    "detail": f"cell c{i+1:02d}={v} has no non-zero 4-neighbour",
                })

        # MULTI_TEAM_NON_MATCH
        m = MATCH_RE.match(name)
        if m:
            a, b = m.group(1).upper(), m.group(2).upper()
            allowed = {0}
            if a in TEAM_TO_IDX:
                allowed.add(TEAM_TO_IDX[a])
            if b in TEAM_TO_IDX:
                allowed.add(TEAM_TO_IDX[b])
            stray = sorted(set(cells) - allowed)
            if stray:
                flags.append({
                    "image": name,
                    "flag": "MULTI_TEAM_NON_MATCH",
                    "detail": f"contains labels {stray} but filename implies only {sorted(allowed - {0})}",
                })

        # SUSPICIOUS_NUM_PLAYERS
        p = prov.get(name, {})
        np_raw = (p.get("num_players") or "").strip()
        if np_raw.isdigit():
            n_players = int(np_raw)
            non_zero = sum(1 for v in cells if v != 0)
            if n_players >= 5 and non_zero < 3:
                flags.append({
                    "image": name,
                    "flag": "SUSPICIOUS_NUM_PLAYERS",
                    "detail": f"num_players={n_players} but only {non_zero} non-zero cells",
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "flag", "detail"])
        w.writeheader()
        w.writerows(flags)

    by_flag: dict[str, int] = {}
    for f in flags:
        by_flag[f["flag"]] = by_flag.get(f["flag"], 0) + 1
    print(f"wrote {OUT.relative_to(REPO_ROOT)}  ({len(flags)} flag rows over {len(set(f['image'] for f in flags))} unique images)")
    for k, v in sorted(by_flag.items()):
        print(f"  {k:<28} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
