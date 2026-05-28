"""T1.5 — Dataset statistics.

Counts images per franchise in dataset/raw/ and dataset/processed/ and prints
a table. Flags any franchise below the 100-image floor required by the spec.

Usage:
    python src/phase1_dataset/dataset_stats.py
"""

from __future__ import annotations

from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_PER_FRANCHISE = 100

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
PROC_DIR = REPO_ROOT / "dataset" / "processed"


def count_images(d: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not d.exists():
        return out
    for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")):
        out[sub.name] = sum(1 for f in sub.iterdir() if f.suffix.lower() in VALID_EXT)
    return out


def main() -> int:
    raw = count_images(RAW_DIR)
    proc = count_images(PROC_DIR)
    franchises = sorted(set(raw) | set(proc))

    print("=" * 60)
    print(f"{'franchise':<14}{'raw':>8}{'processed':>14}{'meets_min':>14}")
    print("-" * 60)
    for f in franchises:
        r = raw.get(f, 0)
        p = proc.get(f, 0)
        meets = "yes" if p >= MIN_PER_FRANCHISE else f"NO ({MIN_PER_FRANCHISE - p} short)"
        print(f"{f:<14}{r:>8}{p:>14}{meets:>14}")
    print("-" * 60)
    print(f"{'TOTAL':<14}{sum(raw.values()):>8}{sum(proc.values()):>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
