"""T4 — Inference pipeline (course-submission deliverable).

Reads the saved model_ipl_jersey_prediction_varun.pkl, takes any 800x600 image, returns
a length-64 prediction vector in row-major c01..c64 order.

In CLI mode, it walks a directory of images and emits a CSV in the spec's
required submission format:

    Image File Name, Train Or Test, c01, c02, ..., c64

Usage as a library:
    from phase3.pipeline import load_model, predict_image
    model = load_model('models/model_ipl_jersey_prediction_varun.pkl')
    cells = predict_image(model, img_bgr_800x600)  # list of 64 ints

Usage as a script (writes outputs/predictions.csv):
    python src/phase3/pipeline.py
        # ^ runs on dataset/processed/, labels every image as train/test
        #   per dataset/features/split.npy and image_names.npy

    python src/phase3/pipeline.py --images <dir> --out <csv>
        # arbitrary directory of 800x600 images, all labelled Test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from phase3.extract import extract_image_features, FEATURE_DIM, N_CELLS

MODEL_PATH_DEFAULT = REPO_ROOT / "models" / "model_ipl_jersey_prediction_varun.pkl"


def load_model(model_path: Path | str = MODEL_PATH_DEFAULT):
    return joblib.load(Path(model_path))


def smooth_proba_map(proba: np.ndarray, alpha: float) -> np.ndarray:
    """3×3 neighbourhood average over the 8×8 cell probability grid for one image.

    proba : (64, n_classes) — row-major cell order.
    alpha : blend weight (0.0 = no smoothing, 1.0 = full neighbourhood average).

    Intuition: a jersey cell is surrounded by similar cells; an isolated red
    banner pixel is not. Smoothing propagates high-confidence neighbour
    predictions to ambiguous cells, boosting recall without retraining.
    """
    if alpha == 0.0:
        return proba
    n_cls  = proba.shape[1]
    grid   = proba.reshape(8, 8, n_cls)
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    nbr    = sum(padded[r:r + 8, c:c + 8] for r in range(3) for c in range(3)) / 9.0
    return (alpha * nbr + (1.0 - alpha) * grid).reshape(64, n_cls)


def _build_ctx_features_single(proba_64: np.ndarray) -> np.ndarray:
    """Build 22-feature context vectors for a single image's 64 cells.

    For each cell: [own_proba(11) | 8-neighbour mean proba(11)].
    """
    n_cls  = proba_64.shape[1]
    ctx    = np.empty((64, n_cls * 2), dtype=np.float32)
    ctx[:, :n_cls] = proba_64
    grid   = proba_64.reshape(8, 8, n_cls)
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    for r in range(8):
        for c in range(8):
            idx       = r * 8 + c
            patch     = padded[r:r + 3, c:c + 3].reshape(9, n_cls)
            ctx[idx, n_cls:] = np.delete(patch, 4, axis=0).mean(axis=0)
    return ctx


def predict_image(model_payload, img_bgr: np.ndarray) -> list[int]:
    if img_bgr.shape[:2] != (600, 800):
        raise ValueError(f"image must be 800x600 BGR, got {img_bgr.shape[:2]}")
    feats  = extract_image_features(img_bgr)
    scaler = model_payload.get("scaler")
    X_in   = scaler.transform(feats) if scaler is not None else feats

    thresholds = model_payload.get("thresholds")
    model      = model_payload["model"]
    if thresholds is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_in)
        alpha = model_payload.get("smoothing_alpha", 0.0)
        if alpha > 0.0:
            proba = smooth_proba_map(proba, alpha)

        stage2 = model_payload.get("stage2_model")
        if stage2 is not None:
            ctx   = _build_ctx_features_single(proba)
            preds = stage2.predict(ctx)
        else:
            preds = np.argmax(proba / thresholds, axis=1)
    else:
        preds = model.predict(X_in)
    return [int(p) for p in preds]


def split_lookup() -> dict[str, str]:
    feat_dir = REPO_ROOT / "dataset" / "features"
    names_path = feat_dir / "image_names.npy"
    split_path = feat_dir / "split.npy"
    if not (names_path.exists() and split_path.exists()):
        return {}
    names = np.load(names_path)
    split = np.load(split_path)
    return {str(n): ("Test" if s == 1 else "Train") for n, s in zip(names, split)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=MODEL_PATH_DEFAULT)
    ap.add_argument("--images", type=Path, default=REPO_ROOT / "dataset" / "processed",
                    help="dir to walk (recursively) for 800x600 images")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "predictions.csv")
    ap.add_argument("--train-test-from-split", action="store_true", default=True,
                    help="label each image as Train/Test using dataset/features/split.npy when available")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 1

    model = load_model(args.model)
    cell_cols = [f"c{i:02d}" for i in range(1, 65)]
    print(f"loaded {args.model.name} (winner: {model.get('model_name', '?')})")

    split = split_lookup() if args.train_test_from_split else {}

    img_paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"images to score: {len(img_paths)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image File Name", "Train Or Test", *cell_cols])
        processed = 0
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None or img.shape[:2] != (600, 800):
                continue
            preds = predict_image(model, img)
            split_label = split.get(p.name, "Test")
            writer.writerow([p.name, split_label, *preds])
            processed += 1
            if processed % 250 == 0:
                print(f"  {processed}/{len(img_paths)}")

    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}  ({processed} predicted rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
