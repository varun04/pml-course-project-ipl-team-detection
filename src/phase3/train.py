"""T3.3 — Train four classifiers on the per-cell feature set, pick the best.

Reads dataset/features/{X,y,image_idx,split,...}.npy and trains:

  1. Multinomial Logistic Regression (class_weight='balanced')
  2. Linear SVC (class_weight='balanced')
  3. Random Forest (class_weight='balanced')
  4. LightGBM Gradient Boosting (class_weight='balanced')

For each, it reports macro-F1 on the held-out 10 % image split, and the
elapsed wall-clock. The best model (by macro-F1) is saved as

    models/model_ipl_jersey_prediction_varun.pkl

containing a dict:
    {
      'scaler'           StandardScaler (fitted on train),
      'model'            best classifier,
      'model_name'       'logreg' / 'svm' / 'rf',
      'feature_names'    list[str],
      'feature_dim'      int,
      'class_names'      dict[int, str],
      'meta'             dict with trained_at, dataset size, scores
    }

The other two classifiers are also persisted as models/model_<name>.pkl for
comparison plots in the slide deck.

Usage:
    python src/phase3/train.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = REPO_ROOT / "dataset" / "features"
MODEL_DIR = REPO_ROOT / "models"
OUT_DIR = REPO_ROOT / "outputs"

CLASS_NAMES = {0: "no_team", 1: "CSK", 2: "DC", 3: "GT", 4: "KKR", 5: "LSG",
               6: "MI", 7: "PBKS", 8: "RR", 9: "RCB", 10: "SRH"}


def load_arrays():
    X = np.load(FEAT_DIR / "X.npy")
    y = np.load(FEAT_DIR / "y.npy")
    image_idx = np.load(FEAT_DIR / "image_idx.npy")
    split = np.load(FEAT_DIR / "split.npy")
    meta = json.loads((FEAT_DIR / "meta.json").read_text())
    return X, y, image_idx, split, meta


def _smooth_proba_map(proba_64: np.ndarray, alpha: float) -> np.ndarray:
    """3×3 neighbourhood average over the 8×8 cell probability grid for one image.

    proba_64 : (64, n_classes)  — row-major cell order.
    alpha    : blend weight (0 = no smoothing, 1 = full neighbourhood average).
    """
    n_cls  = proba_64.shape[1]
    grid   = proba_64.reshape(8, 8, n_cls)
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    nbr    = sum(padded[r:r + 8, c:c + 8] for r in range(3) for c in range(3)) / 9.0
    return (alpha * nbr + (1.0 - alpha) * grid).reshape(64, n_cls)


def _smooth_batch(proba: np.ndarray, cell_img_idx: np.ndarray,
                  alpha: float) -> np.ndarray:
    """Apply _smooth_proba_map per image over a batch of cells."""
    if alpha == 0.0:
        return proba
    out = proba.copy()
    for img_id in np.unique(cell_img_idx):
        sel = cell_img_idx == img_id
        if sel.sum() == 64:
            out[sel] = _smooth_proba_map(proba[sel], alpha)
    return out


def _build_ctx_features(proba: np.ndarray, cell_img_idx: np.ndarray) -> np.ndarray:
    """Build [own_proba(11) | 8-neighbour mean proba(11)] = 22 context features per cell.

    Stage-1 probabilities capture per-cell colour/texture.
    Neighbour probabilities capture spatial consistency — a jersey spans multiple
    cells, so high-confidence neighbours resolve ambiguous cells that stage-1
    misclassifies as 'none'.
    """
    n_cls = proba.shape[1]
    ctx = np.empty((len(proba), n_cls * 2), dtype=np.float32)
    ctx[:, :n_cls] = proba          # own probabilities (always set)
    ctx[:, n_cls:] = proba          # fallback: own if image is incomplete

    for img_id in np.unique(cell_img_idx):
        sel = np.where(cell_img_idx == img_id)[0]
        if len(sel) != 64:
            continue
        p      = proba[sel]                                     # (64, n_cls)
        grid   = p.reshape(8, 8, n_cls)
        padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        nbr    = np.empty((8, 8, n_cls), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                patch      = padded[r:r + 3, c:c + 3].reshape(9, n_cls)
                nbr[r, c]  = np.delete(patch, 4, axis=0).mean(axis=0)  # 8 neighbours
        ctx[sel, n_cls:] = nbr.reshape(64, n_cls)
    return ctx


def tune_thresholds(proba: np.ndarray, y_true: np.ndarray,
                    n_classes: int = 11, n_iter: int = 3) -> np.ndarray:
    """Coordinate-ascent on per-class probability thresholds to maximise macro-F1.

    Prediction rule: argmax(proba / thresholds).
    A lower threshold[c] makes the model more willing to predict class c,
    boosting recall for under-predicted minority classes.
    """
    thresholds = np.ones(n_classes, dtype=np.float64)
    grid = np.linspace(0.05, 1.5, 50)
    for iteration in range(n_iter):
        for c in range(n_classes):
            best_t = thresholds[c]
            best_f1 = f1_score(y_true, np.argmax(proba / thresholds, axis=1),
                               average="macro", zero_division=0)
            for t in grid:
                thresholds[c] = t
                preds = np.argmax(proba / thresholds, axis=1)
                f = f1_score(y_true, preds, average="macro", zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, t
            thresholds[c] = best_t
        overall = f1_score(y_true, np.argmax(proba / thresholds, axis=1),
                           average="macro", zero_division=0)
        print(f"  threshold iter {iteration + 1}/{n_iter}: cal macro-F1 = {overall:.4f}")
    return thresholds


def main() -> int:
    X, y, image_idx, split, meta = load_arrays()

    is_test = split[image_idx] == 1

    # Hold out 20 % of training images as a calibration set for threshold tuning.
    # Splitting by image (not cell) prevents data leakage between fit and cal.
    train_img_ids = np.where(split == 0)[0]
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(train_img_ids)
    n_cal_imgs = max(1, int(0.2 * len(shuffled)))
    cal_img_set = set(shuffled[:n_cal_imgs].tolist())

    is_cal_cell = np.isin(image_idx, np.array(list(cal_img_set))) & ~is_test
    is_fit_cell = (~is_test) & (~is_cal_cell)

    X_fit,  y_fit  = X[is_fit_cell],  y[is_fit_cell]
    X_cal,  y_cal  = X[is_cal_cell],  y[is_cal_cell]
    X_test, y_test = X[is_test],      y[is_test]

    # ── Cap "none" cells to 3× total team cells in the fit split ─────────────
    # Rationale: class_weight='balanced' already balances the loss function,
    # but the model still sees ~48 none cells for every 1 team cell during
    # training. Capping reduces background memorisation and forces the model
    # to learn more general "not-a-jersey" patterns → improves recall.
    none_mask  = (y_fit == 0)
    team_mask  = (y_fit != 0)
    n_team_fit = int(team_mask.sum())
    NONE_CAP   = 3 * n_team_fit          # 3× ratio — tuned heuristic
    rng_cap    = np.random.default_rng(42)

    if none_mask.sum() > NONE_CAP:
        keep_none = rng_cap.choice(np.where(none_mask)[0], size=NONE_CAP, replace=False)
        keep_all  = np.sort(np.concatenate([keep_none, np.where(team_mask)[0]]))
        X_fit, y_fit = X_fit[keep_all], y_fit[keep_all]
        print(f"  none cap  : {none_mask.sum():,} → {NONE_CAP:,}  "
              f"(3× {n_team_fit:,} team cells)")

    print(f"fit  cells : {len(y_fit):,}  ({len(shuffled) - n_cal_imgs} images, none capped)")
    print(f"cal  cells : {len(y_cal):,}  ({n_cal_imgs} images — for threshold tuning)")
    print(f"test cells : {len(y_test):,}  ({int((split == 1).sum())} images)")
    print(f"features   : {X.shape[1]}")

    print("\nfitting StandardScaler on fit split …")
    scaler = StandardScaler().fit(X_fit)
    Xs_fit  = scaler.transform(X_fit).astype(np.float32)
    Xs_cal  = scaler.transform(X_cal).astype(np.float32)
    Xs_test = scaler.transform(X_test).astype(np.float32)

    results = {}
    models = {}

    # 1. Logistic Regression
    print("\n--- Logistic Regression ---")
    t0 = time.time()
    lr = LogisticRegression(
        solver="lbfgs", max_iter=1000,
        class_weight="balanced", n_jobs=-1,
    )
    lr.fit(Xs_fit, y_fit)
    pred = lr.predict(Xs_test)
    f1 = f1_score(y_test, pred, average="macro")
    results["logreg"] = {"macro_f1": float(f1), "fit_seconds": round(time.time() - t0, 2)}
    models["logreg"] = lr
    print(f"  macro-F1 = {f1:.4f}    (fit {results['logreg']['fit_seconds']}s)")

    # 2. Linear SVM
    print("\n--- Linear SVC ---")
    t0 = time.time()
    svm = LinearSVC(class_weight="balanced", max_iter=5000, dual="auto")
    svm.fit(Xs_fit, y_fit)
    pred = svm.predict(Xs_test)
    f1 = f1_score(y_test, pred, average="macro")
    results["svm"] = {"macro_f1": float(f1), "fit_seconds": round(time.time() - t0, 2)}
    models["svm"] = svm
    print(f"  macro-F1 = {f1:.4f}    (fit {results['svm']['fit_seconds']}s)")

    # 3. Random Forest + per-class threshold tuning
    print("\n--- Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_fit, y_fit)  # RF doesn't need scaling

    # Tune thresholds on calibration set (no leakage into test)
    print("  tuning per-class thresholds on calibration split …")
    cal_proba = rf.predict_proba(X_cal)
    thresholds = tune_thresholds(cal_proba, y_cal)

    # Report both raw and threshold-tuned macro-F1 on test
    f1_raw = f1_score(y_test, rf.predict(X_test), average="macro")
    test_proba = rf.predict_proba(X_test)
    pred_tuned = np.argmax(test_proba / thresholds, axis=1)
    f1_tuned = f1_score(y_test, pred_tuned, average="macro")

    fit_time = round(time.time() - t0, 2)
    results["rf"] = {
        "macro_f1":     float(f1_tuned),
        "macro_f1_raw": float(f1_raw),
        "fit_seconds":  fit_time,
    }
    models["rf"] = rf
    print(f"  macro-F1 (raw)    = {f1_raw:.4f}")
    print(f"  macro-F1 (tuned)  = {f1_tuned:.4f}    (fit {fit_time}s)")
    print(f"  thresholds: { {CLASS_NAMES[c]: round(float(thresholds[c]), 3) for c in range(11)} }")

    # 4. LightGBM + per-class threshold tuning
    print("\n--- LightGBM ---")
    t0 = time.time()
    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", n_jobs=-1, random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_fit, y_fit)  # LightGBM handles raw features without scaling

    print("  tuning per-class thresholds on calibration split …")
    lgbm_cal_proba = lgbm.predict_proba(X_cal)
    lgbm_thresholds = tune_thresholds(lgbm_cal_proba, y_cal)

    f1_lgbm_raw = f1_score(y_test, lgbm.predict(X_test), average="macro")
    lgbm_test_proba = lgbm.predict_proba(X_test)
    pred_lgbm_tuned = np.argmax(lgbm_test_proba / lgbm_thresholds, axis=1)
    f1_lgbm_tuned = f1_score(y_test, pred_lgbm_tuned, average="macro")

    fit_time = round(time.time() - t0, 2)
    print(f"  macro-F1 (raw)    = {f1_lgbm_raw:.4f}")
    print(f"  macro-F1 (tuned)  = {f1_lgbm_tuned:.4f}    (fit {fit_time}s)")
    print(f"  thresholds: { {CLASS_NAMES[c]: round(float(lgbm_thresholds[c]), 3) for c in range(11)} }")

    # Spatial smoothing — tune alpha on calibration split, evaluate on test
    print("  tuning spatial smoothing alpha on calibration split …")
    cal_img_idx  = image_idx[is_cal_cell]
    test_img_idx = image_idx[is_test]

    pred_cal_base  = np.argmax(lgbm_cal_proba / lgbm_thresholds, axis=1)
    f1_cal_base    = f1_score(y_cal, pred_cal_base, average="macro", zero_division=0)
    best_alpha, best_alpha_f1 = 0.0, f1_cal_base
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        s    = _smooth_batch(lgbm_cal_proba, cal_img_idx, alpha)
        pred = np.argmax(s / lgbm_thresholds, axis=1)
        f    = f1_score(y_cal, pred, average="macro", zero_division=0)
        if f > best_alpha_f1:
            best_alpha_f1, best_alpha = f, alpha

    s_test       = _smooth_batch(lgbm_test_proba, test_img_idx, best_alpha)
    pred_smooth  = np.argmax(s_test / lgbm_thresholds, axis=1)
    f1_smooth    = f1_score(y_test, pred_smooth, average="macro")
    print(f"  best alpha={best_alpha:.1f}  "
          f"(cal: {f1_cal_base:.4f} → {best_alpha_f1:.4f})")
    print(f"  macro-F1 (smoothed, alpha={best_alpha:.1f}) = {f1_smooth:.4f}")

    results["lgbm"] = {
        "macro_f1":        float(f1_smooth),
        "macro_f1_tuned":  float(f1_lgbm_tuned),
        "macro_f1_raw":    float(f1_lgbm_raw),
        "smoothing_alpha": best_alpha,
        "fit_seconds":     fit_time,
    }
    models["lgbm"] = lgbm

    # ── Stage 2: context-aware re-classification ─────────────────────────────
    # Train a second LightGBM on [own_proba | neighbour_mean_proba] features.
    # Stage-1 cal cells are out-of-sample for stage-1 → no leakage.
    print("\n--- Stage-2 context classifier ---")
    t0_s2 = time.time()

    cal_img_idx  = image_idx[is_cal_cell]
    test_img_idx = image_idx[is_test]

    X2_cal  = _build_ctx_features(lgbm_cal_proba,  cal_img_idx)
    X2_test = _build_ctx_features(lgbm_test_proba, test_img_idx)

    stage2 = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1,
    )
    stage2.fit(X2_cal, y_cal)

    y_s2      = stage2.predict(X2_test)
    f1_s2     = f1_score(y_test, y_s2, average="macro")
    print(f"  context features : 22  (11 own + 11 neighbour mean)")
    print(f"  macro-F1 (stage-2) = {f1_s2:.4f}    (fit {time.time()-t0_s2:.1f}s)")
    print(f"  vs stage-1 tuned  = {f1_lgbm_tuned:.4f}")

    # Update lgbm results to use stage-2 as the final score
    results["lgbm"]["macro_f1"]        = float(f1_s2)
    results["lgbm"]["macro_f1_stage2"] = float(f1_s2)
    models["lgbm_stage2"] = stage2

    # pick winner (RF and LightGBM use tuned F1)
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = models[best_name]
    print(f"\nwinner: {best_name}  (macro-F1 = {results[best_name]['macro_f1']:.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist all four for the comparison slide
    model_thresholds   = {"rf": thresholds, "lgbm": lgbm_thresholds}
    model_smooth_alpha = {"lgbm": best_alpha}
    for name, model in models.items():
        pkl = {"model": model, "scaler": scaler if name not in ("rf", "lgbm") else None}
        if name in model_thresholds:
            pkl["thresholds"] = model_thresholds[name]
        if name in model_smooth_alpha:
            pkl["smoothing_alpha"] = model_smooth_alpha[name]
        joblib.dump(pkl, MODEL_DIR / f"model_{name}.pkl")
    print(f"saved per-model pickles: models/model_logreg.pkl, model_svm.pkl, model_rf.pkl, model_lgbm.pkl")

    # The deliverable pickle
    best_thresholds = model_thresholds.get(best_name)
    payload = {
        "scaler":          scaler if best_name not in ("rf", "lgbm") else None,
        "thresholds":      best_thresholds,
        "smoothing_alpha": model_smooth_alpha.get(best_name, 0.0),
        "stage2_model":    stage2 if best_name == "lgbm" else None,
        "model":         best_model,
        "model_name":    best_name,
        "feature_names": meta["feature_names"],
        "feature_dim":   meta["feature_dim"],
        "class_names":   CLASS_NAMES,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "fit_cells":  int(len(y_fit)),
            "cal_cells":  int(len(y_cal)),
            "test_cells": int(len(y_test)),
            "results":    results,
            "winner":     best_name,
        },
    }
    deliverable = MODEL_DIR / "model_ipl_jersey_prediction_varun.pkl"
    joblib.dump(payload, deliverable)
    print(f"saved deliverable     : {deliverable.relative_to(REPO_ROOT)}")

    # Drop a small results JSON for the report
    (OUT_DIR / "phase3_train_results.json").write_text(json.dumps({
        "winner":  best_name,
        "results": results,
        "n_train": int(len(y_fit)),
        "n_cal":   int(len(y_cal)),
        "n_test":  int(len(y_test)),
    }, indent=2))
    print(f"saved metrics         : {(OUT_DIR / 'phase3_train_results.json').relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
