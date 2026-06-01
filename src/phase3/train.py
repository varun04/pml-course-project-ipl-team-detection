"""T3.3 — Train three classifiers on the per-cell feature set, pick the best.

Reads dataset/features/{X,y,image_idx,split,...}.npy and trains:

  1. Multinomial Logistic Regression (class_weight='balanced')
  2. Linear SVC (class_weight='balanced')
  3. Random Forest (class_weight='balanced')

For each, it reports macro-F1 on the held-out 10 % image split, and the
elapsed wall-clock. The best model (by macro-F1) is saved as

    models/model_group_18_varun.pkl

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


def main() -> int:
    X, y, image_idx, split, meta = load_arrays()

    is_test = split[image_idx] == 1
    X_train, y_train = X[~is_test], y[~is_test]
    X_test, y_test = X[is_test], y[is_test]
    print(f"train cells : {len(y_train):,}")
    print(f"test  cells : {len(y_test):,}")
    print(f"features    : {X.shape[1]}")

    print("\nfitting StandardScaler …")
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train).astype(np.float32)
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
    lr.fit(Xs_train, y_train)
    pred = lr.predict(Xs_test)
    f1 = f1_score(y_test, pred, average="macro")
    results["logreg"] = {"macro_f1": float(f1), "fit_seconds": round(time.time() - t0, 2)}
    models["logreg"] = lr
    print(f"  macro-F1 = {f1:.4f}    (fit {results['logreg']['fit_seconds']}s)")

    # 2. Linear SVM
    print("\n--- Linear SVC ---")
    t0 = time.time()
    svm = LinearSVC(class_weight="balanced", max_iter=5000, dual="auto")
    svm.fit(Xs_train, y_train)
    pred = svm.predict(Xs_test)
    f1 = f1_score(y_test, pred, average="macro")
    results["svm"] = {"macro_f1": float(f1), "fit_seconds": round(time.time() - t0, 2)}
    models["svm"] = svm
    print(f"  macro-F1 = {f1:.4f}    (fit {results['svm']['fit_seconds']}s)")

    # 3. Random Forest
    print("\n--- Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    pred = rf.predict(X_test)
    f1 = f1_score(y_test, pred, average="macro")
    results["rf"] = {"macro_f1": float(f1), "fit_seconds": round(time.time() - t0, 2)}
    models["rf"] = rf
    print(f"  macro-F1 = {f1:.4f}    (fit {results['rf']['fit_seconds']}s)")

    # pick winner
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = models[best_name]
    print(f"\nwinner: {best_name}  (macro-F1 = {results[best_name]['macro_f1']:.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist all three for the comparison slide
    for name, model in models.items():
        joblib.dump({"model": model, "scaler": scaler if name != "rf" else None},
                    MODEL_DIR / f"model_{name}.pkl")
    print(f"saved per-model pickles: models/model_logreg.pkl, model_svm.pkl, model_rf.pkl")

    # The deliverable pickle
    payload = {
        "scaler": scaler if best_name != "rf" else None,
        "model": best_model,
        "model_name": best_name,
        "feature_names": meta["feature_names"],
        "feature_dim": meta["feature_dim"],
        "class_names": CLASS_NAMES,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_cells": int(len(y_train)),
            "test_cells": int(len(y_test)),
            "results": results,
            "winner": best_name,
        },
    }
    deliverable = MODEL_DIR / "model_group_18_varun.pkl"
    joblib.dump(payload, deliverable)
    print(f"saved deliverable     : {deliverable.relative_to(REPO_ROOT)}")

    # Drop a small results JSON for the report
    (OUT_DIR / "phase3_train_results.json").write_text(json.dumps({
        "winner": best_name,
        "results": results,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }, indent=2))
    print(f"saved metrics         : {(OUT_DIR / 'phase3_train_results.json').relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
