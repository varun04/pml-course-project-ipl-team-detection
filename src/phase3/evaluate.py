"""T3.4 — Evaluate the saved model on held-out images.

Loads models/model_group_18_varun.pkl, scores it on the test split, and
produces:

  outputs/phase3_classification_report.csv  per-class precision / recall / F1
  outputs/phase3_confusion_matrix.png       11x11 confusion matrix (heatmap)
  outputs/phase3_compare.png                bar chart of macro-F1 per classifier

Also evaluates the two non-winning classifiers (logreg/svm/rf) so the slide
deck has the comparison.

Usage:
    python src/phase3/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

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
    return X, y, image_idx, split


def predict(model_pkl_path, X_test):
    payload = joblib.load(model_pkl_path)
    model = payload["model"]
    scaler = payload.get("scaler")
    X_in = scaler.transform(X_test) if scaler is not None else X_test
    return model.predict(X_in)


def main() -> int:
    X, y, image_idx, split = load_arrays()
    is_test = split[image_idx] == 1
    X_test, y_test = X[is_test], y[is_test]
    print(f"evaluating on {len(y_test):,} test cells from {int(split.sum())} test images")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The deliverable
    payload = joblib.load(MODEL_DIR / "model_group_18_varun.pkl")
    winner = payload["model_name"]
    print(f"deliverable winner: {winner}")

    y_pred = predict(MODEL_DIR / "model_group_18_varun.pkl", X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0,
                                   labels=list(CLASS_NAMES.keys()),
                                   target_names=[CLASS_NAMES[i] for i in CLASS_NAMES])
    rows = []
    for label, idx in zip(CLASS_NAMES.values(), CLASS_NAMES.keys()):
        r = report[label]
        rows.append({
            "class_idx": idx,
            "class_name": label,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    for k in ("macro avg", "weighted avg"):
        r = report[k]
        rows.append({
            "class_idx": -1,
            "class_name": k,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(OUT_DIR / "phase3_classification_report.csv", index=False)
    print(f"\nclassification report:\n{rep_df.to_string(index=False)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_NAMES.keys()))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[CLASS_NAMES[i] for i in CLASS_NAMES],
                yticklabels=[CLASS_NAMES[i] for i in CLASS_NAMES],
                ax=axes[0], cbar=False)
    axes[0].set_title(f"Confusion matrix (counts) — {winner}, {len(y_test):,} test cells")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[CLASS_NAMES[i] for i in CLASS_NAMES],
                yticklabels=[CLASS_NAMES[i] for i in CLASS_NAMES],
                ax=axes[1], cbar=False, vmin=0, vmax=1)
    axes[1].set_title("Row-normalised (recall)")
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase3_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'phase3_confusion_matrix.png'}")

    # Comparison plot across the three classifiers
    train_results = json.loads((OUT_DIR / "phase3_train_results.json").read_text())
    names = ["logreg", "svm", "rf"]
    macro_f1s = [train_results["results"][n]["macro_f1"] for n in names]
    fit_times = [train_results["results"][n]["fit_seconds"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4c8dff" if n == winner else "#9aa3b2" for n in names]
    bars = ax.bar(names, macro_f1s, color=colors, edgecolor="black", linewidth=0.5)
    for b, f, t in zip(bars, macro_f1s, fit_times):
        ax.text(b.get_x() + b.get_width() / 2, f, f"{f:.3f}\n({t}s)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(macro_f1s) * 1.2)
    ax.set_ylabel("macro-F1 on held-out images")
    ax.set_title(f"Classifier comparison — winner: {winner}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase3_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'phase3_compare.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
