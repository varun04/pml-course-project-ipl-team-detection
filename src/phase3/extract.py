"""T3.1 — Per-cell hand-crafted feature extraction.

Input  : an 800x600 BGR image + cell index (0..63, row-major).
Output : a fixed-length feature vector for that 100x75 cell.

Feature blocks (total 67 floats, all normalized to [0, 1] where applicable):

  H histogram   32  — Hue channel binned (0..180), per-cell normalized.
  S histogram    8  — Saturation channel binned.
  V histogram    8  — Value channel binned.
  Mean BGR       3  — average B, G, R / 255.
  Std  BGR       3  — std B, G, R / 255.
  Position       2  — (row, col) / 7.
  LBP            10 — uniform LBP (P=8, R=1) histogram, density.
  Edge density   1  — Canny edge pixel fraction.

The same routine is used by build_dataset (to make training arrays) and by
pipeline.py (to extract features for an unseen image at inference time).
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import local_binary_pattern

CW, CH = 100, 75
N_CELLS = 64
HSV_H_BINS = 32
HSV_S_BINS = 8
HSV_V_BINS = 8
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2  # 10 for uniform method
FEATURE_DIM = HSV_H_BINS + HSV_S_BINS + HSV_V_BINS + 3 + 3 + 2 + LBP_BINS + 1  # = 67

FEATURE_NAMES = (
    [f"h_{i}" for i in range(HSV_H_BINS)]
    + [f"s_{i}" for i in range(HSV_S_BINS)]
    + [f"v_{i}" for i in range(HSV_V_BINS)]
    + ["mean_b", "mean_g", "mean_r"]
    + ["std_b", "std_g", "std_r"]
    + ["pos_row", "pos_col"]
    + [f"lbp_{i}" for i in range(LBP_BINS)]
    + ["edge_density"]
)


def cell_bbox(cell_idx: int) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) pixel bounds for the given 0-based cell index."""
    r, c = divmod(cell_idx, 8)
    return r * CH, (r + 1) * CH, c * CW, (c + 1) * CW


def extract_cell_features(cell_bgr: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract a feature vector for a single 75x100 BGR cell patch."""
    if cell_bgr.shape[:2] != (CH, CW):
        raise ValueError(f"expected ({CH},{CW}) patch, got {cell_bgr.shape[:2]}")

    out = np.empty(FEATURE_DIM, dtype=np.float32)
    pos = 0

    # 1. HSV histograms (marginal), normalized to a probability mass
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    pixels = float(cell_bgr.shape[0] * cell_bgr.shape[1])
    h_hist = cv2.calcHist([hsv], [0], None, [HSV_H_BINS], [0, 180]).flatten() / pixels
    s_hist = cv2.calcHist([hsv], [1], None, [HSV_S_BINS], [0, 256]).flatten() / pixels
    v_hist = cv2.calcHist([hsv], [2], None, [HSV_V_BINS], [0, 256]).flatten() / pixels
    out[pos:pos + HSV_H_BINS] = h_hist; pos += HSV_H_BINS
    out[pos:pos + HSV_S_BINS] = s_hist; pos += HSV_S_BINS
    out[pos:pos + HSV_V_BINS] = v_hist; pos += HSV_V_BINS

    # 2. Mean and std BGR (normalized to [0,1])
    out[pos:pos + 3] = cell_bgr.reshape(-1, 3).mean(axis=0) / 255.0; pos += 3
    out[pos:pos + 3] = cell_bgr.reshape(-1, 3).std(axis=0) / 255.0; pos += 3

    # 3. Position normalized to [0,1]
    out[pos] = row / 7.0; pos += 1
    out[pos] = col / 7.0; pos += 1

    # 4. LBP (uniform), normalized histogram
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    # density=True returns frequency-density; multiply by bin width (1) to get proportions
    out[pos:pos + LBP_BINS] = lbp_hist; pos += LBP_BINS

    # 5. Canny edge density
    edges = cv2.Canny(gray, 80, 160)
    out[pos] = (edges > 0).mean(); pos += 1

    assert pos == FEATURE_DIM
    return out


def extract_image_features(img_bgr: np.ndarray) -> np.ndarray:
    """For an 800x600 BGR image, return (64, FEATURE_DIM) per-cell features."""
    if img_bgr.shape[:2] != (600, 800):
        raise ValueError(f"expected 800x600 image, got {img_bgr.shape[:2]}")
    out = np.empty((N_CELLS, FEATURE_DIM), dtype=np.float32)
    for idx in range(N_CELLS):
        y0, y1, x0, x1 = cell_bbox(idx)
        r, c = divmod(idx, 8)
        out[idx] = extract_cell_features(img_bgr[y0:y1, x0:x1], r, c)
    return out


if __name__ == "__main__":
    # quick self-test
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:, :, 0] = 50  # arbitrary
    feats = extract_image_features(img)
    print(f"feature dim: {FEATURE_DIM}")
    print(f"output shape: {feats.shape}")
    print(f"first 5 features (cell 0): {feats[0, :5]}")
