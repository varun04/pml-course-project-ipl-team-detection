"""T3.1 — Per-cell hand-crafted feature extraction.

Input  : an 800x600 BGR image + cell index (0..63, row-major).
Output : a fixed-length feature vector for that 100x75 cell.

Feature blocks (total 194 floats, all normalized to [0, 1] where applicable):

  H histogram      32  — Hue channel binned (0..180), per-cell normalized.
  S histogram       8  — Saturation channel binned.
  V histogram       8  — Value channel binned.
  Mean BGR          3  — average B, G, R / 255.
  Std  BGR          3  — std B, G, R / 255.
  Position          2  — (row, col) / 7.
  LBP              10  — uniform LBP (P=8, R=1) histogram, density.
  Edge density      1  — Canny edge pixel fraction.
  HOG              96  — Histogram of Oriented Gradients (8 orientations,
                          25x25 px cells, 1x1 block) — jersey texture/stripes.
  Secondary clr     5  — silver_frac, black_frac, gold_frac, sat_p25, val_p25.
  Dominant pair     8  — k-means k=2 on HSV: (H,S,V,proportion) × 2 clusters,
                          sorted by proportion desc. Captures the two dominant
                          colours and their balance — key for red/navy split.
  Red-zone hist    16  — 8 bins in H[0,25] + 8 bins in H[155,180]. Finer
                          resolution exactly where PBKS and RCB differ in hue.
  Checkered         2  — Response to 2×2 and 4×4 checkerboard kernels.
                          PBKS has a distinctive checkered glove/pad pattern.

The same routine is used by build_dataset (to make training arrays) and by
pipeline.py (to extract features for an unseen image at inference time).
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

CW, CH = 100, 75
N_CELLS = 64
HSV_H_BINS = 32
HSV_S_BINS = 8
HSV_V_BINS = 8
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2          # 10 for uniform method

# HOG: 75×100 cell → 3×4 = 12 cells of 25×25 px, 1×1 block, 8 orientations → 96
HOG_ORIENTATIONS = 8
HOG_PPC          = (25, 25)
HOG_CPB          = (1, 1)
HOG_DIM          = 96

# Secondary colour fractions
SEC_COLOR_DIM = 5

# Dominant colour pair: k=2 k-means on HSV → (H,S,V,prop) × 2, sorted by prop desc
DOM_COLOR_DIM = 8   # 2 clusters × 4 values each

# Red-zone precision histogram: 8 bins in warm-red [0,25] + 8 bins in cool-red [155,180]
RED_ZONE_BINS = 8
RED_ZONE_DIM  = RED_ZONE_BINS * 2   # = 16

# Checkered pattern: response to 2×2 and 4×4 alternating kernels
CHECK_DIM = 2

# Pre-build checkered kernels at module load (avoids repeated allocation)
_KERN_CHECK2 = np.array([[1, -1], [-1, 1]], dtype=np.float32)
_KERN_CHECK4 = np.array(
    [[1,  1, -1, -1],
     [1,  1, -1, -1],
     [-1, -1,  1,  1],
     [-1, -1,  1,  1]], dtype=np.float32) / 4.0

FEATURE_DIM = (
    HSV_H_BINS + HSV_S_BINS + HSV_V_BINS   # 48 — colour histograms
    + 3 + 3 + 2                              # 8  — mean BGR, std BGR, position
    + LBP_BINS + 1                           # 11 — LBP, edge density   → 67 original
    + HOG_DIM                                # 96 — jersey texture
    + SEC_COLOR_DIM                          # 5  — silver/black/gold fractions
    + DOM_COLOR_DIM                          # 8  — dominant colour pair
    + RED_ZONE_DIM                           # 16 — red-zone precision
    + CHECK_DIM                              # 2  — checkered pattern
)                                            # = 194 total

FEATURE_NAMES = (
    [f"h_{i}"       for i in range(HSV_H_BINS)]
    + [f"s_{i}"     for i in range(HSV_S_BINS)]
    + [f"v_{i}"     for i in range(HSV_V_BINS)]
    + ["mean_b", "mean_g", "mean_r"]
    + ["std_b",  "std_g",  "std_r"]
    + ["pos_row", "pos_col"]
    + [f"lbp_{i}"  for i in range(LBP_BINS)]
    + ["edge_density"]
    + [f"hog_{i}"  for i in range(HOG_DIM)]
    + ["silver_frac", "black_frac", "gold_frac", "sat_p25", "val_p25"]
    + ["dom1_h", "dom1_s", "dom1_v", "dom1_prop",
       "dom2_h", "dom2_s", "dom2_v", "dom2_prop"]
    + [f"rz_warm_{i}" for i in range(RED_ZONE_BINS)]
    + [f"rz_cool_{i}" for i in range(RED_ZONE_BINS)]
    + ["check2", "check4"]
)

assert len(FEATURE_NAMES) == FEATURE_DIM, f"{len(FEATURE_NAMES)} != {FEATURE_DIM}"


def cell_bbox(cell_idx: int) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) pixel bounds for the given 0-based cell index."""
    r, c = divmod(cell_idx, 8)
    return r * CH, (r + 1) * CH, c * CW, (c + 1) * CW


def extract_cell_features(cell_bgr: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract a 194-d feature vector for a single 75×100 BGR cell patch."""
    if cell_bgr.shape[:2] != (CH, CW):
        raise ValueError(f"expected ({CH},{CW}) patch, got {cell_bgr.shape[:2]}")

    out = np.empty(FEATURE_DIM, dtype=np.float32)
    pos = 0

    # ── 1. HSV histograms (marginal), normalised to probability mass ──────────
    hsv    = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    pixels = float(cell_bgr.shape[0] * cell_bgr.shape[1])
    h_hist = cv2.calcHist([hsv], [0], None, [HSV_H_BINS], [0, 180]).flatten() / pixels
    s_hist = cv2.calcHist([hsv], [1], None, [HSV_S_BINS], [0, 256]).flatten() / pixels
    v_hist = cv2.calcHist([hsv], [2], None, [HSV_V_BINS], [0, 256]).flatten() / pixels
    out[pos:pos + HSV_H_BINS] = h_hist; pos += HSV_H_BINS
    out[pos:pos + HSV_S_BINS] = s_hist; pos += HSV_S_BINS
    out[pos:pos + HSV_V_BINS] = v_hist; pos += HSV_V_BINS

    # ── 2. Mean and std BGR (normalised to [0,1]) ─────────────────────────────
    out[pos:pos + 3] = cell_bgr.reshape(-1, 3).mean(axis=0) / 255.0; pos += 3
    out[pos:pos + 3] = cell_bgr.reshape(-1, 3).std(axis=0)  / 255.0; pos += 3

    # ── 3. Position normalised to [0,1] ───────────────────────────────────────
    out[pos] = row / 7.0; pos += 1
    out[pos] = col / 7.0; pos += 1

    # ── 4. LBP (uniform), normalised histogram ────────────────────────────────
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS,
                               range=(0, LBP_BINS), density=True)
    out[pos:pos + LBP_BINS] = lbp_hist; pos += LBP_BINS

    # ── 5. Canny edge density ─────────────────────────────────────────────────
    edges = cv2.Canny(gray, 80, 160)
    out[pos] = (edges > 0).mean(); pos += 1

    # ── 6. HOG — jersey stripe / texture direction histogram ──────────────────
    hog_feat = hog(gray, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PPC, cells_per_block=HOG_CPB,
                   feature_vector=True)
    out[pos:pos + HOG_DIM] = hog_feat; pos += HOG_DIM

    # ── 7. Secondary colour fractions — PBKS vs RCB discriminators ────────────
    hsv_flat = hsv.reshape(-1, 3).astype(np.float32)
    h_n = hsv_flat[:, 0] / 180.0
    s_n = hsv_flat[:, 1] / 255.0
    v_n = hsv_flat[:, 2] / 255.0

    # silver/gray: low saturation AND high brightness  (PBKS secondary colour)
    silver_frac = float(((s_n < 0.25) & (v_n > 0.60)).mean())
    # black: very dark pixels  (RCB secondary colour)
    black_frac  = float((v_n < 0.20).mean())
    # gold/yellow: hue ~20-40°, saturated and bright  (RCB accent colour)
    gold_frac   = float(((h_n > 0.11) & (h_n < 0.22) & (s_n > 0.40) & (v_n > 0.40)).mean())
    # saturation 25th percentile — lower for PBKS (silver dilutes it)
    sat_p25     = float(np.percentile(s_n, 25))
    # value 25th percentile — lower for RCB (black pixels pull it down)
    val_p25     = float(np.percentile(v_n, 25))

    out[pos] = silver_frac; pos += 1
    out[pos] = black_frac;  pos += 1
    out[pos] = gold_frac;   pos += 1
    out[pos] = sat_p25;     pos += 1
    out[pos] = val_p25;     pos += 1

    # ── 8. Dominant colour pair (k-means k=2 on HSV) ──────────────────────────
    # Finds the two most-present colours and their proportion.
    # PBKS dominant pair: (scarlet-red, navy-blue)
    # RCB dominant pair:  (crimson-red, black) — different V on the dark cluster
    pixels_hsv = hsv_flat.copy()
    criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centres = cv2.kmeans(
        pixels_hsv, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=2).astype(np.float32)
    order  = np.argsort(-counts)           # sort largest cluster first
    total  = counts.sum()
    for k in order:
        out[pos]     = centres[k, 0] / 180.0   # H normalised
        out[pos + 1] = centres[k, 1] / 255.0   # S normalised
        out[pos + 2] = centres[k, 2] / 255.0   # V normalised
        out[pos + 3] = counts[k] / total        # proportion
        pos += 4

    # ── 9. Red-zone precision histogram ───────────────────────────────────────
    # Standard h_hist bins are 5.6° wide — too coarse to separate PBKS/RCB red.
    # These bins are 3.1° wide, focused on the red wraparound zones.
    h_chan = hsv[:, :, 0]   # uint8 in [0,180]
    warm_hist = cv2.calcHist([h_chan], [0], None,
                             [RED_ZONE_BINS], [0, 25]).flatten() / pixels
    cool_hist = cv2.calcHist([h_chan], [0], None,
                             [RED_ZONE_BINS], [155, 180]).flatten() / pixels
    out[pos:pos + RED_ZONE_BINS] = warm_hist; pos += RED_ZONE_BINS
    out[pos:pos + RED_ZONE_BINS] = cool_hist; pos += RED_ZONE_BINS

    # ── 10. Checkered pattern score ────────────────────────────────────────────
    # PBKS has a distinctive black/white checkered grip pattern on gloves/pads.
    # A checkerboard kernel gives high response on alternating patches.
    gray_f = gray.astype(np.float32) / 255.0
    resp2  = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, _KERN_CHECK2)).mean()
    resp4  = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, _KERN_CHECK4)).mean()
    out[pos] = float(resp2); pos += 1
    out[pos] = float(resp4); pos += 1

    assert pos == FEATURE_DIM
    return out


def extract_image_features(img_bgr: np.ndarray) -> np.ndarray:
    """For an 800×600 BGR image, return (64, FEATURE_DIM) per-cell features."""
    if img_bgr.shape[:2] != (600, 800):
        raise ValueError(f"expected 800x600 image, got {img_bgr.shape[:2]}")
    out = np.empty((N_CELLS, FEATURE_DIM), dtype=np.float32)
    for idx in range(N_CELLS):
        y0, y1, x0, x1 = cell_bbox(idx)
        r, c = divmod(idx, 8)
        out[idx] = extract_cell_features(img_bgr[y0:y1, x0:x1], r, c)
    return out


if __name__ == "__main__":
    import time
    img  = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    t0   = time.time()
    for _ in range(5):
        feats = extract_image_features(img)
    per_img = (time.time() - t0) / 5
    print(f"feature dim  : {FEATURE_DIM}")
    print(f"output shape : {feats.shape}")
    print(f"any NaN/Inf  : {np.isnan(feats).any() or np.isinf(feats).any()}")
    print(f"speed        : {per_img:.3f}s/image  (~{per_img*2724/60:.1f} min for 2724 images)")
