"""Player detection utilities — Phase 3/4.

Three complementary approaches:

1. HOG person detector (cv2 built-in):
   Classical sliding-window HOG + SVM. Gives pixel-level bounding boxes.
   Works well for upright players; misses lying/crouching poses.

2. Connected-component analysis on cell predictions:
   Groups adjacent non-zero cells per team into blobs. Each blob = one
   player region (or a cluster of same-team players).

3. PCA pose estimation (classical):
   For each cell blob, runs PCA on the (row, col) cell coordinates.
   The first principal component gives the blob's main axis direction.
   Angle from vertical → pose class (upright / crouching / lying).
   Each pose uses a different cells-per-player divisor:
     upright   → ÷ 9   (normal standing height in frame)
     crouching → ÷ 7   (shorter, wider stance)
     lying     → ÷ 5   (horizontal, fewer rows activated)

All three are combined in summarize_image(), the main public API.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import Counter
from scipy.ndimage import label as scipy_label

CW, CH   = 100, 75
N_ROWS   = 8
N_COLS   = 8
TEAMS    = ['none', 'CSK', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RR', 'RCB', 'SRH']

# Cells-per-player divisor (calibrated on broadcast cricket images).
# Close-up players fill ~12 cells; distant players ~9; dataset average ~10.
# ÷12 gives the best accuracy across varied shot distances (4/6 tested images).
# Pose-adjusted divisors were tested but hurt accuracy — grouped upright players
# look "lying" to PCA, causing undercounts — so a fixed divisor is used.
AVG_CELLS_PER_PLAYER = 12
CELLS_PER_PLAYER = {'upright': 12, 'crouching': 12, 'lying': 12}  # kept for API compat

# 8-connectivity — diagonal neighbours count as connected.
_STRUCT8 = np.ones((3, 3), dtype=int)


# ── 0. PCA pose estimation ────────────────────────────────────────────────────

def estimate_pose(cell_indices: list[int],
                  n_cols: int = N_COLS) -> tuple[str, float]:
    """Estimate player pose from blob cell indices using PCA orientation.

    Parameters
    ----------
    cell_indices : flat cell indices (0–63, row-major).
    n_cols       : grid width (default 8).

    Returns
    -------
    pose  : 'upright' | 'crouching' | 'lying'
    angle : degrees from vertical (0° = perfectly upright, 90° = lying flat)

    Method
    ------
    Convert flat indices → (row, col) coordinates, centre them, run SVD.
    The first right-singular vector is the blob's principal axis.
    Angle between that axis and the vertical gives the lean angle.
    """
    if len(cell_indices) < 3:
        return 'upright', 0.0   # too few cells to estimate — assume upright

    rows = np.array([idx // n_cols for idx in cell_indices], dtype=float)
    cols = np.array([idx %  n_cols for idx in cell_indices], dtype=float)
    coords = np.column_stack([rows, cols])
    coords -= coords.mean(axis=0)

    # SVD on the centred coordinate matrix
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    main_axis = vt[0]   # (drow, dcol) of principal component

    # Angle from vertical: vertical = (1, 0) in (row, col) space
    # arctan2(|dcol|, |drow|) gives angle from vertical axis
    angle = float(np.degrees(np.arctan2(abs(main_axis[1]), abs(main_axis[0]))))

    if angle < 30:
        pose = 'upright'
    elif angle > 60:
        pose = 'lying'
    else:
        pose = 'crouching'

    return pose, round(angle, 1)


# ── 1. HOG person detector ────────────────────────────────────────────────────

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Non-maximum suppression — removes duplicate HOG detections."""
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0].astype(float), boxes[:, 1].astype(float)
    x2, y2 = x1 + boxes[:, 2],          y1 + boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = np.argsort(-scores)
    keep   = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def detect_players_hog(img_bgr: np.ndarray,
                       win_stride: tuple = (8, 8),
                       padding: tuple   = (8, 8),
                       scale: float     = 1.05,
                       nms_thresh: float = 0.40) -> list[tuple]:
    """Detect people using cv2's built-in HOG + SVM pedestrian detector.

    Returns list of (x, y, w, h) pixel bounding boxes after NMS.
    An empty list means no people were detected.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(
        img_bgr, winStride=win_stride, padding=padding, scale=scale
    )
    if len(boxes) == 0:
        return []
    keep = _nms(boxes, weights.flatten(), nms_thresh)
    return [tuple(int(v) for v in boxes[i]) for i in keep]


# ── 2. Cell-based connected-component detection ───────────────────────────────

def detect_players_cells(cell_preds_64: np.ndarray,
                          min_cells: int = 2) -> list[dict]:
    """Detect players from cell predictions via connected-component analysis.

    Parameters
    ----------
    cell_preds_64 : 1-D array of 64 integers (0 = none, 1–10 = team label).
    min_cells     : blobs smaller than this are treated as background noise.

    Returns
    -------
    List of player dicts (sorted largest → smallest), each containing:
      team_id      int       IPL team label (1–10)
      team_name    str       e.g. 'CSK'
      n_cells      int       number of grid cells occupied
      cell_indices list[int] flat cell indices (0–63, row-major)
      cell_bbox    tuple     (row_min, col_min, row_max, col_max) in grid coords
      pixel_bbox   tuple     (x_min, y_min, x_max, y_max) in pixel coords
    """
    grid   = cell_preds_64.reshape(N_ROWS, N_COLS)
    binary = (grid != 0).astype(np.int8)

    # 8-connectivity so diagonally adjacent cells belong to the same player
    labeled, n_blobs = scipy_label(binary, structure=_STRUCT8)

    players = []
    for blob_id in range(1, n_blobs + 1):
        mask         = labeled == blob_id
        cell_indices = np.where(mask.flatten())[0]

        if len(cell_indices) < min_cells:
            continue                        # too small — likely noise

        labels_in_blob = grid[mask]
        non_zero       = labels_in_blob[labels_in_blob != 0]
        if len(non_zero) == 0:
            continue
        team_id = int(Counter(non_zero.tolist()).most_common(1)[0][0])

        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        # PCA pose estimation — for reporting only, does not affect count.
        # Pose-adjusted divisors were tested but hurt accuracy on groups
        # (grouped upright players look "lying" to PCA → undercount).
        pose, angle = estimate_pose(cell_indices.tolist())

        # Fixed divisor ÷12: best accuracy across varied shot distances.
        est_count = max(1, round(len(cell_indices) / AVG_CELLS_PER_PLAYER))

        players.append({
            'team_id':      team_id,
            'team_name':    TEAMS[team_id],
            'n_cells':      int(len(cell_indices)),
            'pose':         pose,
            'angle_deg':    angle,
            'est_players':  est_count,
            'cell_indices': cell_indices.tolist(),
            'cell_bbox':    (r0, c0, r1, c1),
            'pixel_bbox':   (c0 * CW, r0 * CH, (c1+1)*CW, (r1+1)*CH),
        })

    players.sort(key=lambda p: -p['n_cells'])
    return players


# ── 3. HOG-box → cell mapping ─────────────────────────────────────────────────

def hog_box_to_cells(hog_bbox: tuple, cell_preds_64: np.ndarray) -> dict | None:
    """Map one HOG bounding box to the cell grid and assign a team.

    Returns None if no non-zero cells fall within the box.
    """
    x, y, w, h = hog_bbox
    grid        = cell_preds_64.reshape(N_ROWS, N_COLS)

    # which cell columns/rows does this box span?
    c0 = max(0, x // CW);      c1 = min(N_COLS - 1, (x + w - 1) // CW)
    r0 = max(0, y // CH);      r1 = min(N_ROWS - 1, (y + h - 1) // CH)

    region  = grid[r0:r1+1, c0:c1+1]
    nonzero = region[region != 0]
    if len(nonzero) == 0:
        return None

    team_id = int(Counter(nonzero.tolist()).most_common(1)[0][0])
    return {
        'team_id':   team_id,
        'team_name': TEAMS[team_id],
        'hog_bbox':  hog_bbox,
        'cell_bbox': (r0, c0, r1, c1),
    }


# ── 4. Combined summary ───────────────────────────────────────────────────────

def summarize_image(img_bgr: np.ndarray,
                    cell_preds_64: np.ndarray,
                    run_hog: bool  = True,
                    min_cells: int = 2) -> dict:
    """Full player summary for one image.

    Parameters
    ----------
    img_bgr       : 800×600 BGR image.
    cell_preds_64 : 1-D int array of 64 cell labels from our model.
    run_hog       : whether to run the HOG detector (adds ~50 ms/image).
    min_cells     : minimum blob size for connected-component detection.

    Returns
    -------
    {
      n_players_cells  : int   — player count from connected components
      n_players_hog    : int   — player count from HOG detector
      players          : list  — player dicts from connected-component analysis
      hog_detections   : list  — team assignments for each HOG box
      hog_boxes        : list  — raw (x,y,w,h) HOG bounding boxes
      teams_present    : list  — unique team names detected (sorted)
    }
    """
    # Primary: cell-based detection (always runs)
    players      = detect_players_cells(cell_preds_64, min_cells=min_cells)
    teams_present = sorted(set(p['team_name'] for p in players))

    # Supplementary: HOG bounding boxes (optional, adds pixel-level localisation)
    hog_boxes      = detect_players_hog(img_bgr) if run_hog else []
    hog_detections = [d for box in hog_boxes
                      if (d := hog_box_to_cells(box, cell_preds_64)) is not None]

    # total estimated players = sum of per-blob estimates
    n_est = sum(p['est_players'] for p in players)

    return {
        'n_players_estimated': n_est,           # best overall estimate
        'n_players_cells':     len(players),    # raw blob count
        'n_players_hog':       len(hog_boxes),  # HOG box count
        'players':             players,
        'hog_detections':      hog_detections,
        'hog_boxes':           hog_boxes,
        'teams_present':       teams_present,
    }
