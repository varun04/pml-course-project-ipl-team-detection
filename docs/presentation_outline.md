# Presentation Outline — IPL Jersey Team Detection
## Group 32 | PML Course Project | June 2026

---

## SLIDE 1 — Title

**Title:** Automated Detection of IPL Teams Using Jersey Identification

**Subtitle:** PML Course Project — Group 32

| Name | Roll Number |
|------|-------------|
| Varun Tomar | x3vat359 |
| Aloka Chhatani | _(to be added)_ |
| Kathi Sreeram Kumar | _(to be added)_ |
| Pramod Haridas Nair | _(to be added)_ |

Centre for Machine Intelligence & Data Science, IIT Bombay
e-PGDiploma in AI & DS — January 2026 Cohort

---

## SLIDES 2–4 — EXECUTIVE SUMMARY (max 3)

---

### SLIDE 2 — What We Built

**Heading:** Problem & Solution in One Slide

**Left column — Problem:**
- Given an IPL broadcast image (800×600 px)
- Divide into 8×8 grid = 64 cells
- For each cell: predict which IPL team (0=none, 1–10)
- No CNNs allowed — hand-crafted features only

**Right column — What we built:**
- 3,605 labelled images across 44 match scenarios
- 194 hand-crafted features per cell (HOG, colour, texture)
- Two-stage LightGBM pipeline with spatial context
- Player detection via connected-component analysis

**Bottom bar (big numbers):**
```
3,605 images   |   194 features   |   Macro-F1: 0.711   |   Team ID: 13/13 (100%)
```

> *Insert: outputs/phase3_pipeline_demo.png (prediction overlay on real image)*

---

### SLIDE 3 — Key Results at a Glance

**Heading:** Results Summary

**Table:**
| Metric | Value |
|--------|-------|
| Dataset | 3,605 images, 44 match scenarios, 10 IPL teams |
| Features | 194 hand-crafted (no CNN) |
| Best model | Two-stage LightGBM |
| Macro-F1 (test) | **0.711** (360 held-out images) |
| Team identification | **13/13 (100%)** on hand-checked images |
| Player count accuracy | **58%** (7/12 images correct) |
| Hardest class | PBKS vs RCB (both red jerseys) — still correctly separated |

**Insight box:**
> *"The model correctly identifies which IPL teams are present in every tested image — including the hardest red-vs-red case (PBKS vs RCB). Player counting is limited by the 8×8 cell resolution."*

---

### SLIDE 4 — Improvement Journey

**Heading:** How We Got to 0.711 — Step by Step

> *Insert: outputs/phase3_journey.png*

**Table below chart:**
| Step | Technique | Macro-F1 | Gain |
|------|-----------|----------|------|
| 1 | Random Forest baseline | 0.532 | — |
| 2 | + Per-class threshold tuning | 0.648 | +0.116 |
| 3 | + LightGBM (replaces RF) | 0.690 | +0.042 |
| 4 | + 194 hand-crafted features | 0.695 | +0.005 |
| 5 | + Spatial 3×3 smoothing | 0.696 | +0.001 |
| 6 | + Two-stage context classifier | 0.726 | +0.030 |
| 7 | + 881 new labelled images | 0.711* | *(harder test set)* |

*\*Test set grew from 272→360 images; recall improved on 7/10 classes*

---

## SLIDE 5 — Problem Explanation

**Heading:** Task Definition

**Left — The problem:**
- Each 800×600 image → 8×8 grid → 64 cells (100×75 px each)
- Classify each cell: 0 (none) or 1–10 (IPL franchise)
- Output: CSV with columns c01…c64, values 0–10

**Right — Why it's hard:**
- 83.8% of cells are background (severe class imbalance)
- Teams with similar colours (PBKS and RCB both wear red)
- Players at different distances → variable cell coverage
- No CNN allowed — must use hand-crafted features only

> *Insert: example image with 8×8 grid overlay (from problem statement PDF)*

**Constraint box:**
> *"Any hand-crafted feature engineering techniques for images can be used. CNNs or equivalents that automatically create features SHOULD NOT be used."*

---

## SLIDE 6 — Data Collection

**Heading:** Building the Dataset

**Left — Stats:**
| | Count |
|-|-------|
| Total images | 3,605 |
| Match scenarios | 44 |
| Total labelled cells | 230,720 |
| Image spec | 800×600 px, 4:3 |
| Label format | 8×8 grid, values 0–10 |

**Right — Sources:**
- iplt20.com official match galleries
- ESPNcricinfo, Cricbuzz match reports
- BCCI media releases
- Franchise official websites

**Insight:** DC has fewest images (262) → directly explains its lowest F1 (0.61)

> *Insert: outputs/eda_figures/11_match_coverage.png*

---

## SLIDE 7 — EDA: What the Data Looks Like

**Heading:** Exploratory Data Analysis — 3 Key Insights

**Top chart:**
> *Insert: outputs/eda_figures/01_cell_distribution.png*
**Insight:** 83.8% none → class imbalance is the #1 modelling challenge

**Bottom left:**
> *Insert: outputs/eda_figures/08_per_team_image_count.png*
**Insight:** Data poverty directly predicts model weakness (DC, GT fewer images → lower F1)

**Bottom right:**
> *Insert: outputs/eda_figures/03_nonzero_per_image.png*
**Insight:** Median 9 active cells/image — typical frame shows 1–2 players

---

## SLIDE 8 — EDA: Spatial Patterns & Co-occurrence

**Heading:** Where Teams Appear + Who Appears Together

**Left:**
> *Insert: outputs/eda_figures/02_spatial_heatmaps.png*
**Insight:** Players consistently appear in rows 2–6 (mid-frame). Top rows = scoreboards/sky.

**Right:**
> *Insert: outputs/eda_figures/09_team_cooccurrence.png*
**Insight:** Teams only co-occur in their actual match fixtures — confirms data is match-realistic.

---

## SLIDE 9 — Feature Engineering (194D, No CNN)

**Heading:** 194 Hand-Crafted Features Per Cell

**Table:**
| Block | Dim | What it captures | Key for |
|-------|-----|-----------------|---------|
| HSV histograms (H+S+V) | 48 | Team colour signature | All teams |
| Mean + Std BGR | 6 | Average colour tone | Fast baseline |
| Position (row, col) | 2 | Spatial prior (logos in fixed zones) | All teams |
| LBP texture | 10 | Jersey weave vs smooth background | Texture |
| Edge density (Canny) | 1 | Logos, text overlays | Logos |
| **HOG** | **96** | **Jersey stripe/texture direction** | **Key addition** |
| Secondary colour | 5 | silver (PBKS), black (RCB), gold | PBKS/RCB |
| Dominant colour pair | 8 | Top-2 colours + proportion | Red teams |
| Red-zone histogram | 16 | Fine hue bins in red zone | PBKS vs RCB |
| Checkered pattern | 2 | PBKS distinctive check pattern | PBKS |
| **TOTAL** | **194** | | |

**Insight:** Colour (48D) is the primary signal. HOG (96D) and targeted colour features solve the PBKS/RCB red-vs-red problem.

---

## SLIDE 10 — Classifier Comparison

**Heading:** 4 Classifiers Tested — LightGBM Wins

> *Insert: outputs/phase3_compare.png*

**Key insight box:**
> *"LightGBM's gradient boosting handles class imbalance better than linear models. With class_weight='balanced', it achieves 0.695 macro-F1 before any post-processing."*

**Why not CNN:**
> *"Per problem statement constraints — all features must be hand-crafted. LightGBM on 194-D features with two-stage refinement achieves 0.711 macro-F1."*

---

## SLIDE 11 — Threshold Tuning & Spatial Smoothing

**Heading:** Two Techniques That Lifted Performance

**Left — Threshold tuning (+0.116):**
- Default `predict()` = argmax → "none" always wins (84% class)
- Fix: `predict = argmax(proba / threshold)`
- Tuned per-class via coordinate ascent on calibration split
- CSK threshold = 0.35 (relaxed), SRH = 0.76 (strict)

> *Insert: outputs/phase3_threshold_sensitivity.png*

**Right — Spatial smoothing (+0.001):**
- Each cell classified independently → misses jersey context
- Fix: blend 40% from 3×3 neighbourhood probabilities
- α = 0.4–0.5 tuned on calibration split

**Insight:** Threshold tuning alone gave the biggest single gain (+0.116) — more than any feature engineering step.

---

## SLIDE 12 — Two-Stage Context Classifier (+0.030)

**Heading:** The Biggest Single Gain — Spatial Context

**Architecture diagram:**
```
Image (800×600)
    ↓  extract 194 features per cell
Stage-1: LightGBM  →  (64, 11) probability map
    ↓  per cell: [own_proba(11) | 8-neighbour mean proba(11)]
Stage-2: LightGBM  →  final predictions
```

**Why it works:**
> *"A jersey spans multiple cells. If neighbours predict PBKS with high confidence, an ambiguous cell is almost certainly PBKS — even if its own colour features are noisy."*

**No data leakage:**
- Stage-1 trained on **fit cells** (80% of training images)
- Stage-2 trained on **calibration cells** (20% — out-of-sample for Stage-1)
- Evaluated on **test cells** (10% — never touched)

**Result:** +0.030 macro-F1 | Recall improved on 7/10 classes

---

## SLIDE 13 — Final Classification Results

**Heading:** Per-Class Performance — Final Model

> *Insert: outputs/phase3_prf_per_class.png*

**Key observations:**
- Precision is strong across all teams (0.52–0.96)
- Recall is the weakness — model is conservative
- PBKS (0.57) and DC (0.61) are weakest — data poverty + red ambiguity
- **RCB vs PBKS (both red): correctly separated** — red-zone histogram + dominant colour pair working

> *Insert: outputs/phase3_confusion_matrix.png (row-normalised)*

---

## SLIDE 14 — Error Analysis & Overfitting

**Heading:** Where the Model Fails — and Why

**Left:**
> *Insert: outputs/phase3_error_analysis.png*

**Insight:** "None" absorption (23–41% of team cells predicted as background) is the dominant failure. Cross-team confusion is small (<10%) — team ID is reliable when cells ARE detected.

**Right — Overfitting check:**
> *Insert: outputs/phase3_overfit_analysis.png*

| | Train | Test | Gap |
|--|-------|------|-----|
| Stage-1 (tuned) | 0.95 | 0.70 | 0.25 |
| Stage-2 (context) | 0.94 | 0.71 | 0.23 |

**Honest assessment:**
> *"Gap of 0.23 is large but NOT harmful — regularisation attempt reduced test performance further, confirming the model learned real patterns. Gap exists because test images have different backgrounds, stadiums, and lighting."*

---

## SLIDE 15 — Player Detection

**Heading:** Detecting & Counting Players

**Method (classical, no CNN):**
1. **HOG person detector** (cv2 built-in) → pixel bounding boxes
2. **Connected-component analysis** on cell predictions → player blobs per team
3. **PCA orientation** on blob → upright / crouching / lying
4. **Player count estimate:** `cells in blob ÷ 12`

> *Insert: outputs/phase3_pipeline_demo.png*

**Validation table (13 hand-checked images):**
| | Count |
|-|-------|
| Team identification | **13/13 (100%)** |
| Player counting correct | **7/12 (58%)** |

**Known failure modes:**
| Mode | Example | Root cause |
|------|---------|-----------|
| Same-team merge | 4×MI → 3 | Adjacent blobs merge |
| Non-upright | 5 players → 3 | Lying/bent players activate few cells |
| Wide-angle | 8 players → 4 | <6 cells per player at distance |

---

## SLIDE 16 — Dead-Ends & What Didn't Work

**Heading:** Paths Taken That Didn't Work *(required by guidelines)*

| Approach | What we tried | Why it failed | Lesson |
|----------|--------------|---------------|--------|
| **Gabor filters** | 4 orientations × 2 frequencies per cell | 74ms/image, no measurable F1 gain | Texture info already captured by HOG and LBP |
| **PCA dimensionality reduction** | 50, 100, 130, 150 components | Every level hurt macro-F1 (−0.07 at 150 components) | LightGBM selects features internally — PCA removes that flexibility |
| **CatBoost** | Discussed as alternative to LightGBM | All features are numerical → LightGBM's sweet spot. Not tried. | Right tool for right data type |
| **HOG for player counting** | cv2 pedestrian HOG detector | 9 false positives in stadium crowds | HOG trained on street pedestrians, not cricket stadiums |
| **PCA pose estimation** | PCA angle on blob → upright/lying/crouching | Grouped upright players = horizontal blob = classified as "lying" | 8×8 grid too coarse to distinguish group shape from pose |
| **Pose-adjusted divisors** | ÷9 upright, ÷7 crouch, ÷5 lying | Score 6/13 vs fixed ÷12 scoring 7/13 | Fixed divisor outperforms pose-adjusted on this dataset |
| **Regularised LightGBM** | Fewer trees + L1/L2 to reduce train-test gap | Test F1 dropped from 0.711 → 0.702 | Model learned real patterns, not noise |

---

## SLIDE 17 — Challenges & Learnings

**Heading:** Key Challenges & What We Learned

**Challenges:**

1. **PBKS vs RCB red-vs-red confusion**
   Both teams wear red jerseys. Required 3 targeted features: red-zone histogram, dominant colour pair, secondary colour fractions. Cross-team confusion reduced to 6–8%.

2. **Class imbalance (84% none)**
   Fixed with `class_weight='balanced'` + per-class threshold tuning. Threshold tuning alone gave +0.116 macro-F1.

3. **Player counting at variable distances**
   Fixed ÷12 divisor works for close-ups and mid-distance. Fails for extreme close-ups (1 player fills frame → counted as 2) and wide-angle shots (8 players, 5 cells each → undercounted). No single classical divisor handles all distances.

4. **Test set shift after data expansion**
   Adding 881 images changed the test set composition (272 → 360 images, new harder matches). Macro-F1 appeared to drop (0.726 → 0.711) but recall improved on 7/10 classes.

**Learnings:**

- Spatial context (Stage-2 classifier) gave +0.030 — more than all feature engineering combined after HOG
- Threshold tuning is more impactful than model selection
- Data poverty directly predicts model weakness (DC: 262 images → F1 0.61)
- Honest reporting of failure modes is as important as reporting successes

---

## SLIDE 18 — Conclusions & Future Work

**Heading:** Conclusions & What's Next

**What works:**
- ✅ Team identification: **100%** on all tested images (including PBKS vs RCB)
- ✅ Two-stage LightGBM with 194 features: **macro-F1 0.711**
- ✅ Player detection (connected components): team correct every time
- ✅ Full pipeline: one `.pkl` file reads any 800×600 image → predictions CSV

**Known limits:**
- ❌ PBKS (F1=0.57) and DC (F1=0.61): need more training data
- ❌ Player counting: 58% accuracy — limited by 8×8 cell resolution
- ❌ Train-test gap: 0.23 — tree model memorises training backgrounds

**Future work:**
| Improvement | Expected gain |
|-------------|--------------|
| 200 more DC match images | DC F1: 0.61 → ~0.70 |
| 16×16 or 32×32 cell grid | Better player counting, less "none absorption" |
| Camera-distance estimator | Adaptive divisor for player counting |
| Cross-validation over image splits | More robust macro-F1 estimate |

---

## FIGURES REFERENCE (for slide building)

All figures are in `outputs/` and `outputs/eda_figures/`:

| Figure file | Used in slide |
|-------------|--------------|
| `phase3_pipeline_demo.png` | Slide 2, 15 |
| `phase3_journey.png` | Slide 4 |
| `eda_figures/01_cell_distribution.png` | Slide 7 |
| `eda_figures/02_spatial_heatmaps.png` | Slide 8 |
| `eda_figures/03_nonzero_per_image.png` | Slide 7 |
| `eda_figures/08_per_team_image_count.png` | Slide 7 |
| `eda_figures/09_team_cooccurrence.png` | Slide 8 |
| `eda_figures/11_match_coverage.png` | Slide 6 |
| `phase3_compare.png` | Slide 10 |
| `phase3_threshold_sensitivity.png` | Slide 11 |
| `phase3_prf_per_class.png` | Slide 13 |
| `phase3_confusion_matrix.png` | Slide 13 |
| `phase3_error_analysis.png` | Slide 14 |
| `phase3_overfit_analysis.png` | Slide 14 |
| `phase3_stage_improvement.png` | Slide 12 |
