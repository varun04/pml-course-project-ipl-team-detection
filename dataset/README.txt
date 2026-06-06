IPL Jersey Identification — Dataset README
==========================================

Course      : Programming for Machine Learning and Data Science (PMLDS)
Programme   : e-PGDiploma in AI & DS — January 2026 cohort
Institution : Centre for Machine Intelligence & Data Science, IIT Bombay
Group       : Group 32 — Varun Tomar
Season used : IPL Season 18 (2025) — single season, per FAQ Q4


Dataset summary
---------------
  Total images (processed, labelled) : 3,605
  Total labelled cells (64 per image): 230,720
  Match scenarios covered            : 44
  Image specification                : 800 x 600 px, 4:3 aspect ratio, JPEG
  Label format                       : 8x8 grid, 64 cells per image, values 0-10


Folder layout
-------------
dataset/
  raw/          Original downloads organised by match scenario (36 sub-folders,
                e.g. GTvsMI/, KKRvsCSK/, SRHvsMI/).
  processed/    Final 800x600 JPEG images used for labelling and model training,
                organised by match scenario (36 sub-folders, same structure as raw/).
  features/     Pre-computed feature arrays:
                  X.npy           (230720, 194) float32  per-cell feature matrix
                  y.npy           (230720,)     int8     cell labels 0-10
                  image_idx.npy   (230720,)     int32    maps cell -> image index
                  split.npy       (3605,)       int8     0=train, 1=test
                  image_names.npy (3605,)       str      filenames
                  meta.json                              feature names and sizes
  labels.csv    Ground-truth cell labels (one row per image, columns c01-c64).
  image_stats.csv  Per-image metadata (resolution, SHA256, non-zero cells, etc.)
  _external/    Class-wide master CSV files received from collaborating groups.


Label scheme (matches problem statement)
----------------------------------------
  0   No team / background / crowd / empty pitch
  1   Chennai Super Kings     (CSK)
  2   Delhi Capitals          (DC)
  3   Gujarat Titans          (GT)
  4   Kolkata Knight Riders   (KKR)
  5   Lucknow Super Giants    (LSG)
  6   Mumbai Indians          (MI)
  7   Punjab Kings            (PBKS)
  8   Rajasthan Royals        (RR)
  9   Royal Challengers Bengaluru (RCB)
  10  Sunrisers Hyderabad     (SRH)


Data collection — collaborative approach
-----------------------------------------
All course groups collaborated to build one shared class-wide dataset.
A dedicated labelling tool was created for the class — each group labelled
their portion of images and the labels were merged into a single master CSV.

Image sources (IPL 2025 season only):
  - iplt20.com — official IPL 2025 match photo galleries (primary source)
  - BCCI official media releases
  - ESPNcricinfo, Cricbuzz — match reports and galleries
  - Franchise official websites and social media

All images are used strictly for academic coursework.
No images are redistributed beyond the project submission package.
See docs/T1_1_collection_guide.md for the full source policy.


Dataset constraints (per problem statement)
-------------------------------------------
* Native resolution must be >= 800 x 600. Lower-resolution images excluded.
* All processed images are exactly 800 x 600 (4:3 aspect ratio).
* >= 100 images per franchise (all franchises meet this requirement).
* No CNN-based auto-feature methods used — all features are hand-crafted.


Per-team image counts (images containing at least one cell of that team)
------------------------------------------------------------------------
  Team         Label   Images   Meets >=100?
  ------------------------------------------
  no_team        0        66    (background-only images)
  CSK            1       404    YES
  DC             2       262    YES
  GT             3       316    YES
  KKR            4       554    YES
  LSG            5       324    YES
  MI             6       602    YES
  PBKS           7       575    YES
  RR             8       353    YES
  RCB            9       432    YES
  SRH           10       669    YES
  ------------------------------------------
  TOTAL              3,605 images


Match scenarios (44 total)
---------------------------
  GTvsLSG    :   72    KKRvsCSK   :  131    PBKSvsMI   :   72
  GTvsMI     :  106    KKRvsGT    :   52    RCBvsCSK   :   72
  GTvsPBKS   :   72    KKRvsLSG   :  140    RCBvsDC    :   63
  GTvsRR     :   72    KKRvsPBKS  :  201    RCBvsGT    :  104
  GTvsSRH    :   75    KKRvsRCB   :   67    RCBvsPBKS  :   72
  LSGvsCSK   :   45    KKRvsRR    :   72    RCBvsRR    :   72
  LSGvsMI    :   72    KKRvsSRH   :  255    RCBvsSRH   :   67
  LSGvsRCB   :   71    MIvsCSK    :   75    RRvsCSK    :   72
  LSGvsSRH   :   70    MIvsDC     :   90    RRvsLSG    :   83
  MIvsKKR    :   72    PBKSvsCSK  :  245    RRvsMI     :   72
  MIvsRCB    :   71    PBKSvsDC   :   98    SRHvsDC    :  158
                                            SRHvsMI    :  255
                                            SRHvsPBKS  :  147
                                            SRHvsRR    :   72
  ------------------------------------------------------------------
  TOTAL: 3,605 images across 44 match scenarios


Labelling methodology
---------------------
Each 800x600 image is divided into an 8x8 grid of 64 cells (100x75 px each).
Each cell is assigned a label 0-10 indicating the IPL team (or background).
Labels were assigned by the group using the class-wide labelling tool and
merged with the class master dataset for cross-group consistency.
In case multiple teams appear in a single cell, one team is recorded.


Feature engineering (194 features per cell, no CNNs)
------------------------------------------------------
  HSV histograms (H=32, S=8, V=8)    48 features
  Mean BGR                             3 features
  Std BGR                              3 features
  Position (row, col normalised)       2 features
  LBP texture (uniform, P=8, R=1)     10 features
  Canny edge density                   1 feature
  HOG (8 orient, 25x25 px cells)      96 features
  Secondary colour fractions           5 features
  Dominant colour pair (k-means k=2)   8 features
  Red-zone precision histogram        16 features
  Checkered pattern score              2 features
  TOTAL                              194 features


Model
-----
  Algorithm    : Two-stage LightGBM
  Stage 1      : LightGBM (500 trees) on 194 features, per-class threshold tuning
  Stage 2      : LightGBM context classifier on 22 features
                 [own_proba(11) | neighbour_mean_proba(11)]
  Macro-F1     : 0.712 (test set, 360 images)
  Model file   : models/model_ipl_jersey_prediction_varun.pkl

Class-imbalance handling
------------------------
Three complementary techniques applied to address 83.8% "none" dominance:
  1. class_weight='balanced'     — adjusts loss function weights mathematically
  2. Per-class threshold tuning  — argmax(proba/threshold), tuned on cal split
  3. None-class capping          — none cells capped at 3× total team cells
                                   (139,010 → 81,402 cells, 41% reduction)
                                   Reduces background memorisation while
                                   class_weight corrects the loss function.
