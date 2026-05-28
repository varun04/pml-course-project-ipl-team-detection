# PML Course Project — IPL Team Detection

Course project for *Programming for Machine Learning and Data Science (PMLDS)*, e-PGD AI&DS Jan'26 cohort, CMInDS, IIT Bombay.

Given an 800×600 cricket image, divide it into an 8×8 grid (64 cells) and predict for each cell which IPL franchise (if any) is present. Hand-crafted image features only — no CNNs for feature creation.

**Submission deadline:** 2026-06-06 23:55.

## Repo layout

```
dataset/
  raw/         per-franchise originals + _sources.csv
  processed/   final 800x600 JPEGs (input to labelling & training)
  _rejected/   images that failed the >=800x600 native-resolution check
src/
  phase1_dataset/   collection / validation / resize / stats scripts
notebooks/         exploratory analysis
models/            trained .pkl files
outputs/           predictions CSV
docs/              collection guide, presentation drafts
```

## Phase 1 quickstart (current phase)

1. Read [docs/T1_1_collection_guide.md](docs/T1_1_collection_guide.md) before downloading any image.
2. Save raw images into `dataset/raw/<NN_FRANCHISE>/` and log every download in that folder's `_sources.csv`.
3. Validate the batch:
   ```
   python src/phase1_dataset/validate_images.py            # report only
   python src/phase1_dataset/validate_images.py --apply    # move rejects
   ```
4. Produce the final 800×600 set:
   ```
   python src/phase1_dataset/resize_images.py
   ```
5. Check counts (need ≥100 per franchise):
   ```
   python src/phase1_dataset/dataset_stats.py
   ```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
