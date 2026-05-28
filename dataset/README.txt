IPL Jersey Identification — Dataset README
==========================================

Course      : Programming for Machine Learning and Data Science (PMLDS)
Programme   : e-PGDiploma in AI & DS — January 2026 cohort
Institution : Centre for Machine Intelligence & Data Science, IIT Bombay
Season used : IPL Season 18 (2025)  -- single season, per FAQ Q4

Folder layout
-------------
dataset/
  raw/                 Original downloads, organised per franchise. Each
                       franchise sub-folder also contains _sources.csv
                       (one row per image: filename, source_url,
                       date_downloaded, downloader, notes).
  processed/           Final 800x600 JPEG images, organised per franchise.
                       This is the set used for labelling and training.
  _rejected/           Images moved out by validate_images.py because they
                       failed the >=800x600 resolution check or were
                       unreadable. _rejection_log.csv records the reason.

Franchise sub-folder naming (matches the label scheme in the problem statement)
------------------------------------------------------------------------------
  00_no_team   label 0   No team / empty pitch / crowd / background
  01_CSK       label 1   Chennai Super Kings
  02_DC        label 2   Delhi Capitals
  03_GT        label 3   Gujarat Titans
  04_KKR       label 4   Kolkata Knight Riders
  05_LSG       label 5   Lucknow Super Giants
  06_MI        label 6   Mumbai Indians
  07_PBKS      label 7   Punjab Kings
  08_RR        label 8   Rajasthan Royals
  09_RCB       label 9   Royal Challengers Bengaluru
  10_SRH       label 10  Sunrisers Hyderabad

Sourcing
--------
Every collected image has its origin URL logged in the per-franchise
_sources.csv. Sources span the official IPL site (iplt20.com), official
franchise websites, news outlets (ESPNcricinfo, Cricbuzz, news portals)
and licensed press agencies (Getty, AP, PTI) re-published in those news
items. See docs/T1_1_collection_guide.md for the full source policy.

The dataset is collected strictly for academic coursework. No images are
redistributed beyond the project submission.

Dataset constraints (locked by the problem statement)
-----------------------------------------------------
* Native resolution must be >= 800 x 600. Lower-resolution images are
  never upscaled; they are moved into _rejected/.
* All processed images are exactly 800 x 600 (4:3), produced by
  centre-cropping to 4:3 and downsizing.
* >= 100 images per franchise (target ~120 collected so we can drop some
  during validation / labelling QC).

Counts
------
Final per-franchise counts are produced by:
    python src/phase1_dataset/dataset_stats.py

(Fill in the counts below before submission.)

    franchise       raw    processed
    ----------------------------------
    00_no_team      ___    ___
    01_CSK          ___    ___
    02_DC           ___    ___
    03_GT           ___    ___
    04_KKR          ___    ___
    05_LSG          ___    ___
    06_MI           ___    ___
    07_PBKS         ___    ___
    08_RR           ___    ___
    09_RCB          ___    ___
    10_SRH          ___    ___
    ----------------------------------
    TOTAL           ___    ___
