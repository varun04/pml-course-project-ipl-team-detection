# T1.1 — Image Collection Guide (IPL Season 18, 2025)

This file is the **single source of truth** for how raw images are collected for the dataset. Read this before downloading any image.

## Scope (locked decisions)

- **Season:** IPL 18 — **2025 season only.** Reject any image from a prior or later season (jersey designs change year-to-year, FAQ Q4). Re-locked to 2025 on 2026-05-29 after Group 19 (Tushita / Arijit) shared 140+ pre-labelled images from the 2025 SRHvsDC match. The 12 CSK 2026 images collected before the pivot are archived under `dataset/_archived/2026_pivot/`.
- **Target counts:** ≥100 images per franchise × 10 franchises + a "no-team" bucket (empty pitch, crowd, stadium, logos with no jersey visible) for class 0. Aim for **120+ per franchise** so we can discard some during validation.
- **Aspect / resolution rule (spec):** Only collect images whose native resolution is **≥ 800×600**. Lower-resolution images **must not** be upscaled — they are unusable.

## Approved source list

Record every download with its source URL in a per-franchise log (`dataset/raw/<franchise>/_sources.csv`, schema below). The README we ship at submission will be generated from these logs.

| Tier | Source | Why use it | Watch-outs |
|---|---|---|---|
| 1 | **iplt20.com** — Photos section (official) | Highest quality, season-tagged, authoritative | Limited bulk download; one-at-a-time saves |
| 1 | **BCCI / official franchise sites** (e.g., chennaisuperkings.com, mumbaiindians.com) | Clean studio + match shots, current season jerseys | Often watermarked; keep watermark out of the cell crop if possible |
| 2 | **Getty Images / AP / PTI** (via news embeds) | Match-action photos, high resolution | Copyrighted — use for academic project only; cite source |
| 2 | **News portals** — ESPNcricinfo, Cricbuzz, NDTV Sports, Indian Express, Times of India sports section | Easy season filtering, good variety | Many compressed thumbnails are <800×600 — check before keeping |
| 3 | **Wikimedia Commons** | Open licence (preferred where available) | Sparse coverage of 2025 specifically |
| 3 | **Google Images** filtered by `site:` and Tools → Size → Large | Fallback for under-represented franchises | High noise; verify each image is genuinely 2025 |

**Do not use:**
- AI-generated images
- Fan-art / illustrations / cartoons
- Screenshots from broadcasts at <800×600 effective resolution
- Photos where jerseys cannot be visually verified (e.g., heavy back-lighting, motion blur over the whole player)

## What makes a good image (keep)

- Players clearly in current-season jersey, jersey colour and logos identifiable
- Single team OR distinguishable multi-team scenes (e.g., post-match handshakes — the example image in the problem statement is exactly this)
- A mix of: full-body, half-body, action shots, line-ups, single-player close-ups
- A mix of camera angles, lighting, indoor/outdoor

## What to discard (don't even download)

- **Crowd images where jerseys are mixed and ambiguous** (FAQ Q5 — discard)
- Players not in IPL kit (training jerseys, off-field clothes)
- Images from any season other than 2025
- Heavy artistic filters that change jersey colour

## Per-franchise `_sources.csv` schema

Every team member maintains a `_sources.csv` alongside the images they save. One row per image. Schema:

```
filename,source_url,date_downloaded,downloader,notes
CSK_001.jpg,https://www.iplt20.com/photos/...,2026-05-28,Varun,post-match handshake vs RCB
```

`filename` matches the file actually placed in `dataset/raw/<franchise>/`.

## Naming convention

Inside `dataset/raw/<NN_FRANCHISE>/` save files as `<FRANCHISE>_<###>.jpg` (or `.png`). Example: `CSK_001.jpg`, `CSK_002.jpg`, …, `MI_001.jpg`. Three-digit zero-padded counters keep them sortable.

## Workflow (per team member)

1. Pick the franchise(s) you're assigned.
2. Save each candidate image into `dataset/raw/<NN_FRANCHISE>/`.
3. Append a row to that franchise's `_sources.csv`.
4. When you have a batch (~25 images), run `python src/phase1_dataset/validate_images.py` to catch low-resolution rejects early.
5. When all franchises hit ≥120, run `python src/phase1_dataset/resize_images.py` to produce the final 800×600 set under `dataset/processed/`.

## Class-0 ("no team") guidance

Collect ~150 images that contain **no IPL player in jersey**: empty pitches, stadium wide shots, crowd without identifiable team colour, scoreboard close-ups, stumps, ground staff, logos on banners. These teach the classifier what an empty / non-player cell looks like.

## Legal & sourcing note (for the README we ship)

This dataset is collected strictly for academic coursework. Every image's source URL is recorded. No image is redistributed beyond the project submission.
