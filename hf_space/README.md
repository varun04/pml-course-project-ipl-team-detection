---
title: IPL Team Detection
emoji: 🏏
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---

# IPL Jersey Team Detection

**PML Course Project | IIT Bombay 2026**

Upload any IPL 2025 broadcast image to detect:
- Which IPL teams are present
- Estimated number of players
- Cell-level prediction overlay (8×8 grid)

**Model:** Two-stage LightGBM | **Features:** 194 hand-crafted | **Macro-F1:** 0.712 | No CNNs used
