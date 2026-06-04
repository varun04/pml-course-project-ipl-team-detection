"""IPL Team Detection — Hugging Face Spaces Demo"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
import joblib

# ── Paths — robust for both local and HF Space ────────────────────────────────
APP_DIR    = Path(__file__).resolve().parent          # /app on HF Space
SRC_DIR    = APP_DIR / 'src'
MODEL_FILE = APP_DIR / 'model_ipl_jersey_prediction_varun.pkl'

sys.path.insert(0, str(SRC_DIR))

from phase3.pipeline         import load_model, predict_image
from phase3.player_detection import (detect_players_cells, estimate_pose,
                                      AVG_CELLS_PER_PLAYER, TEAMS)

print(f"Loading model from {MODEL_FILE} ...")
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_FILE}. "
        "Ensure model_ipl_jersey_prediction_varun.pkl is uploaded to the Space."
    )
MODEL = load_model(MODEL_FILE)
print("Model loaded ✅")

COLORS = ['#444a55','#f2c200','#3b7dd8','#0fa3a3','#7a4ad1',
          '#6fd0e8','#0a2a66','#ed1b24','#e6308a','#8b0000','#ff822a']
CW, CH = 100, 75


def predict(input_image: np.ndarray):
    if input_image is None:
        return None, "Please upload an image."

    img_rgb = cv2.resize(input_image, (800, 600), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    preds       = predict_image(MODEL, img_bgr)
    grid        = np.array(preds).reshape(8, 8)
    players     = detect_players_cells(np.array(preds))
    n_players   = sum(p['est_players'] for p in players)
    teams_found = sorted(set(TEAMS[p] for p in preds if p != 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

    axes[0].imshow(img_rgb)
    for i in range(1, 8):
        axes[0].axvline(i * CW, color='white', linewidth=0.9, alpha=0.7)
        axes[0].axhline(i * CH, color='white', linewidth=0.9, alpha=0.7)
    axes[0].set_title('Input image  +  8×8 grid overlay',
                       fontsize=12, fontweight='bold', pad=6)
    axes[0].axis('off')

    axes[1].imshow(img_rgb)
    for idx, v in enumerate(preds):
        r, c = divmod(idx, 8)
        x, y = c * CW, r * CH
        if v != 0:
            axes[1].add_patch(mpatches.FancyBboxPatch(
                (x + 2, y + 2), CW - 4, CH - 4,
                boxstyle='round,pad=2',
                facecolor=COLORS[v], alpha=0.52,
                edgecolor=COLORS[v], linewidth=2.0))
            axes[1].text(x + CW/2, y + CH/2, f'{v}\n{TEAMS[v]}',
                         color='white', fontsize=7.5, ha='center', va='center',
                         fontweight='bold',
                         bbox=dict(facecolor=COLORS[v], alpha=0.80,
                                   pad=2, boxstyle='round,pad=0.3'))
    for i in range(1, 8):
        axes[1].axvline(i * CW, color='white', linewidth=0.5, alpha=0.35)
        axes[1].axhline(i * CH, color='white', linewidth=0.5, alpha=0.35)

    teams_str = ', '.join(teams_found) if teams_found else 'None detected'
    axes[1].set_title(f'Predicted — Teams: {teams_str}  |  Players: ~{n_players}',
                       fontsize=12, fontweight='bold', pad=6)
    axes[1].axis('off')

    fig.tight_layout(pad=1.5)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    from PIL import Image as PILImage
    out_img = np.array(PILImage.open(buf).convert('RGB'))  # RGB — gradio 5.x
    plt.close(fig)

    lines = [
        f"Teams detected    : {', '.join(teams_found) if teams_found else 'none'}",
        f"Estimated players : ~{n_players}", "",
        "Per-team breakdown:",
    ]
    for pl in players:
        pose, angle = estimate_pose(pl['cell_indices'])
        lines.append(f"  {pl['team_name']:<6}  {pl['n_cells']:>2} cells  "
                     f"pose: {pose} ({angle:.0f}°)  → ~{pl['est_players']} player(s)")
    if not players:
        lines.append("  No team cells detected — background / crowd image")
    team_char = {0:'.', 1:'C', 2:'D', 3:'G', 4:'K',
                 5:'L', 6:'M', 7:'P', 8:'R', 9:'B', 10:'S'}
    lines += ["", "Cell grid (8×8):"]
    for row in grid:
        lines.append("  " + " ".join(team_char[v] for v in row))
    lines += ["", "Legend: C=CSK D=DC G=GT K=KKR L=LSG M=MI",
              "        P=PBKS R=RR B=RCB S=SRH .=none"]

    return out_img, "\n".join(lines)


with gr.Blocks(title="IPL Team Detection") as demo:
    gr.Markdown("""
    # IPL Team Detection — Interactive Demo
    **PML Course Project | IIT Bombay 2026**

    Upload any IPL 2025 broadcast image. The model will:
    1. Resize to 800x600 and apply an 8x8 grid (64 cells)
    2. Classify each cell: which IPL team (0=none, 1-10)
    3. Estimate player count via connected-component analysis
    """)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload IPL match image", type="numpy", height=300)
            btn = gr.Button("Run Detection", variant="primary", size="lg")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Result (left: raw + grid | right: prediction)",
                               type="numpy", height=300)
    with gr.Row():
        out_txt = gr.Textbox(label="Detection summary", lines=18, max_lines=25)

    gr.Examples(
        examples=[
            ['examples/GT_player.jpg'],
            ['examples/MI_grouped.jpg'],
            ['examples/RR_vs_CSK.jpg'],
            ['examples/RCB_player.jpg'],
            ['examples/SRH_vs_DC.jpg'],
            ['examples/MI_vs_CSK.jpg'],
        ],
        inputs=inp,
        label="Click any example to test instantly",
    )

    gr.Markdown("""---
    **Model:** Two-stage LightGBM | **Features:** 194 hand-crafted (HSV, HOG, LBP, texture) | **Macro-F1:** 0.712 | No CNNs
    """)

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_txt])
    inp.change(fn=predict, inputs=inp, outputs=[out_img, out_txt])

if __name__ == "__main__":
    demo.launch()
