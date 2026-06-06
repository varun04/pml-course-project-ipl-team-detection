"""IPL Team Detection — Interactive Demo
Run: .venv/bin/python src/app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import gradio as gr

from phase3.pipeline         import load_model, predict_image
from phase3.player_detection import (detect_players_cells, estimate_pose,
                                      AVG_CELLS_PER_PLAYER, TEAMS)

REPO_ROOT  = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / 'models' / 'model_ipl_jersey_prediction_varun.pkl'

# Team colours (hex) — one per label 0-10
COLORS = ['#444a55','#f2c200','#3b7dd8','#0fa3a3','#7a4ad1',
          '#6fd0e8','#0a2a66','#ed1b24','#e6308a','#8b0000','#ff822a']

CW, CH = 100, 75   # cell width, height in pixels

print("Loading model…")
MODEL = load_model(MODEL_PATH)
print("Model loaded ✅")


def predict(input_image: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Parameters
    ----------
    input_image : RGB numpy array (from Gradio image component)

    Returns
    -------
    overlay_img : side-by-side figure as RGB numpy array
    summary_text: plain-text result summary
    """
    if input_image is None:
        return None, "Please upload an image."

    # ── Resize to 800×600 ────────────────────────────────────────────────────
    img_rgb = cv2.resize(input_image, (800, 600), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Run two-stage model ───────────────────────────────────────────────────
    preds = predict_image(MODEL, img_bgr)          # list of 64 ints
    grid  = np.array(preds).reshape(8, 8)

    # ── Player detection ──────────────────────────────────────────────────────
    players      = detect_players_cells(np.array(preds))
    n_players    = sum(p['est_players'] for p in players)
    teams_found  = sorted(set(TEAMS[p] for p in preds if p != 0))

    # ── Build side-by-side figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

    # Left: raw image + grid lines
    axes[0].imshow(img_rgb)
    for i in range(1, 8):
        axes[0].axvline(i * CW, color='white', linewidth=0.9, alpha=0.7)
        axes[0].axhline(i * CH, color='white', linewidth=0.9, alpha=0.7)
    axes[0].set_title('Input image  +  8×8 grid overlay',
                       fontsize=12, fontweight='bold', pad=6)
    axes[0].axis('off')

    # Right: prediction overlay
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
            axes[1].text(
                x + CW / 2, y + CH / 2,
                f'{v}\n{TEAMS[v]}',
                color='white', fontsize=7.5, ha='center', va='center',
                fontweight='bold',
                bbox=dict(facecolor=COLORS[v], alpha=0.80,
                          pad=2, boxstyle='round,pad=0.3'))
    for i in range(1, 8):
        axes[1].axvline(i * CW, color='white', linewidth=0.5, alpha=0.35)
        axes[1].axhline(i * CH, color='white', linewidth=0.5, alpha=0.35)

    teams_str = ', '.join(teams_found) if teams_found else 'None detected'
    axes[1].set_title(
        f'Predicted  —  Teams: {teams_str}  |  Players: ~{n_players}',
        fontsize=12, fontweight='bold', pad=6)
    axes[1].axis('off')

    fig.tight_layout(pad=1.5)
    import io
    buf_io = io.BytesIO()
    fig.savefig(buf_io, format='png', dpi=110, bbox_inches='tight',
                facecolor='white')
    buf_io.seek(0)
    from PIL import Image as PILImage
    buf = np.array(PILImage.open(buf_io))
    plt.close(fig)

    # ── Summary text ──────────────────────────────────────────────────────────
    lines = [
        f"Teams detected    : {', '.join(teams_found) if teams_found else 'none'}",
        f"Estimated players : ~{n_players}",
        "",
        "Per-team breakdown:",
    ]
    for pl in players:
        pose, angle = estimate_pose(pl['cell_indices'])
        lines.append(
            f"  {pl['team_name']:<6}  {pl['n_cells']:>2} cells  "
            f"pose: {pose} ({angle:.0f}°)  → ~{pl['est_players']} player(s)"
        )
    if not players:
        lines.append("  No team cells detected — background / crowd image")
    lines += [
        "",
        "Cell grid (8×8):",
    ]
    team_char = {0:'.', 1:'C', 2:'D', 3:'G', 4:'K',
                 5:'L', 6:'M', 7:'P', 8:'R', 9:'B', 10:'S'}
    for row in grid:
        lines.append("  " + " ".join(team_char[v] for v in row))
    lines += [
        "",
        "Legend: C=CSK D=DC G=GT K=KKR L=LSG M=MI",
        "        P=PBKS R=RR B=RCB S=SRH .=none",
    ]

    return buf, "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="IPL Team Detection") as demo:
    gr.Markdown("""
    # 🏏 IPL Team Detection — Interactive Demo
    **Group 32 | PML Course Project | IIT Bombay 2026**

    Upload any IPL 2025 broadcast image (any size). The model will:
    1. Resize it to 800×600 and apply an 8×8 grid
    2. Classify each of the 64 cells (team 1–10 or background)
    3. Estimate the number of players and their team
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload IPL match image", type="numpy",
                           height=300)
            btn = gr.Button("Run Detection", variant="primary", size="lg")

        with gr.Column(scale=2):
            out_img = gr.Image(label="Result  (left: raw + grid  |  right: prediction)",
                               type="numpy", height=300)

    with gr.Row():
        out_txt = gr.Textbox(label="Detection summary",
                             lines=18, max_lines=25)

    gr.Examples(
        examples=[
            [str(REPO_ROOT / 'dataset' / 'raw' / 'RRvsLSG' / 'RRvsLSG_image_5.jpg')],
            [str(REPO_ROOT / 'dataset' / 'raw' / 'GTvsMI'  / 'GTvsMI_image_239.jpg')],
            [str(REPO_ROOT / 'dataset' / 'raw' / 'RCBvsPBKS'/ 'RCBvsPBKS_image_68.jpg')],
        ],
        inputs=inp,
        label="Example images from dataset",
    )

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_txt])
    inp.change(fn=predict, inputs=inp, outputs=[out_img, out_txt])

    gr.Markdown("""
    ---
    **Model:** Two-stage LightGBM  |  **Features:** 194 hand-crafted (HSV, HOG, LBP, texture)
    **Macro-F1:** 0.712 on held-out test set  |  **No CNNs used**
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True,
                theme=gr.themes.Soft())
