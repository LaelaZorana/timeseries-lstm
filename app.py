"""
Gradio demo for the LSTM time-series forecaster + anomaly detector.

Bespoke UI (the portfolio's design language, not the stock template): a hand-built SVG
chart draws the metric, the LSTM forecast, and the flagged anomalies; metric cards report
detection precision/recall/F1 live; a scenario picker and a sensitivity slider drive it.
Inference runs the real package (timeseries/) with the bundled trained weights.

Run locally:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces this file is the entry point (app_file: app.py).
"""
from __future__ import annotations

import numpy as np
import gradio as gr

from timeseries import (SCENARIOS, anomaly_scores, flag, forecast,
                        load_forecaster, load_norm, score_detection, scenario_series)
from timeseries.model import WINDOW

ACCENT = "#10b981"   # emerald (monitoring)
_MODEL = load_forecaster()
_MEAN, _STD = load_norm()

SCN_LABELS = {
    "calm": "Calm (no incidents)",
    "spikes": "Spikes (fraud bursts)",
    "regime": "Regime shift",
    "crash": "Stress (spikes + shift)",
}


def _svg(values, pred, flags, labels):
    """Hand-built SVG: actual line, forecast (dashed), flagged anomalies, true-anomaly ticks."""
    n = len(values)
    W, H = 720, 280
    pl, pr, pt, pb = 10, 10, 16, 26
    finite = np.concatenate([values, pred[~np.isnan(pred)]]) if np.isfinite(pred).any() else values
    ymin, ymax = float(np.min(finite)), float(np.max(finite))
    if ymax - ymin < 1e-6:
        ymax = ymin + 1.0

    def X(i):
        return pl + (i / max(n - 1, 1)) * (W - pl - pr)

    def Y(v):
        return pt + (1 - (v - ymin) / (ymax - ymin)) * (H - pt - pb)

    actual_pts = " ".join(f"{X(i):.1f},{Y(values[i]):.1f}" for i in range(n))
    fc_pts = " ".join(f"{X(i):.1f},{Y(pred[i]):.1f}" for i in range(n) if not np.isnan(pred[i]))

    # Ground-truth anomaly ticks along the baseline.
    ticks = "".join(
        f'<rect x="{X(i)-1:.1f}" y="{H-pb+4:.0f}" width="2" height="6" rx="1" fill="#fca5a5"/>'
        for i in range(n) if labels[i] == 1
    )
    # Flagged anomalies as rings on the actual line.
    rings = "".join(
        f'<circle cx="{X(i):.1f}" cy="{Y(values[i]):.1f}" r="5.5" fill="none" '
        f'stroke="#ef4444" stroke-width="2.5"/>'
        for i in range(n) if flags[i] == 1
    )
    return f"""
    <svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet" class="ts-svg">
      <polyline points="{fc_pts}" fill="none" stroke="#94a3b8" stroke-width="1.6"
                stroke-dasharray="5 4" opacity="0.9"/>
      <polyline points="{actual_pts}" fill="none" stroke="{ACCENT}" stroke-width="2.4"/>
      {rings}{ticks}
    </svg>
    """


def _card(value, label, tone="ink"):
    return (f'<div class="ts-card ts-{tone}"><div class="ts-card-v">{value}</div>'
            f'<div class="ts-card-l">{label}</div></div>')


def run(scenario: str, sensitivity: float):
    vals, labels, desc = scenario_series(scenario, n=300, seed=11)
    pred = forecast(vals, _MODEL, _MEAN, _STD)
    scores = anomaly_scores(vals, pred)
    flags = flag(scores, threshold=float(sensitivity))
    m = score_detection(flags, labels)

    n_true = int(labels.sum())
    n_flag = int(flags.sum())
    chart = _svg(vals, pred, flags, labels)

    if n_true == 0:
        # Calm: success is NOT firing. Show false-alarm count instead of precision/recall.
        cards = (_card(f"{n_flag}", "false alarms", "good" if n_flag == 0 else "warn")
                 + _card("0", "real anomalies", "ink")
                 + _card("✓" if n_flag == 0 else "!", "clean run" if n_flag == 0 else "noisy", "ink"))
    else:
        cards = (_card(f"{m['precision']*100:.0f}%", "precision", "accent")
                 + _card(f"{m['recall']*100:.0f}%", "recall", "accent")
                 + _card(f"{m['f1']:.2f}", "F1", "accent")
                 + _card(f"{n_flag}/{n_true}", "flagged / real", "ink"))

    legend = (f'<div class="ts-legend">'
              f'<span><i class="ts-line" style="background:{ACCENT}"></i>actual</span>'
              f'<span><i class="ts-dash"></i>LSTM forecast</span>'
              f'<span><i class="ts-ring"></i>flagged anomaly</span>'
              f'<span><i class="ts-tick"></i>true anomaly</span></div>')

    return (f'<div class="ts-result"><div class="ts-desc">{desc} '
            f'<b>Sensitivity z &ge; {sensitivity:.1f}</b></div>'
            f'<div class="ts-cards">{cards}</div>'
            f'<div class="ts-chartcard">{chart}{legend}</div></div>')


CSS = """
:root { --ts-bg1:#ecfdf5; --ts-bg2:#eff6ff; --ts-ink:#0f2a22; --ts-muted:#5b7167;
  --ts-card:#fff; --ts-line:rgba(15,42,34,.09); --ts-accent:#10b981;
  --ts-font:'Plus Jakarta Sans','Inter',system-ui,sans-serif; }

/* Light lock: HF Spaces default to dark mode, but this UI is designed light.
   Override Gradio's dark theme variables so it renders light everywhere. */
:root, .dark, gradio-app.dark {
  color-scheme: light !important;
  --body-background-fill:#ffffff !important;
  --background-fill-primary:#ffffff !important;
  --background-fill-secondary:#f6f6fb !important;
  --block-background-fill:#ffffff !important;
  --block-label-background-fill:#ffffff !important;
  --input-background-fill:#ffffff !important;
  --border-color-primary:rgba(20,16,40,.12) !important;
  --body-text-color:#16131f !important;
  --body-text-color-subdued:#6b6880 !important;
  --block-title-text-color:#16131f !important;
  --block-info-text-color:#6b6880 !important;
}
html, body, gradio-app, .dark { background:#ffffff !important; }

.gradio-container { max-width: 800px !important; background:
  radial-gradient(1200px 480px at 12% -10%, var(--ts-bg1), transparent 60%),
  radial-gradient(1000px 480px at 110% 8%, var(--ts-bg2), transparent 55%) !important; }
.gradio-container, .gradio-container * { font-family: var(--ts-font); }

#ts-head { text-align:center; padding:18px 8px 4px; }
#ts-head .ts-pill { display:inline-block; background:#0f2a22; color:#fff; border-radius:999px;
  padding:5px 13px; font-size:.7rem; font-weight:700; letter-spacing:.08em; margin-bottom:14px; }
#ts-head h1 { margin:0; font-size:2rem; font-weight:800; letter-spacing:-.02em;
  background:linear-gradient(90deg,#059669,#0ea5e9); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; }
#ts-head p { margin:10px auto 0; max-width:600px; color:var(--ts-muted); font-size:1.0rem; line-height:1.55; }

.ts-result { animation: ts-fade .35s ease both; }
@keyframes ts-fade { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }
.ts-desc { text-align:center; color:var(--ts-muted); margin:4px 0 14px; font-size:.96rem; }
.ts-desc b { color:var(--ts-ink); }

.ts-cards { display:flex; gap:12px; margin-bottom:14px; }
.ts-card { flex:1; background:var(--ts-card); border:1px solid var(--ts-line); border-radius:16px;
  padding:16px; text-align:center; box-shadow:0 10px 28px rgba(15,42,34,.05); }
.ts-card-v { font-size:1.7rem; font-weight:800; color:var(--ts-ink); letter-spacing:-.01em; }
.ts-card-l { font-size:.8rem; color:var(--ts-muted); margin-top:3px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.ts-card.ts-accent .ts-card-v { color:var(--ts-accent); }
.ts-card.ts-good .ts-card-v { color:#10b981; }
.ts-card.ts-warn .ts-card-v { color:#ef4444; }

.ts-chartcard { background:var(--ts-card); border:1px solid var(--ts-line); border-radius:18px;
  padding:16px 18px 10px; box-shadow:0 14px 36px rgba(15,42,34,.06); }
.ts-svg { display:block; }
.ts-svg polyline { animation: ts-draw 1s ease-out both; }
@keyframes ts-draw { from{opacity:0} to{opacity:1} }
.ts-legend { display:flex; gap:18px; justify-content:center; flex-wrap:wrap; margin-top:8px;
  font-size:.82rem; color:var(--ts-muted); }
.ts-legend span { display:flex; align-items:center; gap:6px; }
.ts-legend i { display:inline-block; }
.ts-line { width:16px; height:3px; border-radius:2px; }
.ts-dash { width:16px; height:0; border-top:2px dashed #94a3b8; }
.ts-ring { width:11px; height:11px; border:2.5px solid #ef4444; border-radius:50%; }
.ts-tick { width:3px; height:9px; background:#fca5a5; border-radius:1px; }

#ts-controls { gap:14px; }
.ts-footer { margin-top:22px; padding-top:16px; border-top:1px solid var(--ts-line);
  text-align:center; font-size:.88rem; color:var(--ts-muted); line-height:1.9; }
.ts-footer a { text-decoration:none; font-weight:700; color:var(--ts-accent); }
.ts-meta { text-align:center; color:var(--ts-muted); font-size:.82rem; margin-top:10px; }
"""

FOOTER = """
<div class="ts-footer">
📈 LSTM forecasting + anomaly detection by <b>Laela Zorana</b><br>
<a href="https://laelazorana.github.io">Portfolio</a> &middot; <a href="https://www.linkedin.com/in/laela-zorana-362309114">LinkedIn</a> &middot; <a href="https://github.com/LaelaZorana">GitHub</a> &middot; <a href="https://huggingface.co/LaelaZ">Hugging Face</a><br>
<span style="opacity:.7">More demos:</span> <a href="https://huggingface.co/spaces/LaelaZ/distilbert-emotion">Emotion</a> &middot; <a href="https://huggingface.co/spaces/LaelaZ/cnn-gradcam">CNN + Grad-CAM</a> &middot; <a href="https://huggingface.co/spaces/LaelaZ/nn-from-scratch">NN From Scratch</a> &middot; <a href="https://huggingface.co/spaces/LaelaZ/ai-agent-scenario-qc">Scenario QC</a> &middot; <a href="https://huggingface.co/spaces/LaelaZ/rlhf-pairwise-rater">RLHF Rater</a> &middot; <a href="https://huggingface.co/spaces/LaelaZ/scorm-qa-validator">SCORM QA</a>
</div>
"""

theme = gr.themes.Soft(
    primary_hue="emerald", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"),
          "system-ui", "sans-serif"],
)

with gr.Blocks(title="Time-Series Anomaly Detection (LSTM)", theme=theme, css=CSS) as demo:
    gr.HTML(
        '<div id="ts-head"><span class="ts-pill">DEEP LEARNING · TIME SERIES · RISK MONITORING</span>'
        "<h1>Catching anomalies before they hurt</h1>"
        "<p>An LSTM learns the normal rhythm of a daily risk metric, then flags days where "
        "reality drifts too far from its forecast. Pick a scenario and dial the sensitivity. "
        "On held-out data the forecaster beats a naive baseline by 83%.</p></div>"
    )
    with gr.Row(elem_id="ts-controls"):
        scn = gr.Dropdown(choices=[(SCN_LABELS[k], k) for k in SCENARIOS],
                          value="spikes", label="Scenario")
        sens = gr.Slider(2.0, 5.0, value=3.0, step=0.25, label="Sensitivity (z-score threshold)")

    out = gr.HTML()

    scn.change(run, inputs=[scn, sens], outputs=out)
    sens.change(run, inputs=[scn, sens], outputs=out)
    demo.load(run, inputs=[scn, sens], outputs=out)

    gr.HTML(FOOTER)
    gr.HTML('<div class="ts-meta">Runs the real package (timeseries/) with the trained LSTM. '
            'Backtest + detection metrics: <code>python -m timeseries.evaluate</code>. '
            'Lower the threshold to catch more (and risk false alarms); raise it to flag only the clearest.</div>')


if __name__ == "__main__":
    demo.launch()
