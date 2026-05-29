---
title: Time-Series Anomaly Detection (LSTM)
emoji: 📈
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Time-Series Anomaly Detection (LSTM)

An LSTM learns the normal rhythm of a daily risk/operations metric, then flags the days
where reality drifts too far from its forecast. Pick a scenario (calm, fraud-burst spikes,
a regime shift, or a stress mix) and dial the detection sensitivity; the chart shows the
actual series, the LSTM forecast, and the flagged anomalies, with live precision / recall / F1.

This is the build-and-prove pattern I use across my portfolio. I do not just plot a forecast:

- **Backtest:** on a held-out clean series the LSTM's one-step forecast beats a naive
  last-value baseline by ~83% RMSE (it learns the weekly seasonality the baseline misses).
- **Detection:** precision / recall / F1 against injected, labelled anomalies, reported per
  scenario. Sharp spikes are caught cleanly; sustained regime shifts are harder for
  forecast-error detection (only the onset trips it), which the demo shows honestly.

Bridges a risk-monitoring background to deep learning: model the expected behaviour, alert
on deviation.

**Source & full docs:** https://github.com/LaelaZorana/timeseries-lstm
