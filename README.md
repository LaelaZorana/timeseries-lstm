# timeseries-lstm

**🔗 Live demo:** [try it on Hugging Face Spaces](https://huggingface.co/spaces/LaelaZ/timeseries-lstm). Pick a scenario, dial the sensitivity, and watch the LSTM flag anomalies.

An LSTM that **forecasts** a daily risk/operations metric and **detects anomalies** when the
real value drifts too far from the forecast. It is the time-series piece of my portfolio, and
it bridges a risk-monitoring background to deep learning: model the expected behaviour, then alert
on deviation. As with the rest of my work, I publish the model *and* the evidence it works.

## I built it and I verified it

Two honest tests, not a single pretty plot.

**1. Backtest (forecasting).** On a held-out clean series the model never trained on, I compare
the LSTM's one-step forecast against a naive last-value baseline (predict tomorrow == today):

```
LSTM  RMSE: 4.71
naive RMSE: 28.44   (predict today = yesterday)
improvement over naive: 83.4%
```

The baseline collapses on the weekly seasonality, but the LSTM learns it. A forecaster that cannot
beat "tomorrow looks like today" is not worth releasing, so that is the bar.

**2. Anomaly detection.** Precision / recall / F1 against injected, labelled anomalies, by scenario
(z-score threshold 3.0):

```
scenario     prec  recall   f1
spikes       0.86   1.00   0.92    <- sharp spikes caught cleanly
regime       1.00   0.44   0.61
crash        0.86   0.33   0.48
```

Sharp spikes (think fraud bursts) are caught cleanly. Sustained **regime shifts are harder** for
forecast-error detection, because once the level shifts the LSTM adapts within its window and stops
flagging, so only the onset trips it. The demo shows this honestly rather than hiding it, since it
is a real property of the method and naming it is part of the point.

## Why this repo is more than "it forecasts"

The bars above are guarded by **tests**: `test_lstm_beats_naive_baseline` fails CI if the
forecaster stops beating naive by 40%, and `test_detector_catches_spikes` fails if spike recall or
precision drops. The fast tests check the data generator, windowing, and the detector math offline.

```
tests/test_timeseries.py
  test_generate_is_reproducible_and_positive  # seeded data is deterministic
  test_injected_spikes_are_labelled           # ground-truth labels match injected anomalies
  test_make_windows_shapes_and_content        # supervised windowing is correct
  test_detector_scoring_on_known_case         # z-score flag fires on a known large error
  test_score_detection_counts                 # precision/recall/F1 arithmetic
  test_forecast_shape_and_warmup              # NaN warmup, finite after (trained; skips if absent)
  test_lstm_beats_naive_baseline              # backtest bar (trained; skips if absent)
  test_detector_catches_spikes                # detection bar (trained; skips if absent)
```

## How it works (plain version)

A small LSTM reads a 14-day window and predicts the next day. Train it only on **normal** data, so
it learns what normal looks like. At monitoring time, standardize the forecast errors into a rolling
z-score; any day above the sensitivity threshold is flagged. Lower threshold = catch more (and risk
false alarms); higher = flag only the clearest.

## Run it

```bash
pip install -r requirements.txt
python -m timeseries.train --epochs 25     # trains the forecaster, saves weights/
python -m timeseries.evaluate              # backtest + per-scenario detection metrics
pytest -q                                  # data + detector math + forecast bars
python app.py                              # launch the interactive demo locally
```

## Layout

```
timeseries/
  data.py      # seeded synthetic financial-style series + labelled anomalies
  model.py     # LSTM forecaster + windowing + normalisation
  train.py     # train on clean data, save weights
  detect.py    # forecast-error z-score detection + precision/recall/F1
  evaluate.py  # backtest vs naive baseline + per-scenario detection report
tests/         # data + windowing + detector math + forecast/detection bars
app.py         # bespoke demo: custom SVG chart, metric cards, scenario picker, sensitivity slider
weights/       # shipped trained LSTM + normalisation stats
```

Part of my ML portfolio (build + evaluate). License: MIT.
