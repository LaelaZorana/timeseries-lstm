"""
Tests for the time-series forecaster and anomaly detector.

Fast layer (always runs, no trained model): the data generator is reproducible and labels
match injected anomalies; windowing is correct; the detector's scoring math is right on a
hand-built case. Heavy layer (skipped unless trained weights are present): the LSTM beats a
naive baseline on a held-out series, and the detector catches injected spikes, so a
regressed model fails CI rather than the demo.
"""
from __future__ import annotations

import numpy as np
import pytest

from timeseries.data import generate
from timeseries.detect import anomaly_scores, flag, score_detection
from timeseries.model import WINDOW, make_windows, load_forecaster
from timeseries import model as model_mod


def test_generate_is_reproducible_and_positive():
    a, la = generate(n=300, seed=5, n_spikes=4)
    b, lb = generate(n=300, seed=5, n_spikes=4)
    assert np.array_equal(a, b) and np.array_equal(la, lb)   # same seed -> same series
    assert (a > 0).all()                                     # a positive metric
    assert a.shape == (300,) and la.shape == (300,)


def test_injected_spikes_are_labelled():
    _, labels = generate(n=400, seed=3, n_spikes=5)
    assert labels.sum() == 5
    _, labels2 = generate(n=400, seed=3, n_spikes=0, level_shift=True)
    assert labels2.sum() > 0          # the level-shift window is labelled


def test_make_windows_shapes_and_content():
    s = np.arange(20, dtype=np.float32)
    X, y = make_windows(s, window=WINDOW)
    assert X.shape == (20 - WINDOW, WINDOW)
    assert y.shape == (20 - WINDOW,)
    assert np.array_equal(X[0], s[:WINDOW]) and y[0] == s[WINDOW]


def test_detector_scoring_on_known_case():
    # actual vs predicted with one huge error at index 5
    actual = np.array([10.0] * 10, dtype=np.float32)
    predicted = actual.copy()
    predicted[5] = 100.0                                     # large error here
    scores = anomaly_scores(actual, predicted)
    flags = flag(scores, threshold=2.0)
    assert flags[5] == 1 and flags.sum() == 1
    labels = np.zeros(10, dtype=int); labels[5] = 1
    m = score_detection(flags, labels)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_score_detection_counts():
    flags = np.array([1, 1, 0, 0])
    labels = np.array([1, 0, 1, 0])
    m = score_detection(flags, labels)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert abs(m["precision"] - 0.5) < 1e-9 and abs(m["recall"] - 0.5) < 1e-9


# --- Heavy layer: trained model, skipped if weights absent. ---

def test_forecast_shape_and_warmup():
    if not model_mod.WEIGHTS_PATH.exists():
        pytest.skip("trained weights not present (run python -m timeseries.train)")
    from timeseries.model import forecast, load_norm
    s, _ = generate(n=120, seed=8)
    mean, std = load_norm()
    pred = forecast(s, load_forecaster(), mean, std)
    assert pred.shape == s.shape
    assert np.isnan(pred[:WINDOW]).all()        # no forecast without history
    assert np.isfinite(pred[WINDOW:]).all()


def test_lstm_beats_naive_baseline():
    if not model_mod.WEIGHTS_PATH.exists():
        pytest.skip("trained weights not present")
    from timeseries.evaluate import backtest
    res = backtest(verbose=False)
    assert res["lstm_rmse"] < res["naive_rmse"]
    assert res["improvement_pct"] >= 40.0, f"only {res['improvement_pct']:.1f}% over naive"


def test_detector_catches_spikes():
    if not model_mod.WEIGHTS_PATH.exists():
        pytest.skip("trained weights not present")
    from timeseries.evaluate import detection_report
    rep = detection_report(threshold=3.0, verbose=False)
    spikes = rep["per_scenario"]["spikes"]
    assert spikes["recall"] >= 0.8, f"spike recall {spikes['recall']:.2f}"
    assert spikes["precision"] >= 0.7, f"spike precision {spikes['precision']:.2f}"
