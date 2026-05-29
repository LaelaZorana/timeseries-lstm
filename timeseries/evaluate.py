"""
Evaluate the forecaster and the anomaly detector. Two honest tests.

1. BACKTEST (forecasting): on a held-out clean series the model never trained on, compare
   the LSTM's one-step forecast error against a naive last-value baseline (predict tomorrow
   == today). A forecaster that cannot beat "tomorrow looks like today" is not worth
   shipping, so this is the bar that matters.

2. ANOMALY DETECTION: on series with injected, labelled anomalies, report precision /
   recall / F1 of the forecast-error detector.

Run:  python -m timeseries.evaluate
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from timeseries.data import generate, scenario_series, SCENARIOS
from timeseries.detect import anomaly_scores, flag, score_detection
from timeseries.model import WINDOW, forecast, load_forecaster, load_norm


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def backtest(seed: int = 99, verbose: bool = True) -> Dict[str, float]:
    """LSTM forecast RMSE vs naive last-value baseline on a held-out clean series."""
    series, _ = generate(n=600, seed=seed)              # unseen clean data
    model = load_forecaster()
    mean, std = load_norm()
    pred = forecast(series, model, mean, std)

    naive = np.full(len(series), np.nan, dtype=np.float32)
    naive[1:] = series[:-1]                              # predict today = yesterday

    # Compare on the region where the LSTM produces forecasts.
    region = slice(WINDOW, len(series))
    lstm_rmse = _rmse(series[region], pred[region])
    naive_rmse = _rmse(series[region], naive[region])
    improvement = 100.0 * (naive_rmse - lstm_rmse) / naive_rmse

    if verbose:
        print("BACKTEST (held-out clean series, one-step forecast)")
        print(f"  LSTM  RMSE: {lstm_rmse:.3f}")
        print(f"  naive RMSE: {naive_rmse:.3f}  (predict today = yesterday)")
        print(f"  improvement over naive: {improvement:.1f}%\n")
    return {"lstm_rmse": lstm_rmse, "naive_rmse": naive_rmse, "improvement_pct": improvement}


def detection_report(threshold: float = 3.0, verbose: bool = True) -> Dict[str, Dict]:
    """Precision/recall/F1 of the detector across all named scenarios + a pooled score."""
    model = load_forecaster()
    mean, std = load_norm()
    per_scenario: Dict[str, Dict] = {}
    all_flags, all_labels = [], []
    for name in SCENARIOS:
        vals, labels, _ = scenario_series(name, n=300, seed=11)
        pred = forecast(vals, model, mean, std)
        scores = anomaly_scores(vals, pred)
        flags = flag(scores, threshold)
        per_scenario[name] = score_detection(flags, labels)
        all_flags.append(flags)
        all_labels.append(labels)
    pooled = score_detection(np.concatenate(all_flags), np.concatenate(all_labels))

    if verbose:
        print(f"ANOMALY DETECTION (z-score threshold {threshold})")
        print(f"{'scenario':10s} {'prec':>6s} {'recall':>7s} {'f1':>6s}")
        for name, m in per_scenario.items():
            print(f"{name:10s} {m['precision']:6.2f} {m['recall']:7.2f} {m['f1']:6.2f}")
        print(f"{'POOLED':10s} {pooled['precision']:6.2f} {pooled['recall']:7.2f} {pooled['f1']:6.2f}")
    return {"per_scenario": per_scenario, "pooled": pooled}


if __name__ == "__main__":
    backtest(verbose=True)
    detection_report(verbose=True)
