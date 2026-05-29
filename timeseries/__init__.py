"""LSTM time-series forecasting + anomaly detection (build + evaluate).

Public surface:
    generate, scenario_series, SCENARIOS            (timeseries.data)
    LSTMForecaster, load_forecaster, forecast,
    make_windows, load_norm, WINDOW                 (timeseries.model)
    anomaly_scores, flag, detect, score_detection   (timeseries.detect)
"""
from __future__ import annotations

from timeseries.data import SCENARIOS, generate, scenario_series
from timeseries.detect import anomaly_scores, detect, flag, score_detection
from timeseries.model import (WINDOW, LSTMForecaster, forecast, load_forecaster,
                              load_norm, make_windows)

__all__ = [
    "generate", "scenario_series", "SCENARIOS",
    "LSTMForecaster", "load_forecaster", "forecast", "make_windows", "load_norm", "WINDOW",
    "anomaly_scores", "flag", "detect", "score_detection",
]
