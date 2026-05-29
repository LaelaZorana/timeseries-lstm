"""
LSTM forecaster for the time series.

A small LSTM reads a window of past values and predicts the next one. Inputs are
normalized with statistics fit on the training data (stored with the weights) so inference
matches training. The model and the normaliser are deliberately tiny, so the trained
weights ship in the repo and the demo forecasts without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

WINDOW = 14          # two weeks of history
HIDDEN = 32
LAYERS = 1

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "lstm_forecaster.pt"
NORM_PATH = Path(__file__).resolve().parent.parent / "weights" / "norm.json"


class LSTMForecaster(nn.Module):
    """window of (WINDOW,) values -> next value. Single-feature univariate forecaster."""

    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                # x: (B, WINDOW, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # (B,)


def make_windows(series: np.ndarray, window: int = WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D series into (X, y): X[i] = series[i:i+window], y[i] = series[i+window]."""
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def save_norm(mean: float, std: float) -> None:
    NORM_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORM_PATH.write_text(json.dumps({"mean": float(mean), "std": float(std)}))


def load_norm() -> Tuple[float, float]:
    if NORM_PATH.exists():
        d = json.loads(NORM_PATH.read_text())
        return d["mean"], d["std"]
    return 0.0, 1.0


def load_forecaster() -> LSTMForecaster:
    """Load the shipped trained forecaster in eval mode."""
    model = LSTMForecaster()
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


def forecast(series: np.ndarray, model: LSTMForecaster,
             mean: float, std: float, window: int = WINDOW) -> np.ndarray:
    """One-step-ahead forecast for every point that has `window` history.

    Returns an array the same length as `series`; the first `window` entries are NaN
    (no history to forecast from).
    """
    preds = np.full(len(series), np.nan, dtype=np.float32)
    if len(series) <= window:
        return preds
    X, _ = make_windows(series, window)
    Xn = ((X - mean) / std).reshape(len(X), window, 1)
    with torch.no_grad():
        yn = model(torch.from_numpy(Xn)).cpu().numpy()
    preds[window:] = yn * std + mean
    return preds
