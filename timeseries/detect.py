"""
Forecast-error anomaly detection.

The idea, in plain terms: the LSTM learns what a normal next-day value looks like. When the
actual value lands far from the forecast, that gap is suspicious. We standardize the
forecast errors into a rolling z-score and flag any day whose error exceeds a sensitivity
threshold. Higher threshold = fewer, higher-confidence flags.

This is the same pattern used for monitoring operational/risk metrics: model the expected
behaviour, alert on deviation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def anomaly_scores(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Absolute forecast error standardized to a z-score; NaN where no forecast exists."""
    err = np.abs(actual - predicted)
    valid = ~np.isnan(err)
    scores = np.full(len(err), np.nan, dtype=np.float32)
    if valid.sum() < 2:
        return scores
    mu = np.nanmean(err)
    sd = np.nanstd(err)
    if sd < 1e-8:
        scores[valid] = 0.0
        return scores
    scores[valid] = (err[valid] - mu) / sd
    return scores


def flag(scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Binary anomaly flags: 1 where z-score >= threshold."""
    flags = np.zeros(len(scores), dtype=int)
    valid = ~np.isnan(scores)
    flags[valid] = (scores[valid] >= threshold).astype(int)
    return flags


def score_detection(flags: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Precision / recall / F1 of flags against ground-truth labels."""
    tp = int(np.sum((flags == 1) & (labels == 1)))
    fp = int(np.sum((flags == 1) & (labels == 0)))
    fn = int(np.sum((flags == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def detect(actual: np.ndarray, predicted: np.ndarray,
           threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: return (scores, flags)."""
    scores = anomaly_scores(actual, predicted)
    return scores, flag(scores, threshold)
