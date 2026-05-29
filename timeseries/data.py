"""
Synthetic financial-style time series with labelled anomalies.

The series stands in for a daily risk/operations metric (think: transaction volume or a
risk indicator monitored by a bank's risk office). It is built from structure an LSTM can
actually learn, so the model can beat a naive baseline:
  - a weekly seasonal pattern (weekday effect),
  - a slow trend,
  - AR(1) autocorrelated noise,
on a positive base level. Everything is seeded, so the data is reproducible.

Anomalies are injected on top and labelled, so detection can be scored honestly:
  - spikes: short sharp jumps (e.g. a burst of fraudulent activity),
  - level shifts: a sustained regime change over a window.

generate(...) returns (values, labels) where labels[i] == 1 marks an anomalous day.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

WEEK = 7


def _clean_series(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    # Weekly seasonality: a weekday pattern repeated, scaled.
    weekday = np.array([1.00, 1.15, 1.10, 1.05, 1.20, 0.70, 0.55])  # Mon..Sun
    seasonal = weekday[t % WEEK]
    # Slow trend: gentle sinusoidal drift so it is non-stationary but bounded.
    trend = 1.0 + 0.15 * np.sin(2 * np.pi * t / 180.0)
    # AR(1) noise.
    noise = np.zeros(n)
    phi, sigma = 0.6, 0.04
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma)
    base = 100.0
    return base * seasonal * trend * (1.0 + noise)


def generate(n: int = 2000, seed: int = 0, n_spikes: int = 0,
             level_shift: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a series and its anomaly labels.

    Args:
        n: number of days.
        seed: RNG seed (reproducible).
        n_spikes: number of short sharp spikes to inject.
        level_shift: if True, inject one sustained level-shift segment.
    """
    rng = np.random.default_rng(seed)
    values = _clean_series(n, rng)
    labels = np.zeros(n, dtype=int)

    if n_spikes > 0:
        # Avoid the first window so detection always has history before an anomaly.
        idx = rng.choice(np.arange(30, n), size=min(n_spikes, n - 30), replace=False)
        for i in idx:
            sign = 1.0 if rng.random() < 0.7 else -1.0
            values[i] *= 1.0 + sign * rng.uniform(0.35, 0.6)
            labels[i] = 1

    if level_shift:
        start = int(n * 0.6)
        end = min(n, start + max(15, n // 20))
        values[start:end] *= 1.35   # sustained regime change
        labels[start:end] = 1

    return values.astype(np.float32), labels


SCENARIOS = {
    "calm":      dict(n_spikes=0, level_shift=False, desc="A normal stretch, no incidents."),
    "spikes":    dict(n_spikes=6, level_shift=False, desc="Several sharp spikes (e.g. fraud bursts)."),
    "regime":    dict(n_spikes=1, level_shift=True, desc="A sustained level shift (regime change)."),
    "crash":     dict(n_spikes=3, level_shift=True, desc="Spikes plus a regime change (stress)."),
}


def scenario_series(name: str, n: int = 240, seed: int = 7) -> Tuple[np.ndarray, np.ndarray, str]:
    """A short labelled series for a named demo scenario."""
    cfg = SCENARIOS.get(name, SCENARIOS["spikes"])
    vals, labels = generate(n=n, seed=seed, n_spikes=cfg["n_spikes"],
                            level_shift=cfg["level_shift"])
    return vals, labels, cfg["desc"]
