"""
Train the LSTM forecaster on a clean (anomaly-free) series.

Training only on normal data is deliberate: the detector should learn what normal looks
like, so injected anomalies later stand out as large forecast errors. Saves the weights
and the normalisation stats to weights/.

Run:  python -m timeseries.train --epochs 25
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from timeseries.data import generate
from timeseries.model import (LSTMForecaster, WEIGHTS_PATH, WINDOW, make_windows,
                              save_norm)


def train(epochs: int = 25, lr: float = 5e-3, batch_size: int = 64, seed: int = 0) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Long clean training series; a separate clean series for validation.
    train_series, _ = generate(n=2000, seed=1)
    val_series, _ = generate(n=500, seed=2)

    mean, std = float(train_series.mean()), float(train_series.std())
    Xtr, ytr = make_windows(train_series)
    Xva, yva = make_windows(val_series)

    def norm(a):
        return (a - mean) / std

    Xtr_t = torch.from_numpy(norm(Xtr)).unsqueeze(-1)
    ytr_t = torch.from_numpy(norm(ytr))
    Xva_t = torch.from_numpy(norm(Xva)).unsqueeze(-1)
    yva_t = torch.from_numpy(norm(yva))

    model = LSTMForecaster()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    n = len(Xtr_t)
    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for a in range(0, n, batch_size):
            idx = perm[a:a + batch_size]
            optim.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            optim.step()
            running += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), yva_t).item()
        print(f"epoch {ep:2d}  train_mse {running / n:.5f}  val_mse {vloss:.5f}", flush=True)
        if vloss < best:
            best = vloss
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)
            save_norm(mean, std)

    print(f"\nBest val MSE: {best:.5f}  ->  saved {WEIGHTS_PATH}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
