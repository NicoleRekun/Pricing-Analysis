"""Estimate how likely a quote is to win, given its price relative to competitors.

Model (one per customer segment):
    logit P(win) = a + b * (price_index - 1)
    price_index  = quoted price / lowest competitor price

b is negative: the further above the competition you price, the less you win.
Fitted by maximum likelihood with Newton-Raphson (plain NumPy, no black box).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class WinModel:
    segment: str
    a: float
    b: float
    n_quotes: int
    observed_win_rate: float

    def p_win(self, price_index):
        z = self.a + self.b * (np.asarray(price_index, dtype=float) - 1.0)
        return 1.0 / (1.0 + np.exp(-z))

    @property
    def breakeven_index(self) -> float:
        """Price index at which the win rate is 50%."""
        return 1.0 - self.a / self.b


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-6):
    """Return (a, b) for logit P(y=1) = a + b*x."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (y - p) - ridge * w
        H = -(X.T * (p * (1 - p))) @ X - ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(w[0]), float(w[1])


def fit_segments(quotes) -> dict[str, WinModel]:
    """quotes: pandas DataFrame with segment, quoted_price, lowest_competitor_price, won."""
    models = {}
    for seg, g in quotes.groupby("segment"):
        x = (g["quoted_price"] / g["lowest_competitor_price"]).to_numpy() - 1.0
        y = g["won"].to_numpy(dtype=float)
        a, b = fit_logistic(x, y)
        models[seg] = WinModel(seg, a, b, len(g), float(y.mean()))
    return models
