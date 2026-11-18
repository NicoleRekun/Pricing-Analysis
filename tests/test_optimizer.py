"""Tests use small FICTIONAL inputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.optimize import CAP_INDEX, best_price, floor_price  # noqa: E402
from optimizer.winrate import WinModel, fit_logistic  # noqa: E402


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.2, 0.2, 20000)
    p = 1 / (1 + np.exp(-(0.5 - 8.0 * x)))
    y = (rng.random(x.size) < p).astype(float)
    a, b = fit_logistic(x, y)
    assert abs(a - 0.5) < 0.08 and abs(b + 8.0) < 0.6


def test_floor_price():
    assert round(floor_price(80, 20), 2) == 100.0


def test_never_below_margin_floor():
    m = WinModel("t", a=0.0, b=-40.0, n_quotes=1, observed_win_rate=0.5)  # very price-sensitive
    price, _, bind = best_price(m, cost=80, list_price=130, low_comp=100, margin_floor_pct=20)
    assert price >= floor_price(80, 20) - 1e-9
    assert bind.startswith("margin floor")


def test_never_above_competitor_cap_or_list():
    m = WinModel("t", a=5.0, b=-0.1, n_quotes=1, observed_win_rate=0.9)  # barely price-sensitive
    price, _, _ = best_price(m, cost=50, list_price=200, low_comp=100, margin_floor_pct=10)
    assert price <= min(200, CAP_INDEX * 100) + 1e-9


def test_interior_optimum_matches_brute_force():
    m = WinModel("t", a=0.4, b=-9.0, n_quotes=1, observed_win_rate=0.5)
    price, _, bind = best_price(m, cost=70, list_price=150, low_comp=100, margin_floor_pct=5)
    grid = np.linspace(70.0 / 0.95, 115, 100000)
    brute = grid[np.argmax(m.p_win(grid / 100) * (grid - 70))]
    assert abs(price - brute) < 0.2
    assert bind.startswith("none")


def test_demo_data_is_labelled():
    root = Path(__file__).resolve().parent.parent / "data"
    for f in ["products.csv", "quotes.csv", "competitor_prices.csv"]:
        p = root / f
        if p.exists():
            assert p.read_text().startswith("# DEMO DATA ONLY")
