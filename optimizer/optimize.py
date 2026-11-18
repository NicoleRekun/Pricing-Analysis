"""Choose the price that maximises expected gross profit per quote, within guardrails.

    expected profit(price) = P(win | price) * (price - unit cost)

Guardrails (internal information):
  * margin floor   - never below the item class's minimum gross margin
  * competitor cap - never more than CAP_INDEX x the lowest competitor price
  * list cap       - never above list price
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .winrate import WinModel

CAP_INDEX = 1.15
GRID_POINTS = 400


@dataclass
class Recommendation:
    sku: str
    description: str
    item_class: str
    segment: str
    unit_cost: float
    list_price: float
    lowest_competitor_price: float
    floor_price: float
    current_avg_price: float
    current_win_rate: float
    current_profit_per_quote: float
    recommended_price: float
    recommended_win_rate: float
    recommended_margin_pct: float
    recommended_profit_per_quote: float
    quotes_per_year: int
    binding_constraint: str


def floor_price(cost: float, margin_floor_pct: float) -> float:
    return cost / (1 - margin_floor_pct / 100)


def price_bounds(cost, list_price, low_comp, margin_floor_pct):
    lo = floor_price(cost, margin_floor_pct)
    hi = min(list_price, CAP_INDEX * low_comp)
    return lo, hi


def best_price(model: WinModel, cost, list_price, low_comp, margin_floor_pct):
    lo, hi = price_bounds(cost, list_price, low_comp, margin_floor_pct)
    if hi <= lo:  # guardrails conflict: margin floor wins
        p = lo
        return p, float(model.p_win(p / low_comp)), "margin floor (above competitor cap)"
    grid = np.linspace(lo, hi, GRID_POINTS)
    win = model.p_win(grid / low_comp)
    profit = win * (grid - cost)
    k = int(np.argmax(profit))
    if k == 0:
        bind = "margin floor"
    elif k == GRID_POINTS - 1:
        bind = "list price" if hi == list_price else "competitor cap"
    else:
        bind = "none (unconstrained optimum)"
    return float(grid[k]), float(win[k]), bind


def recommend(products: pd.DataFrame, quotes: pd.DataFrame, models: dict[str, WinModel]) -> pd.DataFrame:
    hist = (quotes.assign(profit=lambda d: d["won"] * 0)  # placeholder, filled below
                  .merge(products[["sku", "unit_cost"]], on="sku"))
    hist["profit"] = hist["won"] * (hist["quoted_price"] - hist["unit_cost"])
    grp = hist.groupby(["sku", "segment"]).agg(
        current_avg_price=("quoted_price", "mean"),
        current_win_rate=("won", "mean"),
        current_profit_per_quote=("profit", "mean"),
        n=("won", "size"),
    ).reset_index()

    low = quotes.groupby("sku")["lowest_competitor_price"].first()
    rows = []
    for _, g in grp.iterrows():
        p = products.loc[products["sku"] == g["sku"]].iloc[0]
        m = models[g["segment"]]
        price, win, bind = best_price(m, p["unit_cost"], p["list_price"], low[g["sku"]],
                                      p["margin_floor_pct"])
        rows.append(Recommendation(
            sku=p["sku"], description=p["description"], item_class=p["item_class"],
            segment=g["segment"], unit_cost=p["unit_cost"], list_price=p["list_price"],
            lowest_competitor_price=float(low[g["sku"]]),
            floor_price=round(floor_price(p["unit_cost"], p["margin_floor_pct"]), 2),
            current_avg_price=round(g["current_avg_price"], 2),
            current_win_rate=round(g["current_win_rate"], 4),
            current_profit_per_quote=round(g["current_profit_per_quote"], 2),
            recommended_price=round(price, 2),
            recommended_win_rate=round(win, 4),
            recommended_margin_pct=round((price - p["unit_cost"]) / price * 100, 2),
            recommended_profit_per_quote=round(win * (price - p["unit_cost"]), 2),
            quotes_per_year=int(g["n"]),
            binding_constraint=bind,
        ))
    return pd.DataFrame([asdict(r) for r in rows])


def summarise(rec: pd.DataFrame) -> dict:
    cur = (rec["current_profit_per_quote"] * rec["quotes_per_year"]).sum()
    new = (rec["recommended_profit_per_quote"] * rec["quotes_per_year"]).sum()
    q = rec["quotes_per_year"]
    return {
        "current_gross_profit": round(float(cur), 2),
        "recommended_gross_profit": round(float(new), 2),
        "uplift_pct": round(float((new - cur) / cur * 100), 1),
        "current_win_rate": round(float((rec["current_win_rate"] * q).sum() / q.sum()), 4),
        "recommended_win_rate": round(float((rec["recommended_win_rate"] * q).sum() / q.sum()), 4),
        "pairs": int(len(rec)),
    }
