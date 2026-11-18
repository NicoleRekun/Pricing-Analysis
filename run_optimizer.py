"""Fit win-rate models on the DEMO quote history and recommend prices.

    python data/generate_demo_data.py
    python run_optimizer.py

Writes output/recommendations.csv and docs/data.js (for the dashboard).
Everything is FICTIONAL demo data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optimizer.optimize import CAP_INDEX, recommend, summarise
from optimizer.winrate import fit_segments

BANNER = "=" * 70 + "\n DEMO DATA ONLY: fictional company, products, prices and quotes\n" + "=" * 70


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(Path("data") / name, comment="#")


def main() -> None:
    print(BANNER)
    products, quotes = load("products.csv"), load("quotes.csv")
    models = fit_segments(quotes)

    print("\nWin-rate model by segment  (logit P = a + b*(price/competitor - 1))")
    for m in models.values():
        print(f"  {m.segment:<11} a={m.a:6.2f}  b={m.b:7.2f}  50% win at {m.breakeven_index:.3f}x competitor"
              f"   ({m.n_quotes} quotes, observed win {m.observed_win_rate:.1%})")

    rec = recommend(products, quotes, models)
    s = summarise(rec)
    Path("output").mkdir(exist_ok=True)
    with open("output/recommendations.csv", "w", newline="") as f:
        f.write("# DEMO DATA ONLY - fictional company, products, prices and quotes\n")
        rec.to_csv(f, index=False)

    print(f"\nAcross {s['pairs']} product x segment pairs (demo year of quotes):")
    print(f"  gross profit  {s['current_gross_profit']:>12,.0f}  ->  {s['recommended_gross_profit']:>12,.0f}"
          f"   ({s['uplift_pct']:+.1f}%)")
    print(f"  win rate      {s['current_win_rate']:>12.1%}  ->  {s['recommended_win_rate']:>12.1%}")
    print("\nBinding constraint counts:")
    print(rec["binding_constraint"].value_counts().to_string())

    payload = {
        "notice": "DEMO DATA ONLY - fictional company, products, prices and quotes",
        "cap_index": CAP_INDEX,
        "summary": s,
        "models": {k: {"a": round(v.a, 4), "b": round(v.b, 4), "n": v.n_quotes,
                       "observed": round(v.observed_win_rate, 4)} for k, v in models.items()},
        "rows": json.loads(rec.to_json(orient="records")),
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data.js").write_text("window.PRICING_DATA = " + json.dumps(payload) + ";\n")
    print("\nWrote output/recommendations.csv and docs/data.js")
    print(BANNER)


if __name__ == "__main__":
    main()
