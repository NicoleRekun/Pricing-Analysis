"""
Build the DEMO data set for the pricing optimizer.

!!! EVERYTHING THIS SCRIPT CREATES IS FICTIONAL !!!
Fictional company "Maple Grid Electrical Supply". Products, costs, prices,
competitors, customer segments and the quote history are invented, drawn from
a random generator with a fixed seed. Nothing comes from a real company.

The quote history is simulated from a known "true" win-rate model, so the
optimizer's estimates can be checked against the truth (see tests/).

Run:  python data/generate_demo_data.py
Writes: data/products.csv, data/competitor_prices.csv, data/quotes.csv
"""
import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 2026
NOTICE = "DEMO DATA ONLY - fictional company, products, prices and quotes"

# item class -> (margin floor %, cost range CAD, names)
CLASSES = {
    "WIRE": (12.0, (40, 400), ["Copper building wire 12 AWG 75 m", "Copper building wire 14 AWG 75 m",
                               "Armoured cable 12/2 75 m", "Armoured cable 14/3 75 m", "Low-voltage cable 18/2 150 m",
                               "Ground wire 6 AWG 30 m"]),
    "CONDUIT": (15.0, (5, 60), ["EMT conduit 1/2 in 3 m", "EMT conduit 3/4 in 3 m", "PVC conduit 1 in 3 m",
                                "Flexible conduit 1/2 in 15 m", "Liquid-tight conduit 3/4 in 15 m", "Conduit strap 1/2 in, box of 100"]),
    "BREAKER": (18.0, (20, 250), ["Single-pole breaker 15 A", "Double-pole breaker 30 A", "GFCI breaker 20 A",
                                  "AFCI breaker 15 A", "Load centre 100 A", "Surge protective device"]),
    "LIGHTING": (22.0, (15, 180), ["LED panel 2x4 ft 40 W", "LED high bay 150 W", "LED wall pack 30 W",
                                   "LED strip 5 m 24 V", "LED troffer retrofit kit", "Emergency light unit"]),
    "TOOLS": (25.0, (30, 300), ["Cordless impact driver kit", "Fish tape 30 m", "Digital multimeter",
                                "Conduit bender 1/2 in", "Cable cutter", "Voltage tester"]),
}

# Who we sell to. "Reseller" = supplier/partner accounts that buy to resell.
# TRUE win model (hidden from the optimizer): logit(p) = alpha - beta * (price_index - 1)
# where price_index = quoted price / lowest competitor price.
SEGMENTS = {
    "Contractor": {"alpha": 0.4, "beta": 9.0, "share": 0.45},
    "Industrial": {"alpha": 0.9, "beta": 5.0, "share": 0.30},
    "Reseller":   {"alpha": 0.1, "beta": 15.0, "share": 0.25},
}
COMPETITORS = ["Demo Competitor A", "Demo Competitor B", "Demo Competitor C"]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def build(n_quotes: int = 4000) -> None:
    rng = random.Random(SEED)
    products, comps, quotes = [], [], []

    n = 2000
    for cls, (floor, (lo, hi), names) in CLASSES.items():
        for name in names:
            n += 1
            sku = f"DEMO-{n}"
            cost = round(rng.uniform(lo, hi), 2)
            list_price = round(cost * rng.uniform(1.32, 1.60), 2)
            products.append(dict(sku=sku, description=name, item_class=cls, unit_cost=cost,
                                 list_price=list_price, margin_floor_pct=floor))
            for c in COMPETITORS:
                comps.append(dict(sku=sku, competitor=c,
                                  price=round(list_price * rng.uniform(0.80, 1.04), 2),
                                  observed_on="2026-10-15"))

    low_comp = {}
    for c in comps:
        low_comp[c["sku"]] = min(low_comp.get(c["sku"], 1e9), c["price"])

    seg_names = list(SEGMENTS)
    seg_weights = [SEGMENTS[s]["share"] for s in seg_names]
    # some products are quoted far more often than others
    popularity = {p["sku"]: rng.uniform(0.3, 2.0) for p in products}
    skus = list(popularity)
    sku_w = list(popularity.values())
    by_sku = {p["sku"]: p for p in products}

    for i in range(n_quotes):
        sku = rng.choices(skus, sku_w)[0]
        seg = rng.choices(seg_names, seg_weights)[0]
        p = by_sku[sku]
        # historical habit: reps discount 0-22% off list, a bit more for resellers
        disc = rng.uniform(0.0, 0.22) + (0.03 if seg == "Reseller" else 0.0)
        price = round(p["list_price"] * (1 - disc), 2)
        idx = price / low_comp[sku]
        s = SEGMENTS[seg]
        won = rng.random() < sigmoid(s["alpha"] - s["beta"] * (idx - 1))
        month = 1 + i * 12 // n_quotes
        quotes.append(dict(quote_id=f"DEMO-Q{i + 1:05d}", quote_month=f"2026-{month:02d}", sku=sku,
                           segment=seg, quoted_price=price, lowest_competitor_price=low_comp[sku],
                           won=int(won)))

    for fname, rows in [("products.csv", products), ("competitor_prices.csv", comps), ("quotes.csv", quotes)]:
        with open(HERE / fname, "w", newline="") as f:
            f.write(f"# {NOTICE}\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{NOTICE}\nproducts={len(products)} competitor_prices={len(comps)} quotes={len(quotes)}")


if __name__ == "__main__":
    build()
