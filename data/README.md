# Demo data: fictional only

**Nothing in this folder is real.** `generate_demo_data.py` writes three CSV files for a made-up distributor,
*Maple Grid Electrical Supply*. Each file starts with the line `# DEMO DATA ONLY ...`.

| File | Rows | Contents | How it was made |
|---|---|---|---|
| `products.csv` | 30 | SKUs `DEMO-2001` to `DEMO-2030`, generic product names, cost, list price, margin floor | Random cost per class, random 32-60% markup |
| `competitor_prices.csv` | 90 | `Demo Competitor A/B/C` prices | Random 80-104% of list price |
| `quotes.csv` | 4,000 | Quote ID, month, SKU, segment, quoted price, lowest competitor price, won (1/0) | Random discount off list; win drawn from a known logistic model |

The "true" win model used to simulate outcomes is in the generator (`SEGMENTS`), so the optimizer's estimates can be
checked against it. The fixed seed (2026) gives everyone identical data.
