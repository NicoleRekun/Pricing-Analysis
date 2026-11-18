# Pricing-Analysis: Win-Rate Pricing Optimizer

> [!WARNING]
> **DEMO DATA ONLY.** Every number in this repository belongs to a **fictional** company, *Maple Grid Electrical Supply*.
> Products, costs, list prices, margin floors, competitor prices and the 4,000-quote history are **invented** by
> [`data/generate_demo_data.py`](data/generate_demo_data.py) from a fixed random seed. No real company's data appears here.
> Every CSV file starts with a `# DEMO DATA ONLY` line, and the dashboard and console output carry the same banner.

**Live dashboard:** https://nicolerekun.github.io/Pricing-Analysis/ (fictional demo data)

## The question

What price should a distributor quote for each product and customer type to earn the most gross profit,
given that **higher prices win fewer deals**?

The optimizer combines three kinds of information:

| Input | Examples | Role |
|---|---|---|
| **Internal** | unit cost, list price, margin floor per product class | Guardrails and profit per win |
| **Competitive** | lowest competitor price per product | Defines the price index the market reacts to |
| **Sales outcomes** | past quotes: price offered, customer segment, won or lost | Teaches the model how price changes the chance of winning |

Customer segments are **Contractor**, **Industrial** and **Reseller** (partner accounts that buy to resell).

## How it works

1. **Win-rate model per segment.** Logistic regression fitted by maximum likelihood (NumPy, Newton-Raphson):

   `logit P(win) = a + b × (price ÷ lowest competitor − 1)`

2. **Expected profit.** `P(win | price) × (price − unit cost)`. Each extra dollar of price earns more per win but
   lowers the win rate; the best price balances the two.

3. **Guardrails from internal policy.** The price is searched only between:
   - the **margin floor** price, `cost ÷ (1 − floor %)`
   - the lower of **list price** and **1.15 × lowest competitor**

   If the floor is above the competitor cap, the floor wins and the case is flagged.

4. **Report.** Recommended price, expected win rate, margin, expected profit per quote, and which guardrail (if any)
   decided the answer.

## Results on the demo data

```
Win-rate model by segment  (logit P = a + b*(price/competitor - 1))
  Contractor  a=  0.39  b=  -8.93  50% win at 1.044x competitor
  Industrial  a=  1.02  b=  -4.21  50% win at 1.243x competitor
  Reseller    a=  0.01  b= -16.17  50% win at 1.001x competitor

Across 90 product x segment pairs (demo year of quotes):
  gross profit        95,813  ->       118,256   (+23.4%)
  win rate             60.1%  ->         56.6%
```

The demo history was simulated from known "true" parameters (Contractor b = −9, Industrial −5, Reseller −15),
so the fit can be checked: it recovers them closely. The optimizer raises prices where buyers are
price-tolerant (Industrial) and sharpens them where they are not (Reseller), trading a few points of win rate for
more profit.

## Run it

```bash
pip install -r requirements.txt
python data/generate_demo_data.py    # builds the fictional data set
python run_optimizer.py              # fits models, writes output/ and the dashboard data
pytest -q                            # 6 tests
```

## Layout

```
data/generate_demo_data.py   FICTIONAL data generator (see data/README.md)
optimizer/winrate.py         logistic win-rate model, MLE fit
optimizer/optimize.py        profit-maximising price within guardrails
run_optimizer.py             end-to-end run, writes recommendations and dashboard data
docs/index.html              interactive dashboard (GitHub Pages ready)
output/recommendations.csv   recommended price per product x segment
tests/                       model recovery, guardrails, brute-force optimum check
```

## Earlier work in this repo

`Retail_Price.xlsx` is the public retail price optimisation practice dataset used in this repo's first version (R script). It is unrelated to the demo data above.

## Limits and next steps

- One competitor price per product; real data would weight competitors and track price dates.
- Win rate depends only on price index and segment; adding deal size, customer history and lead time would sharpen it.
- Before use on real data: validate on a hold-out period, run in shadow mode beside analysts, refresh monthly.

---
*Demo data only. Built by Nicole Rekun.*
