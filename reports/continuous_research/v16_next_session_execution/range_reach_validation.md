# v16 Stage B2 — Range-Reach Probability Validation

Date: 2026-08-18 · Data: range_reach_validation.csv · Definitions frozen
in price_domain_methodology.md: BUY reach = next-day LOW ≤ level; SELL
reach = next-day HIGH ≥ level. **Range reach is a daily-range statistic,
never a fill probability** — order existence, queue priority, volume,
and slippage are not modeled; daily bars carry no intraday ordering.

## Calibration quality

| Side | n (level-obs) | Brier | mean predicted | mean realized |
|---|---|---|---|---|
| BUY | 4,302 | 0.146 | 75.0% | 79.8% |
| SELL | 807 | 0.164 | 70.4% | 70.3% |

Decile monotonicity (BUY): predictions rise 17%→93% across deciles and
realized frequencies rise with them (46%→94% from decile 2 up; decile 1
has n=4). SELL deciles similarly monotone. The BUY side is ~5pp
**conservative** (realized reach exceeds predicted) — the safe direction
for a tool that argues for patience; per pre-registration it is NOT
retuned.

Stability: by year, BUY predicted-vs-realized stays within ±12pp
(2022 the most conservative: 75.9% vs 87.6%); **2026 YTD** BUY 72.2% vs
79.2% (fine), while SELL 2026 is the weak spot (69.6% predicted vs
54.2% realized, n=48, Brier 0.24) — high-vol regime again; flagged, not
tuned. By vol regime: within ±7pp everywhere.

## Waiting/pullback tradeoff (entries; n=1,434 per level)

| Level vs reference | Predicted reach | Realized reach | T+1_RANGE_DID_NOT_REACH_LEVEL (realized) |
|---|---|---|---|
| reference (0.0%) | 83.4% | 86.1% | 13.9% |
| −0.5% | 64.8% | 72.2% | 27.8% |
| −1.0% | 51.6% | 60.1% | 39.9% |
| −1.5% | 40.2% | 47.5% | 52.5% |
| −2.0% | 31.3% | 37.6% | 62.4% |

Reading: waiting for a 1% discount historically got range-reached on
~60% of next sessions; a 2% discount, ~38%. The complement is labeled
T+1_RANGE_DID_NOT_REACH_LEVEL — NOT a "missed trade": a manual user can
still act later within the 20-session horizon. This is exactly the
patience-vs-participation tradeoff the nightly reach lines quantify.

No fill guarantees; no path-order claims; nothing retuned post hoc.
