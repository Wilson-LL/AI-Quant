# 08 — Transaction cost & execution audit (B10, B11)

## What the research cost model is

`net60` = 60 bps **round trip** charged on one-way L1 turnover = **30 bps
per side** (`transformer_portfolio.py:161-163`); METHODS.md wrongly
describes it as one-way 60. Shorts are charged the same flat rate with no
borrow fee — so **L/S numbers are not implementable and long-only is the
reference** (CH 1.989 / BR 1.455 at net60).

## What a TWSE long round trip actually costs

| Component | Typical | Notes |
|---|---|---|
| Brokerage commission | 0.1425% list; 0.04–0.08% with common online discounts | charged both sides |
| Securities transaction tax | **0.30%** on sells (0.10% for ETFs) | sell side only |
| Half-spread × 2 | ~0.05–0.20% for TW50-class names at 1–2 ticks | universe is large/mid cap |
| Slippage / impact | ≈ 0 at the user's size (v9-X2: p95 participation 0.4% of ADV at 10M TWD) | |
| **Round trip, fees only** | **0.38–0.59%** (38–59 bps) | ≈ net60 |
| **Round trip incl. spread** | **≈ 60–100 bps** | ≈ net60…net100 |

So net60 ≈ fee-only reality and **net100 is the realistic all-in figure**
for a manual next-open executor. No cost assumption in the 60–100 range
changes any ranking of alternatives (below).

## Sensitivity (long-only, identical protocol; `baselines.csv`)

| Signal | CH net0 | net60 | net100 | net150 | BR net60 | net100 | net150 |
|---|---|---|---|---|---|---|---|
| blend50+band10 (champion) | 2.058 | 1.989 | **1.943** | 1.885 | 1.455 | **1.405** | 1.343 |
| mom126_5 only | 1.797 | 1.708 | 1.648 | 1.574 | 1.361 | 1.301 | 1.227 |
| tf only | 1.820 | 1.733 | 1.675 | 1.602 | 1.193 | 1.136 | 1.066 |

Retention net60→net100 ≈ 0.97; break-even from v9-X1: 632 bps CH / 463
bps BR. **Costs are not the problem** (B16-C rejected); live one-way
turnover is 0.09 per rebalance, ~40% of the backtest's.

## Execution timing (B11)

- Next-open vs next-close: validated equivalent (retention 1.005 / 1.014;
  v16 Task 2). Entrants gap only +12–26 bps median overnight.
- First 5-min / 15-min / VWAP proxies: **not evaluable** — intraday data
  exists only from the v15 collector (Aug-2026 onward, ~30 sessions); the
  historical cache is EOD OHLCV. No fabricated intraday history. Revisit
  when the collector has ≥ 6 months (its stated decision-grade clock).
- Live execution-state layer works but the intraday collector has now
  died silently mid-session twice (08-25, 09-14) with the *supervisor*
  itself vanishing without an exit record. Reliability item, not an
  alpha item — logged for the v15 line.

## Corporate actions (link to R1)

Backtest returns are price returns on unadjusted data: dividends are
neither received nor adjusted. For a long-only TW large-cap book that
understates realized return by roughly the universe yield (~3–4%/yr) and
biases the Q3 labels; see 02/R1 and hypothesis H3.
