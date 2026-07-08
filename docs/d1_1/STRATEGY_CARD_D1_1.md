# Strategy Card — D1.1 Cross-Sectional Momentum (TWSE)

A deterministic, rule-based momentum-factor product. **No machine learning.**
This is a *research prototype* and a **known-factor product, not alpha**
(established in Experiment 4).

## The rule
| step | definition |
|---|---|
| **Signal** | `mom_t = close[t−5] / close[t−131] − 1` (6-month return, skip last 5 days to avoid short-term reversal) |
| **Select** | rank cross-sectionally; **long the top quintile**, **short the bottom quintile** |
| **Size** | **inverse-volatility** — weight ∝ 1 / (trailing 60-day return vol), past-only |
| **Caps** | **20% per sector** and **10% per name** (iterative redistribute; both enforced per leg) |
| **Hold** | **20 trading days**, then rebalance |

Universe: 106 liquid TWSE names across 15 sectors (multi-sector is essential —
in semis-only the signal collapses to a volatility/beta bet; see Exp 2–3).

## Validated performance
Full-sample 2018–2026, net of cost. **Survivorship-biased upward → treat as an
UPPER BOUND.** Long-short book (beta-neutral-ish):

| metric | net@0bps | net@60bps | net@80bps |
|---|--:|--:|--:|
| **Sharpe** | 1.45 | **1.27** | 1.22 |

| | value |
|---|--:|
| annualized return (net@60) | 27.6% |
| max drawdown | −22.1% |
| Calmar | 1.25 |
| turnover | 5.6×/yr |
| breakeven cost | ≈ 390+ bps (cost is not the binding constraint) |
| positive years | 8/9 (only 2022 negative) |
| max name weight | 11% |
| max sector weight | ~21% (soft cap, see notes) |
| max single-sector P&L share | 35% |

Vs alternatives (net@60 Sharpe): Exp-4 **linear ML model 0.62** · A1 equal-weight
momentum 1.22 · D1 (sector cap only) 1.21 · **D1.1 (this) 1.27**. The one-line rule
beats the LSTM-Transformer lineage at a fraction of the complexity.

## Risk profile
- **Factor exposure:** ~pure cross-sectional **momentum** (Exp-4: momentum beta
  0.70, t=9.58; residual alpha t=0.12 ≈ 0). Market/vol/size betas ≈ 0.
- **Worst regime:** 2022 momentum crash (net −1.22%/rebalance). No regime overlay
  helped enough to include (Phase C1 failed its bar).
- **Concentration:** sector P&L still ~35% top-sector (electronics/semis/shipping
  lead); the per-name cap holds single names ≤ ~11%.
- **Long-only variant** (deployable where shorting is hard): higher return but
  carries market beta — risk-adjusted it does **not** beat holding the universe in
  this bull sample. The honest signal is the market-neutral L/S.

## Implementation notes / caveats
- **Sector cap is SOFT:** floored at `max(20%, 1/n_sectors_in_leg)`, so a
  sector-sparse leg can drift to ~21%+ (4 sectors → 25%). Not a hard constraint.
- **Survivorship bias:** currently-listed names only (twstock); all numbers are
  upper bounds.
- **In-sample cap choice:** the 10% cap was selected on the full sample (from a
  pre-declared set, on concentration not Sharpe) — **not yet OOS-validated.**
- **Shorting** TWSE is operationally hard; borrow/uptick constraints not modelled.
- **Costs:** one-way L1 turnover × bps; realistic TWSE round-trip ~60–80 bps tested.
- Not a claim of alpha; it harvests a known premium.

## Reproduce
```
python research/data.py            # (re)build the OHLCV cache (network; ~1-2 min/name)
python research/momentum.py        # A1 raw momentum baseline
python research/portfolio_d1.py    # D1 construction variants (frozen baseline)
python research/d1_1_pername_cap.py# D1.1 per-name cap validation
python research/prototype.py       # D1.1 product: today's book + validation card
python research/test_framework.py  # 8/8 framework tests
```
`generate_book()` emits the current target long/short (and long-only) book;
`backtest_summary()` returns the validated metrics via the same code path.

**Data snapshot & reproducibility.** The cached OHLCV (`research/data_cache/`) is a
**2026-07-06 survivorship snapshot** and is **git-ignored** (not in the repo).
`research/data.py` regenerates it deterministically *given the same TWSE listing*,
but a future refetch uses the then-current listed universe (survivors change,
delisted names never reappear), so exact numbers may differ. To reproduce the
published figures, use the frozen 2026-07-06 cache.

## Out-of-sample validation (WALKFORWARD_D1_1.md)
- **Held-out (select 2018–21, evaluate 2022–26 incl. the crash): OOS Sharpe 1.51**
  (≥ IS 1.20, full 1.27) — survives; not a full-sample overfit.
- **10% cap is robust:** IS-only selection also picks 10% (cap Sharpe flat
  1.18–1.20); adaptive cap-tuning is *worse* (1.00) than fixed-10 (1.13) →
  keep the cap fixed.
- **Survivorship (unmeasurable directly):** L/S > long-only OOS (edge is
  cross-sectional); dropping top-3 names cuts OOS Sharpe 1.51→1.33 (−12%).
- **Reasoned haircut ~20–35%** (survivorship + frictions + favourable OOS regime)
  → **realistic deployable ~0.9–1.1 net Sharpe**, not the headline 1.5. Blended
  cross-regime ~1.1–1.3.

## Robustness (5-hour sprint, cycles C1–C8)
| test | result |
|---|---|
| **C1** stricter OOS (multi-fold + bootstrap) | PASS — folds 80% positive; block-bootstrap 90% CI on net@60 Sharpe **[0.47, 1.99]** > 0 |
| **C2** parameter sensitivity (25 settings) | positive at **100%**, but magnitude-sensitive (median ~1.0, min 0.66) — 1.27 is a favorable pick |
| **C3** implementation realism | PASS — survives TWSE price-limit fills + 150 bps (only 28 unfillable entries) |
| **C4** regime stability | **partial — BULL 1.81 vs BEAR −0.42**, HIGH-vol 0.58 vs LOW-vol 2.35; DD recovers ~8 mo |
| **C5** hard sector cap | keep soft — hard cap counterproductive (22% > 21%), infeasible in a concentrated book |
| **C6** alternative momentum | general momentum (all defs positive 0.73–1.27); composite ~1.08; 126/5 favorable |
| **C7** universe bootstrap (200 draws, drop 20%) | PASS — **100% positive**, Sharpe 5/50/95 = **0.95/1.17/1.39** |
| **C8** factor regression of the book | momentum beta 0.83 (t=21.6); **residual alpha t=1.60 (ns)** → known factor, not alpha |

**Net honest read:** the momentum edge is robust across OOS, universe, cost/friction,
and momentum definition — but the 1.27 headline is a favorable pick, and **bear /
high-vol regimes are the one real vulnerability** (the book carries mild market beta,
C8). **Realistic expected deployable Sharpe ~0.8–1.0** after parameter + regime +
survivorship + friction haircuts. It is a known-factor (momentum) product, not alpha.
*Future robustness options (not adopted):* multi-horizon rank-composite signal (C6,
diversifies signal risk at ~1.08).

## Status
Research milestone, **OOS-validated**. **Not production / not capital-ready.**
Production `train.py` / `inference.py` intentionally untouched. Next gate before any
capital use: model live frictions (borrow/slippage/price limits) and, for genuine
alpha beyond the momentum premium, Path B (orthogonal data).
