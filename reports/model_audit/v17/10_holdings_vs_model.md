# 10 — Actual holdings vs model portfolio (B6, B16-G) — hardened

Four objects are kept separate:

| | Object | Data available? |
|---|---|---|
| A | model **target** portfolio (blend50_band10 book, 22 names, equal-ish weights) | yes — `reports/paper_trading/<date>_blend50_band10_decision_book.csv` |
| B | model **paper** portfolio (ledger of A's snapshots with matured returns) | yes — `PAPER_LEDGER.csv` (gross, overlapping) |
| C | user's **current** holdings | yes — `my_holdings.csv` (11 lots, no entry dates) |
| D | user's **actual historical executions** (fills, dates, prices) | **NO — not recorded anywhere** |

## Verdict on attribution: **ACTUAL_REALIZED_LOSS_ATTRIBUTION_INCOMPLETE**

Because D does not exist, this audit can quantify how far C diverges
from A *today*, and it can characterize B's realized path — but it
**cannot** decompose the user's historical P&L into "model was wrong" vs
"portfolio differed from the model" vs "timing of entries". Statements
elsewhere in this audit that divergence is the *primary* explanation of
realized pain are hereby narrowed to: **current divergence is large and
sufficient to make realized results unrepresentative of the model; its
share of past losses is not measurable.**

## Current-portfolio divergence (C vs A, 2026-09-11 plan; measurable)

| Metric | Value |
|---|---|
| Fraction of priced capital in official model names | **58.2%** (8 of 11 lots) |
| Unselected-held exposure (in universe, not in book) | **22.1%** (2330, rank #45) |
| Outside-model exposure | **19.7%** priced (0050, ETF) + 6669 unpriced (not configured) |
| Omitted model-target exposure (targets of names not held) | **55.0%** of the model book (14 of 22 names unheld) |
| One-way deviation to target, ½·Σ\|w_actual − w_target\| over the union | **0.61** |
| Concentration: HHI 0.136 → effective N ≈ 7.4; top-2 = 41.8%; max weight 22.1% vs model cap 10% | |
| Value-weighted unrealized P&L (cost basis, priced lots) | +1.7% |

Per-lot table: see `latest_next_session_summary.md` (# 我的實際持倉) —
2330 REDUCE_LONG (unselected), 0050/6669 NO_MODEL_OPINION, 1326/2357/2454
/2408 REDUCE_LONG (overweight vs target), 2883/2887/2308 HOLD_LONG,
6446 ADD_LONG.

## Model paper portfolio (B), for context only

32 matured 20d snapshots since 2026-07-07: mean +9.98%, 84% positive
books, but heavily overlapping (daily snapshots of a 20-day return) and
dominated by the late-July rebound after a −18.6% block (2026-07-01).
Effective independent observations ≈ 2. Not comparable to the research
Sharpes (gross, overlapping — timing-audit F4).

## What would make attribution possible

Add to `my_holdings.csv` (optional, backward-compatible): `entry_date`
(and ideally `entry_price`); or keep a simple fills log. With entry
dates the plan can compute post-entry high / drawdown / time-in-position
per lot and the audit can compare each lot against the model's action
history on the same dates. Until then: divergence YES, historical
attribution NO.

## Cost basis (A2/A3) — unchanged

Context only; no action rule reads `avg_cost` (`map_user_action` has no
cost parameter; asserted by test). A4 position-exit research on user
positions: **POSITION_EXIT_RESEARCH_DATA_INSUFFICIENT**; feasible on the
model's paper positions (P7).
