# v14 — Data Availability Report (Task 1)

Read-only inspection, 2026-08-03, branch `research/v14-intraday-conditional-playbook`.

## Inventory

| # | Data | Status |
|---|---|---|
| 1 | Intraday 1-min bars | **ABSENT** |
| 2 | Intraday 5-min bars | **ABSENT** |
| 3 | Tick data | **ABSENT** |
| 4 | Bid/ask quotes | **ABSENT** |
| 5 | Order book | **ABSENT** |
| 6 | Daily OHLCV | **PRESENT** — `research/data_cache/` (110 symbols, 2015→, date/open/high/low/close/volume; `data_cache_full/` adds daily turnover+transactions since ~2026-07) |
| 7 | EOD decision books | PRESENT — daily `*_blend50_band10_decision_book.csv` + per-strategy books |
| 8 | Paper ledger | PRESENT — `PAPER_LEDGER.csv` (matured 1/5/10/20d) |
| 9 | Sector mapping | PRESENT — `SECTOR_MAP` (~10 sectors) |
| 10 | Shortability/margin/borrow | **ABSENT** — only the v12 conservative proxy (low-ADV/rally/price-floor exclusions) |
| 11 | Market calendar / trading hours | **PARTIAL** — trading DATES implicit in cache; **no intraday session-time logic anywhere** (TWSE 09:00–13:30 continuous + auction is not encoded; checkpoint times like "09:30" have no data or code support) |
| 12 | Cost assumptions | PRESENT — net60/100/150 bps conventions; v9 break-evens (632/463 bps); v12 2× short-cost convention. **No intraday slippage/fill model exists** (nothing to calibrate one on) |

Fetch layer: `refresh_data.py` uses `twstock.Stock.fetch(year, month)` —
monthly EOD history only. `twstock` also exposes a REALTIME quote endpoint
(current-snapshot only, no history) — usable by a future collector, not for
backfill. No known free source of historical TWSE minute bars for this
universe.

## Answers

**A. True intraday backtest possible now: NO.**

**B. Missing:** any sub-daily price/volume series (minute bars at minimum),
session-time calendar logic, intraday cost/slippage calibration data,
borrow/shortability data. There is no historical backfill path — intraday
history must be COLLECTED forward from the day a collector starts.

**C. Weak daily-bar proxy that CAN be built:** per-day (gap = open/prev_close,
open-to-close return, gap-to-close, high/low range) conditioned on EOD-book
state (action/score/rank) and prior-day context. This supports: gap-bin ×
EOD-action conditional statistics, open-to-close directional tendencies,
avoid-chase rules at the OPEN only, and range-based risk envelopes.

**D. What the proxy CANNOT conclude:** anything time-of-day (09:30/10:00/
11:00 checkpoint rules — no data at those timestamps); stop-vs-target
ordering (daily bars do not reveal whether the high or the low came first —
path ambiguity; any stop/TP label is structurally biased); VWAP distances;
intraday volume patterns; fill realism; slippage. **Every proxy output is
non-decision-grade and must carry `data_quality=DAILY_BAR_PROXY_ONLY`,
`live_trading_allowed=false`.**

**E. Needed:** a forward intraday collector — poll the TWSE MIS/twstock
realtime endpoint for the ~110-name universe on a 1–5-min cadence during
sessions (09:00–13:30 TW), append to
`research/intraday_cache/` (gitignored), with gap/crash recovery and
staleness flags. Realistic timeline: weeks–months of collection before any
checkpoint rule is testable; ~6+ months before decision-grade. Detailed
plan goes in `intraday_data_acquisition_plan.md` (sprint deliverable).

## Consequence for the sprint (per the pre-registered rule)

No model training on intraday targets; no XL usage; the sprint pivots to
its fallback goal: playbook FRAMEWORK + clearly-labeled daily-bar proxy +
collector plan. Expected verdict family:
DATA_MISSING_BUILD_COLLECTOR_FIRST / DAILY_BAR_PROXY_ONLY_NOT_DECISION_GRADE
/ PLAYBOOK_FRAMEWORK_READY.
