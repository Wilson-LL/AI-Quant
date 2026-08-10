# v15 Intraday Collector MVP — Design

**Data collection only.** No orders, no broker APIs (twstock has none —
the collector performs HTTP GETs against the public TWSE MIS quote
endpoint that twstock wraps). Raw data never leaves the machine and is
never committed (`research/intraday_cache/`, `*.sqlite*` gitignored).

## Source (verified live 2026-08-03)

`twstock.realtime.get([codes])` — batched snapshots with:
latest_trade_price, trade_volume (tick), accumulate_trade_volume,
best_bid/ask price+volume ×5, session open/high/low, exchange timestamp,
success flag. `previous_close` is not served → joined from the EOD cache
at startup. String payloads with "-" sentinels are parsed defensively.

## Polling design

- Universe: `--universe book` (default; latest blend50_band10 decision
  book, 30-40 symbols) or `--universe full` (108 non-ETF names). Book-only
  default keeps request volume minimal.
- Cadence: 60 s cycle (configurable, floor 30 s); symbols chunked 20 per
  request with 1.5 s throttle between chunks (refresh_data.py etiquette).
  Book universe ⇒ 2 requests/min; full ⇒ 6/min.
- Session gating: the loop runs only 08:55–13:35 TW, Mon–Fri, and exits
  immediately otherwise. `--once` (single cycle, any time — testing) and
  `--mock N` (synthetic, no network) bypass the gate.
  **Scheduling the session loop requires separate user approval.**

## Robustness

- SQLite WAL, append-only inserts, commit per cycle → crash loses at most
  one cycle; `INSERT OR IGNORE` on the (symbol, timestamp, source) key
  makes re-runs idempotent.
- Every run is a `collector_runs` row; a crashed run's `running` status is
  marked `aborted` by the next startup (restart-safe, v8-recovery
  convention).
- Per-chunk and per-symbol try/except: one bad symbol or failed request
  emits a quality event and never kills the cycle.
- `source` column separates `TWSE_MIS` from `MOCK` so test data can never
  contaminate real data.

## Pipeline

collect_realtime_quotes.py (snapshots + inline quality events) →
build_1m_bars.py (minute OHLC + cumulative-volume deltas, idempotent) →
quality_report.py (per-day per-symbol coverage/events md, gitignored) →
status.py (runs/coverage/events overview).

## Scheduling (user-approved 2026-08-06)

Two Windows Scheduled Tasks (current user, weekdays):
- `AIQuant-IntradayCollector` 08:54 → `collect_realtime_quotes.py
  --universe book --interval 60` (session gate self-terminates at 13:35).
  The collector waits (≤15 min) for the 08:55 gate when launched early —
  without this the 08:54 task exited before the open (found 2026-08-10:
  run 6 recorded 0 cycles; the day was salvaged with a 09:36 manual start)
- `AIQuant-IntradayPostClose` 13:40 → `research/intraday_collector/
  postclose.bat` (bars + quality report)

Limitations: PC must be on/logged-in at 08:54 (no wake configured);
non-trading holidays yield short dedupe-heavy runs (visible in the daily
quality report; trading-calendar guard is a possible refinement).

## Operational notes

- Run on the same box as daily ops is fine (CPU-trivial); the existing
  "no refresh during GPU training" cache-race rule extends to the
  collector only in that both poll TWSE — stagger is polite but not
  required (different endpoints).
- Timeline to usefulness: ~3 months QA-clean data for first descriptive
  statistics; ~6 months for the earliest decision-grade checkpoint rules
  (see v14 intraday_data_acquisition_plan.md).
