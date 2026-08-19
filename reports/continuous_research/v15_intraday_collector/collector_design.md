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

## Reliability hardening (2026-08-20)

Motivating incidents: 2026-08-18 two collector instances polled
concurrently (unknown system-python launcher + the scheduled venv task;
one never registered a run row); 2026-08-19 the scheduled collector
process died silently at ~09:44 with no captured traceback and no
restart (37 min of session data lost until manual restart).

- **Single-instance lock** (collect_realtime_quotes.py): per-DB OS
  byte-range lock (`msvcrt.locking`) acquired before any DB write; a
  second instance prints a clear message and exits with the distinct
  code **EXIT_ALREADY_RUNNING = 75** (EX_TEMPFAIL — not a crash, not
  success). The OS releases the lock on ANY process death — no
  stale-lockfile recovery needed. Lock file `<db>.collector.lock`
  (scratch-DB tests never collide).
- **Supervisor watchdog** (collector_supervisor.py — the scheduled task
  now runs THIS). Observability contract (precise): child stdout/stderr
  are captured to `research/intraday_cache/supervisor_<date>.log`
  whenever emitted, and every child launch and exit (with return code)
  is always recorded — so an unexpected termination is observable even
  when no Python traceback was produced (hard kill / interpreter
  crash). Behavior inside the supervision window 08:50–13:34:
  rc 75 → **STANDBY** (another collector owns the lock — the 2026-08-18
  mode): retry every 60 s WITHOUT consuming the crash budget, until the
  window ends or the rogue instance dies and this supervisor takes
  over; any other rc → unexpected death: restart after 30 s, bounded by
  **max 10 CONSECUTIVE failures** (a child that stayed alive ≥ 5 min
  before dying resets the consecutive counter — the budget guards
  against crashloops, not against unrelated failures spread over the
  session). Child exit after 13:34 = normal completion. The supervisor
  holds its own `<db>.supervisor.lock`. Run-ledger restart-safety is
  unchanged (dead child's 'running' row → 'aborted' at next child
  startup).
- AIQuant-IntradayCollector task command updated 2026-08-20 to
  `collector_supervisor.py --universe book --interval 60` (same 08:54
  Mon–Fri trigger, same interactive-logon mode). Post-close task
  unchanged.

## Operational notes

- Run on the same box as daily ops is fine (CPU-trivial); the existing
  "no refresh during GPU training" cache-race rule extends to the
  collector only in that both poll TWSE — stagger is polite but not
  required (different endpoints).
- Timeline to usefulness: ~3 months QA-clean data for first descriptive
  statistics; ~6 months for the earliest decision-grade checkpoint rules
  (see v14 intraday_data_acquisition_plan.md).
