# v14 Intraday Data Acquisition Plan (Task 9.5) — PLAN ONLY, no collector built

No historical TWSE minute data is freely backfillable for this universe;
intraday history must be collected FORWARD. Nothing here runs without
separate user approval.

## Proposed collector (research/intraday_collector.py — NOT implemented)

- Source: twstock realtime / TWSE MIS snapshot endpoint (best price, last
  trade, cumulative volume per symbol).
- Cadence: poll the ~110-name universe every 60 s during the session
  (09:00–13:30 TW, Mon–Fri, trading days inferred from the EOD calendar);
  ~270 snapshots/day/symbol; aggregate to 1-min bars (last/cum-volume
  diff) and derive 5-min bars.
- Storage: `research/intraday_cache/<sym>/<YYYY-MM>.csv` (gitignored),
  append-only, crash-safe (rewrite-per-snapshot-batch, dedupe on read —
  same conventions as the EOD cache).
- Integrity: staleness flags (no snapshot > 3 min), session-completeness
  score per day, clock-skew guard, throttle + backoff (TWSE-friendly, same
  etiquette as refresh_data.py).
- Operational cost: a lightweight process running during sessions
  (CPU-trivial); must NOT run on the GPU box while training jobs run
  refresh-sensitive code (existing cache-race rule extends to it).
- VWAP: cumulative turnover/cumulative volume per snapshot when the
  endpoint provides value traded; else derived approximation flagged.

## Accumulation timeline to usefulness

- ~1 month: sanity/QA only (bar integrity, volume shape).
- ~3 months (~60 sessions): first descriptive checkpoint statistics;
  wide CIs; screen-grade at best.
- **~6 months: earliest decision-grade window** for simple checkpoint
  rules (one regime only); a year+ for regime robustness. This is the
  hard gate for everything the v14 framework left as placeholders.

## Alternatives (if faster history is wanted)

Paid/licensed TWSE historical intraday vendors exist; acquiring them is a
budget decision, out of scope. The collector remains worth running
regardless (free, compounding asset).
