# v15 Collector — Database Schema

SQLite (WAL) at `research/intraday_cache/intraday.sqlite` (gitignored).

## intraday_quotes  — one row per (symbol, exchange-timestamp, source)
symbol TEXT · timestamp TEXT (exchange time "YYYY-MM-DD HH:MM:SS") ·
price REAL (NULL when invalid — gap analyzable) · volume REAL (tick) ·
cumulative_volume REAL · bid_price REAL (best bid) · ask_price REAL (best
ask) · open/high/low REAL (session so far) · previous_close REAL (from
EOD cache) · source TEXT ('TWSE_MIS' | 'MOCK') · collected_at TEXT (local
wall clock) · run_id INTEGER.
PK (symbol, timestamp, source) → idempotent re-collection.

## intraday_1m_bars — derived, idempotent (INSERT OR REPLACE)
symbol · bar_time ("YYYY-MM-DD HH:MM") · open/high/low/close (snapshot
prices within the minute) · volume_delta (cumulative-volume diff, clipped
≥0) · amount_delta (NULL until the endpoint's value-traded field is
wired) · source · created_at.
PK (symbol, bar_time, source).

## collector_runs — one row per process run
run_id PK AUTOINCREMENT · started_at · ended_at · mode
('session'|'once'|'mock') · universe · n_symbols · interval_s · cycles ·
quotes_written · events · status ('running'→'completed'/'interrupted';
stale 'running' rows become 'aborted' on next startup).

## data_quality_events
event_id PK · run_id · symbol · event_time · event_type (see
data_quality_spec.md) · detail (≤300 chars) · source.

Indices: intraday_quotes(timestamp), data_quality_events(event_time).
Retention: unbounded for MVP (~1-2 MB/session/40 symbols at 60 s cadence;
years fit in single-digit GB); revisit if the full universe runs.
