# v15 Collector — Data Quality Spec

Checks run INLINE during collection (events land in
`data_quality_events`) plus a per-day report (`quality_report.py`).

| event_type | trigger | smoke-verified |
|---|---|---|
| MISSING_QUOTE | symbol absent from batch response / success=false | ✔ (mock) |
| NO_TRADE_TICK (v15.1, informational) | latest_trade_price='-' — NORMAL TWSE MIS behavior when no trade matched in the latest window (87.8% of day-1 snapshots). Counted per run as ONE summary event (symbol='*'); per-row granularity lives in intraday_quotes (price NULL, bid/ask live) so it never drowns true anomalies | ✔ (mock + day 1) |
| INVALID_PRICE | truly malformed: non-positive or unparseable-and-not-'-' | ✔ (mock) |
| STALE_QUOTE | exchange timestamp unchanged > 180 s during collection | ✔ (day 1: closing-auction freeze 13:25-13:30, expected) |
| CUMVOL_DECREASE | cumulative volume decreases (feed glitch) | ✔ (mock + day 1) |
| VOLUME_JUMP | tick volume z-score > 6 vs session history (≥10 obs) | ✔ (mock + day 1) |
| FETCH_ERROR | request/HTTP failure for a chunk | wraps every request |
| STORE_ERROR | unexpected per-symbol storage failure | wraps every insert |

## v15.1 bar price basis

1m bars carry `price_basis`: **TRADE_PRICE** (all constituent snapshots had
trade prints) · **MIDQUOTE_FALLBACK** (all bid/ask-mid) · **MIXED**.
Midquote bars are STATE PROXIES, not execution prices — any decision-grade
use requires explicit spread/slippage assumptions; never treat the mid as
a fill.

## v15.1 cadence statistics

The collector paces on target times (next_target += interval; sleep the
remainder), absorbing request/throttle time — day-1's 66.8 s drift at a
nominal 60 s is fixed. Measured cadence (mean/median/p95/min/max cycle
seconds) is stored per run in `collector_runs.cadence_json` and surfaced
in the daily quality report.

## Daily quality report (per symbol × source)

snapshot count · distinct minutes · coverage vs expected cycles (270 for
a full 09:00–13:30 session at 60 s) · largest implied gap · event counts
by type · derived bar count · list of symbols below 90% coverage.
Session-completeness is THE health metric — weeks of ≥95% coverage is a
prerequisite for any research use (see v14 acquisition plan).

## Explicit non-guarantees

Snapshots are ~60 s samples, not ticks: intraminute path, true traded
value (amount), odd-lot and auction dynamics are not captured; bid/ask
are top-of-book at sample instants. These limits are inherited by every
derived bar and must be restated in any research that consumes this data.
