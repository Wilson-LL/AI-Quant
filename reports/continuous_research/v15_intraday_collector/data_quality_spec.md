# v15 Collector — Data Quality Spec

Checks run INLINE during collection (events land in
`data_quality_events`) plus a per-day report (`quality_report.py`).

| event_type | trigger | smoke-verified |
|---|---|---|
| MISSING_QUOTE | symbol absent from batch response / success=false | ✔ (mock, 5×) |
| INVALID_PRICE | non-numeric / "-" / ≤0 latest price (row still stored with NULL price so gaps are analyzable) | ✔ (mock) |
| STALE_QUOTE | exchange timestamp unchanged > 180 s during collection | logic in place; needs real session duration to fire (mock runs are sub-second) |
| CUMVOL_DECREASE | cumulative volume decreases (feed glitch) | logic in place |
| VOLUME_JUMP | tick volume z-score > 6 vs session history (≥10 obs) | logic in place; needs ≥10 real cycles |
| FETCH_ERROR | request/HTTP failure for a chunk | wraps every request |
| STORE_ERROR | unexpected per-symbol storage failure | wraps every insert |

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
