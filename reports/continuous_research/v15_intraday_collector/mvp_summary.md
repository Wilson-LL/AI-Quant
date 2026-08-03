# v15 Intraday Collector MVP — Summary

Built and smoke-tested 2026-08-03, branch
`research/v15-intraday-data-collector`. Data collection only — no orders,
no broker APIs, no model training, no production changes.

## Delivered

- `research/intraday_collector/collect_realtime_quotes.py` — batched
  TWSE-MIS polling (twstock), book/full universes, 60 s cadence with
  session gating (08:55–13:35 TW weekdays), inline quality events,
  SQLite WAL storage, restart-safe run ledger, per-symbol fault isolation,
  `--once` and `--mock` test modes.
- `build_1m_bars.py` (idempotent minute bars from cum-volume deltas) ·
  `quality_report.py` (per-day coverage/events md) · `status.py`
  (runs/coverage/events overview).
- Design docs: collector_design.md, database_schema.md,
  data_quality_spec.md. Gitignore hardened (`*.sqlite*`, quality/, logs/).

## Smoke evidence (no session loop was run)

- **Mock pipeline (5 synthetic cycles, 4 symbols + 1 always-missing):**
  20 quotes → 15 bars → quality report; injected anomalies detected
  (5× MISSING_QUOTE, 1× INVALID_PRICE zero-price). Stale/volume-jump
  detectors have correct logic but need real session durations/history to
  fire — first real session will exercise them.
- **Single real snapshot (`--once`, market closed):** 32/32 decision-book
  symbols collected in 4.7 s (2 chunked requests), 0 errors, closing-state
  fields all parsed (price/tick/cum-vol/bid/ask/OHL). MOCK and TWSE_MIS
  data verified separated by the source column.

## Status and next decision

The collector is ready but **NOT scheduled** — starting market-hour
collection needs explicit user approval. Suggested MVP operation once
approved: run `collect_realtime_quotes.py --universe book` during
sessions (manually or via a scheduled task), then
`build_1m_bars.py` + `quality_report.py` after close alongside daily ops;
review coverage for the first week before trusting the pipeline. The
6-month decision-grade clock (v14 plan) starts on the first full
collection day.
