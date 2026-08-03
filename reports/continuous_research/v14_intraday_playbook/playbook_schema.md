# v14 Playbook Schema (Task 2)

A playbook is generated AFTER the previous close / BEFORE open from the
latest EOD decision book. It is a condition table, never orders. Every row
carries `live_trading_allowed=false`; with only EOD data every row carries
`data_quality=DAILY_BAR_PROXY_ONLY`.

## Row schema (CSV columns)

date · symbol · eod_action · eod_target_weight · eod_rank · eod_score ·
direction_bias (long / short_diagnostic / avoid) · opening_gap_bin ·
checkpoint · condition (human-readable trigger) · suggested_action_label ·
confidence (validated / heuristic / none) · risk_level (low/med/high) ·
reason · data_source · data_quality · live_trading_allowed (always false)

## Scenario dimensions

- **Gap bins (10):** ≤−5, (−5,−3], (−3,−2], (−2,−1], (−1,0], (0,1],
  (1,2], (2,3], (3,5], >5 (% vs previous close).
- **Checkpoints:** open · open+15m · open+30m · open+60m · midday ·
  final-30m · pre-close exit. TWSE session 09:00–13:30 (no repo calendar
  encodes intraday hours — flagged). **With EOD data only, ONLY the `open`
  and `pre-close` checkpoints are backed by any data; all intermediate
  checkpoint rows are emitted as FRAMEWORK placeholders with
  confidence=none and reason="pending intraday collector".**
- **Price/volume/market state fields** (return-from-open, VWAP distance,
  same-time volume, index/sector state): schema-reserved, unpopulatable
  until the collector exists.
- **EOD context:** action/weight/rank/score from the blend50_band10 book;
  D1.2/tf/blend scores where available; holdings-overlay mismatch is
  consumed only locally and NEVER written into committed outputs.
- **Short context:** shortability is assumed FALSE unless data exists;
  short rows are `short_diagnostic` only, gated by the v12 proxy flags
  (low-ADV / +30% 20d rally / price floor).

## Action vocabulary

SKIP · WATCH_ONLY · ENTER_LONG_CONDITIONAL · EXIT_LONG · REDUCE_LONG ·
ENTER_SHORT_CONDITIONAL · COVER_SHORT · AVOID_SHORT · FORCE_EXIT_EOD ·
NO_ACTION. "Conditional" labels mean "a condition worth human review", not
an instruction to trade.

## Confidence semantics

- `validated`: the gap×action cell passed the daily-bar proxy rule search
  (train→val→OOS) — still NON-decision-grade (path-blind), but
  empirically grounded in open-to-close space.
- `heuristic`: conservative default (e.g. avoid-chase on >+5% gaps) with
  no cell-level validation.
- `none`: framework placeholder awaiting intraday data.
