# v16 Stage A — Test Summary (Task 16)

Date: 2026-08-18 · Suites: tests/test_holdings.py (new, unit) +
tests/test_user_holdings_overlay.py (7 pre-existing + 9 new end-to-end).
Result: **46/46 OK** (~2.1s), compileall clean. The 7 pre-existing
overlay tests pass UNCHANGED — long-only legacy behavior is numerically
identical (gross_long == gross_exposure when no shorts exist).

## Matrix coverage → test

| # | Case | Test | Result |
|---|---|---|---|
| 1 | legacy positive shares | test_legacy_positive | LONG, qty as-is |
| 2 | legacy negative shares | test_legacy_negative + TestStageA.test_a | SHORT, qty=abs, warned; never negative downstream |
| 3 | new LONG schema | test_new_schema | ok |
| 4 | new SHORT schema | test_new_schema / test_c | ok |
| 5 | invalid side | test_invalid_side + test_f | HoldingsError, rc 2, nothing written |
| 6 | negative shares + explicit side | test_side_sign_contradiction + test_f | hard error, not repaired |
| 7 | zero shares | test_zero_shares + test_g | warned, dropped |
| 8 | duplicate LONG lots | test_duplicate_long_lots + test_d | aggregated, qty-weighted avg cost |
| 9 | duplicate SHORT lots | test_duplicate_short_lots | aggregated |
| 10 | LONG+SHORT same symbol | test_long_short_conflict + test_e | NOT netted; POSITION_CONFLICT_REVIEW on both rows |
| 11 | no position + BUY | test_none_buy | OPEN_LONG_NEW_SIGNAL [HIGH] |
| 12 | no position + HOLD | test_none_hold | OPEN_LONG_EXISTING_TARGET [MEDIUM] — priority strictly below fresh BUY |
| 13 | no position + SELL | test_none_sell | NO_ACTION ("NOT a short") |
| 14 | LONG + HOLD underweight | test_long_hold_underweight | ADD_LONG (8% target, 1% actual) |
| 15 | LONG + HOLD aligned | test_long_hold_aligned | HOLD_LONG (8% vs 7.5%) |
| 16 | LONG + HOLD overweight | test_long_hold_overweight | REDUCE_LONG (8% vs 12%) |
| 17 | LONG + REDUCE | test_long_reduce | compares vs NEW target |
| 18 | LONG + SELL | test_long_sell | EXIT_LONG [HIGH] |
| 19 | SHORT + BUY | test_short_vs_buy_hold + test_c | BUY_TO_COVER [HIGH] |
| 20 | SHORT + HOLD | test_short_vs_buy_hold + test_a | BUY_TO_COVER [HIGH] |
| 21 | SHORT + WATCH | test_short_vs_watch | BUY_TO_COVER [MEDIUM] |
| 22 | SHORT + SELL | test_short_vs_sell_rank_based | rank-based HOLD_SHORT/REDUCE_SHORT/BUY_TO_COVER — never auto-HOLD_SHORT from SELL |
| 23 | outside-universe LONG | test_outside_universe | NO_MODEL_OPINION |
| 24 | outside-universe SHORT | test_outside_universe + test_h | NO_MODEL_OPINION |
| 25 | ETF outside universe | test_h (0050) | NO_MODEL_OPINION |
| 26 | mixed gross exposure | test_exposure_metrics | gross/net documented values |

## Explicit denominator regressions (the audit's corruption findings)

- **Negative legacy shares cannot make the denominator negative or
  inflate other weights**: TestStageA.test_a asserts 2330's weight is
  50,000/60,000 (gross), not the old 50,000/40,000 (net) = 125%; all
  weights ≤ 1.0.
- **All-short portfolio cannot silently NaN-blank every weight**:
  test_b asserts weights are computed (old code: total ≤ 0 → all NaN,
  report rendered as if fine).
- **NaN weights do not silently propagate**: test_a asserts no NaN in
  my_current_weight for valued rows.
- **Duplicate-lot sign flip**: test_d asserts the combined 1303 position
  (16.7% vs 12% target) reads OVERWEIGHT/REDUCE_LONG — per-lot the old
  code reported UNDERWEIGHT twice.
- **No generic HOLD/SELL user_action can ever be emitted**: test_i.

## Also verified

- Real smoke on the live repo (`--strategy blend50_band10`, actual
  my_holdings.csv, read-only): 2 positions, 0 short, outputs written to
  gitignored reports/user_holdings/ — legacy file accepted unchanged.
- Validation failures exit rc 2 with a clear message and write nothing.
