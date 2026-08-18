# v16 Stage B — Test Summary

Date: 2026-08-18 · Full suite: **69/69 OK** (~32s; 23 new Stage B tests
+ the 46 Stage A/overlay tests, all still green). compileall clean on
all changed modules.

## New suites

tests/test_execution_price_bands.py (11 tests) — band engine:
- TWSE tick ladder + buy-down/sell-up rounding (NaN-safe).
- Outcome-column alignment (next_open_gap at row t = open[t+1]/close[t]−1).
- **No-future-leakage (matrix 18)**: absurd post-asof data injected into
  the history leaves strict-window quantiles bit-identical.
- **Fallback hierarchy + insufficient samples (13/14)**: RANKxVOL→VOL→
  GLOBAL→INSUFFICIENT, MIN_CELL_OBS/MIN_POOL enforced.
- **Fresh vs existing aggressiveness (15)**: existing ceiling/chase/
  reference ≤ fresh, same cell.
- **Extreme ±10% gap pools (16/17)**: ordering invariants and finiteness
  hold with limit-move tails.
- ATR guardrail floors the ideal-zone width.
- **No short creation (20, engine side)**: short_cover_bands emits only
  cover/risk keys; posture map contains no short-entry value.
- **Path-ambiguity guard (19)**: validation module contains no
  stop-before-target logic and documents daily OHLC as path-ambiguous.

tests/test_user_next_session_plan.py (12 tests) — plan layer, synthetic
fixture repo (300-session cache clears MIN_POOL):
- (1) fresh BUY unowned → OPEN_LONG_NEW_SIGNAL + complete ordered entry
  bands + sample count; (2) HOLD unowned → OPEN_LONG_EXISTING_TARGET,
  model_position_age_sessions ≥ 3 from book history, WAIT_FOR_PULLBACK;
- (3/4/5) HOLD + aligned/under/overweight LONG → HOLD_LONG / ADD_LONG /
  REDUCE_LONG; (6) SELL unowned → NO_ACTION and absent from the
  high-priority section; (7) SELL held → EXIT_LONG with full sell bands
  (urgent < panic ordering asserted); (8) WATCH → WATCH_LONG with a
  price answer in the report; (9) outside-universe LONG (0050) →
  NO_MODEL_OPINION, no bands, honest report text; (10) SHORT vs bullish
  BUY → BUY_TO_COVER + cover/risk bands + "NO VALIDATED OPEN-SHORT
  MODEL EXISTS" banner; (11) stale book → book_stale metadata + STALE
  BOOK warning; (12) Friday signal → Monday execution date; (13)
  missing holdings file → USER_POSITION_UNKNOWN model-only plan;
- (20) **no OPEN_SHORT/SELL_SHORT/WATCH_SHORT can ever be emitted**
  (legacy-negative, explicit-short, and no-holdings variants) + all
  actions within the Stage-A vocabulary (also asserted at runtime in
  build_plan).

## B27 sweep

Repo grep over changed executable code: OPEN_SHORT / SELL_SHORT /
WATCH_SHORT appear only in documentation strings and in the runtime
assertion that forbids them. No reachable code path emits them.

## Real-repo smoke (B26)

`python research/user_next_session_plan.py` against the actual
gitignored my_holdings.csv (legacy schema, unmodified): 34 rows on the
2026-08-17 book (fresh, stale=false), actions = 22
OPEN_LONG_EXISTING_TARGET + 10 WATCH_LONG + 1 REDUCE_LONG (held 2330,
covered-not-selected, top-half rank soft-reduce) + 1 NO_MODEL_OPINION
(0050 — no fabricated action). Outputs written only under gitignored
reports/user_actions/.

## OOS validation run

price_band_validation.csv/md: 1,703 role-observations, 68 rebalances,
expanding windows, 0 skipped — coverage matches the pre-registered
quantile design (see price_band_validation.md).
