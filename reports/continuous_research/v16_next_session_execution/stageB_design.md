# v16 Stage B — Design Record (Tasks 8/9/13/15)

Date: 2026-08-18 · Branch research/v16-next-session-execution-advisor.
Stage B builds the nightly human-facing layer on the accepted foundation
(timing gate f6d629b NEXT_OPEN_TIMING_VALIDATED; Stage A 91765cc
position-aware semantics). Nothing in Stage B modifies the model, the
decision book, daily_ops.bat, or the intraday collector; nothing places
orders or opens shorts.

## Architecture

```text
MODEL DECISION (blend50_band10 book — read-only, semantics preserved)
        ↓
ACTUAL HOLDINGS (my_holdings.csv → research/holdings.py normalization)
        ↓
USER ACTION (Stage-A map_user_action — model HOLD/SELL never shown raw)
        ↓
NIGHT-BEFORE PRICE BAND (research/execution_price_bands.py — pre-registered
        empirical conditional quantiles + ATR guardrails, expanding window)
        ↓
NEXT-SESSION ACTION PLAN (research/user_next_session_plan.py →
        reports/user_actions/<date>_next_session_action_plan.{csv,md},
        gitignored; + latest_* copies)
        ↓
MANUAL REVIEW (no live refresh yet — Stage C; no automation, no orders)
```

## Key decisions

- **Report population** = actual positions ∪ decision-book rows (BUY /
  HOLD / REDUCE / SELL / WATCH). Unselected universe names are excluded
  (no clutter); model SELL on an unowned name becomes NO_ACTION and is
  kept out of the high-priority section.
- **Freshness semantics** (`signal_freshness`): FRESH_ENTRY (model BUY,
  unowned) vs EXISTING_MODEL_POSITION (model HOLD/REDUCE, unowned —
  reported as OPEN_LONG_EXISTING_TARGET with
  `model_position_age_sessions` counted from dated paper books) vs
  CURRENT_USER_POSITION / WATCH_ONLY / NO_MODEL_OPINION. Existing
  targets get deliberately less aggressive entry bands (one quantile
  notch lower everywhere; invariant asserted in code and tests).
- **Metadata block** (derived in the plan layer — blended_decision_book
  untouched): signal_date, data_asof, generated_at,
  intended_execution_date (next weekday; TWSE holiday calendar does not
  exist in the repo — documented limitation), intended_execution_session
  = NEXT_TWSE_SESSION, timing_validation = NEXT_OPEN_TIMING_VALIDATED,
  book_age_sessions (cache trading dates newer than the book),
  book_stale + prominent report warning when stale. A missing holdings
  file produces a model-only plan flagged USER_POSITION_UNKNOWN.
- **Bands**: engine + methodology in price_band_methodology.md
  (pre-registered), validation in price_band_validation.md. Cells =
  rank bucket × cross-sectional vol tercile with hierarchical fallback
  (RANKxVOL→VOL→GLOBAL; MIN_CELL_OBS 400, MIN_POOL 750); every row
  carries band_sample_count + band_fallback_level. Historical rank
  buckets for calibration come from the frozen BR panel where covered.
- **execution_posture**: conditional night-time intent from the fixed
  action→posture map (methodology doc); live posture is Stage C.
- **Tick rounding**: single helper, TWSE ladder from public
  documentation, buy levels rounded down / sell levels up, explicitly
  flagged unverified against a live feed (also printed in every report
  header).
- **Report sections** (Task 15): 1 HIGH PRIORITY (real user items only)
  · 2 NEW LONG SIGNALS · 3 MODEL POSITIONS I DO NOT OWN · 4 ACTUAL LONG
  POSITIONS (hold/add; no manufactured prices for aligned HOLD_LONG) ·
  5 REDUCE/EXIT (SELL INTO STRENGTH vs RISK EXIT labelled) · 6 ACTUAL
  SHORT POSITIONS (cover/risk bands only, "NO VALIDATED OPEN-SHORT
  MODEL EXISTS" banner) · 7 WATCHLIST (every row has entry reference,
  zone, do-not-chase, invalidation) · 8 NO MODEL OPINION (no fabricated
  advice).

## Explicitly deferred (not in Stage B)

Task 3 (book metadata columns in blended_decision_book.py), Task 10/11
(morning live refresh + bat), Task 12 (daily_ops step 9), scheduling,
any broker/order functionality.
