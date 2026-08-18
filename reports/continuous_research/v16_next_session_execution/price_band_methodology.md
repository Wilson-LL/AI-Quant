# v16 Stage B — Night-Before Price-Band Methodology (pre-registered)

Written 2026-08-18 **before any validation metric was computed**. The
formulas, quantiles, guardrail constants, fallback hierarchy, and
validation metrics below are fixed as of this file; validation results
follow in price_band_validation.md and may NOT retroactively tune them.

Purpose: at 22:00 on trading day T, tomorrow's open is unknown. Bands are
conditional research estimates of reasonable execution regions for the
NEXT session, built ONLY from information available at T's close. They
describe execution quality for an already-validated signal
(NEXT_OPEN_TIMING_VALIDATED, f6d629b); they never re-select stocks and
never turn a model long into a bearish signal.

## Inputs (all known at T close)

previous_close · ATR20 (mean of 20 trailing true ranges) · ATR20_pct =
ATR20 / close · vol20/vol60 (realized, log returns) · cross-sectional
volatility tercile on date T · model rank bucket · user_action family ·
signal_freshness. Forbidden as inputs: anything from T+1 onward (outcome
variables only during validation).

## Empirical outcome variables (per symbol-day, from the EOD cache)

```text
next_open_gap        = open(T+1)/close(T) − 1
next_high_from_close = high(T+1)/close(T) − 1
next_low_from_close  = low(T+1)/close(T) − 1
next_close_from_close= close(T+1)/close(T) − 1
```

## Conditioning cells (deliberately low-dimensional)

Cell key = (rank_bucket, vol_bucket):
- rank_bucket: TOP (model rank ≤ 20% of universe — book candidates),
  MID (20–50%), REST (> 50%); from the decision book / predictions
  (production) or the frozen walk-forward panel scores (validation).
- vol_bucket: cross-sectional tercile of vol20 on date T (LOW/MED/HIGH).

Fallback hierarchy when a cell has < MIN_CELL_OBS = 400 observations:
rank×vol → vol only → global. (No sector level: with a ~108-name
universe the sector cells cannot reliably clear the minimum; recorded as
a deliberate omission.) Every output row records band_sample_count and
band_fallback_level (RANKxVOL / VOL / GLOBAL).

## Calibration windows — no future leakage

Validation: for a signal dated T, quantiles are computed from
observations with date STRICTLY EARLIER than T (expanding window,
minimum 750 pooled observations to emit a band at all).
Production (nightly plan): the full history up to and including T is the
expanding window — at generation time nothing later exists.

## Band formulas (quantiles are of the conditional cell distribution)

Let P = previous_close, g(q) = q-quantile of next_open_gap,
h(q) = q-quantile of next_high_from_close, l(q) = q-quantile of
next_low_from_close, A = ATR20_pct.

**Long entry — FRESH (OPEN_LONG_NEW_SIGNAL):**
```text
reference            = P × (1 + g(0.50))          # ~conditional median next open
ideal_zone           = P × (1 + g(0.25)) … P × (1 + g(0.60))
acceptable_ceiling   = P × (1 + g(0.75))
do_not_chase_above   = P × (1 + g(0.90))
risk_review_below    = P × (1 + min(l(0.10), −K_RISK × A))
```

**Long entry — EXISTING TARGET (OPEN_LONG_EXISTING_TARGET) and ADD_LONG:**
deliberately less aggressive than a fresh signal (the model has held it;
no freshness urgency): shift every quantile down one notch —
```text
reference            = P × (1 + g(0.40))
ideal_zone           = P × (1 + g(0.20)) … P × (1 + g(0.50))
acceptable_ceiling   = P × (1 + g(0.60))
do_not_chase_above   = P × (1 + g(0.75))
risk_review_below    = same as fresh
```
Invariant (tested): the existing-target ceiling and chase levels are ≤
the fresh-signal ones for the same cell.

**Sell side (EXIT_LONG / REDUCE_LONG)** — empirical upside quantiles, not
a blind numeric mirror (gap distributions are asymmetric):
```text
sell_reference           = P × (1 + max(g(0.50), 0))       # sell into strength at/above median open
ideal_sell_zone          = P × (1 + g(0.50)) … P × (1 + h(0.60))
acceptable_sell_floor    = P × (1 + g(0.25))
do_not_panic_sell_below  = P × (1 + max(l(0.10), −K_PANIC × A))
urgent_risk_review_below = P × (1 + min(l(0.05), −K_RISK × A))
```
REDUCE_LONG (sell into strength) and EXIT_LONG (risk exit) share bands;
the report labels the intent differently and EXIT rows carry the urgent
level prominently.

**HOLD_LONG (aligned) — no manufactured trade prices:**
```text
no_action_zone = P × (1 − K_HOLD × A) … P × (1 + K_HOLD × A)
review_below   = P × (1 + min(l(0.05), −K_RISK × A))
review_above   = P × (1 + h(0.95))
```

**Actual shorts (HOLD_SHORT / REDUCE_SHORT / BUY_TO_COVER)** — cover/risk
bands only, NEVER short-entry bands:
```text
cover_reference    = existing-target buy reference (they are buyers)
cover_zone         = existing-target ideal zone
risk_review_above  = P × (1 + max(h(0.90), +K_RISK × A))   # short loses as price rises
```

## Volatility guardrails (single global constant set, not tuned per cell)

```text
K_WIDTH = 0.25   # ideal-zone half-width floor: max(empirical, K_WIDTH × A)
K_RISK  = 1.50   # risk-review distance floor in ATR
K_PANIC = 1.00   # panic-sell distance floor in ATR
K_HOLD  = 1.00   # hold no-action half-width in ATR
```
Chosen a priori from conventional ATR practice; NOT optimized. Any
future change requires re-running the full validation.

## Price rounding

One centralized helper rounds every displayed price to the standard TWSE
price-tick ladder (<10: 0.01 · 10–50: 0.05 · 50–100: 0.10 · 100–500:
0.50 · 500–1000: 1.00 · ≥1000: 5.00), buy-side levels rounded down,
sell-side levels rounded up. **Caveat (also printed in the report): the
tick ladder is implemented from public TWSE documentation and has not
been verified against a live feed in this repo — verify before
operational reliance.** No other exchange rules are assumed.

## execution_posture (night-time = conditional intent, not a live call)

| user_action | posture |
|---|---|
| OPEN_LONG_NEW_SIGNAL | EXECUTE_IN_IDEAL_ZONE (if it opens in/below the ideal zone; otherwise degrade to WAIT_FOR_PULLBACK / DO_NOT_CHASE by the printed levels) |
| OPEN_LONG_EXISTING_TARGET | WAIT_FOR_PULLBACK |
| ADD_LONG | EXECUTE_WITHIN_LIMIT |
| HOLD_LONG | NO_ACTION |
| REDUCE_LONG | SELL_IF_REBOUND |
| EXIT_LONG | SELL_IN_IDEAL_ZONE |
| BUY_TO_COVER / REDUCE_SHORT / HOLD_SHORT | RISK_REVIEW |
| WATCH_LONG | WAIT_FOR_PULLBACK |
| NO_ACTION | NO_ACTION |
| NO_MODEL_OPINION | NO_MODEL_OPINION |
| POSITION_CONFLICT_REVIEW | POSITION_CONFLICT |

## Pre-registered validation metrics (computed by
research/next_session_price_band_validation.py on the frozen BR panel
2021→, expanding windows, rebalance-grid signals)

For entry bands (fresh + existing variants, per year and per vol bucket):
% of next opens below ideal / inside ideal / ideal-to-ceiling / above
ceiling / above do-not-chase; median |reference − next open| (bps); q75
and q90 of the absolute pricing error; conditional next-day high/low
after inside-zone opens vs above-chase opens. For sell bands: % opens
inside ideal sell zone / above / below floor / below panic. Sample
counts everywhere. **Explicitly NOT evaluated: "stop hit before target"
sequencing — daily OHLC is path-ambiguous and intraday ordering is
unknowable (v14 finding); nothing in the validation interprets it.**

Success is descriptive (coverage matches the quantile design within
sampling error, stable by year/regime), not a promise of fills; no
profitability claim is attached to bands.
