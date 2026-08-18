# v16 Stage B2 — Test Summary

> **Review patch 2026-08-19** (interpretation only, amendment A1): the
> chase threshold is presented as "above preferred execution range"
> (execution-quality, not alpha-validity — wording tested); a
> leakage-safe `range_reach_confidence` layer (HIGH/NORMAL/DEGRADED/
> INSUFFICIENT, trailing-drift rule, no date special-casing) annotates
> reach percentages without changing any action or threshold; waiting
> tradeoffs are worded as range reach / T+1_RANGE_DID_NOT_REACH_LEVEL.
> Suite grew to 100/100 (7 patch tests: frozen-constant assertions,
> drift-rule classification incl. GLOBAL/INSUFFICIENT/small-recent,
> leakage safety, AST-level no-date-special-casing, no-missed-trade
> wording, execution-quality md wording, forced-DEGRADED caveat
> rendering).

Date: 2026-08-18 · Full suite: **91/91 OK** (~14s) = 69 Stage A + B1
tests (all still green, none superseded) + 22 new B2 tests. compileall
clean across research/ and tests/.

## B1 baseline preservation (Task 0)

Snapshot taken BEFORE any B2 edit: 69/69 rerun OK; SHA-256 prefixes
recorded (price_band_methodology.md EFCD2CC61625D355,
price_band_validation.csv CBC466DC2B528C20, price_band_validation.md
2CBE98A1485BC476, execution_price_bands.py 37DB197C373A492F); baseline
copies committed-to-be as price_band_methodology_b1_baseline.md and
price_band_validation_b1_baseline.{csv,md}. B1 quantile constants
unchanged in B2 (additive design).

## New suites

tests/test_twse_price_domain.py (11) — official 40.60 anchor
(44.65/36.55); tick boundaries 9.99/10, 49.95/50, 99.9/100, 499.5/500,
999/1000; floor/ceil never produce an illegal tick (500 random + edge
prices); limits never exceed ±10%; missing/invalid reference → UNKNOWN;
ETF → UNKNOWN with no fabricated limits; status never
CONFIRMED_STANDARD_LIMIT; clamp near limit-up/limit-down;
PriceDomainValidationError on out-of-domain and off-tick prices;
unknown-domain still enforces tick legality; legal_domain_position.

tests/test_execution_price_bands.py +6 (B2) — BUY reach definition
(next-day LOW ≤ level, hand-computed), SELL reach definition (next-day
HIGH ≥ level), open-beyond probabilities, NaN safety, cell_full
expanding-window leakage (corrupted future lows leave samples
bit-identical), no-fill-probability wording gate on the plan source.

tests/test_user_next_session_plan.py +5 (B2) — hard gate: every emitted
band and distribution price on NORMAL_DAY_ASSUMPTION rows is a legal
tick inside [legal_limit_down, legal_limit_up]; domain columns (stock →
NORMAL_DAY_ASSUMPTION/PREVIOUS_CLOSE, ETF → UNKNOWN + NaN limits + no
fabricated distributions); expected-open quantiles ordered p10≤…≤p90;
reach probabilities in [0,1] for buy and sell rows + reach-curve CSV
written with BUY and SELL rows; report wording (NOT a fill probability,
legal range, header disclaimer).

## Validation runs (pre-registered metrics, computed after freezing)

- Legal-domain hard gate: **0 violations** over 1,703 observations.
- Open-distribution coverage: 12.3/28.4/52.4/74.9/90.8% at nominal
  10/25/50/75/90; 2026-YTD miscalibration flagged, not tuned.
- Range-reach calibration: BUY Brier 0.146 (pred 75.0% vs realized
  79.8%, conservative), SELL Brier 0.164 (70.4% vs 70.3%); monotone
  deciles; SELL-2026 weak spot flagged.
- Do-not-chase outcome analysis: no 20-session outcome support for the
  chase threshold (honest negative result; threshold stays frozen —
  documented as execution-quality discipline, not alpha protection).
- Waiting tradeoff: reach 86/72/60/48/38% at 0/−0.5/−1/−1.5/−2%
  discounts (realized), with non-reach labeled
  T+1_RANGE_DID_NOT_REACH_LEVEL.

## Safety gates (Task 32)

No reachable OPEN_SHORT/SELL_SHORT/WATCH_SHORT (grep: docstrings +
forbidding assertion only); no broker/network imports in the Stage B
modules (grep clean); daily_ops.bat, blended_decision_book.py, training/
inference code, checkpoints, and the intraday collector untouched (git
status: only .gitignore modified among tracked non-report files); raw
DB, my_holdings.csv, and reports/user_actions/ all gitignored and
unstaged.

## Real-holdings smoke (Task 31)

Same action profile as B1 (22 OPEN_LONG_EXISTING_TARGET, 10 WATCH_LONG,
1 REDUCE_LONG on the held 2330, 1 NO_MODEL_OPINION on 0050 — preserved,
no fabricated bands), now with auction-reference/legal-range/expected-
distribution/reach content on every actionable row (100 B2 content
lines in the generated md) plus the per-symbol reach-curve CSV. All
outputs under gitignored reports/user_actions/.
