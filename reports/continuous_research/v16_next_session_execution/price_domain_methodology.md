# v16 Stage B2 — Price Domain + Distribution + Range-Reach Methodology (pre-registered)

Written 2026-08-18 **before any B2 validation metric was computed**.
Everything below is frozen as of this file; B2 validation results may not
retroactively tune it. Any later correctness fix must be recorded as an
explicit amendment at the bottom. The B1 methodology and validation
results are preserved verbatim as `price_band_methodology_b1_baseline.md`
/ `price_band_validation_b1_baseline.{csv,md}` (SHA-256 prefixes at
snapshot: methodology EFCD2CC61625D355, validation csv CBC466DC2B528C20,
validation md 2CBE98A1485BC476, engine 37DB197C373A492F). B2 is ADDITIVE:
the B1 band quantile constants are NOT changed.

## Auction reference (frozen assumptions)

- Ordinary stock, previous close available → reference =
  previous_close, reference_source = PREVIOUS_CLOSE, confidence MEDIUM,
  status `NORMAL_DAY_ASSUMPTION` (never CONFIRMED — the repo cannot rule
  out ex-date/special adjustments; see twse_price_domain_rules.md).
- Missing/invalid previous close → status `UNKNOWN`, limits NA,
  confidence LOW. Special-day references are never inferred from price
  movement.
- Non-stock instrument types (ETF etc., per SECTOR_MAP type) → status
  `UNKNOWN` (ladder/limit unverified), limits NA. NO_MODEL_OPINION rows
  never receive model-derived bands regardless.

## Legal domain (frozen)

raw_limit_up = ref × 1.10; raw_limit_down = ref × 0.90.
legal_limit_up = greatest legal tick ≤ raw_limit_up; legal_limit_down =
smallest legal tick ≥ raw_limit_down (direction-aware projection onto
the ordinary-stock ladder <10:0.01 / 10–50:0.05 / 50–100:0.10 /
100–500:0.50 / 500–1000:1.00 / ≥1000:5.00). Regression anchor: ref
40.60 → 44.65 / 36.55.

**Hard constraint:** when a legal domain is known, every emitted price
level is clamped into [legal_limit_down, legal_limit_up] and then
projected to a legal tick. A violation raises
`PriceDomainValidationError` — never a silent bad number. Validation
gate: ZERO domain violations across all emitted prices.

## Price rounding directions (frozen; same as B1 where B1 defined them)

- BUY-intent levels: round DOWN to the legal tick.
- SELL/risk-intent levels: round UP.
- Distribution display prices (expected_open_*, expected_low_*,
  expected_high_*): round to the NEAREST legal tick, ties down, then
  clamp to the legal domain when known.
Internal statistics stay continuous; only user-facing prices are
projected onto the grid.

## Normalized coordinates (frozen definitions)

relative_to_reference = price/ref − 1;
distance_to_limit_up_pct = legal_limit_up/price − 1;
distance_to_limit_down_pct = price/legal_limit_down − 1;
legal_domain_position = (price − legal_limit_down) /
(legal_limit_up − legal_limit_down) — execution context ONLY, never a
buy/sell signal by itself.

## Conditional distributions (frozen)

Same conditioning and fallback as B1 (unchanged deliberately):
cells = rank_bucket (TOP≤20% / MID≤50% / REST) × cross-sectional vol20
tercile; fallback RANKxVOL → VOL → GLOBAL; MIN_CELL_OBS 400, MIN_POOL
750; band_sample_count + band_fallback_level recorded on every row.
Calibration windows: strictly < T for validation; ≤ T for the nightly
plan. Quantiles reported: open gap p10/p25/p50/p75/p90; low p10/p25/
p50/p75; high p25/p50/p75/p90 — all relative to the auction reference
(= previous close under the normal-day assumption, so these coincide
with the B1 close-anchored distributions).

## Range-reach probability (frozen definition)

For a BUY level P: reach = share of calibration observations with
next_day_low ≤ P/ref − 1. For a SELL level P: reach = share with
next_day_high ≥ P/ref − 1. Estimated from the SAME expanding
calibration cell as the bands. This is a DAILY-RANGE statistic: it never
asserts order existence, queue priority, volume, zero slippage, or that
the level traded — every report states "range reach, not fill
probability". The non-reach complement is labeled
`T+1_RANGE_DID_NOT_REACH_LEVEL` (a manual user can still act later in
the 20-session horizon).

Reach-curve levels shown in the nightly md (3–5 only): ideal_zone_high,
buy reference, ideal_zone_low, and one conservative level = ref ×
(1 + g(0.10)) for entries; sell_reference, ideal_sell_zone_high, and
acceptable_sell_floor for sells. Additionally: p_open_above_do_not_chase
= share of calibration gaps > the chase gap; p_open_below_panic_level =
share of gaps < the panic gap. The full per-symbol tick-level curve goes
to gitignored reports/user_actions/<date>_price_reach_curve.csv, never
into the md and never committed. range_reach_data_quality = OK when the
cell cleared MIN_CELL_OBS at RANKxVOL or VOL level, DEGRADED for GLOBAL
fallback, NA when no distribution was available.

## Pre-registered validation metrics (computed by the B2 extension of
research/next_session_price_band_validation.py on the frozen BR panel,
expanding windows, same 1,703-observation role grid as B1)

1. **Legal domain** (price_domain_validation): zero-violation hard gate
   over every emitted price of every evaluated observation; legality of
   every tick; the 40.60 anchor; boundary cases 9.99/10, 49.95/50,
   99.9/100, 499.5/500, 999/1000 (unit tests).
2. **Open distribution coverage** (price_domain_validation): empirical
   share of actual next opens ≤ p10/p25/p50/p75/p90 (targets 10/25/50/
   75/90% within sampling error), median |p50 − open| in bps; by year,
   vol regime, and 2026 YTD separately.
3. **Range reach** (range_reach_validation): predicted reach probability
   vs realized frequency in prediction-decile buckets, Brier score,
   buy-side (low ≤ level) at the entry-band levels and sell-side
   (high ≥ level) at the sell-band levels; sample counts; by year, vol
   regime, 2026 YTD separately. No quantile/threshold retuning follows
   from these results.
4. **Do-not-chase outcome analysis** (do_not_chase_outcome_analysis):
   entrant/incumbent observations bucketed by actual next open vs the
   frozen bands (≤ ceiling / ceiling→chase / > chase); per bucket: n,
   mean/median 20-session open-to-open forward return, hit rate, and
   max adverse excursion measured as min(daily lows over the 20-session
   window) vs entry — a LEVEL statistic, no intraday ordering claimed.
   ANALYSIS ONLY: the chase threshold stays frozen regardless of result.
5. **Waiting tradeoff** (in range_reach_validation): for entry
   observations, reach vs non-reach frequencies at discounts of 0.0% /
   −0.5% / −1.0% / −1.5% / −2.0% vs the reference.

## Amendments

**A1 (2026-08-19, user-review patch — interpretation only, zero numeric
changes).**
1. *Do-not-chase reframed as an execution-quality threshold.* The
   do_not_chase_outcome_analysis found no 20-session outcome support
   for an alpha reading of the threshold. The numeric field
   `do_not_chase_above` and its frozen quantiles (fresh g0.90 /
   existing g0.75) are UNCHANGED; user-facing wording now presents it
   as "above preferred execution range" — an unusually expensive entry
   vs the historical next-open distribution — and explicitly states the
   validated 20-session signal was NOT shown to fail above it. No
   `execution_price_warning` column is emitted at night (there is no
   live price at 22:00 to compare; that label belongs to the Stage C
   live refresh).
2. *Range-reach calibration confidence layer* (new, descriptive only):
   `range_reach_confidence` ∈ HIGH / NORMAL / DEGRADED / INSUFFICIENT,
   computed per row at its primary level (buy reference / sell
   reference) from the row's own calibration sample ONLY (< T
   validation, ≤ T production — leakage-safe; no year or date
   special-casing): INSUFFICIENT = no sample; DEGRADED = GLOBAL
   fallback, or trailing-750-observation subsample < 200, or
   |recent-vs-full reach drift| > 10pp; HIGH = RANKxVOL fallback with
   drift ≤ 5pp; NORMAL otherwise. Constants frozen here (RECENT_N 750,
   RECENT_MIN 200, DRIFT_DEGRADED 0.10, DRIFT_HIGH 0.05). It never
   changes actions, selection, or thresholds; DEGRADED rows carry an
   inline lower-confidence caveat (the observed 2026 high-vol SELL-side
   under-realization is expected to surface through this rule, not
   through any hardcoded date).
3. *Waiting-tradeoff wording*: non-reach outcomes are labeled
   T+1_RANGE_DID_NOT_REACH_LEVEL and are never described as missed
   trades/orders/profit.
