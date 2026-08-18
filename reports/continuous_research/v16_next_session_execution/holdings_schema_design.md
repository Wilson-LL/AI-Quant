# v16 Task 7/14 — Holdings Schema Design (LONG/SHORT)

Date: 2026-08-18. Implemented in research/holdings.py (normalization) +
research/user_holdings_overlay.py (consumer). my_holdings.csv itself
remains gitignored personal data; my_holdings.example.csv carries the
reference schema.

## New schema (preferred, human-editable)

```csv
symbol,side,shares,avg_cost,current_price,current_value,account,notes
2330,LONG,200,2356.25,,,CTBC,core semi position
0050,LONG,1000,102.05,,,CTBC,ETF
2376,SHORT,1000,350.00,,,CTBC,short position
```

- `side` = LONG | SHORT (case-insensitive on read, normalized upper).
- `shares` = **positive absolute quantity only**.
- `current_value` = **absolute** market value (never signed) — sign
  ambiguity for shorts is resolved by `side`, not by negative numbers.

## Backward compatibility (legacy schema, no `side` column)

| Legacy input | Normalized to | Notes |
|---|---|---|
| shares > 0 | LONG, qty = shares | unchanged behavior |
| shares < 0 | SHORT, qty = abs(shares) | accepted with a warning recommending the explicit schema |
| shares == 0 | dropped | warning printed; not a position |
| unparseable | side UNKNOWN, qty NaN | flows to REVIEW_MANUALLY, excluded from all arithmetic |

Normalization happens immediately at load (`holdings.load_lots`);
**negative share counts can never reach portfolio arithmetic** — every
downstream quantity is positive-by-construction.

## Hard validation errors (never silently repaired → HoldingsError, rc 2)

- `side` column present with a value other than LONG/SHORT.
- `side` column present but blank on any row.
- `side` present with negative `shares` (side/sign contradiction) —
  the file must be fixed by the user.

## Internal representation (per aggregated position)

`position_side`, `position_qty` (positive), `market_value_abs`
(positive), `signed_exposure_value` (+ for LONG, − for SHORT),
qty-weighted `avg_cost`, `n_lots`, `both_sides` conflict flag.

## Duplicate symbols

Lots are preserved at load, then **aggregated per (symbol, side) before
any weight/action calculation**: total quantity, summed absolute value,
qty-weighted average cost (left blank with a warning if any lot lacks
avg_cost). Rationale: the pre-Stage-A code weighted each lot separately,
which produced sign-flipped over/underweight verdicts (two 5% lots vs an
8.6% target each read "underweight" although the combined 10% position
is overweight) — regression-tested now.

Same symbol with BOTH LONG and SHORT lots: **never netted**. Both rows
are kept, flagged `both_sides`, classified POSITION_CONFLICT, and
user_action = POSITION_CONFLICT_REVIEW (HIGH) with the lots documented
in the report.

## Portfolio denominators (the corruption fix)

```text
gross_long_value   = Σ market_value_abs over LONG positions
gross_short_value  = Σ market_value_abs over SHORT positions
gross_exposure     = gross_long_value + gross_short_value
net_exposure       = gross_long_value − gross_short_value
```

- `my_current_weight` (informational, all rows) = value / gross_exposure.
  Gross is non-negative by construction, so a short can never shrink the
  denominator, inflate other weights, or flip the total ≤ 0 (the old
  failure mode where every weight silently became NaN).
- `my_long_cmp_weight` (used for model-target comparison) =
  value / **gross_long_value**, LONG rows only. Chosen because model
  targets are weights of a long-only book summing to 1 — the comparable
  user quantity is "fraction of my long book", and a SHORT position
  mathematically cannot distort it. SHORT rows carry NaN here and NaN
  `weight_gap` (a short is never an "underweight long").
- With a long-only holdings file, gross_long == gross_exposure and both
  weights coincide exactly with the pre-Stage-A numbers (continuity
  verified by the untouched legacy tests).

## Output columns added to the overlay CSV

position_side, position_qty, market_value_abs, signed_exposure_value,
my_long_cmp_weight, user_action, user_action_priority,
user_action_reason, account (previously read but dropped — now emitted).
Legacy aliases my_shares (= position_qty) and my_current_value
(= market_value_abs) are kept for continuity.
