# v16 Task 5/14 — User Action Mapping (final)

Date: 2026-08-18. Implemented in research/holdings.py::map_user_action;
wired into the overlay as columns user_action / user_action_priority /
user_action_reason. `model_action` is preserved untouched beside it.
The bare labels HOLD and SELL can never appear as a user_action
(regression-tested); OPEN_SHORT and WATCH_SHORT are deliberately not in
the vocabulary — no validated short-side model exists (v12: rejected 6/6).

## Vocabulary

OPEN_LONG_NEW_SIGNAL · OPEN_LONG_EXISTING_TARGET · ADD_LONG · HOLD_LONG ·
REDUCE_LONG · EXIT_LONG · WATCH_LONG · WATCH_NEUTRAL (reserved) ·
HOLD_SHORT · REDUCE_SHORT · BUY_TO_COVER · NO_ACTION · NO_MODEL_OPINION ·
POSITION_CONFLICT_REVIEW

## Precedence

1. LONG+SHORT lots on the same symbol → POSITION_CONFLICT_REVIEW [HIGH].
2. Outside the scored universe (or no data cache) → NO_MODEL_OPINION
   [HIGH if the position is material (> large-gap share of gross), else
   INFO] — regardless of side. Never fabricated advice for e.g. 0050.
3. Otherwise the tables below.

## No actual position (rows for these appear in Stage B's plan; the
   mapping is implemented and unit-tested now)

| Model state | user_action | Priority | Why |
|---|---|---|---|
| BUY | OPEN_LONG_NEW_SIGNAL | HIGH | fresh model entry — highest freshness |
| HOLD, target > 0 | OPEN_LONG_EXISTING_TARGET | MEDIUM | standing target the user never entered — deliberately NOT "HOLD", and deliberately less fresh than BUY |
| REDUCE, target ≥ 4% | OPEN_LONG_EXISTING_TARGET | LOW | meaningful residual target (≥ half a typical 1/22 book slot) |
| REDUCE, target < 4% | WATCH_LONG | LOW | trimmed to a residual — watch only |
| WATCH | WATCH_LONG | LOW | bullish near-miss |
| SELL | NO_ACTION | INFO | model exits its paper long; user holds nothing; **never SELL_SHORT** |
| in universe, no book row | NO_ACTION | INFO | not targeted, not held |

## Actual LONG position

| Model state | user_action | Priority |
|---|---|---|
| SELL | EXIT_LONG | HIGH |
| in book, target > 0, gap < −band | ADD_LONG | MEDIUM |
| in book, target > 0, within ±band | HOLD_LONG | LOW |
| in book, target > 0, gap > band | REDUCE_LONG | MEDIUM (HIGH if gap > 2×band) |
| in book, target > 0, weight not computable | HOLD_LONG | INFO ("comparison unavailable") |
| WATCH (target 0) | HOLD_LONG | MEDIUM — hold under watch |
| unselected, non-WATCH, rank top-half | REDUCE_LONG | MEDIUM (soft reduce) |
| unselected, non-WATCH, rank bottom-half | EXIT_LONG | HIGH |
| unselected, rank unavailable | REDUCE_LONG | MEDIUM (conservative middle) |

REDUCE with target > 0 compares against the NEW (post-trim) target — the
model action itself does not force a user reduce if the user is already
at or below the new target.

## Actual SHORT position (tracking only — no new shorts ever suggested)

| Model state | user_action | Priority |
|---|---|---|
| in book, target > 0 (BUY/HOLD/REDUCE) | BUY_TO_COVER | HIGH — model is bullish against the short |
| WATCH | BUY_TO_COVER | MEDIUM — conflict review |
| SELL or unselected, rank ≤ 30% | BUY_TO_COVER | MEDIUM |
| SELL or unselected, rank 30–50% | REDUCE_SHORT | MEDIUM |
| SELL or unselected, rank > 50% | HOLD_SHORT | LOW |
| rank unavailable | REDUCE_SHORT | MEDIUM |

Every short-row reason carries the disclaimer: SELL is a long-book exit
and is **never treated as proof of bearish alpha**; classification is
risk/conflict-based only. (rank = tf-score percentile over the scored
universe from the latest predictions file, 0 = best; the book's blend
rank governs wherever a book row exists.)

## Alignment tolerance (documented, configurable)

```text
band = max(0.25 × target, 2pp)      # ALIGN_REL = 0.25, ALIGN_FLOOR = 2pp
underweight  : actual − target < −band   → ADD_LONG
aligned      : |actual − target| ≤ band  → HOLD_LONG
overweight   : actual − target > +band   → REDUCE_LONG
```

Rationale: the 2pp floor equals the overlay's long-standing medium_gap
default, below which a difference is operationally meaningless noise for
manual execution; the 25% proportional term scales the corridor with the
target so an 8% target tolerates 6–10% (realistic manual tracking) and a
4% target tolerates 3–5%. Fixed before inspecting any individual
holding; override via holdings.ALIGN_REL / ALIGN_FLOOR.

Worked examples (band for target 8% = max(2pp, 2pp) = 2pp):

| target | actual | gap | verdict |
|---|---|---|---|
| 8% | 1% | −7pp | ADD_LONG |
| 8% | 7.5% | −0.5pp | HOLD_LONG (aligned) |
| 8% | 12% | +4pp | REDUCE_LONG |
| 0% (SELL) | 10% | — | EXIT_LONG |
| 0% (unselected, top-half rank) | 10% | — | REDUCE_LONG |
