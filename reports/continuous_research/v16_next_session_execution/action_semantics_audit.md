# v16 Task 4 — Model Action Semantics Audit (final)

Date: 2026-08-18. Code-traced (not inferred from names); these semantics
are PRESERVED as-is by Stage A — `model_action` is never mutated, and the
new `user_action` layer sits beside it, never on top of it.

## The audited meanings (blend50_band10 decision book)

Assignment: research/blended_decision_book.py:74-89. Reference state
`prev_w` = the model's own previous paper book
(reports/paper_trading/books/<date>_blend50_band10.csv via
paper_trading._prev_book:70-77) — never the user's portfolio.

| Action | Exact condition (w1 = today's target, w0 = previous paper book) | Meaning |
|---|---|---|
| BUY | w1 > 0 and w0 == 0 | new entrant to the model paper book |
| HOLD | w1 > 0 and w1 >= w0 − 1e-6 | incumbent, weight unchanged **or increased** (no ADD label exists; a +1.9pp add on 2026-08-07 is labelled HOLD) |
| REDUCE | w1 > 0 and w1 < w0 − 1e-6 | incumbent, target weight strictly decreased |
| SELL | w1 == 0 (name absent from today's book) | model paper book exits the long — **NOT an open-short signal** |
| WATCH | blend rank ≤ top 30% of universe, not otherwise emitted (:85-89) | bullish-leaning near-miss (band kept incumbents in the slots) — **never bearish** |

## Why SELL can never mean shorting (structural evidence)

1. SELL is the `else` branch of a dictionary lookup defaulting to 0.0 —
   it literally encodes "absent from today's long book".
2. Production weights are clamped to [0, 0.10]:
   transformer_portfolio.py:60 `np.maximum(w, 0)` before the cap
   waterfill; inputs are `np.ones` (paper_trading.py:58). Negative
   target weights are unreachable.
3. No short book exists in any production path: daily_ops' eight steps
   import only cap_weights/NAME_CAP/CAP_TOL; the long-short machinery
   lives solely in the offline backtest function; short_side_v12.py is
   research-only and its verdict REJECTED the short side 6/6.
4. The repo's own consumers agree: the v14 playbook maps SELL → MID
   (neutral pseudo-state, not bottom-quintile) and renders it EXIT_LONG.

## Why unowned names showed as HOLD (the defect Stage A fixes)

Every action is computed from (w1, w0) — two model-side quantities. A
user owning none of a HOLD name was shown "HOLD" although the actionable
truth is "the model maintains a position you don't have". Stage A: that
row's `user_action` is now OPEN_LONG_EXISTING_TARGET (see
user_action_mapping.md); the bare labels HOLD/SELL never appear in the
user_action column (regression-tested).

## Known quirks documented for the record (not changed in Stage A)

- Two labelling implementations coexist: the decision book (1e-6 exact
  comparison, equal weight, band10) and inference_transformer_eod's
  `_target_book.csv` (2pp tolerance, increases = BUY, band 5%,
  inverse-vol). The overlay and Stage A read only the decision book.
- The 10% "band" is a rank/selection band (incumbents keep slots while
  inside top k×1.2), not a weight-tolerance band; WATCH's 30% threshold
  is a third, different cutoff.
- `ranks.nsmallest(0)` in the labelling union is dead code (empty set).
- Sort order of the book CSV is alphabetical by action, then rank.
