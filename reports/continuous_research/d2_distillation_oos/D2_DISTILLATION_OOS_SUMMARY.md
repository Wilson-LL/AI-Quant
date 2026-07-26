# D2 Distillation OOS Walkforward Check — REJECT (0/6 gates)

Completed 2026-07-27. User-approved focused check; scope held: D2 only, no
signal research, no production changes, cache read-only, no new queue.
~1.9 h GPU (champ 4.2 ks + bear 2.7 ks). Full numbers: `D2_OOS_RESULT.json`.

## Question

Can one student model, distilled per-refit from the 7 teacher seeds (50/50
true target + teacher-mean scores), replace the 7-seed ensemble for daily
retrain efficiency?

## Answer: NO — every gate failed.

| Gate | Student | Ensemble (same-run) | Bar | Pass |
|---|---|---|---|---|
| Champ blend Sharpe ≥95% | 1.034 | 2.147 | ≥2.040 | ✗ (48%) |
| Champ DD within 1pp | −19.0% | −10.6% | ≥−11.6% | ✗ |
| Bear blend Sharpe ≥95% | 1.004 | 1.443 | ≥1.371 | ✗ (70%) |
| Bear DD within 1pp | −23.7% | −18.0% | ≥−19.0% | ✗ |
| 2022 not materially worse | −0.45 | −0.15 | ≥−0.30 | ✗ |
| Book overlap ≥0.80 | 0.52 champ / 0.70 bear | — | — | ✗ |

Weight difference (L1, blend book): substantial; top-quintile membership
diverges on roughly half the names in the champion window.

## Validity checks

- **Determinism cross-check PASSED:** the same-run ensemble books reproduce
  the frozen A8 references exactly (2.147 / 1.443) — teachers were trained
  byte-identically, so the student comparison is apples-to-apples.
- **The runtime saving is real but irrelevant:** teachers ≈ 2250 s vs
  student ≈ 340 s per window (≈6.6×) — efficiency was never the problem.

## Why the single-refit val check (96%) was misleading

This is the arc's fourth and sharpest val-IC/book-space dissociation
(after MT, TCN, and the D1 spread finding): the student matches the
ensemble's *average per-date rank correlation* while disagreeing with it on
precisely the tails that books are built from (top-quintile Jaccard 0.52).
Ensemble averaging doesn't just improve mean rank quality — it stabilizes
the extreme ranks, and a single distilled student does not inherit that.
Standing lesson reinforced: **no efficiency or architecture change should
ever be adopted on val-IC evidence alone; book-level dual-window validation
is the only decision-grade test.**

## Disposition

- **7-seed production spec stays** (unchanged throughout).
- **Distillation: REJECTED** (not research-only — 0/6 with large margins).
  The distillation line closes unless fundamentally re-designed (e.g.
  multi-student distillation or rank-tail-weighted objectives — neither is
  proposed; daily retrain at ~5 min is not a real pain point).
