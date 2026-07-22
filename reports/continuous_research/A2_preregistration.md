# Pre-registration — Cycle 8 (A2 confidence filtering + champion reproducibility)

Registered: 2026-07-22 22:35 (before running)

## Part 1 — champion rerun with score_std (reproducibility check)

`train_transformer_eod.walkforward` now emits `score_std` (cross-seed std;
PRODUCTION_EDIT_PLAN.md Edit 1). Rerun the exact champion config
(close_only, seq60, preset B, seeds [0..4], equal-all, rank-20, refit 126,
OOS 2023-01→) → panel `LOOP_A2_champion_std`.

Gates:
- Mean path is untouched by the edit; run-to-run drift comes only from CUDA/AMP
  nondeterminism. Reproducibility PASS if standalone L/S net60 ∈ 1.91 ± 0.15 and
  val-IC within ±0.01 of 0.074. A miss ⇒ flag champion fragility on the scoreboard
  (that itself is a result).

## Part 2 — confidence filtering (uncertainty veto)

On the new panel (and its blend50+band10 book), pre-declared variants:
- **CF1** drop names in the top 33% of same-date score_std before selection
- **CF2** drop top 20%
- **CF3** shrinkage: score_adj = z(score) × (1 − pct_rank(score_std))
Applied to: champion standalone (L/S + LO) and blend50+band10 (L/S + LO).

Gates: net60 improvement ≥ 0.05 with DD not worse by > 2pp on the champion window
⇒ run bear-window validation (A2B) before any promotion. Anything less ⇒ reject
or research-only. All 3 variants reported regardless.

Runtime: ~15 min GPU (7 refits × 5 seeds) + CPU evaluation.
