# Pre-registration — Cycle 10 (E1 close+D1.2 features at champion strength)

Registered: 2026-07-22 23:00 (before running)

Prior evidence: G5 screen (2 seeds): close_d12 LO 1.82 > close_only LO 1.75, L/S
1.49 < 1.53, val IC 0.069 vs 0.071. The 5-seed/preset-B upgrade changed rank-20
close_only from 1.53 → 1.91; feature-set ordering may also change at full strength.
A1 taught us 2-seed screen orderings are unreliable.

Config: feature_set=close_d12, seq 60, preset B, seeds [0..4], equal-all, rank-20
target, refit 126, OOS 2023-01→. Panel `LOOP_E1_close_d12_presetB`.

Gates:
- Selection stays honest: compare val IC vs champion close_only (0.074). If val IC
  is lower, close_only remains production default regardless of OOS numbers
  (report OOS anyway, flagged as not-selected-by-protocol).
- If val IC ≥ champion AND OOS LO/L/S beat champion (1.93/1.91) → bear-window
  validation required before promotion.
- Also report its blend50+band10 with D1.2 (redundancy risk: the feature already
  contains D1.2 — blend gain should shrink; that itself is informative).

Runs batched with A2 in one process (dataset-build amortization).
