# Pre-registration — Cycles D5/D6/D7 (queue v5 CPU construction experiments)

Registered: 2026-07-24 (before running; panels cached, no cache mutation while
the GPU scheduler runs)

Base: blend50 score on the deep-cache reference panels
(LOOP_REF23_champion_deep 2023–26, LOOP_BEARDEEP_rank20_2021 2021–26).

- **D5 ERC weighting:** equal-risk-contribution (inverse-vol normalized to
  risk parity approximation via 1/vol20 weights renormalized under the cap —
  full ERC needs covariance; use the standard 1/σ proxy on the selected names,
  which IS invvol → invvol was REJECTED. True test: 1/σ² weighting (variance
  parity) as the distinct untested cell. Gate: ΔSharpe ≥ −0.05 both panels
  with DD improvement, else reject.
- **D6 top-N LO:** N ∈ {10, 15} equal-weight LO books (vs top-quintile ≈ 21).
  Gate: net60 ≥ quintile LO on both panels (concentration payoff) — else
  reject; concentration risk noted regardless.
- **D7 conservative spec:** band15 + cap 7.5% combined (both individually
  Sharpe-neutral-or-better). Gate: within 0.05 of band10/cap10 refs on both
  panels ⇒ documented as the conservative production variant.

All cells reported; no post-hoc additions.
