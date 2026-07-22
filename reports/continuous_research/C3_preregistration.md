# Pre-registration — Cycle 4 (C3 exposure scaling on blend50+band10)

Registered: 2026-07-22 21:30 (before running)

## Hypothesis

Walk-forward exposure scaling (risk-off gates using only trailing data) reduces the
max drawdown of the promoted blend50+band10 book by ≥ 20% relative, at a net60
Sharpe cost ≤ 0.1, on BOTH panels (champion 2023–26, bear 2021–26). Cash earns 0.

Distinct from rejected C1: C1 shifted weight *between sleeves* (alpha timing);
C3 scales *total exposure* (risk timing). DD-reduction is the goal, not Sharpe.

## Pre-declared variants (all reported; applied to L/S and LO blend50+band10)

- **EX1** market-drawdown gate: exposure 0.5 if equal-weight universe index is
  >10% below its 252d high at decision close, else 1.0.
- **EX2** market-vol target: exposure = clip(0.15 / trailing-60d annualized market
  vol, 0.3, 1.0).
- **EX3** own-equity drawdown gate: exposure 0.5 if the strategy's own matured
  paper equity (returns through rebalance t−2) is >10% below its high, else 1.0.

Turnover accounting: scaled book turnover = e_t·turn_t + |Δe_t|/2.

## Success criteria

Filter-candidate (category C) if maxDD improves ≥ 20% relative with ΔSharpe ≥ −0.1
on both panels. Reject if DD improvement < 20% or Sharpe cost > 0.2 anywhere.
Report full-window + 2022 + 2023+ metrics at 0/60/100/150 bps.
