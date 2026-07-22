# Pre-registration — Cycle 1 (C1 adaptive blend, C2 static blend frontier)

Registered: 2026-07-22 20:40 (before any experiment code was run)

## Hypothesis

**C1:** A walk-forward adaptive TF/D1.2 blend weight — computed only from trailing,
matured information (rolling sleeve Sharpe, rolling signal IC, or market-drawdown
regime state) — achieves L/S net60 Sharpe > static 50/50 on the 2021-01→2026-07
panel (`BEAR_presetB_2021`), without losing to static 50/50 on the 2023–26 subwindow.

**C2:** Establish the static blend frontier (score-level and return-level) on the same
panel as the reference surface.

Rationale: prior sprint showed regime complementarity (2022: TF +0.58 vs D1.2 −1.55)
and that static 50/50 beat both standalones on 2021–26 (1.37 vs 1.00/1.09). If the
complementarity is detectable from trailing data, adaptive weighting should add value;
if not, static 50/50 stands.

## Pre-declared variants (ALL will be reported; no post-hoc additions)

Static, score-level (z-blend): w_tf ∈ {0, 0.3, 0.5, 0.7, 1.0}
Static, return-level (capital split between the two L/S sleeve books): w_tf ∈ {0.3, 0.5, 0.7}

Adaptive, return-level (floor 0.2 / cap 0.8 on w_tf unless stated):
- **AD1** softmax of trailing sleeve Sharpe, last 6 matured rebalances, λ=1
- **AD2** same, last 12 matured rebalances
- **AD3** trailing 126-trading-day mean daily rank IC per signal (matured: IC dates ≤ t−22), w ∝ max(IC,0)+0.05
- **AD4** market-drawdown regime: equal-weight universe index >10% below its 252d high at decision close → w_tf=0.8, else 0.5
- **AD5** momentum-crash detector: mean of last 2 matured D1.2 sleeve rebalance returns < −3% → w_tf=0.8, else 0.5

## Leakage guards

- Sleeve returns of rebalance k mature one day after rebalance k+1 ⇒ decisions at
  rebalance t use sleeve returns up to rebalance **t−2** only.
- Daily IC at date d uses fwd20 from close d+1 ⇒ matured at d+21; decisions at t use
  IC dates ≤ t−22.
- Regime index uses closes ≤ decision date t; execution is at close t+1 (lag 1).
- Return-level weight changes incur turnover: total turn = w·turn_tf + (1−w)·turn_d12 + |Δw|/2 per leg-averaged convention (conservative).

## Success criteria (declared in advance)

- Primary: full-window 2021–26 L/S net60 Sharpe vs static 50/50 recomputed on the
  identical panel/protocol.
- Secondary: 2022 yearly net60 Sharpe, maxDD, Calmar, 2023–26 subwindow Sharpe, net100.
- Multiplicity discipline: 5 adaptive variants are tested. A single variant beating
  static 50/50 by a small margin ⇒ verdict "monitor", not "promote". Promotion requires
  beating static on BOTH windows with margin ≥ 0.1 Sharpe or clear DD improvement,
  and the mechanism must be economically interpretable.

## Budget

CPU-only on cached panels; expected < 15 min wall. No GPU, no retraining.
