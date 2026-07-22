# Cycles 1–2 summary (C1/C2 blends · D1 construction)

2026-07-22 21:05 · CPU-only (cached panels) · full JSON: C1_C2_results.json, D1_results.json

## C1 — adaptive blend weights: REJECTED

All 5 pre-declared walk-forward adaptive variants (rolling sleeve Sharpe ×2, trailing
IC, market-drawdown regime, momentum-crash detector) underperform the static 50/50
score blend on 2021–26 (best adaptive AD5 1.28 vs static 1.37 L/S net60). Regime
switches (AD4 1.27, AD5 1.28) beat performance-chasing (AD1 1.08) but nothing beats
static. Consistent with the recency-weighting rejection: trailing-window adaptation
reacts too late at 20d rebalance granularity. Adaptive weighting is closed unless a
*forward-looking* regime signal (not trailing performance) is found.

## C2 — static blend frontier: finding kept

- Score-level blending strictly dominates return-level sleeve blending
  (50/50: 1.37 vs 1.18) — blend signals before selection, not books after.
- Frontier is flat across w_tf 50–70 (1.33–1.37 full window); 50/50 stays the pick
  (better 2023+ subwindow 2.02, simpler).

## D1 — construction transfer: PROMOTED

Blend50 book with 10% no-trade band, equal weight:

| window | mode | net60 | net100 | maxDD | turn | vs band0 |
|---|---|--:|--:|--:|--:|---|
| 2023–26 | L/S | **1.95** | 1.80 | −12.3% | 0.30 | 1.79, −14.4% |
| 2023–26 | LO | 1.83 | 1.79 | −27.0% | 0.25 | 1.80, −30.1% |
| 2021–26 | L/S | 1.42 | 1.27 | −26.4% | 0.33 | 1.37, −30.5% |
| 2021–26 | LO | 1.48 | 1.43 | −30.8% | 0.28 | 1.42, −34.8% |

- Improvement is monotone 0→5→10, consistent in all 4 panel×mode combos and at
  100/150 bps. Band 15/20 shape-check: plateau (champion L/S 1.78–1.86, bear
  1.41–1.46) — 10% is not a knife-edge; adopted as the pre-registered value.
- Inverse-vol weighting hurts in all 12 cells → rejected for the blend book.
- **New best overall book: blend50 + band10 — beats standalone champion (1.95 vs
  1.91) with lower DD (−12.3% vs −15.0%) and beats it on the bear window (1.42 vs
  1.00), where it also beats D1.2 (1.09) and plain blend (1.37).**

Verdict: category **B (blend candidate) → new recommended production book**,
subject to A1 (rank-10) comparison and paper trading.
