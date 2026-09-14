# 07 — Regime robustness (B9)

Objective, trailing-only labels on the equal-weight universe index
(`regime_labels.csv`): trend = index above/below its own 126-session MA;
vol regime = terciles of trailing 20-session index vol; breadth =
terciles of the share of names above their 60-session MA; shock =
5-session index return < −5%. No hand-labelled periods.

| Regime | CH IC | CH IC>0 | CH top-Q excess (20d) | BR IC | BR IC>0 | BR top-Q excess |
|---|---|---|---|---|---|---|
| BULL trend | 0.080 | 67% | +3.4% | 0.050 | 61% | +2.6% |
| **BEAR trend** | 0.034 | 65% | +0.6% | **−0.006** | 57% | +0.3% |
| LOW vol | 0.106 | 68% | +3.6% | 0.055 | 56% | +2.4% |
| MID vol | 0.054 | 65% | +2.0% | 0.044 | 63% | +1.6% |
| HIGH vol | 0.057 | 67% | +2.6% | **0.006** | 57% | +1.7% |
| BROAD breadth | 0.040 | 63% | +1.6% | 0.030 | 57% | +1.4% |
| MID breadth | 0.111 | 73% | +4.2% | 0.085 | 68% | +3.3% |
| **NARROW breadth** | 0.026 | 59% | +1.1% | **−0.030** | 52% | +0.5% |
| SHOCK 5d (n=13/25) | 0.119 | 85% | +1.9% | 0.081 | 68% | +3.3% |

## Findings

1. **The signal is a bull-market / mid-breadth signal.** In bear trends
   and narrow-breadth markets its IC is ≈ 0 (BR) — not strongly negative,
   but with no selection edge the long book simply rides the market.
2. **2022 is the stress case**: IC −0.057 for the year; the strategy was
   approximately flat net (−0.15 Sharpe), consistent with "no edge + costs".
3. Post-shock periods are actually good for the signal (rebound
   continuation), matching the July-2026 prospective cluster.
4. **Are recent losses regime degradation or normal drawdown?** The 2025
   IC (0.045) was the weakest full year since 2022 while still positive;
   2026 YTD is the strongest. The paper book's −18.6% block (2026-07-01)
   sits inside the backtest long-only drawdown range (−24% to −26%). On
   the model's own terms this is a **normal drawdown, not degradation**.
   The user's realized experience is dominated by a different portfolio
   (10).

Implication for research: any challenger must be reported on the BEAR and
NARROW slices separately; a model that only improves BULL/MID-breadth IC
adds nothing where the current system is weakest.
