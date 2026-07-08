# Out-of-Sample Validation — D1.1 momentum prototype

Does D1.1 survive OOS, and how much is the full-sample number inflated by (a)
selecting the 10% cap on the full sample and (b) survivorship bias? `research/
walkforward_d1_1.py`. No ML; the D1.1 rule is unchanged. net@60bps; the universe
is survivorship-biased (upper bound).

## A. Held-out split — select on IS (2018–2021), evaluate on OOS (2022–2026)
OOS deliberately contains the **2022 momentum crash** — a stern test.

| variant | period | Sharpe | annRet | maxDD | Calmar | P&L share | 2022%/reb | n |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| D1.1 fixed 10% | FULL | 1.27 | 27.6% | −22.1% | 1.25 | 34.8% | −1.22 | 96 |
| D1.1 fixed 10% | IS 18–21 | 1.20 | 24.7% | −15.3% | 1.62 | 59.3% | — | 43 |
| **D1.1 fixed 10%** | **OOS 22–26** | **1.51** | 42.8% | −20.3% | 2.11 | 35.2% | −1.06 | 54 |
| D1 (sector only) | OOS 22–26 | 1.47 | 40.2% | −19.5% | 2.06 | 37.1% | −0.99 | 54 |
| uncapped (no caps) | OOS 22–26 | 1.64 | 48.6% | −20.7% | 2.35 | 45.8% | −1.05 | 54 |

**D1.1 survives OOS** — Sharpe 1.51, *higher* than IS (1.20) and full-sample (1.27).
Not a full-sample overfit that collapses OOS. (Honest: OOS 2022–2026 was a strong
momentum regime — AI/commodities/industrials — so the high OOS number is partly
regime-favourable, not skill. The cross-regime honest range is ~1.1 (IS) to ~1.5
(OOS).)

**Was the 10% cap chosen only because of the full sample?** No.
- IS-only (2018–2021) best name cap by Sharpe = **10.0%** — the same choice, using
  past data only. And IS cap scores are essentially flat: unc 1.18 / 15% 1.19 /
  12.5% 1.19 / **10% 1.20** / 7.5% 1.18 / 5% 1.18. The cap barely moves Sharpe.
- Therefore "IS-selected (10%) OOS" == "D1.1 fixed 10% OOS" == **1.51** identically.
- The per-name cap never earned its place on Sharpe (established in D1.1) — its job
  is **concentration control**, which holds OOS: P&L share 35.2% (D1.1) vs 45.8%
  (uncapped), max sector 21.6% vs 37.5%.

## B. Expanding walk-forward — adaptive cap vs fixed
At each rebalance, pick the name cap with the best trailing net@60 Sharpe over all
past rebalances; apply to the next unseen rebalance (min-train ≈ 3 yrs).

| strategy (OOS 2021-07 → 2026-05, 60 reb) | net@60 Sharpe |
|---|--:|
| **fixed 10%** | **1.13** |
| D1 (sector only) | 1.04 |
| adaptive WF cap | 1.00 |

Adaptive selection (chose 10% × 49, 5% × 11) **underperforms the fixed 10%** — the
caps are within noise, so chasing the trailing-best one only adds whipsaw.
**Conclusion: keep the cap FIXED at 10%; do not adapt it.**

## 7. Survivorship-bias estimate (honest, with current data)
We **cannot** measure true survivorship bias — twstock lists only currently-listed
names; delisted/failed names are absent (two requested codes were `StockIDNotFound`).
All numbers are upper bounds. Two internal proxies:

- **Long-only vs long-short (OOS):** L/S 1.51 > long-only 1.34. The market-neutral
  book is the *stronger* one OOS, so the edge is genuinely cross-sectional, **not**
  merely riding survivor-long winners/beta. (Reassuring.)
- **Drop-top-contributor (OOS):** top-3 names 1303 (materials +21.5%), 1519
  (industrial +19.8%), 1802 (materials +15.4%). Dropping top-1 → Sharpe 1.44;
  dropping top-3 → **1.33** (from 1.51, −12%). The strategy survives losing its 3
  best names but ~12% of the OOS Sharpe rests on them — moderate concentration/luck
  dependence.

**Reasoned haircut.** Full-sample selection is *not* inflating (OOS ≥ IS ≥ full).
The residual risks are survivorship (unmeasurable) + live frictions (borrow,
slippage beyond 60 bps, price limits) + regime-favourable OOS + top-name
dependence. A defensible haircut is **~20–35%** on Sharpe → the ~1.3 blended /
~1.5 OOS figure likely corresponds to a **realistically deployable ~0.9–1.1 net
Sharpe** — still a respectable momentum-factor product, but not the headline number.

## Verdict
**D1.1 is OUT-OF-SAMPLE VALID and REMAINS the recommended prototype.** It survives a
crash-in-OOS held-out test, its 10% cap is robust (IS-selection picks it; caps are
within noise), and adaptive cap-tuning is *worse* than the fixed rule — so the
frozen D1.1 is the right choice. Do not change the rule.

Honest qualifiers to carry forward: (i) OOS 2022–2026 was a favourable momentum
regime; blended cross-regime Sharpe is ~1.1–1.3, not 1.5; (ii) ~12% of OOS Sharpe
depends on 3 names; (iii) survivorship is unmeasurable here → apply the ~20–35%
haircut; (iv) it remains a **known-factor product, not alpha**, and a research
prototype, not a production/capital-ready system.
