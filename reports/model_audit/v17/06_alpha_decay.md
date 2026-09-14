# 06 — Alpha decay & rank quality (B7, B8) — historical and prospective

## Historical (frozen OOS panels; reconstructable, not contaminated)

Bucket mean forward returns by production-blend percentile (per-date means):

| Bucket | CH 5d | CH 20d | CH 40d | BR 5d | BR 20d | BR 40d |
|---|---|---|---|---|---|---|
| top 5% | 1.24% | 5.62% | 11.39% | 1.07% | 5.19% | 11.10% |
| 5–10% | 1.17% | 4.45% | 9.42% | 0.83% | 3.10% | 6.63% |
| 10–20% | 0.81% | 3.41% | 7.37% | 0.51% | 2.29% | 5.24% |
| 20–30% | 0.38% | 1.64% | 3.74% | 0.29% | 1.32% | 2.92% |
| 30–50% | 0.31% | 1.30% | 2.72% | 0.24% | 1.12% | 2.28% |
| bottom 50% | 0.21% | 0.77% | 1.51% | 0.20% | 0.75% | 1.38% |
| top20 − bot20 | 0.86% | 3.67% | 7.87% | 0.55% | 2.56% | 5.87% |

Rank IC (blend vs fwd_20): CH mean 0.067, IC-IR 0.34, positive on 66% of
856 dates; BR 0.032, IR 0.15, positive 59% of 1346. By year: 2021 0.000 ·
**2022 −0.057** · 2023 0.068 · 2024 0.070 · 2025 **0.045** · 2026 YTD 0.097.
Unstable through time: a year at zero, a year negative, weakest recent
full year 2025. Momentum alone: IC 0.063 / 0.032 ≈ blend (see 11).

## Prospective (2026-07-07 → 2026-09-11) — with the statistical caution it deserves

Source: 32 dated production prediction files realized against the cache
(`prospective.csv`, `statistics_summary.json`). **These daily rows are
not independent samples**: a 20-session label overlaps the next 20 rows.

| Horizon | daily obs (matured) | mean IC | positive | ≈ non-overlapping blocks | block-mean IC range over start dates | HAC 95% CI | n_eff (HAC) |
|---|---|---|---|---|---|---|---|
| 5d | 27 | 0.096 | 63% | 5 | −0.06 … +0.29 | **[−0.09, +0.28]** | ~0.5 |
| 10d | 23 | 0.185 | 74% | 3 | −0.18 … +0.45 | [+0.09, +0.28] | ~1.2 |
| 20d | 14 | 0.203 | 86% | **1** | −0.18 … +0.53 | not estimable | ~0 |

Calendar span 66 days (~47 sessions); the matured 20d observations span
2026-07-07 → 08-11 only. Top-quintile 20d +16.8% vs universe +6.3% vs
bottom-quintile +3.6% on those 14 overlapping dates, dominated by the
late-July rebound; the two earliest dates (07-07, 07-22) were negative.

**Reading:** the prospective evidence is *encouraging and consistent with
the historical IC*, and shows no sign of decay — but it is roughly **one
independent 20-session block**, and the 5-day HAC interval includes zero.
It must not be described as prospective validation. It becomes
decision-grade at ≥ 3 non-overlapping blocks (≈ 60 sessions, ~mid-Oct
2026 at the earliest) and genuinely informative at 6+.

## Reading

Historically the ranking is real but modest (IC 0.03–0.07), regime-
dependent (07), and most of the per-rank return comes from the top
decile. Prospectively there is no evidence of degradation; there is also
not yet enough independent data to claim the edge is intact at its
historical magnitude.
