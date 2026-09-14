# 17 — P0-A result: 126-session vs 5-session vs DAILY refit (paired seeds {0,1,2})

**Interval 2026-01-05 → 2026-07-23 is a BURNED DIAGNOSTIC INTERVAL** — it
already influenced research selection. Nothing here is untouched final
OOS, final validation, or prospective confirmation; the numbers are
parity diagnostics, not independent estimates of strategy performance.

Spec: `p0_refit_parity_spec.json` (daily interpretation frozen before any
daily fit). Runner: `research/run_p0_refit_parity.py`. Arm A/B fits
**reused from the P0-A′ cache** (config hash `2e8b…` matched; 87 fits);
arm C: **393 new daily fits, 5.54 h GPU, 0 failed, 0 retried** (33–70 s
each). Identical 131 decision dates in all arms. Parity sanity: the first
daily refit (2026-01-05) reproduced arm A's fits for the same date to four
decimals (val IC +0.0221/+0.0371/+0.0345) — same data, same seeds, same
training → cadence is the only difference.

## Arms (transformer-only unless stated; portfolio = blend50 + band10 long-only)

| Metric | A: 126-session | B: 5-session | C: DAILY |
|---|---|---|---|
| rank IC mean (HAC SE) | **0.1216** (0.068) | 0.1061 (0.074) | 0.1090 (0.073) |
| rank IC std | 0.234 | 0.243 | 0.242 |
| positive-IC frequency | 74.0% | 68.7% | 72.5% |
| top20 − bottom20 fwd_20 spread | 7.30% | 6.81% | 6.79% |
| gross return / block (7 blocks) | 8.42% | 8.19% | 8.19% |
| net60 / net100 Sharpe-like | 3.108 / 3.081 | 3.107 / 3.075 | 3.107 / 3.075 |
| max drawdown (net60) | −5.78% | −5.83% | −5.83% |
| one-way turnover (names changed) / rebalance | 0.223 (9.4) | 0.237 (9.9) | 0.237 (9.9) |
| seed dispersion (mean score std) | 0.0136 | 0.0134 | 0.0138 |
| own validation IC of the fits (mean; p10–p90) | 0.092 (0.03–0.16) | 0.118 (0.04–0.17) | 0.119 (0.04–0.17) |
| Spearman(val IC, subsequent block OOS IC) across refits | n/a (2 refits) | **−0.65** | **−0.50** |

## Paired differences (same 131 dates; HAC lag-20 SE)

| | B − A | C − A | C − B |
|---|---|---|---|
| Δ rank IC | −0.0155 (SE 0.0099) | **−0.0126 (SE 0.0077)** | +0.0030 (SE 0.0036) |
| share of dates with Δ > 0 | 44% | 41% | 37% |
| Δ spread (pp) | −0.49 (0.43) | −0.51 (0.32) | −0.02 |
| Δ positive-IC freq (pp) | −5.3 | −1.5 | +3.8 |
| Δ refit-to-refit rank corr | −0.043 | −0.031 | +0.012 |
| Δ turnover | +0.014 | +0.014 | 0.000 |
| Δ max DD (pp) | −0.05 | −0.05 | 0.00 |
| Δ net60 / net100 | −0.001 / −0.006 | −0.001 / −0.006 | 0 / 0 |
| Δ own val IC | +0.026 | +0.027 | +0.001 |

## Refit shock (same as-of date, IDENTICAL feature windows: previous model vs new model)

| | A (1 boundary) | B (26) | C (130) |
|---|---|---|---|
| same-date score corr | 0.942 | 0.916 | 0.924 |
| same-date rank Spearman | 0.996 | 0.953 | **0.965** |
| same-date top-20% overlap (Jaccard) | 0.833 | 0.887 | **0.908** |
| names entering = leaving top quintile per refit | 2.0 | 1.31 | **1.06** |
| mean |rank move| (of 108) | 2.1 | 4.2 | **3.4** |
| max |rank move| per refit (mean / worst) | 12 / 12 | 65 / 103 | **48 / 106** |

Market-evolution baseline (consecutive DATES, model held fixed within a
block, arm A): rank corr **0.997**, top-Q overlap **0.972**. With daily
refits (arm C, consecutive dates = market + refit): 0.964 / 0.904.

**Derivation of the "≈ 90% refit shock" statement (a DERIVED DIAGNOSTIC,
not an independent statistic).** Define rank *movement* between two
cross-sectional score vectors as `M = 1 − ρ_Spearman`, and assume the two
sources of movement are approximately additive on this scale:

| Quantity (arm, comparison) | ρ | M = 1 − ρ |
|---|---|---|
| market evolution only: arm A, consecutive dates, same model | 0.997 | **0.003** |
| market + refit: arm C, consecutive dates, model refit daily | 0.964 | **0.036** |
| refit shock only: arm C, same date, old vs new model, identical inputs | 0.965 | **0.035** |

Share attributable to refitting: by subtraction
`(M_total − M_market) / M_total = (0.036 − 0.003) / 0.036 = 0.92`;
by direct measurement `M_refit / M_total = 0.035 / 0.036 = 0.97`
(the two differ because additivity is only approximate). Both readings
put the refit share at **≈ 0.9 or above**, hence "≈ 90%". On the
top-quintile-overlap normalization (`M = 1 − Jaccard`): market 0.028,
total 0.096, refit 0.092 → 0.71 by subtraction, 0.96 directly. The
statement therefore holds under both normalizations at the "refit shock
dominates" level; the exact percentage depends on the normalization and
on the additivity approximation and should not be quoted more precisely
than "roughly 90% on the rank-correlation scale". The refit alone churns
~1 top-quintile name per day and occasionally moves a single name 50–100
ranks.

## Answers to the six P0-A questions

1. **Does daily refitting materially reduce predictive quality vs 126-session?** Modestly: ΔIC −0.0126 (≈ −10% relative, 1.6 HAC SE; A ahead on 59% of dates), spread −0.5 pp. Above the pre-registered −0.005 tolerance, below the RISK bar.
2. **Does the 5-session deterioration grow at daily cadence?** No. C − B = +0.003 ± 0.0036: daily ≈ 5-session. The effect is a **step from sparse (126) to dense (≤5) refitting, not a progressive density effect**.
3. **Does band10 still absorb the ranking noise?** Yes — the B and C books are identical to each other and within 0.006 Sharpe / 0.05 pp DD of A.
4. **Does daily retraining raise turnover?** Marginally: +0.014 one-way (+0.6 names per rebalance).
5. **Market evolution vs refit shock?** Refit shock dominates day-to-day rank movement (0.965 same-date refit corr vs 0.997 market-only day-to-day).
6. **Does higher val IC under dense refitting fail to translate?** Yes, quantified: dense arms' own val IC is **+0.027 higher** while their OOS IC is **−0.013 to −0.016 lower**; across refits the correlation between a fit's val IC and its subsequent block OOS IC is **negative (−0.65 B, −0.50 C)** — refits that look best on the (13-month-stale) holdout do worst afterwards.

## Pre-registered classifications (applied mechanically by the runner)

- **P0_A_DAILY: PARITY_WARNING** (5-session re-classified: PARITY_WARNING).
- **REFIT_DENSITY_EFFECT: INCONCLUSIVE** by the frozen rule (C is worse than B on 1 of 7 components; C−A ΔIC is neither ≤ B−A − 0.005 nor ≥ B−A + 0.005). Substantively: a sparse-vs-dense step, flat thereafter.
- **BAND10_ROBUSTNESS: ABSORBS_DIFFERENCE** (Δnet100 −0.006, ΔDD −0.05 pp, Δturnover +0.014).

## What this does and does not support

- Supports: the research references (126-session cadence) are *slightly
  optimistic* descriptions of what daily production ranking delivers at
  the model level; the portfolio the user follows is unaffected by
  cadence because band10 absorbs it; the ranking layer / WATCH list
  carry ~1 name/day of refit-induced churn.
- Does not support: any production change on this evidence alone; P0-B
  (7-seed, ≈17 h) is **not justified** as the next step — the 3-seed
  paired design already answers the density question with tight paired
  SEs, and a 7-seed run would refine magnitude, not direction. The
  cadence question is better pursued as **P6**: does a *weekly or
  monthly* full refit recover the 126-session IC (cheap: reuse arm A/B
  machinery, ~1–2 h), together with **P4** (the negative val→OOS
  correlation points at the stale holdout as the mechanism).
- Not committed automatically; awaiting review.
