# 23 — Preregistration: H-RESIDUAL-SIGNAL and H-FACTOR-PREMIUM

**Status:** frozen before any residual, benchmark or factor-premium result was computed. **Machine-readable (authoritative):** `h_residual_factor_spec.json`. **Code:** `research/h_residual_factor.py`, which refuses to run until the spec is committed and unmodified. **Tests:** `tests/test_h_residual_factor.py`.
**Evidence label:** BURNED MECHANISM DIAGNOSTIC. This is not independent validation and not a trading rule. Zero GPU and no retraining; the only fitted objects are the simple linear benchmarks. Production is unchanged.

## Question

Does the production-like neural model carry cross-sectional ranking information that its own input factors do not explain? In other words, is it **factor repackaging** or does it hold **nonlinear residual signal**? And does the factor it leans on most pay positively out of sample?

## Sample

- **Primary:** P0 arm B, the 5-session refit with three seeds and the production recipe. It covers 131 decision dates from 2026-01-05 to 2026-07-23 in 27 refit blocks. The nine P4-B0 refit blocks are a subset.
- **Secondary:** arm C (daily refit, production cadence) and arm A (126-session refit), on the same block calendar.
- **Prospective, descriptive only:** production 7-seed predictions for dates from 2026-07-24 whose 20-session forward return has matured. The cache ends 2026-09-11, so this is roughly one label window.
- **Uncertainty unit:** the refit block. Seeds are averaged inside the score and never counted. The primary SE is Newey–West with lag 4 on the block series, because 20-session labels overlap about four blocks. Intervals use t with 26 degrees of freedom.

## Benchmarks (training rows only, no tuning)

- **A — 12-1 momentum:** close(t−21) / close(t−252) − 1. This is not a network input.
- **B — D1.2 momentum:** identical to the network input `mom_126_5`, which is about 6-month momentum skipping 5 sessions. So B is `mom_126_5` alone. Session 02 called this "12-1 momentum"; that was a mislabel.
- **C — ridge on the 10 inputs:** refitted per block on matured dates only, with a fixed α = 1.0.

## Residualisation (exact)

Per decision date, across names, using date-t information only:

- s is the rank-Gaussian score of the Transformer.
- Fit s by OLS on an intercept plus a factor set X.
- The fitted value is the factor component. The residual is s minus the fitted value.
- Forward returns never enter this step.

| residual set | X | role |
|---|---|---|
| **R10 (primary)** | the 10 input features | removes any date-t linear combination of the network's own inputs |
| R_ridge | the ridge benchmark score | secondary |
| R_mom | `mom_126_5` | secondary |
| R_mom12_1 | 12-1 momentum | secondary |

- **RESIDUAL_IC** is the Spearman correlation of the residual with the 20-session forward return, per date, averaged within a block.
- **FACTOR_SHARE** is the fraction of the Transformer's rank-Pearson IC carried by the fitted component. That split is exactly additive.

## Method calibration, done before any real result

With plain uniform ranks, a score that was purely a linear mix of its inputs left a spurious residual IC of about 0.026. So rank-Gaussian scores are used instead.

A real-data **placebo** measures the remaining false-positive floor. It is a score built only from `mom_126_5` and `vol_60`, whose residual IC should be zero. If the placebo's mean residual IC is 0.01 or more in absolute value, the result is flagged METHOD_SENSITIVE. PRESENT must then beat the placebo by more than 0.01.

## H-RESIDUAL-SIGNAL classification (R10, arm B)

Rules are checked in this order: PRESENT, ABSENT, WEAK, INCONCLUSIVE.

| label | rule |
|---|---|
| **PRESENT** | mean > +0.01; lower 95% bound > 0; more than 50% of blocks positive; leave-one-block-out minimum > 0 |
| **ABSENT** | upper 95% bound < +0.01 (equivalence) |
| **WEAK** | mean > 0; at least 50% of blocks positive; the CI includes 0 |
| **INCONCLUSIVE** | everything else |

## H-FACTOR-PREMIUM (per factor: `mom_126_5`, 12-1 momentum, `vol_60`)

- **Statistic:** the daily rank IC over the 131 primary dates. The SE is Newey–West with lag 20. Degrees of freedom are the number of non-overlapping 21-session windows minus 1.
- **Also reported:** blocks positive, the two halves split at the median date, the top-minus-bottom quintile spread, the prospective mean, and calendar years 2016–2025 as context only.

Rules are checked in this order: ROBUST, ABSENT, REGIME_DEPENDENT, WEAK, INCONCLUSIVE.

| label | rule |
|---|---|
| **ROBUST** | mean ≥ 0.02; lower bound > 0; at least two-thirds of blocks positive; both halves positive; prospective mean not below −0.05 |
| **ABSENT** | upper bound < 0.01, or the mean and both halves are at or below 0 |
| **REGIME_DEPENDENT** | halves of opposite sign with one half at 0.05 or more in absolute value; or a positive mean with under 60% of blocks positive and one half at 0.05 or more |
| **WEAK** | a positive mean otherwise |
| **INCONCLUSIVE** | everything else |

No label may rest on one block.

## Long-training attribution

- **LONG refits:** decompose each epoch's IC into its R10-fitted part and its residual part, at epoch 3 and at the mean of epochs 50, 75 and 100. FACTOR_SHARE of the early-minus-late gap is the fitted-part gap divided by the total gap.
  - **EXPLAINS_MOST:** pooled share ≥ 0.5, and ≥ 0.5 on at least 2 refits with a non-trivial gap.
  - **PARTIAL:** pooled share from 0.25 up to 0.5.
  - **NOT_EXPLAINED:** pooled share < 0.25.
- **P4-B0 refits:** across the 9 refits, correlate the epoch-3-minus-epoch-15 OOS gap with exposure × momentum payoff, using a permutation p-value.

## Decision tree (no automatic launch)

- **Case A:** ABSENT, with the factor explaining most of the Transformer's behaviour. The network is mostly a factor extractor. Spend no GPU on bigger or longer models. Next come targets, independent data, a point-in-time universe, and corporate-action-adjusted prices.
- **Case B:** PRESENT. Design, but do not launch, an **early** LR decay near the epoch 3–5 step region. Do not use an epoch-50 fork.
- **Case C:** WEAK or INCONCLUSIVE. No architecture scaling. First build more independent sample and a longer prospective shadow history.

## Prior knowledge (disclosed)

- Report 11 found residual IC beyond momentum of 0.017 / −0.001.
- Council Session 02 found, post hoc, epoch-3 residual validation IC beyond the inputs of about 0.001, and momentum OOS IC of 0.33 / 0.26 / −0.02 on the three LONG blocks.
- The P4-B0 curves are already known.
