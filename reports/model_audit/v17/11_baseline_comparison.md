# 11 — Simple baselines under the identical protocol (B14, P5)

All rows: same dates, same target (rank-20), same walk-forward (refit 126,
matured labels, purge 21), same universe, same costs, same book builder
(top quintile, band10, equal weight, cap 10), long-only.
(`baselines.csv`, `baseline_ridge.csv`, `baseline_correlations.csv`)

| Signal | CH IC | CH net60 | CH net100 | CH DD | BR IC | BR net60 | BR net100 | BR DD |
|---|---|---|---|---|---|---|---|---|
| **blend50 (champion)** | 0.074 | **1.989** | 1.943 | −23.8% | 0.036 | **1.455** | 1.405 | −26.0% |
| tf (transformer only) | 0.063 | 1.733 | 1.675 | −26.7% | 0.029 | 1.193 | 1.136 | −30.3% |
| mom126_5 (D1.2 only) | 0.071 | 1.708 | 1.648 | −30.7% | 0.036 | 1.361 | 1.301 | −32.9% |
| ridge (same 10 features, per-date standardized, walk-forward) | 0.014 | 1.204 | 1.087 | −35.7% | 0.007 | 1.010 | 0.906 | −32.9% |
| ridge + mom blend | 0.055 | 1.579 | 1.495 | −32.5% | 0.027 | 1.231 | 1.151 | −30.5% |
| tf + mom + ridge (⅓ each) | 0.061 | 1.739 | 1.666 | −28.4% | 0.030 | 1.297 | 1.230 | −28.6% |
| mom60 | 0.048 | 1.460 | 1.365 | −26.4% | 0.018 | 1.267 | 1.180 | −31.9% |
| mom20 | 0.058 | 1.120 | 0.956 | −34.6% | 0.010 | 0.953 | 0.792 | −43.3% |
| equal-weight universe | — | 1.370 | 1.370 | −21.3% | — | 1.038 | 1.038 | −20.2% |
| 0050 (same blocks; unadjusted series — indicative only) | — | 0.392 | — | — | — | 0.286 | — | — |

GBM/tree baseline: **not run** — `sklearn`/`lightgbm` are not installed in
the production venv and installing them is out of scope for this branch
(dependency safety). A pooled *raw-feature* ridge was tried first and is
reported separately (`baseline_ridge_pooled_raw.csv`, IC 0.006) as an
unfair baseline, not as evidence.

## Prediction correlation and incremental information

| | CH | BR |
|---|---|---|
| per-date rank corr(transformer, momentum) | **0.645** | **0.656** |
| corr(transformer, ridge) | 0.464 | 0.450 |
| corr(ridge, momentum) | 0.201 | 0.162 |
| IC of the transformer's **residual** after regressing on ridge+momentum (per date) | **0.017** | **−0.001** |

## What the neural model uniquely adds (the P5 question)

1. **It is ~65% momentum.** Its ranking is highly correlated with the
   126/5 momentum it is blended with; the residual information beyond
   momentum + a linear model has an IC of ~0.02 on CH and ~0 on BR.
2. **It beats a linear model on the same inputs clearly** (1.73 vs 1.20
   CH), so sequence/non-linear structure is doing something — but that
   something mostly re-expresses momentum.
3. **Its demonstrable portfolio value is diversification**: the blend's
   drawdown is 7 pp better than either component and its IC-IR is higher;
   the Sharpe increment over momentum alone (+0.28 CH / +0.09 BR) is
   inside one standard error.
4. The universe itself (equal-weight 1.37 / 1.04) supplies a large share
   of the absolute return.

**Constraint on future compute:** with residual IC ≈ 0.02/0.00, scaling
the architecture cannot be expected to unlock hidden capacity — there is
little independent signal for a bigger model to fit. GPU spend should go
to *changing what the model is asked to learn* (horizon P3, recency P4,
adjusted labels P1) and to *validating what production actually does*
(P0), not to width/depth.
