# 16 — P0-A′ result: refit-cadence screen (paired seeds {0,1,2}; 126 vs 5 sessions)

Spec: `p0_refit_parity_spec.json` (frozen before the run). Runner:
`research/run_p0_refit_parity.py`. Artifacts: `p0/result_P0_A_prime.json`,
`p0/panel_P0_A_prime_arm{A,B}.csv.gz`, `p0/stability_*.csv`, `p0/fits/`.
Interval 2026-01-05 → 2026-07-23, **131 decision dates, identical in both
arms**. This interval already influenced research selection — this is a
**methodology diagnostic, not OOS validation** of the champion.

Compute: 87 seed-fits (arm A 2 refits × 3 seeds; arm B 27 refits × 3
seeds), **1.28 h GPU**, 0 failed, 0 retried; per-fit 34–68 s, 4–7 epochs.

## A vs B on the same 131 dates (transformer-only = raw ensemble score)

| Metric | A: 126-session | B: 5-session | B − A |
|---|---|---|---|
| rank IC mean (vs fwd_20) | **0.1216** | 0.1061 | **−0.0155** (HAC SE 0.0099; 44% of dates B>A) |
| rank IC std | 0.234 | 0.243 | +0.009 |
| positive-IC frequency | 74.0% | 68.7% | **−5.3 pp** |
| top20 − bottom20 fwd_20 spread | 7.30% | 6.81% | −0.49 pp (HAC SE 0.43) |
| seed dispersion (mean score std) | 0.0136 | 0.0134 | ≈ 0 |
| in-sample val IC of the fits | 0.092 | 0.118 | fits look *better* on their own val set |

## Blend50 + band10 long-only book (7 rebalance blocks — too few to be decisive)

| Metric | A | B | B − A |
|---|---|---|---|
| gross return / block | 8.42% | 8.19% | −0.23 pp |
| net60 / net100 Sharpe-like | 3.108 / 3.081 | 3.107 / 3.075 | −0.001 / −0.006 |
| max drawdown (net60) | −5.78% | −5.83% | −0.05 pp |
| one-way turnover / rebalance | 0.223 | 0.237 | **+0.014** |
| names changed / rebalance | 9.4 | 9.9 | +0.6 |

## Prediction stability at refit boundaries (same date scored by old and new fit)

| | A (1 boundary) | B (26 boundaries) |
|---|---|---|
| score correlation | 0.942 | 0.916 |
| rank correlation | 0.996 | **0.953** |
| top-quintile overlap (Jaccard) | 0.833 | **0.887** |

## Classification (pre-registered thresholds applied mechanically): **PARITY_WARNING**

- Not PARITY_OK: ΔIC −0.0155 is below −0.005 and ~1.6 HAC SE below zero;
  Δpositive-frequency −5.3 pp crosses the −5 pp line.
- Not PARITY_RISK: only one of the four risk votes fired (positive
  frequency); the spread change (−0.49 pp) sits just inside the −0.5 pp
  line, refit-to-refit stability is high (rank corr 0.95, overlap 0.89),
  turnover moved +0.014 (< +0.10), and every book-level metric is
  indistinguishable between arms.

## Reading (what the diagnostic says, and its limits)

1. Denser full refitting of the same seeds **did not improve** ranking
   quality; it moved IC in the *wrong* direction by a small but not-
   noise-level amount, while its own validation IC rose — the project's
   familiar val-IC/OOS dissociation, now observed on the cadence axis.
2. The **book is insensitive**: band10 hysteresis absorbed the extra
   rank movement (turnover +0.014, Sharpe/DD identical). So the *portfolio*
   the user follows is not obviously harmed by daily refits; the *ranking
   layer / WATCH list* is what sees the jitter.
3. Stability of consecutive 5-session refits is high (rank corr 0.95);
   daily refits would add ~26× more boundaries and this number is the one
   to watch in P0-A.
4. Limits: 131 dates ≈ 6–7 independent 20-session blocks; a 3-seed
   ensemble; one half-year that was part of the selection window. The
   result motivates the daily-cadence arm; it does not settle it.

## Implication for P0-A (3 seeds, 126 vs daily)

Justified by this screen: the direction is adverse and non-trivial at
5-session cadence, and daily cadence is the production reality. P0-A
(≈7.4 h) would tell whether the effect grows with density. **Not
launched** — awaiting review per the task's stop rule.
