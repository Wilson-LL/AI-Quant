# Pre-registration — Cycle 19 (RL1 pairwise ranking loss)

Registered: 2026-07-23 12:20 (after Edit 2 smoke tests; before the experiment run)

Hypothesis: optimizing a same-date pairwise logistic ranking loss (the metric
family the model is selected on) instead of MSE on rank targets improves val IC
and OOS books. Last allowed untested Track-A lever (rank-loss family; listwise
deferred unless pairwise shows promise).

Configs: champion in every other respect (close_only, seq60, preset B, 5 seeds,
equal-all, rank-20 target, refit 126, deep cache) with loss="pairwise";
OOS 2023-01→ and 2021-01→. Panels LOOP_RL1_pairwise_{2023,2021}.

Gates vs standing refs:
- Selection metric first: mean val IC vs champion's (REF23 run) — if lower,
  close_only+MSE stays default regardless of OOS (report OOS flagged).
- Standalone vs 1.91/1.91 (2023) and 1.14 (2021); blend50+band10 vs 2.06/−10.7%
  and 1.47/−18.7%. Promote only on dual-window wins (≥ +0.05) via both val-IC
  and OOS; partial → monitor; else reject and close the ranking-loss line.

Smoke evidence (not decision-grade): 3-epoch pairwise val IC 0.094 vs 2-epoch
MSE 0.073 on a 60-stock subset. Runtime ~2h GPU (pairwise epochs ~30% slower).
