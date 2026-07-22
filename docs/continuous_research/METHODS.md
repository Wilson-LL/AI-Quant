# Continuous research loop — methods and protocol

Branch `research/continuous-alpha-loop-4060ti` · started 2026-07-22.

## Protocol (inherited from the 20h sprint, unchanged)

- Chronological walk-forward only; refits every 126 trading days unless stated;
  purge = seq_len + horizon at every split; matured labels only; execution lag 1
  day; no full-sample scalers; quintile books with hard 10% name cap (water-fill)
  and soft 20% sector cap; min_names 60; costs one-way bps on L1 turnover.
- Model selection by validation rank IC only. OOS results reported for every
  config that ran, including failures.
- Every experiment is pre-registered (hypothesis, variants, gates) in
  `reports/continuous_research/*_preregistration.md` BEFORE running. Multiplicity
  discipline: single-window wins are "monitor", promotion needs cross-window
  consistency (champion 2023–26 panel AND bear 2021–26 panel).

## Reusable infrastructure

- Cached walk-forward score panels (`reports/transformer_gpu/panels/*.csv.gz`)
  make Track C/D (blend, construction, filter) experiments CPU-only.
- `research/loop_experiments.py` — GPU configs (one JSON + panel per config,
  crash-safe; batch multiple configs per process: dataset build dominates
  process wall-clock at ~30 min vs ~15 min training).
- `research/adaptive_blend.py`, `research/blend_construction.py`,
  `research/exposure_scaling.py` — panel-level experiment drivers.
- `research/paper_trading.py` — shadow books (backfill/snapshot/evaluate).
- `research/blended_decision_book.py` — Section-9 decision book for the
  promoted blend candidate.

## Key decisions so far (see RESEARCH_SCOREBOARD.md for numbers)

1. Production candidate book = **50/50 z-blend of champion transformer and D1.2
   momentum, top-quintile, equal-weight, 10% no-trade band, hard 10% cap, 20d
   hold, monthly retrain** ("blend50+band10").
2. Static beats adaptive: no trailing-data weight scheme beat static 50/50.
3. Blend at score level, not return level (+0.2 Sharpe).
4. No-trade band 10% is a robust improvement (both panels, both modes, all costs);
   inverse-vol weighting is not.
5. Name cap can be tightened to 7.5% at zero Sharpe cost when concentration
   matters.
6. 2-seed screens mis-rank configs (A1): only 5-seed results are decision-grade.
