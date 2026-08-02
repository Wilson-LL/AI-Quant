# v13 feature quality report (Phase 1)

Leakage: truncation test — per-stock features recomputed with the last 20 days removed must be identical on overlapping dates (covers future-price leakage, rolling-window causality, and timestamp<=prediction). Cross-sectional transforms are per-date by construction; targets asserted absent from inputs.

## close_only

- leakage: production control (unchanged path)
- features 10 · samples 276,679 (100.0% of close_only — the 252d lookbacks cost warmup history)
- non-finite values: 0 · constant features: none
- |corr|>0.90 pairs: 0 (full list in feature_correlation_summary.csv)
- heavy-tailed (|x|>10 on >0.1% rows): none
- top yearly drift (|year mean − overall|/std): vol_20 0.68, mom_126_5 0.68, vol_60 0.63, dist_lo_60 0.56, mom_60 0.54

## v13_f1

- leakage max diff: 0.00e+00 (PASS)
- features 13 · samples 263,611 (95.3% of close_only — the 252d lookbacks cost warmup history)
- non-finite values: 0 · constant features: none
- |corr|>0.90 pairs: 2 (full list in feature_correlation_summary.csv)
- heavy-tailed (|x|>10 on >0.1% rows): none
- top yearly drift (|year mean − overall|/std): true_range 0.66, hl_range 0.66, illiq_z 0.29, overnight_gap 0.16, close_loc 0.11

## v13_f2

- leakage max diff: 0.00e+00 (PASS)
- features 27 · samples 263,611 (95.3% of close_only — the 252d lookbacks cost warmup history)
- non-finite values: 0 · constant features: none
- |corr|>0.90 pairs: 3 (full list in feature_correlation_summary.csv)
- heavy-tailed (|x|>10 on >0.1% rows): none
- top yearly drift (|year mean − overall|/std): vol_20 0.67, true_range 0.66, hl_range 0.66, mom_126_5 0.65, vol_60 0.63

## v13_f3

- leakage max diff: 0.00e+00 (PASS)
- features 32 · samples 263,611 (95.3% of close_only — the 252d lookbacks cost warmup history)
- non-finite values: 0 · constant features: none
- |corr|>0.90 pairs: 4 (full list in feature_correlation_summary.csv)
- heavy-tailed (|x|>10 on >0.1% rows): none
- top yearly drift (|year mean − overall|/std): vol_20 0.67, true_range 0.66, hl_range 0.66, mom_126_5 0.65, vol_60 0.63

## Plumbing fits (2 epochs, small model — NOT signal)

- v13_f1: input_dim 13, ok=True, val_ic(2ep) 0.0699, train 29.1s, VRAM 2.16 GB
- v13_f3: input_dim 32, ok=True, val_ic(2ep) 0.18496, train 35.2s, VRAM 3.37 GB
