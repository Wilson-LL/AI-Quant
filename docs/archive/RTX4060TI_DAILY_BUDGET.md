# RTX 4060 Ti Daily Workflow Budget (measured, 2026-07-22 sprint)

Hardware: RTX 4060 Ti 16 GB (Ada, CC 8.9), 16-core CPU, 34 GB RAM, torch 2.13
nightly cu132, AMP mixed precision. All timings measured on the real 108-stock
frozen-cache universe with the champion configuration (close-only features,
seq_len 60, LSTM-Transformer h64 with cross-attention, 5-seed ensemble,
equal-weight full history, 20d cross-sectional rank target).

## Measured daily cycle

| step | command | measured time |
|---|---|---|
| 1. Data collect (incremental) | `python research/refresh_data.py` | ~2–6 min network-bound (1 req/stock-month, throttled 1.5 s; typically 1 month/stock) |
| 2. Daily retrain (5 seeds, matured labels only) | `python train_transformer_eod.py --mode daily-retrain` | **134.6 s** total (incl. ~35 s dataset build; 22–33 s per seed fit; peak VRAM 2.0 GB) |
| 3. Inference + decision book (108 stocks) | `python inference_transformer_eod.py` | **~40 s** total (dataset rebuild ~35 s; model inference 1.8 s for 5 seeds) |

**Total ≈ 5–10 minutes per day, vs a 12-hour budget → ~70× headroom.**

## Preset costs (walk-forward refit, equal-all history, measured in G9)

| preset | params | seq | seeds | mean fit | daily retrain est | inference (108 names) | peak VRAM | fits 12h? |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| A (h64) | 112,577 | 40 | 3 | 12.7 s | 38 s | 2.8 ms | 1.3 GB | yes |
| **B (h64) champion** | 113,857 | 60 | 5 | 23.0 s | 115 s | 3.1 ms | 2.3 GB | yes |
| C (h128) | 432,449 | 60 | 3 | 41.0 s | 123 s | 4.6 ms | 3.9 GB | yes |

Even preset C at daily cadence uses <0.5% of the 12 h budget. The binding cost
of the whole workflow is the throttled TWSE fetch, not the GPU.

## Notes

- fp32 is ~10× slower than AMP on this torch nightly — always run AMP
  (see RTX4060TI_ENVIRONMENT_CHECK.md).
- Retrain cadence finding (G3): daily/weekly warm-start retraining HURT OOS
  performance; frozen/monthly full retrains performed best. The recommended
  production cadence is **monthly full retrain** (5 seeds, ~2 min) with daily
  collect + inference; daily retrain remains feasible within budget if desired,
  but is not supported by the evidence.
