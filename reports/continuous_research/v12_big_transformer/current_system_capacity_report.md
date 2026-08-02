# v12 — Current System Capacity Report

Read-only inspection, 2026-07-30, branch `research/v12-big-transformer-recency-short`.
Sources: model.py, train_transformer_eod.py, dataset_transformer_eod.py,
inference_transformer_eod.py, research/{gpu_research_scheduler,run_queue_v11,
blended_decision_book,paper_trading,transformer_portfolio,refresh_data}.py,
plus measured numbers from the v11 queue (2026-07-30). No code modified.

## 1–3. Parameters and checkpoint

- **Total parameters: 113,857** (exact, instantiated) — preset B, input_dim 10.
- **Checkpoint size: 0.45 MB fp32** per seed (`daily_seed{0..6}.pt`, measured);
  7-seed ensemble ≈ 3.2 MB total.
- Module split (exact): LSTM 19,456 · input_proj 4,160 · pos_emb 3,840 ·
  cross-attn (+kv proj) 17,344 · TransformerEncoder×2 66,944 · MLP head 2,113.

## 4–5. Memory (measured, v11)

- Training: **2.0 GB peak allocated / 2.7 GB reserved** at batch 1024 (7-seed
  walkforward runs: ~2.16 GB). Whole dataset lives GPU-resident
  (X tensor 664 MB, `to_gpu()`); there is **no DataLoader** — batches are
  index-gathers on GPU, so pin_memory/num_workers are not applicable.
- Inference: ~1 GB-scale; scores 108 stocks in seconds (`predict_idx`,
  batch 8192).

## 6. Epochs / early stopping

Preset B: `max_epochs=25`, `patience=3` on **val rank-IC**, `min_epochs=2`;
best-IC state restored at the end (best-by-val, in memory). Learning curves
(`hist`: per-epoch train_loss/val_ic) exist inside `fit_one` but per-epoch
curves are not persisted — v12 will log them.

## 7–8. Split and window logic

- Research walkforward: refit every 126 trading days from `oos_start`; at
  each refit, train = all matured samples before the refit date, val = last
  slice (val-rank-IC early stop), OOS = the next 126-day block. Scores are
  strictly out-of-sample.
- Daily retrain: `matured_train_val()` — train on all matured labels to the
  latest date, small recent val slice.
- **Training window: ALL available history (2015→)** in both paths.

## 9–12. History weighting

- Equal-weight full history is the champion default (recency weighting was
  tested and REJECTED in the early sprints — "recency" appears as a closed
  line; `mode_daily_retrain(recency=None)` comment: "recency weighting hurt
  OOS").
- **Sample-weight machinery EXISTS**: `fit_one(weights=...)` takes
  per-sample weights (normalized to mean 1), plumbed via
  `matured_train_val(recency=...)` — so v12's recency research needs NO
  training-module changes, only research-side weight construction.
- **No hard date cutoff exists** anywhere in the training path.

## 13–14. Checkpoints and AMP

- Save: `checkpoints/transformer_eod/daily_seed{s}.pt` + `daily_manifest.json`
  (production, will NOT be touched). Load: `inference_transformer_eod.
  load_ensemble()` rebuilds nets from the saved cfg — any preset shape
  round-trips (verified again in v11 smoke).
- **AMP: yes, mandatory-on for CUDA** (`torch.amp.autocast` + `GradScaler`).

## 15–17. Not present

- **Gradient checkpointing: none.** `nn.TransformerEncoder` used plainly.
- **Gradient accumulation: none.** One optimizer step per batch.
- **Multi-GPU: none.** Single RTX 4060 Ti 16 GB; no DDP/DataParallel.

These three are exactly what v12's runner must add (research-only) to make
XL training feasible.

## 18. Inference modes

Single deterministic pass per seed; 7-seed mean + std (`score_std` feeds the
confidence terciles). **No MC-dropout, no multi-snapshot, no perturbation /
robustness scoring, no repeated-pass uncertainty** — all v12-new.

## 19. Short-side evaluation

`transformer_portfolio.backtest_scores(mode=...)` already supports
`long_short` (equal-weight top vs bottom quintile, costs, band, caps) and
`long_only`; every standing reference (2.147/1.443) IS a long-short net60
number. **What does NOT exist:** short-leg-only attribution (return / hit
rate / DD of the short side alone), borrow/liquidity/squeeze proxies,
asymmetric gross configurations (100/50 etc.), net-exposure control,
short-name caps, or a short-only diagnostic book. All v12-new, research-only.

## 20. What can be researched without touching production

| axis | mechanism | production edit needed? |
|---|---|---|
| model size | preset `overrides` (derived presets, v11 mechanism) | no |
| recency weights / windows | `fit_one(weights=...)` + research-side weight builders | no |
| epochs / patience / min_epochs | preset keys (`max_epochs`, `patience`) + `fit_one(max_epochs=, min_epochs=)` | no (preset keys exist) |
| micro-batch / accumulation / grad-ckpt | v12 runner wraps its own fit loop OR flag-gated cfg keys default-OFF | small flag-gated edits only if needed |
| long-short configs | research evaluators on saved panels (CPU) | no |
| deep inference | new research script reading v12 checkpoints | no |

Production surface (daily retrain defaults, `checkpoints/transformer_eod/`,
daily_manifest, blend book, paper ledger, holdings overlay, data_cache)
requires zero changes for the entire v12 program.
