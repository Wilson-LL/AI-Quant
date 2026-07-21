# RTX 4060 Ti Sprint Plan — Transformer Daily-Retrain (20h)

Branch: `research/transformer-4060ti-daily-retrain-20h` · Start: 2026-07-22 00:42

## 0. Reality check vs the brief

The brief references prior sprint artifacts (GPU_RUNBOOK.md, TRANSFORMER_*_REPORT.md,
dataset_transformer_eod.py, train_transformer_eod.py, inference_transformer_eod.py,
inference_transformer_gpu.py, research/transformer_portfolio.py, research/refresh_data.py,
research/daily_inference.py, docs/transformer_daily/) and branches
(`research/transformer-gpu-confirmation-20h` etc.). **None of these exist in this
repository.** What exists:

### 1. What was already built
- `model.py` — `LSTM_CondTransformer`: LSTM → projection + positional embedding →
  **cross-attention (query = LSTM states, key/value = projected raw input)** →
  Transformer encoder → MLP head. Exactly the required architecture.
- `train.py` / `inference.py` / `dataset.py` — original barrier-target (+12%/−6%/20d)
  pipeline over live twstock fetches; chronological split with purge already fixed.
- `research/` — the D1.1 momentum research stack: cached OHLCV panel (119 TWSE stocks,
  2018-01-02 → 2026-07-07, `date,open,high,low,close,volume` + SECTOR_MAP), causal
  features, targets registry, IC/walk-forward/backtest evaluation framework, momentum
  baseline (D1.2 lineage), D1.1 portfolio construction with inverse-vol + 10% name cap
  + soft sector cap. D1.1 validated: L/S net@60 Sharpe 1.27 full-sample, OOS 1.51,
  realistic deployable ~0.8–1.0.

### 2. What was only CPU-tested / not tested
- The transformer EOD daily-retrain pipeline itself **does not exist and must be built
  this sprint** (this is the main work, reusing model.py + research/ as required).
- Original train.py was written for CUDA but with no AMP, no recency weighting, no
  cross-sectional target, no matured-label logic, no portfolio layer.

### 3. What must be confirmed on the RTX 4060 Ti
Confirmed already (G0 partial): CUDA works, AMP stable at 10.6× speedup, epoch @200k
samples ≈ 17 s, 119-stock inference ≈ 6 ms, peak VRAM 317 MB. Remaining: end-to-end
train → inference → decision book on real data, and the full experiment queue.

### 4. Scripts ready to reuse as-is
`model.py`, `research/data.py`, `research/features.py`, `research/targets.py`,
`research/evaluation.py`, `research/momentum.py`, `research/portfolio_d1.py`,
`research/d1_1_pername_cap.py`.

### 5. Scripts to build (new) or modify
| file | action |
|---|---|
| `dataset_transformer_eod.py` | NEW — panel → sequence samples, feature sets, cross-sectional targets, matured-label masking, recency weights, purged chronological splits |
| `train_transformer_eod.py` | NEW — AMP GPU trainer (reuses `LSTM_CondTransformer` + train.py patterns: AdamW, grad clip, early stop, checkpointing), weighted per-sample loss, seed ensemble, walk-forward runner with retrain cadence, `--mode daily-retrain` |
| `inference_transformer_eod.py` | NEW — load ensemble → predict all stocks → decision book CSV/MD with hard 10% cap |
| `research/transformer_portfolio.py` | NEW — portfolio constructions + backtest of score panels (reuses D1.1 cap/inverse-vol logic) |
| `research/refresh_data.py` | NEW — incremental cache refresh via twstock (network-optional; sprint runs on the frozen 2026-07-06 snapshot for reproducibility) |
| `model.py`, `train.py`, `inference.py`, `dataset.py` | UNTOUCHED (production; per PRODUCTION_EDIT_PLAN.md) |

## Data honesty — "full field"

The reproducible frozen cache holds OHLCV only. twstock also serves `turnover`,
`transaction`, `change`, but a full-history refetch is ~12k rate-limited requests
(hours, ban risk) and would break snapshot reproducibility. Therefore:
- `change` = exact (derivable from close). `turnover` ≈ close×volume (tight proxy).
- `transaction` (trade count) and avg-trade-size features are **unavailable offline**;
  documented as such. `refresh_data.py` supports fetching true full fields incrementally
  for future use.
- Sector comes from SECTOR_MAP → sector-relative features are fully available.

## Evaluation protocol (fixed before running)

- Universe: 117 non-ETF cached names (0050/0056 reserved as market proxy).
- **Screening protocol** (G1/G2/G4/G5): walk-forward, refit every 126 trading days,
  OOS 2023-01 → 2026-07 (7 refits). At each refit date t0: train only on samples whose
  **label window fully matures ≤ t0**; val = last 10% of train dates (purged) for early
  stopping + seed averaging; predict daily scores for the next 126 days. 2 seeds for
  screening, 3–5 for champions.
- Purge/embargo: `seq_len + horizon` samples at every boundary.
- Scaling: features are per-day causal ratios (scale-free) + per-date cross-sectional
  z-scores — no full-sample scaler anywhere.
- Portfolio metrics: 20d-hold top/bottom-quintile L/S and long-only top-quintile,
  net at 0/60/100/150 bps; IC, rank IC, turnover, max DD, yearly breakdown.
- **Cadence protocol** (G3): champion config, OOS last ~2y; frozen / quarterly /
  monthly(20d) / weekly(5d, warm-start) / daily(warm-start) + daily-from-scratch
  feasibility on a 1-month sample, extrapolated.
- Model selection: val rank IC only. No OOS-driven tuning; all runs reported.

## 6. Experiment queue

| # | experiment | configs × fits | est. GPU time |
|---|---|---|---|
| G0 | CUDA readiness + real-data smoke train/inference/book | 1 | 15 min |
| G1 | Close-only Preset A, seq 40 vs 60, 20d rank target, HL126 | 2 | ~1.0 h |
| G2 | Recency: equal / roll 3y/1.5y/1y / HL 63/126/252/378 / 5y-cap+HL126 (+6m fine-tune if time) | 9 | ~2.5 h |
| G3 | Cadence: frozen/quarterly/monthly/weekly-warm/daily-warm (+scratch sample) | 5 | ~3 h |
| G4 | Targets: 5d/10d/20d rank, 20d excess-vs-universe, vs-sector, barrier diagnostic | 6 | ~2 h |
| G5 | Features: close / close+D1.2 / OHLC-range / volume-block / sector-rel / curated full / full+D1.2 | 7 | ~2 h |
| G6 | Hybrid: standalone vs D1.2-filter vs rank-blend vs sleeve | CPU | ~0.5 h |
| G7 | Portfolio: top-N/quintile, L/S, rank-wt, inv-vol, caps, bands, 5/10/20d holds | CPU | ~1 h |
| G8 | Daily production workflow end-to-end + budget timing | 1 | 0.5 h |

GPU total ≈ 12–13 h, run in background while docs/analysis proceed. Presets: A (h64,
seq40), B (h64, seq60, 5 seeds), C (h128, seq60) — C only if time allows.

## 7–8. Expected runtime / outputs
Each experiment writes JSON + MD under `reports/transformer_gpu/` (git-kept, small);
checkpoints under `checkpoints/transformer_eod/` (git-ignored). Daily workflow outputs
per prompt §9 paths. Heartbeat every ~30 min to RUN_STATUS_TRANSFORMER_4060TI_20H.md.

## 9. Acceptance bar
As per brief §12: beat D1.2 (net@60 L/S Sharpe ~1.27 full / ~1.5 OOS window, realistic
0.8–1.0) OOS after costs, competitive @100bps, no material DD worsening, stable walk-
forward, sane turnover, hard 10% cap, feasible ≤12h daily cycle, no leakage. Otherwise
classify as sleeve/filter/research per brief.

## 10. Rollback plan
All new code is additive (new files); production train/inference/dataset/model
untouched. Rollback = delete new files / reset branch; protected branches never
touched; nothing pushed.
