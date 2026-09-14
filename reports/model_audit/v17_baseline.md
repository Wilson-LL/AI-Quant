# v17 Baseline Freeze — production system as of 2026-09-14

Documentation only. Accepted production commit: **`d2966f0`** (main; feat:
add full-universe priority ranking), parent lineage 8f5186e ← aa1f79d ←
e49c81e ← 7999d91. Live model manifest asof **2026-09-11** (preset B,
7 seeds, `tgt_rank_20`, horizon 20). Every fact below cites its source.

## Model

| Item | Value | Source |
|---|---|---|
| Architecture | `LSTM_CondTransformer`: LSTM(64, 1 layer) → Linear(64→64) + learned positional embedding → cross-attention (4 heads; query = LSTM states, key/value = Linear(raw inputs)) with residual → `TransformerEncoder`(2 layers, d=64, 4 heads, FFN 128, dropout 0.2, post-LN) → last-timestep readout → Linear(64→32)-ReLU-Dropout(0.2)-Linear(32→1) | `model.py:4-71` |
| Parameters | ≈114k (470 KB checkpoints) | `checkpoints/transformer_eod/daily_seed*.pt` |
| Preset | **B**: hidden 64, trans_layers 2, lstm_layers 1, heads 4, ff 128, dropout 0.2, seq_len 60, max_epochs 25, patience 3 | `train_transformer_eod.py:42-43` |
| Sequence length | **60** sessions, window T−59..T inclusive, readout at T | `dataset_transformer_eod.py:409-419`, `model.py:69` |
| Feature set | **close_only, 10 features**, all per-stock trailing, close-price-only: log_ret_1, mom_5, mom_20, mom_60, mom_126_5, vol_20, vol_60, dist_hi_60, dist_lo_60, px_over_ma20; hard clip ±10; **no scaler, no z-score, no cross-sectional transform** | `dataset_transformer_eod.py:53-64, 222-223, 378` |
| Target | **tgt_rank_20** = per-date percentile rank of `fwd_20 = close(T+21)/close(T+1) − 1` mapped to [−1, 1]; NaN when < 30 names/date | `dataset_transformer_eod.py:260-266, 330-337` |
| Horizon / exec lag | 20 sessions / 1 (enter at T+1 close in the label) | same |
| Training window | all matured history from cache start (2015-01 for 105/108 names), equal weight, no recency | `train_transformer_eod.py:535, 549`; `dataset_transformer_eod.py:501` |
| Validation | most recent ~10% of matured dates (~1 year, ~260 dates); purge 21 dates (horizon+lag) between train and val; **val is never trained on** | `dataset_transformer_eod.py:480-512` |
| Early stopping | argmax of per-date Spearman val IC; patience 3, min epochs 2; live seeds stop at 4–7 epochs | `train_transformer_eod.py:366-380` |
| Optimizer | AdamW lr 3e-4 constant (no schedule), weight decay 1e-4, grad-clip 1.0, batch 1024, AMP | `train_transformer_eod.py:217-220, 254-255, 325, 361` |
| Loss | MSE on the rank target (weighted branch with all-ones weights in production) | `train_transformer_eod.py:356-358` |
| Ensemble | **7 seeds (0–6)**, arithmetic mean of raw scores; `score_std` = seed dispersion | `train_transformer_eod.py:553`; `inference_transformer_eod.py:181-182` |
| Retrain cadence | **full refit from scratch every session** (daily_ops step 2) | `daily_ops.bat` |
| Determinism | torch/np seeds only; no cuDNN determinism flags | `train_transformer_eod.py:229-230` |

## Decision layer

| Item | Value | Source |
|---|---|---|
| Blend | blend50 = 0.5·z(tf score) + 0.5·z(mom); z per date over the merged cross-section | `blended_decision_book.py:51-53` |
| D1.2 component | mom126_5 = close(T−5)/close(T−131) − 1 (126-session momentum, 5-session skip); needs ≥132 rows | `blended_decision_book.py:49` |
| top_frac | 0.20 → k = max(3, round(0.2·n)) = 22 of 108 | `paper_trading.py:39, 49` |
| band10 | hysteresis: entry needs rank ≤ k; exit only when rank > int(k·1.2) = 26; incumbents keep slots in previous-book order | `paper_trading.py:44-60` |
| Weights | equal → `cap_weights`: hard 10% name cap (water-fill), soft 20% sector cap | `transformer_portfolio.py:22-81` |
| WATCH | universe blend rank ≤ int(0.3·n) = 32 and not already emitted | `blended_decision_book.py:70-89` |
| Portfolio size | ~22 names, long-only in production (L/S only in research metrics) | ledger `n_names` mean 21.74 |
| Costs (research) | `net60` = 60 bps round trip = **30 bps per side** on one-way L1 turnover; L/S shorts charged the same flat rate, no borrow cost | `transformer_portfolio.py:161-163` |
| Execution convention | label/backtest: T+1 close → T+21 close; user executes at **T+1 open** — NEXT_OPEN_TIMING_VALIDATED (retention 1.005 CH L/S, 1.014 BR) | `v16_next_session_execution/next_open_execution_audit.md` |
| Price bands | conditional next-session quantiles by rank×vol cell (MIN_CELL_OBS 400 / MIN_POOL 750), fallback RANK×VOL→VOL→GLOBAL; ATR guardrails K_WIDTH .25 / K_RISK 1.5 / K_PANIC 1.0 / K_HOLD 1.0 (untuned); tick-rounded; clamped to TWSE ±10% legal domain (NORMAL_DAY_ASSUMPTION) | `execution_price_bands.py:18-34`, `twse_price_domain.py` |
| Holdings mapping | `holdings.map_user_action` 24-rule table; USER_ACTIONS 14 values; no OPEN_SHORT; ALIGN_REL .25 / ALIGN_FLOOR 2pp / REDUCE_ENTRY_MIN 4% | `holdings.py:41-47, 227-363` |
| Publication gate | PARTIAL_COVERAGE_MIN = 0.99 over the **cached** universe (108/108 pass, 107 pass, 106 block); pipeline_gate stages refresh/retrain/inference/book with exit-code + fresh-mtime contract | `user_next_session_plan.py:809`; `pipeline_gate.py` |
| Universe | configured 112 (SECTOR_MAP, hand-curated 2026-07-06 snapshot) → model-eligible 110 (non-ETF) → scored 108 (2809/2888 never cached) | `data.py:64-124`; `dataset_transformer_eod.py:289` |
| Ranking layer | blend50 full cross-section exported by the book step; `universe_ranking.py` ranks all 108, strength bands ≤.10/.20/.50/.80, agreement = seed-std terciles, 5 holdings-first priority tiers | `research/universe_ranking.py` |

## Standing research references (frozen panels, end 2026-07-23)

| Window | Panel | blend50+band10 L/S net60 | L/O net60 | n rebalances |
|---|---|---|---|---|
| CH 2023-01→ | SCHED_A8_seeds7_full | 2.147 (seed-set 0–6; disjoint seeds 1.843) | 1.989 | 42 |
| BR 2021-01→ | SCHED_BEAR_A8_seeds7_full | 1.443 (disjoint 1.301) | 1.455 | 67 |

Bootstrap (200×, drop-20% names): CH p5 1.61 / p50 1.92 / p95 2.17; BR p5 1.14 / p50 1.37 / p95 1.54. **Planning numbers are the medians (1.92 / 1.37)**; the point estimates are one seed draw and sit near the bootstrap p95.

## Validation protocol actually implemented

Chronological walk-forward, `oos_start` 2023-01-01 (CH) or 2021-01-01 (BR), refit every 126 sessions (~6 months), matured labels only, purge 21 (not seq_len+horizon as METHODS.md states), selection by val IC at screen level, adoption by OOS book metrics on the same windows (see 05_validation_methodology.md for the consequences).
