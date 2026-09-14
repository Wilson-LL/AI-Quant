# 01 — Current production pipeline (data → model → portfolio → execution → holdings)

Companion to `../v17_baseline.md` (parameter freeze). Commit `d2966f0`.

```
TWSE (twstock, monthly STOCK_DAY)            research/refresh_data.py [1/9]
  raw OHLCV, UNADJUSTED for corporate actions  → research/data_cache/<sym>.csv (108 of 112 configured)
        │  gate: newest-date coverage ≥ 99% of cached universe (pipeline_gate refresh)
        ▼
dataset_transformer_eod.build_dataset("close_only", seq_len=60, horizons=(20,))
  per-stock trailing features (10, close-only, clip ±10, no scaler)
  target tgt_rank_20 = per-date pct-rank of close(T+21)/close(T+1) − 1 → [−1,1]
  window admitted iff all 60 rows finite; ETFs excluded; ~247k train / ~28k val samples
        ▼
train_transformer_eod.py --mode daily-retrain [2/9]         (GPU, ~7 min)
  matured labels only (label_end ≤ latest date); chronological 90/10 split, purge 21
  7 seeds × LSTM_CondTransformer preset B; AdamW 3e-4, MSE, early stop on val rank IC
  → checkpoints/transformer_eod/daily_seed0..6.pt + daily_manifest.json
        │  gate: manifest asof == newest cache date
        ▼
inference_transformer_eod.py [3/9]
  latest-date windows for all scoreable names (≥60 else abort); ensemble mean + seed std
  → reports/transformer_gpu/<asof>_predictions.csv (108 rows: stock, score, score_std, sector, vol_20)
  (+ a research-only inverse-vol/band0.05 target_book — NOT the production book)
        │  gate: 4 dated artifacts fresh (mtime after step start)
        ▼
research/blended_decision_book.py [4/9]
  mom126_5 from cache; z_tf, z_mom per date; blend = 0.5 z_tf + 0.5 z_mom
  _book_from_scores: top 20% (k=22), band10 hysteresis vs previous book, equal weight,
  hard 10% name cap, soft 20% sector cap → BUY/HOLD/REDUCE/SELL + WATCH (top 30%)
  → reports/paper_trading/<asof>_blend50_band10_decision_book.csv (+ full-universe scores CSV)
        │  gate: dated book present
        ▼
paper_trading.py snapshot/evaluate [5-6/9] → books/ + PAPER_LEDGER.csv (gross, overlapping)
daily_diff_report.py [7/9] → daily_diffs/DIFF_<asof>.md
user_holdings_overlay.py [8/9] (my_holdings.csv vs book)
user_next_session_plan.py --nightly [9/9]
  holdings (Stage-A schema) × book × universe rank → map_user_action (14-value vocabulary)
  price bands (rank×vol conditional quantiles, TWSE legal domain) → plan CSV/MD
  universe_ranking.py → latest_universe_ranking.{csv,md}
  simplified_reports → latest_next_session_summary.md  (v17: holdings-first, gated)
        ▼
morning_execution_plan.bat (user, ≥09:02)
  intraday collector (Task Scheduler 08:54, supervisor + single-instance lock) → SQLite
  refresh_execution_prices.py: live bid/ask/trade → execution state vs night bands
  → latest_live_execution_summary.md (v17: holdings-first, gated)
        ▼
USER executes manually at next open (validated equivalent to the T+1-close convention)
```

## Where the validated strategy lives vs what the user actually holds

The validated object is the **blend50+band10 long-only book** (22 names, equal-ish weights). Everything after step 4 is presentation. The user's real portfolio (`my_holdings.csv`) is mapped *onto* that book by `map_user_action`; it is never forced to equal it. See `10_holdings_vs_model.md` for how far apart they currently are.

## Two books are produced daily (source of confusion)

`inference_transformer_eod.py` still writes a research `target_book` (inverse-vol weights, band 0.05 of the *universe*) while `blended_decision_book.py` writes the production book (equal weight, band 0.10 of the *book size*). Only the latter matches the backtest. Recommendation (no change made): label the former explicitly as non-production or stop emitting it.
