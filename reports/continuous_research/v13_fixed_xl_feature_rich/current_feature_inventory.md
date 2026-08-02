# v13 — Current Feature & Data Inventory

Read-only inspection, 2026-08-01, branch `research/v13-fixed-xl-feature-rich`.
Sources: dataset_transformer_eod.py, research/data.py, research/refresh_data.py,
train/inference modules, data_cache contents. No code modified.

## 1–3. Existing feature sets (all already implemented in `dataset_transformer_eod.py`)

| feature_set | dim | columns |
|---|---|---|
| **close_only** (production) | 10 | log_ret_1, mom_5, mom_20, mom_60, mom_126_5, vol_20, vol_60, dist_hi_60, dist_lo_60, px_over_ma20 |
| close_d12 | 11 | close_only + xs_d12_rank (per-date pct rank of mom_126_5) |
| close_regime | 14 | close_only + mkt_dd, mkt_vol20, mkt_mom20, breadth (causal market context) |
| close_range | 13 | close_only + hl_range, atr_ratio, close_loc |
| close_liq | 12 | close_only + vol_z, turn_z |
| ohlc_range | 8 | log_ret_1, hl_range, true_range, atr_ratio, close_loc, oc_ret, overnight_gap, range_z |
| volume_block | 6 | log_ret_1, vol_z, turn_z, vol_ma_ratio, log_vol_chg, illiq_z (Amihud-style) |
| sector_rel | 8 | ret_vs_sector, mom20_vs_sector, mom60_vs_sector, vol20_vs_sector + 4 base |
| curated_full | 14 | curated union incl. range, volume, sector-relative |
| full_d12 | 15 | curated_full + xs_d12_rank |

All are **causal by construction** (rolling windows on past data; per-date
cross-sectional transforms; sector fallback to universe mean when n<3;
warmup NaNs excluded by the all-finite-window rule). `turn` is a PROXY
(`close × volume`); the true turnover field is absent offline.

**Prior evidence (critical):** `close_d12`, `close_range`, `close_liq`,
full-field/curated sets and regime features were all screened in v1–v7 at
the PRODUCTION-SIZE model and REJECTED (scoreboard; RESEARCH_LINES_CLOSED
"until real full-field data ~2027-01"). v13's hypothesis is precisely that
a 114k-param model couldn't exploit them but a 1 GB model might — so these
rejections are priors, not blockers, and the small-model screen doubles as
a replication control.

## 4. Targets

`tgt_rank_{5,10,20}` (per-date cross-sectional rank of forward 5/10/20d
returns) built by `build_dataset(horizons=(5,10,20))`; production uses
`tgt_rank_20`. Raw `fwd_{5,10,20}` also carried in panels. No other target
families exist (risk-proxy targets are a CLOSED line).

## 5. Universe

- **110 symbols cached / 108 scored** — hand-curated liquid TWSE large/mid
  caps (`DEFAULT_UNIVERSE` + `MULTI_SECTOR_NEW` in `research/data.py`);
  survivorship-biased by construction.
- ETFs **0050 / 0056 are cached but excluded from the model universe**
  (tagged sector "etf"; usable as market/dividend-style PROXY inputs).
- `SECTOR_MAP` exists for every name (~10 sectors: semis, electronics,
  financials, materials, telecom, panels, optics, biotech, ...).
- No explicit liquidity filter (the universe is curated-liquid); no
  market-cap data.

## 6. Cached raw columns

`research/data_cache/<sym>.csv`: **date, open, high, low, close, volume**
— nothing else. NOT present: true turnover value, transaction count,
market cap, margin/short balances, institutional or foreign flows,
futures, fundamentals. `research/data_cache_full/` has been accumulating
**true turnover + transaction count** since ~2026-07 via
`refresh_data.py --full-fields`; it reaches useful history ~2027-01 (the
pre-authorized E2 revisit).

## 7. Feature families derivable from the EXISTING cache (no new data)

Tiers 1–6 of the v13 design are all derivable today:
OHLCV/range/gap/true-range (Tier 1 — mostly already coded in
`ohlc_range`); multi-horizon momentum/reversal/vol/drawdown/MA-distance/
volume z (Tier 2 — mostly coded); per-date z-scores/ranks, sector-relative,
vol- and liquidity-adjusted momentum (Tier 3 — partially coded:
`xs_d12_rank`, `sector_rel`); market index (equal-weight or 0050 close),
market vol, breadth, sector momentum (Tier 4 — coded in `close_regime`
except sector momentum); ADV/dollar-volume, Amihud illiquidity, volume
shock, liquidity rank (Tier 5 — partially coded: `illiq_z`, `vol_z`;
`adv_frame()` exists in queue_v9_lib); downside momentum, squeeze/rally
flags, vol expansion (Tier 6 — formulas exist in short_side_v12.py, not
yet as model features).

## 8. Families requiring NEW data (not available locally)

Tier 7 entirely: margin trading balance, short interest/lending,
institutional/foreign/dealer flows, futures context, financial statements,
revenue, earnings revisions, valuations, dividends, analyst data. TWSE/TPEX
publish several of these daily (margin, institutional flows) — acquiring
them means extending `refresh_data.py` with new endpoints + a new cache
directory, which is a SEPARATE approval (no scraping without it). Nearest
real enrichment: true turnover/transactions in `data_cache_full`, usable
~2027-01.

## Infrastructure notes for v13

- `build_dataset(feature_set=...)` handles any registered set; adding new
  sets = extending `_stock_features`/`FEATURE_COLS`/`_panel_xs`
  (research-only additions to the shared module — flag-gated by the
  feature_set key itself, absent from production paths, anchor-verified).
- Input width flows through `build_net` (`input_dim = len(FEATURE_COLS[fs])`)
  into the LSTM and `cross_kv_proj` only — for XL2 (h1024/L24) going from
  10 → ~44 features adds ≈174k params (≈0.7 MB on 1.19 GB): **not a
  bottleneck**; pos-emb/cross-attention logic is width-agnostic.
- X tensor VRAM scales with feature count: 276k × 60 × F × 4B ≈ 0.66 GB at
  F=10 → ≈2.9 GB at F=44 (fp32, GPU-resident). Fits alongside XL2's ~6.5 GB
  training footprint; fp16 X is the fallback lever.
- Checkpoint save/load and inference compatibility are width-agnostic
  (cfg + feature_set stored in the checkpoint; `build_net_from_ck` sizes
  input_dim from `FEATURE_COLS[feature_set]`).
