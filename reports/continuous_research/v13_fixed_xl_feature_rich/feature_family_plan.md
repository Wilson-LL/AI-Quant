# v13 Feature Family Plan (Task 2)

Tiers map onto staged sets F0–F7 (exact columns in feature_set_specs.md).
All of Tiers 1–6 are derivable from the existing OHLCV cache (turnover =
close×volume PROXY until data_cache_full matures ~2027-01). Tier 7 is
external-only and requires a separate data-acquisition approval — nothing
is scraped or downloaded in v13.

| Tier | Family | v13 coverage | Notes |
|---|---|---|---|
| 0 | close_only baseline | F0 | production control |
| 1 | OHLCV price/volume | F1 (13) | oc/gap/range/true-range/close-loc/volume z/illiquidity — mostly pre-existing `ohlc_range`+`volume_block` code |
| 2 | multi-horizon technicals | F2 (+14) | momentum 5–252, vol 5/20/60, rolling DD 252, MA distances; reversal horizons are the short-momentum columns (sign learned by the model) |
| 3 | cross-sectional normalized | F3 (+5) | per-date pct-ranks (mom20, d12, vol20, liquidity) + vol-adjusted momentum z |
| 4 | market/sector context | F4 (+5) | sector-relative ret/mom/vol + sector momentum (SECTOR_MAP exists); market-level context (mkt_dd/breadth) exists as `close_regime` — regime features are a CLOSED line at small scale, folded in only via the sector channel here |
| 5 | liquidity/execution | F5 (+3) | ADV shock/trend, 60d volume z (Amihud illiq_z already in F1) |
| 6 | short-side/risk | F6 (+4) | downside momentum, squeeze flag (+30%/20d), vol expansion, max daily ret 20d |
| 7 | external (margin, flows, fundamentals…) | F7 — NOT BUILT | no local data; acquisition plan = separate proposal |

Prior evidence honestly stated: richer sets at PRODUCTION size were
rejected in v1–v7 (close_range/close_liq/close_d12/curated_full/regime).
v13 tests whether that verdict was a capacity limit. The small-model screen
(Phase 2) therefore doubles as a replication control, and the fixed-XL
screen (Phase 3) is the actual hypothesis test.
