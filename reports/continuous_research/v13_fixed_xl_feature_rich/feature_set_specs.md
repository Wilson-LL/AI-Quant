# v13 Feature Set Specs (Task 4)

Implemented in `dataset_transformer_eod.py` as feature_set keys
`v13_f1 … v13_f6` (additive, research-only; production `close_only` path
anchor-verified bit-identical after the edit). All features are causal:
rolling windows over data ≤ t; per-date cross-sectional transforms use only
same-date symbols; sector stats fall back to the universe mean when a
sector has <3 names that day. NaN handling: the existing all-finite-window
rule — a sample enters the dataset only if every feature in its seq_len
window is finite (252d lookbacks push warmup to ~383 trading days; sample
counts reported in the quality report). Scaling: features are ratio/z/rank
constructions (no raw levels except none); `xs_voladj_mom_z` clipped ±5.
Targets are NEVER inputs (asserted in the quality battery).

## F1 — OHLCV-derived (13)
log_ret_1=ln(c/c₋₁) · oc_ret=c/o−1 · overnight_gap=o/c₋₁−1 ·
hl_range=(h−l)/c · true_range=TR/c · atr_ratio=TR/c over its 20d mean −1 ·
close_loc=(c−l)/(h−l)−0.5 · range_z=20d z of hl_range ·
vol_z=20d z of volume · turn_z=20d z of turnover-proxy (c×v) ·
vol_ma_ratio=v/20d-mean −1 · log_vol_chg=ln((v+1)/(v₋₁+1)) ·
illiq_z=252d z of |ret|/turnover (Amihud).

## F2 — + multi-horizon technicals (27)
mom_{5,10,20,60}=pct_change · mom_126_5=c₋₅/c₋₁₃₁−1 · mom_252 ·
vol_{5,20,60}=rolling std of log_ret_1 · dist_hi_60 / dist_lo_60 =
c/rolling max/min(60)−1 · px_over_ma20 / px_over_ma60 ·
dd_252=c/rolling max(252, min 60)−1.

## F3 — + cross-sectional per-date (32)
xs_mom20_rank, xs_d12_rank, xs_vol20_rank, xs_adv_rank = per-date pct
rank −0.5 of (mom_20, mom_126_5, vol_20, log ADV20) ·
xs_voladj_mom_z = per-date z of mom_126_5/(vol_60+ε), clipped ±5.
(adv20l=ln(20d median c×v +1) is a helper level, ranked only — not an input.)

## F4 — + sector-relative (37)
ret_vs_sector, mom20_vs_sector, mom60_vs_sector = value − same-date sector
mean (universe-mean fallback, n<3) · vol20_vs_sector · sec_mom20 = sector
mean mom_20 broadcast.

## F5 — + liquidity/execution (40)
adv_shock = turn/20d-median −1 · adv_trend = 20d/60d median turnover −1 ·
vol_z60 = 60d volume z.

## F6 — + short-side risk (44)
down_mom_20 = min(mom_20, 0) · squeeze_flag = 1 if 20d return > +30% ·
vol_expand = vol_5/vol_60 −1 · maxret_20 = max daily log ret in 20d.

## Memory impact (X tensor, 276k samples × seq60, fp32)
F0 0.66 GB · F1 0.86 · F2 1.79 · F3 2.12 · F4 2.45 · F5 2.65 · F6 2.91 GB
— GPU-resident alongside the model; fp16 X is the fallback lever for XL
runs. Inference memory scales identically; checkpoint size impact <1 MB
even at XL2 (input width touches only the LSTM input matrix and
cross_kv_proj).
