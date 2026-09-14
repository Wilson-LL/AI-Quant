# 02 — Data integrity & leakage audit (B1, B4, B5)

Method: code audit with file:line evidence **plus mechanical tests**
(`tests/test_leakage_alignment.py`, 5 tests, all pass) that perturb
synthetic data and assert invariance.

## Proven clean (mechanically)

| Property | Proof |
|---|---|
| Every production feature at T is invariant to prices after T | `test_close_only_features_ignore_future`: scrambling all closes/volumes after T leaves all 10 `close_only` features at ≤T bit-identical; sanity test confirms features do respond to past changes |
| Target at T uses only closes in (T+1 … T+21] | `test_fwd_ret_window_is_t_plus_1_to_t_plus_21`: changing close(T) or anything after T+21 leaves the label unchanged; last 21 labels immature (NaN) |
| Train/val purge ≥ horizon+lag, no overlap, OOS untouched | `test_purge_and_maturity`, `test_no_train_val_overlap`: max train label-end < min val feature date; gap ≥ 21; both ≤ refit date |
| No full-sample scaler | grep + read: only `np.clip(±10)`, data-independent (`dataset_transformer_eod.py:378`) |
| Cross-sectional transforms are per-date only; none used by close_only | `dataset_transformer_eod.py:129-140, 205` |
| Early stopping never sees the OOS block | maturity + purge arithmetic in `walkforward` (`train_transformer_eod.py:420-426, 468`) |
| No same-bar execution; panel `fwd_h` == cache lag-1 to 1.4e-7 | queue_v9 X3 and v16 Task 2 cross-checks |

## Findings (ranked)

**R1 — Prices are unadjusted for corporate actions (🔴 material).** `data.py:164-171` stores raw twstock closes; no adjustment code exists anywhere. Consequences: (a) `fwd_20` labels spanning an ex-dividend date (TWSE: concentrated Jun–Aug) are biased down by the yield → the cross-sectional rank target systematically penalizes high-yield names each Q3 — a learnable but non-tradeable pattern; (b) backtest long returns exclude dividends while including ex-div drops (understates long-only, overstates short-leg); (c) stock dividends/capital reductions inject discrete jumps into log_ret_1/mom/vol features. **Not a look-ahead leak; a label-quality and attribution bias.** Research-only fix path: build an adjustment-factor table (needs a dividend/corporate-action source) and re-run the frozen panels — see hypothesis H1.

**R2 — The most recent ~1 year is never trained on (🔴 for a "daily-adaptive" system).** `matured_train_val` reserves the newest 10% of matured dates (~260 sessions) for early stopping only; the 21-day purge is discarded entirely. Every daily retrain therefore learns from data ending ~13 months before today and selects its epoch on an almost-constant one-year set. Combined with R6 this means the daily retrain buys little adaptation. Research question H2 (train-on-all + fixed epoch budget, or rolling val).

**R3 — Reported OOS windows double as selection sets (🟠).** ~111 panels / ~20 challengers evaluated on the same 2023→ window; scheduler gates on OOS Sharpe (`gpu_research_scheduler.py:200-206`); A8 adoption cited OOS numbers. The "two-window consistency" rule uses windows that overlap 64%. Consequence: 2.147 is a maximum over trials, not an unbiased estimate (see 05).

**R4 — n = 42 rebalances → Sharpe SE ≈ ±0.6 annualized (🟠).** See 05/bootstrap.

**R5 — Cost convention (🟠).** net60 = 30 bps/side; docs say "one-way 60"; shorts charged like longs with no borrow/SBL model → L/S numbers are not implementable; **long-only 1.989 (CH) / 1.455 (BR) are the honest references**. See 08.

**R6 — Production retrain cadence never validated (🟡).** All panels use refit-every-126; the only cadence study (G3) found weekly/daily *warm* retrains strongly negative (−2.6 / −1.3 Sharpe) and daily *full* refit was never run. Production runs a daily full refit. Verified (tests/test_seed_reproducibility.py): production retrains the SAME fixed seed IDs 0–6 from scratch each session (torch + numpy seeded per fit; batch order from the seeded torch RNG; no cuDNN determinism flags, so CUDA training is not bitwise reproducible) — so weights change as the expanding dataset changes and through non-deterministic CUDA kernels, not because new seeds are drawn. Given R2, the newest data enters only the holdout, so most nightly weight movement is not new information; the band absorbs some of the resulting rank jitter.

**R7 — Two `band` conventions** (fraction of book size vs fraction of universe) and band applied to the long leg only in L/S backtests. Presentation/consistency issue; production long-only book uses the validated convention.

**R8 — METHODS.md overstates the purge** (claims seq_len+horizon=81; code uses 21, which is sufficient). Doc fix.

**R9 — Feature/target/recency choices trace to 2-seed screens** the repo itself later declared invalid; G5 actually ranked close_d12/sector_rel above close_only at 2 seeds. Later 5–7-seed confirmations of close_only stand, so this is provenance, not a live defect.

**R10 — hygiene**: no-op chronology assertion (`dataset_transformer_eod.py:510`); `score_std` unbiased/biased mismatch inference vs walk-forward; unclipped target path in production; no cuDNN determinism (seeds not bit-reproducible); halted-name horizon drift (label counts 20 own rows, not 20 calendar sessions).

## P1 — corporate-action distortion, quantified from the cache (`audit_v17_corporate_actions.py`)

No dividend/split table exists locally (twstock raw; `data_cache_full`
has OHLCV/turnover only), so events were *detected* as symbol-days whose
close-to-close return trails the equal-weight universe by > 4 pp. **Every
such print is a DETECTED CORPORATE-ACTION CANDIDATE, not a CONFIRMED
corporate action** — ordinary idiosyncratic crashes are included.
Seasonality is the tell: candidate prints per session run 1.1–1.8 in
Oct–May but **2.1 (Jun), 3.3 (Jul), 2.6 (Aug)** — the TWSE ex-dividend
season. Seasonal excess ≈ **0.64 events per stock-year** (≈ 830 events
over 12 years × 108 names); median residual drop on those days **−5.4%**
(p25 −6.8%) against a 20-day cross-sectional return std of 8.5% → a
**≈ 0.6 σ downward shift of the label** for an affected name.
Share of 21-session label windows containing a candidate print: 24%
overall (30% in Jun–Sep) — an upper bound that includes ordinary crashes;
the seasonal-excess share plausibly attributable to corporate actions is an
**ESTIMATE from the candidate detector of ≈ 5–8% of all windows**,
concentrated in Q3 and in high-yield names. Exact attribution requires a
true point-in-time corporate-action table.

Features: `log_ret_1`/`mom_5` are hit for 1–5 sessions after an event;
`mom_20/60/126_5`, `dist_hi_60`, `px_over_ma20` carry a −yield bias for
their full lookback (up to 6 months) — so the model *sees* the same
artefact it is being taught to rank on.

**CORPORATE_ACTION_BIAS: MATERIAL** — a qualitative classification
supported by the seasonal excess of candidate prints and their size, not by
confirmed events (seasonal, name-specific; not a look-ahead leak; magnitude
of the effect on the reported Sharpe unknown until an adjusted panel exists). **Data requirement:** a per-symbol table
of ex-dividend / ex-rights dates with cash amount and share ratio (TWSE
"除權息" announcements; e.g. FinMind `TaiwanStockDividend`, TWSE TWT49U
export, or a broker export) from 2015 → build an adjustment-factor
series → an *isolated* research-adjusted cache; production data untouched
until reviewed.

## Session/timestamp alignment

Nightly refresh runs after 22:00 (TWSE EOD published ~14:30–17:00); feature rows end at T close; the decision is consumed at T+1 open; the label starts at T+1 close. Next-open vs next-close was validated equivalent (retention 1.005/1.014). No intraday data enters the model.
