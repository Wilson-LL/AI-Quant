# 15 — P0: exact production-training parity validation (design; NOT run)

## The mismatch, precisely

| | Validated research (SCHED_A8 / BEAR panels) | Production (`daily_ops.bat` step 2) |
|---|---|---|
| Refit cadence | every **126 sessions** | **every session** (full refit from scratch) |
| Training data at refit | all matured labels ≤ refit date − 1 | all matured labels ≤ latest cache date |
| Holdout | newest 10% of matured dates, purge 21 | identical rule |
| Early stopping | argmax val rank IC, patience 3, min 2 | identical |
| Architecture / optimizer / seeds | preset B, AdamW 3e-4, MSE, seeds 0–6 (fixed IDs) | identical — same fixed seed IDs retrained from scratch each session |
| Ensemble | mean of 7 seed scores | identical |
| Predictions used for a date T | from the model refit ≤ 126 sessions earlier | from the model refit **that night** |
| Downstream | blend50 / band10 / net60 / T+1 close (≈ next open) | identical |

Everything is identical except cadence — and cadence changes two things
the research never measured: (1) how much *newer* data the model has
seen (≤ 1 day vs ≤ 126 days stale, both still ~13 months behind via the
holdout), and (2) **day-to-day retraining variation**: every night the same fixed
seeds 0–6 are retrained from scratch on a slightly larger dataset (and
on non-deterministic CUDA kernels), so rank order can move without new
information reaching the training set; the band absorbs some of this,
but the ranking layer and WATCH list see all of it. G3's weekly/daily *warm*
refits were strongly negative; daily *full* refit is unmeasured.

## Paired-seed design (0B)

Cadence is the ONLY experimental variable. Every comparison uses the
same seed IDs in both arms, the same decision dates, features, target,
training/validation/early-stopping rules, blend50, top_frac, band10,
costs and execution timing. Fixed 3-seed subset = **{0, 1, 2}** (the
first three production seeds; the project's standing 3-seed screen set).

| Stage | Arm A | Arm B |
|---|---|---|
| P0-A′ (screen) | seeds {0,1,2}, refit every 126 sessions | seeds {0,1,2}, refit every 5 sessions |
| P0-A | seeds {0,1,2}, 126 | seeds {0,1,2}, daily |
| P0-B (confirmation) | seeds {0..6}, 126 | seeds {0..6}, daily |

A 3-seed arm is never compared with the 7-seed champion panel.

## Design

For each historical decision date T in a chosen span:
1. build the dataset as of T (`build_dataset` on the cache truncated to T
   — no future rows; identical to what `refresh → retrain` would have seen);
2. `matured_train_val(refit_rank = T)` (same holdout/purge);
3. train S seeds with `fit_one` (same preset/optimizer/early stop);
4. score the cross-section at T with the seed mean (+ std);
5. write a panel `date, stock, score, score_std` — the "daily-refit
   panel" — for the span.
Then compare, on the same dates, against the existing 126-refit panel
(`SCHED_A8_seeds7_full` scores for those dates): rank IC vs realized
fwd_20, blend50+band10 net60/net100 return, max DD, turnover, top−bottom
20d spread, **rank stability** (day-to-day Spearman of the score vector;
book Jaccard), and score correlation between the two panels.

No future-trained weights are reused: each date's model is trained
fresh from data ≤ T. The existing `walkforward()` already implements
steps 1–5 with `refit_every` as a parameter; P0 = `refit_every=1` over a
bounded span (plus a per-date dataset truncation guard, since the
current routine builds the dataset once for the full cache — that is
fine because labels are masked by maturity and features are causal
(02), but it will be asserted explicitly).

## Runtime estimate (measured, not guessed)

Live manifest 2026-09-11: per-seed fit on the full 2015→ dataset took
44–97 s (4–7 epochs, AMP, batch 1024) → **≈ 70 s per seed-fit**; dataset
build ≈ 30 s once per run; scoring negligible.

| Stage | Span | Seeds | Refits | GPU time |
|---|---|---|---|---|
| **P0-A** | 2026-01-05 → 2026-07-23 (126 sessions, latest half-year of the frozen panel) | 3 | 126 | 126 × 3 × 70 s ≈ **7.4 h** (one overnight) |
| P0-A′ (optional cheap prior) | same span, refit every 5 sessions | 3 | 26 | ≈ 1.5 h |
| **P0-B** | 2025-07-24 → 2026-07-23 (252 sessions) | 7 | 252 | 252 × 7 × 70 s ≈ **34 h** (≈ 4 overnights) |
| P0-B (reduced) | 2026-01-05 → 2026-07-23 | 7 | 126 | ≈ **17 h** (2 overnights) |

Recommendation: run **P0-A′ then P0-A** (≈ 9 h total); proceed to the
reduced P0-B only if P0-A shows a difference larger than the champion's
own 3-seed-vs-7-seed noise. Full P0-B (34 h) is not proposed unless the
reduced run is ambiguous. GPU windows must avoid 22:00 daily_ops and
should be sequential with it (VRAM is sufficient to co-run, but the
retrain's own timing/gates should not be perturbed).

## Decision rule (pre-registered)

Compare the denser-cadence arm vs the 126-refit arm (same seeds) on the identical dates:
- if daily-refit rank IC ≥ 126-refit IC − 0.005 **and** net100 Sharpe ≥
  126-refit − 0.15 **and** rank stability not materially worse → cadence
  is **PARITY_OK**; the research references may be read as describing
  production;
- if daily-refit is materially worse (IC −0.01 or Sharpe −0.3 or DD +5 pp
  or day-to-day rank autocorrelation < 0.95) → **PRODUCTION_RESEARCH_PARITY_RISK**;
  the immediate remedy is operational (refit weekly/monthly with the
  validated cadence), which needs no model change and would be proposed
  as a separate approved production change;
- if daily-refit is materially *better* → record it, but it does not
  change the champion's references (they would become conservative).
