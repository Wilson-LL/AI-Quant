# 05 — Validation methodology, multiple testing & winner's curse (B4, B5, B15)

## What is implemented

Chronological walk-forward: `oos_start` 2023-01 (CH) / 2021-01 (BR),
refit every 126 sessions (~6 months; docs wrongly say "monthly"), train
= all matured history, val = newest 10% of matured dates, purge = horizon
+ lag = 21 (sufficient: equals the label overlap), OOS block scored by the
7-seed mean; portfolio metrics on non-overlapping 20-session blocks.
Purged/embargoed CV beyond the existing purge is not required by the
target structure. Selection rule on paper: screen by val IC, adopt by
dual-window OOS book metrics — **so the OOS windows are the selection set**.

## How much selection has touched the CH/BR windows

- `reports/transformer_gpu/panels/`: **111 OOS panels**, all on
  2023→ or 2021→. Excluding smoke, seed-set replications, distillation
  students and identical re-runs, roughly **60–70 materially distinct
  model/strategy comparisons** (targets ×6, feature sets ×13, lookbacks
  ×4, architectures ×10, training regimes ×20, losses ×4, blends/
  constructions ×15+, overlays ×8) were scored on the same windows.
- The construction grid in this audit adds 40 more cells; the baseline
  suite ~10 more.
- The bear window contains the champion window (64% of dates shared), so
  "consistency across two windows" is ≈1.4 independent checks.
- Direct evidence of winner's curse already in the record: the identical
  config on disjoint seeds scored **1.843 vs 2.147** (−0.30); the
  champion point sits at its own bootstrap p95; the v13 f2 "2.101" screen
  collapsed to 1.60 on confirmation.

## Selection-adjusted sanity check (no false precision)

Deflated Sharpe (Bailey & López de Prado) on the champion's long-only
net60 per-block series (`audit_v17_statistics.py`): CH n=42, per-block
SR 0.56, skew +0.29, kurtosis 4.8; BR n=67, SR 0.41, skew −0.11, kurt 2.6.

| Assumption: N distinct trials / std of trial Sharpes (ann.) | Expected max of null trials (ann.) | CH: P(true SR > that) | BR: P(true SR > that) |
|---|---|---|---|
| 10 / 0.20 | 0.32 | 0.998 | 0.993 |
| 30 / 0.35 | 0.73 | 0.984 | 0.943 |
| 70 / 0.35 | 0.84 | 0.974 | 0.909 |
| 70 / 0.50 | 1.20 | 0.909 | 0.709 |

Plain probabilistic Sharpe: P(true SR > 0) ≈ 1.0 both windows;
**P(true SR > 1.0): CH 0.95, BR 0.84**.

**What this establishes.** With any defensible trial count (30–70) and
trial dispersion (0.35–0.5), the champion's long-only result is very
unlikely to be a pure selection artefact (DSR ≥ 0.9 on CH, ≥ 0.7 on BR):
the strategy almost certainly has a positive, probably > 1.0, long-only
Sharpe on this survivor universe. **What it does not establish** is the
magnitude: the point 1.99 (CH) carries a sampling SE of ~0.6 and a
selection haircut of order 0.5–1.0 under the same assumptions, and the
survivor universe adds an unquantified upward bias (03). A candid central
expectation is **CH long-only ≈ 1.0–1.7, BR ≈ 0.7–1.3**, and the honest
label for every historical number remains **SURVIVOR-UNIVERSE UPPER-BOUND
REFERENCE**. The audit did not compute a single "corrected Sharpe" because
the trial set is not a null ensemble (most trials were deliberate
degradations) and no per-trial return series exist for a PBO estimate.

## Production ≠ validated protocol

Validated = 126-session refit with a 10% holdout; production = **daily
full refit** with the same holdout. Never validated; nearest tested
cadences (weekly/daily *warm* retrain, G3) were strongly negative; daily
full refit untested → **P0 in 12/15**.

## Protocol for v17 challengers (pre-registered)

- Selection windows = CH + BR (already burned; used for screening only).
- Final untouched test = prospective period from 2026-07-24 (panel
  freeze), evaluated once, champion vs ≤ 1 finalist, then shadow. It is
  currently ~1 independent block and must accumulate (06).
- Bar: beat the champion's bootstrap **p50** (not point) on both windows
  on the scorecard in 12, with disjoint-seed replication within 0.15.
- Multiplicity: ≤ 12 GPU configurations in the whole v17 programme;
  the family-wise count is reported with every result.
