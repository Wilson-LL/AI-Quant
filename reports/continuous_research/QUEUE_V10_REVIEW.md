# Queue v10 — Pre-Start Review Summary

Status: review artifact for QUEUE_V10_PROPOSAL.md. **v10 NOT started.**
Nothing in this document changes code, production, or data.

## 1–3. Experiments, compute class, runtime

| ID | Experiment | GPU-heavy? | Screen runtime | If promoted (7-seed dual-window) |
|---|---|---|---|---|
| A1 (CS1) | cross-sectional attention head on champion trunk | **GPU** | ~40 min | +3–4 h |
| A2 (CS2) | inverted (names-as-tokens) attention variant | **GPU** | ~30 min | +3–4 h (mutually exclusive with A1 adoption) |
| A3 (TCN) | temporal-convolution sanity anchor | **GPU** | ~20 min | promotion not expected (anchor) |
| B1 (MT1) | multi-task heads (rank-20 + rank-5 + vol), joint training | **GPU** | ~30 min | +3–4 h |
| B2 (Q1) | quantile head + uncertainty-damped sizing | **GPU** + CPU book pass | ~30 min | +3–4 h |
| C1 (REV1) | 5d-reversal third blend leg | **CPU pre-screen** (~20 min, zero GPU); GPU branch only if CPU gate passes | ~25 min (conditional) | +3–4 h |
| C2 (RT1) | market-residualized target (inversion tripwire) | **GPU** | ~25 min | +3–4 h |
| D1 (AUG1) | input augmentation vs seed-set variance (2 cells × 2 seed-sets) | **GPU** | ~1.5 h | +3–4 h |
| D2 (DIST1) | 7-seed → 1 student distillation | **GPU** | ~30 min | +3–4 h (runs last, conditional on champion surviving A–C) |

## 4. Total expected runtime

- Screens: ~4.5 h GPU (+20 min CPU for C1 pre-screen).
- Promotions: PROMOTE_TOP_K = 2 across all families → at most 2 full 7-seed
  dual-window re-runs ≈ 6–8 h.
- **Expected total: 12–16 h GPU.** Worst case (every screen promotes —
  historically never happens): ~28 h.

## 5. Closed lines explicitly avoided

Sequence length (40/60/90/120), layer count, dropout/weight-decay, ranking
losses (pairwise + listwise), risk-proxy targets (avoid-bottom, dd-adjusted),
sector-excess target, regime FEATURES (model inputs), extra feature sets
(close_range / close_liq / close_d12 / full-field until ~2027-01), seed-count
expansion (7 fixed; 14 closed), recency weighting, warm-start, rolling
windows, preset-diversity blends, confidence filtering (seed-disagreement),
exposure gates (EX1/EX2/EX3 — EX3 disarmed in v9), adaptive blend weights
(B4 re-confirmed the closure in v9). Near-line distinctions are argued in the
proposal: B1 = joint training (not the closed multi-horizon *ensemble*);
B2 = model-native uncertainty (not the closed ensemble-external filtering);
C2 = market-residual (not the closed sector-excess).

## 6. What counts as ADOPTION

Four mandatory steps, no exceptions: (1) within-family val-IC win at screen;
(2) 7-seed re-run (3-seed screens are known-unreliable); (3) dual-window
blend books at/above the seed-robust ranges 1.85–2.15 / 1.30–1.45 without
material bear-DD degradation; (4) explicit user sign-off. Special cases:
D1 adopts on cross-seed-set val-IC spread halving at flat-or-better mean
(variance is the success metric, not level); D2 adopts on student ≥0.95×
ensemble val IC AND OOS blend within 0.1 of the 7-seed books; B2 must first
hold val IC ≥0.048, then win on book metrics.

## 7. What counts as REJECTION

Screen val IC below champion/family bar → dead at screen (cheap, most likely
outcome per history: 14+ challengers rejected to date). Promoted but either
window below range → REJECT. C2: val IC ≥0.055 with OOS IC <0.02 → inversion
#3, residual-target line closed permanently. A3 is expected to lose (anchor);
it "rejects" by default and only an upset promotes.

## 8. Highest-value experiments (ranked)

1. **A1** — the largest untested inductive bias (cross-sectional attention);
   only axis with plausible step-change upside.
2. **D1** — directly attacks the loop's known worst fragility (seed-set
   variance 1.843 vs 2.147); even a null is decision-relevant.
3. **B1** — cheapest mainstream lever left; single pre-registered cell.
4. **D2** — near-guaranteed practical payoff (7× cheaper retrain) at low
   risk, independent of alpha outcomes.
5. C1 — canonical complement to momentum; CPU gate makes it nearly free.
Lower: B2, A2 (contingent on A1), C2 (informative either way), A3 (anchor).

## 9. Most likely to overfit (ranked)

1. **B2** — the sizing rule can fit noise even with one pre-registered rule.
2. **C2** — inversion risk is the experiment's explicit subject (tripwired).
3. **A1/A2** — new capacity on ~108 names; val-IC gate + 7-seed rule contain it.
4. C1 — small pre-registered blend-weight grid.
Lowest: A3 (anchor), D1 (pure regularization), D2 (compression, not search).

## 10. Cut list for a shorter run

Safe cuts, in order: **A2** (redundant until A1 reads out), **A3** (anchor,
nice-to-have), **B2** (highest overfit risk), **C2** (informative but not
urgent), **C1 GPU branch** (keep the free CPU pre-screen), **D2** (defer to
after the alpha verdicts). Keeping A1 + B1 + D1 preserves ~80% of the queue's
expected information value.

## 11. Production files modified?

**Production defaults/specs: NO — untouched throughout.** Honest caveat:
A1/A2/B1/B2/D2 require flag-gated additions to the shared training module
(`train_transformer_eod.py` / model code), which the production daily retrain
also imports. Mitigation is pre-registered in the proposal: all new code
paths land behind flags that default OFF, and the champion path is
re-verified bit-identical (determinism re-run of a known config) BEFORE any
v10 experiment starts. A3/C1-CPU/C2/D1 need no shared-module edits beyond a
feature/target column and an augmentation hook (also flag-gated).

## 12. data_cache mutation required?

**No.** All new columns (residual target, reversal feature, teacher scores)
are derived in-memory or written to derived files outside `data_cache/`.
The cache is read-only for all of v10. (The independent daily-ops refresh
continues appending new EOD rows on its own cycle — unrelated to v10, and
the existing rule stands: GPU queue never runs during a refresh.)

---

## Option A — FULL v10 (12–16 h GPU)

All nine experiments, phased: A1/A2/A3 screens → B1/B2/C2 screens + C1 CPU
pre-screen → D1 → up to 2 promotions (7-seed dual-window, auto bear-spawn,
config-dedup active) → D2 last. Full information: architecture question
settled from two angles + anchor, all head/objective/signal/procedure axes
read out, inversion tripwire data point collected.

## Option B — SHORT v10 (4–6 h GPU, highest-value subset)

| Item | GPU time |
|---|---|
| A1 cross-sectional attention screen | ~40 min |
| B1 multi-task heads screen | ~30 min |
| D1 augmentation vs seed variance | ~1.5 h |
| C1 CPU pre-screen (bonus, zero GPU) | 0 |
| ONE promotion slot (PROMOTE_TOP_K=1), best val-IC winner | ~3–4 h |
| **Total** | **~4–6 h** |

Cut: A2, A3, B2, C2, C1-GPU, D2 — all resumable later as v10b without
re-running anything (scheduler dedup guarantees no duplicate configs).
Option B preserves the three highest-value questions (new architecture axis,
cheapest untested lever, seed-variance fix) at ~1/3 the GPU budget; its main
loss is settling the architecture family from one angle instead of two and
deferring the distillation payoff.
