# Continuous Research Checkpoint — 2026-07-23 01:45

Loop start 2026-07-22 20:39 · elapsed ~5.1h · cycles completed: 14 experiments /
verdicts across 8 committed checkpoints (121bd54 → bb28a5b).

## 1. Experiments since loop start

| ID | hypothesis | verdict |
|---|---|---|
| C1 | adaptive blend weights from trailing data | REJECT (all 5 ≤ static 1.37) |
| C2 | static blend frontier | score-level ≫ return-level (+0.2); 50/50 best |
| D1 | no-trade band / invvol on blend book | **PROMOTE band10**; reject invvol |
| F1 | paper-trading scaffold | tool shipped; 76 shadow books, ledger live |
| C3 | exposure-scaling risk gates | REJECT (EX3 L/S → monitor) |
| D1b | blend60/70 + band10 | REJECT (below blend50+band10 both panels) |
| F2 | blended decision-book generator | tool shipped (Section-9 columns) |
| D3 | h5/h10 holds; 7.5% cap | REJECT short holds; cap-7.5% Sharpe-free option |
| R1 | universe bootstrap of promoted book | PASS (100% positive, p5 1.52/1.15) |
| A1 | rank-10 target @ champion strength | REJECT standalone (1.51 vs 1.91) |
| B2 | multi-horizon r20+r10 ensembles | REJECT (bear gate) |
| A1B | rank-10 blend bear validation | REJECT — line closed (LO 1.97 → 0.96 OOW) |
| V1 | Section-6 validation profile | PASS (no losing year either panel) |
| A2a | champion reproducibility + score_std edit | PASS — exact reproduction |
| E1 | close+D1.2 features @ 5 seeds | REJECT (1.37/1.50 vs 1.91/1.93) |
| A2b | seed-disagreement confidence filter | REJECT (drops high-signal names) |

## 2. Leaderboard (current)

**Production candidate (unchanged since cycle 2, survived 8 challengers):**
50/50 z-blend(champion TF, D1.2) · top-quintile · equal-weight · 10% no-trade band
· hard 10% cap · 20d hold · monthly retrain — **L/S 1.95 net60 / DD −12.3%
(2023–26); 1.42 / −26.4% (2021–26); no losing year; bootstrap-robust.**
LO variant 1.83 / −27.0%. Optional: 7.5% cap (free), EX3 DD gate (bear-helpful).

Standalone champion transformer: 1.91/1.93 (reproduced exactly).
D1.2: 1.64/1.77. mom20: 0.40.

## 3. Rejected this window

Adaptive weights, return-level blending, invvol, exposure gates EX1/EX2,
h5/h10, rank-10 line (incl. its seductive bull-window LO 1.97), multi-horizon
ensembles, close_d12 features, confidence filtering. Full table in
RESEARCH_SCOREBOARD.md.

## 4. Promoted

blend50+band10 (cycle 2, D1) — subsequently validated (R1, V1) and unbeaten.

## 5. Code changes

New: research/{adaptive_blend, blend_construction, exposure_scaling,
paper_trading, blended_decision_book, loop_experiments}.py; docs/continuous_research/METHODS.md.
Modified: train_transformer_eod.py (score_std, Edit 1 — validated by exact champion
reproduction), dataset_transformer_eod.py (2 additive targets), research/refresh_data.py
(--backfill-start). PRODUCTION_EDIT_PLAN.md updated before pipeline edits.

## 6. Runtime summary

GPU jobs: A1 (35 fits), A1B (55 fits), BATCH1 (70 fits) — all completed, AMP,
peak ~6.3 GB VRAM, GPU 83–95% utilized. Key finding: per-process dataset build
(~30 min) dominates wall-clock → configs are now batched per process.
CPU panel experiments: ~20 runs, each < 5 min. Everything fits the 12h daily
budget trivially (full daily production workflow remains ~5–10 min).

## 7. Best next hypotheses (queue v2)

1. BEAR-DEEP: bear-window rerun on the 2015-backfilled cache (in progress —
   backfill ~65% done, BATCH2 queued).
2. B3 vol-adjusted target; B4 avoid-bottom target (targets built + smoke-tested).
3. A3 regularization one-knob checks.
4. F3 paper-trading daily cadence.

## 8. Did the best candidate change?

Yes, in cycle 2: static 50/50 blend → **blend50 + 10% no-trade band** (better on
both panels, both modes, all cost levels). Unchanged since despite 8 challenges;
every challenger rejection tightened the evidence around it.
