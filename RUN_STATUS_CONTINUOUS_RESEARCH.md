# Continuous Research Loop — Run Status

Branch: research/continuous-alpha-loop-4060ti · RTX 4060 Ti (CUDA OK, torch 2.13 nightly cu132)

## Heartbeat log

### 2026-07-22 20:39 — loop start (elapsed 0h)
- Cycle: 0 (setup)
- Read prior sprint state: champion TF 1.91 L/S net60 (2023–26), blend 1.37 (2021–26).
- Verified CUDA + GPU (RTX 4060 Ti). Cached walk-forward panels found → panel-based
  experiments (Tracks C/D) run CPU-only.
- Created RESEARCH_SCOREBOARD.md with seeded baselines + 8-item hypothesis queue.
- Next: Cycle 1 = C1/C2 adaptive + static blend sweep on bear panel (2021–26).
- Blockers: none.

### 2026-07-22 21:08 — cycles 1–2 done (elapsed 0.5h)
- Cycle 1 (C1/C2, CPU): adaptive blends REJECTED (all 5 ≤ static 1.37); score-level
  blending dominates return-level (+0.2); frontier flat 50–70% TF.
- Cycle 2 (D1, CPU): **blend50+band10 = new best book** — L/S 1.95 net60 / DD −12.3%
  (2023–26), 1.42 / −26.4% (bear window). Invvol rejected. Band plateau 10–20%.
- Files: research/adaptive_blend.py, research/blend_construction.py,
  research/loop_experiments.py (new); results + prereg in reports/continuous_research/.
- GPU: A1 (rank-10 target, preset B, 5 seeds) training in background since 20:53.
- Next: A1 readout → cycle 3; then F1 paper-trading scaffold.
- Blockers: none.

### 2026-07-22 21:42 — cycles 3–5 done (elapsed 1.0h)
- Cycle 3 (F1): paper-trading scaffold built (`research/paper_trading.py`,
  backfill/snapshot/evaluate). 76 books backfilled 2025-01→2026-07; ledger shows
  17 matured rebalances/strategy: tf 2.28, d12 2.07, blend50 2.04, band10 2.02
  (gross LO ann Sharpe), band10 lowest turnover (0.20 vs 0.30).
- Cycle 4 (C3): exposure scaling — REJECT overall vs pre-registered gates;
  EX3 own-equity DD gate on L/S = MONITOR (bear: 1.46/−19.5% vs 1.42/−26.4%;
  champion cost −0.05).
- Cycle 4b (D1b): blend60/70+band10 fail gate (below blend50+band10 both panels).
- Cycle 5 (F2): blended decision-book generator built
  (`research/blended_decision_book.py`); 2026-07-07 book: 22 names, maxW 10.0%,
  21 HOLD / 1 BUY / 1 SELL / 9 WATCH.
- GPU: A1 still training (~50 min, GPU 83–95%; 10d target converging slower).
- Next: A1 readout; D3 holding/cap sweep on blend book; A2 needs score_std panels.
- Blockers: none.
