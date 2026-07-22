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

### 2026-07-22 22:45 — cycles 6–8 in flight (elapsed 2.1h)
- Cycle 6 (D3): h5/h10 rejected for blend book (h20 best both panels); 7.5% name
  cap Sharpe-neutral (adopt-optional). Commit de6a4e7.
- Cycle 6b (R1): blend50+band10 universe bootstrap PASS (100% positive; p5 1.52
  champ / 1.15 bear; drop-top-3 1.77/1.16).
- Cycle 7 (A1): rank-10 standalone REJECTED (1.51 vs 1.91). Rank-10×D1.2 blend
  h10+band10 LO 1.97 → bear validation A1B running on GPU (2/11 refits).
- Cycle 8 prep (A2): PRODUCTION_EDIT_PLAN Edit 1 documented; train_transformer_eod
  walkforward now emits score_std (additive column; mean path untouched); champion
  reproducibility rerun queued after A1B.
- Validation: full Section-6 profile for blend50+band10 written
  (VALIDATION_blend50_band10.json): no losing year either panel; 2022 −0.04 vs
  D1.2 −1.55; IC positive every year except 2022; rank-ac 0.993.
- Device: RTX 4060 Ti busy (A1B); CPU experiments continuing.
- Blockers: none.

### 2026-07-22 23:45 — cycles 7/9 verdicts, BATCH1 launched (elapsed 3.1h)
- A1B landed (11 refits, 1286s train): rank-10 bear standalone 1.08/1.21; its
  blend h10+band10 **0.92/0.96 — the bull-window LO 1.97 collapsed**. Rank-10
  line CLOSED. Cross-window gates worked exactly as designed.
- B2 bear verdicts: MH1 1.11/1.22, MH2 1.39/1.43 — both fail vs blend50+band10
  (1.42/1.48). Multi-horizon ensembles REJECTED.
- **blend50+band10 survives all challengers to date; remains the production
  candidate.**
- BATCH1 (A2 champion reproducibility + score_std test, then E1 close_d12 at
  5 seeds) launched on GPU (bcbhdx7sd).
- Runtime note: per-process dataset build ≈ 30 min dominates GPU jobs → batch
  configs per process from now on (documented in METHODS.md).
- Next after BATCH1: A2 confidence-filter evaluation (CPU); E1 gate check;
  refresh_data --backfill-start design (2015–17 cache depth).
- Blockers: none.

### 2026-07-23 00:15 — backfill tool built; data-hygiene incident handled (elapsed 3.6h)
- refresh_data.py gained --backfill-start (commit 24f63c7); 2330 verified
  (+734 rows to 2015-01-05).
- INCIDENT: full-universe backfill was launched while BATCH1 (GPU) was mid-run —
  E1's dataset rebuild would have read a partially-backfilled cache. Backfill
  STOPPED after 0 additional stocks (only 2330 from the manual test). Verified
  harmless: dataset builder nulls targets on dates with <30 names, so 2330's
  solo 2015–16 rows contribute zero training samples. Full backfill deferred
  until BATCH1 completes. Lesson recorded: never mutate data_cache while a GPU
  experiment process may rebuild datasets.
- BATCH1: A2 wf1/7 val_ic +0.0706 (champion ref 0.074 — tracking).
- Blockers: none.

### 2026-07-23 01:15 — BATCH1 verdicts; queue v2; backfill running (elapsed 4.6h)
- A2a reproducibility: **PASS exact** (1.91/1.93/0.0716 = original). score_std
  Edit 1 validated → committing production edit.
- E1 close_d12 @5 seeds: **REJECT** (1.37/1.50 vs 1.91/1.93). Close-only stands.
- A2b confidence filtering: **REJECT** — drop-uncertain halves Sharpe; seed
  disagreement marks high-signal names, not bad ones.
- Queue v1 exhausted → queue v2 generated (BEAR-DEEP, vol-adj target,
  avoid-bottom target, regularization check, paper cadence).
- Full-universe 2015 backfill running in background (bfnnbkslw, ~2.2h).
- GPU idle until backfill completes (BEAR-DEEP requires consistent cache).
- Blockers: none.

### 2026-07-23 03:30 — backfill done; BATCH2 launched (elapsed 6.9h)
- Backfill complete: 106 stocks, +77,064 rows, 0 fetch failures; cache now
  2015-01→2026-07 for the full universe. 6h checkpoint report committed (6f0a486).
- BATCH2 launched (bbj2i28zf): BEARDEEP rank-20 2021→, B3 voladj, B4 avoid-bot,
  A3 dropout-0.3, A3 wd-5e-4 — 5 configs, one dataset build, ~2–3h.
- Comparability note for readout: B3/B4/A3 train on the deep cache; the 1.91
  champion reference is shallow-cache. Fair deep-cache 2023–26 reference =
  BEARDEEP panel's 2023+ subwindow (refit-grid offset caveat).
- Forward cache refresh (2026-07-08→22) deferred until BATCH2 completes —
  cache-mutation discipline.
- Blockers: none.

### 2026-07-23 06:00 — BATCH2 verdicts (elapsed 9.4h)
- BEARDEEP: deep cache lifts bear-window blend50+band10 to **1.47 / DD −18.7%**
  (new standing reference, was 1.42/−26.4%); tf standalone 1.14. Deep training
  trades some 2022 crash-adaptivity (tf −0.05 vs +0.58) for higher overall level;
  blend stays ≈ flat in 2022.
- B3 voladj target: REJECT (1.44/1.72). B4 avoid-bottom: REJECT — val IC +0.146
  vs OOS IC −0.069 inversion; veto variants also fail (1.45/0.93 vs 1.86).
- A3: champion regularization confirmed at plateau (1.86/1.90, no val-IC case).
- Next: forward cache refresh (07-08→22), daily retrain + inference on fresh
  data, paper snapshot + blended decision book = live F3 cadence demo.
- Blockers: none.
