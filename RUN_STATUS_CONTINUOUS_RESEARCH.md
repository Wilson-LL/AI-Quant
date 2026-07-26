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

### 2026-07-23 06:45 — live daily cycle done; C4 rejected; WD1 running (elapsed 10.1h)
- F3 live cadence: refresh (+1,100 rows → 2026-07-22) → retrain 267s → inference
  → paper books + decision book (22 names, 21 HOLD/1 BUY/1 SELL, maxW 10.0%).
  Whole cycle ~11 min — deep cache doubles retrain time, still ~65× in budget.
- C4 sector-neutral scoring: REJECT (costs 0.4–1.1 Sharpe both panels; sector
  tilt is load-bearing alpha). Commit 677e864.
- WD1 (3y-window member for train-window-diversity ensemble) on GPU (bjwbhwxx0).
- Blockers: none.

### 2026-07-23 08:50 — WD1 rejected; REF23 references set; BATCH3 running (elapsed 12.2h)
- WD1: roll-3y member collapses on deep cache → all window-diversity ensembles
  rejected (best 1.30 < 1.47 ref). Line closed. Commit 5f624a4.
- REF23: deep-cache champion 2023–26 = 1.91/1.91 standalone (training-depth
  insensitive); **blend50+band10 = 2.06 L/S / DD −10.7% / LO 1.95 (B4d, new
  standing reference)**. Commit 0f292c8.
- BATCH3 (preset C both windows → architecture-diversity blend gates) on GPU.
- Blockers: none.

### 2026-07-23 14:45 — queue v3 experimental items complete (elapsed 18.1h)
- BATCH3 preset-C blend: REJECT (ties champ window, loses bear). BATCH4
  excess-sector: REJECT (promising list emptied). RL1 pairwise loss: REJECT by
  val-IC gate (0.036 < 0.050; OOS confirms). Commits 5e50dcc→this.
- 12h checkpoint written (830bee1). Edit 2 (pairwise loss) stays in the
  codebase, default MSE path verified identical.
- **Loop state: OPERATIONAL CADENCE.** All Track A–E levers tested; 13
  challengers rejected against blend50+band10 (refs 2.06/−10.7% and
  1.47/−18.7%). Daily cycle validated at ~11 min. Next experimental triggers:
  new regime evidence in the paper ledger, full-field data (~2027-01), or
  user direction.
- Blockers: none (idle pending next TWSE close for the daily cycle).

### 2026-07-23 15:40 — loop continued on user instruction; queue v4 (elapsed 19h)
- Daily refresh: +0 rows (TWSE 2026-07-23 not yet published; retry after E3).
- Queue v4: (1) **E3 close+market-regime features** — the one untested Track-E
  set, targets the 2022 weakness at feature level (distinct from rejected
  portfolio-level C1/C3); (2) daily cycle when 07-23 data lands; (3) B6
  drawdown-adjusted target only if E3 shows the regime axis carries signal.
- close_regime implemented (4 causal market columns; smoke-tested) — commit
  4e19cf3. E3 dual-window on GPU (b3782mpks).
- Blockers: none.

### 2026-07-23 17:45 — E3 rejected; queue v4 experimental items closed (elapsed 21.1h)
- E3 market-regime features: REJECT — IC diluted, 2022 worsened (−0.30 vs
  −0.15), blends below both refs. A4 (head-level) and B6 (conditional) cancelled
  per pre-registration. Lesson logged: marginal val-IC edges (±0.005) are not
  decision-grade; dual-window OOS gates decided correctly.
- Daily refresh retried ×2: +0 rows — TWSE 2026-07-23 EOD not yet published.
- **Loop state: operational cadence.** 14 challengers rejected across queues
  v1–v4; blend50+band10 unbeaten (refs 2.06/−10.7% and 1.47/−18.7%). Next
  triggers: TWSE data publication (daily cycle), paper-ledger regime evidence,
  full-field data (~2027-01), user direction.
- Blockers: none.

### 2026-07-24 — GPU RESEARCH MODE (user-directed); queue v5 running
- GPU check: RTX 4060 Ti, 13.1/16.4 GB free; **FFXIV + Discord + browsers hold
  ~3 GB (user warned; proceeding — VRAM ample; max_concurrent stays 1 while
  the game runs).** Hourly watcher stopped (scheduler owns daily ops, runs
  them only at queue completion — no cache-mutation races).
- Built: gpu_research_scheduler.py (crash-safe queue, OOM half-seed retry,
  val-IC auto-promotion, bear auto-spawn, JSONL logging); 2 new targets
  (spread, dd-adjusted); 2 curated feature sets (close_range, close_liq);
  listwise loss (Edit 3). Commit 146022c.
- Queue v5: smoke + 12 screens + 7-seed full + auto-promotions. Scheduler
  running (bh3mp3q9n): smoke mid-walkforward, GPU 90%, 5.8 GB.
- CPU cells done meanwhile: D5 varpar REJECT, D6 topN REJECT, D7
  band15+cap7.5 = documented conservative option (1.98/−10.6% & 1.46/−16.1%).
- Next: scheduler completion → screen verdicts, promotion readouts, checkpoint.
- Blockers: none.

### 2026-07-24 20:30 — GPU-mode day summary (elapsed since mode start ~9h)
- Queue v5: 19 GPU configs, 0 unresolved failures → **7-seed spec ADOPTED**
  (2.147/−10.6% & 1.443/−18.0%, refs B4e/B4f); seq axis closed (40<60>90>120);
  ranking-loss family closed; E4/E5 features rejected; B6 inversion #2.
- Queue v6: 9-seed = saturation (keep 7). Edit 4 wires 7 seeds into daily
  retrain (tested live: 7 ckpts / 304 s); 07-24 books on adopted spec.
- D7b: conservative construction (band15+cap7.5) = recommended deployment spec
  on the 7-seed signal (Sharpe-identical, bear DD +2.8pp, 2022 −0.08).
- Queue v7 running (GPU): disjoint-seed replication ×2 windows, refit-63
  sensitivity, crash-first 2022 window. CPU: 7-seed bootstrap running.
- GPU utilization: near-continuous training since mode start; VRAM ~2.2–2.9 GB
  alongside FFXIV (~3 GB); 0 OOM.
- Blockers: none.

### 2026-07-26 — crash recovery + queue v8 completion (14-seed verdict: keep 7)
- **07-25 01:42 incident:** CUDA `unknown error` (driver/GPU wedge) killed
  ENS14_2021 at wf 4/11 and BEAR_ENS14_2023 at spawn; the PC then shut down
  unexpectedly. Scheduler finalized all files before exit — recovery audit
  found **zero corruption** (all JSON/JSONL/gzip verified; ENS14_2023 result
  cross-consistent and VALID). Full audit: `RECOVERY_AFTER_SHUTDOWN.md`.
  Pre-crash ENS14_2021 partials (wf 1–3 val_ics) declared INVALID, unused.
- **Resume (user-approved):** failed items re-pended (original errors archived
  in `gpu_scheduler/queue_v8_failure_archive_20260725.json`), ENS14_2023 kept
  done (not rerun). Pre-flight: CUDA smoke + AMP OK, nvidia-smi healthy, no
  stray python, correct branch. Scheduler resumed 07-26 00:23, completed 05:25:
  **3 done, 0 failed** — the CUDA error did not recur (post-reboot, under the
  same FFXIV load that accompanied the original crash).
- **Queue v8 verdict (scoreboard V8):** convergence hypothesis CONFIRMED —
  champ blend 2.125 (midpoint 1.995), bear blend 1.383 (midpoint 1.372). The
  adopt-14 gate passes by the letter, but the bear margin (+0.011) is parity
  within seed noise, bear DD is 4.6pp worse (−22.6% vs −18.0%) and 2022 is
  −0.43 vs −0.15 at 2× retrain cost → **recommendation: keep 7-seed spec;
  adoption decision left to user.** Production unchanged (7 seeds, D7b
  band15+cap7.5 remains recommended deployment variant).
- Scheduler design note: bear auto-spawn duplicated ENS14_2021 exactly
  (bit-identical books — determinism reconfirmed; ~2.4h GPU on a redundant
  run; dedup-by-config worth adding before the next queue).
- Daily ops ran at queue completion: +0 rows (TWSE weekend, nothing to infer).
- Blockers: none. Queue v8 CLOSED; no new queue started (per user direction).

### 2026-07-26 — user decision: keep 7 seeds; spawn-dedup fix landed
- **User confirmed KEEP 7** — 14-seed line closed on the scoreboard (V8);
  production spec unchanged (7-seed, blend50+band10; D7b band15+cap7.5 stays
  the recommended deployment variant).
- **Spawn-dedup fix** in `gpu_research_scheduler.py`: `promote()` and
  `spawn_bear()` now dedup candidates by normalized run-defining config
  (`_run_key`: feature_set/target/seq_len/seeds + run_one defaults), not just
  id — closes the v8 gap where BEAR_ENS14_2023 duplicated ENS14_2021 under a
  different id (~2.4h GPU). Verified by replaying the v8 queue: duplicate
  spawn skipped, genuine spawn (uncovered window) still queued, default
  normalization correct.
- Next queue: none started (awaiting direction).

### 2026-07-26 — queue v9 complete (deployment validation, CPU-only): 13/13, 0 failed
- **User-approved v9 ran as designed**: signal frozen to the three 7-seed
  panels, zero GPU training, cache read-only, production untouched. All four
  comparison books (blend / D7b / tf / D1.2) in every experiment. Runtime
  ~65 min CPU (R3 bootstrap 57 min). Results:
  `reports/continuous_research/queue_v9/` (per-experiment JSON +
  QUEUE_V9_SUMMARY.md); scoreboard row V9.
- **Deployment case execution-validated:** costs (break-even 632/463 bps),
  capacity (10M TWD trivial), delay (T+2 cost 0.226 ≤ 0.30; same-bar audit
  clean), settlement (constrained fills unnecessary), sector caps ~free.
  **D7b (band15+cap7.5) survives its grid** — band15_cap10 looked better on
  raw Sharpe (2.21 CH) but missed the pre-registered 1pp-DD dominance bar;
  not adopted (no post-hoc grid picking).
- **Track 4 all rejected at their bars → no crash overlay exists**: EX3
  formally DISARMED (blend +4.0pp DD but d7b +2.2pp < 3pp), tf hedge sleeve
  +1.6pp < 3pp, regime downweight rejected as pre-declared (bear side did
  improve — 2022 −0.15→+0.04 — but CH flat; single crash sample, recorded).
  Bear risk is managed by construction only; this is a documented, accepted
  state, not an oversight.
- **New operational tooling (P1/P2 passed):** matured forward ledger
  (17/20 obs — realized-vs-backtest evidence gate ACTIVATES in ~3
  rebalances) and daily diff report. Both should join the daily-ops cycle.
- **Open flag (R5):** bear-window drop-top-5 retention on d7b = 0.667
  (<0.70); top contributor 1519 at 9–12% of PnL. The single number to watch
  as the paper ledger matures.
- Blockers: none. v10 (GPU research queue) proposed separately — NOT started.
