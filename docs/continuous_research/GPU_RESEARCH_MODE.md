# GPU research mode (queue v5+)

Activated 2026-07-24 on user instruction: research-first, GPU-saturated; the
TWSE watcher is subordinate (daily ops run only between/after GPU jobs).

## Scheduler

`research/gpu_research_scheduler.py run` executes
`reports/continuous_research/gpu_scheduler/queue_v5.json` sequentially:

- **max_concurrent_gpu_jobs = 1.** One CUDA training process; datasets are
  GPU-resident and epochs saturate the device. Note: this desktop also runs
  FFXIV/Discord/browsers (~3 GB VRAM); 13 GB remains — safe for every queued
  config, but concurrency 2 is NOT enabled while the game is running.
- Dataset builds are the wall-clock bottleneck (~35 min); configs carry an
  `order` field grouping identical (feature_set, seq_len) so each dataset is
  built once per group.
- Crash-safe: queue JSON rewritten after every job (status pending/running/
  done/failed + error text). Failures are recorded, never skipped silently.
- OOM: empty_cache → retry once with half the seeds (logged) → failed status.
- Auto-promotion: when all screens are done, top-2 by mean VAL IC (selection
  metric only) are cloned to 5-seed configs with blend evaluation; any
  promoted config whose blend L/S net60 ≥ champion-window ref − 0.05 spawns a
  bear-window run automatically.
- Per-job JSONL log: start/elapsed/VRAM/params/val-IC in
  `gpu_scheduler/scheduler_log.jsonl`.

## Queue v5 (registered before launch)

12 screens (3 seeds, 2023 window): layers 1/3 · dropout 0.1 · wd 1e-5/1e-3 ·
spread target · drawdown-adjusted target · listwise loss · seq 90/120 ·
close+range · close+liquidity; 1 full run (7-seed ensemble); plus auto-spawned
promotions/bear runs, and CPU construction experiments (ERC weighting, top-N,
conservative spec) run panel-side. Closed lines from queues v1–v4 (see
scoreboard) are NOT re-run.

Acceptance gates: unchanged — dual-window (2023–26 AND 2021–26) comparison
against champion transformer (1.91), blend50+band10 refs (2.06/−10.7% and
1.47/−18.7%), D1.2, mom20, equal-weight universe; val-IC selection discipline;
hard 10% name cap; costs 0/60/100/150 bps.

## Verdict duty

Screen results are decision-grade only for *promotion*; final keep/reject
verdicts require the 5-seed dual-window treatment (A1/E1 precedent: 2-seed and
3-seed orderings are noisy).
