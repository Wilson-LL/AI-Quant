# Recovery Audit — Unexpected Shutdown During Queue v8

Audited: 2026-07-25 (post-shutdown), branch `research/continuous-alpha-loop-4060ti`

## TL;DR

- **No file corruption. No data loss. No repair needed.** All JSON/JSONL files
  parse; the ENS14_2023 panel gzip verifies; the scheduler wrote its final
  state (`queue complete: 1 done, 2 failed`) before the machine went down.
- **ENS14_2023 (14-seed, champion window) completed fully and its results are
  trustworthy.** Panel, queue result block, JSONL entry, and run log all agree.
- **ENS14_2021 crashed mid-run** (CUDA `unknown error` at wf 4/11) and
  **BEAR_ENS14_2023 never ran** (same error immediately, GPU already dead).
  Both are already marked `failed` in the queue — their metrics are **INVALID**
  and must not be used.
- The CUDA "unknown error" at 01:42:03 preceded the shutdown — this looks like
  a GPU/driver-level failure that took the machine down, not a clean power cut
  mid-write. The scheduler survived long enough to finalize all files.

## Git state

- Branch: `research/continuous-alpha-loop-4060ti` ✓ (as expected)
- HEAD: `7f77864` — "research(loop): v7 verdicts - seed-sensitivity flag, refs
  restated as ranges" (2026-07-25 00:12:55 +0800)
- Uncommitted (exactly the three expected files, preserved untouched):
  - `reports/continuous_research/gpu_scheduler/queue_v8.json` (M)
  - `reports/continuous_research/gpu_scheduler/scheduler_log.jsonl` (M)
  - `reports/continuous_research/gpu_scheduler/scheduler_run6.log` (M)
- Diff vs HEAD is exactly what a healthy run would produce: ENS14_2023
  `running` → `done` + full result block; two `failed` entries with error
  strings; 3 JSONL lines appended; 33 log lines appended. Nothing anomalous.

## Timeline of the final run (scheduler_run6.log + scheduler_log.jsonl)

| Time (local) | Event |
|---|---|
| 00:12:22 | ENS14_2023 started (14 seeds, OOS 2023→) |
| ~01:17:19 | ENS14_2023 **completed** (3897.1 s, 7/7 refits, panel written 01:17:12) |
| ~01:17 | ENS14_2021 started (14 seeds, OOS 2021→, 11 refits planned) |
| ~01:37 | wf 3/11 (2022-01-13) finished — last healthy GPU work |
| 01:42:03 | ENS14_2021 **FAILED** during wf 4/11: `AcceleratorError: CUDA error: unknown error` |
| 01:42:03 | BEAR_ENS14_2023 auto-spawned, **FAILED immediately** with the same CUDA error (GPU unusable) |
| 01:42:03 | Scheduler wrote final queue state and exited: `queue complete: 1 done, 2 failed` |
| after 01:42 | Machine shut down (no files touched after 01:42:03) |

## Queue v8 status: completed-with-failures, not truncated

`queue_v8.json` is valid JSON and was finalized by the scheduler itself:

1. **ENS14_2023** — `done`. VALID (see integrity checks below).
2. **ENS14_2021** — `failed` (CUDA unknown error at wf 4/11). INVALID.
3. **BEAR_ENS14_2023** — `failed` (0 refits, never trained). INVALID.

## Last fully completed experiment

**ENS14_2023** (14-seed ensemble, champion window 2023→2026). Headline blend
(band10): LS net60 Sharpe **2.125** (net0 2.353, DD −10.1%), LO net60 **1.97**.
Raw 14-seed book: LS net60 1.679. mean_val_ic 0.04829 over 7 refits,
peak VRAM 2160 MB, 3897 s.

Gate context: expected blend in [1.85, 2.15]; 2.125 is in-range and at/above
the seed-set midpoint (~2.00) for this window. **The adopt-14 gate requires
BOTH windows, and the bear window never completed — so the gate is
undecidable and the 7-seed spec stays adopted by default (per the
pre-registered "else 7 seeds stay" clause).**

## Experiment interrupted by the shutdown

**ENS14_2021** — completed only wf 1–3 of 11 (val_ic +0.1358 / +0.1500 /
+0.0092), then CUDA failure. No result block, no panel, no JSONL success
entry. The three partial per-refit val_ic values in `scheduler_run6.log` are
**not decision-grade and are hereby marked INVALID** — do not cite them.
**BEAR_ENS14_2023** produced zero output (also INVALID, trivially).

## Integrity checks performed

| Check | Result |
|---|---|
| `queue_v5/v6/v7/v8.json` parse as JSON | all VALID |
| `scheduler_log.jsonl` — 32 lines, each parses; ends with newline | VALID, not truncated |
| `scheduler_run6.log` — ends with scheduler's own completion line | complete, not truncated |
| `SCHED_ENS14_2023.csv.gz` — `gzip -t` | OK (91,277 rows, 2023-01-03 → 2026-07-24) |
| ENS14_2023 cross-consistency (queue result vs JSONL vs run log) | val_ic 0.04829, elapsed 3897.1 s, VRAM 2160.1 MB, blend Sharpes — all three sources agree |
| Panels for failed runs | correctly absent (no orphan partials) |
| Repo-wide scan for files modified after 01:17 (excl. caches) | only the 3 scheduler files (01:42:03) — no stray partial outputs |
| `RUN_STATUS_CONTINUOUS_RESEARCH.md` | intact; last entry 07-24 20:30 (pre-dates v8 — needs a post-recovery entry, not yet written per instructions) |
| `RESEARCH_SCOREBOARD.md` | intact; committed at 7f77864; no v8 rows yet (correct — v8 unresolved) |
| Checkpoint `CONTINUOUS_RESEARCH_CHECKPOINT_20260725_0040.md` | intact; describes v8 as "running" (superseded by this audit) |

No repaired copies were created because no file was found partially written.

## Post-audit smoke tests (all PASS)

- `.venv` python 3.12.3, numpy 2.4.4, pandas 3.0.3, torch 2.13.0.dev20260522+cu132
- `torch.cuda.is_available()` → True; RTX 4060 Ti detected; 1024×1024 GPU
  matmul OK; AMP autocast matmul OK; VRAM 15.9/17.2 GB free
- `nvidia-smi`: driver 591.86, 47 °C, healthy after reboot
- `import gpu_research_scheduler` (pulls in dataset_transformer_eod,
  train_transformer_eod, transformer_portfolio) → OK; PRESETS A/B/C present;
  `require_cuda()` → OK

The CUDA `unknown error` does not reproduce post-reboot — consistent with a
transient driver/GPU wedge that the reboot cleared.

## Recommendation (pending user approval — nothing resumed yet)

**Resume queue v8; do not discard, do not fully rerun.**

- ENS14_2023 is complete and verified — rerunning it would waste ~65 min of
  GPU for a determinism-confirmed pipeline (spawn-inherit bug #3 audit showed
  reruns reproduce).
- Exact resume point: reset `ENS14_2021` and `BEAR_ENS14_2023` from `failed`
  to `pending` (keep ENS14_2023 `done`), then restart the scheduler on
  `queue_v8.json`. Its crash-safe design skips `done` items.
- Precondition: the CUDA `unknown error` + shutdown suggests a driver/GPU-level
  fault. Before resuming: verify `nvidia-smi` is healthy and run a short CUDA
  smoke test (tensor op + one tiny training step). If the error recurs,
  investigate driver/thermals/PSU before any long run.
- Keep-alive note: ENS14_2021 is the longest queued run (~150 min expected);
  if the shutdown was thermal/power related, consider running it with the
  GPU otherwise idle (no FFXIV) as a discriminating test.
