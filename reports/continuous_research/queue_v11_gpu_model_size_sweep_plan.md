# Queue v11 — GPU Model-Size Sweep (pre-registered plan)

Status: **PLAN ONLY — no full GPU runs until user approval.**
Branch: `research/v11-gpu-model-size-sweep`. Date: 2026-07-30.
Runner: `research/run_queue_v11.py`. Outputs:
`reports/continuous_research/v11_gpu_model_size_sweep/`.

Research question (single axis): does a larger or reshaped
LSTM_CondTransformer (width / depth / sequence length) improve **book-level
OOS performance** of the production blend, at the byte-identical A8 training
protocol? Secondary: instrument GPU utilization and determine whether a
larger batch (target ~9–11 GB VRAM) is a safe throughput win.

Explicitly out of scope (per user rules C–F): feature sets other than
close_only, new targets, new portfolio rules, combined questions.

## 1. Pipeline inspection summary (Task 1)

- **Architecture:** `model.py` `LSTM_CondTransformer` (LSTM → proj+pos-emb →
  cross-attn → TransformerEncoder → MLP head). Untouched by this queue.
- **Hyperparameters:** all from `PRESETS` (`train_transformer_eod.py:38-45`);
  production preset B = h64/L2/H4/ff128/do0.2/seq60. The scheduler's
  `_preset_for()` already supports per-run `overrides` via derived presets —
  no new mechanism needed for this sweep.
- **Seeds:** production 7-seed hardcoded in `mode_daily_retrain` (line 547);
  research runs pass explicit seed lists.
- **Checkpoints:** production `checkpoints/transformer_eod/daily_seed{s}.pt`
  (+ manifest); inference (`load_ensemble`) rebuilds nets from the SAVED cfg,
  so any preset shape round-trips automatically. v11 research checkpoints go
  to `checkpoints/v11_sweep/` (gitignored) and never touch the daily files.
- **AMP:** already mandatory-on for CUDA (`torch.amp.autocast` +
  `GradScaler`, `train_transformer_eod.py:250,304,339`). No change needed;
  disabling AMP is NOT offered (it would itself be a protocol change).
- **Batch size:** fixed `fit_one` kwarg (1024). v11 adds a flag-gated preset
  key `batch` (absent from A/B/C → default path **verified bit-identical**
  by pre/post anchor: seeds 0–1 val ICs 0.11329/0.12393 and panel checksums
  byte-equal before and after the edit).
- **GPU memory logging:** `peak_vram_mb` (max_memory_allocated) already in
  every walkforward result; v11 adds max_reserved, total VRAM, per-rung
  batch/VRAM rows.
- **OOM handling:** scheduler `run_one` retries once with half the seeds;
  v11's batch probe adds `empty_cache` + backoff so no probe rung can kill
  the queue.

## 2. Candidates (Task 2)

All candidates: `close_only`, `tgt_rank_20`, horizon 20, refit 126, MSE,
equal-weight history, lr 3e-4, wd 1e-4, **batch 1024**, seeds 0–6 at
confirmation stage, dual windows CH (oos 2023-01-01) and BR (oos
2021-01-01), blend+band10 book evaluation — i.e. the A8 protocol with only
the architecture axis varied.

| id | hidden | layers | heads | ff | seq_len | note |
|---|---|---|---|---|---|---|
| V11_A_baseline | 64 | 2 | 4 | 128 | 60 | baseline reproduction + env-drift check |
| V11_B_wider96 | 96 | 2 | 4 | 192 | 60 | width axis, untested point |
| V11_C_wider128 | 128 | 2 | **8** | 256 | 60 | heads=8 chosen: keeps head_dim=16 as in production h64/H4; the h128/H4/ff256 shape already exists as preset C (standalone DD-alternative, Sharpe 1.58 / DD −10.5%) so H8 is the new information |
| V11_D_deeper64 | 64 | 3 | 4 | 128 | 60 | depth axis |
| V11_E_longer90 | 64 | 2 | 4 | 128 | 90 | **re-run of a CLOSED axis** (scoreboard P5s: seq90 5-seed blend 1.815 with family-best val IC 0.054 — rejected). Included at user request; treat as replication, expected to fail |
| V11_F_wider96_longer90 | 96 | 2 | 4 | 192 | 90 | width×length interaction — genuinely new cell despite seq90 prior |
| V11_G_wider128_deeper3 | 128 | 3 | 8 | 256 | 60 | OPTIONAL — runs only if B–F train stably (no OOM retries, no divergence) |

Prior-evidence flags (pre-registered so hindsight can't move the bar):
seq90 lost to seq60 in v1–v7 AND in the P5s 5-seed promotion; preset C
(h128) was kept only as a standalone-DD alternative, never a blend winner.

## 3. Execution protocol

**Recommended: phased (screens select, never adopt — meta-rule 3).**

- **Phase 0 — baseline anchor:** V11_A at 7 seeds, both windows. Doubles as
  environment-drift check: CH blend net60 must land in 1.85–2.15 (single-set
  ref 2.147) and BR in 1.30–1.45 (ref 1.443), else STOP and investigate
  before any candidate runs.
- **Phase 1 — screen:** A + B–F (G held) at **3 seeds (0–2), CH window
  only**, blend book evaluated. A's own 3-seed screen is the comparison
  point (3-seed candidate vs 7-seed baseline would confound ensemble size).
- **Phase 2 — confirm:** top ≤2 candidates whose screen blend CH net60 ≥
  V11_A 3-seed screen value, promoted to 7 seeds × dual windows.
- **Phase 3 — seed-robustness battery (mandatory before any adoption
  talk, meta-rule 2):** any Phase-2 gate-passer re-runs with disjoint seeds
  10–16 on both windows.

Alternative (user-specified literal): all 7 candidates × 7 seeds × 2
windows ≈ 15–25 h GPU. The phased path covers the same decision space in
≈ 8–12 h and is the default proposal. **The user chooses at approval time.**

Calibration (v8 records, h64): 7-seed CH ≈ 33 min, 7-seed BR ≈ 80 min,
peak VRAM ≈ 2.2 GB. Width/depth/length multipliers measured by the smoke
probe before the full run.

## 4. GPU utilization / VRAM track (Task 3)

Separate from the candidate comparison — **batch stays 1024 in all
candidate runs** because batch size is part of the optimization recipe;
mixing it into the sweep would confound the architecture axis.

- Instrumentation logged per run: GPU name, total VRAM, max allocated, max
  reserved, batch, AMP flag, per-seed train time, total time
  (`queue_v11_gpu_usage.csv`).
- Auto-batch probe (`run_queue_v11.py probe`): batch ladder 1024 → 2048 →
  4096 → 8192 → 12288 → 16384, one short fit per rung, stop when peak
  allocated reaches `--target-vram-gb` (default 10, range 9–11) or on OOM
  (empty_cache → back off → record → continue). Never crashes the queue.
- If a larger batch reaches the VRAM target with good throughput, adopting
  it for RESEARCH runs requires a val-IC + book parity check at that batch
  first (a batch change is a recipe change; lr scaling questions belong to
  a future pre-registered queue, not this one). Production daily retrain
  stays at 1024 regardless.
- Making the model artificially large to burn VRAM is explicitly not a goal.

## 5. Validation metrics per candidate (Task 4)

From the standard panel/book machinery (`run_one` → `backtest_scores`):
CH & BR long-short and long-only net60 (+net0/net100/net150 cost curve),
Sharpe, max DD, avg turnover, rank IC, yearly table incl. 2022; blend book
vs standalone; top-quintile overlap + holdings Jaccard vs V11_A panel;
sector concentration and top-contributor share; n names; per-seed val-IC
spread. Operational: train time, peak VRAM (alloc+reserved), checkpoint
size, and an explicit **inference-compat test** (save seed-0 checkpoint in
the daily format → rebuild via `build_net_from_ck` → score latest date →
finite scores required).

## 6. Acceptance gates (Task 5, pre-registered)

Immediate reject if any of: val-IC-only improvement (meta-rule 1); bear DD
>1pp worse than baseline without ≥0.10 bear Sharpe gain; 2022 blend Sharpe
worse than baseline by >0.15; blend turnover >+20% vs baseline; one-seed
dependence (screen edge vanishes when best seed dropped); unstable training
(divergence / repeated OOM retries); inference-compat failure; any change
required to the production pipeline beyond a preset entry.

Adoption consideration requires beating or matching the V11_A 7-seed
baseline on ALL of: CH L/S net60 Sharpe, BR Sharpe, max DD, 2022, turnover-
adjusted net (net100), seed robustness (incl. Phase-3 disjoint battery),
operational feasibility. Anything less →
`KEEP_CURRENT_PRODUCTION` / `REJECT_ALL_CHALLENGERS`.

Verdict vocabulary (queue_v11_verdict.md): KEEP_CURRENT_PRODUCTION ·
PROMOTE_CHALLENGER_FOR_FURTHER_VALIDATION · PROMOTE_CHALLENGER_TO_PAPER_ONLY
· REJECT_ALL_CHALLENGERS. **No automatic promotion to production in any
outcome; adoption is always a separate user decision.**

## 7. Outputs (Task 6)

`reports/continuous_research/v11_gpu_model_size_sweep/`:
queue_v11_plan.md (copy of this file) · queue_v11.json (crash-safe state) ·
queue_v11_results.{csv,md} · queue_v11_gpu_usage.csv ·
queue_v11_book_metrics.csv · queue_v11_verdict.md ·
queue_v11_run_manifest.json · logs/<item>.log per candidate run.
Panels continue to land in `reports/transformer_gpu/panels/` (gitignored);
research checkpoints in `checkpoints/v11_sweep/` (gitignored).

## 8. Safety rails

Production defaults, daily decision book, blend50_band10, paper ledger,
holdings overlay, data_cache: untouched. The only shared-module change is
the anchor-verified `batch` preset key. CUDA `unknown error` → stop the
queue and write a hardware note (no retry loops). Commits only at clean
stop points after user approval, never including data_cache CSVs or
checkpoints.
