# Queue v12 — Big LSTM+Transformer / Recency / Epoch Depth / Long-Short / Deep Inference

**PLAN ONLY. Pre-registered 2026-07-30. No phase runs without explicit user
approval; Phases 2/3/4/5 each require their own approval.**
Branch `research/v12-big-transformer-recency-short`.
Runner (to be implemented after plan approval):
`research/run_queue_v12_big_transformer.py`.
Outputs: `reports/continuous_research/v12_big_transformer/`.

Research questions (from the user's brief): (1) can a much larger
LSTM+Transformer beat production by ≥1% relative at book level; (2) do more
epochs help or overfit; (3) should recent data weigh more; (4) should
training exclude data older than 1/2/3 years; (5) does deep GPU inference
add rank-stability/uncertainty value; (6) are useful long-short/short
signals available; (7) can any of it improve book metrics without worsening
DD, turnover, bear, or 2022.

Standing context the plan must respect: v11 just rejected h96–h128 /
L3 / seq90 at book level; equal-history beat recency in the early sprints;
seven recorded val-IC/book dissociations. v12 is a deliberate, deeper
re-examination of those axes at much larger scale — expected-loss but
worth-knowing research, pre-registered so hindsight can't move the bars.

## 1. Model bands (estimator-verified, exact param counts)

1 GiB fp32 = 268,435,456 params. Full table in
`model_capacity_estimates.{csv,md}`. Selected candidates:

| band | config (seq60, close_only) | params | ckpt fp32 | est. train VRAM | est. min/epoch |
|---|---|---|---|---|---|
| S | h64/L2/H4/ff128 (production) | 113,857 | 0.45 MB | 2 GB (measured) | 0.2 (measured) |
| M2 | h256/L4/H8/ff1024 | 3.79 M | 14.5 MB | ~1.5 GB | ~1 |
| M3 | h256/L6/H8/ff1024 | 5.37 M | 20.5 MB | ~1.7 GB | ~1 |
| L1 | h512/L8/H8/ff2048 | 27.7 M | 105 MB | ~3.1 GB | 4–8 |
| L3 | h768/L8/H12/ff3072 | 62.1 M | 237 MB | ~4.5 GB | 9–17 |
| XL1 | h1024/L16/H16/ff4096 | 211.1 M | 805 MB | ~11 GB | 29–58 |
| **XL2** | **h1024/L24/H16/ff4096** | **311.9 M** | **1.19 GB ≥ target** | ~15.8 GB → needs micro-batch 128 + grad-ckpt | 43–86 |

**1 GB feasibility verdict:** trainable on this GPU (XL2 fits with AMP +
micro-batch ≤128 + gradient accumulation + gradient checkpointing; XL1 at
805 MB fits comfortably), **but a full A8 walkforward at XL scale is
infeasible** (25 epochs × 7 seeds × 8 refits ≈ weeks). XL therefore runs a
pre-registered REDUCED protocol: 1 seed, single latest refit, epoch budget
fitted to ~5 h (≈6–8 epochs XL2, ≈8–12 XL1), val-IC learning curve + a
single-refit book diagnostic. This makes the 1 GB checkpoint a real,
measured artifact and gives a first signal read — while the *decision-grade*
model-size evidence comes from M/L bands, which can run the genuine
walkforward protocol. XL cannot be promoted directly from the reduced
protocol; an XL promotion path would require an L-band success first.

## 2. Phases (each launch gated on user approval)

- **Phase 0 — baseline anchor.** V11's Phase-0 panels (2.147 CH / 1.443 BR,
  exact reproduction, 2026-07-30) are reused as the anchor — re-running
  costs 2 h and the panels are on disk. A drift re-check runs only if the
  cache has been refreshed since.
- **Phase 1 — feasibility smoke (~1–2 h GPU).** Estimator table (done);
  instantiate M2/L1/XL1/XL2; tiny fit (1 seed, 2 epochs, micro-batched)
  per band; measure real epoch time + peak VRAM vs estimates; save/reload
  each checkpoint via the daily format incl. the ~1.19 GB XL2 artifact;
  inference-compat check; OOM-backoff exercise; verify zero production
  files touched.
- **Phase 2 — recency / window / epoch screen (CPU-light GPU, S-band
  model only, 3 seeds, CH window).** Grid below, evaluated at book level.
  Rationale: recipe questions are answered fastest on the cheap model, and
  v12's big models then train with the winning recipe only.
  - Hard windows: 1y / 2y / 3y / all-history (baseline).
  - Exponential half-life: 63 / 126 / 252 / 504 trading days.
  - Hybrid: 3y window × half-life 126 and 252.
  - Calendar-aware bands: 6m=1.0, 6–12m=0.75, 1–2y=0.5, 2–3y=0.25, >3y=0.
  - Epoch axis (on the best-2 window recipes + baseline): max_epochs
    25/50/100 × patience 3/5/10/20 (selected cells, not full cross), plus
    min_epochs 10/25 floors on the 50-epoch cells; per-epoch learning
    curves (train loss, val IC, epoch time, peak VRAM) logged to
    `epoch_curves.csv`; overfit diagnostics (val-IC peak epoch, best-vs-last
    checkpoint) to `epoch_depth_report.md`.
  - Windows/weights are built as per-sample weight vectors via the EXISTING
    `fit_one(weights=...)` path — data_cache untouched, no cache pruning;
    any derived recent panels land in
    `reports/transformer_gpu/v12_big_transformer/recent_panels/` (gitignored).
- **Phase 3 — big-model screen (3 seeds, CH, winning recipe from Phase 2 or
  baseline recipe if Phase 2 rejects everything).** M2, M3, L1, L3 full
  walkforward screens (feasible: 1–17 min/epoch); XL1 + XL2 reduced
  protocol as defined above. Includes the largest feasible checkpoint
  candidate (XL2, 1.19 GB) by construction.
- **Phase 4 — confirmation (top ≤2 from Phase 3, 7 seeds, dual windows CH
  2023→ / BR 2021→).** Full book-level validation, long-only AND
  long-short evaluation, disjoint-seed battery (10–16) required before any
  adoption talk (standing meta-rule 2).
- **Phase 5 — deep inference (up to 5 h, separate approval).** Runs against
  the production 7-seed ensemble regardless of Phase-4 outcome (it is an
  inference-layer question), plus any Phase-4 survivor. Details §4.

## 3. Long-short / short-account research (Task 7; CPU, panel-based)

Evaluated on saved panels (no new training): production tf panel, D1.2,
blend50 — each in configurations 100/0 (long-only baseline), 100/50,
100/100, market-neutral (net 0), and a short-only diagnostic list
(review-only, never an order list).

- Short-leg attribution: short-leg return/hit-rate/DD in isolation,
  maximum adverse excursion, worst-5 short positions, short-side sector
  and top-contributor concentration.
- **Squeeze/borrow proxy (labeled PROXY — real borrow data unavailable):**
  exclude/flag names with high short-side risk = small ADV (bottom
  liquidity tercile by 20d median close×volume), extreme 20d up-moves
  (rally-squeeze proxy), and price < a floor. Conservative costs: short
  leg charged 2× the long-side cost curve (borrow+locate margin) in a
  sensitivity table at 60/100/150/200 bps.
- Constraints grid: single-name short cap 5%, sector short cap 20%,
  liquidity cap (position ≤ 5% of 20d ADV), net/gross exposure reported
  per configuration.
- Gates: short leg must add Sharpe vs the 100/0 baseline AFTER the 2×
  cost assumption, survive the bear window and the 2021 rally, not blow
  the squeeze proxy, and keep short-leg DD and turnover acceptable. A good
  long model is NOT assumed to short well; the short leg must prove value
  independently (long leg frozen while short leg is ablated).
- Outputs: `short_side_report.md`, `short_side_metrics.csv`,
  `short_candidate_diagnostics.csv`.

## 4. Deep inference design (Task 8 — `research/deep_inference_v12.py`)

Modes (composable, `--max-inference-hours 5` hard budget, stops cleanly):
deterministic single-pass (control) · 7-seed ensemble (current production
behavior, baseline) · multi-snapshot ensemble (best-by-val + last + epoch
snapshots from v12 training) · multi-window (latest vs previous refit
checkpoints) · MC-dropout (N configurable, uncertainty only) · input
perturbation (price noise σ∈{0.1%,0.5%}, drop-last-k-days k∈{1,3,5},
truncated-history replays). Per-stock outputs: mean/std score, rank
stability across passes, top-quintile inclusion probability,
bottom-quintile inclusion probability (short candidates), seed and
snapshot disagreement, high-confidence long/short lists and an
unstable-avoid list. Book-level comparison vs normal inference decides
value: **if the deep-inference book is not measurably better (or the
uncertainty signal does not correlate with realized rank error), verdict
REJECT_DEEP_INFERENCE — burning 5 h must buy information.**
Output: `queue_v12_deep_inference_report.md` + deep score panel (gitignored).

## 5. Validation gates (Task 9, pre-registered)

Adoption consideration requires ALL of: ≥ **+1% relative** OOS blend
book Sharpe after costs vs the V11/V12 Phase-0 baseline (i.e. CH ≥ 2.169
given 2.147); max DD not worse; bear-window Sharpe not worse; 2022 not
worse by >0.15; turnover within +20%; no concentration blow-up (top-5
contributor share, sector shares vs baseline); stable across seeds AND the
disjoint battery; daily runtime within the 5 h budget. Immediate rejects:
val-IC-only edges; sub-noise margins (< the ±0.10 seed-range width);
DD-degrading wins; frequent OOM; checkpoint-large-but-book-worse;
recency that wins only in-sample/val; short books failing the cost/squeeze
gates. Explicit non-criteria (recorded verbatim): **1 GB is not enough,
epochs are not enough, recency is not enough, short capability is not
enough, val IC is never enough.**

## 6. Runner and outputs (Tasks 10–11)

`research/run_queue_v12_big_transformer.py` with commands: inspect ·
estimate · materialize · dry-run · smoke · run --phase 0..4 · report ·
status (deep inference lives in `deep_inference_v12.py`, Phase 5).
Feasibility features (research-only): AMP (already on) · gradient
checkpointing (encoder-layer wrap) · gradient accumulation
(micro_batch/accum_steps/effective_batch in configs) · OOM catch →
empty_cache → halve micro-batch → retry → mark failed · full VRAM/runtime
logging per epoch and per candidate. No DataLoader changes: the dataset is
GPU-resident by design (pin_memory/num_workers not applicable — documented
in the capacity report).
Outputs under `reports/continuous_research/v12_big_transformer/`: the 18
files enumerated in the task (capacity report + estimates done; plan =
this file; manifest/results/gpu_usage/book_metrics/recency/epoch/
short-side/deep-inference/verdict produced by their phases).
Verdict vocabulary: KEEP_CURRENT_PRODUCTION · REJECT_BIG_MODEL ·
REJECT_RECENCY_WEIGHTING · REJECT_EPOCH_EXPANSION · REJECT_SHORT_SIDE ·
REJECT_DEEP_INFERENCE · PROMOTE_TO_FURTHER_VALIDATION ·
PROMOTE_TO_PAPER_ONLY · PROMOTE_TO_PRODUCTION_CANDIDATE (axis verdicts are
independent — e.g. big model rejected while short-side passes). Nothing
promotes automatically.

## 7. Safety

Production untouched: daily retrain defaults, `checkpoints/transformer_eod/`,
daily_manifest, blend50_band10, paper ledger, holdings overlay, data_cache
(read-only; no deletion of historical cache files — any pruning requires
explicit user approval and would happen as derived research panels, never
in place). Research paths: `checkpoints/v12_big_transformer/`,
`reports/transformer_gpu/v12_big_transformer/`, v12 report dir + logs/ —
all large artifacts gitignored (`.gitignore` updated: v12 dirs, logs,
*.pt/*.pth/*.ckpt, derived panels). CUDA unknown error → stop, hardware
note, no retry loops. Commits only of small summary/doc files, only after
explicit user approval.
