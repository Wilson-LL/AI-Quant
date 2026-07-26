# Queue v10 Proposal — GPU Research Queue (New Signal / Model Axes)

Status: **PROPOSED — NOT STARTED** (2026-07-26). Awaiting user approval.

Framing per user direction: genuinely new signal/model research, not
deployment construction. Every axis below is one the loop has NOT tested —
closed lines (seq length, layers, dropout/wd, ranking losses, risk-proxy /
sector-excess targets, regime features, extra feature sets, seed counts,
recency, warm-start, rolling windows, preset-diversity blends, confidence
filtering, exposure gates) stay closed.

Standing gates, unchanged: val-IC is the selection metric (within-family
only — the B4/B6 inversion lesson); dual-window OOS vs seed-robust ranges
**1.85–2.15 (2023→) / 1.30–1.45 (2021→)** with bootstrap medians 1.92/1.37 as
planning numbers; blend books decide, hard caps enforced; 3-seed screens are
known-unreliable → any promoted config re-runs at 7 seeds before judgment.
Scheduler `_run_key` dedup is active — no config may duplicate a prior run.

Hardware envelope: RTX 4060 Ti 16 GB; screens (3-seed, champion window)
~15–30 min; full 7-seed dual-window runs ~2.5–4 h. Proposed total ≈ 20–28 h
GPU if every screen promotes (realistically less — most screens die).

## Phase A — architecture family (the core question)

The champion is a per-stock temporal transformer. The untested inductive
bias is CROSS-SECTIONAL: letting names see each other at prediction time.

### A1 — cross-sectional attention head (CS1)
1. **Hypothesis:** attending across the ~108-name cross-section at each date
   (one attention block over per-stock embeddings, before the head) captures
   relative-value structure the per-stock model cannot; val IC ≥ champion's
   0.050 and OOS blend within tolerance of refs.
2. **Config:** champion trunk frozen-ish (preset B, close_only, seq60,
   rank-20) + 1 cross-sectional attention layer on the date's embedding
   batch; 3-seed screen, champion window.
3. **Runtime:** ~40 min screen (cross-attention adds ~20%).
4. **Gate:** val IC ≥ 0.050 to promote (within-family comparison vs A2/A3
   screens); promoted → 7-seed dual-window + blend books vs ranges.
5. **Baseline:** champion 7-seed (val IC 0.050, blend 2.147/1.443).
6. **GPU-heavy.** Requires a model edit (new module) — signal generation
   changes are the POINT of v10, per user framing.
7. **Outputs:** SCHED panels + queue results as usual.
8. **Overfit risk:** moderate — new capacity; controlled by val-IC gate and
   the 7-seed re-run rule.
9. **Why:** the single most-cited modern result on equity panels
   (iTransformer-style variate attention) and the largest untested
   architecture axis available on this hardware.

### A2 — inverted attention variant (CS2)
Same family as A1 (mutually exclusive adoption): attention ONLY across names
(tokens = stocks, channels = time features), no temporal attention. 3-seed
screen; same gate/baseline as A1. ~30 min. Tests whether temporal attention
is even needed once cross-sectional structure is available.

### A3 — TCN baseline (sanity anchor)
1. **Hypothesis (expected: loses):** a small temporal-convolution net at
   matched parameter count cannot reach champion val IC — attention earns
   its cost. If it CAN, that's important negative information about the
   champion's complexity.
2. **Config:** 4-block dilated TCN, close_only seq60 rank-20, 3 seeds.
3. **Runtime:** ~20 min. **Gate:** descriptive anchor; promotes only if val
   IC beats champion (unexpected). **Overfit risk:** low. **Why:** every
   architecture claim in the scoreboard is attention-vs-attention; this
   anchors the family externally.

## Phase B — head / objective (model-native, not loss-swap)

### B1 — multi-task heads (MT1)
1. **Hypothesis:** jointly predicting rank-20 + rank-5 + 20d realized vol
   (shared trunk, 3 heads, fixed loss weights 1.0/0.3/0.3) regularizes the
   trunk and lifts rank-20 val IC. Distinct from the closed multi-horizon
   *ensemble* line (B2): this is joint TRAINING, one model.
2. **Config:** preset B + 2 aux heads; 3-seed screen; aux weights
   pre-registered (no weight search — one cell).
3. **Runtime:** ~30 min. **Gate:** val IC vs champion; promote → 7-seed
   dual-window. **Overfit risk:** low-moderate (single pre-registered cell).
4. **Why:** the cheapest mainstream trick not yet tried; clean single-cell test.

### B2 — quantile head for uncertainty-aware sizing (Q1)
1. **Hypothesis:** a pinball-loss quantile head (q10/q50/q90) gives
   model-native uncertainty; sizing by q50 with q90−q10 width as a weight
   damper beats point-estimate sizing on book metrics at equal val IC.
   Distinct from closed A2b (seed-disagreement filtering — that was
   ensemble-external; this is model-internal).
2. **Config:** champion trunk + quantile head, 3 seeds; book construction:
   damped weights w ∝ 1/width within the selected quintile (one rule,
   pre-registered).
3. **Runtime:** ~30 min screen + CPU book pass.
4. **Gate:** two-stage — val IC must hold (≥0.048), then blend books vs refs
   dual-window at 7 seeds. **Overfit risk:** moderate (sizing rule could
   fit noise; single rule mitigates). **Why:** uncertainty sizing is the
   only sizing lever never tested at the model level.

## Phase C — new signal families (cheap screens first)

### C1 — short-horizon reversal family (REV1)
1. **Hypothesis:** 5d reversal (z of −ret_5d) carries independent alpha on
   TWSE names; as a THIRD blend leg (tf + mom126 + rev5) it lifts the blend
   above the current range top without hurting the bear window.
2. **Config:** CPU screen first (no GPU): rev5 standalone + 3-way blend
   grids {10%,20%} rev weight on frozen panels. GPU only if the CPU screen
   passes: retrain with rev5 as an input feature (close_rev feature set)
   — 3-seed screen.
3. **Runtime:** CPU ~20 min; GPU branch ~25 min.
4. **Gate:** CPU: 3-way blend must beat 2-way on BOTH windows to justify the
   GPU branch. GPU: standard val-IC gate. **Overfit risk:** moderate (weight
   grid is small and pre-registered). **Why:** momentum is the only
   cross-sectional anomaly family in the book; reversal is its canonical
   complement and is untested.

### C2 — market-residualized target (RT1)
1. **Hypothesis (inversion-risk flagged):** rank of market-residual returns
   (name return minus beta×universe-proxy return) is a cleaner learning
   target than raw rank-20. Sector-excess (ES) closed REJECT; market-level
   residualization is untested and mechanically different.
2. **Config:** new target column (CPU precompute, cache read-only), preset
   B, 3 seeds. **Runtime:** ~25 min.
3. **Gate:** val IC within-family only, and a PRE-REGISTERED INVERSION CHECK
   (B4/B6 pattern): if val IC ≥ 0.055 but OOS IC < 0.02 → record inversion
   #3 and close the residual-target line permanently.
4. **Overfit risk:** the risk IS the val/OOS inversion — explicitly gated.
   **Why:** cheap, and either outcome (works / third inversion data point)
   is informative.

## Phase D — training procedure

### D1 — input augmentation (AUG1)
1. **Hypothesis:** feature-noise injection (σ=0.1 on z-scored inputs) or
   date-block dropout (drop 10% of training dates per epoch) reduces
   seed-set variance — the v7 finding (1.843 vs 2.147 across seed sets) —
   even if mean val IC is flat. Success metric is VARIANCE, not level.
2. **Config:** 2 cells (noise / date-dropout), 3 seeds ×2 seed-sets (0–2,
   10–12) to measure cross-set spread at screen scale.
3. **Runtime:** ~1.5 h (4 short runs). **Gate:** adopt path only if
   cross-set val-IC spread halves at flat-or-better mean; then 7-seed
   dual-window. **Overfit risk:** low (regularization). **Why:** directly
   attacks the loop's known largest fragility (seed-set sensitivity).

### D2 — ensemble distillation (DIST1, deployment-motivated but model-side)
1. **Hypothesis:** a single student trained on the 7-seed ensemble's soft
   scores recovers ≥95% of ensemble val IC at 1/7 inference+retrain cost.
2. **Config:** teacher = frozen 7-seed scores (existing checkpoints);
   student = preset B, MSE on teacher scores + true target (50/50), 3 seeds.
3. **Runtime:** ~30 min. **Gate:** student val IC ≥ 0.95× ensemble; OOS
   blend within 0.1 of 7-seed books dual-window. **Overfit risk:** low.
4. **Why:** if it works, daily retrain drops from ~5 to ~1 min and the
   deployment story simplifies; if not, quantifies what the ensemble buys.

## Sequencing & budget

Phase A screens first (A1/A2/A3 ≈ 1.5 h) — the architecture question gates
everything downstream. Then B1/B2 + C2 screens (~1.5 h), C1 CPU screen in
parallel, D1 (~1.5 h). Promotions (val-IC winners only, PROMOTE_TOP_K=2)
re-run at 7 seeds dual-window (~3–4 h each) with auto bear-spawn
(config-dedup active). D2 runs last, only if the 7-seed spec is still
champion after A/B/C verdicts. Worst case ≈ 28 h GPU; expected ≈ 12–16 h.

## Guardrails

- NOT STARTED until user approval; production untouched throughout v10.
- Model-code edits (new modules/heads) land behind flags; the champion
  training path stays byte-identical and is re-verified (determinism check)
  before any v10 run.
- Every adoption requires: within-family val-IC win → 7-seed re-run →
  dual-window blend gate vs ranges → explicit user sign-off.
- Closed lines stay closed; C2 carries a pre-registered inversion tripwire.
- Daily ops (P1 ledger, P2 diff) continue independently; GPU queue never
  runs during cache refresh (existing scheduler rule).
