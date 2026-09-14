# 12 — Prioritized challenger matrix, compute plan, promotion standard (revised)

**Status: PROPOSED — no GPU experiment has been run. Awaiting review.**
Hardware: RTX 4060 Ti 16 GB, torch 2.13 nightly cu132; production uses
≤ 4.2 GB; nothing below changes the CUDA environment.

## Guiding conclusions (what the audits established)

- The transformer is ~65% momentum; residual IC beyond momentum + linear
  ≈ 0.02 (CH) / 0.00 (BR) → **architecture scaling is not proposed** (11).
- Production's daily full refit was never validated → **P0 first** (15).
- Labels/features carry a material (qualitative, candidate-detector-based) seasonal corporate-action bias →
  **P1 before target-horizon research** (02).
- Survivor universe → all references are upper bounds; P2 is a data task.
- IC and top−bottom spread rise 20d→40d while the book holds for months
  → **P3 is the one model hypothesis with positive evidence** (04).

## Priority order and budget

| P | Item | What runs | GPU | Gate to proceed |
|---|---|---|---|---|
| **P0** | Production-parity: denser refit vs validated 126-refit, PAIRED seeds (design in 15) | P0-A′ seeds {0,1,2}: 126 vs 5-session cadence over 2026-H1 (≈1.5–2 h) → P0-A seeds {0,1,2}: 126 vs daily (≈7.4 h) → P0-B seeds {0..6}: 126 vs daily (≈17 h reduced) only if P0-A differs materially | **≈ 9 h, up to 26 h** | outcome PARITY_OK vs PRODUCTION_RESEARCH_PARITY_RISK decides whether an operational cadence change is proposed before any model research |
| **P1** | Corporate-action correctness | 0 GPU: acquire dividend/ex-rights table → adjustment factors → isolated research cache → re-score the frozen champion panels on adjusted returns (CPU) | 0 (data task) | quantifies the bias on the references; defines whether P3/P4 must be run on adjusted data |
| **P2** | Point-in-time universe requirements | 0 GPU: data acquisition spec (03); no modelling | 0 | labels stay "upper bound" until done |
| **P3** | 40-session-target challenger (train `tgt_rank_40`, same arch/features/seeds/walk-forward/costs), evaluated at 20- AND 40-session rebalance | 3-seed CH screen (≈45 min) → 7-seed CH+BR + disjoint seeds (≈2.5 h × 2) | **≈ 6 h** | after P0 result and P1 bias estimate; run on adjusted data if P1 says the bias moves ranks |
| **P4** | Training recency: (A) current policy, (B) expanding window with only the required 21-purge+val block held out, then refit on all data at a fixed epoch budget, (C) rolling recent-N-year window (N = 3, 5) — (D) recency weighting only if B/C justify it (v12 rejected 10 forms) | 3-seed CH screens (≈45 min each × 3) → best two at 7 seeds dual-window (≈5 h) | **≈ 7 h** | after P0 |
| **P5** | Simple baselines | **done (CPU)**: momentum, D1.2, ridge, tf, blend; GBM deferred (no sklearn/lightgbm in venv — requires an approved dependency) | 0 | informs P3/P4 expectations |
| **P6** | Refit-cadence optimization (weekly / monthly / 126) | only if P0 shows a cadence effect; reuse P0 panels | 0–4 h | after P0 |
| **P7** | Position-exit overlays on the model's paper positions (entry dates known) | CPU on frozen panels | 0 | when P1-adjusted returns exist (overlays are sensitive to ex-div drops) |

**Total proposed GPU budget: ≈ 22 h baseline (P0-A′/A 9 h + P3 6 h + P4 7 h), up to ≈ 39 h if reduced P0-B is triggered** — 3–5 overnight sessions, none launched before review, none overlapping the 22:00 daily_ops window. Recommended reordering: P5 is already done, so P3 and P4 run after P0 and the P1 bias estimate; P6/P7 remain conditional.

Not proposed (archived negatives respected): hidden/depth/FFN scaling,
21 seeds, seq 90/120, feature-rich XL, exotic ranking losses, cross-
sectional attention, TCN, multi-task 5/10/20, recency weighting as a
first-line change.

## Champion promotion standard (unchanged in substance, restated)

Production remains CHAMPION throughout. A challenger may enter **shadow
mode** only if ALL hold on both CH and BR with disjoint-seed replication
within 0.15 Sharpe:
1. mean OOS rank IC at the 20-session evaluation horizon ≥ champion + 0.005 and positive-IC share ≥ champion's;
2. long-only net100 Sharpe ≥ champion bootstrap p50 (CH 2.06 / BR 1.47);
3. max DD ≤ champion + 2 pp or Calmar ≥ 1.1× champion;
4. one-way turnover ≤ 1.5× champion;
5. BEAR-trend and NARROW-breadth IC not worse by > 0.02; 2022 net Sharpe ≥ champion's;
6. no val-IC-only win (dissociation tripwire).
Then ≥ 3 independent 20-session prospective blocks in shadow (both models
score daily; only the champion drives the book) before promotion is
reconsidered on prospective data. Historical winners are never promoted
directly.
