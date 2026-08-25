# Current System State — 2026-07-27

Snapshot at the close of the v10 research arc (D2 rejection accepted by user;
arc closed). No GPU queue running; no experiments pending; loop is in
operational cadence.

## Repository

- **Branch:** `research/continuous-alpha-loop-4060ti`
- **Latest commit:** `1344795` — "D2 distillation OOS check - REJECT 0/6,
  line closed, 7-seed stays"
- Environment: RTX 4060 Ti 16 GB, `.venv` python 3.12, torch 2.13 nightly
  cu132 (AMP mandatory). Panels/checkpoints are gitignored; docs, queue
  states and metrics JSONs are committed.

## Production default (ADOPTED)

**blend50 + band10, 7-seed** — 50/50 z-blend of:
- Champion transformer: `LSTM_CondTransformer` preset B (h64, 2 trans
  layers, seq60), close-only features, `tgt_rank_20`, **7-seed ensemble
  (seeds 0–6)**, monthly (126d) walk-forward retrain, MSE loss,
  equal-weight full history;
- D1.2 momentum: mom126 skip-5, cross-sectional z.

Construction: top-quintile, equal weight, **10% no-trade band, hard 10%
name cap**, soft 20% sector cap, 20d hold, next-session execution
(exec_lag=1, same-bar audit clean).

## Recommended deployment variant (documented option)

**D7b: band15 + cap7.5** on the same 7-seed signal — Sharpe-identical to
default (2.15 / 1.44), bear DD −15.2% (+2.8pp better), 2022 −0.08, turnover
0.29/0.23, maxW 7.5%. Shadow books remain band10 until a quarter boundary.

## Champion metrics (standing references)

- Seed-robust ranges (v7): **L/S net60 ≈ 1.85–2.15 (2023→26), ≈ 1.30–1.45
  (2021→26)**; bootstrap medians **1.92 / 1.37** are the planning numbers;
  single-seed-set points 2.147 / 1.443 (seeds 0–6) sit at the range tops.
- Crash-first window (2022→): 1.32 (band10) / 1.47 (D7b).
- Execution validation (v9): break-even **632 bps champ / 463 bps bear**;
  net150 1.81/1.13; capacity trivial ≤10M TWD; T+2 delay cost 0.226;
  settled-cash constraint immaterial; 30% sector cap ≈ free.
- Bootstrap (R3, D7b): CH p5 1.70/p50 1.94; BR p5 1.11/p50 1.33; 100%
  positive draws all windows.

## What is adopted

7-seed ensemble (A8); blend50+band10 book; D7b as recommended deployment
variant; daily ops pipeline; P1 matured paper ledger + P2 daily diff
(operational tooling, v9).

## What is rejected (v10 arc, all at pre-registered gates)

Cross-sectional attention; 5d reversal; input augmentation; MT multi-task
heads (seed-luck, killed by disjoint-seed battery); residual target; TCN;
B2 quantile as blend challenger (bear 1.274 < 1.30 floor); **D2 distillation
(0/6 gates — final act of the arc)**. Earlier: ~20 challengers v1–v7 (see
RESEARCH_LINES_CLOSED.md and the scoreboard).

## What remains monitor-only / parked

- **Quantile-standalone lead (parked):** pinball training improves the
  standalone transformer (+0.36 champ) but not the blend. Any revisit needs
  fresh pre-registration and is adaptive-weight-adjacent.
- **No crash overlay exists by decision:** EX3 disarmed, hedge sleeve and
  regime downweight rejected (v9). Bear risk is managed by construction
  (D7b band15) only — documented, accepted state.
- **P1 evidence gate:** arms at 20 matured 20d observations (~3 rebalances
  out as of 07-26).

## Do not reopen without new evidence

Everything in RESEARCH_LINES_CLOSED.md. The two standing meta-rules that
guard the loop:
1. **Never adopt on val-IC evidence alone** — four dissociations recorded
   (D1, MT, TCN, D2): val IC does not see book-space/tail behavior.
2. **Single 7-seed draws are not decision-grade** — seed-set replication
   (disjoint seeds 10–16) is mandatory before any adoption talk (v7 + MT
   precedents).
