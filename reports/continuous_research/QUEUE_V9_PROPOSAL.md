# Queue v9 Proposal — Deployment Realism & Paper-Trading Readiness

Status: **PROPOSED — NOT STARTED** (2026-07-26). Awaiting user approval.
Production defaults unchanged: 7-seed blend50+band10 (D7b band15+cap7.5 =
recommended deployment variant). 14-seed line closed (V8, user decision).

## Design principles

1. **Signal is frozen.** Every experiment reuses existing 7-seed panels:
   `SCHED_A8_seeds7_full.csv.gz` (2023→), `SCHED_BEAR_A8_seeds7_full.csv.gz`
   (2021→), `SCHED_W22_oos2022_start.csv.gz` (2022→). Zero new GPU trainings →
   zero seed expansion, zero `_run_key` collisions (scheduler dedup stays
   active but has nothing to catch), and structurally bounded overfitting:
   we vary only construction / execution / measurement layers on a fixed score.
2. **Dual-window gates everywhere** (2023–26 and 2021–26), with the 2022
   crash-first window (W22) as a third read where DD behavior matters.
   Standing refs: seed-robust ranges L/S 1.85–2.15 / 1.30–1.45; bootstrap
   medians 1.92 / 1.37 as planning numbers; D7b 2.15 / 1.44, bear DD −15.2%.
3. **Closed lines respected.** Three Track-4 items are adjacent to closed
   lines (EX1/EX2 exposure gates, adaptive weights, regime features). They are
   included only as pre-registered characterization runs with explicit
   expected outcomes and high adoption bars — the new information justifying a
   look is the W22 crash-first window and the 2022 seed-dependence discovered
   in v7, neither of which existed when those lines closed.
4. **CPU-only.** Total compute ≈ 2–3 h CPU + ~1 h tooling build (P1/P2).
   Daily inference in Track 3 touches the GPU for seconds as usual.

## Summary table

| ID | Track | One-liner | Compute | Runtime |
|---|---|---|---|---|
| C1 | 1 | band{10,15}×cap{7.5,10} construction grid, LS+LO, 3 windows | CPU | ~15 min |
| X1 | 2 | cost curve 60/100/150/200 bps + break-even bps | CPU | ~5 min |
| X2 | 2 | ADV-proxy liquidity caps × capital scale | CPU | ~15 min |
| X3 | 2 | execution delay T+1 (verify) / T+2 / T+3 | CPU | ~10 min |
| X4 | 2 | LO sell-first / settled-cash constraint sim | CPU | ~30 min (build+run) |
| P1 | 3 | matured forward-return ledger, 1/5/10/20d, 3 books | CPU | ~40 min build |
| P2 | 3 | daily diff report in the daily-ops cycle | CPU | ~30 min build |
| B2 | 4 | EX3 own-equity DD exposure gate (armed monitor, first eval) | CPU | ~10 min |
| B3 | 4 | transformer short-sleeve hedge on LO deployment book | CPU | ~20 min |
| B4 | 4 | regime-conditional D1.2 downweight (expected REJECT) | CPU | ~10 min |
| R3 | 5 | universe bootstrap (200×, drop-20%) on D7b spec, 3 windows | CPU | ~20 min |
| R4 | 5 | hard sector-cap stress 30%/20% + drop-largest-sector | CPU | ~10 min |
| R5 | 5 | drop-top-N contributor stress, N∈{1,3,5} | CPU | ~10 min |

13 experiments. Suggested order: C1 → (X1, X3, R3, R4, R5 on the C1-confirmed
spec) → X2, X4 → B2, B3, B4 → P1, P2 (infrastructure last, so the ledger
starts on the finalized spec).

---

## Track 1 — Deployment construction robustness

### C1 — construction grid
1. **ID:** V9-C1
2. **Hypothesis:** D7b (band15+cap7.5) is non-dominated across the full 2×2
   grid; the two never-tested cells (band10+cap7.5, band15+cap10) do not beat
   it on the (Sharpe, bear DD, turnover) triple.
3. **Config:** 7-seed blend50 scores; band ∈ {10%, 15%} × cap ∈ {7.5%, 10%} ×
   mode ∈ {LS, LO} × window ∈ {2023→, 2021→, 2022→}; net0/60/100/150; yearly
   lines; turnover & max-DD table.
4. **Runtime:** ~15 min CPU (24 re-books from 3 panels).
5. **Gate (pre-registered):** D7b stays recommended unless another cell has
   Sharpe within 0.05 tol on BOTH gate windows AND bear DD ≥1pp better AND
   turnover no worse. Band10+cap10 stays shadow-book spec regardless (quarter
   boundary rule).
6. **Baseline:** band10+cap10 (2.147/1.443) and D7b (2.15/1.44/−15.2%).
7. **Compute:** CPU-only.
8. **Outputs:** `reports/continuous_research/v9/C1_construction_grid.json`, `.md` summary table.
9. **Overfit risk:** Low-moderate (4 cells; mitigated by pre-registered
   dominance rule and dual-window gate — we are completing a matrix, not
   searching it).
10. **Why worth running:** the deployment spec freeze (needed before paper
    trading means anything) currently rests on 2 of 4 grid cells.

## Track 2 — Execution realism

### X1 — cost sensitivity & break-even
1. **ID:** V9-X1
2. **Hypothesis:** chosen deployment spec keeps Sharpe ≥1.0 at 150 bps on the
   champion window and ≥0.8 on the bear window; break-even cost >250 bps.
3. **Config:** net at 60/100/150/200 bps round-trip (200 = new column) on the
   C1-chosen spec + shadow spec, both windows; per-year net150/net200 lines;
   linear break-even bps per book from turnover.
4. **Runtime:** ~5 min CPU.
5. **Gate:** descriptive; deployment WARNING flag if net150 <1.0 (champ) or
   <0.7 (bear) — that would mean the edge depends on optimistic cost fills.
6. **Baseline:** existing net0–150 columns (e.g. 7-seed blend champ net150 1.78).
7. **Compute:** CPU-only.
8. **Outputs:** `v9/X1_cost_curve.json`, break-even table in `.md`.
9. **Overfit risk:** none (pure re-pricing of fixed books).
10. **Why:** actual TWSE all-in retail costs are uncertain (tax 30 bps sell
    side alone); we need the margin of safety quantified before live paper P&L
    is interpreted.

### X2 — liquidity-aware position limits
1. **ID:** V9-X2
2. **Hypothesis:** at retail/small-fund scale (≤10M TWD) ADV-based position
   caps never bind; visible degradation starts only ≥100M TWD.
3. **Config:** ADV proxy = 20d median(close×volume) (cache is OHLCV-only — no
   turnover field; proxy documented as limitation). Cap per-name daily trade
   at f×ADV, f ∈ {0.5%, 1%, 2%}; capital ∈ {1M, 10M, 100M TWD}; unfillable
   residue carried or dropped (both variants); chosen spec, both windows.
4. **Runtime:** ~15 min CPU.
5. **Gate:** descriptive capacity curve; flag if 10M TWD at f=1% costs >0.1
   Sharpe.
6. **Baseline:** uncapped books (C1).
7. **Compute:** CPU-only.
8. **Outputs:** `v9/X2_capacity.json`, capacity curve table.
9. **Overfit risk:** none (constraint only removes freedom).
10. **Why:** the 108-stock universe includes thin names; paper trading at an
    assumed capital needs to know whether backtest weights are even fillable.

### X3 — rebalance delay / no same-bar execution
1. **ID:** V9-X3
2. **Hypothesis:** dataset `exec_lag=1` already guarantees no same-bar
   execution (verify and document); alpha decays gracefully: T+2 costs ≤0.15
   Sharpe, T+3 ≤0.30.
3. **Config:** re-book chosen spec with execution shifted +1 and +2 days
   beyond current T+1, both windows; verification cell asserting no signal
   date == execution date in the panel pipeline.
4. **Runtime:** ~10 min CPU.
5. **Gate:** if T+2 loss >0.3 → deployment REQUIRES guaranteed next-open
   execution; note in deployment spec. Otherwise PASS with decay curve.
6. **Baseline:** production T+1 books.
7. **Compute:** CPU-only.
8. **Outputs:** `v9/X3_delay.json`, decay table + same-bar audit note.
9. **Overfit risk:** none.
10. **Why:** a monthly-retrain 20d-hold signal should be delay-robust; if it
    is not, that is a red flag about where the backtest edge lives.

### X4 — long-only cash-settlement constraint
1. **ID:** V9-X4
2. **Hypothesis:** with turnover 0.23–0.29 and 20d holds, a sell-first /
   buy-with-settled-cash (T+2 TW settlement) LO simulation costs <0.05 Sharpe
   vs the idealized simultaneous rebalance.
3. **Config:** LO book of chosen spec; sells execute at T+1 open, buys
   constrained by cash settled by that date, unfilled buys carried ≤3 days;
   no leverage, no margin; both windows.
4. **Runtime:** ~30 min CPU (small simulator build + run).
5. **Gate:** if cost ≥0.05 Sharpe or fills carried >3 days on >5% of
   rebalances → paper book generator must adopt the constrained fill logic.
6. **Baseline:** idealized LO book (C1).
7. **Compute:** CPU-only.
8. **Outputs:** `v9/X4_cash_constraint.json` + fill-shortfall log.
9. **Overfit risk:** none.
10. **Why:** this is the last idealization separating the LO backtest from an
    actually executable retail book.

## Track 3 — Paper-trading readiness

### P1 — matured forward-return ledger
1. **ID:** V9-P1
2. **Hypothesis (operational):** realized 20d book returns track backtest
   within the bootstrap CI once ≥20 matured observations exist.
3. **Config:** ledger keyed by (snapshot_date, book ∈ {tf-standalone, blend,
   D1.2}); records positions at snapshot, then fills realized 1/5/10/20d
   forward returns as dates mature from the daily cache; backfills from
   existing `paper_trading.py` snapshots; daily-ops hook appends new rows.
4. **Runtime:** ~40 min build; seconds/day thereafter.
5. **Gate:** operational — ledger populates with no NaN for all matured dates;
   evidence gate ARMED (not judged) until ≥20 matured 20d obs, then
   realized-vs-backtest IC and Sharpe comparison auto-reported.
6. **Baseline:** backtest books + R2/R3 bootstrap CIs.
7. **Compute:** CPU (daily inference step already uses GPU for seconds, unchanged).
8. **Outputs:** `reports/continuous_research/paper_ledger/ledger.csv`,
   `LEDGER_STATUS.md` (auto-updated).
9. **Overfit risk:** none — this is measurement, and it is the *anti*-overfit
   instrument: live-forward evidence.
10. **Why:** everything upstream is in-sample by construction; this is the
    only mechanism that can eventually validate or kill the strategy on
    unseen data.

### P2 — daily diff report
1. **ID:** V9-P2
2. **Hypothesis (operational):** a per-day diff makes silent pipeline drift
   visible within one daily cycle.
3. **Config:** after each daily book generation emit: entries/exits vs prior
   day, weight deltas >1pp, band/cap bindings hit, sector shares, estimated
   cost of the day's trades at 60/150 bps, and tf-vs-blend-vs-D1.2 rank
   disagreement (top-quintile overlap). Appended to daily ops (runs only at
   queue completion / idle, per existing rule).
4. **Runtime:** ~30 min build; <1 min/day.
5. **Gate:** operational — report generated on 3 consecutive daily cycles
   without error; anomaly lines (e.g. turnover >2× trailing median) flagged.
6. **Baseline:** current daily ops output (books only, no diff).
7. **Compute:** CPU-only.
8. **Outputs:** `reports/continuous_research/daily_diffs/DIFF_<date>.md`.
9. **Overfit risk:** none.
10. **Why:** the loop runs unattended; regressions like the duplicate-date
    cache quirk were caught by luck, not instrumentation.

## Track 4 — Bear / crash robustness

*(Static 7-seed blend and D7b across all three windows are covered by C1's
window sweep — not duplicated here. The three items below are the dynamic
overlays. All three are closed-line-adjacent and pre-registered accordingly.)*

### B2 — EX3 own-equity drawdown exposure gate
1. **ID:** V9-B2
2. **Hypothesis:** scaling gross to 50% while the book's own trailing DD
   exceeds −10% cuts bear-window max-DD by ≥3pp at ≤0.05 champion-window
   Sharpe cost. (EX3 was armed as a monitor at line-closure of EX1/EX2
   index-level gates but never evaluated; W22 window is new information.)
3. **Config:** primary cell pre-registered: trigger −10%, scale 0.5, re-entry
   on DD recovery to −5%. Sensitivity cells (secondary, not adoption-eligible):
   trigger −8%/−12%, scale 0.7. Chosen spec, windows 2021→ and 2022→ plus
   champion-window cost check.
4. **Runtime:** ~10 min CPU.
5. **Gate:** adopt as armed monitor policy ONLY if primary cell meets both
   numbers; otherwise REJECT and formally disarm EX3 (closing the last
   exposure-gate thread).
6. **Baseline:** unscaled chosen-spec books; D7b bear DD −15.2%.
7. **Compute:** CPU-only.
8. **Outputs:** `v9/B2_ex3_gate.json`.
9. **Overfit risk:** moderate — 1 crash episode, 5 cells; mitigated by single
   pre-registered primary cell and adopt-as-monitor-only (not always-on).
10. **Why:** it's the standing armed monitor; either it earns its arming with
    numbers or it gets disarmed — both outcomes clean up loop state.

### B3 — transformer short-sleeve hedge on LO book
1. **ID:** V9-B3
2. **Hypothesis:** a 15%-gross short sleeve from the transformer's bottom
   quintile, active only under the B2 trigger, improves LO bear DD by ≥3pp at
   ≤0.1 LO Sharpe cost — giving the realistic (long-only-constrained TW
   retail) deployment a crash lever it currently lacks.
3. **Config:** LO chosen spec + short sleeve 15% gross, bottom-quintile tf
   scores, same band/cap discipline; variants: always-on vs triggered (B2
   primary trigger); windows 2021→, 2022→, champion cost check.
4. **Runtime:** ~20 min CPU.
5. **Gate:** triggered variant must beat plain LO on (bear DD ≥3pp better,
   champ Sharpe cost ≤0.1) AND beat the B2 pure-descaling result — else
   REJECT (descaling dominates hedging).
6. **Baseline:** LO chosen spec; LO bear DD ≈ −28 to −34% is the pain point.
7. **Compute:** CPU-only.
8. **Outputs:** `v9/B3_hedge_sleeve.json`.
9. **Overfit risk:** moderate (2 variants × trigger reuse; one crash episode).
10. **Why:** LS book hedges itself; the LO book — the likely real deployment —
    has no crash protection at all, and its DD numbers show it.

### B4 — regime-conditional D1.2 downweight (expected REJECT)
1. **ID:** V9-B4
2. **Hypothesis (pre-declared expected REJECT, B6s precedent):** shifting
   blend weights 50/50 → 70/30 (tf/mom) while TAIEX < 126d MA improves 2022
   without hurting elsewhere. Adjacent to TWO closed lines (adaptive weights;
   regime features) — run purely as cheap characterization to close the
   question at book level with the new W22 window.
3. **Config:** single pre-registered rule (no grid): index-below-126d-MA →
   70/30, else 50/50; chosen spec; all 3 windows.
4. **Runtime:** ~10 min CPU.
5. **Gate:** adoption bar deliberately high — BOTH gate windows improve ≥0.1
   AND 2022 improves ≥0.2; anything less = REJECT and the adaptive-weight
   line stays closed with book-level evidence added.
6. **Baseline:** static 50/50 chosen spec.
7. **Compute:** CPU-only.
8. **Outputs:** `v9/B4_regime_weights.json`.
9. **Overfit risk:** high in spirit (regime rule, one crash) — which is why it
   is single-cell, pre-declared expected-REJECT, and adoption-gated at a
   level a lucky draw won't clear.
10. **Why:** cheap, closes a recurring "but what if we downweight momentum in
    crashes" question with numbers instead of assertion.

## Track 5 — Robustness audit

### R3 — universe bootstrap on deployment spec
1. **ID:** V9-R3
2. **Hypothesis:** D7b/C1-chosen spec matches the band10 spec's bootstrap
   profile: 100% positive draws, champ p5 ≥1.5, bear p5 ≥1.1, medians within
   0.1 of 1.92/1.37.
3. **Config:** existing `research/bootstrap_7seed.py` extended to the chosen
   construction; 200 draws, drop 20% of names; windows 2023→, 2021→, 2022→.
4. **Runtime:** ~20 min CPU.
5. **Gate:** PASS if hypothesis numbers met; else the spec choice reverts to
   whichever grid cell bootstraps better (name-composition robustness beats
   point estimates — R2 lesson).
6. **Baseline:** R2 (band10: champ p5 1.61/p50 1.92; bear p5 1.14/p50 1.37).
7. **Compute:** CPU-only.
8. **Outputs:** `v9/R3_bootstrap_deploy.json`, log.
9. **Overfit risk:** none (robustness measurement).
10. **Why:** the deployment spec was chosen on point estimates; R2 showed
    point estimates sit near p95 — the chosen spec must survive the same
    honesty check.

### R4 — sector concentration stress
1. **ID:** V9-R4
2. **Hypothesis:** a hard 30% sector cap on longs costs ≤0.15 Sharpe; the
   drop-largest-sector book stays >60% of baseline Sharpe (V1 found soft
   sector share avg 37%, p95 57% — C4 showed sector-*neutrality* is too
   expensive; a loose hard cap is the untested middle ground).
3. **Config:** hard sector caps {30%, 20%} + drop-largest-sector rerun;
   chosen spec, both gate windows.
4. **Runtime:** ~10 min CPU.
5. **Gate:** descriptive; deployment note records the cost curve. If 30% cap
   costs >0.3 Sharpe → concentration premium is load-bearing → paper ledger
   must track realized sector shares (P2 hook).
6. **Baseline:** uncapped chosen spec; C4 (full neutrality: −0.4 to −1.1).
7. **Compute:** CPU-only.
8. **Outputs:** `v9/R4_sector_stress.json`.
9. **Overfit risk:** none.
10. **Why:** 57% p95 single-sector share is a real deployment risk; we should
    know its price before it shows up in live drawdown.

### R5 — drop-top-N contributor stress
1. **ID:** V9-R5
2. **Hypothesis:** no single name contributes >15% of total PnL; dropping the
   top-5 contributors (chosen ex-post = worst case) retains >70% of Sharpe on
   both gate windows.
3. **Config:** per-name realized contribution table; re-book with top-{1,3,5}
   contributors excluded; chosen spec, both windows.
4. **Runtime:** ~10 min CPU.
5. **Gate:** flag CONCENTRATION-RISK if either number violated; feeds max-cap
   discussion (7.5% vs 10%) with evidence.
6. **Baseline:** R1/R2 drop-top-3 results (1.77 / 1.16).
7. **Compute:** CPU-only.
8. **Outputs:** `v9/R5_droptop.json` + contribution table.
9. **Overfit risk:** none (adversarial stress, ex-post worst case).
10. **Why:** distinguishes "broad cross-sectional edge" from "three lucky
    names" — the single most common way backtests of small universes lie.

---

## Explicit non-goals / guardrails

- No new GPU trainings, no seed-count changes (7 stays; 14 closed).
- No reopening of closed lines beyond the three flagged, pre-registered
  Track-4 characterization runs.
- 2021–26 and 2023–26 dual-window gates apply to every adoption decision;
  2022-first window used as DD evidence.
- Scheduler `_run_key` dedup remains active; v9 adds no configs to the GPU
  queue, so no collision is possible by construction.
- Nothing here modifies production defaults; adoption of any v9 outcome is a
  separate, explicit step.
