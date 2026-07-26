# Queue v9 Summary — Deployment Validation (CPU-only, signal frozen)

Completed 2026-07-26. 13/13 done, 0 failed, ~65 min CPU (R3 bootstrap 57 min).
Signal frozen to the 7-seed panels; zero GPU training; cache read-only;
production defaults unchanged throughout. Full per-experiment JSON in this
directory; queue state in `queue_v9.json`.

## Verdict table

| ID | Verdict | Headline numbers |
|---|---|---|
| C1 | **D7b stays recommended** | band15_cap10 posts best raw Sharpe (2.21 CH / 1.45 BR / 1.52 W22) but bear DD only 0.1pp better than D7b — misses the pre-registered 1pp dominance bar. Band15 (not cap) is the bear-DD driver. |
| X1 | **PASS** | net150: 1.81 CH / 1.13 BR; net200: 1.62 / 0.95; break-even 632 bps CH, 463 bps BR — wide cost margin of safety. |
| X2 | **PASS** | 10M TWD: p95 participation 0.4% of ADV20, ~0.3% of traded weight capped, est impact ≈0. 100M TWD impact −0.004 — capacity is not a binding constraint at plausible scale. (ADV proxy = close×volume; OHLCV-only cache.) |
| X3 | **PASS — delay-robust; same-bar audit clean** | Panel fwd_h == cache lag-1 returns exactly (no same-bar execution anywhere). T+2 cost 0.226 CH / ~0 BR (≤0.30 bar); T+3 non-monotonic (0.11 CH) — timing noise, not cliff decay. |
| X4 | **PASS — idealized rebalance safe** | Worst-case settled-cash (all buys T+3): cost ≤ +0.004 across all books/windows (mostly slightly negative = in-sample noise, not edge). Paper book needs no constrained-fill logic. |
| P1 | **OPERATIONAL** | Ledger populated (88 rows, 4 books × 1/5/10/20d). Evidence gate ARMED at 17/20 matured 20d obs — activates in ~3 rebalances. `reports/paper_trading/LEDGER_STATUS.md`. |
| P2 | **OPERATIONAL** | Diff report clean on 3 most recent snapshots → `reports/continuous_research/daily_diffs/`. Add to daily ops cycle. |
| B2 | **REJECT — EX3 DISARMED** | Primary cell (−10%→50%, re-enter −5%): blend bear DD −18.0→−14.0% (+4.0pp ✓) but d7b only −15.2→−13.0% (+2.2pp < 3pp bar). Exposure-gate thread now closed with numbers on both index-level (EX1/EX2) and own-equity (EX3) triggers. |
| B3 | **REJECT** | Triggered tf short sleeve: LO bear DD improves only ~1.6pp (bar: 3pp). Descaling (B2) dominates hedging, and neither clears its bar. LO book keeps no crash overlay. |
| B4 | **REJECT (as pre-declared)** | Bear side did improve (blend 2022: −0.15→+0.04; d7b BR Sharpe 1.44→1.54) but champion window did not improve ≥0.1 — fails the deliberately high bar. Adaptive-weight line stays closed; the bear-side observation is one crash sample, recorded not acted on. |
| R3 | **PASS** | D7b bootstrap (200×): CH p5 1.70/p50 1.94; BR p5 1.11/p50 1.33; W22 p5 1.02; 100% positive everywhere — matches R2's profile; deployment spec is name-composition-robust. |
| R4 | **PASS — premium not load-bearing** | Hard 30% sector cap costs ≤0.001 Sharpe (!); even 20% cap and drop-largest-sector remain viable. Sector concentration is incidental, not the source of edge. |
| R5 | **CONCENTRATION FLAG (bear window)** | Top name (1519) = 9–12% of positive PnL (<15% ✓). Drop-top-5: CH retention 0.80–0.85 ✓, but BR/d7b 0.667 < 0.70 — bear-window Sharpe leans on top names. Feeds the cap discussion; D7b's 7.5% cap does not fix name reliance. |

## Deployment picture after v9

The 7-seed blend deployment case is now **execution-validated**: costs (break-even
>460 bps), capacity (≥10M TWD trivially), delay (T+2 tolerable), settlement
(no constrained-fill logic needed), sector concentration (cap essentially
free if wanted) all pass. The two honest caveats are (1) bear-window
dependence on top contributors (R5 flag — the one number to watch in the
paper ledger) and (2) no crash overlay exists: all three Track-4 levers
failed their bars, so bear risk is managed by construction (D7b band15) and
position discipline only.

Operational tooling now live: P1 matured ledger (evidence gate arms at 20
obs) and P2 daily diff — both should join the daily-ops cycle.
