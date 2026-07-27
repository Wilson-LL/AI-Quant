# Open Watchlist — What the Loop Is Actually Watching

No experiments are running or pending. These are the live monitoring items,
in priority order, with their triggers.

## 1. P1 paper ledger — evidence gate (~3 rebalances out)

At **20 matured 20d observations** per strategy (17/20 as of 07-26) the
realized-vs-backtest comparison becomes decision-grade: realized annualized
Sharpe per book vs the bootstrap CIs (champ p5 1.61 / p50 1.92; bear p5
1.14 / p50 1.37). Check `LEDGER_STATUS.md` daily.
**Trigger:** realized blend Sharpe below the p5 of its window → open a
divergence investigation (first live-forward falsification test the
strategy has ever faced).

## 2. Realized-vs-backtest divergence (continuous)

Before the gate arms, watch the per-snapshot `ret_20d` rows in
`PAPER_LEDGER.csv` for systematic sign/magnitude drift vs backtest
expectations (~+1.5–2.5%/20d in normal regimes). Persistent 1d/5d underperformance with normal 20d = execution-timing issue, not alpha decay.

## 3. Top-contributor concentration — name 1519 (R5 flag)

The one open flag from v9: bear-window drop-top-5 retention 0.667 (<0.70);
1519 alone = 9–12% of positive PnL. **Trigger:** 1519 (or any single name)
exceeding ~15% of realized ledger PnL, or appearing at max weight for many
consecutive books → consider the 7.5% cap variant (D7b) which is
Sharpe-free insurance, already validated.

## 4. Daily diff anomalies (P2)

`DIFF_<asof>.md` flags: one-way turnover >0.50 (band should keep it ~0.15–
0.35) or top sector >50%. Either → do not trade the book mechanically;
check for cache corruption / duplicate dates / score explosion first.

## 5. Sector concentration

v9-R4 priced the concentration premium at ~zero (30% hard cap costs 0.001
Sharpe) but soft-cap books can still reach p95 ~57% single-sector.
**Trigger:** sector share >50% on consecutive books → the free 30% hard cap
is the pre-validated response (needs only a pre-registered construction
switch, not research).

## 6. 2022-like regime behavior

No crash overlay exists (all Track-4 levers failed their bars — accepted
state). The defensive posture is construction-only: D7b band15 (bear DD
−15.2% vs −18.0%). **Trigger for re-opening bear research:** paper ledger
shows a drawdown exceeding ~−12% (2/3 of the backtest bear DD) or the
market proxy enters a sustained below-126d-MA regime with the blend
underperforming its 2022 backtest analogue (2022 was −0.15 blend / −0.08
D7b — approximately flat is the expectation, not profit).

## 7. Quantile-standalone lead (parked, not scheduled)

Pinball-loss training improves the standalone transformer (+0.36 champ /
+0.05 bear vs champion standalone; val IC 0.063) but not the 50/50 blend.
Parked because acting on it is adaptive-weight-adjacent (closed line).
**Trigger to revisit:** the paper ledger's realized data ever shows the
momentum leg persistently dragging the blend (which would justify a
pre-registered blend-weight/quantile revisit), or full-field data arrives
(~2027-01) and a new feature generation opens anyway. `quantile` flag is in
the training module, default OFF, ready.

## Calendar items

- **~2027-01:** `data_cache_full` reaches ~6 months of true
  turnover/transaction history → E2 full-field revisit becomes eligible
  (the one pre-authorized future research line).
- **Quarter boundary:** decide whether shadow books switch band10 → D7b.
