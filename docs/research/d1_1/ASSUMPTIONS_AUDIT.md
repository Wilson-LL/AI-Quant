# AI-Quant — Assumptions Audit & Research Roadmap

Lead researcher's audit of the **implicit assumptions** baked into the current
system. Scope is deliberately *everything except neural-network architecture*
(per instruction): target, holding period, portfolio construction, ranking,
entry, exit, evaluation, and the data/universe those rest on.

Some cards cite **real evidence** already measured on 7 cached TWSE stocks
(2018→2026, Y=20) via the `research/` framework; others are flagged as
*reasoning, to be tested* by a named experiment.

## Scoring method

Each assumption gets three 1–5 scores and a derived priority:

- **U — Upside** if the assumption is wrong and we fix it (1 marginal → 5 could transform OOS performance).
- **L — Likelihood** the current choice is actually suboptimal (1 probably fine → 5 almost certainly wrong).
- **C — Cost** to change *and* fairly test (1 hours → 5 weeks / needs infra we lack, e.g. torch).
- **RPS = U × L / C** (higher = do sooner). Range ≈ 1–15.

The formula rewards cheap, high-confidence, high-upside changes and penalises
expensive speculative ones — exactly the ordering a research program should follow.

---

## Ranked summary (top 20)

| # | Assumption | Area | U | L | C | **RPS** |
|--:|------------|------|--:|--:|--:|--:|
| 1 | Negative class merges "no barrier touched" with real losers | Target | 3 | 5 | 1 | **15.0** |
| 2 | Validation "PnL" is a scalar proxy, not a real strategy backtest | Evaluation | 5 | 5 | 2 | **12.5** |
| 3 | No transaction costs / slippage / TWSE daily price-limit modelled | Portfolio/Eval | 4 | 5 | 2 | **10.0** |
| 4 | Fixed **absolute** barriers (12%/6%) ignore per-stock volatility | Target/Exit | 4 | 4 | 2 | **8.0** |
| 5 | Binary triple-barrier is the right target vs a continuous cross-sectional return | Target | 4 | 4 | 2 | **8.0** |
| 6 | Prices assumed adjusted for splits/dividends (unverified) | Data/Target | 4 | 4 | 2 | **8.0** |
| 7 | Market gate = binary threshold on mean prob > 0.52 | Portfolio/Entry | 3 | 4 | 2 | **6.0** |
| 8 | Universe is ~90 tech/semis names → correlated "diversification" | Portfolio | 3 | 4 | 2 | **6.0** |
| 9 | Holding period undefined; exit logic not actually implemented | Exit/Holding | 4 | 4 | 3 | **5.3** |
| 10 | Hand-picked *current* universe → survivorship bias in backtest | Evaluation | 3 | 5 | 3 | **5.0** |
| 11 | Equal-weight top-10, no conviction or volatility position sizing | Portfolio | 3 | 3 | 2 | **4.5** |
| 12 | Probability rank == expected-return rank (calibration is OFF) | Ranking | 3 | 3 | 2 | **4.5** |
| 13 | 20-day prediction horizon is optimal | Holding | 3 | 3 | 2 | **4.5** |
| 14 | 2:1 reward/risk (12/6) is on the efficient frontier | Exit | 3 | 3 | 2 | **4.5** |
| 15 | Fixed ensemble weights 0.25/0.55/0.15/0.05 across horizons | Ranking | 2 | 4 | 2 | **4.0** |
| 16 | All four horizon models predict the **same** 20d/12/6 label | Target/Ranking | 2 | 3 | 2 | **3.0** |
| 17 | 40-day lookback window is sufficient/optimal | Model input | 2 | 3 | 2 | **3.0** |
| 18 | top_k=30 (rank) and top-10 (trade) cutoffs are arbitrary | Portfolio/Ranking | 2 | 3 | 2 | **3.0** |
| 19 | Entry for all names simultaneously at signal close, infinite liquidity | Entry | 2 | 4 | 3 | **2.7** |
| 20 | Long-only; no short side | Portfolio | 2 | 2 | 4 | **1.0** |

---

## Detailed cards (rank order)

### 1. Negative class merges "no barrier touched" with real losers — RPS 15.0 · Target
- **Description:** In `dataset.build_samples`, if neither +12% nor −6% is hit within 20d the label defaults to `0` — the same class as a genuine −6% loss.
- **Why it exists:** Simplicity — a single binary "did the winning move happen" flag.
- **Evidence for:** For a pure "will it pop +12%" detector, non-pops are legitimately the negative class.
- **Evidence against (measured):** On real data, **45.5% of the `0` class had a *positive* 20d forward return**; the class's median forward return is −0.56% (essentially flat). The label conflates "went nowhere" with "lost money," injecting large label noise. `corr(label, fwd_ret)=0.636` vs `0.719` for a plain return sign.
- **Upside if changed:** Cleaner labels → higher achievable IC for *any* model; likely the single cheapest accuracy win.
- **Cost:** Trivial — a relabel / three-class or return-based target already in `targets.py`.
- **Priority:** Do first.

### 2. Validation "PnL" is a scalar proxy, not a real strategy backtest — RPS 12.5 · Evaluation
- **Description:** `train.backtest` selects the model by `Σ prob·tp / −prob·sl` over the top-30 samples of a **shuffled, pooled** validation set. Model selection + early stopping ride on this one number.
- **Why it exists:** A quick scalar to rank epochs without building a portfolio simulator.
- **Evidence for:** Correlates loosely with "picks tend to be winners."
- **Evidence against:** It is not the deployed strategy (no market gate, no holding, no rebalancing, no costs, no cross-sectional structure); it is computed on pooled overlapping windows (leakage, now partly fixed); probability-weighting the payoff double-counts confidence. A model can win this metric and lose money live.
- **Upside if changed:** Foundational — *every* other decision inherits the referee's validity. Enables honest keep/revert calls.
- **Cost:** Low — the `research/evaluation.py` walk-forward + net top-k backtest already replaces it; needs wiring into training selection.
- **Priority:** Do first (it gates all others).

### 3. No transaction costs / slippage / price-limit modelled — RPS 10.0 · Portfolio/Eval
- **Description:** Nothing anywhere charges commission, spread, market impact, or accounts for TWSE's ±10% daily price limit (a +12% signal may be unfillable on a limit-up day).
- **Why it exists:** Frictionless backtests are easier and look better.
- **Evidence for:** For very low-turnover, large-cap trading, costs are small.
- **Evidence against:** 20-day holding with top-10 rebalencing is meaningful turnover; TWSE round-trip cost (fee+tax ≈ 0.4–0.6%) plus slippage can erase a thin edge. Signals that *require* a limit-up move are systematically unfillable.
- **Upside if changed:** Prevents adopting strategies that are only profitable on paper — protects the whole program from false positives.
- **Cost:** Low — cost already parametrised in `backtest_topk` (`cost_bps`); add a price-limit fill rule.
- **Priority:** High.

### 4. Fixed absolute barriers (12%/6%) ignore per-stock volatility — RPS 8.0 · Target/Exit
- **Description:** The same ±12%/−6% thresholds apply to a low-vol ETF (0050) and a high-vol small-cap.
- **Why it exists:** One global rule is simple and matches a fixed human risk appetite.
- **Evidence for:** Uniform thresholds give a consistent per-trade risk budget.
- **Evidence against:** A 12% move in 20d is routine for some names and near-impossible for others, so the label's base rate and meaning vary wildly by stock (base rate already 19% pooled but surely dispersed). Vol-scaled barriers (k·σ) equalise difficulty and information content.
- **Upside if changed:** More comparable labels across the universe; better cross-sectional ranking.
- **Cost:** Low — `targets.py` can add a σ-scaled barrier variant.
- **Priority:** High.

### 5. Binary triple-barrier vs continuous cross-sectional return — RPS 8.0 · Target
- **Description:** The system *ranks stocks cross-sectionally* (top-k) but trains on a *single-asset, path-dependent binary*. Ranking is usually best learned from a continuous (or cross-sectionally ranked) forward return.
- **Why it exists:** The binary matches the human "will it hit my TP" framing.
- **Evidence for:** Binary targets are robust to outliers.
- **Evidence against (measured):** `corr(TB, fwd_ret)=0.636 < corr(sign, fwd_ret)=0.719`; a binary throws away magnitude, which is exactly what a ranker needs to separate a +12% name from a +40% name.
- **Upside if changed:** Directly targets the quantity the portfolio monetises.
- **Cost:** Low — continuous / vol-adjusted targets already implemented; Experiment 1 (in flight) measures it.
- **Priority:** High.

### 6. Prices assumed adjusted for splits/dividends (unverified) — RPS 8.0 · Data/Target
- **Description:** twstock `close`/`capacity` are used raw. If not back-adjusted, ex-dividend and rights-issue gaps look like real −X% drops.
- **Why it exists:** Convenience of the data source.
- **Evidence for:** Large-caps' single-day dividend gaps are modest.
- **Evidence against:** TWSE names often pay 3–6% annual dividends in one ex-day gap — big enough to trip the −6% stop and mislabel a healthy stock as a loser, and to corrupt every return. This is a silent, systematic label/return bug.
- **Upside if changed:** Removes systematic mislabelling; correct returns everywhere.
- **Cost:** Low to *check* (compare a known ex-div day); moderate to source adjusted prices.
- **Priority:** High — verify immediately (cheap, potentially invalidates other results).

### 7. Market gate = binary threshold on mean prob > 0.52 — RPS 6.0 · Portfolio/Entry
- **Description:** If the universe's mean probability < 0.52, *all* trading is suppressed; else full exposure.
- **Why it exists:** Crude regime filter to sit out bad markets.
- **Evidence for:** Some market-timing overlays do reduce drawdown.
- **Evidence against:** 0.52 is an un-calibrated magic number tied to the (mis-specified) probability scale; a binary all-or-nothing switch is high-variance vs a continuous exposure scaler; never validated as adding *net* value vs just reducing exposure.
- **Upside if changed:** Smoother exposure, better risk-adjusted return, or removal if it's not paying.
- **Cost:** Low — test as an overlay in the backtest.
- **Priority:** Medium-high.

### 8. Universe is ~90 tech/semis names → correlated book — RPS 6.0 · Portfolio
- **Description:** The universe (`generate_stocks_json.py`) is overwhelmingly semis/electronics. Top-10 "diversified" picks are highly co-moving.
- **Why it exists:** Data-extraction scope (OTC excluded) and author familiarity.
- **Evidence for:** Concentrated exposure to a strong sector can outperform in trend.
- **Evidence against:** Effective breadth ≪ 10; a sector shock hits all positions together; risk is understated by naive equal-weighting.
- **Upside if changed:** Real diversification / correlation-aware sizing lowers drawdown per unit return.
- **Cost:** Low-moderate — broaden universe, add sector/correlation constraints.
- **Priority:** Medium-high.

### 9. Holding period undefined; exit not implemented — RPS 5.3 · Exit/Holding
- **Description:** `inference.py` outputs a probability and a top-10 list but specifies **no exit** — the 12/6/20d barriers exist only in the *label*, not in any live rule. Effective holding period is unstated.
- **Why it exists:** The project stops at signal generation.
- **Evidence for:** A discretionary trader could apply exits manually.
- **Evidence against:** Backtest realism and live PnL are undefined without an exit policy; label horizon (20d) and any real hold may diverge.
- **Upside if changed:** Makes PnL well-defined and optimisable; aligns train horizon with hold.
- **Cost:** Moderate — implement explicit exit (barrier and/or time) in the backtester.
- **Priority:** Medium.

### 10. Hand-picked current universe → survivorship bias — RPS 5.0 · Evaluation
- **Description:** The stock list is today's liquid names; delisted/shrunken names are absent from the historical backtest.
- **Why it exists:** Easiest way to get a list.
- **Evidence for:** Keeps data clean and liquid.
- **Evidence against:** Systematically inflates historical returns (losers that died are excluded); a well-known backtest killer.
- **Upside if changed:** Honest OOS estimates; avoids over-optimism.
- **Cost:** Moderate — need point-in-time constituents (harder with twstock).
- **Priority:** Medium; at minimum, disclose and bound the bias.

### 11. Equal-weight top-10, no conviction/vol sizing — RPS 4.5 · Portfolio
- **Description:** `trades = topk[:10]`, equal weight. No scaling by score strength or inverse volatility.
- **Why it exists:** Simplicity.
- **Evidence against:** Discards conviction information; equal dollar ≠ equal risk across different-vol names.
- **Upside:** Better risk-adjusted return from score/vol weighting.
- **Cost:** Low — weighting variants in `backtest_topk`.
- **Priority:** Medium.

### 12. Probability rank == expected-return rank; calibration OFF — RPS 4.5 · Ranking
- **Description:** Ranking uses raw sigmoid probs; alpha calibration is disabled (`calibration_candidates=[1.0]`, commit "set calibration to off").
- **Why it exists:** Calibration search was noisy on the tiny val set, so it was switched off.
- **Evidence against:** Uncalibrated probabilities need not be monotonic in expected return; the top-k cut is sensitive to miscalibration.
- **Upside:** Better selection at the top of the book.
- **Cost:** Low — evaluate rank-IC of calibrated vs raw scores.
- **Priority:** Medium.

### 13. 20-day prediction horizon is optimal — RPS 4.5 · Holding
- **Description:** Horizon fixed at 20 trading days.
- **Why it exists:** Reasonable swing-trade horizon.
- **Evidence against:** Never swept; optimal horizon is an empirical question and interacts with cost/turnover.
- **Upside:** Could materially change IC and net Sharpe.
- **Cost:** Low — horizon sweep in the framework.
- **Priority:** Medium.

### 14. 2:1 reward/risk (12/6) is efficient — RPS 4.5 · Exit
- **Description:** +12% TP / −6% SL chosen a priori.
- **Why it exists:** Clean 2:1 story.
- **Evidence against:** Not derived from the return/MFE-MAE distribution; the efficient (tp, sl, hold) point is measurable.
- **Upside:** Better per-trade expectancy.
- **Cost:** Low — grid-search on `mfe_mae` distributions.
- **Priority:** Medium.

### 15. Fixed ensemble weights 0.25/0.55/0.15/0.05 — RPS 4.0 · Ranking
- **Description:** Horizon models blended with hard-coded weights.
- **Why it exists:** Hand-tuned intuition (favour 1-year).
- **Evidence against:** Not learned or validated; likely off the optimum and regime-dependent.
- **Upside:** Modest ranking improvement.
- **Cost:** Low — fit weights on OOS IC (constrained regression).
- **Priority:** Medium-low.

### 16. All four horizons predict the same 20d/12/6 label — RPS 3.0 · Target/Ranking
- **Description:** Only the *training window length* differs; the target is identical.
- **Why it exists:** Reuse of one pipeline.
- **Evidence against:** Different lookbacks may better predict different horizons; identical targets make the ensemble members redundant/correlated.
- **Upside:** More diverse, complementary ensemble members.
- **Cost:** Moderate.
- **Priority:** Low (revisit after target work).

### 17. 40-day lookback is sufficient/optimal — RPS 3.0 · Model input
- **Description:** Input window fixed at 40 days.
- **Evidence against:** Untested; feature-dependent.
- **Upside/Cost:** Modest / low. **Priority:** Low.

### 18. top_k=30 / top-10 cutoffs are arbitrary — RPS 3.0 · Portfolio/Ranking
- **Description:** Rank list of 30, trade 10; both hard-coded.
- **Evidence against:** Breadth vs concentration trade-off is measurable (IC decay by rank).
- **Upside/Cost:** Modest / low. **Priority:** Low.

### 19. Entry simultaneously at signal close, infinite liquidity — RPS 2.7 · Entry
- **Description:** Implicit fill of all picks at the decision price.
- **Evidence against:** Impact/slippage and price limits bite, especially on the strongest signals; no staggering.
- **Upside/Cost:** Moderate / moderate. **Priority:** Low (fold into cost model, #3).

### 20. Long-only; no shorts — RPS 1.0 · Portfolio
- **Description:** System only goes long the top names.
- **Why it exists:** Shorting TWSE is operationally hard (borrow, uptick rules).
- **Evidence against:** A short leg on the bottom-ranked names could hedge market beta.
- **Upside/Cost:** Modest / high (operational). **Priority:** Low.

---

## Research roadmap

Sequenced by RPS and dependency. Every phase ends with an **accept/revert gate**:
a change is kept only if it improves the pre-registered OOS metric (net top-k
Sharpe and/or rank-IC vs realised return) beyond noise, logged in `RESEARCH_LOG.md`.

**Phase 0 — Referee & data integrity (foundational; mostly built).**
- ✅ Leakage-free global chronological split (done, `dataset.py`).
- ✅ Walk-forward + net top-k backtest referee, unit-tested (`research/`, 6/6).
- ▶ **#6 verify price adjustment** (cheap, can invalidate everything downstream).
- ▶ **#2 wire the real backtest into model selection**; **#3 turn on costs + price-limit fills**; **#10 document/bound survivorship bias.**
- *Gate:* referee reproduces a sane baseline number with costs on.

**Phase 1 — Target & label (highest RPS, cheapest).**
- **#1 fix the class-merging** (return-based or 3-class label).
- **#5 continuous / cross-sectional** target vs baseline (Experiment 1, in flight).
- **#4 vol-scaled barriers.**
- *Gate:* pick the target with best OOS rank-IC vs realised return **and** best net Sharpe under the Phase-0 referee. Record keep/revert.

**Phase 2 — Holding & exit.**
- **#13 horizon sweep** (5/10/20/40/60) × **#9 explicit exit policy** (barrier vs time vs trailing) × **#14 (tp,sl) grid** on MFE/MAE distributions.
- *Gate:* net Sharpe on the frontier, costs included.

**Phase 3 — Portfolio construction & ranking.**
- **#11 conviction/vol weighting**, **#18 breadth (k) via IC-decay**, **#8 correlation/sector-aware sizing + broader universe**, **#7 market gate as a continuous overlay (or drop)**, **#12 calibration**, **#15 learned ensemble weights**.
- *Gate:* each overlay must add net risk-adjusted return as a standalone ablation.

**Phase 4 — Entry/execution realism.**
- **#19 staggered entry, liquidity filters, impact** folded into the cost model.

**Phase 5 — Model class (only now, and still not "a bigger NN").**
- Establish the best **linear / gradient-boosted** baseline on the winning target.
  A neural architecture is considered *only if* it beats that baseline OOS, net of
  costs, by more than its added complexity and compute risk. (Blocked here anyway:
  no torch in this environment.)

**Operating rule:** work strictly top-down in RPS order; never advance a phase on
an in-sample or leaky result; log every experiment (hypothesis stated first,
then result, then keep/revert) in `RESEARCH_LOG.md`.
