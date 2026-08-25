# Roadmap — A → D → C (Momentum Product, engineered)

Pre-registered plan for building a cost- and risk-managed cross-sectional
**momentum** product for TWSE, per `PATH_ANALYSIS.md`. Grounded in Experiments
1–4 (`RESEARCH_LOG.md`). **No deep learning** — Exp-4 proved a 1-line momentum
factor captures everything the LSTM-Transformer learned, so the baseline is the
simplest thing that works and each layer must *earn* its complexity.

Guardrails (apply to every phase):
- **No neural nets / DL.** Linear ranks, closed-form sizing, rule-based overlays only.
- **Simplest baseline first.** A layer is kept only if it beats the previous layer
  on the pre-registered metric, net of cost, out-of-sample.
- **Pre-register before running.** Hypothesis + success/stop criteria are fixed in
  this file before each experiment; results and keep/revert go to `RESEARCH_LOG.md`.
- **Honest accounting.** Every number is net-of-realistic-cost, survivorship-aware
  (upper bound), and — for signal-quality claims — market/vol/momentum/size-neutral.

## Shared evaluation protocol
- **Universe:** the 106-name, 15-sector cache (survivorship-biased upward — stated
  on every result).
- **Data split:** expanding walk-forward where any parameter is fit; embargo = the
  holding horizon. Pure rank rules need no fitting (report full-sample OOS-equivalent).
- **Cost model:** TWSE round-trip ≈ **60 bps** anchor (fee ~0.1425%/side +
  transaction tax 0.30% on sells + slippage), always reported as a sweep
  {0,10,30,60,100} bps with the **breakeven** bps.
- **Two lenses, both reported:** (i) **beta/vol/sector-neutral long-short** =
  signal quality; (ii) **long-only top-k** (with an index/beta hedge noted) = the
  deployable product, since shorting TWSE is operationally hard.
- **Headline metrics:** net Sharpe, max drawdown, Calmar, worst-year Sharpe,
  annualized turnover, factor-regression alpha t-stat (must stay honest: the goal
  is *harvesting a known premium well*, not claiming new alpha).

## Baseline B0 — the simplest momentum product (reference for all phases)
Cross-sectional rank by a single canonical momentum signal — **126-day (6-month)
return skipping the most recent 5 days** (skip avoids short-term reversal) —
equal-weight top-k long / bottom-k short (and a long-only top-k variant),
rebalanced every 20 trading days. No ML, no optimization. This *is* the value
demonstrator; A/D/C must beat it.

---

## Phase A1 — Signal hygiene & turnover reduction
**Objective:** capture the momentum premium at materially lower turnover/cost than
the Exp-4 model (~100%/rebalance, breakeven ~124 bps) without losing predictive
power.

**Hypothesis:** momentum is slow-decaying (Exp-5: IC flat from 20→60 d), so
signal smoothing, longer/overlapping holds, and no-trade rebalance bands cut
turnover with minimal IC loss, raising net-of-cost Sharpe.

**Deliverables:**
- `research/momentum.py` — B0 signal + rank portfolio.
- Turnover-reduction variants: (a) EWMA/averaged signal, (b) overlapping
  sub-portfolios (fractional rebalancing), (c) no-trade rank bands (hysteresis),
  (d) longer holding.
- A comparison harness reusing `evaluation.py`.

**Evaluation metrics:** gross & net@60bps Sharpe, annualized turnover, cross-
sectional IC, breakeven bps, max drawdown.

**Success criteria:** a variant that **cuts annualized turnover ≥40%** while
**losing ≤10% of gross IC**, and whose **net@60bps Sharpe ≥ the Exp-4 model's
net@60bps** (and > B0's). Breakeven bps materially higher than 124.

**Stop criteria:** if no variant keeps net Sharpe within noise of the best gross
config, OR realistic-cost (60 bps) Sharpe stays **< 0.3** across all variants —
momentum isn't viable net of TWSE costs → halt A→D→C and re-open Path B.

**STATUS: CLEARED by the raw baseline B0 (see RESEARCH_LOG Phase A1).**
Assumption update that changes the plan: the Exp-4 model's ~100%/rebalance turnover
was an **ML artifact, not inherent to momentum**. Raw 6-month momentum turns over
only ~8.1×/yr with **breakeven ~394 bps** — so **transaction cost is NOT the binding
constraint**, and the A1 turnover-reduction variants (EWMA/bands/overlap) are
**low-value polish, deprioritised**. B0 already beats the ML (net@60 Sharpe 1.13 vs
0.62 @30) and is more stable (8/9 positive years). The two real problems surfaced
by A1 are **(i) concentration** (semis+electronics+shipping ≈ 85% of L/S P&L) and
**(ii) the 2022 momentum-crash drawdown** — i.e. D1 and C1 are where the value is.

---

## Phase D1 — Portfolio construction layer
**A1 evidence raises this to the top priority** (concentration is the dominant
residual risk; cost is already handled).
**Objective:** harvest the A1 signal more efficiently — higher risk-adjusted
return and lower drawdown via sizing, weighting, and constraints (not new signal).

**Hypothesis:** equal-weight is inefficient and concentrated (Exp-4: electronics
+61%, shipping +43% of P&L). Inverse-vol / vol-target sizing + sector caps +
covariance-shrinkage weighting improves net Sharpe and cuts concentration &
drawdown for the same signal.

**Deliverables:**
- Position sizing: inverse-volatility and portfolio vol-target.
- Constraints: per-sector weight cap; per-name cap.
- Risk weighting: shrinkage covariance (Ledoit-Wolf-style, closed-form) →
  risk-parity / min-variance tilt; turnover-penalized rebalancing.
- Ablation harness (each component on/off).

**Evaluation metrics:** net Sharpe, max drawdown, Calmar, realized vol vs target,
sector concentration (max-sector P&L share / Herfindahl), turnover.

**Success criteria:** vs A1, either **net Sharpe +≥0.15** at equal drawdown, **or
max drawdown −≥25%** at equal net return; **max single-sector P&L share cut
below ~35%**; realized vol tracks target within ±20%.

**Stop criteria:** no construction beats equal-weight *after* its added turnover
cost; OR covariance estimates are unstable OOS (realized vol >1.5× target) and
shrinkage doesn't fix it → keep the simplest sizing (inverse-vol) and move on.

**STATUS: PASSED (see RESEARCH_LOG Phase D1).** Recommended rule = **inverse-vol
sizing + 20% sector cap**: preserves Sharpe (1.21 vs 1.22), cuts maxDD 25%
(−29%→−22%), cuts max sector weight 32%→20% and P&L share 49%→37%, cost-robust to
80 bps. Passed via the maxDD-−25% criterion + large concentration reduction; no
variant improved Sharpe (construction adds risk-adjusted value, not signal — as
expected) and none reached the P&L-share<35% target (best 37%). Scope note: kept it
simple (caps + inverse-vol); did **not** pursue full covariance/min-variance
optimisation — the simple rule already clears the gate and avoids overfitting.
**Unresolved by D1: the 2022 crash (negative for every variant) → C1.**

---

## Phase C1 — Risk / regime overlay (defensive only)
**D1 evidence sharpens the target:** the single remaining weakness is the 2022
momentum-crash year (net Sharpe ≈ −1.0 to −1.4 across all D1 variants); C1's job is
to cut that left tail without materially hurting the other 8 years.
**Objective:** reduce momentum's regime crashes (Exp-4: 2018 L/S Sharpe −2.4,
2022 −0.2) via an exposure scaler, **without materially hurting net return**.

**Hypothesis:** momentum crashes cluster in market reversals / high-volatility
regimes, so a simple rule-based exposure scaler (universe trend sign, realized
vol, or a recent-market-drawdown flag) cuts left-tail drawdowns. This principled
overlay replaces the baseline's crude `mean prob > 0.52` gate.

**Deliverables:**
- Exposure-scaling overlays (evaluated separately, simplest first): (a) universe
  200-day trend on/off, (b) inverse realized-vol scaling (vol-target at book
  level), (c) momentum-crash indicator (scale down after large market drawdown).
- Overlay applied to the D1 product; walk-forward with embargo.

**Evaluation metrics:** max drawdown, worst-year Sharpe, Calmar, net Sharpe,
time-in-market, added turnover, tail (5% CVaR of rebalance returns).

**Success criteria:** **max drawdown and worst-year loss each cut ≥25%** with
**net-return drag ≤10%** → **Calmar improves**. Must hold OOS, not just in the two
known bad years.

**Stop criteria:** the overlay only helps in-sample / fails walk-forward; OR
return drag exceeds the drawdown benefit (Calmar not improved) → **drop the
overlay**; ship the D1 product unhedged and document that timing added no value.

**STATUS: FAILED — no overlay adopted (see RESEARCH_LOG Phase C1).** Four
pre-declared overlays (trend / vol / crash-flag / combined) tested; best cut maxDD
only 13% (bar was 20%) with no Sharpe gain. Root cause: D1's inverse-vol + caps
already de-risked the 2022 crash, leaving little tail to remove; timing gives back
as much upside as crash avoided. Per plan we did **not force a regime model**. Ship
D1 unhedged. (O1 trend is a benign optional Sharpe/cost improver, +0.07 Sharpe, but
does not meet the crash objective and is not adopted by default.)

---

## A → D → C outcome
- **A1 PASS** — 6-month momentum quintile L/S beats the ML baseline (Sharpe 1.22 vs
  0.62) at lower turnover; more cost-robust and more stable (8/9 positive years).
- **D1 PASS** — inverse-vol + 20% sector cap cuts maxDD ~25% and concentration
  (sector weight 32%→20%, P&L share 49%→37%) at preserved Sharpe, cost-robust to 80bps.
- **C1 FAIL** — no regime overlay clears the crash-reduction bar; ship D1 unhedged.

**Final product = the D1 rule** (deterministic, no ML): 6-month momentum (skip 5d),
top/bottom quintile, inverse-vol sizing, 20% sector cap, 20-day hold. ~1.2 net
Sharpe / ~−22% maxDD, survivorship-inflated (upper bound). A **momentum-factor
product, not alpha** — better than the LSTM-Transformer baseline at a fraction of
the complexity.

**Prototype: `research/prototype.py`** — clean, deterministic pipeline. Now
implements **D1.1** (`generate_book()` / `backtest_summary()`, same code path as
the research backtest). Does not touch train.py/inference.py.

### D1.1 refinement — per-name cap (PASSED)
`research/d1_1_pername_cap.py`. Added a **10% per-name cap** to the frozen D1 rule.
Effect vs D1: max name weight 13.5%→11.0%, sector P&L share 37%→35%, at preserved
net Sharpe (1.21→1.27, within noise) and flat drawdown (−21.9%→−22.1%); broad
across years, not one lucky year. Caps ≤7.5% start worsening drawdown; 5% notably
so — so 10% is the conservative choice. **D1 remains frozen/reproducible in
portfolio_d1.py.** Note: dual name+sector capping is iterative → realized max sector
weight ~21% (~1% over the 20% target) when the name cap binds.

**Final deployable rule = D1.1:** 6-month momentum (skip 5d) → top/bottom quintile
→ inverse-vol sizing → **20% sector cap + 10% per-name cap** → 20-day hold.
~1.27 net@60bps Sharpe / −22% maxDD (survivorship-inflated upper bound). A
momentum-factor product, not alpha.

---

## Sequence-level gates
- **Advance** A1→D1→C1 only when the current phase clears its success bar; a failed
  phase reverts to the prior layer (the product still ships at that layer).
- **Overall success:** a documented product that beats the LSTM-Transformer
  baseline on net-of-cost Sharpe **and** max drawdown, with honest (survivorship-
  and factor-aware) accounting.
- **Overall stop:** if A1 stops (momentum not viable net of cost), do not proceed to
  D/C — escalate to the Path-B (orthogonal-data) decision instead.
- **Non-goal (explicit):** claiming alpha. This track harvests a *known premium*
  well; the factor-alpha t-stat is reported to keep us honest, not to be maximized.
