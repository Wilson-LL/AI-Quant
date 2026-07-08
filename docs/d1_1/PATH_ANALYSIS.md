# Path Analysis — where to take AI-Quant next

Decision memo comparing four forward paths, grounded in the evidence from
Experiments 1–4 (see `RESEARCH_LOG.md`). Written **before** committing to Path A,
at the user's request.

## What we actually know (the evidence base)

- **There is one capturable signal in TWSE price/volume: the cross-sectional
  momentum premium.** Exp-4 factor regression: momentum beta 0.70 (t=9.58),
  residual alpha 0.62%/yr (**t=0.12 — zero**). Everything the ML learned is
  momentum.
- Gross L/S Sharpe 0.82; net@30bps 0.62; realistic-cost (~60–80bps TWSE
  round-trip) ≈ 0.4–0.5. Turnover ~100%/rebalance (~12.5×/yr).
- It is **unstable** (5/9 positive years; −2.4 Sharpe in 2018, −0.2 in 2022) and
  **concentrated** (electronics +61%, shipping +43% of P&L).
- The universe is **survivorship-biased upward**, so all figures are upper bounds.
- **Environment constraint:** we have OHLCV via `twstock` and numpy/pandas only —
  **no torch, no fundamentals/alt-data feeds.** This materially gates Path B.

Implication that shapes the whole memo: **Paths A, C, D are complementary layers
on the *same* momentum signal; Path B is the only one that seeks a *new* signal.**

## Scoring key
- **Upside** — incremental net, risk-adjusted, realistic-cost performance vs the
  current momentum baseline (1 marginal → 5 transformative).
- **Difficulty** / **Cost** — research hardness / build effort (1 low → 5 high).
- **P(success)** — probability of clearing a pre-registered success bar (defined
  per path), given our data/tools.
- **Timeline** — focused research iterations here (days), or data-gated (weeks+).
- **EV score** = Upside × P(success) ÷ Cost (rough triage number).

## Summary

| Path | Upside | Difficulty | Cost | P(success) | Timeline | **EV** |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| **A. Momentum product** | 3 | 2 | 2 | **75%** | days | **1.13** |
| **D. Portfolio optimization layer** | 3 | 3 | 3 | **75%** | days–1 wk | **0.75** |
| **C. Market timing / regime overlay** | 2.5 | 4 | 3 | 40% | 1–2 wks | 0.33 |
| **B. Residual alpha beyond momentum** | 5 | 5 | 4.5 | 15–20% | weeks+ (data-gated) | 0.22 |

---

## Path A — Momentum product
**Thesis:** stop pretending it's alpha; harvest the momentum premium *well* —
turnover control, vol-target sizing, sector caps, a drawdown overlay.

- **Expected upside (3/5):** Bounded by the premium itself (a known, somewhat
  crowded factor), but engineering is high-leverage: cutting 100%/rebalance
  turnover and adding vol-targeting + sector caps plausibly lifts realistic-cost
  Sharpe from ~0.45 to ~0.7–0.9 and roughly halves max drawdown. No new alpha,
  cleaner factor capture.
- **Research difficulty (2/5):** Momentum is the most-studied factor in equities;
  the toolkit (holding overlap, no-trade bands, time-series-momentum overlays) is
  standard and low-risk-of-self-deception.
- **Implementation cost (2/5):** Almost entirely refinements to the existing
  framework (backtest, neutralization, cost model already built).
- **P(success) 75%:** Success bar = a robust, realistic-cost, better-than-NN-baseline
  product. Main downside risk: after honest costs + survivorship correction the
  edge is thin (~0.4 Sharpe). High confidence it *works*, medium on *how much*.
- **Timeline:** Short — a few research iterations (days).

## Path B — Search for residual alpha beyond momentum
**Thesis:** find signal orthogonal to momentum/vol/size — the only route to *real*
alpha — via data the price series doesn't contain.

- **Expected upside (5/5):** If found, genuine alpha is uncorrelated with the
  crowded factor, defensible, and the only thing that makes this a *quant* edge
  rather than a factor ETF. This is the high-ceiling path.
- **Research difficulty (5/5):** Requires orthogonal data (fundamentals, earnings
  revisions, ownership/institutional flows, alt-data). Point-in-time integrity for
  fundamentals is a notorious look-ahead minefield; the search is overfit-prone and
  most candidates die at the cost/robustness gate.
- **Implementation cost (4.5/5):** New data acquisition + PIT storage + pipelines.
  **Blocked in the current environment** — `twstock` is price/volume only; needs
  external data feeds we don't have.
- **P(success) 15–20%:** Exp-4 already showed OHLCV has *no* residual alpha; finding
  orthogonal signal that survives costs and isn't overfit is genuinely low-base-rate.
- **Timeline:** Weeks-to-months, gated on data acquisition (not doable here today).

## Path C — Market timing / regime detection
**Thesis:** scale exposure with a regime signal to dodge momentum's crashes
(2018, 2022) — improve Sharpe and drawdown via timing.

- **Expected upside (2.5/5):** Mostly *defensive*. Momentum's worst years are
  regime-driven, so a trend/vol-based exposure scaler could cut drawdown and lift
  Sharpe. But timing rarely *adds return*; the honest evidence is that reliable
  timing alpha is rare, while vol-targeting/trend overlays help risk.
- **Research difficulty (4/5):** Regime/timing is among the most overfit-prone
  areas in the field — few effective regimes, tiny sample of macro cycles, easy to
  fool yourself. Demands ruthless OOS discipline.
- **Implementation cost (3/5):** Reuses the framework; add an exposure-scaling
  signal (trend of the universe, realized vol, breadth). The baseline already has a
  crude `mean prob > 0.52` gate to replace with something principled.
- **P(success) 40%:** As a *return* enhancer, low (~20%); as a *drawdown-reducer*
  overlay on momentum, medium (~40–50%) — vol-targeting/trend overlays have real
  supporting evidence. Success bar = lower drawdown without killing net return.
- **Timeline:** Medium (1–2 weeks; small macro sample means slow confidence).

## Path D — Portfolio optimization layer
**Thesis:** harvest whatever signal exists more efficiently — covariance-aware /
risk-parity weighting, vol-targeting, sector caps, turnover-penalized rebalancing.

- **Expected upside (3/5):** Doesn't create alpha but is a reliable *multiplier* on
  the signal. Exp-4 exposed exactly the inefficiencies it fixes: equal-weight,
  100% turnover, 65% P&L in two sectors. Correcting these dependably improves net
  Sharpe and drawdown for *any* underlying signal.
- **Research difficulty (2.5/5):** Well-understood (mean-variance, risk parity,
  constrained optimization); main pitfall is covariance estimation error (needs
  shrinkage / factor covariance).
- **Implementation cost (3/5):** New code — covariance estimator, a constrained
  optimizer (no cvxpy; simple risk-parity / analytic solutions in numpy), turnover
  penalty. Moderate.
- **P(success) 75%:** Improving net risk-adjusted metrics over equal-weight is
  nearly mechanical (turnover reduction + vol-target reliably help). Success bar =
  better net Sharpe / lower DD than equal-weight top-k.
- **Timeline:** Short–medium (days to ~1 week).

---

## Critical observation

**A, C, and D are not competitors — they are three layers of one product.** A
proper "momentum product" (A) *is* a clean signal + a portfolio layer (D) + a
risk/exposure overlay (a disciplined slice of C). Treating them as rival paths is a
false choice. **B is the genuinely different bet** — a new signal source — and it
is the only path to alpha that isn't a known premium, but it is high-risk and
**blocked without new data** in this environment.

## Recommendation

**Pursue Path A as the spine, with Path D folded in and a narrow, risk-only slice
of Path C — sequenced, each gated on net-of-realistic-cost, survivorship-aware OOS
evidence. Scope Path B as a parallel, longer-horizon track, explicitly contingent
on acquiring orthogonal data; do not block the near-term deliverable on it.**

Rationale:
1. **Highest EV and only fully-doable-here path.** A (1.13) and D (0.75) dominate;
   both run entirely on data/tools we have. B, despite the highest ceiling, is
   data-blocked and low base-rate; C is defensive and overfit-prone.
2. **It produces a defensible, better-than-baseline deployable system now** — a
   cost- and risk-managed momentum factor product will almost certainly beat the
   LSTM-Transformer baseline (which is a *worse*, higher-turnover momentum/beta
   proxy) at a fraction of the complexity.
3. **It builds the substrate the other paths need anyway.** C and D only make sense
   on top of a clean signal; and any future B alpha must be sized/executed through
   the same portfolio + risk layer.

**Concrete sequence (each an experiment with a pre-registered gate):**
- **A1** signal hygiene: reduce turnover (holding overlap, no-trade bands, signal
  smoothing) — target same gross IC at far lower turnover/cost.
- **D1** portfolio layer: vol-target sizing + sector caps + covariance shrinkage —
  target higher net Sharpe & lower DD than equal-weight.
- **C1** risk overlay: vol-target / trend exposure scaler to cap the 2018/2022-style
  drawdowns — target lower max-DD without material net-return loss.
- **B (parallel, unfunded until data):** define the orthogonal-data wishlist and the
  residual-alpha test (Exp-4 factor regression is the template); begin only when a
  fundamentals/flows feed is available.

**One-line answer:** build the momentum product properly (A+D, then C-for-risk);
treat true-alpha search (B) as the real long game, gated on new data — not a bigger
neural net, which Exp-4 shows has nothing left to learn from price alone.
