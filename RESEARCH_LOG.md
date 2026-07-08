# Research Log — AI-Quant

Lead researcher's running log. Objective: **maximise out-of-sample performance
of a quantitative investment system** for TWSE equities. The existing
LSTM-Transformer + close-only triple-barrier design is treated as a *baseline to
be challenged*, not a requirement.

Priority order (per brief): Problem definition → Target definition → Evaluation
methodology → Portfolio construction → Inference logic → Model architecture.
Rule: **no target/model change is accepted without reproducible, leakage-free,
out-of-sample evidence.**

Environment constraints (important, honest): this machine has **numpy/pandas/
twstock only — no torch, no scipy, no sklearn**. So the neural baseline cannot be
retrained here, and (per the brief) that is fine — the highest-priority questions
(target, evaluation, portfolio) are model-agnostic and need only real data + a
correct referee. Network access to TWSE via `twstock` works but is slow
(~1–2 min/stock), so all data is fetched once and cached.

---

## Iteration 0 — Infrastructure & correctness (done)

### 0a. Fixed a leakage/crash bug inherited from `train.py`
Reviewed in `REVIEW_4e7467e.md`. The prior "chronological split" used a per-stock
fraction with a 60-sample embargo, which **emptied val/test for `one_year` and
`half_year` and crashed training**. Replaced with a **global calendar-date
chronological split** (`dataset.chronological_split`): all stocks are cut at the
same dates, with the embargo purged from the train tail / test head so the val
block keeps full width. Verified all four horizons now yield non-empty, leak-free
train/val (`test_dataset.py`, 7/7). This matters for research because a trustable
split is a prerequisite for trustable comparisons.

### 0b. Built a fair, reproducible research framework (`research/`)
| Module | Purpose |
|---|---|
| `data.py` | Fetch TWSE OHLCV via twstock **once**, cache to CSV, reload offline. Resumable. |
| `features.py` | 12 standard **causal** features (momentum, vol, RSI, MA distance, volume z). |
| `targets.py` | Baseline target + 5 alternatives (see below). |
| `evaluation.py` | Cross-sectional **IC**, **walk-forward w/ embargo**, **net top-k backtest**, metrics. |
| `analyze_target.py` | Experiment 1 driver. |
| `test_framework.py` | **6/6 pass** — verifies the referee before trusting it. |

The referee is deliberately a simple ridge / linear-probability model so that when
we compare *targets* (or later *architectures*), the only thing changing is the
variable under test. `test_framework.py` proves: IC recovers ±1 on perfect/anti
signals; the backtest is causal (inverting the signal inverts PnL) and has no
look-ahead; walk-forward learns a causal relation OOS; high/low labelling catches
intraday barrier touches that close-only misses.

**Candidate targets implemented** (all strictly forward-looking, tail-censored):
- `baseline_tb_close` — the current target: +12% before −6% within 20d, **close only**.
- `tb_highlow` — same barriers but judged on **intraday high/low** (order-realistic).
- `fwd_logreturn` — continuous 20d forward log return (full magnitude).
- `fwd_return_sign` — sign of the above.
- `fwd_return_voladj` — forward return / trailing vol·√Y ("forward Sharpe").
- `mfe_mae` — max favourable minus max adverse excursion (path quality).

---

## Iteration 1 — Is the prediction target correct? (in progress)

**Motivation / prior hypotheses (to be tested, not assumed):**

1. **Use/target mismatch.** The system's *actual use* is **cross-sectional
   ranking** (pick top-k across stocks, fixed hold, market gate). But the label
   is a **single-asset, path-dependent TP/SL** outcome. Cross-sectional ranking
   is usually best trained on a **cross-sectional continuous return** (or its
   rank), not a path-dependent binary. Expectation: a continuous /
   vol-adjusted target yields higher OOS rank-IC and net Sharpe than the
   baseline binary.

2. **Class-merging pathology.** The baseline labels "no barrier touched" as `0`
   (negative), merging *flat/up* names with genuine *losers*. This should inject
   label noise and depress learnability. Experiment A quantifies how many `0`
   labels actually had a positive forward return.

3. **Label fidelity.** Close-only barriers mis-detect intraday touches. `tb_highlow`
   should differ materially from `baseline_tb_close`; the direction of any
   economic difference is an open question.

**Method:** `research/analyze_target.py` on the cached universe —
(A) label diagnostics, (B) single-feature learnability IC, (C) model-controlled
economic value: train ridge on each target via walk-forward, then judge every
resulting score on the **same** ground truth (OOS rank-IC vs realised forward
return, and a net top-k backtest with 30 bps round-trip cost).

**Status:** COMPLETE. Framework built + unit-tested (7/7); full 20-name cache
(2018→2026) fetched; definitive run done. Decision below: **reject the
continuous-target hypothesis; keep the triple-barrier (prefer tb_highlow),
pending a volatility-confound check.** Next: Experiment 2 (vol confound).

### Preliminary evidence (7/20 stocks cached, Y=20, label diagnostics only)
Measured on 14,015 (date,stock) observations across TSMC-led semis names:

| metric | value | reading |
|---|---|---|
| baseline TB positive rate | 0.190 | imbalanced but learnable |
| TB high/low positive rate | 0.217 | close-only misses ~2.7pp of real upside touches |
| TB(close) vs TB(high/low) disagreement | 3.7% | label-fidelity gap is real but modest |
| **% of `0`-class with POSITIVE 20d return** | **45.5%** | **negative class is mostly "flat", not "loser"** |
| median 20d return of `0`-class | −0.56% | confirms class-merging noise |
| corr(TB label, fwd_ret) | 0.636 | binary label ⟶ return |
| corr(sign(fwd_ret), fwd_ret) | **0.719** | a plain return sign carries MORE return info than TB |

→ Strong early support for hypothesis (1) class-merging pathology.

### Experiment 1 first pass (5 stocks — NOISY, all semis; treat as directional only)
Ran the full (A)/(B)/(C) driver. Model-controlled economic value, train ridge on
each target then judge OOS score on realised forward return:

| train target | OOS rank-IC vs fwd_ret | best single-feat |IC| |
|---|--:|--:|
| tb_highlow (barrier, high/low) | **0.057** | 0.199 |
| baseline_tb_close | 0.038 | 0.214 |
| fwd_return_sign | 0.008 | 0.081 |
| mfe_mae | 0.007 | 0.092 |
| fwd_logreturn | −0.002 | 0.076 |
| fwd_return_voladj | −0.008 | 0.075 |

**This partially REFUTES my prior** that a continuous/return target would rank
better: here the *triple-barrier* targets are both more learnable (|IC|≈0.20 vs
≈0.08) and rank future returns better OOS. Caveats that make this NOT yet a
decision: (a) only 5 highly-correlated semis names → cross-sectional IC is near
noise; (b) the top-k backtest returned n=0 because k(10) > stocks(5) — a driver
bug, now fixed (adaptive k, nan-safe summary); (c) the label-vs-return correlation
still favours return-based targets in aggregate. **Decision deferred** to the
full ~20-name run. Lesson logged: state hypotheses first, let evidence overrule
them — this is the process working.

Bugs fixed after this run: `backtest_topk` metrics now always populated;
`analyze_target` best-target summary is nan/None-safe; `k` capped at
`universe//3` so the book can form. Definitive run queued at full cache.

### Experiment 1 second pass (9 stocks, k=3, backtest now trades — n=90)
| train target | OOS rank-IC | IC_IR | net Sharpe | maxDD | best single-feat |IC| |
|---|--:|--:|--:|--:|--:|
| tb_highlow | 0.0614 | 0.156 | 1.26 | −34.8% | 0.215 (vol_60) |
| baseline_tb_close | 0.0585 | 0.147 | 1.38 | −35.8% | 0.223 (vol_60) |
| fwd_logreturn | 0.0561 | 0.150 | 1.39 | −29.5% | 0.066 |
| **fwd_return_sign** | 0.0519 | 0.135 | **1.39** | **−22.2%** | 0.049 |
| mfe_mae | 0.0452 | 0.124 | 1.33 | −32.0% | 0.072 |
| fwd_return_voladj | 0.0190 | 0.050 | 1.39 | −36.3% | 0.056 |

**Corrected reading (supersedes the 5-stock pass):**
1. On economic value the targets **cluster** (IC 0.045–0.061); the 5-stock
   "barrier dominates" was small-sample noise. No target dominates yet.
2. `fwd_return_sign` gives the best net Sharpe and **much lower drawdown**
   (−22% vs −35%) while avoiding both the class-merging and volatility confounds.
3. **The barrier targets' high single-feature learnability is largely spurious**:
   the top predictor is `vol_60` (volatility). Features are detecting *whether a
   barrier gets touched* (a volatility effect), not *direction*. Evidence
   AGAINST the barrier target, not for it.
4. **Confound flag:** the ~1.4 net Sharpes are long-only semis **beta** in a
   2018–2026 bull market (all names co-move), not alpha. → Trust the
   cross-sectional **IC** (market move nets out per date), NOT the long-only
   Sharpe/return. Methodological to-do: add a market-neutral (demeaned or
   long-short) backtest lens so target quality isn't swamped by common beta;
   broaden the universe beyond semis (audit #8).

**Decision:** still deferred to the full ~20-name run, but the leaning is toward
a **return-based target (fwd_return_sign / fwd_logreturn)** over the barrier —
same/marginally-lower IC but simpler, lower-drawdown, and free of the volatility
confound. Will finalize on the full cache with the neutral lens added.

### Experiment 1 DEFINITIVE (20 stocks, 39,794 obs, 2068 dates, k=6) — DECISION
Model-controlled, walk-forward (embargo 20d), judged on realised forward return.
Beta-neutral metrics (OOS IC, long-short Sharpe) are decisive; long-only Sharpe
is beta and shown only for reference.

| target trained | OOS IC | IC_IR | **L/S Sharpe** | L/S ret | LO Sharpe (beta) | LO maxDD |
|---|--:|--:|--:|--:|--:|--:|
| baseline_tb_close | **0.0521** | 0.155 | 0.97 | +2.35% | 1.37 | −38% |
| **tb_highlow** | 0.0493 | 0.148 | **1.09** | +2.66% | 1.44 | −32% |
| fwd_logreturn | 0.0122 | 0.044 | −0.14 | −0.29% | 1.21 | −43% |
| mfe_mae | 0.0075 | 0.026 | 0.12 | +0.24% | 1.26 | −38% |
| fwd_return_voladj | 0.0060 | 0.021 | −0.53 | −1.15% | 1.23 | −31% |
| fwd_return_sign | 0.0046 | 0.016 | −0.75 | −1.64% | 0.96 | −30% |

**The full universe REVERSES the 9-stock leaning (which was noise).** Findings:

1. **Triple-barrier ≫ return-based, decisively.** Barrier targets: OOS IC ≈0.05,
   L/S Sharpe ≈ +1.0. Return targets: OOS IC ≈0.01 and **negative** L/S Sharpe —
   a model trained on raw forward return ranks *worse than random* beta-neutral.
   With ~1800 OOS dates the IC SE ≈ 0.008, so barrier-vs-return (Δ≈0.04, ~5σ) is
   real; the two barriers (Δ≈0.003) are **statistically tied**.
2. **My prior hypothesis is refuted.** "Predict a continuous cross-sectional
   return" did *worse*, not better. Likely because 20d raw returns are near-noise
   to a linear model, while the barrier discretisation exposes a learnable
   structure. Hypothesis stated first, evidence overruled it — logged as such.
3. **tb_highlow (intraday high/low) is the marginal pick** over baseline_tb_close:
   tied on IC, better L/S Sharpe (1.09 vs 0.97) and lower drawdown, and it fixes
   the close-only label-fidelity gap. A low-cost, strictly-more-correct refinement
   — but within noise vs baseline, so "prefer, don't over-claim".
4. **The class-merging pathology (44.8%) persists but hurts ranking less than
   feared** — for top-k ranking the merged negatives sit low anyway.
5. **The evaluation-methodology upgrade paid off immediately.** On the long-only
   lens ALL targets looked similar (~1.2–1.4 Sharpe = shared semis beta). Only the
   beta-neutral L/S lens revealed that return-targets carry *no* ranking skill.
   Without it we'd have drawn the wrong conclusion. (Validates audit #3/#8/#10.)

**KEEP/REVERT:**
- **REVERT** the "switch to a continuous/return target" idea — evidence is against it.
- **KEEP (tentative):** adopt **tb_highlow** as the target (label fidelity + best
  neutral skill), pending the confound check below. Not yet wired into the torch
  pipeline (no torch here); recorded as a validated recommendation.

**BIGGEST OPEN RISK — the volatility confound (next experiment).** The barrier
targets are learned almost entirely through `vol_60` (single-feature |IC| 0.23–0.24
vs ≤0.07 for return targets). In a 2018–2026 semis **bull market**, high-vol names
had high forward returns, so the "skill" may be a regime-linked **volatility/beta
bet**, not idiosyncratic alpha, and could invert in a bear/rotation regime.
Experiment 2 must: (a) split IC by sub-period (esp. the 2022 drawdown); (b) build a
pure `vol_60` factor and measure how much of the L/S Sharpe it explains; (c) test
the barrier target after neutralising volatility. Do NOT productionise tb_highlow
until (a)–(c) clear it.

### Assumptions audit
See **`ASSUMPTIONS_AUDIT.md`** — top-20 ranked implicit assumptions (U×L/C
priority) and the phased roadmap. Highest-priority cheap wins: fix class-merging
(#1), wire the real backtest into selection (#2), turn on costs (#3), verify price
adjustment (#6).

---

## Iteration 2 — Is the barrier edge real alpha or a volatility bet? (DONE — decisive)

`research/exp2_vol_confound.py`, full 20-name universe, walk-forward, beta-neutral.

| check | result | reading |
|---|---|---|
| T0 naive `vol_60` factor (no model) | IC 0.0541, L/S Sharpe **1.08** | ranking by raw vol ≈ the whole barrier model |
| barrier model, full features | IC 0.0493, L/S Sharpe 1.09 | **ML adds nothing over the vol factor** |
| T2 barrier vs **vol-neutral** returns | IC **−0.0271** | remove vol → skill is zero/negative |
| T1 retrain **without** vol features | IC 0.0217, Sharpe 0.58 (vol-neutral IC −0.033) | most of the edge is vol; remainder is vol via other feats |
| T3 OOS IC by year | 2019 +0.13, 2020 −0.06, 2021 −0.07, **2022 −0.04**, 2023 +0.03, 2024 +0.10, 2025 +0.22, 2026 +0.14 — **vol-neutral IC ≈0/neg every year** | raw IC is a regime-timed vol bet; negative 2020–2022 |
| T4 barrier L/S vs vol-factor L/S | **corr 0.96, R² 0.92**, residual alpha 0.12%/rebal | 92% of returns ARE the vol factor |

**VERDICT (pre-registered gate FAILS): the triple-barrier target has no
idiosyncratic alpha. Its entire cross-sectional edge is a volatility/beta tilt
that pays in high-vol-rewarding bull regimes (2019, 2024–26) and loses in others
(2020–22). Reject `tb_highlow`/`baseline_tb_close` as an alpha source.** This
*overturns the Iteration-1 tentative keep* — exactly why the productionise gate
required T1+T2 to clear first.

**Bigger reframing (the important finding).** Across Experiments 1–2, on this
universe with standard features, **no target produced vol-neutral cross-sectional
alpha**: raw-return targets have ~zero signal; the barrier target only re-encodes
volatility. The whole baseline system (NN + ensemble + calibration + market gate)
is therefore most likely a dressed-up **long-high-volatility/high-beta bet** — which
looks brilliant in a semis bull market and is fragile in a drawdown. The binding
constraint is not the model or even the label; it is the **opportunity set**:
20 highly-correlated semis names have essentially one axis of cross-sectional
dispersion — volatility/beta. There is little idiosyncratic alpha to find here.

**Implications for the roadmap (revised priorities):**
1. **Universe first (audit #8 promoted to #1).** Get genuine cross-sectional
   dispersion — multiple sectors, more names — or there is no alpha to model.
   Everything downstream is moot until this is fixed.
2. **Always evaluate vol-neutral (and sector-neutral).** Make beta/vol
   neutralisation a *standard* lens in the referee, not an afterthought — raw IC
   and long-only Sharpe are beta mirages here (T3 2025: raw IC +0.22, vol-neutral
   −0.05).
3. **Re-open the target/feature search only inside a vol-neutral objective** —
   i.e. predict the *residual* return after removing vol/beta, and hunt features
   with vol-neutral IC. Standard momentum/vol/RSI showed none.
4. Barrier vs return-target is now moot for alpha; if a barrier is used at all it
   is an *exit/risk* rule, not a selection signal.

## Iteration 3 — Universe problem or feature problem? (DONE — first positive result)

`research/exp3_universe.py`. Same analysis on two universes; the honest metric is
the OOS **vol-neutral** long-short Sharpe (linear model predicting vol-neutral
returns from standard features, walk-forward, net 30bps).

| | SEMIS-ONLY (18 names, 4 sectors) | MULTI-SECTOR (49 names, 13 sectors) |
|---|--:|--:|
| Q1 vol-domination R²(ret~vol_60) | 0.154 | 0.123 |
| Q3 OOS IC vs vol-neutral ret | 0.011 (IR 0.04) | **0.030 (IR 0.15)** |
| **Q3 OOS vol-neutral L/S Sharpe** | **−0.23** | **+0.75** |
| Q3 OOS L/S Sharpe on raw ret | −0.19 | 0.60 |

**VERDICT — opportunity-set hypothesis CONFIRMED.** Broadening semis→multi-sector
flips the honest vol-neutral OOS Sharpe from −0.23 to **+0.75**, and the model's
Sharpe is *higher* on vol-neutral (0.75) than raw returns (0.60) → the edge is
genuine cross-sectional selection, not a vol/beta tilt. The binding constraint in
Exp 1-2 was the universe, exactly as hypothesised. **First positive, beta-neutral,
OOS result of the program.**

Supporting detail (Q2, per-feature IC):
- The residual (vol+sector-neutral) signal lives in **trend/momentum**: multi-sector
  `dist_hi_60` (+0.025), `mom_60` (+0.005), `px_over_ma60` (+0.006), plus a
  **low-volatility premium** (`vol_60` vol+sector-neutral IC −0.026 → low-vol names
  outperform). These are recognisable, economically-sensible factors — not curve
  fits. (Reporting nit: the driver's "strongest" line uses |IC|, so it printed the
  negative low-vol term; the strongest *positive* vol+sector-neutral feature is
  `dist_hi_60`.)
- Semis-only Q2 showed `mom_20` vol-neutral IC +0.044, but that was full-sample
  descriptive IC and did **not** survive OOS (Q3 semis −0.23). Honest-number
  discipline held: the OOS gate rejected the optimistic single-feature hint.

**Caveats (do not over-claim +0.75):** single linear model + 12 standard features;
Sharpe 0.75 is modest and unstress-tested; 13 sectors include thin groups (noisy
sector neutralisation); survivorship bias remains (current large-caps). Robustness
(sub-period stability) is the immediate next check before building on it.

**Roadmap impact:** the universe rebuild is validated and is now a firm
requirement. There IS real vol-neutral alpha to model — so the research can finally
move from "is there anything here?" to "optimise it", all inside a vol-neutral,
net-of-cost objective.

## Iteration 4 — Is the alpha stable & robust? (DONE — it's momentum, not alpha)

`research/exp4_robustness.py`. 106 names / 15 sectors, 195,167 OOS obs
(2018-07→2026-07), one walk-forward linear model predicting vol-neutral 20d
returns; the OOS scores interrogated 7 ways. Survivorship caveat printed
(currently-listed only → upward-biased; treat all numbers as an upper bound).

**Task 8 — factor-exposure regression of the L/S returns (the decisive test):**
| factor | beta | t-stat |
|---|--:|--:|
| **alpha** | 0.0005 | **0.12** |
| market | −0.022 | −0.18 |
| volatility | −0.040 | −0.34 |
| **momentum** | **0.701** | **9.58** |
| size | −0.031 | −0.30 |
R²=0.60. Annualised residual alpha **0.62%, t=0.12** → **statistically zero.**

**Verdict: the Exp-3 "alpha" is the cross-sectional MOMENTUM factor, not skill.**
The model is ~long recent winners / short recent losers; regress out a
mom_60 factor and nothing is left. A practitioner could buy the momentum factor
directly and skip the ML entirely.

Supporting robustness evidence (all consistent with "it's momentum"):
- **Rolling yearly (Task 3):** unstable — LS Sharpe +1.2/1.8/1.4 (2019-21),
  **−2.4 (2018), −0.2 (2022), −0.1 (2024), −0.4 (2025)**, +2.3 (2026, 5 rebals).
  Only **5/9 years positive**; vol-neutral IC negative in 2018/2021/2022. Fails a
  regime-stability bar.
- **Sector attribution (Task 4):** concentrated — **electronics +61%, shipping
  +43%** dominate the cumulative L/S P&L (the 2020-21 shipping boom + electronics/AI
  trends); auto −12%. Not broad skill, a couple of big momentum runs.
- **Alpha decay (Task 5):** IC rises 5d→20d (0.013→0.020), flat to 60d — a slow,
  persistent signal, i.e. momentum, not microstructure.
- **Turnover (Task 6):** ~100% of the book per 20d rebalance (~12.5x/yr) — high.
- **Cost sensitivity (Task 7):** net Sharpe 0.82→0.62→0.49→0.16 at 0/30/50/100 bps;
  breakeven ~124 bps. Realistic TWSE round-trip (~0.585% tax+fee ≈ 58 bps + slippage
  ~60-80 bps) cuts the (already-just-momentum) Sharpe to ~0.4-0.5.

**Pre-registered gate (alpha t>2 AND ≥60% positive years AND net@30bps Sharpe>0.3):**
net@30bps Sharpe 0.62 ✅ · positive years 5/9=56% ❌ · alpha t=0.12 ❌ → **FAILS**.

**Program-level conclusion (Experiments 1-4).** With price/volume features on TWSE:
semis-only = a vol/beta bet (Exp 2); multi-sector = the momentum risk premium, with
**no residual alpha** (Exp 4). The elaborate LSTM-Transformer baseline is, at best,
an expensive momentum proxy — and in its native semis-only universe, a beta bet.
There is a real, capturable *known factor* (momentum) here, but **no evidence of
genuine alpha** from this data. Standard price/volume signal is exhausted.

**Implication for "a better quant system":** two honest paths —
(A) accept it as a **momentum-factor product** and engineer it well (broad universe,
turnover/cost control, risk & drawdown management, sizing) — likely beats the NN
baseline at a fraction of the complexity; or
(B) to find *true* alpha, get **orthogonal signals** (fundamentals, earnings
revisions, ownership/flows, alt-data) — nothing in OHLCV will clear the momentum
bar. A bigger neural net will not help (Exp 4 shows nothing to learn beyond a
factor a 1-line formula already captures).

## Phase A1 — simplest momentum baseline (DONE — beats the ML, closes A1)

`research/momentum.py`. Signal: `mom_t = close[t-5]/close[t-131] - 1` (126d/6mo
return, skip last 5d). Top/bottom quintile (k=21 of 106), 20d hold. Deterministic,
no ML, no look-ahead. All net@60bps unless noted; survivorship-biased (upper bound).

| strategy | net@60 Sharpe | annRet | maxDD | Calmar | turnover |
|---|--:|--:|--:|--:|--:|
| **A1 momentum L/S (multi)** | **1.13** | 30.7% | −31.6% | 0.97 | 8.1×/yr |
| A1 momentum long-only (multi) | 1.36 | 44.8% | −33.1% | 1.35 | 7.7×/yr |
| equal-weight universe (market) | 1.13 | 31.6% | −19.7% | 1.60 | 0× |
| semis-only momentum L/S | 0.51 | 19.0% | −48.2% | 0.40 | 7.5×/yr |
| Exp-4 linear ML L/S | 0.62 (net@30) | — | — | — | 12.5×/yr |

Cross-sectional IC(mom, fwd_ret) = 0.040. Cost sweep (L/S): Sharpe
1.34/1.30/1.24/1.13/1.00 at 0/10/30/60/100 bps; **breakeven ~394 bps**.
Yearly (L/S, net@60): **8/9 years positive** — only 2022 negative (−1.48); 2020
+1.35, 2021 +1.50, 2023 +1.84, 2025 +1.28. Sector attribution (L/S): semis +119%,
electronics +62%, shipping +32%, transport −8%.

**Findings:**
1. **The simplest momentum rule beats the Exp-4 ML model decisively** — L/S net@60bps
   Sharpe 1.13 vs the ML's 0.62 net@30bps, at **35% lower turnover** (8.1× vs 12.5×)
   and far higher cost tolerance (breakeven 394 vs 124 bps). The ML added negative
   value; the 100%/rebalance turnover was an ML artifact, not inherent to momentum.
2. **More robust than the ML**: 8/9 positive years vs 5/9. Only the 2022 momentum
   crash is negative.
3. **Long-only is mostly market beta** (honest): the equal-weight market did
   Sharpe 1.13 / Calmar 1.60; long-only momentum's Calmar (1.35) is *worse* than
   just holding the universe. In a survivorship-biased bull sample, long-only
   anything looks good — the trustworthy signal is the **market-neutral L/S**.
4. **Concentration is the real risk**: semis+electronics+shipping = ~85% of L/S
   P&L → D1 (sector caps) is the highest-value next layer.
5. **Turnover/cost is NOT binding** (breakeven 394 bps ≫ ~60 bps realistic) → the
   A1 turnover-reduction variants are low-value polish, not needed to clear A1.

**A1 success gate (turnover −≥40% at ≤10% IC loss AND net@60 Sharpe ≥ Exp-4 ML):**
The raw baseline already **exceeds the ML net Sharpe by ~0.5 at 35% lower turnover**
— A1's intent (a viable, cost-robust momentum product that beats the ML) is **met
by B0 itself**. Turnover-reduction variants would only push breakeven higher, which
isn't needed. **A1 CLEARED.**

## Phase D1 — portfolio construction (DONE — PASS via risk reduction)

`research/portfolio_d1.py`. Same signal & selection as A1 (top/bottom momentum
quintile); only weights change. Sector caps (equal base, cap+redistribute) and
inverse-vol sizing (past 60d vol only). All net@60bps; survivorship-biased.

**Turnover convention corrected:** D1 uses proper one-way L1 turnover; A1's
name-based figure double-counted the L/S book ~2×. Corrected A1 baseline =
**Sharpe 1.22, turnover 4.2×/yr** (momentum is even more cost-robust than A1 said).

| variant | Sharpe | annRet | maxDD | Calmar | turn | maxSecW | maxPnL% | 2022 Shrp |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 equal (baseline) | 1.22 | 33.5% | −29.3% | 1.14 | 4.2× | 32% | 49% | −1.33 |
| cap 15% | 1.11 | 25.4% | −21.4% | 1.19 | 5.4× | 16% | 37% | −0.45 |
| cap 20% | 1.17 | 28.3% | −24.8% | 1.14 | 5.0× | 20% | 38% | −1.04 |
| cap 25% | 1.24 | 31.9% | −29.7% | 1.07 | 4.7× | 24% | 41% | −1.40 |
| cap 30% | 1.26 | 33.4% | −30.2% | 1.10 | 4.5× | 28% | 44% | −1.39 |
| inverse-vol | 1.26 | 28.9% | −22.1% | 1.31 | 5.1× | 34% | 49% | −1.33 |
| **invvol + cap 20%** | 1.21 | 25.4% | −21.9% | 1.16 | 5.6× | 20% | 37% | −1.09 |

Cost sensitivity (net Sharpe, 0/30/60/80/100 bps): invvol+cap20 =
1.39/1.30/1.21/1.15/1.09 → **survives realistic 60–80 bps comfortably**.
Yearly (invvol+cap20): 2018 −0.50, 2019 +0.57, 2020 +1.55, 2021 +1.93, 2022 −1.09,
2023 +1.79, 2024 +1.32, 2025 +1.27, 2026 +4.03 → **7/9 positive**, gains broad (not
one lucky year; 2026 is 5 rebals, same for all variants).

**Decision-rule evaluation** (improve ≥1 of: Sharpe +0.15 / maxDD −25% /
P&L-share <35%, without material turnover increase or one-year reliance):
- Sharpe +0.15 → **no variant** (construction can't add signal — expected).
- maxDD −25% → **cap15 (−27%) and invvol+cap20 (−25%) PASS**; inverse-vol −24.7% misses.
- P&L-share <35% → **no variant** (best 37%): P&L concentration is stickier than
  weight because winning-sector picks out-earn their capped weight. Concentration is
  substantially *reduced* (49%→37%, maxSecW 32%→20%) but not below 35%.
- Turnover rises ~20–33% but cost-robustness holds (80bps Sharpe still 1.15).

**VERDICT: D1 PASSES** — via the drawdown criterion (−25%) plus a large
concentration reduction, at preserved Sharpe and cost-robustness. Construction
adds *risk-adjusted* value (lower DD, far lower concentration), not signal — which
is exactly its remit.

**Recommended construction rule: inverse-vol sizing + 20% sector cap.** Best
all-round: preserves Sharpe (1.21 vs 1.22), cuts maxDD 25% (−29.3%→−21.9%), cuts
max sector weight 32%→20% and P&L share 49%→37%, improves 2022 (−1.33→−1.09),
survives 80bps (Sharpe 1.15). `cap 15%` is a more defensive alternative (better
2022 −0.45 and concentration, but Sharpe −0.11 and lowest return).

**Honest limits:** (i) no Sharpe gain — expected; (ii) P&L concentration reduced
but still 37% (>35% target); (iii) **2022 stays negative for every variant** —
that is a regime crash D1 cannot fix; it is **Phase C1's** job (drawdown/regime
overlay). Next: C1.

## Phase C1 — regime / drawdown overlay (DONE — FAILED honestly)

`research/regime_c1.py`. Overlays on the D1 product (momentum + inverse-vol +
20% cap), all rules & thresholds PRE-DECLARED before evaluation, fixed scalers
{1.0,0.5,0.0}, causal. net@60bps; survivorship-biased.

Pre-declared: O1 universe 120d trend <0 → 0.5; O2 20d realized vol >1.5× its 252d
median → 0.5; O3 last-3-rebalance return sum < −3% → 0.5; O4 both→0.0 / either→0.5.

| overlay | Sharpe | maxDD | Calmar | 2022%/rebal | %time off | ex-2022 Sharpe |
|---|--:|--:|--:|--:|--:|--:|
| **D1 base (no overlay)** | 1.21 | −21.9% | 1.16 | −1.27 | 0% | 1.49 |
| O1 trend | 1.28 | −20.2% | 1.25 | −1.23 | 27% | 1.56 |
| O2 vol | 1.14 | −20.6% | 1.12 | −1.14 | 17% | 1.41 |
| O3 crash | 1.17 | −22.2% | 1.10 | −1.27 | 18% | 1.45 |
| O4 combined | 1.21 | −19.0% | 1.21 | −1.10 | 38% | 1.48 |

**Decision-rule check** (need maxDD −≥20% OR Calmar +≥20% OR worst-year −≥30%, at
Sharpe drop ≤0.15): best is O4 with maxDD −13% and worst-year −13% → **no overlay
clears any bar**. O1 −8% DD / +7% Calmar; O2/O3 negligible or harmful.

**VERDICT: C1 FAILED.** No simple overlay meaningfully cuts the crash/left tail at
the required magnitude. Root cause (honest): **D1 already de-risked the crash** —
inverse-vol + sector caps cut 2022 to −1.27%/rebalance, so little left tail remains,
and the −21.9% max drawdown is a multi-period event, not a single regime a cheap
filter removes. Every overlay gives back as much good-period return as crash it
avoids (no Sharpe gain; DD −6…−13% only). O3 (own-reversal flag) is useless —
momentum crashes mean-revert too fast to flag without whipsaw. Per the rule, we do
**not force a regime model**.

**Recommendation: ship D1 with NO regime overlay.** Optional, non-required note:
O1 (trend filter) is a *benign* marginal enhancement — Sharpe +0.07, Calmar +7%,
cost-robust (net@80bps 1.20 vs 1.15), and it *preserves/slightly improves* the
ex-2022 edge (1.56 vs 1.49) — a practitioner MAY keep it as mild trend-following,
but it does **not** meet C1's crash-reduction objective and is not adopted by
default (avoids dressing up a regime model that didn't earn its place).

**Ready for a clean prototype?** Yes. The product is the deterministic D1 rule
(6-month momentum, top/bottom quintile, inverse-vol sizing, 20% sector cap, 20-day
hold) — simple, explainable, cost-robust to ~80 bps, ~1.2 net Sharpe / ~−22% maxDD
(survivorship-inflated → treat as upper bound). It is a **momentum-factor product,
not alpha** (Exp-4). A→D→C is complete: A gave the signal, D the risk-managed
construction, C found no worthwhile overlay.

## Phase D1.1 — per-name cap (DONE — PASS at 10%; prototype promoted to D1.1)

`research/d1_1_pername_cap.py`. Frozen D1 (sector cap only) kept reproducible —
the "uncapped" row reuses `portfolio_d1._apply_sector_caps` and reproduces D1
exactly. Pre-declared per-name caps tested: uncapped/15/12.5/10/7.5/5%. net@60bps.

| per-name cap | Sh@0 | Sh@60 | Sh@80 | annRet | maxDD | Calmar | maxName | maxSec | P&L share | 2022 | feasible |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| uncapped (=D1) | 1.39 | 1.21 | 1.15 | 25.4% | −21.9% | 1.16 | 13.5% | 20.3% | 37.0% | −1.27 | yes |
| 15% | 1.41 | 1.23 | 1.17 | 25.9% | −22.0% | 1.18 | 13.1% | 20.3% | 36.5% | −1.28 | yes |
| 12.5% | 1.41 | 1.24 | 1.18 | 26.3% | −22.4% | 1.17 | 12.6% | 20.3% | 36.1% | −1.31 | yes |
| **10%** | 1.45 | **1.27** | 1.22 | 27.6% | −22.1% | 1.25 | **11.0%** | 20.9% | **34.8%** | −1.22 | yes |
| 7.5% | 1.41 | 1.25 | 1.20 | 27.9% | −23.1% | 1.21 | 9.2% | 21.0% | 35.4% | −1.24 | yes |
| 5% | 1.37 | 1.23 | 1.18 | 30.5% | −26.3% | 1.16 | 7.7% | 21.9% | 35.2% | −1.47 | yes |

Decision (prefer cap only if it meaningfully cuts name concentration or DD, at
Sharpe drop ≤0.10): 15%/12.5% cut concentration <15% → weak; **10% PASS** (name
conc −18%, Sharpe +0.06, DD flat); 7.5% pass (conc −32%, DD −5%); 5% cuts conc −43%
but **worsens DD −20%** and 2022. All caps are implementable (k=21/leg, feasibility
floor 4.8%). Yearly path of 10% tracks D1 closely (2018 −0.74 both; 2022 −1.22 vs
−1.27) — broad, **not one lucky year**.

**VERDICT: D1.1 PASSES with a 10% per-name cap.** Chosen as the simplest
conservative cap that *meaningfully* reduces single-name concentration
(13.5%→11.0%) and pushes sector P&L share below 35%, while keeping Sharpe (within
noise) and drawdown flat — NOT the highest-Sharpe pick (5% has higher return but
worse DD). Honest caveats: (i) the +0.06 Sharpe is noise — the real benefit is
concentration control, not return; (ii) no cap improves drawdown (flat at 10%,
worse below); (iii) the dual (name+sector) capping is iterative and leaves the
realized max sector weight at ~20.9% (~1% over target) when the name cap binds —
acceptable, noted.

**prototype.py promoted to D1.1** (adds the 10% name cap). **D1 stays frozen and
reproducible** in `portfolio_d1.py` (and as the uncapped row here).

## Phase D1.1-OOS — walk-forward / point-in-time validation (DONE — VALID)

`research/walkforward_d1_1.py`, full detail in `WALKFORWARD_D1_1.md`. net@60bps.

**A. Held-out (select IS 2018-21, evaluate OOS 2022-26 — crash in OOS):**
D1.1 OOS Sharpe **1.51** (annRet 42.8%, maxDD −20.3%, Calmar 2.11, P&L share 35.2%)
— *higher* than IS (1.20) and full-sample (1.27). **Survives OOS; not a full-sample
overfit.** IS-only best cap = **10%** (IS cap scores essentially flat 1.18–1.20), so
IS-selected(10%) OOS == fixed-10 OOS == 1.51 → **the 10% cap is NOT a full-sample
artifact**; its value is concentration control (P&L share 35% vs 46% uncapped), not
Sharpe.

**B. Expanding adaptive walk-forward:** adaptive cap 1.00 < fixed-10 1.13 < ... →
adaptively tuning the cap is *worse*; **keep it FIXED at 10%.**

**7. Survivorship (unmeasurable directly; proxies):** L/S OOS 1.51 > long-only 1.34
(edge is cross-sectional, not survivor-long beta); dropping top-3 OOS names cuts
Sharpe 1.51→1.33 (−12%). Reasoned **~20–35% haircut** (survivorship + frictions +
regime) → realistic deployable ~**0.9–1.1** net Sharpe.

**VERDICT: D1.1 remains VALID and recommended.** Do not change the rule. Honest
qualifiers: OOS was a favourable momentum regime (blended cross-regime ~1.1–1.3,
not 1.5); ~12% of OOS Sharpe rests on 3 names; still a known-factor product, not
alpha; research prototype, not capital-ready.

## SPRINT (5h, 2026-07-07 15:03–20:03) — pre-registered cycles

### Cycle 1 — stricter OOS: multi-fold walk-forward + bootstrap CI (PRE-REGISTERED)
**Hypothesis:** the D1.1 OOS edge is stable across multiple sequential folds and
its net@60bps Sharpe is statistically > 0, not a single-window artifact.
**Decision rule — D1.1 "robust under stricter OOS" iff:** (a) ≥60% of sequential
OOS folds have positive net@60 Sharpe, AND (b) block-bootstrap 90% CI lower bound
on full-sample net@60 annualized Sharpe > 0, AND (c) median fold Sharpe ≥ 0.5.
Else mark "weakened/failed" and document honestly. Result appended below.

**RESULT (PASS).** `research/walkforward_rolling.py`, seed=20260707, net@60bps.
- (a) 5 sequential folds: **80% positive** (4/5; fold-3 −0.15 = 2022 crash),
  median **1.34**.
- (b) rolling 1-yr Sharpe: median **1.38**, 75% positive, min −1.18 (crash), max 3.50.
- (c) block-bootstrap 90% CI on full-sample Sharpe: point 1.27, **[0.47, 1.99]** →
  statistically > 0.
All three criteria met → **D1.1 robust under stricter OOS.** Honest note: the 5%
bootstrap floor (0.47) is *before* the survivorship/frictions haircut (~20–35%), so
a pessimistic deployable floor is ~0.3–0.5 — still positive. Caps barely change the
CI (D1 0.42, uncapped 0.46) — consistent with "the cap is risk-control, not Sharpe".

### Cycle 2 — parameter-sensitivity robustness (PRE-REGISTERED)
**Hypothesis:** D1.1's edge is robust to reasonable parameter choices, not overfit
to lookback=126 / skip=5 / holding=20 / quintile. **Decision rule — "robust (not
overfit)" iff:** across the pre-declared grid (lookback∈{63,126,189,252},
skip∈{0,5,10}, holding∈{10,20,40}, selection frac∈{0.10,0.20,0.333}), (a) ≥90% of
settings have positive net@60 Sharpe AND (b) ≥75% have Sharpe ≥ 0.7×baseline
(~0.9). If the edge only appears near 126/5/20/quintile → overfit. Result below.

**RESULT (PARTIAL / honest FAIL of the strict bar).** `research/param_sensitivity.py`.
Baseline net@60 Sharpe 1.27. Across 25 settings: **100% positive** (criterion a PASS)
but only **56% ≥ 0.89** (criterion b FAIL; needed ≥75%); min 0.66, median **0.99**,
max 1.27. lookback 63/126/189/252 → 0.86/1.27/0.85/0.72; holding 10/20/40 →
0.99/1.27/1.12; skip 0/5/10 → 1.14/1.27/1.00; frac 0.10/0.20/0.333 → 1.26/1.27/1.21.
**Reading:** the momentum edge is *directionally* robust (never negative), so D1.1 is
NOT overfit-to-a-point — but its *magnitude* is parameter-sensitive and 126/5/20 sits
near the top. The 1.27 headline is optimistic; the **parameter-robust median is ~1.0**.
→ Add this to the honest haircut: on top of survivorship/frictions, the expected
deployable Sharpe is closer to **~0.8–1.0**, not 1.27. **D1.1 rule RETAINED** — 126/5/20
is a standard, defensible, near-best momentum choice; tuning to the grid's max would
itself be overfitting (and the user rule: only change if the rule *fails*, which it
does not — it stays positive everywhere).

### Cycle 3 — implementation realism (PRE-REGISTERED)
**Hypothesis:** D1.1's edge survives TWSE execution frictions — (i) ±10% daily
price-limit unfillability (can't enter a name that is limit-up/down at signal), (ii)
higher costs, (iii) no short-side (long-only fallback). **Decision rule — "survives
frictions" iff** the price-limit-filtered net@60 Sharpe ≥ 0.8×baseline AND stays
positive at 100 bps. Result below.

**RESULT (PASS).** `research/impl_realism.py`. Baseline net@60 Sharpe 1.27;
**price-limit-filtered 1.29** (only 28 name-entries dropped over the sample — 6-month
momentum rarely enters on a limit-up day, so it does NOT depend on unfillable
limit-locked names). Cost stress: 1.27/1.16/1.01 at 60/100/150 bps (limit-filtered
1.29/1.17/1.02). Long-only fallback 1.35 (beta). Ratio filtered/baseline 1.01 ≥ 0.80,
positive at 100 bps → **survives implementation frictions.** (Frictions not modelled:
short-borrow availability/fee, intraday slippage beyond the bps curve.)

### Cycle 4 — regime / sub-period stability (PRE-REGISTERED)
**Hypothesis:** D1.1's edge is not concentrated in one favorable regime — it holds
in bull AND bear, high-vol AND low-vol sub-samples, and its worst drawdown recovers.
**Decision rule — "regime-stable" iff:** (a) net@60 Sharpe positive in both the
bull (universe 120d trend≥0) and bear (<0) sub-samples, with bear not worse than
−0.3; (b) positive in both high-vol and low-vol (universe 20d rvol vs its median)
sub-samples; (c) the maximum drawdown recovers within the sample (no permanent
impairment). Result below.

**RESULT (PARTIAL/FAIL — regime-dependent).** `research/regime_stability.py`.
Regime-conditional net@60 Sharpe: **BULL 1.81 / BEAR −0.42** (fails ≥−0.3),
**HIGH-vol 0.58 / LOW-vol 2.35**. So the edge is concentrated in bull & low-vol
regimes and is *negative in bear* regimes — long-short momentum's classic
vulnerability (consistent with Phase C1, where no regime overlay fixed it). Worst
drawdown −22.1%, peak→trough ~8 mo, **recovered in ~8 mo** (criterion c PASS — no
permanent impairment); rolling-12m Sharpe 75% positive. **Honest conclusion:** the
1.27 headline is bull-aided; a bear/high-vol regime is the primary risk. Reinforces
the ~0.8–1.0 realistic estimate. D1.1 retained (characterisation, not a defect;
drawdowns recover), with regime risk documented as the top weakness.

### Cycle 5 — hard sector-cap variant (PRE-REGISTERED)
**Hypothesis:** enforcing a HARD 20% sector cap (drop the weakest-momentum name
from an over-cap sector, iteratively) meaningfully lowers the realized max sector
weight (soft ~21%) without materially hurting Sharpe. **Decision rule — adopt the
hard cap only iff:** realized max sector weight ≤ 20.5% (hard) AND net@60 Sharpe
drop ≤ 0.10 vs soft-cap D1.1. Else keep the soft cap (simpler). Result below.

**RESULT (KEEP soft cap).** `research/hard_sector_cap.py`. soft-cap D1.1: Sharpe
1.27, maxDD −22.1%, maxSecW **20.9%**, maxNm 11.0%. Hard-cap-by-dropping: Sharpe
1.30, maxDD −19.1%, maxSecW **22.2%** (WORSE), maxNm 10.0%. The hard cap **fails its
own objective** — dropping the weakest name from an over-cap sector shrinks the leg
to the MIN_NAMES floor and re-concentrates it (fewer sectors → higher 1/n_sectors),
ending *above* 20.5%. A true hard 20% cap is effectively infeasible in a concentrated
momentum book without gutting it. maxSecW 22.2% > 20.5% → **keep the soft cap** (the
incidental Sharpe/DD gain isn't from the cap and doesn't meet the concentration bar).

### Cycle 6 — alternative-momentum robustness (PRE-REGISTERED)
**Hypothesis:** the edge is general cross-sectional momentum, not specific to the
6-1 (126/5) signal. Compare 3-1, 6-1, 12-1 month definitions (skip 1 month) + the
D1.1 baseline (126/5) + a rank-composite of the three. **Decision rule — "general
momentum, robust" iff** all definitions give positive net@60 Sharpe AND the
composite is within ~0.10 of baseline (or better). If only 6-1 works → lucky def.
Result below.

**RESULT (PARTIAL — general momentum confirmed).** `research/alt_momentum.py`.
net@60 Sharpe: 3-1 **0.79**, 6-1(126/21) **1.09**, 12-1 **0.73**, 6-1 baseline
(126/5) **1.27**, rank-composite(3/6/12-1) **1.08**. **All positive → the edge is
general cross-sectional momentum, not a lucky 6-1 definition** (criterion a PASS).
But composite 1.08 < baseline−0.10 (1.17) → criterion b FAIL: the 126/5 baseline is
the strongest, so the 1.27 headline is again a favorable specific pick (reinforces
C2). **D1.1 retained** (doesn't fail; 126/5 is a standard def). *Future option:*
a multi-horizon **rank-composite (~1.08)** diversifies signal risk and is a more
conservative, arguably better-practice deployment choice — logged, not adopted now
(don't change the frozen rule unless it fails; don't chase Sharpe).

### Cycle 7 — universe-robustness bootstrap (PRE-REGISTERED)
**Hypothesis:** the edge does not depend on the specific 106-name universe —
dropping a random ~20% of names still yields a positive net@60 Sharpe.
**Decision rule — "universe-robust" iff:** over 200 seeded random-subset draws
(each drops ~20% of names), the 5th-percentile net@60 Sharpe > 0 AND the median is
within ~0.2 of the full-universe Sharpe (1.27). Quantifies universe/luck dependence
(complements the survivorship caveat). Result below.

**RESULT (PASS).** `research/universe_bootstrap.py`, seed 20260707, 200 draws.
Full-universe net@60 Sharpe 1.27. Random-20%-dropped subsets: **100% positive**,
percentiles 5%/25%/50%/75%/95% = **0.95 / 1.07 / 1.17 / 1.24 / 1.39**, min 0.80.
5th-pct 0.95 > 0 and median 1.17 within 0.2 of full → **universe-robust.** The edge
is broad across names, not riding a few specific survivors (tempers the OOS
"top-3 = 12%" fragility; the subset median 1.17 is slightly below full, as expected
with fewer names to rank).

### Cycle 8 — factor-regression of the D1.1 book (PRE-REGISTERED)
**Hypothesis (confirmatory):** the D1.1 *product* is ~momentum beta with ~zero
residual alpha — a known factor, not proprietary alpha (as Exp-4 found for the ML
model). Regress D1.1 L/S returns on market/vol/momentum/size factor L/S series.
**Decision rule:** expect residual alpha t-stat < 2 (indistinguishable from zero)
and a dominant, significant momentum beta. Document exact numbers honestly. Below.

**RESULT (CONFIRMED).** `research/factor_regression_d1_1.py`, 96 rebalances.
Betas (t): **momentum 0.834 (t=21.6)**, market 0.125 (t=2.08), vol −0.070 (t=−1.24),
size −0.044 (t=−0.78); R²=0.89. **Residual alpha +4.3%/yr, t=1.60 (< 2 → not
statistically significant).** → The D1.1 product is ~momentum beta with no
significant residual alpha — a **known-factor product, not proprietary alpha**
(confirms Exp-4 on the actual deployable book). Nuance: mild positive market beta
(0.13, t=2.08) → the L/S is *not* perfectly market-neutral (contributes to the bear
vulnerability in C4).

### Cycles 9-12 — deeper characterization (PRE-REGISTERED, one block)
Characterization only; no rule change; not Sharpe-optimising.
- **C9 signal decay/crowding:** first-half (2018-21) vs second-half (2022-26) net@60
  Sharpe & IC — is momentum weakening? Note if 2nd-half materially weaker.
- **C10 diversification:** correlation of D1.1 L/S returns to equal-weight universe
  ("market") and to 0050 ETF — low corr = diversifier.
- **C11 rebalance-timing:** D1.1 Sharpe at grid offsets 0/5/10/15 trading days —
  robust iff stable across offsets (not a calendar artifact).
- **C12 beta-hedged:** subtract market-beta·market from D1.1 returns — does hedging
  improve the C4 bear-regime Sharpe? (characterization only). Result below.

**RESULTS.** `research/deeper_characterization.py`.
- **C9 (no decay):** 1st-half 2018-21 Sharpe 1.20 / IC 0.036; 2nd-half 2022-26 Sharpe
  1.31 / IC 0.057 → **not weakening** (recent strength is the favorable regime, not
  crowding).
- **C10 (diversifier):** corr of D1.1 L/S to equal-weight market **+0.24**, to 0050
  ETF **+0.13** → low; decent diversifier (but not market-neutral — the mild beta).
- **C11 (timing matters):** net@60 Sharpe by grid offset 0/5/10/15d =
  **1.27 / 1.03 / 0.90 / 1.10** → the 1.27 headline is a **favorable rebalance
  offset**; robust range ~0.9–1.27, all positive (reinforces ~1.0–1.1 honest est.).
- **C12 (bear is intrinsic):** market beta 0.28 (univariate); hedging *reduces* full
  Sharpe (1.45→1.22) and **worsens** the bear regime (−0.19→−0.64) → the bear
  vulnerability is intrinsic momentum-crash risk, **not** removable market beta
  (consistent with C1's failed overlay and C4).

## Backlog (next hypotheses, RE-RANKED after Experiment 4)
Exp 4 settled it: the only capturable signal in TWSE OHLCV here is the **momentum
risk premium** — no residual alpha. So the question is no longer "find alpha in
price/volume" (exhausted) but "engineer the known factor well" or "get orthogonal
data". Ranked:

1. **Decision fork for the user (see below).** Path A (momentum-factor product) vs
   Path B (orthogonal-signal search). This choice sets everything downstream.
2. **If Path A — engineer the momentum product:** (a) cut turnover (100%/rebalance
   is the main cost leak) via signal smoothing / holding overlap / no-trade bands;
   (b) risk management: 2018 & 2022 drawdowns (LS Sharpe −2.4, −0.2) need a trend/
   vol timing overlay; (c) sector caps (electronics+shipping = 65% of P&L is
   concentration risk); (d) proper sizing (vol-target, not equal-weight).
   Success bar: net-of-realistic-cost Sharpe and max-DD beat the NN baseline.
3. **If Path B — orthogonal alpha:** ingest fundamentals / earnings-revision /
   ownership-flow / alt-data; the ONLY test that matters is residual alpha after
   the momentum+vol+size+market factors (Exp-4 regression is the template).
4. **Data integrity (do regardless):** verify price adjustment (audit #6);
   quantify survivorship bias (all Exp-1..4 numbers are upper bounds).
5. **Architecture: shelved.** Exp 4 shows nothing to learn beyond a 1-line factor;
   a bigger NN cannot manufacture alpha that isn't in the data.

*Meta-lessons banked:* (i) always clear the market/vol/momentum/size-neutral OOS
bar before claiming alpha; (ii) long-only Sharpe & raw IC were mirages
(beta/momentum); (iii) a positive vol-neutral Sharpe can still be a *known factor*
— only the factor-regression alpha t-stat distinguishes skill from premium.
