# AI-Quant — TWSE Quant Research

A Taiwan Stock Exchange (TWSE) quantitative research repository containing **two
distinct things**:

1. **The original ML system** (`model.py`, `train.py`, `inference.py`,
   `dataset.py`) — an LSTM-Transformer that predicts a +12%/−6% triple-barrier
   label. It is the *original* codebase, **not** the currently validated approach.
2. **The D1.1 research prototype** (`research/`) — a **deterministic, rule-based
   cross-sectional momentum-factor** pipeline that research found to be simpler,
   more robust, and better-performing than the ML system, at a fraction of the
   complexity.

> ⚠️ **Honest framing:** D1.1 is **not** a proprietary alpha model. It is a
> deterministic TWSE cross-sectional **momentum-factor** prototype — it harvests a
> *known risk premium*, not a proprietary edge. It is a **research prototype, not a
> capital-ready trading system.** See [Caveats](#4-important-caveats).

---

## 1. Project overview

- **What AI-Quant originally was:** a GPU-trained `LSTM_CondTransformer` ensemble
  (four horizon models + a market-regime gate) that ranks TWSE tech stocks by their
  probability of hitting +12% before −6% within 20 trading days. Still present under
  the repo root (`model.py`, `train.py`, `inference.py`, `dataset.py`), but research
  (below) showed this target mostly captured **volatility/beta**, not skill, and the
  ML did **not** beat a simple momentum rule.
- **What the current `research/d1-1-momentum-prototype` branch contains:** a full
  research program (Experiments 1–4 + an A→D→C engineering arc + OOS validation + a
  5-hour robustness sprint) culminating in the **D1.1** momentum prototype, plus the
  scripts that produced every number, and honest documentation of what does and
  doesn't work.
- **Production vs research separation:** the root ML files (`train.py`,
  `inference.py`, `model.py`, `dataset.py`) are treated as **production/legacy and
  are intentionally left untouched** by the research work. Everything in `research/`
  is self-contained and does not import or modify them.

---

## 2. Current validated prototype: D1.1

A single deterministic rule (no ML, no fitted parameters):

| element | definition |
|---|---|
| **Signal** | `mom_t = close[t-5] / close[t-131] - 1` (6-month momentum, skip last 5 days) |
| **Selection** | rank the cross-section; **long the top quintile, short the bottom quintile** (~21 names/leg) |
| **Sizing** | **inverse-volatility** (∝ 1 / trailing 60-day return vol), past-only |
| **Sector cap** | **20% per sector** — *soft* (floored at 1/n_sectors; realized ~20–21%) |
| **Per-name cap** | **10% per name** |
| **Hold** | **20 trading days**, then rebalance |
| **Universe** | 106 liquid TWSE names across 15 sectors (ETFs excluded from selection) |
| **Cost** | headline **net@60 bps**; turnover ~5–6×/yr; breakeven ~390 bps |

- **Long-short** is the research book (market-neutral-ish; the honest signal).
- **Long-only** (top quintile) is the practical variant where shorting is hard —
  but it **carries market beta**, so in a bull sample it looks strong on return
  without adding risk-adjusted value over just holding the universe.

Historical (net@60 bps, **survivorship-biased upper bound**): Sharpe **~1.27**,
ann. return 27.6%, max drawdown −22.1%, Calmar 1.25. **Read the realistic
expectation as ~0.8–1.0**, not 1.27 (see findings/caveats).

---

## 3. Key research findings

Full detail in `docs/d1_1/RESEARCH_LOG.md`. In short:

1. **The original ML/barrier target mostly captured volatility/beta, not alpha.**
   In the semis-only universe, the triple-barrier signal's edge was ~92% a
   volatility/beta bet (Experiment 2); factor-neutral alpha was ~zero.
2. **A multi-sector universe revealed a genuine momentum premium.** Broadening
   beyond semiconductors turned a zero/negative vol-neutral edge into a positive one
   (Experiment 3).
3. **ML did not beat simple momentum.** A one-line momentum rule (net@60 Sharpe
   ~1.22) outperformed the LSTM-Transformer-lineage model (~0.62) at lower turnover
   (Phase A1) — and factor regression shows the product is momentum beta (β=0.83,
   t=21.6) with residual alpha **t=1.60 (not significant)**.
4. **D1.1 survived OOS + robustness + cost + implementation checks.** Held-out OOS
   (2022–26, crash included) Sharpe ~1.51; block-bootstrap 90% CI **[0.47, 1.99]**;
   100% of random-universe subsets positive; survives TWSE price-limit fills and
   150 bps costs; positive across all reasonable parameter choices and momentum
   definitions.
5. **Realistic deployable Sharpe is ~0.8–1.0, not the 1.27 headline.** The headline
   is a favorable parameter/rebalance-offset pick; the representative value across
   settings is ~1.0, and **bear/high-vol regimes are the one intrinsic
   vulnerability** (not fixable by regime overlay or beta-hedge).

---

## 4. Important caveats

- **Research prototype, not capital-ready.** No live execution, borrow, or
  risk-limit integration.
- **Known-factor momentum product, not proprietary alpha** (residual alpha not
  statistically significant).
- **Survivorship bias.** The universe is current-listed names only (via `twstock`);
  delisted names are absent → **all figures are an upper bound**. Results depend on
  the **2026-07-06 data snapshot**; a later refetch will differ.
- **TWSE shorting assumptions.** The long-short book assumes shortability at the
  signal price with no borrow fee — often unrealistic. Long-only avoids this but
  carries market beta.
- **Soft sector cap** may drift to ~20–21% (occasionally higher for sector-sparse
  legs); it is not a hard 20%.
- **No live-trading integration.** This is a manual, human-in-the-loop research tool.
- **No guarantee of future returns.** Past (survivorship-biased) backtest ≠ future
  live performance.

---

## 5. Repository structure

**Research prototype & validation (`research/`)**

| file | purpose |
|---|---|
| `research/prototype.py` | **The D1.1 product** — `generate_book()` (today's long/short & long-only book) + `backtest_summary()` (validation card). Same code path as the research backtest. |
| `research/momentum.py` | Momentum signal + the A1 raw-momentum baseline backtest. |
| `research/portfolio_d1.py` | D1 construction (inverse-vol sizing + 20% sector cap); the *frozen* D1 baseline. |
| `research/d1_1_pername_cap.py` | D1.1 = D1 + 10% per-name cap; the validation that adopted it. |
| `research/walkforward_d1_1.py` | OOS validation — held-out (2018–21 → 2022–26) + adaptive walk-forward + survivorship-fragility proxies. |
| `research/walkforward_rolling.py` | Stricter OOS — multi-fold + block-bootstrap confidence interval on Sharpe. |
| `research/param_sensitivity.py` | Robustness across lookback/skip/holding/quintile. |
| `research/impl_realism.py` | TWSE price-limit fills + cost stress + long-only fallback. |
| `research/test_framework.py` | **Correctness tests for the measurement engine (8/8).** Synthetic data; no cache needed. |
| `research/data.py`, `features.py`, `targets.py`, `evaluation.py` | Data cache, features, candidate targets, and the shared backtest/IC/neutralization engine. |
| `research/exp2_vol_confound.py`, `exp3_universe.py`, `exp4_robustness.py`, `regime_c1.py`, `regime_stability.py`, `hard_sector_cap.py`, `alt_momentum.py`, `universe_bootstrap.py`, `factor_regression_d1_1.py`, `deeper_characterization.py`, `analyze_target.py` | The experiment/robustness scripts behind the findings. |

**Documentation**

| file | purpose |
|---|---|
| `docs/d1_1/OPERATION_MANUAL_D1_1.md` | How to run and interpret the prototype (operator's guide). |
| `docs/d1_1/STRATEGY_CARD_D1_1.md` | One-page strategy card + robustness table. |
| `docs/d1_1/SPRINT_5H_REPORT.md` | Final report of the 5-hour robustness sprint (12 cycles). |
| `docs/d1_1/WALKFORWARD_D1_1.md`, `docs/d1_1/REVIEW_D1_1.md` | OOS validation write-up; pre-commit release review. |
| `docs/d1_1/RESEARCH_LOG.md` | The full running research log (all experiments + cycles). |
| `docs/d1_1/ROADMAP_A_D_C.md`, `docs/d1_1/PATH_ANALYSIS.md`, `docs/d1_1/ASSUMPTIONS_AUDIT.md` | The A→D→C plan, path comparison, and assumptions audit. |

**Original ML system (legacy, untouched):** `model.py`, `train.py`, `inference.py`,
`dataset.py`, `generate_stocks_json.py` — see [§11](#11-the-original-ml-system-legacy).

**Data cache:** `research/data_cache/*.csv` is **git-ignored** (not tracked). It is
a regenerable, ~2026-07-06 survivorship snapshot; see Quickstart to build it.

---

## 6. Quickstart

From the repo root, on the research branch. (Prepend `PYTHONIOENCODING=utf-8` on a
non-UTF console so unicode prints cleanly.)

```bash
# activate environment (Windows venv is already present in .venv/)
git checkout research/d1-1-momentum-prototype

# check the data cache (expect ~106-108 CSVs)
ls research/data_cache/*.csv | wc -l

# build/regenerate the full data cache (only if missing; ~1-2 min/name, resumable, needs network)
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'research'); from data import build_cache, SECTOR_MAP; build_cache(list(SECTOR_MAP))"

# run tests (expect 8/8)
.venv/Scripts/python.exe research/test_framework.py

# generate today's target book + validation card
.venv/Scripts/python.exe research/prototype.py

# D1.1 construction/cap validation
.venv/Scripts/python.exe research/portfolio_d1.py
.venv/Scripts/python.exe research/d1_1_pername_cap.py

# OOS validation (held-out + adaptive walk-forward)
.venv/Scripts/python.exe research/walkforward_d1_1.py
```

---

## 7. How to operate the D1.1 prototype

Manual, human-in-the-loop — **not** an automated trader.

- **Data refresh:** rebuild the cache ~once per rebalance cycle (~monthly) before
  generating a new book (Quickstart command; resumable).
- **Target book generation:** `research/prototype.py` prints the LONG/SHORT legs
  (and a long-only variant). Programmatically: `from prototype import generate_book`.
- **Rebalance interpretation:** designed for a ~20-trading-day hold. For each name,
  `trade = target_weight − current_weight`; exit names not in the book. Caps are
  already embedded in the target weights.
- **Columns to inspect:** `side` (LONG/SHORT), `weight` (signed; LONG +, SHORT −;
  each leg sums to 100%), `mom` (signal), `sector`, `asof`. Confirm max name ≤10%
  and max sector ~20%.
- **What NOT to automate:** don't wire to live execution; don't auto-rebalance on a
  timer without review; don't re-tune parameters to raise backtest Sharpe; don't
  treat the long-only book's return as skill (it's beta).

Full detail: `docs/d1_1/OPERATION_MANUAL_D1_1.md`.

---

## 8. Testing and validation

- **Correctness:** `research/test_framework.py` (8/8) verifies the measurement engine
  — IC sign recovery, backtest causality (no look-ahead), long-short neutrality,
  factor neutralization, walk-forward learning. **If this is not 8/8, do not trust
  any backtest.**
- **Expected outputs:** `prototype.py` → strategy card (net Sharpe ~1.27 @60 bps) +
  a 21-name LONG leg and 21-name SHORT leg summing to 100% each, max name ≤10%, max
  sector ~20%. `portfolio_d1.py` → `invvol + cap 20% … 1.21`. `d1_1_pername_cap.py`
  → `name cap 10.0% … 1.27`, `uncapped (=D1)` reproduces D1. `walkforward_d1_1.py` →
  `D1.1 fixed 10% [OOS] … ~1.51`.
- A higher backtest Sharpe from tweaking a parameter is **not** an improvement.

---

## 9. Development guardrails

- **No production changes without review** — `model.py`, `train.py`, `inference.py`,
  `dataset.py` are legacy and must not be modified as part of research work.
- **`research/` is separate from production** — self-contained; does not import the
  ML pipeline.
- **No pushing or merging without human approval.** Work stays on the research
  branch; commits are local.
- **Avoid Sharpe-chasing and overfitting** — every experiment pre-registers its
  hypothesis and decision rule; the frozen D1.1 rule is not changed to chase
  backtest numbers.

---

## 10. Recommended next steps

1. **Human review of the research branch**, then decide merge policy.
2. **Clean PR / squash-merge** — the branch's history briefly added the 108-file
   data cache (now git-ignored); a **squash merge keeps those blobs off `main`**.
3. **Keep `research/data_cache/` out of `main`** (it is already git-ignored;
   regenerate via `research/data.py`).
4. **For genuine alpha beyond the momentum premium → "Path B": orthogonal data**
   (fundamentals, earnings revisions, ownership/flows, alt-data). Price/volume signal
   is exhausted here, and a bigger neural net will not help (Experiment 4). This
   requires data `twstock` does not provide. See `docs/d1_1/PATH_ANALYSIS.md`.

---

## 11. The original ML system (legacy)

Still in the repo root, unchanged: `model.py` (`LSTM_CondTransformer`), `train.py`,
`inference.py`, `dataset.py`, `generate_stocks_json.py`. It trains a four-horizon
ensemble on a +12%/−6%/20-day triple-barrier label with a market-regime gate.

**Status (important, corrected):** this is the **original** system, **not** the
currently validated approach. Research (Experiments 1–4) found its target largely
encoded **volatility/beta rather than skill**, and a simple momentum rule beat it
out-of-sample at lower complexity and turnover. It is retained for reference and is
**not** recommended as the basis for decisions. Note also that it requires `torch`
(not listed in `requirements.txt`) and GPU, and reads live data via `twstock`.

---

## License

See `LICENSE`.
