# Operation Manual — D1.1 Momentum Prototype

Operator's guide for running and interpreting the D1.1 research prototype. This
manual does **not** change the strategy; it documents the pipeline exactly as
built in `research/`. Production `train.py` / `inference.py` are unrelated to this
prototype and are **not** used here.

> **Status:** research prototype, **not capital-ready**. See §7.

---

## 1. What this pipeline is

- **It is NOT a machine-learning model.** There is no training, no weights, no
  neural network, no fitted parameters. (The repo's `train.py`/`inference.py`
  LSTM-Transformer is a *separate*, older system that this prototype deliberately
  does not touch or use.)
- **It IS a deterministic, rule-based cross-sectional momentum-factor prototype.**
  Given the same input data it always produces the same output. Every number is a
  closed-form calculation (ranks, inverse-volatility weights, capping rules).
- **It harvests a known risk premium (cross-sectional momentum), not proprietary
  alpha** (factor regression: momentum beta 0.83, residual alpha t=1.60 — not
  significant). Treat it as a factor product, not an edge.

---

## 2. The exact D1.1 rule

| element | definition |
|---|---|
| **Signal** | 6-month momentum, skip last week: `mom_t = close[t-5] / close[t-131] - 1` (LOOKBACK=126, SKIP=5 trading days) |
| **Universe** | 106 liquid TWSE names across 15 sectors = all `SECTOR_MAP` entries **except ETFs** (`0050`, `0056` are excluded from selection; they exist in the cache only as market proxies) |
| **Sector map** | `research/data.py :: SECTOR_MAP` — a hard-coded `{stock_id: sector}` dict (semis, electronics, financials, materials, shipping, transport, consumer, retail, auto, industrial, biotech, telecom, construction, optics, panels) |
| **Selection** | rank the cross-section by `mom` each rebalance; **long the top quintile, short the bottom quintile**. Quintile size `k = max(3, round(0.20 × n_names))` ≈ **21 per leg** |
| **Sizing** | **inverse-volatility**: weight ∝ `1 / (trailing 60-day std of daily log returns)`, normalized within each leg. Past-only |
| **Sector cap** | **20% per sector per leg** — SOFT (floored at `1/n_sectors_in_leg`; realized ~20–21%). See §7 |
| **Per-name cap** | **10% per name** (NAME_CAP=0.10), enforced jointly with the sector cap by iterative clip-and-redistribute |
| **Rebalance** | every **20 trading days** (HOLDING=20); non-overlapping grid |
| **Cost** | headline **net@60 bps** round-trip on one-way L1 turnover; reported as a sweep {0, 60, 80} bps. Turnover ≈ 5–6×/yr; breakeven ≈ 390 bps |

Deployable variants emitted by the tool: **long_short** (market-neutral-ish, the
validated book) and **long_only** (top quintile only; carries market beta).

---

## 3. Required input files

- **Data cache:** `research/data_cache/<stock_id>.csv`, one file per name, columns
  `date, open, high, low, close, volume`. This directory is **git-ignored** (not in
  the repo) — you must generate it once.
- **Regenerate the full cache** (fetches every name in `SECTOR_MAP`, ~1–2 min each
  via `twstock`, resumable — already-cached names are skipped):
  ```
  .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'research'); \
    from data import build_cache, SECTOR_MAP; build_cache(list(SECTOR_MAP))"
  ```
  (`.venv/Scripts/python.exe research/data.py` alone only fetches the 20-name
  `DEFAULT_UNIVERSE` — use the command above for the full 106-name universe.)
- **Scripts that depend on the cache:** every `research/*.py` analysis/prototype
  script (`prototype.py`, `portfolio_d1.py`, `d1_1_pername_cap.py`,
  `walkforward_d1_1.py`, `walkforward_rolling.py`, `param_sensitivity.py`,
  `impl_realism.py`, `regime_stability.py`, `hard_sector_cap.py`, `alt_momentum.py`,
  `universe_bootstrap.py`, `factor_regression_d1_1.py`,
  `deeper_characterization.py`). `research/test_framework.py` uses **synthetic
  data only** and needs no cache.
- **Survivorship caveat:** `twstock` lists only *currently-listed* names; delisted
  names are absent. The published results use the **2026-07-06 snapshot**. A later
  refetch pulls the then-current survivor universe and **will differ**. All figures
  are an **upper bound** (see §7).

---

## 4. Main commands

All commands use the repo's venv Python. On the local console prepend
`PYTHONIOENCODING=utf-8` (Git Bash) so unicode prints cleanly. Run from repo root.

| purpose | command |
|---|---|
| **Run tests** | `.venv/Scripts/python.exe research/test_framework.py` (expect **8/8**) |
| **Generate today's target book** + validation card | `.venv/Scripts/python.exe research/prototype.py` |
| **Main backtest** (D1 construction variants) | `.venv/Scripts/python.exe research/portfolio_d1.py` |
| **D1.1 per-name cap validation** | `.venv/Scripts/python.exe research/d1_1_pername_cap.py` |
| **OOS validation** (held-out + adaptive WF) | `.venv/Scripts/python.exe research/walkforward_d1_1.py` |
| **OOS validation** (multi-fold + bootstrap CI) | `.venv/Scripts/python.exe research/walkforward_rolling.py` |
| **Robustness — parameters** | `.venv/Scripts/python.exe research/param_sensitivity.py` |
| **Robustness — implementation/costs** | `.venv/Scripts/python.exe research/impl_realism.py` |
| **Robustness — regime** | `.venv/Scripts/python.exe research/regime_stability.py` |
| **Robustness — universe bootstrap** | `.venv/Scripts/python.exe research/universe_bootstrap.py` |
| **Factor attribution of the book** | `.venv/Scripts/python.exe research/factor_regression_d1_1.py` |
| **Programmatic use** | `from prototype import generate_book, backtest_summary` (after `sys.path.insert(0,'research')`) |

`walkforward_rolling.py` and `universe_bootstrap.py` are the heaviest (~1–4 min).

---

## 5. Expected outputs

### Target book (`generate_book()` / `prototype.py`)
Columns: **`side, stock, sector, weight, mom, asof`**.
- **`side`** — `LONG` or `SHORT`.
- **`weight`** — signed portfolio weight. **LONG weights are positive, SHORT
  negative.** Each leg's absolute weights sum to **1.0** (100% gross per leg). No
  single name exceeds 10%; no single sector exceeds ~20% (soft).
- **`mom`** — the momentum signal value (e.g. `+0.75` = +75% 6-month return).
- **`asof`** — the date of the latest price used for that name's signal.
- **Reading sector exposure:** the tool prints `sector wts:` per leg — the summed
  absolute weight per sector. Confirm no sector materially exceeds 20%.

### Validation card (`backtest_summary()` / `prototype.py`)
- **net Sharpe** at 0/60/80 bps — risk-adjusted return; the **60 bps** figure is the
  headline. Historical ≈ **1.27 @60 bps** (survivorship-inflated; expect ~0.8–1.0
  realistically — §7).
- **ann. return / max drawdown / Calmar** — return, worst peak-to-trough, and their
  ratio.
- **turnover** — annualized fraction of book traded (~5–6×).
- **concentration** — max name %, max sector %, max single-sector P&L share.

### Validation-metric interpretation (other scripts)
- **bootstrap CI** (`walkforward_rolling.py`): the 90% CI on Sharpe **[0.47, 1.99]**
  — the lower bound > 0 means "statistically positive," not "1.27 is reliable."
- **regime split** (`regime_stability.py`): BULL 1.81 vs **BEAR −0.42** — the book
  loses in bear/high-vol regimes (its key risk).
- **factor alpha t-stat** (`factor_regression_d1_1.py`): **t=1.60 (<2)** → no
  significant alpha; it's momentum beta.
- A higher backtest Sharpe from tweaking a parameter is **not** an improvement —
  do not chase it (see §6).

---

## 6. Manual operating workflow

This is a **manual, human-in-the-loop research tool**, not an automated trader.

1. **Update data:** refresh the cache about **once per rebalance cycle (~monthly)**,
   before generating a new book: rerun the §3 regenerate command (resumable). Note
   that refetching changes the survivor universe slightly.
2. **Rebalance cadence:** the strategy is designed for a **20-trading-day
   (~monthly)** rebalance. Do not rebalance more often (the edge is slow-decaying;
   over-trading only adds cost).
3. **Generate the target book:** `research/prototype.py` → read the LONG/SHORT legs
   and weights.
4. **Compare current holdings vs target:** for each name, `trade = target_weight −
   current_weight`; trade the differences. Names not in the book → exit. Respect the
   per-name (10%) and sector (~20%) caps already embedded in the target weights.
5. **What NOT to do automatically:**
   - Do **not** wire this to live order execution — it is not capital-ready.
   - Do **not** auto-rebalance on a timer without human review.
   - Do **not** re-tune the momentum/cap parameters to raise backtest Sharpe.
   - Do **not** treat the long-only book's high return as skill (it is market beta).
   - Do **not** deploy into a confirmed bear/high-vol regime without accepting the
     documented drawdown risk.

---

## 7. Safety and caveats

- **Not capital-ready.** A research prototype for study, not a trading system. No
  live execution, borrow, or risk-limit integration.
- **Not proprietary alpha.** It is the cross-sectional **momentum premium** (a known
  factor). Factor regression: residual alpha t=1.60 (not significant).
- **Shorting assumptions.** The headline long-short book assumes TWSE names are
  shortable at the signal price with no borrow fee — often unrealistic. The
  long-only variant avoids shorting but carries **market beta**.
- **Survivorship bias.** Current-listed universe only → all figures are an **upper
  bound**; results depend on the 2026-07-06 snapshot.
- **Sector cap is SOFT.** Floored at `1/n_sectors_in_leg`; realized max sector
  weight ~20–21%, occasionally higher for sector-sparse legs. It is not a hard 20%.
- **Realistic expectation.** The headline **~1.27 net@60 bps Sharpe is optimistic**
  (favorable parameters, rebalance offset, and survivorship). The honest
  **deployable expectation is ~0.8–1.0**, with **bear/high-vol regimes the primary,
  intrinsic risk** (not fixable by regime overlay or beta-hedge). Max drawdown
  ~−22% (recovers ~8 months historically).

---

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| **Missing data cache** (`Universe cache too small — run research/data.py first` / `SystemExit`) | Cache not built. Run the §3 regenerate command. |
| **Failed data fetch** (`ReadTimeout`, `StockIDNotFoundError`) | Transient network / VPN issue → just rerun (resumable, skips cached). `StockIDNotFound` = a delisted/invalid code; it is skipped and the universe is slightly smaller — this is expected and safe. |
| **No output / seems hung** | Heavy scripts (`walkforward_rolling.py`, `universe_bootstrap.py`) take 1–4 min and buffer stdout to a file when redirected — run with `-u` or wait. Use CPU-time to confirm progress, not stdout. |
| **Unicode/`cp950` print error** | Prepend `PYTHONIOENCODING=utf-8` to the command. |
| **Tests fail** (`research/test_framework.py` not 8/8) | The measurement engine is broken — do **not** trust any backtest until fixed. Re-check you are on the research branch with an unmodified `research/` tree; `git status` should show no unexpected edits. |
| **Target book has too few names** (`< 2k` per rebalance / empty legs) | Cache too small or too many names failed to fetch. Confirm `ls research/data_cache/*.csv \| wc -l` ≈ 106–108 non-ETF names; refetch missing ones. |
| **Numbers differ from the report** | Expected if the cache was refetched on a later date (survivorship snapshot changed). Use the frozen 2026-07-06 cache to reproduce exactly. |

---

## 9. Quickstart (fresh clone → today's book)

```bash
# 0. From the repo root, on the research branch.
git checkout research/d1-1-momentum-prototype

# 1. Build the data cache once (~1-2 min per name; resumable; needs network).
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'research'); \
  from data import build_cache, SECTOR_MAP; build_cache(list(SECTOR_MAP))"

# 2. Verify the measurement engine.
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/test_framework.py   # expect 8/8

# 3. Generate today's target book + validation card.
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/prototype.py
```
Then read the LONG/SHORT legs, sizes, and sector exposure per §5, and follow the
manual workflow in §6. Nothing is committed, pushed, or traded automatically.

---
*This manual documents the prototype as built; it does not alter the D1.1 rule.
Production train.py / inference.py are untouched and unrelated.*
