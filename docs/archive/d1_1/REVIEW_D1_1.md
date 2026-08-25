# Release Review — D1.1 momentum prototype

Final pre-commit review of the D1.1 research milestone. Verdict: **no blocking
issues** — safe to commit as a *research milestone* (not a production trading
system). Numbers are full-sample and survivorship-biased (upper bound); caveats
below.

## Checklist

### 1. git diff summary
- Tracked, modified: `dataset.py`, `train.py`, `test_dataset.py` — these belong to
  the **earlier leakage-fix workstream** (chronological split), *not* D1.1. Excluded
  from this commit.
- New (untracked), part of D1.1 milestone: `research/*.py` (11 modules),
  `research/data_cache/*.csv` (108 cached OHLCV files), and docs
  (`RESEARCH_LOG.md`, `ROADMAP_A_D_C.md`, `PATH_ANALYSIS.md`, `ASSUMPTIONS_AUDIT.md`,
  `REVIEW_D1_1.md`, `STRATEGY_CARD_D1_1.md`).
- New but unrelated: `REVIEW_4e7467e.md` (reviews the leakage fix) — excluded.
- `__pycache__/` — build artifacts, excluded.

### 2. Files changed (to be committed)
Research code: `research/{data,features,targets,evaluation,momentum,portfolio_d1,
d1_1_pername_cap,prototype,exp2_vol_confound,exp3_universe,exp4_robustness,
regime_c1,analyze_target,test_framework}.py`; `research/data_cache/*.csv` (for exact
reproducibility). Docs: the six `.md` files listed above. **Not** committed:
`train.py`, `dataset.py`, `test_dataset.py`, `inference.py`, `REVIEW_4e7467e.md`,
`__pycache__/`.

### 3. train.py / inference.py untouched?
- **inference.py: UNTOUCHED** (clean, `git diff` empty). ✓
- **train.py: not touched by the momentum arc.** It carries uncommitted edits from
  the *earlier* leakage fix; those are **excluded** from this commit, so this commit
  does not modify or include it. ✓ (Honest note: the working tree still holds those
  prior edits, which are a separate, unrelated change.)

### 4. D1 baseline still reproducible?
**Yes.** `portfolio_d1.py` was not modified during D1.1. Re-run gives the exact D1
numbers: `invvol + cap 20%` → Sharpe 1.21, annRet 25.4%, maxDD −21.9%, Calmar 1.16,
turnover 5.6×, maxSec 20.3%, P&L share 37.0%. D1.1's "uncapped" row also reuses the
frozen `_apply_sector_caps` path and reproduces the same. ✓

### 5. D1.1 prototype output matches the validation artifact?
**Yes — by construction and by value.** `prototype.backtest_summary()` calls
`d1_1_pername_cap.backtest(..., name_cap=0.10)` (same code path). Both report
Sharpe@60 1.27, maxDD −22.1%, Calmar 1.25, max name 11%, max sector ~21%, P&L share
35%, 96 rebalances. ✓

### 6. Is the 10% per-name cap causal (no look-ahead)?
**Yes.** `_leg_weights` builds weights from `sub["vol"]` (trailing 60-day realized
vol, `rolling(60).std()` of past returns) and momentum selection (`close[t-5]/
close[t-131]`, past only). `_cap_weights` is a pure weight transform on the current
cross-section. Forward return (`fwd_ret`) is used **only** as the realised P&L label
in the backtest, never in signal, sizing, or capping. ✓

### 7. Is the 20% sector cap hard, or can it drift to ~21%?
**Soft cap — it can drift.** `_cap_weights` (and D1's `_apply_sector_caps`) set
`eff_sec = max(0.20, 1/n_sectors_in_leg)` for feasibility. When a leg spans ≤5
sectors the effective cap loosens (4 sectors → 25%); a final renormalisation can
also nudge sector sums slightly. Realised average max-sector weight is 20.3% (D1)
and 20.9% (D1.1), and can exceed 20% when the leg is sector-sparse. **Not a bug**
(intended feasibility behaviour), but "20% sector cap" is approximate, not a hard
constraint. Documented in the strategy card. A hard cap would require dropping the
weakest-momentum name from an over-represented sector (changes selection) — deferred.

### 8. Do all tests pass?
**Yes.** `test_dataset.py` 7/7 (leakage-free split); `research/test_framework.py`
8/8 (IC sign, backtest causality, long-short neutrality, factor neutralisation,
walk-forward learning, triple-barrier/high-low, forward-return censoring). ✓

### 9. Hidden look-ahead / survivorship / implementation issues?
- **Look-ahead: none found.** Signal, vol sizing, and caps are strictly past-only;
  forward returns are labels only; rebalances are non-overlapping (`dates[::20]`).
- **Survivorship: present and DOCUMENTED (not hidden).** twstock exposes only
  currently-listed names; two requested codes were `StockIDNotFound` (delisted).
  All headline numbers are therefore **upper bounds**. Stated everywhere.
- **In-sample parameter selection (the key honest caveat):** the momentum rule
  parameters are standard/pre-declared, but the **10% cap was chosen after seeing all
  six caps on the full 2018–2026 sample** — no held-out OOS. Mitigated by choosing on
  concentration (not Sharpe-max) from a pre-declared set, but it is **not a clean
  walk-forward validation.** → next experiment.
- **Shorting assumption:** the headline L/S assumes shortable TWSE names (hard in
  practice); a long-only variant is emitted (carries market beta). Documented.
- **Turnover convention:** one-way L1 (A1's earlier name-based figure double-counted
  ~2×); consistent across D1/D1.1.
- **Data staleness:** the live book uses each name's latest row (all ~2026-07-06);
  negligible.

## Blocking issues
**None.** All are documented characteristics/caveats appropriate to a research
milestone, not defects.

## Recommended next experiment
A **walk-forward / point-in-time OOS validation** of D1.1 (rolling train/test
windows; quantify the survivorship haircut) to confirm the full-sample numbers hold out of
sample before any capital or production use. Then Path B (orthogonal data) for real
alpha.
