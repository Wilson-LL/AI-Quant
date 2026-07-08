# 5-Hour Autonomous Research Sprint — Final Report

**Window:** 2026-07-07 15:03 → 20:03 (local). **Active research concluded 18:01
(~3h)** — see §10 for why (genuine queue exhausted; continuing would be make-work,
which the brief forbade). **Branch:** `research/d1-1-momentum-prototype`. **Baseline
at start:** `c9195b8`. No merges, no pushes, no production files touched.

## 1. What was attempted
Stress-test and characterise the D1.1 momentum prototype under stricter OOS,
parameter, universe, cost, regime, and factor assumptions — to establish an
*honest* deployable expectation and find any hidden fragility. 12 pre-registered
research cycles (each: hypothesis + decision rule fixed *before* running).

## 2. What was implemented (all in research/, no ML, deterministic/seeded)
| module | cycle |
|---|---|
| walkforward_rolling.py | C1 multi-fold walk-forward + block-bootstrap CI |
| param_sensitivity.py | C2 lookback/skip/holding/frac grid |
| impl_realism.py | C3 TWSE price-limit fills + cost stress + long-only |
| regime_stability.py | C4 bull/bear, high/low-vol, drawdown recovery |
| hard_sector_cap.py | C5 hard vs soft sector cap |
| alt_momentum.py | C6 3-1/6-1/12-1 + rank-composite |
| universe_bootstrap.py | C7 200 seeded random-subset draws |
| factor_regression_d1_1.py | C8 market/vol/mom/size OLS of the book |
| deeper_characterization.py | C9-12 decay, diversification, timing, beta-hedge |

## 3. Tests run
`research/test_framework.py` — **8/8 pass** before and after every cycle (checked
at each of 10 checkpoints). No framework regressions. D1 baseline reproducibility
re-verified (unchanged). No new unit tests were required (cycles are analyses over
the existing, tested engine).

## 4. Experiments run
12 cycles (C1–C12). All net-of-cost (net@60bps unless noted), survivorship-biased
(upper bound), and reproducible (fixed seeds where stochastic).

## 5. Results (headline per cycle)
| # | cycle | verdict | key number |
|---|---|---|---|
| C1 | stricter OOS | **PASS** | bootstrap 90% CI Sharpe **[0.47, 1.99]** > 0; folds 80% positive |
| C2 | parameter sensitivity | partial | **100% positive**, median ~1.0, min 0.66 |
| C3 | implementation realism | **PASS** | limit-filtered 1.29; 1.02 at 150 bps |
| C4 | regime stability | **partial** | BULL 1.81 / **BEAR −0.42**; DD recovers ~8 mo |
| C5 | hard sector cap | keep soft | hard cap 22% > soft 21% (infeasible) |
| C6 | alternative momentum | partial | all defs +ve (0.73–1.27); composite 1.08 |
| C7 | universe bootstrap | **PASS** | 200 draws **100% positive**; 5/50/95 = 0.95/1.17/1.39 |
| C8 | factor regression | **CONFIRMED** | momentum β 0.83 (t=21.6); alpha t=1.60 (ns) |
| C9 | calendar decay | no decay | 2nd-half 1.31 ≥ 1st-half 1.20 |
| C10 | diversification | good | corr market +0.24, 0050 +0.13 |
| C11 | rebalance timing | robust-ish | Sharpe 0.90–1.27 by offset (1.27 favorable) |
| C12 | beta-hedge | intrinsic bear risk | hedge worsens bear (−0.19→−0.64) |

**Synthesis.** The momentum edge is genuinely robust where it counts —
out-of-sample (statistically > 0), across the universe (100% of subsets), across
cost/limit frictions, across momentum definitions, and it isn't decaying. It is a
**known-factor (momentum) product, not proprietary alpha** (C8, α t=1.60). Three
consistent honest qualifiers emerged repeatedly:
1. **The 1.27 headline is a favorable pick** — parameters (C2), momentum definition
   (C6), and even rebalance-calendar offset (C11) all show ~1.0–1.1 as the
   representative value. **Realistic expected deployable Sharpe ~0.8–1.0** after
   parameter + regime + survivorship + friction haircuts.
2. **Bear / high-vol regimes are the one real vulnerability** (C4), and it is
   *intrinsic momentum-crash risk*, not removable market beta (C1 overlay failed;
   C12 hedge worsens it).
3. **Survivorship remains an unquantifiable upward bias** (current-listed universe).

## 6. Commits created (10, all on the research branch)
```
5309954 C1 stricter OOS          13dd94c C2 param sensitivity
d4ba22d C3 impl realism          320b5b9 C4 regime stability
38e0485 C5 hard sector cap       7288fe6 C6 alt-momentum
c0f29c7 C7 universe bootstrap    eb31e2a C8 factor regression
866bae4 consolidation            ec95539 C9-12 deeper characterization
```
Every commit: tests 8/8, no production files staged, documented, reproducible.

## 7. Files changed
New: 9 research modules (see §2) + this report. Updated: `RESEARCH_LOG.md` (all
cycles), `RUN_STATUS.md` (9 heartbeats), `STRATEGY_CARD_D1_1.md` (C1–C8 robustness
table). **Untouched:** train.py, inference.py, dataset.py, test_dataset.py.

## 8. Failed / rejected ideas (kept honest, not hidden)
- **C5 hard sector cap** — rejected; counterproductive (re-concentrates to 22%).
- **C2 / C6 / C11** — the 1.27 headline does *not* generalise; the honest number is
  ~1.0. Recorded, not buried.
- **C4 / C12** — no way found to remove the bear-regime loss (overlay failed in
  Phase C1; beta-hedge worsens it). Documented as an accepted, intrinsic risk.
- **Multi-horizon composite signal (C6, ~1.08)** — a legitimate future robustness
  option; deliberately **not adopted** (don't change the frozen rule; don't chase).

## 9. Current repo status
Branch `research/d1-1-momentum-prototype`, 10 sprint commits ahead of `c9195b8`.
Working tree clean except the pre-existing (unrelated) leakage-fix edits to
train.py/dataset.py/test_dataset.py and REVIEW_4e7467e.md, which remain unstaged and
excluded by design. `.gitignore` keeps `research/data_cache/` out of the tree
(regenerable via research/data.py; results depend on the 2026-07-06 survivorship
snapshot). Tests 8/8. **D1.1 rule unchanged** — the sprint validated and
characterised it; nothing warranted a change.

## 10. Recommended next action
1. **Human review of this branch, then decide merge policy** (squash-merge keeps the
   one-off data blobs out of `main`). The branch is a defensible *research milestone*,
   not a production system.
2. **Update the deployable expectation to ~0.8–1.0 Sharpe** (not 1.27) in any
   downstream use — the sprint's clearest quantitative outcome.
3. **If pursued further:** (a) a real short-borrow/slippage model and a beta-neutral
   construction to attack the bear risk head-on; (b) evaluate the multi-horizon
   composite (C6) for signal diversification. Neither is expected to change the core
   conclusion.
4. **For genuine alpha beyond the momentum premium → Path B (orthogonal data:
   fundamentals/flows/alt-data).** Price/volume signal is exhausted; a bigger neural
   net will not help (Exp-4/C8). This needs data twstock does not provide.

*Active research concluded ~3h in because the pre-registered high-value + deeper-
characterisation queue (12 cycles) was genuinely exhausted; per the brief, continuing
would have meant make-work / Sharpe-chasing, which was explicitly forbidden. The
5-hour timebox was the ceiling, not a quota to fill.*
