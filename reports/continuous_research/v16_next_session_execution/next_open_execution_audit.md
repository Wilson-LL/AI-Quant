# v16 Task 2 — Next-Open Execution Audit: Results & Verdict

Date: 2026-08-18 · Runner: research/next_open_execution_backtest.py
Thresholds pre-registered in execution_cost_methodology.md BEFORE these
numbers were computed. Frozen 7-seed panels; identical signal, ranking,
selection, weights, rebalance dates and turnover across conventions; only
the executable-price column differs. CPU-only; no retraining.

## Integrity gates (all pass)

- Synthetic forward-return alignment (lag 0/1/2): OK.
- Rebuilt close-lag-1 vs panel fwd_h: max |diff| 1.41e-07 over 229k rows
  (the same magnitude the v9 X3 same-bar audit recorded).
- Reproduction gate A (raw panel, L/S, net60): **CH 2.147 = 2.147, BR
  1.443 = 1.443** — exact.
- Identity E==A (paired coverage): block returns agree to ≤3.6e-9. Note:
  unrestricted E sees one extra rebalance because the live cache has
  matured ~21 sessions of labels that were immature at panel freeze
  (2026-05-25..07-17); diagnosed, benign, check made paired.
- Replica engine (used only for the gap-filter variant) matches
  backtest_scores to ≤0.001 Sharpe on every window/mode/convention.
- External cross-check: the T+2-close decay cost measured here (CH L/S
  2.147→1.921 = **0.226**) equals the queue_v9 X3 delay audit's 0.226.

## Headline: A (T+1 close, all validated evidence) vs B (T+1 open, the user's actual workflow)

net60 Sharpe, common-coverage subset, identical books:

| Window | Mode | A: T+1 close | B: T+1 open | Retention B/A | ΔDD (pp) | Δann.ret (pp) |
|---|---|---|---|---|---|---|
| CH 2023→ | long_short | 2.147 | **2.158** | 1.005 | −0.04 (better) | −1.4 |
| CH 2023→ | long_only | 1.989 | **1.918** | 0.964 | −0.35 (better) | −1.9 |
| BR 2021→ | long_short | 1.443 | **1.463** | 1.014 | −1.92 (better) | +1.2 |
| BR 2021→ | long_only | 1.455 | **1.505** | 1.034 | +0.66 (worse) | +0.7 |

Turnover is identical by construction (selection does not depend on the
outcome column). Cost ladder for B (= convention C): see
signal_decay_report.csv — B remains ≥1.15 net150 in the bear window L/S
and ≥1.39 net150 long-only.

Year-by-year (net60): B tracks A within ±0.2 Sharpe in every ordinary
year. 2022 (bear window): A −0.15 / B −0.17 L/S (mean −0.21% vs −0.22%
per rebalance — a 0.01pp difference); long-only 2022 A +0.05 / B −0.06
(mean +0.11% vs −0.11%, a 0.22pp difference). Both far inside the
pre-registered 2pp guard.

## Corporate-action robustness (pre-registered |gap|>5% filter, paired)

| Window/mode | A filtered | B filtered | B−A |
|---|---|---|---|
| CH L/S | 1.904 | 1.859 | −0.045 |
| CH LO | 1.849 | 1.754 | −0.095 |
| BR L/S | 1.448 | 1.476 | +0.028 |
| BR LO | 1.464 | 1.520 | +0.056 |

Dropped observations: 21–81 per window/mode (of ~1,000–1,900 name-
rebalances). The verdict category is unchanged under the filter. Two
honest observations: (1) the top-20 extreme gaps are dominated not by
dividends but by ±10% limit-move days — 14 of 20 are the 2025-04-09
market-wide limit-up rebound — so the filter is primarily a
market-regime filter here; (2) removing those names lowers BOTH
conventions' absolute Sharpe (extreme-gap entries carried positive
signal), which means avoiding extreme-gap days would have cost
performance, not saved it. Dividend ex-date contamination (timing_audit
F5) remains a real absolute-level caveat but does not affect the
open-vs-close comparison, which is paired by construction.

## Pre-registered verdict evaluation

- ΔSharpe(A−B) ≤ 0.30 both windows (L/S): −0.011 / −0.020 → PASS
  (B is not worse at all).
- B floors (L/S ≥1.55 CH / ≥1.00 BR): 2.158 / 1.463 → PASS.
- 2022 guard (≤2pp per-rebalance degradation): 0.01pp → PASS.
- Long-only secondary guard (LO retention within 0.20 of L/S): min LO
  retention 0.964 vs LS 1.005 → PASS.
- Robustness guard (filter must not change category): unchanged → PASS.

## VERDICT

```text
NEXT_OPEN_TIMING_VALIDATED

reference_T1_close_net60   : CH 2.147 / BR 1.443 (L/S)   [LO 1.989 / 1.455]
actual_T1_open_net60       : CH 2.158 / BR 1.463 (L/S)   [LO 1.918 / 1.505]
relative_sharpe_retention  : 1.005 / 1.014 (L/S)         [LO 0.964 / 1.034]
DD_change                  : −0.04pp / −1.92pp (better)  [LO −0.35 / +0.66pp]
return_change (ann, net60) : −1.4pp / +1.2pp             [LO −1.9 / +0.7pp]
turnover_change            : 0.000 (identical books by construction)
```

Historical OOS evidence supports next-open execution: over both
validation windows, executing the 22:00 signal at the next session's open
is statistically indistinguishable from the backtested next-close
convention, at every cost level tested, in both book modes, before and
after the extreme-gap robustness filter. **This is a statement about
historical out-of-sample evidence, not a guarantee of future
profitability.**

Why the result is unsurprising in hindsight (see
next_open_gap_attribution.md): the blend signal is slow — entrant names
gap only ~+12–26 bps (median) overnight against the entry, one gap per
20-session hold, and the diagnostic same-close convention F is NOT better
than A/B (CH L/S 2.115 vs 2.147/2.158) — the alpha accrues over the
20-day hold, not in the first overnight.

## Scope and caveats

- Convention comparison is exact-paired; absolute Sharpe levels inherit
  the known caveats (survivorship-curated universe, dividend-unadjusted
  prices, net60 cost model, open prices assumed fillable at the print).
- The open-execution model assumes fills at the official open price; real
  fills near the 09:00 auction carry spread/impact not modeled here —
  that is Task 9/10 territory (price bands + live advisor), not a timing
  question.
- Windows end 2026-07-23 (panel freeze). 2026 rows are YTD.
