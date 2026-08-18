# v16 Task 2 — Execution-Timing Audit: Methodology & Pre-registered Gates

Written 2026-08-18 **before any experimental result was computed**
(reproduction check A excepted — its expected values 2.147/1.443 are the
standing references, documented long before v16). CPU-only; frozen panels;
no retraining; cache read-only.

## Signal (identical across all conventions)

Frozen 7-seed walkforward score panels, unchanged bytes:
- CH (champion window, OOS 2023→): reports/transformer_gpu/panels/SCHED_A8_seeds7_full.csv.gz
- BR (bear window, OOS 2021→):     reports/transformer_gpu/panels/SCHED_BEAR_A8_seeds7_full.csv.gz

Score = production blend: 0.5·z(tf score) + 0.5·z(D1.2 momentum), per date
(transformer_hybrid.merged / queue_v9_lib.with_score — same code as v9).
Portfolio construction = the validated engine transformer_portfolio.
backtest_scores, untouched: top-quintile, equal weight, hard 10% name cap,
soft 20% sector cap, no-trade band 0.10, rebalance every 20 panel dates,
min_names 60. Only `ret_col` differs between conventions.

## Executable-price conventions (outcome columns built from the raw cache)

All columns constructed from research/data_cache/*.csv (date-deduped,
sorted), indices aligned exactly like dataset_transformer_eod._fwd_ret.
For a signal row dated T (indices in trading sessions):

| Key | Convention | Entry | Exit | Formula |
|---|---|---|---|---|
| A | Existing reference | T+1 close | T+21 close | close[t+21]/close[t+1] − 1 (must equal panel fwd_h) |
| B | **User's actual workflow** | **T+1 open** | **T+21 open** | open[t+21]/open[t+1] − 1 |
| C | = B with cost ladder | | | net0/60/100/150 on B |
| D | One-session delay | T+2 open | T+22 open | open[t+22]/open[t+2] − 1 |
| E | Identity check | T+1 close | T+21 close | rebuilt independently; must equal A |
| F | Diagnostic upper bound (NOT available to a 22:00 workflow) | T close | T+20 close | close[t+20]/close[t] − 1 |
| — | Decay point | T+2 close | T+22 close | close[t+22]/close[t+2] − 1 (cross-check vs queue_v9 X3 fwd_lag2) |

Blocks chain exactly under every convention (each block's exit price is
the next block's entry price), so compounding is internally consistent —
convention B is a true open-executed portfolio, not a column swap with
mismatched endpoints.

## No-lookahead assertion (programmatic)

For every constructed column: outcome uses only prices at t+lag or later
(lag ≥ 1 except diagnostic F, which is labeled as unavailable). Assertions
in the runner: (1) rebuilt A equals the panel's own fwd_h to ≤ 1e-6 max
abs diff (the panel's fwd_h was GPU-produced by the audited pipeline;
agreement proves index alignment); (2) a synthetic 30-row series with
known open/close patterns yields hand-computed values for every column.
open(T+1) never enters signal formation — the score panel is frozen bytes
from runs that predate v16.

## Coverage control

backtest_scores drops rows with NaN in the active ret_col, so conventions
with longer forward reach (D) lose extra tail dates. To keep selection
identical, all conventions are ALSO run on the common-coverage subset
(rows where every needed column is finite). Both sets are reported;
convention comparisons use the common subset; the reproduction gate for A
uses the raw panel (to match the standing references exactly).

## Cost formula (the validated backtest convention — NOT the daily_diff 2×)

cost_per_rebalance = n_legs × (cb/1e4) × turn, where turn = per-leg
one-way L1 turnover (Σ|Δw|/2, averaged over legs), cb ∈ {0,60,100,150}
(round-trip bps; 60 bps round-trip = 30 bps/side), n_legs = 2 for L/S,
1 for long-only (transformer_portfolio.py:161-172). The daily_diff
report's 2× display convention (F3, timing_audit.md) is explicitly not
used.

## Books and windows

Both modes on both windows: long_short (the methodology behind the
standing references — primary for the verdict) and long_only (the actual
deployment book — reported alongside). Yearly net60 breakdown incl. 2022,
2023, 2024, 2025, 2026-YTD; hit rate; avg per-rebalance return; turnover;
max DD.

## Overnight-gap attribution

Per rebalance date T and selected name: overnight_gap = open(T+1)/close(T)
− 1; intraday_after_open = close(T+1)/open(T+1) − 1. Broken down by book
role (fresh entrant vs incumbent vs exiting name — reconstructed from
consecutive books via holdings_walk-style replica), score quintile, rank
bucket, and sector (only where a bucket has ≥ 200 observations). Questions
answered: do entrants gap against the entry; where does the A-vs-B
difference concentrate; do avoidable gap regimes exist.

## Corporate-action robustness (pre-registered filter)

The cache is dividend/split-unadjusted (timing_audit.md F5). Filter,
fixed before results: an observation (rebalance T, name s) is
"extreme-gap" iff |open(T+1)/close(T) − 1| > 0.05. Robustness variant:
drop extreme-gap names from the book at that rebalance (same names
dropped under EVERY convention — the drop is defined by the gap, not the
convention — keeping the comparison paired), renormalize remaining
weights proportionally. Report headline metrics both ways plus the count
and a list of the 20 most extreme gaps (expect TWSE ex-dividend clustering
Jul–Sep). Nothing is silently deleted; both versions are published.

## Pre-registered reproduction gate

A (raw panel, L/S blend band10 cap10, net60) must reproduce the standing
references: CH 2.147, BR 1.443, tolerance ±0.005 (deterministic panel +
deterministic engine → expected exact). E must equal A (≤1e-9 on block
returns). Failure of either ⇒ verdict BACKTEST_EXECUTION_MISMATCH_FOUND,
stop, investigate before any interpretation.

## Pre-registered verdict thresholds (recorded before computing B)

Primary metric: net60 Sharpe, L/S, per window; retention =
sharpe(B)/sharpe(A). The ±0.30 absolute-drop bar reuses the only
precedent in this repo for an accepted execution perturbation: the v9
pre-registered T+2 delay bar (≤0.30 Sharpe, measured cost 0.226, verdict
"delay-robust"). Sampling noise context: with ~44 (CH) / ~68 (BR)
non-overlapping blocks, 1 SE of the annualized Sharpe is roughly 0.5–0.6,
so drops under ~0.3 are statistically indistinguishable from zero.

- **NEXT_OPEN_TIMING_VALIDATED**: sharpe(A) − sharpe(B) ≤ 0.30 in BOTH
  windows, AND B ≥ 1.55 (CH) and ≥ 1.00 (BR) net60, AND B's 2022 yearly
  net60 mean (BR window) not worse than A's by > 2pp per rebalance.
- **NEXT_OPEN_EDGE_WEAKER_BUT_USABLE**: not validated, but
  sharpe(A) − sharpe(B) ≤ 0.70 in both windows, AND B ≥ 1.20 (CH) and
  ≥ 0.85 (BR), AND B's 2022 mean not worse than A's by > 3pp per
  rebalance.
- **NEXT_OPEN_TIMING_NOT_SUPPORTED**: anything below the usable bar.
- **BACKTEST_EXECUTION_MISMATCH_FOUND**: reproduction gate failed.

Secondary guard: if the long-only retention is worse than the L/S
retention by more than 0.20 (ratio terms), the verdict is downgraded one
level (the deployment book must not hide behind the reference book).
Robustness guard: if the corporate-action-filtered comparison changes the
verdict category, report the WORSE category and flag data-integrity as
the binding issue.

These thresholds are final as of this file's creation; results follow in
next_open_execution_audit.md.
