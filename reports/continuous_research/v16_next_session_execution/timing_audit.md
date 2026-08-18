# v16 Timing Audit — Signal vs Execution Timing (Task 1)

Date: 2026-08-18 · Branch: research/v16-next-session-execution-advisor
Read-only audit; no production behavior changed. Production config verified
from `checkpoints/transformer_eod/daily_manifest.json`: close_only features,
tgt_rank_20, horizon 20, preset B (seq 60), 7 seeds.

## 1. The complete timestamp trace

| Stage | Timing | Evidence |
|---|---|---|
| Feature cutoff | Rolling window of the last 60 sessions ending **at and including T's close**; longest lookback ~T−190 (mom_126_5 needs T−131 + seq 60) | dataset_transformer_eod.py:409-417, :59 |
| Signal timestamp | Date T (the latest cached EOD date); scores computed from the window ending T | inference_transformer_eod.py:145-148 |
| Inference wall-clock | Whenever daily_ops runs (user: ~22:00 TW, after T's close; TWSE EOD publishes by ~14:30+) | daily_ops.bat:3, DAILY_OPERATION_RUNBOOK.md:9 |
| Decision-book timestamp | Signal date T in the **filename only**; no machine-readable signal_date column | blended_decision_book.py:108, :101-103 |
| Label entry price | **close(T+1)** — `fwd_20[t] = close(t+21)/close(t+1) − 1`, exec_lag=1 | dataset_transformer_eod.py:260-266, :310 |
| Label exit price | **close(T+21)** (20 sessions after entry) | same |
| Backtest rebalance | Every 20th panel date; block return = dot(weights, fwd_h) — i.e. **enter T+1 close, exit T+21 close**, blocks chain exactly | transformer_portfolio.py:111, :136-138 |
| Paper ledger execution | Independently hard-coded to the same convention: entry `all_dates[pos+1]` **close**, matured h sessions later | paper_trading.py:142-153 |
| `intended_execution_date` | Descriptive string "next trading day after \<asof\>" — written at blended_decision_book.py:97 and inference_transformer_eod.py:129, **read by nothing**; accounting re-derives T+1 independently. It also omits the word "close" | repo-wide grep: zero reads |
| User's actual workflow | Run daily_ops ~22:00 after T closes → **manually trade after T+1 opens ~09:00** | user statement, v16 spec |

Same-bar audit on record: queue_v9 X3 rebuilt lag-1 returns from the raw
cache and matched the panel's fwd_h to max abs diff 1.4e-07
(reports/continuous_research/queue_v9/X3_result.json) — nobody is cheating
with same-close fills. That audit says nothing about open-vs-close at T+1.

## 2. The central distinction

```text
SIGNAL LAG:        T close → T+1 execution          ALIGNED
```
Features use only data published by T's close; TWSE EOD is available hours
before 22:00; every target/backtest/ledger applies exec_lag=1. Running
daily_ops at 22:00 and acting the next session is exactly the intended lag.

```text
EXECUTION POINT:   T+1 OPEN (user) vs T+1 CLOSE (all validated evidence)
                                                    UNVALIDATED
```
Every published number — the tgt_rank_20 target itself, all
transformer_portfolio Sharpes, the standing references 2.147/1.443, the
paper ledger — assumes fills at **T+1 close**. The user fills at **T+1
open**. No open-price return series has ever been constructed in this repo
(the `open` column is cached for all 110 names since 2015 and is loaded,
but no return/backtest path reads it; the only consumers are two
non-production feature blocks). The single place the repo says "close" out
loud is DAILY_OPERATION_RUNBOOK.md:115; every user-facing string says only
"next session". The existing delay stress (T+2/T+3 close, worst cost 0.226
Sharpe, queue_v9 X3) bounds a different perturbation — a full extra
session, not the open-vs-close difference — and cannot be cited as cover.

**This gap is what Task 2 measures.** The prices exist in the cache; the
frozen 7-seed panels (SCHED_A8_seeds7_full 2023→, SCHED_BEAR_A8_seeds7_full
2021→) provide the exact signals; the comparison is CPU-only.

## 3. Secondary findings (documented, deliberately NOT fixed in v16.1)

None of these invalidates Task 2 (rationale after each). All are candidates
for later v16 tasks or separate fixes, gated on user approval.

**F1 — Partial-publication risk in the refresh guard.** daily_ops.bat:13
branches only on the literal `+0 rows`. A partially-published session
(e.g. +5 of ~108 names) flows through: inference takes asof = max(date),
only complete 60-day windows produce samples, and `make_decision_book`
happily emits a 3-name book that passes its own assertions
(inference_transformer_eod.py:134-137). `n_stocks_scored` in the metrics
JSON would reveal it, but nothing asserts a floor; the runbook (:73-75)
delegates this check to the human. *Task 2 unaffected: it uses frozen
panels, not the daily path.*

**F2 — Stale-book behavior on +0-rows days.** On skip, dated artifacts are
correctly left untouched, but user_holdings_overlay.py:552-553 rebinds to
the newest book of any age and rewrites `latest_user_holdings_overlay.
{csv,md}` with a fresh mtime, undated filename, and no staleness banner —
the exact files the runbook tells the user to read. A stale book can be
mistaken for today's. *Task 2 unaffected; the fix is Task 3/12
(book_stale metadata), deferred.*

**F3 — daily_diff cost-convention mismatch (2×).** The validated backtest
charges net60 = 60 bps round-trip = 30 bps/side on one-way turnover
(transformer_portfolio.py:161-163,171-172; ×2 legs only for L/S). The
daily diff prints `2*turn*60/1e4` — 60 bps per side, twice the backtest's
charge (daily_diff_report.py:62-64) — and docs/continuous_research/
METHODS.md:10 describes the diff's convention, not the backtest's. The
published Sharpes follow the code, not the doc. *Task 2 will use the
backtest convention and document the formula (execution_cost_
methodology.md).*

**F4 — Paper-ledger metrics are gross and overlapping.** Ledger returns
are explicitly gross (paper_trading.py:182) and snapshots are daily while
returns span 20 days, yet Sharpe is annualized as if blocks were
independent (:186). Ledger Sharpes are not comparable to the net60
non-overlapping references. *Task 2 unaffected: it uses the
non-overlapping engine.*

**F5 — Prices unadjusted for corporate actions.** research/data.py:164-171
stores raw twstock prices; no adjustment code exists repo-wide. TWSE cash
dividends create real ex-date down-gaps that enter features and the 20d
label as genuine negative returns. This affects close- and open-based
conventions alike, so the A-vs-B *comparison* remains fair, but the
overnight-gap leg of the open convention is the more directly exposed one.
*Handled inside Task 2 by a pre-registered extreme-gap robustness filter
(both versions reported, nothing silently deleted); a data-pipeline
redesign is out of scope.*

## 4. Conclusion

The 22:00 workflow is correctly lagged and safely after publication; the
execution *point* (next open) is the untested half. Until Task 2 reports,
the honest status is: **execution timing validated: NO (pending
next-open audit)**.
