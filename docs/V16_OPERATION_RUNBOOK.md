# V16 Operation Runbook — Next-Session Execution Advisor

Layered stack (never collapsed into one BUY/HOLD/SELL label):
MODEL (what AI-Quant wants) → POSITION-AWARE LAYER (what applies to my
actual holdings) → NIGHT PRICE LAYER (historically reasonable
next-session prices) → LIVE MARKET LAYER (actual bid/ask/trade/open vs
the night plan) → **MANUAL HUMAN DECISION**. No automatic trading, no
broker APIs, no guaranteed fills, no short-entry model.

Validated foundation: NEXT_OPEN_TIMING_VALIDATED (f6d629b) — running
daily_ops at ~22:00 on day T and executing manually after the next
session opens is the historically supported convention.

## NIGHT (trading day T, ~22:00)

1. `.\daily_ops.bat` — nine steps: refresh EOD → retrain → inference →
   decision book → paper snapshot → paper evaluate → daily diff →
   holdings overlay → **[9/9] next-session user action plan**
   (`research/user_next_session_plan.py --nightly`).
2. Read **`reports\user_actions\latest_next_session_action_plan.md`** —
   the main nightly report (per-action price bands, legal domain,
   expected-open distribution, range-reach probabilities).

## MORNING (trading day T+1)

1. The v15 collector starts by itself at 08:54
   (AIQuant-IntradayCollector; post-close processing at 13:40).
2. After ~09:02: `.\morning_execution_plan.bat` (MANUAL — deliberately
   not scheduled; see "Remaining deployment step").
3. Read **`reports\user_actions\latest_live_execution_plan.md`**.
4. Manual review → manual broker order entry if desired.

## Behavior reference

| Situation | Behavior |
|---|---|
| No new EOD session (+0 rows / same-session rerun) | daily_ops passes the refresh outcome EXPLICITLY (`--eod-refresh-status NO_NEW_SESSION_DATA`) — step 9 hard-gates on it regardless of which output files exist. With a standing plan: its dates are reported and it stays byte-unchanged. With no standing plan: **NO_STANDING_ACTION_PLAN** — nothing is generated (first-deploy/cleaned-output safety). A stale diagnostic never masquerades as a fresh plan. |
| Manual `--nightly` run without a refresh result | **EOD_REFRESH_STATUS_UNKNOWN** — production generation refused by default. Explicit manual recovery: `--allow-current-book-recovery` (clearly-labeled RECOVERY plan re-issued from the standing book with a re-derived intended session; never used by daily_ops.bat). |
| Intended date proves to be a TWSE holiday | The intended date is a **weekday estimate (UNVERIFIED_FOR_HOLIDAYS** — no trading calendar exists in the repo/deps, and none is faked). The morning wrapper refuses with SESSION_MISMATCH and is never rolled forward automatically; re-issue via the manual recovery command above, which preserves signal_date so the plan's age stays visible. |
| Newest EOD session covers <90% of the universe | **PARTIAL_PUBLICATION_SUSPECTED** (pre-registered gate): no new actionable plan; loud warning; re-run after publication completes. |
| Decision book older than newest cache date | **STALE_BOOK**: no new actionable plan (the book was not regenerated). |
| `my_holdings.csv` missing | Steps 8 and 9 are skipped with a clear message; the model pipeline still succeeds (exit 0). No fake "no holdings" actions. |
| Morning run before 09:02 | **WAITING_FOR_MARKET_OPEN** — normal actionable refresh begins at open + 120s (MIN_AFTER_OPEN_SECONDS, centralized in `market_readiness.py`; a data-quality gate, not tuned from returns). The wrapper polls (default 15s, max 5min) and exits **MARKET_DATA_NOT_READY** on timeout. |
| No/old current-session collector rows | **WAITING_FOR_MARKET_DATA** → same wait/timeout. Individual missing or stale symbols never block the session — C1 row-level gates handle them. |
| Run after 13:30 | **MARKET_CLOSED** — actionable refresh refused; `--diagnostic` produces an explicitly labeled HISTORICAL_SESSION_DIAGNOSTIC instead. |
| Plan date ≠ session date | **SESSION_MISMATCH / LIVE_PLAN_DATE_MISMATCH** — nothing actionable; yesterday's plan is never silently applied. |
| Observed TWSE price outside the assumed ±10% domain | **PRICE_DOMAIN_ASSUMPTION_CONFLICT** — the quote is NOT called illegal (a special auction reference may apply; the repo cannot verify references). Price preserved unchanged; row not actionable; manual review. |
| Quote STALE/MISSING | Row not actionable; no live limit suggested. |
| Only a midquote / stale trade available | Shown as a **state proxy** ("market-state proxy ≈ X"), confidence DEGRADED, never phrased as "buy/sell at X". |
| Price above the preferred execution range | Execution-quality context only ("unusually expensive vs the historical next-open distribution") — the validated model signal is NOT invalidated; the decision stays manual. |
| Actual shorts in holdings | Risk-tracked (HOLD_SHORT/REDUCE_SHORT/BUY_TO_COVER). NO VALIDATED OPEN-SHORT MODEL EXISTS; no short entry is ever suggested. |
| 0050 / outside model universe | NO_MODEL_OPINION — no fabricated bands or actions, live data notwithstanding. |

## Exit codes (nightly step 9 and morning refresh)

0 = success OR a legitimately-skipped/warned state (warnings are printed
loudly: NO_NEW_SESSION_DATA, PARTIAL_PUBLICATION_SUSPECTED, STALE_BOOK,
MISSING_HOLDINGS). Nonzero = genuine failure (2 = missing/invalid
inputs, 3 = market data not ready / market closed in normal mode).

## Partial-publication threshold (policy adopted 2026-08-19)

`PARTIAL_COVERAGE_MIN = 0.99` — a DATA-INTEGRITY threshold (not an
alpha parameter; never optimized from returns). Interpretation for the
~108-name universe: full publication passes (108/108); at most one
missing universe name may pass (107/108 ≈ 99.07% — tolerates a single
idiosyncratic suspended/missing name); two or more missing names block
(106/108 ≈ 98.15%). This is intentionally conservative because the
production model is CROSS-SECTIONAL: multi-name partial publication can
materially alter top-book membership, rank percentiles, z-scores, and
target weights. `PARTIAL_PUBLICATION_SUSPECTED` means no new actionable
nightly plan is generated and the standing plan is never overwritten.

## Remaining deployment step (deliberately NOT done)

`AIQuant-MorningExecutionPlan` is NOT scheduled. One real-session manual
smoke of `morning_execution_plan.bat` on the next trading morning is
required before deciding whether to schedule it. The existing collector
tasks are untouched.
