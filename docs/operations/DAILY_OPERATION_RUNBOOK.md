# Daily Operation Runbook

## Canonical production directory (from 2026-09-22)

Production runs from a **dedicated worktree on `main`**:
`C:\Users\wilso\source\code\AI-Quant-prod`. `daily_ops.bat` and
`morning_execution_plan.bat` are run **from there only**. The original clone
`C:\Users\wilso\source\code\AI-Quant` is a research-only worktree (branch
`research/v17-production-audit-and-holdings`). Never switch its branch to
deploy, and never run production from it.

Update production with a fast-forward of `main`:
`git -C ..\AI-Quant-prod merge --ff-only <reviewed release branch>`, then
`git push origin main`. Research branches are never merged wholesale.

The production worktree needs local, git-ignored runtime state. None of it
is copied from research outputs:

| item | requirement |
|---|---|
| `research\data_cache\` | EOD cache. Seeded from the immutable DATA_BASELINE_V2 copy (`AI-Quant-backups\eod_cache_data_baseline_v2_20260921`, aggregate SHA-256 `c37dc970…`); afterwards maintained only by `daily_ops.bat` (append-only). |
| `.venv\` | **Shared environment:** a directory junction `AI-Quant-prod\.venv` → `AI-Quant\.venv` (created 2026-09-22). Verified with Python 3.12.3, torch 2.13.0.dev20260522+cu132, CUDA 13.2, RTX 4060 Ti. Production therefore depends on the research clone's venv. **Never install, upgrade or remove packages in it for research** without re-validating production; if research needs different packages, give research its own venv. |
| `my_holdings.csv` | **AUTHORITATIVE PRODUCTION HOLDINGS** = `C:\Users\wilso\source\code\AI-Quant-prod\my_holdings.csv` (initialised 2026-09-22 from the research clone's file, SHA-256 verified). It feeds the integrity gate's held-symbol hard block and steps 8–9. Edit holdings **only here**. |
| `reports\paper_trading\` books, decision books, ledger | the standing production lineage (previous book = band10 incumbents and the gate's held set) |
| `reports\transformer_gpu\<date>_*` | the previous neural target book (band) and predictions |
| `reports\user_actions\`, `reports\user_holdings\` | the previous standing plan (continuity and recovery) |
| `checkpoints\` | not needed: the daily retrain trains from scratch |

**Holdings source of truth.**
- `AI-Quant\my_holdings.csv` in the research clone is **non-authoritative**. It is a stale copy kept only for reference.
- There is no research → production write path. All code resolves `my_holdings.csv` relative to its own worktree root, so research code run in the research clone can only read or write the research clone's copy.
- Research that needs holdings must take a read-only snapshot of the production file, never a link that writes back.
- Do not point research tools at `AI-Quant-prod`.

The initial state migration is recorded in `reports\operations\PROD_RUNTIME_MIGRATION_MANIFEST.json`: source, destination, SHA-256 and reason for each file. Copied history is continuity input only. Freshness is enforced by:
- the step-start markers and dated-artifact gates for retrain, inference and book;
- the next-session plan's book-date checks (a book older than the newest cache date publishes no plan; an existing plan for the same book date is not regenerated).

**Intraday tasks (migrated 2026-09-25).** `AIQuant-IntradayCollector` (Mon–Fri 08:54) and `AIQuant-IntradayPostClose` (Mon–Fri 13:40) run from `AI-Quant-prod`. Their universe is the production decision book, and they read/write `AI-Quant-prod\research\intraday_cache\intraday.sqlite`, migrated with the SQLite backup API and verified (see `reports\operations\INTRADAY_DB_MIGRATION.json`). `morning_execution_plan.bat` therefore needs no `--db` override when run from prod. The research clone's intraday DB is a retained fallback: never let both worktrees write a SQLite DB concurrently, and run the morning plan only from prod.

## Data integrity statuses (integrity gate, step 1)

| status | meaning | pipeline |
|---|---|---|
| SAFE | no current model symbol has a broken 191-session window | runs |
| DEGRADED | one or more symbols that are not REAL_HELD excluded (DATA_INTEGRITY_FAILURE: no score, no rank). This includes names that are only in the previous model book; valid/eligible >= 99% | runs; exclusions listed in the report and book md |
| UNSAFE_FOR_NEW_MODEL_OUTPUT | a REAL_HELD symbol (open position in the canonical `my_holdings.csv`) invalid, or valid coverage < 99%, or structural date defects | aborts; the previous plan is untouched |

**When to start the nightly cycle.** Wait until BOTH exchanges have published the session, then start:
- TWSE: the market-wide all-stock table (MI_INDEX, ALLBUT0999) returns OK for the date with hundreds of traded symbols, and several liquid control symbols return that date from the production STOCK_DAY source;
- TPEx: the TPEx open-data mainboard close-quotes feed (`https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes`) carries the same session date. The universe has one TPEx-listed name (5903), and TPEx publishes later than TWSE; starting before it publishes leaves 5903 with a stale tail.

Readiness is a MARKET-SESSION question, never a single-symbol one: a symbol with no row after the session is confirmed is SYMBOL_NO_DATA, not an unpublished session. No individual symbol (2207, 5903, ...) is required to trade.

Under DEGRADED, portfolio construction keeps its size. The book size, band10 pool, neural k/band and watch list are sized on REFERENCE_ELIGIBLE_UNIVERSE_N = valid names plus every TEMPORARILY UNAVAILABLE name (invalid required window, or stale tail because its source had not published yet), while candidates are valid names only. Stale tails count for sizing only: the 99% valid-model coverage rule uses window-invalid names only, and the latest-date refresh gate keeps governing stale tails (user decision 2026-09-24, option 1). An excluded name's slot therefore goes to the next valid name. **Held vs model book.** The hard-block source is REAL_HELD_SYMBOLS only: the canonical `my_holdings.csv` open positions. The gate reports REAL_HELD, PREVIOUS_BOOK, their intersection and BOOK_ONLY separately. The previous decision book (PREVIOUS_BOOK_SYMBOLS) is band10 / hysteresis state and never blocks publication. A book-only name whose input is invalid leaves the model book with action `SELL`. Its `caveats` field starts with `DATA_INTEGRITY_EXCLUSION; NOT AN ALPHA-DRIVEN SELL; signal_driven=false`, and it has no score and no rank. Book md, overlay, plan reason, simplified status and daily diff all label it DATA_INTEGRITY_EXCLUSION. Because the name is not in `my_holdings.csv`, it maps to user action `NO_ACTION` and is never a user sell instruction. Real holdings outside the model universe (e.g. the ETF 0050, or the unconfigured 6669) have no model input window. They are reported as NO_MODEL_OPINION by holdings completeness. Confirmed symbol no-trade days live in `reports\data_integrity\symbol_no_trade_registry.csv`. They are never filled, and they still break contiguity.

## Commands

All commands run from the production worktree, using the venv python:
`.venv\Scripts\python.exe` (never system python; torch nightly + AMP live
in the venv).

**Run the whole cycle with `.\daily_ops.bat`** — it executes the 9 steps
below with fail-fast integrity gates (2026-08-24 incident fix). Manual
step-by-step runs are for debugging only.

## The 9-step nightly cycle (after TWSE close; bat-managed)

```text
[1/9] research\refresh_data.py        EOD cache refresh (bounded retry
      passes on empty/rate-limited responses; explicit PARTIAL warning)
      -> +0 rows: steps 2-7 are skipped, step 9 reports
         NO_NEW_SESSION_DATA and leaves the standing plan untouched
      -> gate: newest-date coverage must be >= 99% of the cached
         universe (partial TWSE publication aborts the pipeline here)
[2/9] train_transformer_eod.py --mode daily-retrain   (7-seed ensemble)
      -> gate: daily_manifest.json asof == newest cache date
[3/9] inference_transformer_eod.py    (+ thin-universe guard: >= 60
      scored names or inference aborts)
      -> gate: dated predictions/target_book/metrics/report all present
[4/9] research\blended_decision_book.py
      -> gate: dated blend50_band10 decision book present
[5/9] research\paper_trading.py snapshot
[6/9] research\paper_trading.py evaluate
[7/9] research\daily_diff_report.py
[8/9] research\user_holdings_overlay.py --strategy blend50_band10
      (skipped with a message when my_holdings.csv is absent)
[9/9] research\user_next_session_plan.py --nightly
      --eod-refresh-status <NEW_SESSION_DATA|NO_NEW_SESSION_DATA>
      (the bat passes the refresh outcome explicitly; UNKNOWN status
      refuses production generation)
```

**Exit-code / gate contract (replaces the old "ignore exit codes"
heuristic):** the bat writes a step-start marker before each GPU step
and calls `research\pipeline_gate.py <stage> --exit-code <rc>
--since-marker <marker>`. A step passes only when its expected
current-asof artifacts exist **with mtimes after the step started**. A
nonzero exit is tolerated in exactly one case — every expected artifact
was freshly written and only the torch/CUDA teardown crashed. Any other
nonzero exit, and any zero exit without fresh artifacts (silent no-op),
aborts the pipeline: downstream steps do not run and the standing user
plan is left untouched.

The main outputs to read afterwards:
`reports\user_actions\latest_next_session_summary.md` (simplified) and
`latest_next_session_action_plan.md` (technical). The next morning,
`.\morning_execution_plan.bat` refreshes live guidance (see
`docs\operations\V16_OPERATION_RUNBOOK.md`).

## User holdings overlay (step 8)

Compares actual holdings against the model book — review tool only; it never
generates orders.
- Enable: copy `my_holdings.example.csv` → `my_holdings.csv` (repo root) and
  fill in `symbol,shares[,avg_cost,current_price,current_value,account,notes]`.
  Missing prices fall back to the latest data_cache close. Keep symbols as
  strings (`0050` keeps its leading zero).
- If `my_holdings.csv` does not exist the step prints a skip message and the
  daily run continues — it never fails the cycle.
- Outputs: `reports/user_holdings/<asof>_user_holdings_overlay.{csv,md}` +
  `latest_user_holdings_overlay.{csv,md}` + a summary JSON. Same-day reruns
  overwrite deterministically.
- Read the md: High-priority table first (SELL/REDUCE on names held, |gap| >
  5pp, large uncovered positions), then not-covered / not-selected lists.
  `NOT_IN_MODEL_UNIVERSE` = no coverage, not bearish; `REDUCE` compares to
  the model's previous paper book, not the real portfolio.
- Options: `--strategy blend50_band10|blend50|d12|tf` (d7b has no daily
  books), `--date YYYY-MM-DD`, `--large-gap 0.05 --medium-gap 0.02`,
  `--dry-run`.
- Privacy: `my_holdings.csv` and `reports/user_holdings/` are gitignored —
  never commit them.

## How to check whether TWSE data landed

Step 1's last log line: `Refresh done: +N rows across M stocks in Ss`.
- `+0 rows` → no new session published yet (TWSE EOD can lag; retry later
  or next day).
- Expect roughly `+1 row × ~108 stocks` on a normal trading day.
- Partial publication no longer needs a manual eye: empty/rate-limited
  responses are retried in bounded passes, still-missing symbols are
  listed in an explicit `WARNING`, and the coverage gate (≥99% of the
  cached universe at the newest date) aborts the pipeline before the
  model ever sees a partial cross-section.

## What to inspect each day

| File | What to look for |
|---|---|
| `reports/paper_trading/<asof>_blend50_band10_decision_book.md` | actions line, sector exposure, max weight |
| `reports/continuous_research/daily_diffs/DIFF_<asof>.md` | Anomalies section (turnover >0.50, sector >50%) |
| `reports/paper_trading/PAPER_REPORT.md` | matured 20d summary per strategy |
| `reports/paper_trading/LEDGER_STATUS.md` | maturity counts; evidence-gate status |
| `reports/transformer_gpu/<asof>_train_log.md` | seed val ICs (healthy range roughly +0.10…+0.16 for daily retrain) |

## How to confirm no cache-mutation race

Rule: **never run refresh_data.py while any GPU training/backtest process is
running** (they read the cache). Check before refreshing:
`Get-Process python*` → must show nothing (or only your current shell).
The GPU scheduler enforces this itself (daily ops only at queue completion);
manual runs must respect it too. Cache CSVs may contain duplicate dates
after interrupted refreshes — readers dedupe (`drop_duplicates("date")`),
but if a refresh was interrupted mid-write, re-run step 1 (it is resumable).

## How to confirm max weight ≤ 7.5% / 10%

- Decision book md header prints `max weight X%` — must be **≤ 10.0%**
  (band10+cap10 production book).
- If running the D7b variant, its book must show **≤ 7.5%**.
- Ledger cross-check: `PAPER_LEDGER.csv` column `max_w` per snapshot.
- Any breach → construction bug: do not act on the book; investigate
  `cap_weights` inputs before the next cycle.

## How to review BUY / SELL / WATCH

Open `<asof>_blend50_band10_decision_book.md`:
- Header: `actions: REDUCE:n, WATCH:n, HOLD:n, BUY:n, SELL:n` + sector line.
- Table columns: symbol, model_score, rank, action, target_weight,
  previous_weight, weight_change, sector, confidence, holding_horizon_days.
- BUY = new entry; SELL = exits book (rank fell below band); HOLD/REDUCE =
  incumbent weight maintenance; WATCH = inside the widened band, not held.
- Sanity: ~22 names, 20d horizon, execution is the NEXT session — never
  same-day. (Backtests use next close; executing at the next open is
  validated equivalent — NEXT_OPEN_TIMING_VALIDATED, v16 timing audit.)
  Caveat line applies (survivorship-biased universe).

## How to review matured returns (1d / 5d / 10d / 20d)

`reports/paper_trading/PAPER_LEDGER.csv` — one row per (snapshot, strategy)
with `ret_1d/5d/10d/20d`, `hit_*`, `turnover`, `max_w` (NaN = not matured
yet). `PAPER_REPORT.md` aggregates matured 20d by strategy;
`LEDGER_STATUS.md` shows maturity counts and the evidence gate (activates at
20 matured 20d obs → compare realized ann Sharpe vs bootstrap CI: champ
p5 1.61/p50 1.92, bear p5 1.14/p50 1.37).

## Emergency stop / recovery

**Stop:** `Get-Process python* | Stop-Process -Force` (books/queues are
crash-safe — state files rewritten after every unit of work).

**After a crash / unexpected shutdown:**
1. Follow the recovery-audit pattern in
   `docs\archive\RECOVERY_AFTER_SHUTDOWN.md`
   (2026-07-25 precedent): git status → inspect newest logs/queue JSONs →
   validate JSON/JSONL/gzip integrity → mark anything partial as invalid.
2. Never trust partial metrics from an interrupted run; re-pend and re-run.
3. CUDA `unknown error`: reboot first, verify
   `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`
   and `nvidia-smi` before resuming. If it recurs on an idle GPU, stop and
   write a hardware/driver stability note — do not retry in a loop.
4. Interrupted refresh: just re-run `refresh_data.py` (resumable,
   idempotent); dedupe on read handles duplicate dates.
5. Nothing in daily ops writes to git — recovery never needs git surgery
   beyond checking `git status` is clean of surprises.
