# Daily Operation Runbook

All commands from the repo root (`C:\Users\wilso\source\code\AI-Quant`),
using the venv python: `.venv\Scripts\python.exe` (never system python;
torch nightly + AMP live in the venv).

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
