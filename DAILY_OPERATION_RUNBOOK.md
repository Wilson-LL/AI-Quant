# Daily Operation Runbook

All commands from the repo root (`C:\Users\wilso\source\code\AI-Quant`),
using the venv python: `.venv\Scripts\python.exe` (never system python;
torch nightly + AMP live in the venv). Teardown exit codes from torch can be
nonzero on process exit — ignore exit codes if the logged output says the
step completed.

## Daily cycle (after TWSE close, ~14:30+ TW time; total ~12 min)

```powershell
# 1. Refresh EOD cache (network-bound; appends missing months only)
.venv\Scripts\python.exe research\refresh_data.py
#    optional: --full-fields  (also accumulates turnover/transaction in
#    research\data_cache_full for the ~2027-01 full-field revisit)

# 2. Retrain the 7-seed ensemble (only if refresh added rows; ~5 min)
.venv\Scripts\python.exe train_transformer_eod.py --mode daily-retrain

# 3. Daily inference -> <asof>_predictions.csv
.venv\Scripts\python.exe inference_transformer_eod.py

# 4. Blended decision book (BUY/SELL/HOLD/REDUCE/WATCH)
.venv\Scripts\python.exe research\blended_decision_book.py

# 5. Paper book snapshots (4 strategies) + matured ledger update
.venv\Scripts\python.exe research\paper_trading.py snapshot
.venv\Scripts\python.exe research\paper_trading.py evaluate

# 6. Daily diff report (entries/exits, weight deltas, anomalies)
.venv\Scripts\python.exe research\daily_diff_report.py
```

Steps 2–6 are pointless when step 1 reports `+0 rows` (weekend/holiday/EOD
not yet published) — stop after step 1 in that case.

## How to check whether TWSE data landed

Step 1's last log line: `Refresh done: +N rows across M stocks in Ss`.
- `+0 rows` → no new session published yet (TWSE EOD can lag; retry later
  or next day; the loop historically retried ×2 then moved on).
- Expect roughly `+1 row × ~108 stocks` on a normal trading day. Partial
  (+N ≪ 108) → exchange still publishing; re-run step 1 later before
  proceeding.

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
- Sanity: ~22 names, 20d horizon, execution is NEXT session at close —
  never same-day. Caveat line applies (survivorship-biased universe).

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
1. Follow the recovery-audit pattern in `RECOVERY_AFTER_SHUTDOWN.md`
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
