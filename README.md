# AI-Quant

TWSE equity decision-support system: a nightly EOD model pipeline plus a
live-session execution advisor. **AI-Quant never places broker orders —
every trade decision and order entry is manual.**

## Daily Use

1. **Update actual positions** (what you really hold):
   `my_holdings.csv`
   (schema: `symbol,side,shares,avg_cost,current_price,current_value,account,notes`;
   `side` = LONG/SHORT; see `my_holdings.example.csv`)

2. **Evening, after EOD data is available (~22:00):**
   `.\daily_ops.bat`

3. **Read tomorrow's simplified plan:**
   `reports\user_actions\latest_next_session_summary.md`

4. **During the next trading session, when live guidance is needed
   (from ~09:02):**
   `.\morning_execution_plan.bat`

5. **Read:**
   `reports\user_actions\latest_live_execution_summary.md`

6. **Optional technical detail:**
   `reports\user_actions\latest_next_session_action_plan.md`
   `reports\user_actions\latest_live_execution_plan.md`

Notes on what these do:

- `my_holdings.csv` represents your actual positions; the reports
  separate what the *model* wants from what applies to *you*.
- `daily_ops.bat` runs the 9-step nightly pipeline (EOD refresh → model
  retrain → inference → decision book → paper tracking → holdings
  overlay → next-session action plan) with fail-fast integrity gates;
  it produces the next-session decision layer.
- `morning_execution_plan.bat` refreshes the nightly plan against live
  market data (bid/ask/trades from the intraday collector) and writes
  today's execution guidance.
- Dated copies of every report are kept under
  `reports\user_actions\history\YYYY-MM\`.
- If no plan was generated (no new EOD session, partial publication,
  stale book), `reports\user_actions\latest_nightly_status.md` says why.
- Reference prices are historical, conditional research estimates —
  never guaranteed fills. Nothing here is investment advice.

## Automation (Windows Task Scheduler)

Two scheduled tasks run without user action on trading weekdays:

- **AIQuant-IntradayCollector** (08:54) — supervised TWSE intraday quote
  collection (`research\intraday_collector\collector_supervisor.py`,
  single-instance locks + watchdog restarts; session ends 13:35).
- **AIQuant-IntradayPostClose** (13:40) — builds 1-minute bars and the
  daily data-quality report from the collected session.

The PC must be on and logged in for the tasks to fire. The morning
execution refresh is deliberately NOT scheduled — run it manually.

## Architecture (short)

- **Model**: LSTM-CondTransformer (`model.py`), close-only features,
  cross-sectional 20-day rank target, 7-seed ensemble, retrained daily
  (`train_transformer_eod.py` → `inference_transformer_eod.py`).
- **Decision layer**: 50/50 z-blend with momentum, top-quintile book,
  10% name cap, no-trade band (`research\blended_decision_book.py`);
  paper-trading tracking (`research\paper_trading.py`).
- **User layer** (v16): position-aware actions from your holdings
  (`research\holdings.py`, `research\user_holdings_overlay.py`),
  TWSE price-domain + empirical execution bands
  (`research\twse_price_domain.py`, `research\execution_price_bands.py`),
  nightly plan (`research\user_next_session_plan.py`) and live refresh
  (`research\intraday_advisor\`), rendered as simplified Traditional-
  Chinese summaries (`research\simplified_reports.py`).
- **Integrity**: `research\pipeline_gate.py` blocks partial EOD
  publication and stale artifacts; execution timing is validated for
  next-open execution (see `reports\continuous_research\
  v16_next_session_execution\`).

Environment: Windows + Python venv (`.venv\`) with a working
PyTorch/CUDA install (RTX 4060 Ti; see
`docs\operations\RTX4060TI_ENVIRONMENT_CHECK.md`). `requirements.txt`
lists the non-torch basics; the exact torch/CUDA build is documented in
the environment doc rather than pinned here.

## Documentation map

- `docs\operations\` — runbooks (daily pipeline, v16 workflow,
  GPU environment).
- `docs\` — current architecture (`CURRENT_SYSTEM_STATE.md`,
  `TRANSFORMER_CONFIG_GUIDE.md`).
- `docs\research\` — standing research conclusions
  (`RESEARCH_SCOREBOARD.md`, `RESEARCH_LINES_CLOSED.md`,
  `OPEN_WATCHLIST.md`, `METHODS.md`, D1-line conclusions).
- `docs\archive\` — historical checkpoints, sprint diaries, superseded
  plans (including `d1_1\` and the purge manifest).
- `reports\continuous_research\` — committed research evidence per
  line (v9–v16).
- `research\legacy_lstm\` — the frozen pre-transformer stack.

## Troubleshooting

- **daily_ops aborted with `ERROR: ... failed - daily pipeline
  aborted`** — an integrity gate fired (partial EOD publication, failed
  retrain/inference, stale artifacts). The previous standing plan is
  untouched. Re-run after the cause clears; `latest_nightly_status.md`
  and the console output name the failing stage.
- **`NO_NEW_SESSION_DATA`** — no newly published EOD session (weekend,
  holiday, TWSE lag, or same-session rerun). The standing plan remains
  the actionable one.
- **Morning refresh says `MARKET_DATA_NOT_READY` / `SESSION_MISMATCH`**
  — the session isn't open ≥2 minutes yet, the collector has no fresh
  rows, or the plan's intended date doesn't match today (e.g. holiday).
  Never act on a mismatched plan.
- **Collector**: check `research\intraday_cache\supervisor_<date>.log`
  (every launch/exit is recorded) and
  `python research\intraday_collector\status.py` for a DB overview.

## Tests

`.venv\Scripts\python.exe -m unittest discover -s tests` (~230 tests,
no network, no orders).
