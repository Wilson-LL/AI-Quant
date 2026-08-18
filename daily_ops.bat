@echo off
REM AI-Quant daily operation cycle (mirrors DAILY_OPERATION_RUNBOOK.md).
REM Run from the repo root after TWSE close (~14:30+ TW time).
REM Torch teardown exit codes can be nonzero - steps are not gated on exit
REM codes; judge each step by its logged output (see the runbook).
setlocal
cd /d "%~dp0"

echo === [1/9] refresh EOD cache ===
set "REFRESH_LOG=%TEMP%\aiquant_refresh_log.txt"
.venv\Scripts\python.exe research\refresh_data.py > "%REFRESH_LOG%" 2>&1
type "%REFRESH_LOG%"
REM Explicit EOD refresh contract for step 9 (v16 C2 patch A): the BAT
REM knows the refresh outcome, so it is passed explicitly - the Python
REM layer never infers it from files or console text.
set "EOD_REFRESH_STATUS=NEW_SESSION_DATA"
findstr /C:"+0 rows" "%REFRESH_LOG%" >nul
if %errorlevel%==0 (
    echo +0 rows: no new TWSE session published - skipping steps 2-7.
    set "EOD_REFRESH_STATUS=NO_NEW_SESSION_DATA"
    goto :overlay
)

echo === [2/9] daily retrain ===
.venv\Scripts\python.exe train_transformer_eod.py --mode daily-retrain

echo === [3/9] inference ===
.venv\Scripts\python.exe inference_transformer_eod.py

echo === [4/9] blended decision book ===
.venv\Scripts\python.exe research\blended_decision_book.py

echo === [5/9] paper snapshot ===
.venv\Scripts\python.exe research\paper_trading.py snapshot

echo === [6/9] paper evaluate ===
.venv\Scripts\python.exe research\paper_trading.py evaluate

echo === [7/9] daily diff report ===
.venv\Scripts\python.exe research\daily_diff_report.py

:overlay
echo === [8/9] user holdings overlay ===
if not exist my_holdings.csv (
    echo [8/9] user holdings overlay SKIPPED: my_holdings.csv not found. Copy my_holdings.example.csv to my_holdings.csv to enable.
    echo [9/9] Next-session user plan SKIPPED: my_holdings.csv not found.
    goto :done
)
.venv\Scripts\python.exe research\user_holdings_overlay.py --strategy blend50_band10

echo === [9/9] next-session user action plan ===
REM Nightly gates (NO_NEW_SESSION_DATA / PARTIAL_PUBLICATION_SUSPECTED /
REM STALE_BOOK) live in the Python layer - no logic duplicated here.
REM The refresh outcome is passed EXPLICITLY (patch A contract).
.venv\Scripts\python.exe research\user_next_session_plan.py --nightly --eod-refresh-status %EOD_REFRESH_STATUS%

:done
endlocal
