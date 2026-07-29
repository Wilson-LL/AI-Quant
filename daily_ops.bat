@echo off
REM AI-Quant daily operation cycle (mirrors DAILY_OPERATION_RUNBOOK.md).
REM Run from the repo root after TWSE close (~14:30+ TW time).
REM Torch teardown exit codes can be nonzero - steps are not gated on exit
REM codes; judge each step by its logged output (see the runbook).
setlocal
cd /d "%~dp0"

echo === [1/8] refresh EOD cache ===
set "REFRESH_LOG=%TEMP%\aiquant_refresh_log.txt"
.venv\Scripts\python.exe research\refresh_data.py > "%REFRESH_LOG%" 2>&1
type "%REFRESH_LOG%"
findstr /C:"+0 rows" "%REFRESH_LOG%" >nul
if %errorlevel%==0 (
    echo +0 rows: no new TWSE session published - skipping steps 2-7.
    goto :overlay
)

echo === [2/8] daily retrain ===
.venv\Scripts\python.exe train_transformer_eod.py --mode daily-retrain

echo === [3/8] inference ===
.venv\Scripts\python.exe inference_transformer_eod.py

echo === [4/8] blended decision book ===
.venv\Scripts\python.exe research\blended_decision_book.py

echo === [5/8] paper snapshot ===
.venv\Scripts\python.exe research\paper_trading.py snapshot

echo === [6/8] paper evaluate ===
.venv\Scripts\python.exe research\paper_trading.py evaluate

echo === [7/8] daily diff report ===
.venv\Scripts\python.exe research\daily_diff_report.py

:overlay
echo === [8/8] user holdings overlay ===
if not exist my_holdings.csv (
    echo my_holdings.csv not found. Skipping user holdings overlay. Copy my_holdings.example.csv to my_holdings.csv to enable.
    goto :done
)
.venv\Scripts\python.exe research\user_holdings_overlay.py --strategy blend50_band10

:done
endlocal
