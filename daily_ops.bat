@echo off
REM AI-Quant daily operation cycle (mirrors docs\operations\DAILY_OPERATION_RUNBOOK.md).
REM Run from the repo root after TWSE close (~14:30+ TW time).
REM
REM FAIL-FAST (2026-08-24 incident fix): every critical step is gated.
REM GPU steps (retrain/inference) are gated on their EXPECTED dated
REM artifacts via research\pipeline_gate.py rather than ERRORLEVEL,
REM because torch teardown exit codes can be nonzero on success; the
REM artifact gate also blocks stale outputs from masquerading as fresh
REM ones. Non-GPU steps are additionally ERRORLEVEL-gated. On any gate
REM failure the pipeline ABORTS: no downstream step runs and the
REM previous standing user plan is left untouched.
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
REM Coverage gate BEFORE the GPU retrain: a partially published session
REM (2026-08-24 incident: 34/108) must not reach model training.
set "STEP=EOD refresh coverage"
.venv\Scripts\python.exe research\pipeline_gate.py refresh
if errorlevel 1 goto :pipefail

REM Step-start markers let the gate require artifact mtimes AFTER the
REM step began: a nonzero exit passes ONLY when every expected dated
REM artifact was freshly written by this very run (the documented torch
REM teardown case); any other nonzero exit, and any zero exit without
REM fresh artifacts, aborts.
set "STEP_MARKER=%TEMP%\aiquant_step_start.marker"

echo === [2/9] daily retrain ===
type nul > "%STEP_MARKER%"
.venv\Scripts\python.exe train_transformer_eod.py --mode daily-retrain
set "STEP_RC=%errorlevel%"
set "STEP=daily retrain"
.venv\Scripts\python.exe research\pipeline_gate.py retrain --exit-code %STEP_RC% --since-marker "%STEP_MARKER%"
if errorlevel 1 goto :pipefail

echo === [3/9] inference ===
type nul > "%STEP_MARKER%"
.venv\Scripts\python.exe inference_transformer_eod.py
set "STEP_RC=%errorlevel%"
set "STEP=inference"
.venv\Scripts\python.exe research\pipeline_gate.py inference --exit-code %STEP_RC% --since-marker "%STEP_MARKER%"
if errorlevel 1 goto :pipefail

echo === [4/9] blended decision book ===
set "STEP=blended decision book"
type nul > "%STEP_MARKER%"
.venv\Scripts\python.exe research\blended_decision_book.py
set "STEP_RC=%errorlevel%"
if not "%STEP_RC%"=="0" goto :pipefail
.venv\Scripts\python.exe research\pipeline_gate.py book --exit-code %STEP_RC% --since-marker "%STEP_MARKER%"
if errorlevel 1 goto :pipefail

echo === [5/9] paper snapshot ===
set "STEP=paper snapshot"
.venv\Scripts\python.exe research\paper_trading.py snapshot
if errorlevel 1 goto :pipefail

echo === [6/9] paper evaluate ===
set "STEP=paper evaluate"
.venv\Scripts\python.exe research\paper_trading.py evaluate
if errorlevel 1 goto :pipefail

echo === [7/9] daily diff report ===
set "STEP=daily diff report"
.venv\Scripts\python.exe research\daily_diff_report.py
if errorlevel 1 goto :pipefail

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
exit /b 0

:pipefail
echo.
echo ERROR: %STEP% failed - daily pipeline aborted.
echo No new decision book/user action plan generated.
echo Previous standing plan remains untouched.
endlocal
exit /b 1
