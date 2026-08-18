@echo off
REM v16 Stage C2 - MANUAL morning live execution refresh.
REM Run after ~09:02 on a trading morning (the v15 collector starts at
REM 08:54 via its own scheduled task; this wrapper is NOT scheduled -
REM one real-session manual smoke is required before any scheduling
REM decision).
REM
REM Reads the nightly plan + the intraday DB, waits briefly for the
REM session-level market-readiness gate, writes the live report, and
REM prints a short summary. It NEVER places orders, NEVER retrains,
REM NEVER re-runs inference.
setlocal
cd /d "%~dp0"

if not exist reports\user_actions\latest_next_session_action_plan.csv (
    echo No nightly plan found: reports\user_actions\latest_next_session_action_plan.csv
    echo Run daily_ops.bat the evening before. Nothing to refresh.
    goto :done
)

.venv\Scripts\python.exe research\intraday_advisor\refresh_execution_prices.py --wait-until-ready

echo.
echo Main report: reports\user_actions\latest_live_execution_plan.md
echo Manual review only - no orders are placed by this tool.

:done
endlocal
