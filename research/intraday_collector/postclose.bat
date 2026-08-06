@echo off
REM v15 post-close processing (scheduled 13:40 TW weekdays): build 1m bars
REM and the daily quality report for today's collected session.
REM Data collection pipeline only - no orders, no broker APIs.
cd /d "C:\Users\wilso\source\code\AI-Quant"
.venv\Scripts\python.exe research\intraday_collector\build_1m_bars.py
.venv\Scripts\python.exe research\intraday_collector\quality_report.py
