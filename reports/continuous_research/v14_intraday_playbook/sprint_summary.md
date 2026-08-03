# v14 Sprint Summary — Precomputed Intraday Conditional Playbook

20-hour time-box; actual spend ≈ 6 h (no GPU training — correctly skipped
per the data gate). Branch `research/v14-intraday-conditional-playbook`.

## What was established

1. **No intraday data exists and none is backfillable** (data report).
   True intraday validation requires a forward collector; earliest
   decision-grade window ≈ 6 months after it starts
   (intraday_data_acquisition_plan.md).
2. **The playbook framework is built and produces output today**:
   `generate_conditional_playbook.py` emits a 560-row conditional table
   for the next session from the latest EOD book (33 book symbols + 7
   short-diagnostic names passing the v12 proxy screen), with gap-bin ×
   checkpoint rows, action labels, confidence tiers, and
   `live_trading_allowed=false` / `DAILY_BAR_PROXY_ONLY` on every row.
   Checkpoint rows are explicit placeholders pending real data.
3. **The daily-bar proxy found NO exploitable conditional entry edge.**
   Day-frame: 142,667 symbol-days (2021→2026-07). All-days open-to-close
   long baseline after 30 bps costs: **−0.39%/day, 36.5% win rate** —
   day-trading is negative by default. Of ~80 pre-registered cells, the
   only train-significant POSITIVE cells (deep ≤−5% down-gaps) failed the
   2024 validation year: **0 of 5 possible rules survived** the
   pre-registered gates. The playbook therefore ships with zero
   `validated` entry cells.
4. **The proxy DOES robustly support avoidance**: chasing +2–5% up-gaps
   loses −0.3…−0.7%/day across all EOD-action groups (|t| 2.9–5.8), and
   shorting into ≥+5% up-gaps loses −1.31%/day (t −4.7) — the
   avoid-chase / AVOID_SHORT heuristics in the playbook are empirically
   grounded even at proxy grade.
5. **Offline-GPU role is designed, not trained** (offline_gpu_strategy_
   plan.md): heavy compute belongs in nightly scenario-table precompute /
   MC path simulation / policy distillation-to-table — all gated on
   collector data; no XL until sample counts justify it (v13 lesson).

## What was NOT established (and cannot be, yet)

Any time-of-day rule, stop/TP ordering, fill/slippage realism, VWAP
behavior, or short-side mechanics. Nothing in this sprint is
decision-grade for live intraday trading, and the deliverables say so on
every row.

## Deliverables

data_availability_report.md · playbook_schema.md · target_design.md ·
daily_bar_proxy_limitations.md · intraday_data_acquisition_plan.md ·
offline_gpu_strategy_plan.md · rule_search_plan.md ·
backtest_results.{csv,md} · selected_rules.json (empty selection — a
result, not a failure) · sprint_summary.md · queue_v14_verdict.md ·
scripts: research/intraday_playbook/{generate_conditional_playbook,
backtest_conditional_playbook,search_playbook_rules}.py · sample playbook
(gitignored output dir).

## Natural next step (needs approval; NOT started)

Implement and start the intraday collector (cheap, CPU-only, compounding
asset). Revisit checkpoint rules ~6 months after it runs. The playbook
generator can meanwhile run daily in framework mode if wanted (review
tool; zero live authority).
