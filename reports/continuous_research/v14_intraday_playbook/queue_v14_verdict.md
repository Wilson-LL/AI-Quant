# Queue v14 verdict — Precomputed Intraday Conditional Playbook (20h sprint)

## PRIMARY VERDICT: DATA_MISSING_BUILD_COLLECTOR_FIRST
### Secondary: PLAYBOOK_FRAMEWORK_READY · DAILY_BAR_PROXY_ONLY_NOT_DECISION_GRADE

Sprint executed 2026-08-03 (~6 h of the 20 h box; GPU training correctly
never started — the Task-1 data gate fired).

## Findings

1. **The core research question cannot be answered yet.** No sub-daily
   data exists anywhere in the repo and TWSE minute history is not freely
   backfillable — intraday validation requires the forward collector
   (plan written; ~6 months to decision-grade after start).
2. **The framework half of the hypothesis is validated**: heavy-compute-
   before-open → lightweight-table-during-session is implemented and
   produces a real playbook today (560 rows, next session, from the EOD
   book). Live path = table lookup; no GPU, no orders, no broker.
3. **The proxy half returned a null with teeth**: across 142,667 symbol-
   days and ~80 pre-registered gap×EOD-state cells, ZERO conditional
   entry rules survived train→validation after 30/60 bps costs. Open-to-
   close day-trading on this universe is baseline-negative (−0.39%/day
   net). What IS robust is avoidance: gap-chasing and squeeze-shorting
   cells lose consistently — the playbook's conservative heuristics have
   empirical support; its entry cells have none.
4. **Task-8 gates:** unmet by construction (data insufficient; fills/
   paths unmodeled) — hence no verdict above framework-ready is available,
   exactly as the pre-registered fallback anticipated.

## Standing rules going forward

- No intraday model training (any size) until the collector has ≥3 months
  of QA-clean data; XL-class models additionally gated on sample-count
  justification (v13 lesson).
- The playbook generator may run daily as a REVIEW tool only; every row
  carries live_trading_allowed=false until true-data validation exists.
- The collector itself is NOT built or running — separate approval.

## Artifacts

See sprint_summary.md §Deliverables. Committable set = docs + 3 scripts +
backtest csv/md + selected_rules.json (~small); playbook outputs and
day-frame are gitignored.
