# Incident record — intraday collector supervisor exited mid-session (2026-09-14)

Record only. No fix in the v17 branch; proposed follow-up branch
`fix/intraday-supervisor-self-recovery`. Scheduled tasks were NOT modified.
All timestamps verified from `research/intraday_cache/supervisor_2026-09-14.log`,
`intraday.sqlite` (`collector_runs`, `intraday_quotes`) and `schtasks /query /v`.

| Item | Evidence |
|---|---|
| Task launch | Task Scheduler Last Run Time **08:54:00**; supervisor log `[supervisor 08:54:01] launching collector (attempt 1)` |
| Boundary exit + watchdog save | run 39 exited at the 08:55 gate (`outside session window … exiting`, rc 0); supervisor logged `child DIED IN-SESSION rc=0 after 59s (1/10)` and relaunched at **08:55:30** (run 40) — the same benign boundary artefact seen 08-25 |
| Last good quote | run 40: first quote 08:55:31, **last quote 10:14:33**, 1,798 quotes |
| Supervisor disappearance | no log line after the 08:55:30 launch; no `child exited` record; no process at 10:39 (0 collector-family processes) |
| Collector disappearance | run 40 left `status='running'` with no `ended_at` until the restart marked it `aborted` at 10:39:54 |
| Task Scheduler view | Status **Ready**, **Last Result −1073741510 = 0xC000013A (STATUS_CONTROL_C_EXIT)** — the scheduler recorded the supervisor process *ending* with a Ctrl-C/console-close exit status, i.e. it did not believe the task was still running |
| Windows task history | `Microsoft-Windows-TaskScheduler/Operational` returned no events for today (history appears disabled); no further exit reason available |
| Restart | manual (`collector_supervisor.py --universe book --interval 60`, hidden window) at **10:39:53** → run 41, first quote **10:39:55** |
| Data gap | **10:14:33 → 10:39:55 ≈ 25 min** of the session; run 41 then collected normally (1,446 quotes by 11:43) |
| Prior occurrence | 2026-08-25: run 22 aborted the same way (last quote 10:44:46; restart 10:50:07; run 23 completed at 13:35) |

Interpretation (not yet root-caused): STATUS_CONTROL_C_EXIT on the
supervisor is consistent with the process group receiving a console
control event (console window closed / Ctrl-C / logoff-type signal)
rather than a Python exception — which is exactly the case the current
watchdog cannot cover (it watches the child, nobody watches the
supervisor). Both events occurred ~10:15–10:45 on a session day.

Follow-up scope (separate branch): make the supervisor ignore console
control events (or run it via a wrapper that does), have the scheduled
task restart on exit with a bounded retry, add a session-time liveness
check (post-close report or the morning refresh flags a stale collector),
and enable Task Scheduler history so the exit reason is captured.
