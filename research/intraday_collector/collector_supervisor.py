"""v15 hardening — collector supervisor (watchdog).

Motivation (two failures in one week): 2026-08-19 the scheduled
collector process died at ~09:44 with no restart and no recorded
evidence; 2026-08-18 a duplicate instance raced the scheduled one.

Observability contract (precise): child stdout/stderr are captured to a
dated log file WHENEVER EMITTED, and the supervisor always records every
child launch and every child exit with its return code — so an
unexpected termination is observable even when the child produced no
Python traceback (hard kill / interpreter crash).

State machine per child exit (rc), while inside the supervision window:
  rc == EXIT_ALREADY_RUNNING (75)  -> STANDBY: another collector owns
      the DB collector lock (the 2026-08-18 mode). Wait STANDBY_DELAY_S
      and try again. Standby NEVER consumes the crash budget and can
      repeat until the window ends — if the rogue collector dies later,
      this supervisor takes over.
  other rc                         -> unexpected in-session death:
      consecutive_failures += 1, restart after RESTART_DELAY_S. A child
      that stayed alive >= HEALTHY_RUNTIME_S before dying resets the
      consecutive counter first (the budget bounds REPEATED unhealthy
      failures, not the lifetime total across the session).
      consecutive_failures > max_restarts -> give up (crashloop guard).
Exit after the window closes -> normal session completion.

The supervisor holds its own per-DB lock so two supervisors cannot
fight. Run-ledger restart-safety unchanged (dead child's 'running' row
-> 'aborted' at next child startup). Data collection only — no orders,
no broker APIs.

Usage (the scheduled task points here as of this hardening):
  python research/intraday_collector/collector_supervisor.py
      [--universe book] [--interval 60] [--db PATH]
      [--max-restarts 10] [--restart-delay 30]
      [--standby-delay 60] [--healthy-runtime 300]
      [--window-start HH:MM] [--window-end HH:MM]   (test overrides)
      [--child-cmd "..."]                            (test stub)
"""

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from collect_realtime_quotes import (DB_DEFAULT,  # noqa: E402
                                     EXIT_ALREADY_RUNNING,
                                     acquire_single_instance_lock)

# Supervision window: slightly wider than the child's 08:55-13:35 gate so
# an early scheduled start (08:54) is supervised and a normal 13:35 child
# exit is recognized as session end, not a crash.
WINDOW_START = _dt.time(8, 50)
WINDOW_END = _dt.time(13, 34)
MAX_RESTARTS = 10          # max CONSECUTIVE unhealthy failures
RESTART_DELAY_S = 30
STANDBY_DELAY_S = 60       # retry cadence while another collector owns
#                            the lock (never consumes the crash budget)
HEALTHY_RUNTIME_S = 300    # child alive >= this before dying -> the
#                            consecutive-failure counter resets first


def in_window(now, start, end):
    return now.weekday() < 5 and start <= now.time() <= end


def supervise(child_cmd, db, log_path, start, end, max_restarts,
              restart_delay, standby_delay=STANDBY_DELAY_S,
              healthy_runtime=HEALTHY_RUNTIME_S,
              now_fn=_dt.datetime.now, sleep_fn=time.sleep):
    """Supervision loop; returns (launches, outcome). Deterministic for
    tests via now_fn/sleep_fn and window/delay overrides. Outcomes:
    OUTSIDE_WINDOW / SESSION_COMPLETE / MAX_RESTARTS_EXHAUSTED."""
    launches = 0
    consecutive_failures = 0
    log = open(log_path, "a", encoding="utf-8", buffering=1)

    def note(msg):
        log.write(f"[supervisor {now_fn().strftime('%H:%M:%S')}] "
                  f"{msg}\n")
        print(f"[supervisor] {msg}")

    try:
        if not in_window(now_fn(), start, end):
            note("outside supervision window - nothing to do")
            return 0, "OUTSIDE_WINDOW"
        while True:
            launches += 1
            note(f"launching collector (attempt {launches}): "
                 f"{' '.join(child_cmd)}")
            t_start = now_fn()
            proc = subprocess.Popen(child_cmd, stdout=log, stderr=log,
                                    cwd=ROOT)
            rc = proc.wait()
            t_end = now_fn()
            note(f"child exited rc={rc}")
            if not in_window(t_end, start, end):
                if rc == EXIT_ALREADY_RUNNING:
                    note("another collector owned the lock through "
                         "session end - standby concluded")
                else:
                    note("session window closed - normal completion")
                return launches, "SESSION_COMPLETE"
            if rc == EXIT_ALREADY_RUNNING:
                # STANDBY (2026-08-18 mode): another collector owns the
                # DB collector lock. Not a crash; the budget is
                # untouched. Keep retrying so we take over the moment
                # the other instance dies or releases.
                note("STANDBY: another collector owns the DB collector "
                     f"lock - retrying in {standby_delay:.0f}s (crash "
                     "budget not consumed)")
                sleep_fn(standby_delay)
                continue
            runtime = (t_end - t_start).total_seconds()
            if runtime >= healthy_runtime and consecutive_failures:
                note(f"child was healthy for {runtime:.0f}s before "
                     "dying - consecutive-failure counter reset")
                consecutive_failures = 0
            consecutive_failures += 1
            note(f"child DIED IN-SESSION rc={rc} after {runtime:.0f}s "
                 f"(consecutive failure {consecutive_failures}/"
                 f"{max_restarts}); any child output was captured "
                 "above; launch/exit are recorded even without a "
                 "traceback")
            if consecutive_failures > max_restarts:
                note("max consecutive failures exhausted - giving up; "
                     "manual restart required")
                return launches, "MAX_RESTARTS_EXHAUSTED"
            sleep_fn(restart_delay)
    finally:
        log.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["book", "full"],
                    default="book")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--max-restarts", type=int, default=MAX_RESTARTS)
    ap.add_argument("--restart-delay", type=float,
                    default=RESTART_DELAY_S)
    ap.add_argument("--standby-delay", type=float,
                    default=STANDBY_DELAY_S)
    ap.add_argument("--healthy-runtime", type=float,
                    default=HEALTHY_RUNTIME_S)
    ap.add_argument("--window-start", default=None,
                    help="HH:MM test override")
    ap.add_argument("--window-end", default=None,
                    help="HH:MM test override")
    ap.add_argument("--child-cmd", default=None,
                    help="test stub command (space-separated)")
    a = ap.parse_args(argv)

    lock = acquire_single_instance_lock(a.db, name="supervisor")
    if lock is None:
        print("[supervisor] another supervisor already holds the lock "
              f"for {a.db} - exiting")
        return 0

    def _t(s, default):
        if not s:
            return default
        h, m = s.split(":")
        return _dt.time(int(h), int(m))

    start = _t(a.window_start, WINDOW_START)
    end = _t(a.window_end, WINDOW_END)
    if a.child_cmd:
        child = a.child_cmd.split()
    else:
        child = [sys.executable,
                 os.path.join(HERE, "collect_realtime_quotes.py"),
                 "--universe", a.universe, "--interval",
                 str(int(a.interval)), "--db", a.db]
    log_dir = os.path.dirname(os.path.abspath(a.db))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"supervisor_{_dt.date.today().isoformat()}.log")
    launches, outcome = supervise(child, a.db, log_path, start, end,
                                  a.max_restarts, a.restart_delay,
                                  standby_delay=a.standby_delay,
                                  healthy_runtime=a.healthy_runtime)
    print(f"[supervisor] done: {launches} launch(es), {outcome} "
          f"(log: {log_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
