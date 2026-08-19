"""Tests for the v15 collector reliability hardening (2026-08-20):
single-instance lock + supervisor watchdog.

Deterministic: scratch lock/DB paths, stub children, injected clocks.
Never touches the production DB or network.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_collector_reliability -v
"""

import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLL = os.path.join(REPO, "research", "intraday_collector")
sys.path.insert(0, COLL)

from collect_realtime_quotes import (EXIT_ALREADY_RUNNING,  # noqa: E402
                                     acquire_single_instance_lock)
import collector_supervisor as sup  # noqa: E402

PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PY):
    PY = sys.executable

HOLDER = """
import sys, time
sys.path.insert(0, r"{coll}")
from collect_realtime_quotes import acquire_single_instance_lock
lock = acquire_single_instance_lock(r"{db}")
print("HELD" if lock else "DENIED", flush=True)
time.sleep(60)
"""


class TestSingleInstanceLock(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lock_test_")
        self.db = os.path.join(self.dir, "scratch.sqlite")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _spawn_holder(self):
        p = subprocess.Popen(
            [PY, "-c", HOLDER.format(coll=COLL, db=self.db)],
            stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline().strip()
        return p, line

    def test_second_acquire_denied_while_held(self):
        p, line = self._spawn_holder()
        try:
            self.assertEqual(line, "HELD")
            self.assertIsNone(acquire_single_instance_lock(self.db))
        finally:
            p.kill()
            p.wait()

    def test_lock_released_on_process_death(self):
        # the 08-19 failure mode: holder dies -> lock must free itself
        p, line = self._spawn_holder()
        self.assertEqual(line, "HELD")
        p.kill()
        p.wait()
        lock = None
        for _ in range(20):                 # OS release is near-immediate
            lock = acquire_single_instance_lock(self.db)
            if lock:
                break
            time.sleep(0.25)
        self.assertIsNotNone(lock)
        lock.close()

    def test_per_db_isolation(self):
        # scratch DBs never collide with each other / production
        other = os.path.join(self.dir, "other.sqlite")
        l1 = acquire_single_instance_lock(self.db)
        l2 = acquire_single_instance_lock(other)
        self.assertIsNotNone(l1)
        self.assertIsNotNone(l2)
        l1.close()
        l2.close()

    def test_collector_contention_distinct_exit_code(self):
        # end-to-end: a real second collector (mock mode) must exit with
        # EXIT_ALREADY_RUNNING (75) — not 0, not a crash code — with the
        # single-instance message, writing nothing
        p, line = self._spawn_holder()
        try:
            self.assertEqual(line, "HELD")
            r = subprocess.run(
                [PY, os.path.join(COLL, "collect_realtime_quotes.py"),
                 "--mock", "1", "--db", self.db],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, EXIT_ALREADY_RUNNING)
            self.assertEqual(EXIT_ALREADY_RUNNING, 75)
            self.assertIn("another collector instance", r.stdout)
            self.assertFalse(os.path.isfile(self.db))   # nothing written
        finally:
            p.kill()
            p.wait()

    def test_supervisor_lock_independent_of_collector_lock(self):
        c = acquire_single_instance_lock(self.db, name="collector")
        s = acquire_single_instance_lock(self.db, name="supervisor")
        self.assertIsNotNone(c)
        self.assertIsNotNone(s)
        c.close()
        s.close()


class TestSupervisor(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sup_test_")
        self.db = os.path.join(self.dir, "scratch.sqlite")
        self.log = os.path.join(self.dir, "sup.log")
        self.marker = os.path.join(self.dir, "launches.txt")
        # stub child: appends a marker line, exits rc 1 (in-session death)
        self.dying_child = [PY, "-c",
                            f"open(r'{self.marker}','a').write('x');"
                            "import sys; sys.exit(1)"]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _clock(self, *times):
        """now_fn yielding the given HH:MM datetimes then repeating last."""
        seq = [dt.datetime(2026, 8, 20, *t) for t in times]  # a Thursday

        def now_fn():
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return now_fn

    def test_restarts_until_exhausted(self):
        # child dies in-session every time -> initial + max_restarts
        n, outcome = sup.supervise(
            self.dying_child, self.db, self.log,
            dt.time(9, 0), dt.time(13, 30), max_restarts=2,
            restart_delay=0, now_fn=lambda: dt.datetime(2026, 8, 20, 10),
            sleep_fn=lambda s: None)
        self.assertEqual(n, 3)
        self.assertEqual(outcome, "MAX_RESTARTS_EXHAUSTED")
        with open(self.marker) as f:
            self.assertEqual(len(f.read()), 3)
        with open(self.log, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("DIED IN-SESSION", text)
        self.assertIn("max consecutive failures exhausted", text)

    def test_exit_after_window_is_normal(self):
        # first now() in window (loop entry + launch), then post-window
        # -> child exit is session completion, no restart
        now_fn = self._clock((10, 0), (10, 0), (13, 40))
        n, outcome = sup.supervise(
            self.dying_child, self.db, self.log,
            dt.time(9, 0), dt.time(13, 34), max_restarts=5,
            restart_delay=0, now_fn=now_fn, sleep_fn=lambda s: None)
        self.assertEqual(n, 1)
        self.assertEqual(outcome, "SESSION_COMPLETE")

    def test_outside_window_no_launch(self):
        n, outcome = sup.supervise(
            self.dying_child, self.db, self.log,
            dt.time(9, 0), dt.time(13, 34), max_restarts=5,
            restart_delay=0,
            now_fn=lambda: dt.datetime(2026, 8, 20, 15),
            sleep_fn=lambda s: None)
        self.assertEqual(n, 0)
        self.assertEqual(outcome, "OUTSIDE_WINDOW")
        self.assertFalse(os.path.isfile(self.marker))

    def test_weekend_no_launch(self):
        n, outcome = sup.supervise(
            self.dying_child, self.db, self.log,
            dt.time(9, 0), dt.time(13, 34), max_restarts=5,
            restart_delay=0,
            now_fn=lambda: dt.datetime(2026, 8, 22, 10),  # Saturday
            sleep_fn=lambda s: None)
        self.assertEqual(outcome, "OUTSIDE_WINDOW")

    def test_child_output_captured_in_log(self):
        # the 08-19 gap: a crash must leave evidence in the log
        crashing = [PY, "-c", "raise RuntimeError('boom-traceback')"]
        sup.supervise(crashing, self.db, self.log, dt.time(9, 0),
                      dt.time(13, 30), max_restarts=1, restart_delay=0,
                      now_fn=lambda: dt.datetime(2026, 8, 20, 10),
                      sleep_fn=lambda s: None)
        with open(self.log, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("boom-traceback", text)
        self.assertIn("RuntimeError", text)

    def _stepping_clock(self, start_hm=(10, 0), step_s=60):
        """now_fn advancing step_s per call from a Thursday base."""
        state = {"t": dt.datetime(2026, 8, 20, *start_hm)}

        def now_fn():
            state["t"] += dt.timedelta(seconds=step_s)
            return state["t"]
        return now_fn

    # ---- 2026-08-20 review patch: standby + consecutive budget ----

    def test_standby_does_not_consume_crash_budget(self):
        # child always exits EXIT_ALREADY_RUNNING (rogue collector owns
        # the lock). With max_restarts=3 the supervisor must keep
        # standing by far past 3 attempts, until the window closes.
        contended = [PY, "-c", "import sys; sys.exit(75)"]
        n, outcome = sup.supervise(
            contended, self.db, self.log,
            dt.time(9, 0), dt.time(10, 10), max_restarts=3,
            restart_delay=0, standby_delay=0, healthy_runtime=300,
            now_fn=self._stepping_clock(start_hm=(10, 0), step_s=10),
            sleep_fn=lambda s: None)
        self.assertEqual(outcome, "SESSION_COMPLETE")   # window end
        self.assertGreater(n, 3)                        # budget untouched
        with open(self.log, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("STANDBY", text)
        self.assertIn("crash budget not consumed", text)
        self.assertNotIn("MAX", outcome)

    def test_takeover_after_rogue_dies(self):
        # child exits 75 while a flag file exists; the flag disappears
        # after two standby waits (the rogue "dies") -> next launch runs
        # normally to the window end -> SESSION_COMPLETE, 3 launches.
        flag = os.path.join(self.dir, "rogue_alive")
        open(flag, "w").close()
        child = [PY, "-c",
                 f"import os, sys; sys.exit(75 if os.path.exists("
                 f"r'{flag}') else 0)"]
        waits = {"n": 0}

        def sleep_fn(s):
            waits["n"] += 1
            if waits["n"] == 2:
                os.remove(flag)
        # window sized so launches 1-2 (standby) stay inside and the
        # takeover child's exit lands after the window end
        clock = self._stepping_clock(start_hm=(10, 0), step_s=10)
        n, outcome = sup.supervise(
            child, self.db, self.log,
            dt.time(9, 0), dt.time(10, 2), max_restarts=2,
            restart_delay=0, standby_delay=0, healthy_runtime=300,
            now_fn=clock, sleep_fn=sleep_fn)
        self.assertEqual(outcome, "SESSION_COMPLETE")
        self.assertEqual(n, 3)          # 2 standby + 1 real takeover
        with open(self.log, encoding="utf-8") as f:
            self.assertEqual(f.read().count("STANDBY"), 2)

    def test_healthy_runtime_resets_consecutive_counter(self):
        # each child "runs" 10 min (clock steps 300s; t_start->t_end =
        # 2 calls = 600s >= healthy 300s) then dies rc=1. With
        # max_restarts=1 an unreset counter would exhaust at the 2nd
        # death; the reset keeps it supervising until the window ends.
        n, outcome = sup.supervise(
            self.dying_child, self.db, self.log,
            dt.time(9, 0), dt.time(11, 0), max_restarts=1,
            restart_delay=0, standby_delay=0, healthy_runtime=300,
            now_fn=self._stepping_clock(start_hm=(9, 30), step_s=300),
            sleep_fn=lambda s: None)
        self.assertEqual(outcome, "SESSION_COMPLETE")
        self.assertGreater(n, 2)        # survived >2 healthy-then-die
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("counter reset", f.read())

    def test_quick_crashes_still_exhaust_budget(self):
        # near-zero runtime (step 1s) -> no reset -> consecutive budget
        # exhausts exactly as before
        n, outcome = sup.supervise(
            self.dying_child, self.db, self.log,
            dt.time(9, 0), dt.time(13, 30), max_restarts=2,
            restart_delay=0, standby_delay=0, healthy_runtime=300,
            now_fn=self._stepping_clock(start_hm=(10, 0), step_s=1),
            sleep_fn=lambda s: None)
        self.assertEqual(n, 3)
        self.assertEqual(outcome, "MAX_RESTARTS_EXHAUSTED")

    def test_cli_second_supervisor_denied(self):
        s = acquire_single_instance_lock(self.db, name="supervisor")
        try:
            rc = sup.main(["--db", self.db, "--child-cmd", "unused",
                           "--window-start", "00:01",
                           "--window-end", "00:02"])
            self.assertEqual(rc, 0)
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
