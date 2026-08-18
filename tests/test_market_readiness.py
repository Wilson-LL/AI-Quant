"""Tests for the C2 session-level market-readiness gate + wrapper CLI.

Deterministic clocks and temp SQLite mocks; no real DB writes, no
network, no training.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_market_readiness -v
"""

import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research", "intraday_advisor"))
sys.path.insert(0, os.path.join(REPO, "research"))

import market_readiness as mr  # noqa: E402
import refresh_execution_prices as rep  # noqa: E402
from test_live_market_state import make_db, row  # noqa: E402
from test_refresh_execution_prices import plan_row  # noqa: E402

D = "2026-08-18"


def T(hhmmss):
    return dt.datetime.fromisoformat(f"{D} {hhmmss}")


class TestReadiness(unittest.TestCase):

    def _assess(self, rows, now, intended=D):
        p = make_db(rows)
        try:
            return mr.assess(p, intended, now=now)
        finally:
            os.unlink(p)

    # C2-21.1: before 09:00
    def test_1_before_open(self):
        s, _ = self._assess([], T("08:55:00"))
        self.assertEqual(s, "WAITING_FOR_MARKET_OPEN")

    # C2-21.2: 09:00:30 still below the minimum-after-open gate
    def test_2_below_min_after_open(self):
        s, d = self._assess([row("2330", f"{D} 09:00:01")], T("09:00:30"))
        self.assertEqual(s, "WAITING_FOR_MARKET_OPEN")
        self.assertIn("120", d)

    # C2-21.3: after gate + no current rows
    def test_3_no_rows(self):
        s, _ = self._assess([], T("09:03:00"))
        self.assertEqual(s, "WAITING_FOR_MARKET_DATA")

    # C2-21.4: after gate + fresh rows
    def test_4_ready(self):
        s, _ = self._assess([row("2330", f"{D} 09:02:30")], T("09:03:00"))
        self.assertEqual(s, "MARKET_READY")

    # C2-21.5: stale rows only
    def test_5_stale_rows(self):
        s, _ = self._assess([row("2330", f"{D} 09:03:00")], T("09:30:00"))
        self.assertEqual(s, "WAITING_FOR_MARKET_DATA")

    # C2-21.6: session mismatch
    def test_6_session_mismatch(self):
        s, _ = self._assess([], T("09:03:00"), intended="2026-08-19")
        self.assertEqual(s, "SESSION_MISMATCH")

    # C2-21.7: market closed
    def test_7_market_closed(self):
        s, _ = self._assess([row("2330", f"{D} 13:30:00")], T("14:00:00"))
        self.assertEqual(s, "MARKET_CLOSED")

    def test_constants_preregistered(self):
        self.assertEqual(mr.MIN_AFTER_OPEN_SECONDS, 120)
        self.assertEqual(mr.POLL_SECONDS_DEFAULT, 15)
        self.assertEqual(mr.MAX_WAIT_SECONDS_DEFAULT, 300)


class TestWrapperCLI(unittest.TestCase):
    """refresh_execution_prices.main readiness behavior (C2-21.7-10)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="c2_cli_")
        cls.plan = os.path.join(cls.dir, "plan.csv")
        pd.DataFrame([plan_row("2330", "OPEN_LONG_NEW_SIGNAL")]).to_csv(
            cls.plan, index=False)
        cls.db = make_db([row("2330", f"{D} 09:02:30", price=99.5,
                              bid=99.4, ask=99.5)])
        cls.out = os.path.join(cls.dir, "out")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)
        os.unlink(cls.db)

    def _main(self, *extra):
        return rep.main(["--plan", self.plan, "--db", self.db,
                         "--out-dir", self.out, *extra])

    # market closed in normal mode -> refused, no actionable report
    def test_closed_refused(self):
        rc = self._main("--now", f"{D} 14:05:00")
        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(
            os.path.join(self.out, "latest_live_execution_plan.md")))

    # C2-21.9: max-wait timeout -> MARKET_DATA_NOT_READY, rc 3
    def test_wait_timeout(self):
        rc = self._main("--now", f"{D} 08:30:00", "--wait-until-ready",
                        "--max-wait-seconds", "0", "--poll-seconds", "1")
        self.assertEqual(rc, 3)

    # ready session -> normal actionable run succeeds
    def test_ready_success(self):
        rc = self._main("--now", f"{D} 09:03:00")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(
            os.path.join(self.out, "latest_live_execution_plan.md")))

    # C2-21.8: diagnostic mode bypasses readiness, labeled historical
    def test_diagnostic_bypass(self):
        rc = self._main("--now", "2026-08-19 01:30:00",
                        "--session-date", D, "--diagnostic")
        self.assertEqual(rc, 0)
        with open(os.path.join(self.out,
                               "latest_live_execution_plan.md"),
                  encoding="utf-8") as f:
            self.assertIn("HISTORICAL_SESSION_DIAGNOSTIC", f.read())

    # C2-21.11/12: one missing symbol never blocks the session
    def test_missing_symbol_row_level_only(self):
        plan2 = os.path.join(self.dir, "plan2.csv")
        pd.DataFrame([plan_row("2330", "OPEN_LONG_NEW_SIGNAL"),
                      plan_row("9999", "OPEN_LONG_NEW_SIGNAL")]).to_csv(
            plan2, index=False)
        rc = rep.main(["--plan", plan2, "--db", self.db,
                       "--out-dir", self.out, "--now", f"{D} 09:03:00"])
        self.assertEqual(rc, 0)
        live = pd.read_csv(os.path.join(
            self.out, "latest_live_execution_plan.csv"),
            dtype={"symbol": str}).set_index("symbol")
        self.assertTrue(live.loc["2330", "action_valid_now"])
        self.assertFalse(live.loc["9999", "action_valid_now"])
        self.assertEqual(live.loc["9999", "quote_freshness"], "MISSING")


if __name__ == "__main__":
    unittest.main(verbosity=2)
