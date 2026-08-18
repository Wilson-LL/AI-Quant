"""End-to-end tests for refresh_execution_prices.py (v16 Stage C1).

Temp plan CSV + temp SQLite mocks; covers the C1-22 scenario matrix
(session mismatch, stale plan, zone/limit/expensive/gap states, sell
states, HOLD, NO_MODEL_OPINION, shorts, illegal-domain observation, no
short creation).

Run: .venv\\Scripts\\python.exe -m unittest tests.test_refresh_execution_prices -v
"""

import datetime as dt
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research", "intraday_advisor"))
sys.path.insert(0, os.path.join(REPO, "research"))

import refresh_execution_prices as rep  # noqa: E402
from test_live_market_state import SCHEMA, row  # noqa: E402

D = "2026-08-18"
NOW = dt.datetime.fromisoformat(f"{D} 10:00:30")


def plan_row(sym, ua, **kw):
    base = {
        "symbol": sym, "signal_date": "2026-08-17",
        "intended_execution_date": D, "book_stale": False,
        "model_action": "BUY", "user_action": ua,
        "signal_freshness": "FRESH_ENTRY",
        "reference": 99.5, "ideal_zone_low": 98.5, "ideal_zone_high": 100.0,
        "acceptable_ceiling": 101.0, "do_not_chase_above": 102.5,
        "risk_review_below": 96.0,
        "sell_reference": 100.5, "ideal_sell_zone_low": 100.5,
        "ideal_sell_zone_high": 103.0, "acceptable_sell_floor": 98.5,
        "do_not_panic_sell_below": 96.5, "urgent_risk_review_below": 95.5,
        "no_action_zone_low": 98.0, "no_action_zone_high": 102.0,
        "review_below": 95.0, "review_above": 104.5,
        "cover_zone_low": 98.5, "cover_zone_high": 100.5,
        "risk_review_above": 103.0,
        "auction_reference_price": 100.0,
        "auction_reference_source": "PREVIOUS_CLOSE",
        "legal_limit_down": 90.0, "legal_limit_up": 110.0,
        "price_domain_status": "NORMAL_DAY_ASSUMPTION",
        "expected_open_p10": 98.9, "expected_open_p25": 99.4,
        "expected_open_p50": 100.0, "expected_open_p75": 100.5,
        "expected_open_p90": 101.5,
        "buy_reference_reach_probability": 0.72,
        "range_reach_confidence": "NORMAL",
    }
    base.update(kw)
    return base


class TestRefresh(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="c1_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _run(self, plan_rows, db_rows, session=D, now=NOW,
             diagnostic=False):
        plan_p = os.path.join(self.dir, "plan.csv")
        pd.DataFrame(plan_rows).to_csv(plan_p, index=False)
        db_p = os.path.join(self.dir, "mock.sqlite")
        if os.path.exists(db_p):
            os.unlink(db_p)
        con = sqlite3.connect(db_p)
        con.execute(SCHEMA)
        for r in db_rows:
            con.execute("INSERT INTO intraday_quotes VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)
        con.commit()
        con.close()
        live, meta = rep.refresh(plan_p, db_p, session, now=now,
                                 diagnostic=diagnostic)
        return live.set_index("symbol"), meta

    def _md(self, live, meta):
        out = os.path.join(self.dir, "out")
        rep.write_report(live.reset_index(), meta, out)
        with open(os.path.join(out, "latest_live_execution_plan.md"),
                  encoding="utf-8") as f:
            return f.read()

    # mock 11: session mismatch
    def test_11_session_mismatch(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", price=99.5, ask=99.5)],
            session="2026-08-19", now=dt.datetime.fromisoformat(
                "2026-08-19 10:00:30"))
        self.assertEqual(meta["mode"], "REJECTED")
        self.assertEqual(live.loc["2330", "signal_validity"],
                         "SESSION_MISMATCH")
        self.assertFalse(live["action_valid_now"].any())
        self.assertIn("LIVE_PLAN_DATE_MISMATCH",
                      self._md(live, meta))

    # mock 12: stale plan
    def test_12_stale_plan(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL", book_stale=True)],
            [row("2330", f"{D} 10:00:00", price=99.5, ask=99.5)])
        self.assertEqual(meta["mode"], "REJECTED")
        self.assertEqual(live.loc["2330", "signal_validity"],
                         "STALE_PLAN")
        self.assertFalse(live["action_valid_now"].any())

    # mock 13: ideal-zone BUY
    def test_13_ideal_zone_buy(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", price=99.5, bid=99.4,
                 ask=99.5, o=100.0)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"], "IN_IDEAL_ZONE")
        self.assertEqual(r["execution_quality"], "GOOD")
        self.assertTrue(r["action_valid_now"])
        self.assertEqual(r["live_price_source"], "BEST_ASK")
        self.assertAlmostEqual(r["suggested_limit_reference"], 99.5)
        self.assertEqual(r["suggested_limit_reason"],
                         "ASK_INSIDE_IDEAL_ZONE")
        self.assertEqual(r["signal_validity"], "VALIDATED_MODEL_SIGNAL")
        self.assertEqual(r["open_percentile_approx"], "P50_P75")
        md = self._md(live, meta)
        self.assertIn("1. ACTIONABLE NOW", md)
        self.assertIn("range reach, NOT a fill probability", md)

    # mock 14: acceptable BUY
    def test_14_acceptable_buy(self):
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", ask=100.5)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"],
                         "ABOVE_IDEAL_WITHIN_LIMIT")
        self.assertTrue(r["action_valid_now"])
        self.assertEqual(r["suggested_limit_reason"],
                         "ASK_WITHIN_ACCEPTABLE_LIMIT")
        self.assertLessEqual(r["suggested_limit_reference"], 101.0)

    # mock 15: above preferred range — expensive but signal intact
    def test_15_above_preferred_range(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", ask=103.0, o=103.0)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"],
                         "ABOVE_PREFERRED_EXECUTION_RANGE")
        self.assertEqual(r["execution_quality"], "EXPENSIVE")
        self.assertEqual(r["signal_validity"], "VALIDATED_MODEL_SIGNAL")
        self.assertTrue(r["action_valid_now"])   # expensive != invalid
        self.assertTrue(np.isnan(r["suggested_limit_reference"]))
        self.assertEqual(r["suggested_limit_reason"],
                         "CURRENT_PRICE_ABOVE_PREFERRED_RANGE")
        md = self._md(live, meta)
        self.assertIn("validated model signal remains intact", md)
        self.assertNotIn("MODEL INVALID", md)
        self.assertEqual(r["open_percentile_approx"], "ABOVE_P90")

    # mock 16: gapped below risk band
    def test_16_gapped_through_risk(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", ask=95.0, o=95.2)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"],
                         "GAPPED_THROUGH_RISK_REVIEW")
        self.assertFalse(r["action_valid_now"])
        self.assertEqual(r["suggested_limit_reason"],
                         "RISK_REVIEW_REQUIRED")
        md = self._md(live, meta)
        self.assertNotIn("bargain", md.lower())

    # mock 17: ideal SELL
    def test_17_ideal_sell(self):
        live, _ = self._run(
            [plan_row("2330", "EXIT_LONG", model_action="SELL")],
            [row("2330", f"{D} 10:00:00", bid=101.0, ask=101.5)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"], "IN_IDEAL_SELL_ZONE")
        self.assertTrue(r["action_valid_now"])
        self.assertEqual(r["live_price_source"], "BEST_BID")
        self.assertEqual(r["suggested_limit_reason"],
                         "BID_INSIDE_IDEAL_SELL_ZONE")
        self.assertAlmostEqual(r["suggested_limit_reference"], 101.0)

    # mock 18: panic SELL -> urgent review, no wait-for-rebound
    def test_18_panic_sell(self):
        live, meta = self._run(
            [plan_row("2330", "EXIT_LONG", model_action="SELL")],
            [row("2330", f"{D} 10:00:00", bid=96.0)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"], "URGENT_RISK_REVIEW")
        self.assertFalse(r["action_valid_now"])
        md = self._md(live, meta)
        self.assertNotIn("wait for a rebound", md.lower())
        self.assertNotIn("WAIT_FOR_REBOUND", md)

    # mock 19: HOLD_LONG — no manufactured trade
    def test_19_hold_long(self):
        live, _ = self._run(
            [plan_row("2330", "HOLD_LONG", model_action="HOLD")],
            [row("2330", f"{D} 10:00:00", price=100.5, bid=100.0,
                 ask=100.5)])
        r = live.loc["2330"]
        self.assertEqual(r["live_execution_state"], "NO_ACTION_IN_RANGE")
        self.assertTrue(r["action_valid_now"])
        self.assertEqual(r["suggested_limit_reason"],
                         "NO_ACTION_REQUIRED")
        self.assertTrue(np.isnan(r["suggested_limit_reference"]))

    # mock 20: NO_MODEL_OPINION
    def test_20_no_model_opinion(self):
        live, _ = self._run(
            [plan_row("0050", "NO_MODEL_OPINION", model_action="",
                      price_domain_status="UNKNOWN",
                      legal_limit_down=np.nan, legal_limit_up=np.nan)],
            [row("0050", f"{D} 10:00:00", price=106.0)])
        r = live.loc["0050"]
        self.assertEqual(r["signal_validity"], "NO_MODEL_OPINION")
        self.assertFalse(r["action_valid_now"])

    # mock 21: existing SHORT (cover uses buy hierarchy)
    def test_21_short_positions(self):
        live, meta = self._run(
            [plan_row("2330", "BUY_TO_COVER", model_action="HOLD"),
             plan_row("1303", "HOLD_SHORT", model_action="")],
            [row("2330", f"{D} 10:00:00", bid=99.0, ask=99.5),
             row("1303", f"{D} 10:00:00", price=100.0, bid=100.0,
                 ask=100.5)])
        r = live.loc["2330"]
        self.assertEqual(r["live_price_source"], "BEST_ASK")   # buy side
        self.assertIn(r["live_execution_state"],
                      ("IN_IDEAL_ZONE", "BELOW_IDEAL_ZONE"))
        md = self._md(live, meta)
        self.assertIn("NO VALIDATED OPEN-SHORT MODEL EXISTS", md)

    # mock 22 (SUPERSEDED by the C1 correctness patch): a trusted TWSE
    # quote outside a NORMAL_DAY_ASSUMPTION domain is an ASSUMPTION
    # CONFLICT (special reference may apply), never an "illegal price".
    def test_22_domain_assumption_conflict(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", price=115.0, bid=114.5,
                 ask=115.0, o=115.0)])
        r = live.loc["2330"]
        self.assertEqual(r["domain_validation_status"],
                         "PRICE_DOMAIN_ASSUMPTION_CONFLICT")
        self.assertIn("PRICE_DOMAIN_ASSUMPTION_CONFLICT", r["errors"])
        self.assertNotIn("LIVE_PRICE_DOMAIN_ERROR", r["errors"])
        self.assertFalse(r["action_valid_now"])
        # observed price preserved exactly, never clamped/rewritten
        self.assertAlmostEqual(r["live_price"], 115.0)
        self.assertAlmostEqual(r["ask"], 115.0)
        # signal validity is NOT invalidated by price
        self.assertEqual(r["signal_validity"], "VALIDATED_MODEL_SIGNAL")
        md = self._md(live, meta)
        self.assertIn("PRICE DOMAIN REVIEW REQUIRED", md)
        self.assertIn("does NOT establish that the market quote is "
                      "illegal", md)
        self.assertNotIn("illegal TWSE price", md)

    # patch test 1: inside domain -> DOMAIN_OK
    def test_22a_domain_ok(self):
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", price=99.5, bid=99.4,
                 ask=99.5, o=100.0)])
        self.assertEqual(live.loc["2330", "domain_validation_status"],
                         "DOMAIN_OK")

    # patch test 3: UNKNOWN domain -> no fabricated bound validation
    def test_22b_unknown_domain(self):
        live, _ = self._run(
            [plan_row("0050", "NO_MODEL_OPINION", model_action="",
                      price_domain_status="UNKNOWN",
                      legal_limit_down=np.nan, legal_limit_up=np.nan)],
            [row("0050", f"{D} 10:00:00", price=250.0)])   # far off ref
        r = live.loc["0050"]
        self.assertEqual(r["domain_validation_status"], "UNKNOWN_DOMAIN")
        self.assertEqual(r["errors"], "")

    # patch test 5: structurally malformed price -> DATA_VALIDATION_ERROR
    def test_22c_malformed_data(self):
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", price=100.37, bid=99.4,
                 ask=100.37)])                     # off the tick grid
        r = live.loc["2330"]
        self.assertEqual(r["domain_validation_status"],
                         "DATA_VALIDATION_ERROR")
        self.assertIn("DATA_VALIDATION_ERROR", r["errors"])
        self.assertFalse(r["action_valid_now"])

    # patch test 6: genuinely CONFIRMED domain -> hard error is reachable
    def test_22d_confirmed_domain_violation(self):
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL",
                      price_domain_status="CONFIRMED_STANDARD_LIMIT")],
            [row("2330", f"{D} 10:00:00", ask=115.0)])
        r = live.loc["2330"]
        self.assertEqual(r["domain_validation_status"],
                         "LIVE_PRICE_DOMAIN_ERROR")
        self.assertFalse(r["action_valid_now"])

    # mock 7-adjacent: stale quote gates actionability
    def test_stale_quote_gates(self):
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 09:30:00", price=99.5, ask=99.5)])
        r = live.loc["2330"]
        self.assertEqual(r["quote_freshness"], "STALE")
        self.assertFalse(r["action_valid_now"])

    # mock 8-adjacent: missing symbol
    def test_missing_symbol_gates(self):
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")], [])
        r = live.loc["2330"]
        self.assertEqual(r["quote_freshness"], "MISSING")
        self.assertFalse(r["action_valid_now"])
        self.assertEqual(r["live_price_source"], "NONE")

    # midquote proxy -> degraded wording, not actionable
    def test_midquote_degraded(self):
        live, meta = self._run(
            [plan_row("2330", "HOLD_LONG", model_action="HOLD")],
            [row("2330", f"{D} 10:00:00", price=None, bid=99.0,
                 ask=100.0)])
        r = live.loc["2330"]
        self.assertEqual(r["live_price_source"], "MIDQUOTE_STATE_PROXY")
        self.assertEqual(r["execution_reference_confidence"], "DEGRADED")
        md = self._md(live, meta)
        self.assertIn("market-state proxy", md)

    # historical diagnostic never claims live actionability
    def test_historical_diagnostic(self):
        live, meta = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", price=99.5, ask=99.5)],
            now=dt.datetime.fromisoformat("2026-08-19 01:30:00"))
        self.assertEqual(meta["mode"], "HISTORICAL_SESSION_DIAGNOSTIC")
        self.assertFalse(live["action_valid_now"].any())
        self.assertIn("HISTORICAL_SESSION_DIAGNOSTIC",
                      self._md(live, meta))

    # mock 23: no short creation anywhere
    def test_23_no_short_creation(self):
        with open(os.path.join(REPO, "research", "intraday_advisor",
                               "refresh_execution_prices.py"),
                  encoding="utf-8") as f:
            src = f.read()
        for bad in ("guaranteed fill", "expected fill",
                    "fill_probability"):
            self.assertNotIn(bad, src)
        live, _ = self._run(
            [plan_row("2330", "OPEN_LONG_NEW_SIGNAL")],
            [row("2330", f"{D} 10:00:00", ask=99.5)])
        self.assertFalse({"OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"}
                         & set(live["user_action"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
