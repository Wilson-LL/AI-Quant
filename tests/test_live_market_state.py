"""Unit tests for research/intraday_advisor/live_market_state.py (C1).

Deterministic temp-SQLite mocks of the audited v15 schema. READ-ONLY
behavior verified; no real DB touched.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_live_market_state -v
"""

import datetime as dt
import os
import sqlite3
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research", "intraday_advisor"))

import live_market_state as lms  # noqa: E402

D = "2026-08-18"
NOW = dt.datetime.fromisoformat(f"{D} 10:00:30")

SCHEMA = ("CREATE TABLE intraday_quotes (symbol TEXT, timestamp TEXT, "
          "price REAL, volume REAL, cumulative_volume REAL, "
          "bid_price REAL, ask_price REAL, open REAL, high REAL, "
          "low REAL, previous_close REAL, source TEXT, "
          "collected_at TEXT, run_id INTEGER, "
          "PRIMARY KEY (symbol, timestamp, source))")


def make_db(rows):
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    con = sqlite3.connect(f.name)
    con.execute(SCHEMA)
    for r in rows:
        con.execute("INSERT INTO intraday_quotes VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)
    con.commit()
    con.close()
    return f.name


def row(sym, ts, price=None, bid=None, ask=None, o=100.0,
        source="TWSE_MIS", collected=None):
    return (sym, ts, price, 10.0, 100.0, bid, ask, o, 101.0, 99.0,
            100.0, source, collected or ts.replace(" ", "T"), 1)


class TestMarketState(unittest.TestCase):

    def _state(self, rows, sym):
        p = make_db(rows)
        try:
            return lms.load_session_state(p, D, now=NOW).get(sym)
        finally:
            os.unlink(p)

    # mock 1: fresh trade
    def test_fresh_trade(self):
        st = self._state([row("2330", f"{D} 10:00:00", price=99.5,
                              bid=99.4, ask=99.5)], "2330")
        self.assertEqual(st["quote_freshness"], "FRESH")
        self.assertEqual(st["last_trade_price"], 99.5)
        self.assertEqual(lms.actionable_price(st, "buy"),
                         (99.5, "BEST_ASK"))

    # mock 2: '-' no-trade -> NULL price with live bid/ask
    def test_no_trade_tick(self):
        st = self._state([row("2330", f"{D} 10:00:00", price=None,
                              bid=99.4, ask=99.5)], "2330")
        self.assertIsNone(st["last_trade_price"])
        self.assertEqual(lms.actionable_price(st, "buy"),
                         (99.5, "BEST_ASK"))
        self.assertEqual(lms.actionable_price(st, "sell"),
                         (99.4, "BEST_BID"))

    # mock 3/4/5: bid/ask combinations
    def test_bid_only_ask_only(self):
        st = self._state([row("2330", f"{D} 10:00:00", bid=99.4)], "2330")
        self.assertEqual(lms.actionable_price(st, "sell"),
                         (99.4, "BEST_BID"))
        self.assertEqual(lms.actionable_price(st, "buy")[1], "NONE")
        st = self._state([row("2330", f"{D} 10:00:00", ask=99.6)], "2330")
        self.assertEqual(lms.actionable_price(st, "buy"),
                         (99.6, "BEST_ASK"))
        self.assertEqual(lms.actionable_price(st, "sell")[1], "NONE")

    # mock 6: midquote fallback is a STATE proxy (state side)
    def test_midquote_state_proxy(self):
        st = self._state([row("2330", f"{D} 10:00:00", price=None,
                              bid=99.0, ask=100.0)], "2330")
        p, src = lms.actionable_price(st, "state")
        self.assertEqual(src, "MIDQUOTE_STATE_PROXY")
        self.assertAlmostEqual(p, 99.5)
        # stale trade never masquerades as actionable
        st2 = self._state([row("2330", f"{D} 09:00:00", price=98.0)],
                          "2330")
        p2, src2 = lms.actionable_price(st2, "buy")
        self.assertEqual(src2, "STALE_TRADE_STATE_PROXY")

    # mock 7: stale quote
    def test_stale_quote(self):
        st = self._state([row("2330", f"{D} 09:30:00", price=99.0,
                              bid=99.0, ask=99.1)], "2330")
        self.assertEqual(st["quote_freshness"], "STALE")
        self.assertGreater(st["quote_age_seconds"], lms.AGING_S)

    def test_aging_quote(self):
        st = self._state([row("2330", f"{D} 09:57:00", price=99.0)],
                         "2330")
        self.assertEqual(st["quote_freshness"], "AGING")

    # mock 8: missing symbol
    def test_missing_symbol(self):
        st = self._state([row("2330", f"{D} 10:00:00")], "9999")
        self.assertIsNone(st)

    # mock 9: duplicate latest rows across sources -> only TWSE_MIS used
    def test_duplicate_rows_other_source_ignored(self):
        st = self._state([row("2330", f"{D} 10:00:00", price=99.5),
                          row("2330", f"{D} 10:00:00", price=1.0,
                              source="MOCK")], "2330")
        self.assertEqual(st["last_trade_price"], 99.5)

    # mock 10: exchange ts older than collected_at -> age from exchange ts
    def test_age_uses_exchange_timestamp(self):
        st = self._state([row("2330", f"{D} 09:58:00",
                              collected=f"{D}T10:00:29")], "2330")
        self.assertGreater(st["quote_age_seconds"],
                           st["collected_age_seconds"])
        self.assertEqual(st["quote_freshness"], "AGING")

    def test_latest_trade_vs_latest_quote(self):
        # trade at 09:59, later quote row with no trade: last trade kept
        st = self._state([row("2330", f"{D} 09:59:00", price=99.0),
                          row("2330", f"{D} 10:00:00", price=None,
                              bid=99.1, ask=99.2)], "2330")
        self.assertEqual(st["exchange_ts"], f"{D} 10:00:00")
        self.assertEqual(st["last_trade_price"], 99.0)
        self.assertEqual(st["last_trade_ts"], f"{D} 09:59:00")

    def test_thresholds_preregistered(self):
        self.assertEqual(lms.FRESH_S, 120)
        self.assertEqual(lms.AGING_S, 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
