# -*- coding: utf-8 -*-
"""Tests for research/simplified_reports.py (compact zh-TW summaries).

Deterministic synthetic frames; presentation-only assertions.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_simplified_reports -v
"""

import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))

import simplified_reports as sr  # noqa: E402

BANNED = ("P10", "P25", "P50", "P75", "P90", "range-reach", "reach",
          "confidence", "PRICE_DOMAIN_ASSUMPTION_CONFLICT",
          "MIDQUOTE_STATE_PROXY", "STALE_TRADE_STATE_PROXY",
          "DATA_VALIDATION_ERROR", "LIVE_PRICE_DOMAIN_ERROR",
          "OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT", "guaranteed")


def live_row(sym, ua, state="", src="BEST_ASK", price=100.0,
             fresh="FRESH", sug=99.5, **kw):
    base = {"symbol": sym, "user_action": ua,
            "live_execution_state": state, "live_price_source": src,
            "live_price": price, "quote_freshness": fresh,
            "suggested_limit_reference": sug,
            "signal_validity": "VALIDATED_MODEL_SIGNAL",
            "domain_validation_status": "DOMAIN_OK",
            "night_reference": 99.0,
            "night_above_preferred_range": 102.5, "errors": ""}
    base.update(kw)
    return base


LIVE_META = {"session_date": "2026-08-19", "mode": "LIVE",
             "market_data": "FRESH",
             "refresh_time": "2026-08-19 09:03:12"}


def plan_row(sym, ua, **kw):
    base = {"symbol": sym, "user_action": ua, "ideal_zone_low": 98.5,
            "ideal_zone_high": 100.0, "acceptable_ceiling": 101.0,
            "do_not_chase_above": 102.5, "reference": 99.5,
            "sell_reference": 100.5, "cover_reference": 99.0}
    base.update(kw)
    return base


PLAN_META = {"signal_date": "2026-08-18",
             "intended_execution_date": "2026-08-19"}


def section(md, title):
    m = re.split(r"^## ", md, flags=re.M)
    for part in m[1:]:
        if part.startswith(title):
            return part
    return ""


class TestLiveSummary(unittest.TestCase):

    def _md(self, rows, meta=LIVE_META):
        return sr.live_summary_md(pd.DataFrame(rows), meta)

    # genuine entries in 買進參考, WATCH never
    def test_buy_section_membership(self):
        md = self._md([live_row("3037", "OPEN_LONG_NEW_SIGNAL",
                                "IN_IDEAL_ZONE", price=1135, sug=1135),
                       live_row("1303", "OPEN_LONG_EXISTING_TARGET",
                                "BELOW_IDEAL_ZONE", price=192, sug=192),
                       live_row("9910", "ADD_LONG",
                                "ABOVE_IDEAL_WITHIN_LIMIT"),
                       live_row("6414", "WATCH_LONG", "IN_IDEAL_ZONE")])
        buy = section(md, "買進參考")
        for s in ("3037", "1303", "9910"):
            self.assertIn(s, buy)
        self.assertNotIn("6414", buy)
        self.assertIn("6414", section(md, "觀察"))
        self.assertIn("NEW", buy)              # fresh-signal marker
        # compact statuses
        self.assertIn("可考慮", buy)
        self.assertIn("尚可", buy)
        self.assertIn("觀察名單尚非正式買進訊號", md)

    # expensive genuine entry only in 價格偏高
    def test_expensive_only_there(self):
        md = self._md([live_row("2351", "OPEN_LONG_EXISTING_TARGET",
                                "ABOVE_PREFERRED_EXECUTION_RANGE",
                                price=186.5, sug=np.nan,
                                night_reference=176.0)])
        self.assertNotIn("2351", section(md, "買進參考"))
        exp = section(md, "價格偏高")
        self.assertIn("2351", exp)
        self.assertIn("176", exp)
        self.assertIn("不代表模型訊號失效", exp)

    # REDUCE/EXIT only in 減碼 / 賣出參考
    def test_sell_section(self):
        md = self._md([live_row("2330", "REDUCE_LONG",
                                "IN_IDEAL_SELL_ZONE", src="BEST_BID",
                                price=2400, sug=2405),
                       live_row("2412", "EXIT_LONG",
                                "ABOVE_IDEAL_SELL_ZONE", src="BEST_BID")])
        sell = section(md, "減碼 / 賣出參考")
        self.assertIn("2330", sell)
        self.assertIn("減碼", sell)
        self.assertIn("賣出", sell)
        self.assertNotIn("2330", section(md, "買進參考"))

    # missing sell quote -> 注意, no fabricated number
    def test_missing_sell_quote(self):
        md = self._md([live_row("2330", "REDUCE_LONG", "", src="NONE",
                                price=np.nan, fresh="MISSING",
                                sug=np.nan)])
        self.assertEqual(section(md, "減碼 / 賣出參考"), "")
        att = section(md, "注意")
        self.assertIn("2330", att)
        self.assertIn("即時報價不足", att)

    # risk/data gates -> 注意 with simple wording, no internal codes
    def test_attention_wording(self):
        md = self._md([live_row("3333", "OPEN_LONG_NEW_SIGNAL",
                                "GAPPED_THROUGH_RISK_REVIEW"),
                       live_row("4444", "OPEN_LONG_NEW_SIGNAL",
                                "IN_IDEAL_ZONE",
                                domain_validation_status=
                                "PRICE_DOMAIN_ASSUMPTION_CONFLICT"),
                       live_row("0050", "NO_MODEL_OPINION", "",
                                src="NONE", price=np.nan, sug=np.nan,
                                fresh="STALE")])
        att = section(md, "注意")
        self.assertIn("跌破風險區", att)
        self.assertIn("價格基準可能特殊", att)
        self.assertIn("0050：模型未涵蓋", att)

    # one stock = one primary section
    def test_single_section_per_stock(self):
        rows = [live_row("1111", "OPEN_LONG_NEW_SIGNAL",
                         "ABOVE_PREFERRED_EXECUTION_RANGE"),
                live_row("2222", "REDUCE_LONG", "IN_IDEAL_SELL_ZONE",
                         src="BEST_BID"),
                live_row("3333", "WATCH_LONG", "IN_IDEAL_ZONE"),
                live_row("4444", "OPEN_LONG_NEW_SIGNAL",
                         "IN_IDEAL_ZONE")]
        md = self._md(rows)
        for sym in ("1111", "2222", "3333", "4444"):
            hits = sum(sym in sec for sec in
                       re.split(r"^## ", md, flags=re.M)[1:])
            self.assertEqual(hits, 1, sym)

    # no technical internals ever leak into the simple report
    def test_no_internal_wording(self):
        md = self._md([live_row("1111", "OPEN_LONG_NEW_SIGNAL",
                                "IN_IDEAL_ZONE"),
                       live_row("2222", "REDUCE_LONG", "", src="NONE",
                                price=np.nan, fresh="MISSING",
                                sug=np.nan),
                       live_row("3333", "WATCH_LONG",
                                "ABOVE_ACCEPTABLE_LIMIT")])
        for bad in BANNED:
            self.assertNotIn(bad, md, bad)
        self.assertIn("不保證成交", md)
        self.assertIn("系統不會自動下單", md)
        self.assertIn("latest_live_execution_plan.md", md)

    # compactness on an ordinary day
    def test_compact_line_count(self):
        rows = ([live_row(f"1{i:03d}", "OPEN_LONG_EXISTING_TARGET",
                          "IN_IDEAL_ZONE") for i in range(5)]
                + [live_row("2330", "REDUCE_LONG", "IN_IDEAL_SELL_ZONE",
                            src="BEST_BID")]
                + [live_row(f"2{i:03d}", "WATCH_LONG", "IN_IDEAL_ZONE")
                   for i in range(6)]
                + [live_row("0050", "NO_MODEL_OPINION", "", src="NONE",
                            price=np.nan, sug=np.nan, fresh="STALE")])
        md = self._md(rows)
        self.assertLessEqual(len(md.splitlines()), 35)


class TestNightSummary(unittest.TestCase):

    def _md(self, rows):
        return sr.night_summary_md(pd.DataFrame(rows), PLAN_META)

    def test_structure(self):
        md = self._md([plan_row("1111", "OPEN_LONG_NEW_SIGNAL"),
                       plan_row("2222", "OPEN_LONG_EXISTING_TARGET"),
                       plan_row("3333", "ADD_LONG"),
                       plan_row("4444", "WATCH_LONG"),
                       plan_row("5555", "REDUCE_LONG"),
                       plan_row("6666", "EXIT_LONG"),
                       plan_row("7777", "BUY_TO_COVER"),
                       plan_row("0050", "NO_MODEL_OPINION")])
        buy = section(md, "明日買進參考")
        for s in ("1111", "2222", "3333"):
            self.assertIn(s, buy)
        self.assertNotIn("4444", buy)
        self.assertIn("98.5–100", buy)
        self.assertIn("4444", section(md, "明日觀察"))
        sell = section(md, "明日減碼 / 賣出參考")
        self.assertIn("5555", sell)
        self.assertIn("空單回補", sell)
        self.assertIn("0050：模型未涵蓋", section(md, "注意"))
        for bad in BANNED:
            self.assertNotIn(bad, md, bad)
        self.assertIn("latest_next_session_action_plan.md", md)

    def test_missing_sell_reference(self):
        md = self._md([plan_row("8888", "EXIT_LONG",
                                sell_reference=np.nan)])
        self.assertIn("| — |", section(md, "明日減碼 / 賣出參考"))

    def test_compact(self):
        md = self._md([plan_row(f"1{i:03d}", "OPEN_LONG_EXISTING_TARGET")
                       for i in range(6)]
                      + [plan_row(f"2{i:03d}", "WATCH_LONG")
                         for i in range(8)])
        self.assertLessEqual(len(md.splitlines()), 30)

    def test_gitignored_paths(self):
        with open(os.path.join(REPO, ".gitignore"),
                  encoding="utf-8") as f:
            self.assertIn("reports/user_actions/", f.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
