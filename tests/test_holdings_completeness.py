"""Track A (v17): every actual position appears in BOTH user summaries.

Covers the 19 acceptance cases: nightly + live completeness for LONG /
SHORT / ETF / outside-scope / no-cache / not-in-book / not-in-WATCH /
ranked-but-unselected / missing-quote holdings; avg_cost + P&L
arithmetic; cost basis never changes the model action; no short-entry
actions; determinism; stable paths; the A6 gate itself.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))
sys.path.insert(0, os.path.join(REPO, "research", "intraday_advisor"))

import simplified_reports as sr  # noqa: E402
import user_next_session_plan as unsp  # noqa: E402
import holdings as hold  # noqa: E402

DATE = "2026-01-09"
META_N = {"intended_execution_date": "2026-01-12", "signal_date": DATE}
META_L = {"refresh_time": "2026-01-12 09:05:00", "mode": "LIVE",
          "market_data": "FRESH"}


def prow(sym, ua, side="", qty=np.nan, cost=np.nan, close=np.nan,
         model_action="", model_rank=np.nan, pri="INFO", reason="",
         **kw):
    base = {"symbol": sym, "user_action": ua, "position_side": side,
            "position_qty": qty, "avg_cost": cost, "previous_close": close,
            "model_action": model_action, "model_rank": model_rank,
            "user_action_priority": pri, "user_action_reason": reason,
            "ideal_zone_low": 98.5, "ideal_zone_high": 100.0,
            "acceptable_ceiling": 101.0, "sell_reference": 100.5,
            "cover_reference": 99.0}
    base.update(kw)
    return base


def lrow(sym, ua, side="", qty=np.nan, cost=np.nan, close=np.nan,
         live=np.nan, model_action="", model_rank=np.nan, pri="INFO",
         **kw):
    base = {"symbol": sym, "user_action": ua, "position_side": side,
            "position_qty": qty, "avg_cost": cost, "previous_close": close,
            "live_price": live, "live_price_source": "BEST_ASK",
            "quote_freshness": "FRESH", "model_action": model_action,
            "model_rank": model_rank, "user_action_priority": pri,
            "user_action_reason": "", "live_execution_state": "",
            "suggested_limit_reference": np.nan, "night_reference": np.nan,
            "signal_validity": "VALIDATED_MODEL_SIGNAL",
            "execution_quality": "NORMAL"}
    base.update(kw)
    return base


def section(md, title):
    lines = md.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith("# ") or ln.startswith("## "):
            on = ln.strip("# ").strip() == title
            continue
        if on:
            out.append(ln)
    return "\n".join(out)


HOLDINGS_MATRIX = [
    # sym, ua, side, qty, cost, close, model_action, rank   (case)
    ("1101", "HOLD_LONG", "LONG", 1000, 50.0, 55.0, "HOLD", 3),  # in book
    ("1102", "REDUCE_LONG", "LONG", 1000, 100.0, 82.0, "", np.nan),  # ranked, unselected, at a loss
    ("1103", "HOLD_SHORT", "SHORT", 500, 80.0, 70.0, "", np.nan),  # SHORT outside book
    ("0050", "NO_MODEL_OPINION", "LONG", 1000, 102.0, np.nan, "", np.nan),  # ETF / scope
    ("6669", "NO_MODEL_OPINION", "LONG", 30, 3379.0, np.nan, "", np.nan),  # no cache
    ("1104", "HOLD_LONG", "LONG", 200, 30.0, 33.0, "WATCH", 9),  # WATCH holding
    ("1105", "EXIT_LONG", "LONG", 100, 10.0, 12.0, "SELL", 40),  # not in WATCH, exit
]


class TestNightlyCompleteness(unittest.TestCase):

    def _plan(self):
        rows = [prow(s, ua, side, q, c, cl, ma, rk)
                for s, ua, side, q, c, cl, ma, rk in HOLDINGS_MATRIX]
        rows.append(prow("2222", "OPEN_LONG_NEW_SIGNAL", model_action="BUY",
                         model_rank=1))
        rows.append(prow("3333", "WATCH_LONG", model_action="WATCH",
                         model_rank=12))
        return pd.DataFrame(rows)

    # 1/3/4/5/6/11: every holdings row appears nightly, for every case
    def test_every_holding_appears_nightly(self):
        md = sr.night_summary_md(self._plan(), META_N,
                                 universe_ranks={"1102": 33, "1103": 90})
        sect = section(md, "我的實際持倉")
        for s, *_ in HOLDINGS_MATRIX:
            self.assertEqual(sect.count(f"| {s} |"), 1, s)  # exactly once
        self.assertNotIn("| 2222 |", sect)   # unowned never listed here
        # holdings section is the FIRST section
        self.assertLess(md.index("# 我的實際持倉"), md.index("## 明日買進參考"))

    # 7/8/9: avg_cost read and P&L arithmetic (LONG and SHORT)
    def test_pnl_arithmetic(self):
        md = sr.night_summary_md(self._plan(), META_N)
        sect = section(md, "我的實際持倉")
        long_row = [ln for ln in sect.splitlines() if "| 1102 |" in ln][0]
        self.assertIn("| 100 |", long_row)         # avg_cost
        self.assertIn("-18.0%", long_row)          # 82/100 - 1
        short_row = [ln for ln in sect.splitlines() if "| 1103 |" in ln][0]
        self.assertIn("+12.5%", short_row)         # short: 1 - 70/80
        c = unsp.cost_basis_context("LONG", 1000, 100.0, 82.0)
        self.assertAlmostEqual(c["unrealized_pnl_pct"], -0.18)
        self.assertAlmostEqual(c["unrealized_pnl"], -18000.0)
        c = unsp.cost_basis_context("SHORT", 500, 80.0, 70.0)
        self.assertAlmostEqual(c["unrealized_pnl_pct"], 0.125)
        self.assertAlmostEqual(c["unrealized_pnl"], 5000.0)
        c = unsp.cost_basis_context("LONG", 10, np.nan, 82.0)
        self.assertTrue(np.isnan(c["unrealized_pnl_pct"]))  # not faked

    # 12: NO_MODEL_OPINION = manual review wording, N/A rank
    def test_no_model_opinion_semantics(self):
        md = sr.night_summary_md(self._plan(), META_N)
        row = [ln for ln in section(md, "我的實際持倉").splitlines()
               if "| 0050 |" in ln][0]
        self.assertIn("模型未涵蓋", row)
        self.assertIn("N/A", row)
        self.assertIn("需要人工檢視，不代表買進/賣出訊號", row)
        self.assertNotIn("減碼", row)

    # 13: cost basis never changes the model action (A3)
    def test_cost_basis_does_not_change_action(self):
        md = sr.night_summary_md(self._plan(), META_N)
        row = [ln for ln in section(md, "我的實際持倉").splitlines()
               if "| 1102 |" in ln][0]
        self.assertIn("-18.0%", row)
        self.assertIn("減碼", row)      # REDUCE_LONG stands despite the loss
        # and the mapper has no cost input at all
        import inspect
        self.assertNotIn("avg_cost",
                         inspect.signature(hold.map_user_action).parameters)

    # 14: no short-entry actions anywhere in the vocabulary / wording
    def test_no_short_entry_actions(self):
        for bad in ("OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"):
            self.assertNotIn(bad, hold.USER_ACTIONS)
            self.assertNotIn(bad, sr.ACTION_ZH)
        md = sr.night_summary_md(self._plan(), META_N)
        for bad in ("OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"):
            self.assertNotIn(bad, md)

    # 17: deterministic
    def test_deterministic(self):
        a = sr.night_summary_md(self._plan(), META_N)
        b = sr.night_summary_md(self._plan(), META_N)
        self.assertEqual(a, b)

    # A6 gate: a holdings-file symbol absent from the plan is still
    # rendered (PLAN_MISSING) — and a genuinely missing one raises
    def test_gate_never_drops(self):
        md = sr.night_summary_md(self._plan(), META_N,
                                 holdings_symbols={"1101", "9999"})
        self.assertIn("| 9999 |", section(md, "我的實際持倉"))
        self.assertIn("夜間計畫未含", md)
        with self.assertRaises(sr.HoldingsCoverageError):
            sr.check_holdings_coverage({"1101", "7777"}, {"1101"}, "t")


class TestLiveCompleteness(unittest.TestCase):

    def _live(self):
        rows = [lrow(s, ua, side, q, c, cl, live=(cl * 1.01 if
                                                   pd.notna(cl) else np.nan),
                     model_action=ma, model_rank=rk)
                for s, ua, side, q, c, cl, ma, rk in HOLDINGS_MATRIX]
        # 10: holding with NO live quote at all
        rows.append(lrow("1106", "HOLD_LONG", "LONG", 100, 20.0, 21.0,
                         live=np.nan, live_price_source="NONE",
                         quote_freshness="MISSING", model_action="HOLD",
                         model_rank=5))
        rows.append(lrow("2222", "OPEN_LONG_NEW_SIGNAL", live=99.0,
                         model_action="BUY", model_rank=1,
                         live_execution_state="IN_IDEAL_ZONE",
                         suggested_limit_reference=99.5))
        return pd.DataFrame(rows)

    # 2/3/4/5/6/10/11: every holding appears live, incl. missing quote
    def test_every_holding_appears_live(self):
        md = sr.live_summary_md(self._live(), META_L)
        sect = section(md, "我的實際持倉")
        for s, *_ in HOLDINGS_MATRIX:
            self.assertEqual(sect.count(f"| {s} |"), 1, s)
        row = [ln for ln in sect.splitlines() if "| 1106 |" in ln][0]
        self.assertIn("（前收）", row)        # fell back to previous close
        self.assertIn("+5.0%", row)           # 21/20 - 1 on the fallback
        self.assertLess(md.index("# 我的實際持倉"), md.index("## 買進參考"))

    def test_live_gate_from_holdings_file(self):
        meta = dict(META_L, holdings_symbols={"1101", "8888"})
        md = sr.live_summary_md(self._live(), meta)
        self.assertIn("| 8888 |", section(md, "我的實際持倉"))

    def test_universe_rank_used_when_no_book_rank(self):
        meta = dict(META_L, universe_ranks={"1102": 33})
        md = sr.live_summary_md(self._live(), meta)
        row = [ln for ln in section(md, "我的實際持倉").splitlines()
               if "| 1102 |" in ln][0]
        self.assertIn("#33", row)

    def test_live_deterministic(self):
        self.assertEqual(sr.live_summary_md(self._live(), META_L),
                         sr.live_summary_md(self._live(), META_L))


class TestEndToEndNightly(unittest.TestCase):
    """Real build_plan -> write_report on a fixture root: the holdings
    file is the source of truth and every symbol reaches the summary
    (18: paths unchanged)."""

    def test_real_plan_pipeline(self):
        from test_user_next_session_plan import _cache_csv, _write
        root = tempfile.mkdtemp(prefix="hold_e2e_")
        try:
            for i, sym in enumerate(("2330", "1303", "2887")):
                _cache_csv(os.path.join(root, "research", "data_cache",
                                        f"{sym}.csv"), seed=3 + i,
                           base=100.0 * (i + 1))
            _write(os.path.join(root, "reports", "transformer_gpu",
                                f"{DATE}_predictions.csv"),
                   "stock,score,score_std\n2330,0.10,0.01\n1303,0.20,0.01\n"
                   "2887,0.30,0.01\n")
            _write(os.path.join(root, "reports", "paper_trading",
                                f"{DATE}_blend50_band10_decision_book.csv"),
                   "symbol,model_score,rank,action,target_weight,"
                   "previous_weight,weight_change,sector,confidence\n"
                   "2887,2.0,1,BUY,0.08,0.0,0.08,financials,low\n"
                   "1303,1.5,2,HOLD,0.12,0.12,0.0,materials,low\n")
            hp = os.path.join(root, "my_holdings.csv")
            _write(hp, "symbol,side,shares,avg_cost,current_price,"
                       "current_value,account,notes\n"
                       "1303,LONG,1000,150,,,a,\n"      # in book
                       "2330,LONG,100,500,,,a,\n"       # ranked, unselected
                       "6669,LONG,30,3379,,,a,\n"       # no cache, no scope
                       "2887,SHORT,100,300,,,a,\n")     # SHORT vs BUY
            plan, meta = unsp.build_plan(root, hp, DATE, use_panel=False)
            out = os.path.join(root, "reports", "user_actions")
            unsp.write_report(plan, meta, out)
            p = os.path.join(out, "latest_next_session_summary.md")
            self.assertTrue(os.path.isfile(p))
            with open(p, encoding="utf-8") as f:
                md = f.read()
            sect = section(md, "我的實際持倉")
            for s in ("1303", "2330", "6669", "2887"):
                self.assertEqual(sect.count(f"| {s} |"), 1, s)
            # cost basis context landed in the plan CSV
            csv = pd.read_csv(os.path.join(
                out, "latest_next_session_action_plan.csv"),
                dtype={"symbol": str})
            for c in ("unrealized_pnl_pct", "price_vs_cost_pct",
                      "reference_price"):
                self.assertIn(c, csv.columns)
            r = csv[csv["symbol"] == "6669"].iloc[0]
            self.assertTrue(pd.isna(r["unrealized_pnl_pct"]))  # no price
            self.assertEqual(r["user_action"], "NO_MODEL_OPINION")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
