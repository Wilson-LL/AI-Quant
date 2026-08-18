"""Unit tests for research/user_holdings_overlay.py.

Builds a synthetic repo tree in a temp dir (fake data_cache, predictions,
decision book, holdings file) — touches nothing in the real repo.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_user_holdings_overlay -v
"""

import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))

import user_holdings_overlay as uho  # noqa: E402

DATE = "2026-07-28"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TestOverlay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="uho_test_")
        r = cls.root
        # fake data_cache (2330 has a duplicate last date -> dedupe keep last)
        _write(os.path.join(r, "research", "data_cache", "2330.csv"),
               "date,open,high,low,close,volume\n"
               "2026-07-27,490,510,485,495,1000\n"
               "2026-07-28,495,505,490,999,1000\n"
               "2026-07-28,495,505,490,500,1000\n")
        for sym, px in (("1303", 100.0), ("0050", 50.0), ("2887", 40.0)):
            _write(os.path.join(r, "research", "data_cache", f"{sym}.csv"),
                   "date,open,high,low,close,volume\n"
                   f"2026-07-28,{px},{px},{px},{px},1000\n")
        # 2412: cache file exists but is unreadable (wrong columns) -> no price
        _write(os.path.join(r, "research", "data_cache", "2412.csv"),
               "garbage_a,garbage_b\n1,2\n")
        # model universe = latest predictions
        _write(os.path.join(r, "reports", "transformer_gpu",
                            f"{DATE}_predictions.csv"),
               "stock,score,score_std\n2330,0.1,0.01\n1303,0.2,0.01\n"
               "2887,0.3,0.01\n2412,0.15,0.01\n")
        # decision book: 1303 HOLD (target 12%), 2887 BUY, 2412 HOLD;
        # 2330 in universe but NOT in the book
        _write(os.path.join(r, "reports", "paper_trading",
                            f"{DATE}_blend50_band10_decision_book.csv"),
               "symbol,model_score,rank,action,target_weight,previous_weight,"
               "weight_change,sector,confidence\n"
               "1303,3.5,1,HOLD,0.12,0.12,0.0,materials,low\n"
               "2887,0.5,20,BUY,0.08,0.0,0.08,financials,low\n"
               "2412,0.4,21,HOLD,0.05,0.05,0.0,telecom,low\n")
        # holdings: total known value = 79,000
        cls.holdings = os.path.join(r, "my_holdings.csv")
        _write(cls.holdings,
               "symbol,shares,avg_cost,current_price,current_value,account,notes\n"
               "2330,100,450,,,CTBC,\n"          # 50,000 (63%) not selected
               "1303,100,90,,,CTBC,\n"           # 10,000 (12.7%) ~ target 12%
               "2887,100,38,,,CTBC,\n"           # 4,000  (5.1%)  BUY
               "0050,100,,,,long_term,etf\n"     # 5,000  not in universe
               "9910,50,55,,10000,short,\n"      # 10,000 not in data_cache
               "2412,100,,,,CTBC,no price\n")    # value unknown

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run(self):
        rc = uho.run(self.root, self.holdings, "blend50_band10", None,
                     medium_gap=0.02, large_gap=0.05)
        self.assertEqual(rc, 0)
        p = os.path.join(self.root, "reports", "user_holdings",
                         "latest_user_holdings_overlay.csv")
        return pd.read_csv(p, dtype={"symbol": str}).set_index("symbol")

    def test_1_missing_holdings_exits_gracefully(self):
        rc = uho.run(self.root, os.path.join(self.root, "nope.csv"),
                     "blend50_band10", None, 0.02, 0.05)
        self.assertEqual(rc, 0)

    def test_2_in_book_classifications(self):
        ov = self._run()
        # 1303: HOLD, my 12.66% vs target 12% -> within medium gap
        self.assertEqual(ov.loc["1303", "classification"], "MATCH_TARGET")
        self.assertEqual(ov.loc["1303", "model_action"], "HOLD")
        # 2887: BUY row is classified MODEL_BUY
        self.assertEqual(ov.loc["2887", "classification"], "MODEL_BUY")
        # 2330: in universe, not in book
        self.assertEqual(ov.loc["2330", "classification"],
                         "IN_UNIVERSE_NOT_SELECTED")
        # dedupe of duplicate cache dates: last close 500 -> value 50,000
        self.assertAlmostEqual(ov.loc["2330", "latest_price"], 500.0)
        self.assertAlmostEqual(ov.loc["2330", "my_current_value"], 50000.0)

    def test_3_not_in_universe(self):
        ov = self._run()
        self.assertEqual(ov.loc["0050", "classification"],
                         "NOT_IN_MODEL_UNIVERSE")
        self.assertEqual(ov.loc["9910", "classification"], "NOT_IN_DATA_CACHE")
        # 9910 is 12.7% of the portfolio with no coverage -> HIGH
        self.assertEqual(ov.loc["9910", "review_priority"], "HIGH")

    def test_4_missing_price_no_crash(self):
        ov = self._run()
        self.assertEqual(ov.loc["2412", "classification"], "PRICE_MISSING")
        self.assertTrue(pd.isna(ov.loc["2412", "latest_price"]))
        self.assertTrue(pd.isna(ov.loc["2412", "my_current_weight"]))

    def test_5_same_day_rerun_no_duplicates(self):
        ov1 = self._run()
        ov2 = self._run()
        self.assertEqual(len(ov1), 6)
        self.assertEqual(len(ov2), 6)
        self.assertFalse(ov2.index.duplicated().any())
        dated = os.path.join(self.root, "reports", "user_holdings",
                             f"{DATE}_user_holdings_overlay.csv")
        self.assertTrue(os.path.isfile(dated))

    def test_6_leading_zeros_preserved(self):
        ov = self._run()
        self.assertIn("0050", ov.index)

    def test_7_unknown_strategy_lists_available(self):
        rc = uho.run(self.root, self.holdings, "d7b", None, 0.02, 0.05)
        self.assertEqual(rc, 0)  # graceful, no crash


class TestStageA(unittest.TestCase):
    """v16 Stage A end-to-end: LONG/SHORT schema, denominator safety,
    duplicate-lot aggregation, conflicts, user_action wiring."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="uho_stageA_")
        r = cls.root
        for sym, px in (("2330", 500.0), ("1303", 100.0), ("2887", 40.0),
                        ("0050", 50.0)):
            _write(os.path.join(r, "research", "data_cache", f"{sym}.csv"),
                   "date,open,high,low,close,volume\n"
                   f"2026-07-28,{px},{px},{px},{px},1000\n")
        _write(os.path.join(r, "reports", "transformer_gpu",
                            f"{DATE}_predictions.csv"),
               "stock,score,score_std\n2330,0.1,0.01\n1303,0.2,0.01\n"
               "2887,0.3,0.01\n2412,0.15,0.01\n")
        _write(os.path.join(r, "reports", "paper_trading",
                            f"{DATE}_blend50_band10_decision_book.csv"),
               "symbol,model_score,rank,action,target_weight,previous_weight,"
               "weight_change,sector,confidence\n"
               "1303,3.5,1,HOLD,0.12,0.12,0.0,materials,low\n"
               "2887,0.5,20,BUY,0.08,0.0,0.08,financials,low\n"
               "2412,0.4,21,HOLD,0.05,0.05,0.0,telecom,low\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run_file(self, text, expect_rc=0):
        p = os.path.join(self.root, "holdings_case.csv")
        _write(p, text)
        rc = uho.run(self.root, p, "blend50_band10", None,
                     medium_gap=0.02, large_gap=0.05)
        self.assertEqual(rc, expect_rc)
        if expect_rc != 0:
            return None
        out = os.path.join(self.root, "reports", "user_holdings",
                           "latest_user_holdings_overlay.csv")
        return pd.read_csv(out, dtype={"symbol": str})

    def test_a_legacy_negative_shares_denominator_safety(self):
        # 2330 LONG 100 (50,000) + 1303 legacy -100 (SHORT 10,000).
        ov = self._run_file("symbol,shares\n2330,100\n1303,-100\n")
        ov = ov.set_index("symbol")
        # SHORT normalized, never negative qty
        self.assertEqual(ov.loc["1303", "position_side"], "SHORT")
        self.assertEqual(ov.loc["1303", "position_qty"], 100.0)
        self.assertAlmostEqual(ov.loc["1303", "signed_exposure_value"],
                               -10000.0)
        # REGRESSION: gross denominator (60,000), not net (40,000) — the
        # old code gave 2330 weight 50,000/40,000 = 125%
        self.assertAlmostEqual(ov.loc["2330", "my_current_weight"],
                               50000.0 / 60000.0, places=4)
        self.assertTrue((ov["my_current_weight"].dropna() <= 1.0).all())
        # long-target comparison uses gross LONG only
        self.assertAlmostEqual(ov.loc["2330", "my_long_cmp_weight"], 1.0)
        # no silent NaN propagation: valued rows all have weights
        self.assertFalse(ov["my_current_weight"].isna().any())
        # SHORT vs bullish book HOLD (target 12%) -> cover review, not
        # "underweight long"
        self.assertEqual(ov.loc["1303", "user_action"], "BUY_TO_COVER")
        self.assertEqual(ov.loc["1303", "user_action_priority"], "HIGH")
        self.assertNotEqual(ov.loc["1303", "classification"],
                            "UNDERWEIGHT_VS_MODEL")

    def test_b_all_short_portfolio_no_nan_blankout(self):
        # REGRESSION: old code -> total <= 0 -> every weight silently NaN.
        ov = self._run_file("symbol,shares\n1303,-100\n").set_index("symbol")
        self.assertAlmostEqual(ov.loc["1303", "my_current_weight"], 1.0)
        self.assertEqual(ov.loc["1303", "position_side"], "SHORT")

    def test_c_new_schema_short_vs_buy(self):
        ov = self._run_file(
            "symbol,side,shares,avg_cost,current_price,current_value,"
            "account,notes\n"
            "2330,LONG,100,450,,,CTBC,\n"
            "2887,SHORT,100,38,,,CTBC,short test\n").set_index("symbol")
        self.assertEqual(ov.loc["2887", "position_side"], "SHORT")
        self.assertEqual(ov.loc["2887", "classification"],
                         "ACTUAL_SHORT_POSITION")
        self.assertEqual(ov.loc["2887", "user_action"], "BUY_TO_COVER")
        # short never enters the long-comparison denominator
        self.assertAlmostEqual(ov.loc["2330", "my_long_cmp_weight"], 1.0)
        self.assertTrue(pd.isna(ov.loc["2887", "my_long_cmp_weight"]))
        self.assertTrue(pd.isna(ov.loc["2887", "weight_gap"]))

    def test_d_duplicate_lots_aggregate_sign_flip_regression(self):
        # Two 1303 lots (50+50 @100 = 10,000) + 2330 (50,000):
        # combined 1303 = 16.7% vs target 12% -> OVERWEIGHT. Per-lot the
        # old code said 8.3% vs 12% -> UNDERWEIGHT on both rows (sign flip).
        ov = self._run_file("symbol,side,shares,avg_cost,current_price,"
                            "current_value,account,notes\n"
                            "2330,LONG,100,,,,a,\n"
                            "1303,LONG,50,90,,,a,\n"
                            "1303,LONG,50,110,,,b,\n")
        self.assertEqual(int((ov["symbol"] == "1303").sum()), 1)  # one row
        r = ov.set_index("symbol").loc["1303"]
        self.assertEqual(r["position_qty"], 100.0)
        self.assertAlmostEqual(r["my_avg_cost"], 100.0)   # qty-weighted
        self.assertEqual(r["classification"], "OVERWEIGHT_VS_MODEL")
        self.assertEqual(r["user_action"], "REDUCE_LONG")

    def test_e_long_short_conflict_not_netted(self):
        ov = self._run_file("symbol,side,shares\n"
                            "2887,LONG,100\n2887,SHORT,40\n")
        rows = ov[ov["symbol"] == "2887"]
        self.assertEqual(len(rows), 2)                      # NOT netted
        self.assertTrue((rows["classification"] == "POSITION_CONFLICT").all())
        self.assertTrue(
            (rows["user_action"] == "POSITION_CONFLICT_REVIEW").all())
        self.assertTrue((rows["user_action_priority"] == "HIGH").all())

    def test_f_validation_errors_rc2(self):
        self._run_file("symbol,side,shares\n2330,LNG,100\n", expect_rc=2)
        self._run_file("symbol,side,shares\n2330,SHORT,-100\n", expect_rc=2)

    def test_g_zero_shares_dropped(self):
        ov = self._run_file("symbol,shares\n2330,100\n2887,0\n")
        self.assertNotIn("2887", set(ov["symbol"]))

    def test_h_outside_universe_short_no_model_opinion(self):
        ov = self._run_file("symbol,side,shares\n"
                            "2330,LONG,100\n0050,SHORT,100\n"
                            ).set_index("symbol")
        self.assertEqual(ov.loc["0050", "user_action"], "NO_MODEL_OPINION")

    def test_i_no_generic_hold_output(self):
        # the bare label HOLD must never appear as a user_action
        for text in ("symbol,shares\n2330,100\n1303,100\n",
                     "symbol,side,shares\n1303,SHORT,10\n"):
            ov = self._run_file(text)
            self.assertNotIn("HOLD", set(ov["user_action"]))
            self.assertNotIn("SELL", set(ov["user_action"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
