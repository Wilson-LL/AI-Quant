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


if __name__ == "__main__":
    unittest.main(verbosity=2)
