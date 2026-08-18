"""Unit tests for research/holdings.py (v16 Stage A).

Covers the Stage A test matrix: schema parsing/normalization (legacy and
side-based), validation errors, duplicate-lot aggregation, LONG+SHORT
conflicts, exposure conventions, and the complete model-action ->
user-action mapping including short-position and no-position branches.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_holdings -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))

import holdings as hold  # noqa: E402


def _tmp_csv(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                    encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


def _load(text):
    p = _tmp_csv(text)
    try:
        return hold.load_lots(p)
    finally:
        os.unlink(p)


class TestSchema(unittest.TestCase):

    # -- matrix 1: legacy positive shares
    def test_legacy_positive(self):
        lots, w = _load("symbol,shares,avg_cost\n2330,100,500\n")
        self.assertEqual(lots.loc[0, "position_side"], "LONG")
        self.assertEqual(lots.loc[0, "position_qty"], 100.0)
        self.assertTrue(lots.loc[0, "legacy_schema"])

    # -- matrix 2: legacy negative shares -> SHORT with positive qty
    def test_legacy_negative(self):
        lots, w = _load("symbol,shares\n2330,100\n1303,-200\n")
        r = lots[lots["symbol"] == "1303"].iloc[0]
        self.assertEqual(r["position_side"], "SHORT")
        self.assertEqual(r["position_qty"], 200.0)   # normalized positive
        self.assertTrue(any("negative-share" in x for x in w))
        # negative shares must never survive normalization
        self.assertTrue((lots["position_qty"].dropna() >= 0).all())

    # -- matrix 3/4: new schema LONG and SHORT
    def test_new_schema(self):
        lots, w = _load("symbol,side,shares,avg_cost\n"
                        "2330,LONG,200,2356.25\n2376,SHORT,1000,350\n")
        self.assertEqual(list(lots["position_side"]), ["LONG", "SHORT"])
        self.assertTrue((lots["position_qty"] > 0).all())
        self.assertFalse(lots.loc[0, "legacy_schema"])

    # -- matrix 5: invalid side value
    def test_invalid_side(self):
        with self.assertRaises(hold.HoldingsError):
            _load("symbol,side,shares\n2330,LNG,100\n")

    # -- matrix 6: explicit side + negative shares = hard error, no repair
    def test_side_sign_contradiction(self):
        for side in ("LONG", "SHORT"):
            with self.assertRaises(hold.HoldingsError):
                _load(f"symbol,side,shares\n2330,{side},-100\n")

    def test_blank_side_in_new_schema(self):
        with self.assertRaises(hold.HoldingsError):
            _load("symbol,side,shares\n2330,,100\n")

    # -- matrix 7: zero shares -> warned and dropped
    def test_zero_shares(self):
        lots, w = _load("symbol,shares\n2330,0\n1303,100\n")
        self.assertEqual(list(lots["symbol"]), ["1303"])
        self.assertTrue(any("zero-share" in x for x in w))

    def test_unparseable_shares_kept_unknown(self):
        lots, w = _load("symbol,shares\n2330,abc\n")
        self.assertEqual(lots.loc[0, "position_side"], "UNKNOWN")
        self.assertTrue(np.isnan(lots.loc[0, "position_qty"]))


class TestAggregation(unittest.TestCase):

    # -- matrix 8: duplicate LONG lots aggregate (the sign-flip bug fix)
    def test_duplicate_long_lots(self):
        lots, _ = _load("symbol,side,shares,avg_cost\n"
                        "2330,LONG,100,400\n2330,LONG,100,600\n")
        pos, w = hold.aggregate_positions(lots)
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos.loc[0, "position_qty"], 200.0)
        self.assertAlmostEqual(pos.loc[0, "avg_cost"], 500.0)  # qty-weighted
        self.assertEqual(pos.loc[0, "n_lots"], 2)
        self.assertTrue(any("aggregated" in x for x in w))

    # -- matrix 9: duplicate SHORT lots aggregate
    def test_duplicate_short_lots(self):
        lots, _ = _load("symbol,side,shares\n"
                        "2376,SHORT,300\n2376,SHORT,700\n")
        pos, _ = hold.aggregate_positions(lots)
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos.loc[0, "position_side"], "SHORT")
        self.assertEqual(pos.loc[0, "position_qty"], 1000.0)

    # -- matrix 10: simultaneous LONG + SHORT -> both kept, flagged, unnetted
    def test_long_short_conflict(self):
        lots, _ = _load("symbol,side,shares\n"
                        "2330,LONG,100\n2330,SHORT,40\n")
        pos, w = hold.aggregate_positions(lots)
        self.assertEqual(len(pos), 2)                 # NOT netted to 60
        self.assertTrue(pos["both_sides"].all())
        self.assertTrue(any("NOT" in x and "netted" in x.lower() or
                            "netted" in x for x in w))

    # -- matrix 26: gross/net exposure with mixed LONG/SHORT
    def test_exposure_metrics(self):
        lots, _ = _load("symbol,side,shares\n"
                        "2330,LONG,100\n1303,SHORT,50\n")
        pos, _ = hold.aggregate_positions(lots)
        pos["market_value_abs"] = [50000.0, 10000.0]
        m = hold.exposure_metrics(pos)
        self.assertEqual(m["gross_long_value"], 50000.0)
        self.assertEqual(m["gross_short_value"], 10000.0)
        self.assertEqual(m["gross_exposure"], 60000.0)
        self.assertEqual(m["net_exposure"], 40000.0)
        # denominators can never be negative under this convention
        self.assertGreaterEqual(m["gross_exposure"], 0.0)
        self.assertGreaterEqual(m["gross_long_value"], 0.0)


def ua(**kw):
    base = dict(position_side="NONE", model_action="", model_target=np.nan,
                in_universe=True, in_book=False, cmp_weight=np.nan,
                universe_rank_pct=None, conflict=False, material=False)
    base.update(kw)
    return hold.map_user_action(**base)


class TestUserActionMapping(unittest.TestCase):

    def test_open_short_not_in_vocabulary(self):
        self.assertNotIn("OPEN_SHORT", hold.USER_ACTIONS)
        self.assertNotIn("WATCH_SHORT", hold.USER_ACTIONS)

    # -- matrix 11: no position + BUY -> fresh signal, high freshness
    def test_none_buy(self):
        a, pri, _ = ua(model_action="BUY", model_target=0.08, in_book=True)
        self.assertEqual(a, "OPEN_LONG_NEW_SIGNAL")
        self.assertEqual(pri, "HIGH")

    # -- matrix 12: no position + HOLD -> standing target, NOT HOLD_LONG,
    #    lower freshness priority than a fresh BUY
    def test_none_hold(self):
        a, pri, _ = ua(model_action="HOLD", model_target=0.08, in_book=True)
        self.assertEqual(a, "OPEN_LONG_EXISTING_TARGET")
        self.assertEqual(pri, "MEDIUM")

    def test_none_reduce_by_remaining_target(self):
        a, pri, _ = ua(model_action="REDUCE", model_target=0.06, in_book=True)
        self.assertEqual((a, pri), ("OPEN_LONG_EXISTING_TARGET", "LOW"))
        a, _, _ = ua(model_action="REDUCE", model_target=0.02, in_book=True)
        self.assertEqual(a, "WATCH_LONG")

    def test_none_watch(self):
        a, _, _ = ua(model_action="WATCH", model_target=0.0, in_book=True)
        self.assertEqual(a, "WATCH_LONG")

    # -- matrix 13: no position + SELL -> NO_ACTION, never a short
    def test_none_sell(self):
        a, _, reason = ua(model_action="SELL", model_target=0.0, in_book=True)
        self.assertEqual(a, "NO_ACTION")
        self.assertIn("NOT a short", reason)

    # -- matrix 14: LONG + HOLD underweight (target 8%, actual 1%)
    def test_long_hold_underweight(self):
        a, _, _ = ua(position_side="LONG", model_action="HOLD",
                     model_target=0.08, in_book=True, cmp_weight=0.01)
        self.assertEqual(a, "ADD_LONG")

    # -- matrix 15: LONG + HOLD aligned (target 8%, actual 7.5%)
    def test_long_hold_aligned(self):
        a, _, _ = ua(position_side="LONG", model_action="HOLD",
                     model_target=0.08, in_book=True, cmp_weight=0.075)
        self.assertEqual(a, "HOLD_LONG")

    # -- matrix 16: LONG + HOLD overweight (target 8%, actual 12%)
    def test_long_hold_overweight(self):
        a, _, _ = ua(position_side="LONG", model_action="HOLD",
                     model_target=0.08, in_book=True, cmp_weight=0.12)
        self.assertEqual(a, "REDUCE_LONG")

    # -- matrix 17: LONG + REDUCE compares vs the NEW target
    def test_long_reduce(self):
        a, _, _ = ua(position_side="LONG", model_action="REDUCE",
                     model_target=0.06, in_book=True, cmp_weight=0.06)
        self.assertEqual(a, "HOLD_LONG")
        a, _, _ = ua(position_side="LONG", model_action="REDUCE",
                     model_target=0.04, in_book=True, cmp_weight=0.10)
        self.assertEqual(a, "REDUCE_LONG")

    # -- matrix 18: LONG + SELL -> EXIT_LONG
    def test_long_sell(self):
        a, pri, _ = ua(position_side="LONG", model_action="SELL",
                       model_target=0.0, in_book=True, cmp_weight=0.10)
        self.assertEqual((a, pri), ("EXIT_LONG", "HIGH"))

    # target 0 / unselected long: documented rank rule
    def test_long_unselected_rank_rule(self):
        a, _, _ = ua(position_side="LONG", cmp_weight=0.10,
                     universe_rank_pct=0.4)
        self.assertEqual(a, "REDUCE_LONG")
        a, _, _ = ua(position_side="LONG", cmp_weight=0.10,
                     universe_rank_pct=0.9)
        self.assertEqual(a, "EXIT_LONG")

    # -- matrix 19/20: SHORT against bullish model state -> cover review
    def test_short_vs_buy_hold(self):
        for act in ("BUY", "HOLD"):
            a, pri, reason = ua(position_side="SHORT", model_action=act,
                                model_target=0.08, in_book=True)
            self.assertEqual((a, pri), ("BUY_TO_COVER", "HIGH"))
            self.assertIn("no validated short-side model", reason)

    # -- matrix 21: SHORT + WATCH -> conflict review, not silent hold
    def test_short_vs_watch(self):
        a, pri, _ = ua(position_side="SHORT", model_action="WATCH",
                       model_target=0.0, in_book=True)
        self.assertEqual((a, pri), ("BUY_TO_COVER", "MEDIUM"))

    # -- matrix 22: SHORT + SELL must NOT auto-imply HOLD_SHORT
    def test_short_vs_sell_rank_based(self):
        a, _, _ = ua(position_side="SHORT", model_action="SELL",
                     in_book=True, universe_rank_pct=0.2)
        self.assertEqual(a, "BUY_TO_COVER")      # model leans positive
        a, _, _ = ua(position_side="SHORT", model_action="SELL",
                     in_book=True, universe_rank_pct=0.45)
        self.assertEqual(a, "REDUCE_SHORT")
        a, _, reason = ua(position_side="SHORT", model_action="SELL",
                          in_book=True, universe_rank_pct=0.8)
        self.assertEqual(a, "HOLD_SHORT")        # rank-based, not SELL-based
        self.assertIn("never bearish alpha", reason)

    # -- matrix 23/24/25: outside universe -> NO_MODEL_OPINION for any side
    def test_outside_universe(self):
        for side in ("LONG", "SHORT", "NONE"):
            a, _, _ = ua(position_side=side, in_universe=False)
            self.assertEqual(a, "NO_MODEL_OPINION")
        _, pri, _ = ua(position_side="LONG", in_universe=False, material=True)
        self.assertEqual(pri, "HIGH")

    def test_conflict_dominates(self):
        a, pri, _ = ua(position_side="LONG", model_action="BUY",
                       model_target=0.08, in_book=True, conflict=True)
        self.assertEqual((a, pri), ("POSITION_CONFLICT_REVIEW", "HIGH"))

    def test_all_outputs_in_vocabulary(self):
        cases = [
            dict(), dict(model_action="BUY", model_target=0.08, in_book=True),
            dict(position_side="LONG", model_action="SELL", in_book=True),
            dict(position_side="SHORT", universe_rank_pct=0.7),
            dict(in_universe=False), dict(conflict=True),
        ]
        for kw in cases:
            a, pri, _ = ua(**kw)
            self.assertIn(a, hold.USER_ACTIONS)
            self.assertIn(pri, ("HIGH", "MEDIUM", "LOW", "INFO"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
