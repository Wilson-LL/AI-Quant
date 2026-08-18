"""Unit tests for research/execution_price_bands.py (v16 Stage B).

Covers: tick rounding, outcome-column alignment, expanding-window
leakage safety, fallback hierarchy, minimum sample counts, fresh-vs-
existing aggressiveness invariant, extreme-gap distributions, and the
structural absence of short-entry bands.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_execution_price_bands -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))

import execution_price_bands as epb  # noqa: E402


def synth_frames(n_sym=4, n_days=300, gap_scale=0.005, seed=7,
                 extreme=()):
    """Deterministic synthetic cache frames. extreme: list of
    (sym_idx, day_idx, gap) to inject extreme opening gaps."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    frames = {}
    for i in range(n_sym):
        c = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        gaps = rng.normal(0.0005, gap_scale, n_days)
        for si, di, g in extreme:
            if si == i:
                gaps[di] = g
        o = np.empty(n_days)
        o[0] = c[0]
        o[1:] = c[:-1] * (1 + gaps[1:])
        h = np.maximum(o, c) * 1.005
        lo = np.minimum(o, c) * 0.995
        frames[f"90{i:02d}"] = pd.DataFrame(
            {"date": dates, "open": o, "high": h, "low": lo, "close": c,
             "volume": 1000.0})
    return frames


class TestTickRounding(unittest.TestCase):

    def test_ladder(self):
        for px, tick in ((5, 0.01), (25, 0.05), (77, 0.10), (250, 0.50),
                         (700, 1.0), (2400, 5.0)):
            self.assertEqual(epb.twse_tick(px), tick)

    def test_buy_down_sell_up(self):
        self.assertAlmostEqual(epb.round_to_tick(77.34, "buy"), 77.30)
        self.assertAlmostEqual(epb.round_to_tick(77.34, "sell"), 77.40)
        self.assertAlmostEqual(epb.round_to_tick(2403.0, "buy"), 2400.0)
        self.assertTrue(np.isnan(epb.round_to_tick(np.nan, "buy")))


class TestFeatures(unittest.TestCase):

    def test_outcome_alignment(self):
        f = synth_frames(1, 60)["9000"]
        feat = epb.symbol_features(f)
        # row t outcome = open[t+1]/close[t] - 1 (outcome only)
        t = 30
        self.assertAlmostEqual(
            feat.loc[t, "next_open_gap"],
            f.loc[t + 1, "open"] / f.loc[t, "close"] - 1, places=12)
        self.assertTrue(np.isnan(feat.iloc[-1]["next_open_gap"]))


class TestCalibrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 450 sessions x 4 symbols so each vol tercile clears MIN_CELL_OBS
        cls.frames = synth_frames(n_days=450)
        cls.hist = epb.build_history(cls.frames)
        cls.cal = epb.BandCalibrator(cls.hist)
        cls.asof = cls.hist["date"].max() - pd.Timedelta(days=30)

    # matrix 18: no future leakage in calibration
    def test_no_future_leakage(self):
        h2 = self.hist.copy()
        future = h2["date"] >= self.asof
        h2.loc[future, "next_open_gap"] = 0.5    # absurd future data
        cal2 = epb.BandCalibrator(h2)
        q1, n1, _ = self.cal.cell(self.asof, None, "MED", strict=True)
        q2, n2, _ = cal2.cell(self.asof, None, "MED", strict=True)
        self.assertEqual(n1, n2)
        for k in q1:
            if np.isfinite(q1[k]):
                self.assertAlmostEqual(q1[k], q2[k], places=12,
                                       msg=f"future leaked into {k}")

    # matrix 13/14: fallback hierarchy + insufficient sample counts
    def test_fallback_hierarchy(self):
        # rank buckets absent in this hist -> rank x vol falls back to VOL
        q, n, fb = self.cal.cell(self.asof, "TOP", "MED", strict=True)
        self.assertEqual(fb, "VOL")
        self.assertGreaterEqual(n, epb.CONFIG["MIN_CELL_OBS"])
        # unknown vol bucket -> GLOBAL
        q, n, fb = self.cal.cell(self.asof, None, None, strict=True)
        self.assertEqual(fb, "GLOBAL")
        # tiny pool -> INSUFFICIENT, no band
        early = self.hist["date"].min() + pd.Timedelta(days=5)
        q, n, fb = self.cal.cell(early, None, "MED", strict=True)
        self.assertIsNone(q)
        self.assertEqual(fb, "INSUFFICIENT")

    # matrix 15: fresh vs existing aggressiveness
    def test_fresh_more_aggressive_than_existing(self):
        q, _, _ = self.cal.cell(self.asof, None, "MED", strict=True)
        f = epb.entry_bands(100.0, 0.02, q, fresh=True)
        e = epb.entry_bands(100.0, 0.02, q, fresh=False)
        self.assertLessEqual(e["acceptable_ceiling"],
                             f["acceptable_ceiling"] + 1e-9)
        self.assertLessEqual(e["do_not_chase_above"],
                             f["do_not_chase_above"] + 1e-9)
        self.assertLessEqual(e["reference"], f["reference"] + 1e-9)

    # matrix 16/17: extreme positive / negative gap distributions
    def test_extreme_gap_distributions(self):
        ex = [(0, d, 0.10) for d in range(50, 60)] + \
             [(1, d, -0.10) for d in range(50, 60)]
        hist = epb.build_history(synth_frames(extreme=ex))
        cal = epb.BandCalibrator(hist)
        q, n, _ = cal.cell(hist["date"].max(), None, None, strict=True)
        f = epb.entry_bands(100.0, 0.02, q, fresh=True)
        s = epb.sell_bands(100.0, 0.02, q)
        # ordering invariants hold even with +/-10% tails in the pool
        self.assertLess(f["ideal_zone_low"], f["ideal_zone_high"])
        self.assertLessEqual(f["acceptable_ceiling"],
                             f["do_not_chase_above"])
        self.assertLess(f["risk_review_below"], f["ideal_zone_low"])
        self.assertLess(s["do_not_panic_sell_below"], s["sell_reference"])
        self.assertLess(s["urgent_risk_review_below"],
                        s["do_not_panic_sell_below"] + 1e-9)
        for v in list(f.values()) + list(s.values()):
            self.assertTrue(np.isfinite(v))

    def test_atr_guardrail_floors_zone_width(self):
        q, _, _ = self.cal.cell(self.asof, None, "LOW", strict=True)
        big_atr = 0.10
        f = epb.entry_bands(100.0, big_atr, q, fresh=True)
        width = f["ideal_zone_high"] - f["ideal_zone_low"]
        self.assertGreaterEqual(
            width, 2 * epb.CONFIG["K_WIDTH"] * big_atr * 100.0 - 1e-9)


class TestNoShortCreation(unittest.TestCase):

    # matrix 20 (engine side): no short-entry bands, no short postures
    def test_short_cover_only(self):
        hist = epb.build_history(synth_frames())
        cal = epb.BandCalibrator(hist)
        q, _, _ = cal.cell(hist["date"].max(), None, None, strict=True)
        b = epb.short_cover_bands(100.0, 0.02, q)
        self.assertEqual(set(b), {"cover_reference", "cover_zone_low",
                                  "cover_zone_high", "risk_review_above"})

    def test_posture_map_has_no_short_entry(self):
        for v in epb.POSTURE.values():
            self.assertNotIn("SHORT", v)
        for k in ("OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"):
            self.assertNotIn(k, epb.POSTURE)

    # matrix 19: validation never interprets daily OHLC path ordering
    def test_validation_module_is_path_blind(self):
        p = os.path.join(REPO, "research",
                         "next_session_price_band_validation.py")
        with open(p, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("stop_hit", src)
        self.assertNotIn("hit_before", src)
        self.assertIn("path-ambiguous", src)


class TestB2Reach(unittest.TestCase):
    """Range-reach probability definitions + leakage (Stage B2)."""

    def setUp(self):
        self.samples = {
            "gap": np.array([-0.02, -0.01, 0.0, 0.01, 0.02]),
            "low": np.array([-0.03, -0.02, -0.01, 0.0, 0.01]),
            "high": np.array([-0.01, 0.0, 0.01, 0.02, 0.03])}

    def test_buy_reach_definition(self):
        # BUY reach = share of next-day LOWS at or below the level
        self.assertAlmostEqual(
            epb.reach_prob_buy(self.samples, 100.0, 99.0), 0.6)   # -1%
        self.assertAlmostEqual(
            epb.reach_prob_buy(self.samples, 100.0, 97.0), 0.2)   # -3%
        self.assertAlmostEqual(
            epb.reach_prob_buy(self.samples, 100.0, 101.0), 1.0)

    def test_sell_reach_definition(self):
        # SELL reach = share of next-day HIGHS at or above the level
        self.assertAlmostEqual(
            epb.reach_prob_sell(self.samples, 100.0, 101.0), 0.6)  # +1%
        self.assertAlmostEqual(
            epb.reach_prob_sell(self.samples, 100.0, 103.0), 0.2)
        self.assertAlmostEqual(
            epb.reach_prob_sell(self.samples, 100.0, 99.0), 1.0)

    def test_open_beyond(self):
        self.assertAlmostEqual(epb.prob_open_beyond(
            self.samples, 100.0, 101.5, "above"), 0.2)
        self.assertAlmostEqual(epb.prob_open_beyond(
            self.samples, 100.0, 98.5, "below"), 0.2)

    def test_nan_safety(self):
        self.assertTrue(np.isnan(epb.reach_prob_buy(None, 100.0, 99.0)))
        self.assertTrue(np.isnan(
            epb.reach_prob_buy(self.samples, 100.0, np.nan)))

    def test_cell_full_leakage(self):
        frames = synth_frames(n_days=450)
        hist = epb.build_history(frames)
        asof = hist["date"].max() - pd.Timedelta(days=30)
        h2 = hist.copy()
        h2.loc[h2["date"] >= asof, "next_low_from_close"] = -0.5
        q1, n1, f1, s1 = epb.BandCalibrator(hist).cell_full(
            asof, None, "MED", strict=True)
        q2, n2, f2, s2 = epb.BandCalibrator(h2).cell_full(
            asof, None, "MED", strict=True)
        self.assertEqual(len(s1["low"]), len(s2["low"]))
        self.assertAlmostEqual(float(s1["low"].mean()),
                               float(s2["low"].mean()), places=12)

    def test_no_fill_probability_wording(self):
        # the plan layer must call these range-reach, never fill prob
        p = os.path.join(REPO, "research", "user_next_session_plan.py")
        with open(p, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("fill_probability", src)
        self.assertIn("NOT a fill probability", src)


class TestReviewPatch(unittest.TestCase):
    """2026-08-19 review patch: frozen thresholds, confidence layer."""

    def test_chase_quantiles_unchanged_from_b2(self):
        # numeric thresholds frozen: interpretation changed, values not
        self.assertEqual(epb.CONFIG["fresh"]["chase"], 0.90)
        self.assertEqual(epb.CONFIG["existing"]["chase"], 0.75)
        self.assertEqual(epb.CONFIG["fresh"]["ceil"], 0.75)
        self.assertEqual(epb.CONFIG["existing"]["ceil"], 0.60)
        self.assertEqual(epb.REACH_CONF, {"RECENT_N": 750,
                                          "RECENT_MIN": 200,
                                          "DRIFT_DEGRADED": 0.10,
                                          "DRIFT_HIGH": 0.05})

    def _samples(self, old_low, new_low, n_old=1500, n_new=750):
        # date-sorted pooled sample: old regime first, recent regime last
        return {"gap": np.zeros(n_old + n_new),
                "low": np.concatenate([np.full(n_old, old_low),
                                       np.full(n_new, new_low)]),
                "high": np.concatenate([np.full(n_old, -old_low),
                                        np.full(n_new, -new_low)])}

    def test_confidence_degraded_on_regime_drift(self):
        # old regime reaches -1% often, recent regime never
        s = self._samples(old_low=-0.02, new_low=0.0)
        conf, drift = epb.reach_confidence(s, 100.0, 99.0, "buy", "VOL")
        self.assertEqual(conf, "DEGRADED")
        self.assertLess(drift, -0.10)     # recent under-realizes

    def test_confidence_normal_and_high(self):
        s = self._samples(old_low=-0.02, new_low=-0.02)
        conf, drift = epb.reach_confidence(s, 100.0, 99.0, "buy", "VOL")
        self.assertEqual(conf, "NORMAL")
        self.assertAlmostEqual(drift, 0.0)
        conf, _ = epb.reach_confidence(s, 100.0, 99.0, "buy", "RANKxVOL")
        self.assertEqual(conf, "HIGH")
        conf, _ = epb.reach_confidence(s, 100.0, 99.0, "buy", "GLOBAL")
        self.assertEqual(conf, "DEGRADED")   # global fallback

    def test_confidence_insufficient_and_small_recent(self):
        conf, _ = epb.reach_confidence(None, 100.0, 99.0, "buy", "VOL")
        self.assertEqual(conf, "INSUFFICIENT")
        s = self._samples(-0.02, -0.02, n_old=50, n_new=50)
        conf, _ = epb.reach_confidence(s, 100.0, 99.0, "buy", "VOL")
        self.assertEqual(conf, "DEGRADED")   # recent < RECENT_MIN

    def test_confidence_leakage_safe(self):
        # confidence at asof must ignore post-asof data entirely
        frames = synth_frames(n_days=450)
        hist = epb.build_history(frames)
        asof = hist["date"].max() - pd.Timedelta(days=30)
        h2 = hist.copy()
        h2.loc[h2["date"] >= asof, ["next_low_from_close",
                                    "next_high_from_close"]] = -0.5
        for h in (hist, h2):
            cal = epb.BandCalibrator(h)
            q, n, fb, s = cal.cell_full(asof, None, "MED", strict=True)
            c, d = epb.reach_confidence(s, 100.0, 99.0, "buy", fb)
            if not hasattr(self, "_ref"):
                self._ref = (c, d)
        self.assertEqual((c, d), self._ref)

    def test_no_date_special_casing(self):
        # confidence must come from a principled rule, not year checks.
        # Docstrings may cite dates; CODE must not compare against them.
        import ast
        import re
        for mod in ("execution_price_bands.py", "twse_price_domain.py",
                    "user_next_session_plan.py"):
            with open(os.path.join(REPO, "research", mod),
                      encoding="utf-8") as f:
                src = f.read()
            self.assertIsNone(
                re.search(r"==\s*20\d\d|year\s*[=<>]|\.year\b", src),
                msg=mod)
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Constant) and \
                        isinstance(node.value, int):
                    self.assertNotEqual(node.value, 2026, msg=mod)

    def test_confidence_invariant_to_shuffled_history(self):
        # chronological regime shift: old period reaches -1% always,
        # recent period never -> DEGRADED with negative drift. The
        # production path must restore date order before "recent"
        # slicing, so a shuffled history must give IDENTICAL results.
        n_old, n_new = 1250, 750
        dates = pd.bdate_range("2018-01-02", periods=n_old + n_new)
        hist = pd.DataFrame({
            "date": dates,
            "stock": ["90%02d" % (i % 7) for i in range(len(dates))],
            "vol_bucket": "MED", "rank_bucket": np.nan,
            "next_open_gap": 0.0,
            "next_low_from_close": np.concatenate(
                [np.full(n_old, -0.02), np.full(n_new, 0.0)]),
            "next_high_from_close": np.concatenate(
                [np.full(n_old, 0.02), np.full(n_new, 0.0)]),
        })
        asof = dates[-1] + pd.Timedelta(days=5)
        ref = []
        for seed in (None, 0, 1, 42):
            h = hist if seed is None else \
                hist.sample(frac=1, random_state=seed)
            q, n, fb, s = epb.BandCalibrator(h).cell_full(
                asof, None, "MED", strict=True)
            conf, drift = epb.reach_confidence(s, 100.0, 99.0, "buy", fb)
            ref.append((conf, round(drift, 12), n, fb,
                        tuple(round(q[k], 12) for k in sorted(q)
                              if np.isfinite(q[k]))))
        # identical label, drift, sample count, fallback, quantiles
        self.assertTrue(all(r == ref[0] for r in ref[1:]))
        self.assertEqual(ref[0][0], "DEGRADED")
        self.assertLess(ref[0][1], -0.10)    # recent under-realizes
        # future rows are still excluded by the strict < T rule
        h3 = pd.concat([hist, hist.tail(50).assign(
            date=asof + pd.Timedelta(days=10),
            next_low_from_close=-0.5)], ignore_index=True)
        q3, n3, _, s3 = epb.BandCalibrator(h3).cell_full(
            asof, None, "MED", strict=True)
        self.assertEqual(n3, ref[0][2])
        c3, d3 = epb.reach_confidence(s3, 100.0, 99.0, "buy", "VOL")
        self.assertEqual((c3, round(d3, 12)), (ref[0][0], ref[0][1]))
        # thresholds untouched by the ordering fix
        self.assertEqual(epb.CONFIG["MIN_POOL"], 750)
        self.assertEqual(epb.CONFIG["MIN_CELL_OBS"], 400)
        self.assertEqual(epb.REACH_CONF["RECENT_N"], 750)

    def test_no_missed_trade_wording(self):
        with open(os.path.join(REPO, "research",
                               "user_next_session_plan.py"),
                  encoding="utf-8") as f:
            src = f.read()
        for bad in ("missed trade", "failed order", "missed profit"):
            self.assertNotIn(bad, src)
        self.assertIn("T+1_RANGE_DID_NOT_REACH_LEVEL", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
