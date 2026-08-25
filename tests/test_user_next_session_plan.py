"""End-to-end tests for research/user_next_session_plan.py (v16 Stage B).

Synthetic fixture repo (cache with 300 sessions of history so calibration
clears MIN_POOL, predictions, decision book incl. BUY/HOLD/SELL/WATCH,
paper-book history for position age, holdings variants). Touches nothing
in the real repo.

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_user_next_session_plan -v
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

import user_next_session_plan as unsp  # noqa: E402
import holdings as hold  # noqa: E402

DATE = "2026-01-09"          # a Friday -> weekend test


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _cache_csv(path, n=300, base=100.0, seed=3, end=DATE):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=end, periods=n)
    c = base * np.cumprod(1 + rng.normal(0, 0.01, n))
    gaps = rng.normal(0.0005, 0.005, n)
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1] * (1 + gaps[1:])
    h = np.maximum(o, c) * 1.005
    lo = np.minimum(o, c) * 0.995
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                       "open": o.round(2), "high": h.round(2),
                       "low": lo.round(2), "close": c.round(2),
                       "volume": 1000})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


class TestPlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="unsp_test_")
        r = cls.root
        cls.closes = {}
        for i, sym in enumerate(("2330", "1303", "2887", "2412", "0050")):
            df = _cache_csv(os.path.join(r, "research", "data_cache",
                                         f"{sym}.csv"), seed=3 + i,
                            base=100.0 * (i + 1))
            cls.closes[sym] = float(df["close"].iloc[-1])
        _write(os.path.join(r, "reports", "transformer_gpu",
                            f"{DATE}_predictions.csv"),
               "stock,score,score_std\n2330,0.10,0.01\n1303,0.20,0.01\n"
               "2887,0.30,0.01\n2412,0.05,0.01\n")
        # book: 2887 fresh BUY, 1303 HOLD, 2412 SELL, 2330 WATCH
        _write(os.path.join(r, "reports", "paper_trading",
                            f"{DATE}_blend50_band10_decision_book.csv"),
               "symbol,model_score,rank,action,target_weight,"
               "previous_weight,weight_change,sector,confidence\n"
               "2887,2.0,1,BUY,0.08,0.0,0.08,financials,low\n"
               "1303,1.5,2,HOLD,0.12,0.12,0.0,materials,low\n"
               "2412,0.2,3,SELL,0.0,0.05,-0.05,telecom,low\n"
               "2330,0.4,4,WATCH,0.0,0.0,0.0,semis,low\n")
        # paper-book history: 1303 present 3 sessions back -> age >= 3
        for d in ("2026-01-06", "2026-01-07", "2026-01-08", DATE):
            _write(os.path.join(r, "reports", "paper_trading", "books",
                                f"{d}_blend50_band10.csv"),
                   "stock,weight,score,rank\n1303,0.12,1.5,2\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _plan(self, holdings_text=None, date=DATE):
        hp = os.path.join(self.root, "holdings_case.csv")
        if holdings_text is None:
            if os.path.isfile(hp):
                os.unlink(hp)
        else:
            _write(hp, holdings_text)
        plan, meta = unsp.build_plan(self.root, hp, date, use_panel=False)
        return plan.set_index(
            plan["symbol"] + "|" + plan["position_side"].fillna("")), meta

    def _md(self, plan, meta):
        out = os.path.join(self.root, "reports", "user_actions")
        unsp.write_report(plan.reset_index(drop=True), meta, out)
        with open(os.path.join(out, "latest_next_session_action_plan.md"),
                  encoding="utf-8") as f:
            return f.read()

    # matrix 1: fresh BUY + no position
    def test_1_fresh_buy(self):
        plan, meta = self._plan("symbol,shares\n2330,100\n")
        r = plan.loc["2887|"]
        self.assertEqual(r["user_action"], "OPEN_LONG_NEW_SIGNAL")
        self.assertEqual(r["signal_freshness"], "FRESH_ENTRY")
        self.assertEqual(r["execution_posture"], "EXECUTE_IN_IDEAL_ZONE")
        for c in ("reference", "ideal_zone_low", "ideal_zone_high",
                  "acceptable_ceiling", "do_not_chase_above",
                  "risk_review_below"):
            self.assertTrue(np.isfinite(r[c]), c)
        self.assertLess(r["ideal_zone_low"], r["ideal_zone_high"])
        self.assertLessEqual(r["acceptable_ceiling"],
                             r["do_not_chase_above"])
        self.assertGreater(r["band_sample_count"], 0)

    # matrix 2: HOLD + no position -> existing target with age
    def test_2_hold_unowned(self):
        plan, _ = self._plan("symbol,shares\n2330,100\n")
        r = plan.loc["1303|"]
        self.assertEqual(r["user_action"], "OPEN_LONG_EXISTING_TARGET")
        self.assertEqual(r["signal_freshness"], "EXISTING_MODEL_POSITION")
        self.assertGreaterEqual(r["model_position_age_sessions"], 3)
        self.assertEqual(r["execution_posture"], "WAIT_FOR_PULLBACK")

    # matrix 3/4/5: HOLD + aligned / underweight / overweight LONG
    def test_3_hold_long_weight_cases(self):
        c1303 = self.closes["1303"]
        c2330 = self.closes["2330"]
        # aligned: 1303 ~12% of gross long
        q = 0.12 / (1 - 0.12) * (100 * c2330) / c1303
        plan, _ = self._plan(f"symbol,shares\n2330,100\n1303,{q:.0f}\n")
        self.assertEqual(plan.loc["1303|LONG", "user_action"], "HOLD_LONG")
        # underweight: tiny position
        plan, _ = self._plan(f"symbol,shares\n2330,100\n1303,1\n")
        self.assertEqual(plan.loc["1303|LONG", "user_action"], "ADD_LONG")
        # overweight: dominate the long book
        plan, _ = self._plan(f"symbol,shares\n2330,1\n1303,10000\n")
        self.assertEqual(plan.loc["1303|LONG", "user_action"],
                         "REDUCE_LONG")

    # matrix 6: SELL + no position -> NO_ACTION, out of high-priority
    def test_6_sell_unowned(self):
        plan, meta = self._plan("symbol,shares\n2330,100\n")
        r = plan.loc["2412|"]
        self.assertEqual(r["user_action"], "NO_ACTION")
        md = self._md(plan, meta)
        hi = md.split("## 1. HIGH PRIORITY")[1].split("## 2.")[0]
        self.assertNotIn("2412", hi)

    # matrix 7: SELL + LONG -> EXIT_LONG with sell bands
    def test_7_sell_held(self):
        plan, _ = self._plan("symbol,shares\n2412,100\n")
        r = plan.loc["2412|LONG"]
        self.assertEqual(r["user_action"], "EXIT_LONG")
        self.assertEqual(r["execution_posture"], "SELL_IN_IDEAL_ZONE")
        for c in ("sell_reference", "ideal_sell_zone_low",
                  "ideal_sell_zone_high", "acceptable_sell_floor",
                  "do_not_panic_sell_below", "urgent_risk_review_below"):
            self.assertTrue(np.isfinite(r[c]), c)
        self.assertLess(r["urgent_risk_review_below"],
                        r["do_not_panic_sell_below"] + 1e-9)

    # matrix 8: WATCH + no position -> WATCH_LONG with price answers
    def test_8_watch(self):
        plan, meta = self._plan("symbol,shares\n1303,1\n")
        r = plan.loc["2330|"]
        self.assertEqual(r["user_action"], "WATCH_LONG")
        self.assertTrue(np.isfinite(r["reference"]))
        md = self._md(plan, meta)
        self.assertIn("watch entry reference", md)

    # matrix 9: outside-universe LONG -> NO_MODEL_OPINION, no bands
    def test_9_outside_universe(self):
        plan, meta = self._plan("symbol,shares\n2330,100\n0050,500\n")
        r = plan.loc["0050|LONG"]
        self.assertEqual(r["user_action"], "NO_MODEL_OPINION")
        self.assertTrue(np.isnan(r["reference"]))
        md = self._md(plan, meta)
        self.assertIn("No AI-Quant target / action / execution band "
                      "available", md)

    # matrix 10: existing SHORT + bullish model -> BUY_TO_COVER + bands
    def test_10_short_vs_bullish(self):
        plan, meta = self._plan("symbol,side,shares\n2887,SHORT,100\n")
        r = plan.loc["2887|SHORT"]
        self.assertEqual(r["user_action"], "BUY_TO_COVER")
        self.assertEqual(r["execution_posture"], "RISK_REVIEW")
        self.assertTrue(np.isfinite(r["cover_reference"]))
        self.assertTrue(np.isfinite(r["risk_review_above"]))
        md = self._md(plan, meta)
        self.assertIn("NO VALIDATED OPEN-SHORT MODEL EXISTS", md)

    # matrix 11: stale book warning
    def test_11_stale_book(self):
        extra = os.path.join(self.root, "research", "data_cache",
                             "2330.csv")
        with open(extra, "a", encoding="utf-8") as f:
            f.write("2026-01-12,100,101,99,100,1000\n")
        try:
            plan, meta = self._plan("symbol,shares\n2330,100\n")
            self.assertTrue(meta["book_stale"])
            self.assertEqual(meta["book_age_sessions"], 1)
            md = self._md(plan, meta)
            self.assertIn("STALE BOOK", md)
        finally:
            df = pd.read_csv(extra)
            df[df["date"] != "2026-01-12"].to_csv(extra, index=False)

    # matrix 12: Friday signal -> Monday execution date
    def test_12_weekend(self):
        plan, meta = self._plan("symbol,shares\n2330,100\n")
        self.assertEqual(meta["signal_date"], "2026-01-09")   # Friday
        self.assertEqual(meta["intended_execution_date"], "2026-01-12")
        self.assertEqual(unsp.next_twse_session("2026-01-10"),
                         "2026-01-12")

    # missing holdings file -> model-only plan, flagged
    def test_13_user_position_unknown(self):
        plan, meta = self._plan(None)
        self.assertFalse(meta["user_position_known"])
        md = self._md(plan, meta)
        self.assertIn("USER_POSITION_UNKNOWN", md)
        self.assertEqual(plan.loc["2887|", "user_action"],
                         "OPEN_LONG_NEW_SIGNAL")

    # ---- Stage B2: legal domain, distributions, reach ----

    def test_b2_all_prices_legal_hard_gate(self):
        import twse_price_domain as tpd
        plan, _ = self._plan("symbol,side,shares\n2330,LONG,100\n"
                             "2412,LONG,50\n2887,SHORT,10\n")
        band_cols = [c for c in unsp.BAND_COLS] + \
            [c for c in unsp.DOMAIN_COLS if c.startswith("expected_")]
        for _, r in plan.iterrows():
            if r["price_domain_status"] != "NORMAL_DAY_ASSUMPTION":
                continue
            for c in band_cols:
                v = r[c]
                if pd.isna(v):
                    continue
                self.assertTrue(tpd.is_legal_tick(v), (r["symbol"], c, v))
                self.assertGreaterEqual(v, r["legal_limit_down"] - 1e-9,
                                        (r["symbol"], c))
                self.assertLessEqual(v, r["legal_limit_up"] + 1e-9,
                                     (r["symbol"], c))

    def test_b2_domain_columns(self):
        plan, _ = self._plan("symbol,shares\n2330,100\n0050,500\n")
        r = plan.loc["2887|"]           # in-universe stock
        self.assertEqual(r["price_domain_status"], "NORMAL_DAY_ASSUMPTION")
        self.assertEqual(r["auction_reference_source"], "PREVIOUS_CLOSE")
        self.assertTrue(np.isfinite(r["legal_limit_up"]))
        self.assertLessEqual(r["legal_limit_up"],
                             r["auction_reference_price"] * 1.10 + 1e-9)
        # ETF outside universe: no fabricated domain or distributions
        e = plan.loc["0050|LONG"]
        self.assertEqual(e["price_domain_status"], "UNKNOWN")
        self.assertTrue(np.isnan(e["legal_limit_up"]))
        self.assertTrue(np.isnan(e["expected_open_p50"]))

    def test_b2_expected_open_ordered(self):
        plan, _ = self._plan("symbol,shares\n2330,100\n")
        r = plan.loc["2887|"]
        seq = [r[f"expected_open_p{p}"] for p in (10, 25, 50, 75, 90)]
        self.assertTrue(all(np.isfinite(v) for v in seq))
        self.assertTrue(all(seq[i] <= seq[i + 1] + 1e-9
                            for i in range(4)))

    def test_b2_reach_probabilities(self):
        plan, meta = self._plan("symbol,shares\n2412,100\n")
        b = plan.loc["2887|"]           # fresh BUY, no position
        for c in ("buy_reference_reach_probability",
                  "ideal_low_reach_probability",
                  "p_open_above_do_not_chase"):
            self.assertTrue(0.0 <= b[c] <= 1.0, c)
        s = plan.loc["2412|LONG"]       # EXIT_LONG
        self.assertTrue(0.0 <= s["sell_reference_reach_probability"] <= 1)
        self.assertTrue(0.0 <= s["p_open_below_panic_level"] <= 1)
        # reach curve CSV written, buy + sell rows present
        out = os.path.join(self.root, "reports", "user_actions")
        unsp.write_report(plan.reset_index(drop=True), meta, out)
        curve = pd.read_csv(os.path.join(
            out, "history", DATE[:7], f"{DATE}_price_reach_curve.csv"),
            dtype={"symbol": str})
        self.assertIn("BUY", set(curve["side"]))
        self.assertIn("SELL", set(curve["side"]))
        self.assertTrue(curve["range_reach_probability"].between(0, 1)
                        .all())

    def test_b2_md_wording(self):
        plan, meta = self._plan("symbol,shares\n2412,100\n")
        md = self._md(plan, meta)
        self.assertIn("NOT a fill probability", md)
        self.assertIn("legal range:", md)
        self.assertIn("never fill", md)   # header disclaimer
        self.assertNotIn("fill probability: ", md)

    # ---- 2026-08-19 review patch: wording + confidence ----

    def test_p1_execution_quality_wording(self):
        plan, meta = self._plan("symbol,shares\n2412,100\n")
        md = self._md(plan, meta)
        self.assertIn("above preferred execution range", md)
        self.assertIn("did NOT show", md)                # no alpha claim
        self.assertIn("T+1_RANGE_DID_NOT_REACH_LEVEL", md)
        self.assertNotIn("do not chase above:", md)      # old phrasing
        for bad in ("missed trade", "failed order", "missed profit",
                    "loses alpha", "signal is bad"):
            self.assertNotIn(bad, md)
        # numeric schema field preserved for compatibility
        self.assertIn("do_not_chase_above", plan.columns)
        self.assertTrue(np.isfinite(plan.loc["2887|",
                                             "do_not_chase_above"]))

    def test_p2_confidence_column_and_caveat(self):
        plan, meta = self._plan("symbol,shares\n2412,100\n")
        self.assertIn("range_reach_confidence", plan.columns)
        vals = set(plan["range_reach_confidence"])
        self.assertTrue(vals <= {"HIGH", "NORMAL", "DEGRADED",
                                 "INSUFFICIENT", "NA"})
        # actions/thresholds are untouched by confidence
        self.assertEqual(plan.loc["2412|LONG", "user_action"],
                         "EXIT_LONG")
        # force DEGRADED on the sell row -> inline caveat + header note
        p2 = plan.reset_index(drop=True).copy()
        m = p2["symbol"].eq("2412") & p2["position_side"].eq("LONG")
        p2.loc[m, "range_reach_confidence"] = "DEGRADED"
        p2.loc[m, "reach_drift_recent"] = -0.15
        out = os.path.join(self.root, "reports", "user_actions")
        unsp.write_report(p2, meta, out)
        with open(os.path.join(out,
                               "latest_next_session_action_plan.md"),
                  encoding="utf-8") as f:
            md = f.read()
        self.assertIn("reach confidence DEGRADED", md)
        self.assertIn("under-realized", md)
        self.assertIn("lower confidence", md)
        self.assertIn("reach-calibration confidence DEGRADED", md)

    # matrix 20 + B27: no short-creation action can ever be emitted
    def test_20_no_open_short(self):
        for text in ("symbol,shares\n2330,100\n1303,-50\n",
                     "symbol,side,shares\n2887,SHORT,100\n",
                     None):
            plan, _ = self._plan(text)
            self.assertFalse({"OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"}
                             & set(plan["user_action"]))
            self.assertTrue(set(plan["user_action"])
                            <= set(hold.USER_ACTIONS))


class TestNightly(TestPlan):
    """C2 --nightly gates. Inherits the TestPlan fixture (and re-runs its
    tests harmlessly under this class name)."""

    def _nightly(self, holdings_text, out_name,
                 status="NEW_SESSION_DATA", recovery=False):
        hp = os.path.join(self.root, "holdings_case.csv")
        if holdings_text is None:
            if os.path.isfile(hp):
                os.unlink(hp)
        else:
            _write(hp, holdings_text)
        out = os.path.join(self.root, "reports", out_name)
        rc = unsp.nightly(self.root, hp, out, eod_refresh_status=status,
                          allow_recovery=recovery)
        return rc, out

    def _hash(self, path):
        with open(path, "rb") as f:
            return hash(f.read())

    # C2-20.1/2: fresh session -> plan; same-session rerun -> untouched
    def test_n1_fresh_then_no_new_data(self):
        rc, out = self._nightly("symbol,shares\n2330,100\n", "n1")
        self.assertEqual(rc, 0)
        latest = os.path.join(out, "latest_next_session_action_plan.csv")
        dated = os.path.join(out, "history", DATE[:7],
                             f"{DATE}_next_session_action_plan.csv")
        self.assertTrue(os.path.isfile(latest))
        self.assertTrue(os.path.isfile(dated))     # dated -> history/YYYY-MM
        h1 = self._hash(latest)
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("FRESH_PLAN_GENERATED", f.read())
        # rerun: latest actionable plan must NOT be overwritten
        rc2, _ = self._nightly("symbol,shares\n2330,999\n", "n1")
        self.assertEqual(rc2, 0)
        self.assertEqual(self._hash(latest), h1)
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("NO_NEW_SESSION_DATA", f.read())

    # C2-20.5: partial publication blocks the actionable plan
    def test_n2_partial_publication_blocked(self):
        extra = os.path.join(self.root, "research", "data_cache",
                             "2330.csv")
        with open(extra, "a", encoding="utf-8") as f:
            f.write("2026-01-12,100,101,99,100,1000\n")
        try:
            rc, out = self._nightly("symbol,shares\n2330,100\n", "n2")
            self.assertEqual(rc, 0)     # warning, not pipeline failure
            self.assertFalse(os.path.isfile(os.path.join(
                out, "latest_next_session_action_plan.csv")))
            with open(os.path.join(out, "latest_nightly_status.md"),
                      encoding="utf-8") as f:
                s = f.read()
            self.assertIn("PARTIAL_PUBLICATION_SUSPECTED", s)
            self.assertIn("1/4", s)
        finally:
            df = pd.read_csv(extra)
            df[df["date"] != "2026-01-12"].to_csv(extra, index=False)

    # C2-20.4: stale book (full newer session, book not regenerated)
    def test_n3_stale_book_blocked(self):
        paths = [os.path.join(self.root, "research", "data_cache",
                              f"{s}.csv") for s in ("2330", "1303",
                                                    "2887", "2412")]
        for p in paths:
            with open(p, "a", encoding="utf-8") as f:
                f.write("2026-01-12,100,101,99,100,1000\n")
        try:
            rc, out = self._nightly("symbol,shares\n2330,100\n", "n3")
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.isfile(os.path.join(
                out, "latest_next_session_action_plan.csv")))
            with open(os.path.join(out, "latest_nightly_status.md"),
                      encoding="utf-8") as f:
                self.assertIn("STALE_BOOK", f.read())
        finally:
            for p in paths:
                df = pd.read_csv(p)
                df[df["date"] != "2026-01-12"].to_csv(p, index=False)

    # C2-20.3: missing holdings file -> graceful model-only run
    def test_n4_missing_holdings(self):
        rc, out = self._nightly(None, "n4")
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("FRESH_PLAN_GENERATED", f.read())

    def test_n5_gate_policy(self):
        # data-integrity policy (user-approved 2026-08-19): allow at
        # most ~one missing name in a ~108-name cross-sectional universe
        self.assertEqual(unsp.PARTIAL_COVERAGE_MIN, 0.99)
        self.assertGreaterEqual(108 / 108, unsp.PARTIAL_COVERAGE_MIN)
        self.assertGreaterEqual(107 / 108, unsp.PARTIAL_COVERAGE_MIN)
        self.assertLess(106 / 108, unsp.PARTIAL_COVERAGE_MIN)
        # generic ratio behavior (fixture-independent)
        self.assertLess(0.75, unsp.PARTIAL_COVERAGE_MIN)   # 3/4 blocks
        self.assertGreaterEqual(1.0, unsp.PARTIAL_COVERAGE_MIN)

    # ---- patch A: explicit EOD refresh-status contract ----

    # A5.1: +0 rows + existing standing plan -> byte-unchanged
    def test_a1_no_new_with_standing_plan(self):
        rc, out = self._nightly("symbol,shares\n2330,100\n", "a1")
        latest = os.path.join(out, "latest_next_session_action_plan.csv")
        h1 = self._hash(latest)
        rc2, _ = self._nightly("symbol,shares\n2330,100\n", "a1",
                               status="NO_NEW_SESSION_DATA")
        self.assertEqual(rc2, 0)
        self.assertEqual(self._hash(latest), h1)
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            s = f.read()
        self.assertIn("NO_NEW_SESSION_DATA", s)
        self.assertIn("standing plan: signal date", s)

    # A5.2: +0 rows + NO existing plan -> hard gate, nothing generated
    def test_a2_no_new_without_standing_plan(self):
        rc, out = self._nightly("symbol,shares\n2330,100\n", "a2",
                                status="NO_NEW_SESSION_DATA")
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isfile(os.path.join(
            out, "latest_next_session_action_plan.csv")))
        self.assertFalse(os.path.isfile(os.path.join(
            out, f"{DATE}_next_session_action_plan.csv")))
        self.assertFalse(os.path.isfile(os.path.join(
            out, "history", DATE[:7],
            f"{DATE}_next_session_action_plan.csv")))
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            s = f.read()
        self.assertIn("NO_NEW_SESSION_DATA", s)
        self.assertIn("NO_STANDING_ACTION_PLAN", s)

    # A5.6: UNKNOWN status (direct manual run) -> refused by default
    def test_a3_unknown_status_refused(self):
        rc, out = self._nightly("symbol,shares\n2330,100\n", "a3",
                                status="UNKNOWN")
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isfile(os.path.join(
            out, "latest_next_session_action_plan.csv")))
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("EOD_REFRESH_STATUS_UNKNOWN", f.read())

    # A5.7: explicit recovery mode -> clearly labeled, plan generated
    def test_a4_recovery_mode(self):
        rc, out = self._nightly("symbol,shares\n2330,100\n", "a4",
                                status="UNKNOWN", recovery=True)
        self.assertEqual(rc, 0)
        latest = os.path.join(out, "latest_next_session_action_plan.csv")
        self.assertTrue(os.path.isfile(latest))
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            s = f.read()
        self.assertIn("RECOVERY_PLAN_GENERATED", s)
        self.assertIn("RECOVERY MODE", s)
        # intended date re-derived from today, not from the old signal
        plan = pd.read_csv(latest, dtype={"symbol": str})
        import datetime as dtm
        self.assertGreaterEqual(
            str(plan["intended_execution_date"].iloc[0]),
            dtm.date.today().isoformat())

    # A5.8: daily_ops passes the explicit status in both branches
    def test_a5_bat_contract(self):
        with open(os.path.join(REPO, "daily_ops.bat"),
                  encoding="utf-8") as f:
            bat = f.read()
        self.assertIn('set "EOD_REFRESH_STATUS=NEW_SESSION_DATA"', bat)
        self.assertIn('set "EOD_REFRESH_STATUS=NO_NEW_SESSION_DATA"', bat)
        self.assertIn("--eod-refresh-status %EOD_REFRESH_STATUS%", bat)
        self.assertNotIn("--allow-current-book-recovery", bat)

    # patch B: calendar-uncertainty labels on every plan
    # ---- 2026-08-25 cleanup: dedup-gate history migration (A-D) ----

    def test_h1_history_dated_plan_blocks_duplicate(self):
        # A: a dated plan in history/YYYY-MM blocks same-session regen
        rc, out = self._nightly("symbol,shares\n2330,100\n", "h1")
        latest = os.path.join(out, "latest_next_session_action_plan.csv")
        h1 = self._hash(latest)
        rc2, _ = self._nightly("symbol,shares\n2330,999\n", "h1")
        self.assertEqual(rc2, 0)
        self.assertEqual(self._hash(latest), h1)   # byte-unchanged
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("NO_NEW_SESSION_DATA", f.read())

    def test_h2_legacy_root_dated_plan_still_blocks(self):
        # B: transition safety — a legacy flat-root dated file is still
        # recognized by the gate
        out = os.path.join(self.root, "reports", "h2")
        os.makedirs(out, exist_ok=True)
        _write(os.path.join(out,
                            f"{DATE}_next_session_action_plan.csv"),
               f"signal_date,intended_execution_date\n{DATE},2026-01-12\n")
        rc, out2 = self._nightly("symbol,shares\n2330,100\n", "h2")
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isfile(os.path.join(
            out, "latest_next_session_action_plan.csv")))
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("NO_NEW_SESSION_DATA", f.read())

    def test_h3_missing_dated_plan_generates(self):
        # C: a genuinely missing dated plan does NOT falsely trigger the
        # gate — generation proceeds (fresh fixture dir)
        rc, out = self._nightly("symbol,shares\n2330,100\n", "h3")
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "latest_nightly_status.md"),
                  encoding="utf-8") as f:
            self.assertIn("FRESH_PLAN_GENERATED", f.read())

    def test_h4_latest_paths_unchanged(self):
        # D: the four human-facing latest_* paths stay exactly in the
        # user_actions root
        rc, out = self._nightly("symbol,shares\n2330,100\n", "h4")
        for n in ("latest_next_session_action_plan.csv",
                  "latest_next_session_action_plan.md",
                  "latest_next_session_summary.md"):
            self.assertTrue(os.path.isfile(os.path.join(out, n)), n)

    def test_b_calendar_estimate_labels(self):
        plan, meta = self._plan("symbol,shares\n2330,100\n")
        self.assertEqual(meta["intended_execution_date_source"],
                         "WEEKDAY_CALENDAR_ESTIMATE")
        self.assertEqual(meta["intended_execution_date_confidence"],
                         "UNVERIFIED_FOR_HOLIDAYS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
