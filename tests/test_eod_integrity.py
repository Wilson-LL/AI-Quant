"""H-DATA-INTEGRITY: EOD cache trading-session integrity, refresh root-cause
fix, gap repair, dataset/inference gap guard and the longitudinal gate.
Synthetic caches in temp dirs only; the live cache is never touched."""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import data as D  # noqa: E402
import eod_integrity as E  # noqa: E402
import refresh_data as RD  # noqa: E402

CAL = pd.bdate_range("2024-01-01", "2025-12-31")      # synthetic market calendar (Mon-Fri)
HOLIDAY = pd.Timestamp("2024-02-14")                   # a weekday closure: no symbol trades
CAL = CAL[CAL != HOLIDAY]


def ohlcv(dates, seed=0):
    rng = np.random.default_rng(seed)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), 2)
    return pd.DataFrame({"date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"), "open": c,
                         "high": np.round(c * 1.01, 2), "low": np.round(c * 0.99, 2), "close": c,
                         "volume": np.arange(len(dates)) + 1000})


def write_cache(d, sym, dates, seed=0):
    df = ohlcv(dates, seed)
    df.to_csv(os.path.join(d, f"{sym}.csv"), index=False, lineterminator="\r\n")
    return df


REQ = E.required_windows()


class TestAuditor(unittest.TestCase):

    def audit(self, dates):
        return E.audit_symbol("X", pd.Series(pd.DatetimeIndex(dates)), CAL, CAL.max(), REQ)

    def test_01_full_missing_month_detected(self):
        a = self.audit(CAL[~((CAL.year == 2024) & (CAL.month == 6))])
        self.assertEqual(a["full_month_gaps"], 1)
        self.assertIn("2024-06", a["full_month_list"])
        self.assertEqual(a["missing_sessions"], int(((CAL.year == 2024) & (CAL.month == 6)).sum()))

    def test_02_single_missing_session_detected(self):
        a = self.audit(CAL.delete(300))
        self.assertEqual(a["missing_sessions"], 1)
        self.assertEqual(a["gap_intervals"], 1)
        self.assertEqual(a["full_month_gaps"], 0)

    def test_03_weekend_not_a_gap(self):
        cal = E.derive_calendar({f"S{i}": pd.Series(CAL) for i in range(12)})
        self.assertFalse((cal.dayofweek >= 5).any())
        self.assertEqual(self.audit(CAL)["missing_sessions"], 0)

    def test_04_exchange_holiday_not_a_gap(self):
        cal = E.derive_calendar({f"S{i}": pd.Series(CAL) for i in range(12)})
        self.assertNotIn(HOLIDAY, cal)
        a = E.audit_symbol("X", pd.Series(CAL), cal, cal.max(), REQ)
        self.assertEqual(a["missing_sessions"], 0)

    def test_03b_saturday_makeup_session_is_a_session(self):
        sat = pd.Timestamp("2024-03-02")
        d = {f"S{i}": pd.Series(CAL.append(pd.DatetimeIndex([sat])).sort_values()) for i in range(12)}
        self.assertIn(sat, E.derive_calendar(d))

    def test_05_duplicate_date_detected(self):
        dates = list(CAL) + [CAL[10]]
        self.assertEqual(self.audit(sorted(dates))["duplicate_dates"], 1)

    def test_06_unsorted_dates_detected(self):
        dates = list(CAL)
        dates[5], dates[6] = dates[6], dates[5]
        self.assertGreater(self.audit(dates)["non_monotonic_steps"], 0)

    def test_07_fresh_tail_with_historical_gap(self):
        a = self.audit(CAL.delete(20))                     # hole early in 2024, tail current
        self.assertTrue(a["tail_fresh"])
        self.assertEqual(a["LEVEL_C_history"], "INCOMPLETE")
        self.assertEqual(a["LEVEL_A_live_inference"], "PASS")
        b = self.audit(CAL.delete(len(CAL) - 5))            # hole inside the current window
        self.assertTrue(b["tail_fresh"])
        self.assertEqual(b["LEVEL_A_live_inference"], "FAIL")

    def test_required_window_derived_from_feature_code(self):
        self.assertEqual(E.feature_lookback("close_only"), 131)   # mom_126_5 = c.shift(5)/c.shift(131)
        self.assertEqual(REQ["REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS"], 191)
        self.assertEqual(REQ["REQUIRED_TRAINING_CONTIGUOUS_SESSIONS"], 212)

    def test_wed_to_fri_is_not_one_session(self):
        mon = pd.Timestamp("2024-06-03")
        cal = pd.bdate_range(mon, periods=6)                # Mon..Mon
        dates = cal.delete(3)                               # Thu missing
        gb = E.gap_before(dates, cal)
        self.assertEqual(list(gb), [False, False, False, True, False])


class FakeDay(SimpleNamespace):
    pass


def payload(dates, base=100.0):
    return [FakeDay(date=d.to_pydatetime(), open=base, high=base + 1, low=base - 1, close=base,
                    capacity=1000) for d in pd.DatetimeIndex(dates)]


class TestRefreshAndRepair(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._lc, self._cp = RD.load_cached, RD.cache_path
        RD.load_cached = lambda s: pd.read_csv(os.path.join(self.tmp, f"{s}.csv"), parse_dates=["date"])
        RD.cache_path = lambda s: os.path.join(self.tmp, f"{s}.csv")

    def tearDown(self):
        RD.load_cached, RD.cache_path = self._lc, self._cp
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_08_retry_does_not_forget_a_failed_month(self):
        write_cache(self.tmp, "S", CAL[CAL < "2025-03-01"])
        months = {(2025, 2): CAL[(CAL >= "2025-02-01") & (CAL < "2025-03-01")],
                  (2025, 3): CAL[(CAL >= "2025-03-01") & (CAL < "2025-04-01")],
                  (2025, 4): CAL[(CAL >= "2025-04-01") & (CAL < "2025-05-01")],
                  (2025, 5): CAL[(CAL >= "2025-05-01") & (CAL < "2025-05-20")]}
        fail = {(2025, 4)}

        def fetch(y, m):
            if (y, m) in fail:
                raise ConnectionError("boom")
            return payload(months.get((y, m), []))
        today = pd.Timestamp("2025-05-19")
        RD.refresh_stock("S", today, fetch_fn=fetch, throttle_s=0)
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"), parse_dates=["date"])["date"]
        self.assertEqual(d.max(), months[(2025, 3)].max())        # nothing appended past the failure
        fail.clear()                                               # retry pass
        RD.refresh_stock("S", today, fetch_fn=fetch, throttle_s=0)
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"), parse_dates=["date"])["date"]
        self.assertEqual(E.audit_symbol("S", d, CAL[CAL <= d.max()], d.max(), REQ)["missing_sessions"], 0)

    def test_09_empty_payload_stays_unresolved(self):
        write_cache(self.tmp, "S", CAL[~((CAL.year == 2024) & (CAL.month == 6))])
        before = open(os.path.join(self.tmp, "S.csv"), "rb").read()
        state = os.path.join(self.tmp, "state.json")
        for _ in range(RD.REPAIR_MAX_ATTEMPTS + 1):
            log, st = RD.repair_gaps(self.tmp, CAL, [("S", "2024-06")], state_path=state,
                                     fetch_factory=lambda s: (lambda y, m: []), sleep_fn=lambda x: None)
        self.assertEqual(st["S|2024-06"]["state"], E.UNRESOLVED)
        self.assertNotEqual(st["S|2024-06"]["state"], E.CONFIRMED_NO_DATA)
        self.assertEqual(open(os.path.join(self.tmp, "S.csv"), "rb").read(), before)

    def test_10_gap_merge_is_idempotent_and_preserves_existing_lines(self):
        hole = CAL[(CAL.year == 2024) & (CAL.month == 6)]
        write_cache(self.tmp, "S", CAL.difference(hole))
        p = os.path.join(self.tmp, "S.csv")
        orig = open(p, "rb").read()
        rows = pd.DataFrame([{"date": d, "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 7}
                             for d in hole])
        self.assertEqual(RD.merge_missing_rows(p, rows, hole), len(hole))
        after = open(p, "rb").read()
        for line in orig.split(b"\r\n"):
            self.assertIn(line, after.split(b"\r\n"))
        self.assertEqual(RD.merge_missing_rows(p, rows, hole), 0)
        self.assertEqual(open(p, "rb").read(), after)

    def test_16_no_forward_fill_or_interpolation(self):
        hole = CAL[(CAL.year == 2024) & (CAL.month == 6)]
        write_cache(self.tmp, "S", CAL.difference(hole))
        state = os.path.join(self.tmp, "state.json")
        partial = hole[:5]                                          # source only has 5 of the sessions
        log, st = RD.repair_gaps(self.tmp, CAL, [("S", "2024-06")], state_path=state,
                                 fetch_factory=lambda s: (lambda y, m: payload(partial)),
                                 sleep_fn=lambda x: None)
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"), parse_dates=["date"])["date"]
        self.assertEqual(len(set(d) & set(hole)), 5)                # only returned sessions added
        self.assertEqual(st["S|2024-06"]["state"], E.CONFIRMED_NO_DATA)
        self.assertEqual(len(st["S|2024-06"]["no_data_sessions"]), len(hole) - 5)


class TestDatasetAndInferenceGuard(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._cd = D.CACHE_DIR
        D.CACHE_DIR = self.tmp
        import dataset_transformer_eod as DS
        self.DS = DS

    def tearDown(self):
        D.CACHE_DIR = self._cd
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, guard=True):
        return self.DS.build_dataset("close_only", seq_len=60, horizons=(20,), universe=["9001", "9002", "9003"],
                                     verbose=False, gap_guard=guard, calendar=CAL)

    def test_13_contiguous_data_unchanged_by_guard(self):
        for i, s in enumerate(("9001", "9002", "9003")):
            write_cache(self.tmp, s, CAL, seed=i)
        a, b = self.build(True), self.build(False)
        self.assertTrue(np.array_equal(a["X"], b["X"], equal_nan=True))
        self.assertTrue(np.array_equal(a["targets"]["fwd_20"], b["targets"]["fwd_20"], equal_nan=True))
        self.assertEqual(a["integrity"]["rejected_by_stock"], {})

    def test_12_training_sample_crossing_gap_rejected(self):
        write_cache(self.tmp, "9001", CAL.delete(300), seed=0)          # one missing session
        write_cache(self.tmp, "9002", CAL, seed=1)
        write_cache(self.tmp, "9003", CAL, seed=2)
        a, b = self.build(True), self.build(False)
        # a 191-row window has 190 row-to-row transitions: one hole kills exactly 190 windows
        self.assertEqual(a["integrity"]["rejected_by_stock"].get("9001"), REQ["REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS"] - 1)
        s = a["stocks"].index("9001")
        ends = a["dates"][a["date_rank"][a["stock_idx"] == s]]
        dates = pd.DatetimeIndex(CAL.delete(300))
        missing = CAL[300]
        for e in pd.DatetimeIndex(ends):                   # no kept window spans the hole
            i = dates.get_loc(e)
            start = dates[max(0, i - 190)]
            self.assertFalse(start < missing < e)
        # labels spanning the hole are gone too
        fa = a["targets"]["fwd_20"][a["stock_idx"] == s]
        fb = b["targets"]["fwd_20"][b["stock_idx"] == b["stocks"].index("9001")]
        self.assertLess(np.isfinite(fa).sum(), np.isfinite(fb).sum())

    def test_11_current_inference_window_crossing_gap_rejected(self):
        write_cache(self.tmp, "9001", CAL.delete(len(CAL) - 3), seed=0)  # hole in the current window
        write_cache(self.tmp, "9002", CAL, seed=1)
        write_cache(self.tmp, "9003", CAL, seed=2)
        a = self.build(True)
        asof = str(CAL.max())[:10]
        self.assertEqual(a["integrity"]["latest_status"]["9001"]["status"], "WINDOW_CROSSES_DATA_GAP")
        s = a["stocks"].index("9001")
        self.assertFalse(((a["stock_idx"] == s) & (a["dates"][a["date_rank"]] == CAL.max())).any())
        import inference_transformer_eod as INF
        f = INF.integrity_failures(a, asof)
        self.assertEqual(f["symbol"].tolist(), ["9001"])
        self.assertEqual(f["status"].iloc[0], E.DATA_INTEGRITY_FAILURE)
        self.assertEqual(f["reason"].iloc[0], "WINDOW_CROSSES_DATA_GAP")


class TestGates(unittest.TestCase):

    def setUp(self):
        import pipeline_gate as G
        self.G = G
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "research", "data_cache"))
        self.syms = G.universe()[:12]

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, sym, dates, seed=0):
        write_cache(os.path.join(self.root, "research", "data_cache"), sym, dates, seed)

    def test_14_cross_sectional_99pct_gate_unchanged(self):
        from user_next_session_plan import PARTIAL_COVERAGE_MIN
        self.assertEqual(PARTIAL_COVERAGE_MIN, 0.99)
        for i, s in enumerate(self.syms):
            self.write(s, CAL if i else CAL[:-1], i)            # one name one day short
        ok, msg = self.G.check(self.root, "refresh")
        self.assertFalse(ok)                                    # 11/12 < 99%
        self.write(self.syms[0], CAL, 0)
        ok, _ = self.G.check(self.root, "refresh")
        self.assertTrue(ok)

    def test_15_integrity_gate_blocks_and_standing_plan_preserved(self):
        for i, s in enumerate(self.syms):
            self.write(s, CAL.delete(len(CAL) - 4) if i == 0 else CAL, i)
        ok_refresh, _ = self.G.check(self.root, "refresh")
        ok_int, msg = self.G.check(self.root, "integrity")
        self.assertTrue(ok_refresh)                              # "12/12 at the newest date" ...
        self.assertFalse(ok_int)                                 # ... does NOT pass a hidden hole
        self.assertIn("RECENT_WINDOW_INTEGRITY", msg)
        self.write(self.syms[0], CAL, 0)
        self.assertTrue(self.G.check(self.root, "integrity")[0])
        bat = open(os.path.join(REPO, "daily_ops.bat"), encoding="utf-8", errors="replace").read()
        i_gate = bat.index("pipeline_gate.py integrity")
        self.assertIn("if errorlevel 1 goto :pipefail", bat[i_gate:i_gate + 120])
        for later in ("--mode daily-retrain", "inference_transformer_eod.py", "blended_decision_book.py",
                      "user_next_session_plan.py --nightly"):
            self.assertLess(i_gate, bat.index(later))
        pipefail = bat[bat.rindex("\n:pipefail"):]          # the label, not a goto target
        self.assertNotIn("user_next_session_plan", pipefail)
        self.assertIn("Previous standing plan remains untouched", pipefail)


class TestFrozenInputBook(unittest.TestCase):

    def test_17_frozen_input_decision_book_byte_identical(self):
        """Regenerate the latest runtime decision book from frozen inputs with
        THIS branch's code and compare bytes. Runtime files live in the
        production working tree (AIQUANT_RUNTIME_ROOT, default: this repo)."""
        rt = os.environ.get("AIQUANT_RUNTIME_ROOT", REPO)
        pt = os.path.join(rt, "reports", "paper_trading")
        books = sorted(f for f in os.listdir(pt) if f.endswith("_blend50_band10_decision_book.csv")) \
            if os.path.isdir(pt) else []
        if not books or not os.path.isdir(os.path.join(rt, "research", "data_cache")):
            self.skipTest("no runtime decision book / cache available")
        asof = books[-1][:10]
        import blended_decision_book as B
        import paper_trading as P
        saved = (D.CACHE_DIR, P.BOOK_DIR, B.ROOT, B.PT_DIR)
        out = tempfile.mkdtemp()
        try:
            D.CACHE_DIR = os.path.join(rt, "research", "data_cache")
            P.BOOK_DIR = os.path.join(pt, "books")
            B.ROOT, B.PT_DIR = rt, out
            B.build(asof)
            for n in (f"{asof}_blend50_band10_decision_book.csv", f"{asof}_blend50_band10_decision_book.md"):
                h = [hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() for d in (pt, out)]
                self.assertEqual(h[0], h[1], n)
        finally:
            D.CACHE_DIR, P.BOOK_DIR, B.ROOT, B.PT_DIR = saved
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
