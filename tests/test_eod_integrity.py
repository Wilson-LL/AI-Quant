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
        self.assertNotIn(st["S|2024-06"]["state"], E.COMPLETE_STATES + E.RESOLVED_EXCEPTION_STATES)
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
        self.assertEqual(st["S|2024-06"]["state"], E.MARKET_OPEN_SYMBOL_NO_DATA)
        self.assertEqual(len(st["S|2024-06"]["no_data_sessions"]), len(hole) - 5)


class TestSymbolNoTradeSemantics(unittest.TestCase):
    """MARKET OPEN does not imply EVERY STOCK HAS A ROW."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.hole = CAL[(CAL.year == 2024) & (CAL.month == 6)]
        write_cache(self.tmp, "S", CAL.difference(self.hole))
        self.state = os.path.join(self.tmp, "state.json")
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fetch_factory(self, days):
        def factory(sym):
            def fetch(y, m):
                self.calls.append((sym, y, m))
                return payload(days)
            return fetch
        return factory

    def repair(self, days, registry=None, runs=1):
        for _ in range(runs):
            log, st = RD.repair_gaps(self.tmp, CAL, [("S", "2024-06")], state_path=self.state,
                                     registry_path=registry, fetch_factory=self.fetch_factory(days),
                                     sleep_fn=lambda x: None)
        return st["S|2024-06"]

    def test_nt1_genuine_missing_download_is_repaired(self):
        st = self.repair(self.hole)
        self.assertEqual(st["state"], E.FETCH_OK)
        self.assertEqual(st["last_request"]["rows_added"], len(self.hole))
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"))["date"]
        a = E.audit_symbol("S", d, CAL, CAL.max(), REQ)
        self.assertEqual(a["missing_sessions"], 0)

    def test_nt2_registry_confirmed_suspension_is_not_fetched_and_still_breaks_contiguity(self):
        reg = os.path.join(self.tmp, "registry.csv")
        pd.DataFrame({"symbol": "S", "date": self.hole.strftime("%Y-%m-%d"),
                      "status": E.CONFIRMED_SYMBOL_NO_TRADE, "reason": "exchange-announced suspension",
                      "source": "TWSE announcement (test)"}).to_csv(reg, index=False)
        st = self.repair(self.hole, registry=reg)
        self.assertEqual(st["state"], E.CONFIRMED_SYMBOL_NO_TRADE)
        self.assertEqual(self.calls, [])                          # no download for confirmed days
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"))["date"]
        self.assertEqual(len(set(pd.to_datetime(d)) & set(self.hole)), 0)   # nothing fabricated
        gb = E.gap_before(pd.DatetimeIndex(pd.to_datetime(d)), CAL)
        self.assertTrue(gb.any())                                  # still a break for the model
        exc = E.missing_exceptions(None, reg)
        a = E.audit_symbol("S", d, CAL, CAL.max(), REQ, exc)
        self.assertEqual(a["missing_confirmed_symbol_no_trade"], len(self.hole))
        self.assertEqual(a["missing_unrepaired"], 0)

    def test_nt3_unresolved_symbol_specific_no_data(self):
        served = self.hole[:3]                                     # source serves 3 days only
        st = self.repair(served)
        self.assertEqual(st["state"], E.MARKET_OPEN_SYMBOL_NO_DATA)
        self.assertEqual(st["suspension_status"], E.SUSPENSION_STATUS_UNKNOWN)
        self.assertEqual(len(st["no_data_sessions"]), len(self.hole) - 3)
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"))["date"]
        a = E.audit_symbol("S", d, CAL, CAL.max(), REQ, E.missing_exceptions(self.state, None))
        self.assertEqual(a["missing_market_open_symbol_no_data"], len(self.hole) - 3)
        self.assertEqual(a["LEVEL_C_history"], "INCOMPLETE")

    def test_nt4_unresolved_never_silently_becomes_complete(self):
        self.repair(self.hole[:3], runs=5)                         # many later runs
        st = self.repair(self.hole, runs=2)                        # even if the source changes
        self.assertEqual(st["state"], E.MARKET_OPEN_SYMBOL_NO_DATA)
        self.assertNotIn(st["state"], E.COMPLETE_STATES)
        self.assertEqual(len(self.calls), 1)                       # no endless re-downloading
        tmp2 = tempfile.mkdtemp()
        try:                                                       # empty payloads end UNRESOLVED
            write_cache(tmp2, "T", CAL.difference(self.hole))
            s2 = os.path.join(tmp2, "state.json")
            for _ in range(RD.REPAIR_MAX_ATTEMPTS + 2):
                log, st2 = RD.repair_gaps(tmp2, CAL, [("T", "2024-06")], state_path=s2,
                                          fetch_factory=lambda s: (lambda y, m: []), sleep_fn=lambda x: None)
            self.assertEqual(st2["T|2024-06"]["state"], E.UNRESOLVED)
            self.assertNotIn(st2["T|2024-06"]["state"], E.COMPLETE_STATES)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)


class TestSnapshotAndTailAppend(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "cache")
        os.makedirs(self.cache)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_is_exact_and_verifier_detects_changes(self):
        hole = CAL[(CAL.year == 2024) & (CAL.month == 6)]
        write_cache(self.cache, "A", CAL.difference(hole))
        write_cache(self.cache, "B", CAL, seed=1)
        man = E.snapshot_cache(self.cache, os.path.join(self.tmp, "snap"))
        self.assertTrue(man["verified"])
        self.assertEqual(man["file_count"], 2)
        with self.assertRaises(FileExistsError):
            E.snapshot_cache(self.cache, os.path.join(self.tmp, "snap"))
        rows = pd.DataFrame([{"date": d, "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 1}
                             for d in hole])
        RD.merge_missing_rows(os.path.join(self.cache, "A.csv"), rows, hole)
        df, tot = E.verify_against_snapshot(os.path.join(self.tmp, "snap"), self.cache, CAL)
        self.assertEqual(tot["existing_rows_modified_or_removed"], 0)
        self.assertEqual(tot["rows_added"], len(hole))
        self.assertEqual(tot["unexplained_added_dates"], 0)
        p = os.path.join(self.cache, "B.csv")                      # tamper with one existing row
        b = open(p, "rb").read().replace(b",", b";", 3)
        open(p, "wb").write(b)
        df, tot = E.verify_against_snapshot(os.path.join(self.tmp, "snap"), self.cache, CAL)
        self.assertGreater(tot["existing_rows_modified_or_removed"], 0)

    def test_tail_refresh_appends_without_rewriting_history(self):
        write_cache(self.cache, "S", CAL[CAL < "2025-06-01"])
        p = os.path.join(self.cache, "S.csv")
        before = open(p, "rb").read()
        lc, cp = RD.load_cached, RD.cache_path
        RD.load_cached = lambda s: pd.read_csv(os.path.join(self.cache, f"{s}.csv"), parse_dates=["date"])
        RD.cache_path = lambda s: os.path.join(self.cache, f"{s}.csv")
        june = CAL[(CAL >= "2025-06-01") & (CAL < "2025-06-20")]

        def fetch(y, m):
            return payload(june if (y, m) == (2025, 6) else CAL[(CAL.year == y) & (CAL.month == m)])
        try:
            n, _ = RD.refresh_stock("S", pd.Timestamp("2025-06-19"), throttle_s=0, fetch_fn=fetch)
        finally:
            RD.load_cached, RD.cache_path = lc, cp
        after = open(p, "rb").read()
        self.assertEqual(n, len(june))
        self.assertTrue(after.startswith(before))                 # history untouched, rows appended

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
        self.assertIn(E.UNSAFE, msg)                             # 11/12 valid < 99%
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


class TestConfirmedNoTradeRegistry(unittest.TestCase):
    """Decision 2026-09-21: 2207 / 2025-12-18 = CONFIRMED_NO_TRADE. Resolves the
    repair-queue item; never fills a row; never restores window contiguity."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.day = CAL[-30]
        write_cache(self.tmp, "S", CAL.delete(len(CAL) - 30))
        self.state = os.path.join(self.tmp, "state.json")
        self.month = str(self.day.to_period("M"))
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_repair(self, registry=None):
        month_days = CAL[(CAL.to_period("M") == self.day.to_period("M")) & (CAL != self.day)]

        def factory(sym):
            def fetch(y, m):
                self.calls.append((y, m))
                return payload(month_days)          # the source serves every day but this one
            return fetch
        return RD.repair_gaps(self.tmp, CAL, [("S", self.month)], state_path=self.state,
                              registry_path=registry, fetch_factory=factory,
                              sleep_fn=lambda x: None)[1][f"S|{self.month}"]

    def registry(self):
        reg = os.path.join(self.tmp, "registry.csv")
        pd.DataFrame({"symbol": ["S"], "date": [str(self.day)[:10]], "status": ["CONFIRMED_NO_TRADE"],
                      "reason": ["zero-trade day"], "source": ["TWSE MI_INDEX (test)"]}).to_csv(reg, index=False)
        return reg

    def test_nt5_confirmed_no_trade_resolves_queue_item(self):
        st = self.run_repair()
        self.assertEqual(st["state"], E.MARKET_OPEN_SYMBOL_NO_DATA)       # before the decision
        st = self.run_repair(self.registry())
        self.assertEqual(st["state"], E.CONFIRMED_SYMBOL_NO_TRADE)        # upgraded, resolved
        self.assertEqual(st["state"], "CONFIRMED_NO_TRADE")
        self.assertEqual(len(self.calls), 1)                              # no retry download
        exc = E.missing_exceptions(self.state, self.registry())
        self.assertEqual(exc[("S", str(self.day)[:10])], E.CONFIRMED_SYMBOL_NO_TRADE)

    def test_nt6_confirmed_no_trade_does_not_resolve_window_continuity(self):
        self.run_repair(self.registry())
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"))["date"]
        a = E.audit_symbol("S", d, CAL, CAL.max(), REQ, E.missing_exceptions(self.state, self.registry()))
        self.assertEqual(a["LEVEL_A_live_inference"], "FAIL")
        self.assertEqual(a["recent_window_missing"], 1)
        wf = E.window_failure_reasons(self.tmp, ["S"], CAL, CAL.max(), registry_path=self.registry())
        self.assertEqual(wf["S"]["reason"], E.CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW)
        wf2 = E.window_failure_reasons(self.tmp, ["S"], CAL, CAL.max(),
                                       registry_path=os.path.join(self.tmp, "none.csv"))
        self.assertEqual(wf2["S"]["reason"], E.MISSING_SESSION_IN_REQUIRED_WINDOW)

    def test_nt7_confirmed_no_trade_synthesizes_no_row(self):
        before = open(os.path.join(self.tmp, "S.csv"), "rb").read()
        self.run_repair(self.registry())
        self.run_repair(self.registry())
        after = open(os.path.join(self.tmp, "S.csv"), "rb").read()
        self.assertEqual(before, after)                                   # byte-identical file
        self.assertNotIn(str(self.day)[:10].encode(), after)

    def test_nt8_confirmed_no_trade_is_not_a_downloadable_request(self):
        d = pd.read_csv(os.path.join(self.tmp, "S.csv"))["date"]
        a = E.audit_symbol("S", d, CAL, CAL.max(), REQ)
        plan = E.backfill_plan([a], CAL, CAL.max(), REQ)
        self.assertEqual(len(plan), 1)                                    # without the registry
        conf = E.load_no_trade_registry(self.registry())
        plan = E.backfill_plan([a], CAL, CAL.max(), REQ, confirmed=conf)
        self.assertEqual(len(plan), 0)                                    # no longer downloadable

    def test_repo_registry_has_2207_with_evidence(self):
        reg = E.load_no_trade_registry(E.NO_TRADE_REGISTRY_DEFAULT)
        self.assertIn(("2207", "2025-12-18"), reg)
        r = pd.read_csv(E.NO_TRADE_REGISTRY_DEFAULT, dtype=str)
        row = r[(r["symbol"] == "2207") & (r["date"] == "2025-12-18")].iloc[0]
        self.assertEqual(row["status"], "CONFIRMED_NO_TRADE")
        self.assertIn("MI_INDEX", row["source"])


class TestLiveIntegrityPolicy(unittest.TestCase):
    """Symbol-level live policy: derived from the 99% rule, never a fixed
    one-symbol allowance; a held invalid symbol always blocks."""

    U = [f"{1000 + i}" for i in range(108)]

    def test_p1_107_of_108_degraded(self):
        p = E.classify_live_integrity(self.U, ["1005"], set())
        self.assertEqual(p["status"], E.DEGRADED)
        self.assertTrue(p["publication_allowed"])
        self.assertEqual((p["valid"], p["eligible"]), (107, 108))
        self.assertEqual(p["excluded"], ["1005"])

    def test_p2_106_of_108_blocked(self):
        p = E.classify_live_integrity(self.U, ["1005", "1006"], set())
        self.assertEqual(p["status"], E.UNSAFE)
        self.assertFalse(p["publication_allowed"])

    def test_p3_threshold_is_the_ratio_not_a_count(self):
        u = [str(i) for i in range(300)]
        self.assertEqual(E.classify_live_integrity(u, ["1", "2", "3"], set())["status"], E.DEGRADED)
        self.assertEqual(E.classify_live_integrity(u, ["1", "2", "3", "4"], set())["status"], E.UNSAFE)
        self.assertEqual(E.classify_live_integrity(u[:50], ["1"], set())["status"], E.UNSAFE)

    def test_p4_held_invalid_blocks_even_at_99pct(self):
        u = self.U[:106] + ["2883", "3443"]
        for held_sym in ("2883", "3443"):
            p = E.classify_live_integrity(u, [held_sym], {"2883", "3443"})
            self.assertEqual(p["status"], E.UNSAFE)                      # never "99% is enough"
            self.assertFalse(p["publication_allowed"])
            self.assertEqual(p["held_invalid"], [held_sym])
            self.assertGreaterEqual(p["valid_ratio"], 0.99)

    def test_p5_clean_is_safe(self):
        self.assertEqual(E.classify_live_integrity(self.U, [], {"1000"})["status"], E.SAFE)
        self.assertEqual(E.VALID_MODEL_COVERAGE_MIN, 0.99)


class TestHeldSymbolGateRegression(unittest.TestCase):
    """End-to-end integrity gate on a synthetic runtime root with the full universe."""

    def setUp(self):
        import pipeline_gate as G
        self.G = G
        self.root = tempfile.mkdtemp()
        self.cache = os.path.join(self.root, "research", "data_cache")
        os.makedirs(self.cache)
        self.syms = G.universe()
        self.hole = CAL.delete(len(CAL) - 40)
        for i, s in enumerate(self.syms):
            write_cache(self.cache, s, CAL, i)
        self.n = len(self.syms)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def holdings(self, syms):
        pd.DataFrame({"symbol": syms, "shares": [1000] * len(syms)}).to_csv(
            os.path.join(self.root, "my_holdings.csv"), index=False)

    def standing_book(self, syms):
        pt = os.path.join(self.root, "reports", "paper_trading")
        os.makedirs(pt, exist_ok=True)
        pd.DataFrame({"symbol": syms, "target_weight": [1 / len(syms)] * len(syms)}).to_csv(
            os.path.join(pt, "2025-12-01_blend50_band10_decision_book.csv"), index=False)

    def test_g1_one_nonheld_invalid_is_degraded_and_listed(self):
        self.assertGreaterEqual((self.n - 1) / self.n, 0.99)
        write_cache(self.cache, "2207", self.hole, 7)
        self.holdings(["2883", "3443"])
        ok, msg = self.G.check(self.root, "integrity")
        self.assertTrue(ok, msg)
        self.assertIn("DEGRADED", msg)
        self.assertIn("2207", msg)

    def test_g2_held_invalid_blocks_2883(self):
        write_cache(self.cache, "2883", self.hole, 7)
        self.holdings(["2883", "3443"])
        ok, msg = self.G.check(self.root, "integrity")
        self.assertFalse(ok)
        self.assertIn(E.UNSAFE, msg)
        self.assertIn("2883", msg)

    def test_g3_A_book_only_invalid_is_degraded_not_blocked(self):
        """3443 invalid, in the previous model book only, NOT in my_holdings.csv:
        previous-book membership is model state, never a hard-block source."""
        write_cache(self.cache, "3443", self.hole, 7)
        self.standing_book(["3443", "2330"])
        self.holdings(["2883"])
        ok, msg = self.G.check(self.root, "integrity")
        self.assertTrue(ok, msg)
        self.assertIn("DEGRADED", msg)
        self.assertIn("3443", msg)
        li = self.G.longitudinal_integrity(self.root)
        hb = li["HOLDINGS_VS_BOOK"]
        self.assertEqual(hb["REAL_HELD_SYMBOLS"], ["2883"])
        self.assertEqual(hb["PREVIOUS_BOOK_SYMBOLS"], ["2330", "3443"])
        self.assertEqual(hb["BOOK_ONLY"], ["2330", "3443"])
        self.assertEqual(li["DATA_INTEGRITY_STATUS"]["held_invalid"], [])

    def test_g5_B_invalid_in_holdings_and_book_blocks(self):
        write_cache(self.cache, "2883", self.hole, 7)
        self.holdings(["2883", "0050"])
        self.standing_book(["2883", "2330"])
        ok, msg = self.G.check(self.root, "integrity")
        self.assertFalse(ok)
        self.assertIn(E.UNSAFE, msg)
        self.assertIn("2883", msg)

    def test_g6_C_real_holding_not_in_book_blocks(self):
        """2330: real holding, in the model universe, NOT in the previous book.
        (0050 / 6669 are real holdings outside the model universe -- ETF /
        unconfigured -- so they have no model input window to invalidate; they
        stay NO_MODEL_OPINION under holdings completeness.)"""
        write_cache(self.cache, "2330", self.hole, 7)
        self.holdings(["2330", "2883", "0050", "6669"])
        self.standing_book(["2883", "3443"])              # 2330 not in the model book
        ok, msg = self.G.check(self.root, "integrity")
        self.assertFalse(ok)
        self.assertIn("2330", msg)

    def test_g7_held_symbols_is_real_holdings_only(self):
        self.holdings(["0050"])
        self.standing_book(["3443"])
        self.assertEqual(E.held_symbols(self.root)[0], {"0050"})
        self.assertEqual(E.previous_book_symbols(self.root)[0], {"3443"})

    def test_g4_F_two_nonheld_invalid_below_99pct_blocks(self):
        self.assertLess((self.n - 2) / self.n, 0.99)
        for s in ("2207", "1101"):
            write_cache(self.cache, s, self.hole, 7)
        ok, msg = self.G.check(self.root, "integrity")
        self.assertFalse(ok)
        self.assertIn("< 99%", msg)


class TestExclusionBookSemantics(unittest.TestCase):
    """An excluded symbol gets no stale/neutral/bottom/synthetic score: it is
    absent from the cross-section, and every z/rank/band/weight is computed
    over valid names only (identical to a universe without that name)."""

    def build(self, cache_syms, pred_syms, integ_rows, prev_book=None):
        import blended_decision_book as B
        import paper_trading as P
        root, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        bdir = os.path.join(root, "nobooks")
        if prev_book is not None:                            # previous paper band10 book
            os.makedirs(bdir)
            pd.DataFrame({"stock": list(prev_book), "weight": 1.0 / len(prev_book)}).to_csv(
                os.path.join(bdir, "2025-12-01_blend50_band10.csv"), index=False)
        cache = os.path.join(root, "research", "data_cache")
        os.makedirs(cache)
        for s in cache_syms:
            write_cache(cache, s, CAL, int(s))              # per-symbol seed: same data in both builds
        gd = os.path.join(root, "reports", "transformer_gpu")
        os.makedirs(gd)
        asof = str(CAL[-1])[:10]
        sc = {s: np.random.default_rng(int(s)).normal() for s in cache_syms}
        pd.DataFrame({"stock": pred_syms, "score": [sc[s] for s in pred_syms],
                      "score_std": [0.1] * len(pred_syms), "sector": "x", "vol_20": 0.2}).to_csv(
            os.path.join(gd, f"{asof}_predictions.csv"), index=False)
        pd.DataFrame(integ_rows, columns=["symbol", "status", "reason", "last_cached_date", "asof",
                                          "detail", "held"]).to_csv(
            os.path.join(gd, f"{asof}_data_integrity.csv"), index=False)
        saved = (D.CACHE_DIR, P.BOOK_DIR, B.ROOT, B.PT_DIR)
        try:
            D.CACHE_DIR, P.BOOK_DIR, B.ROOT, B.PT_DIR = cache, bdir, root, out
            db = B.build(asof)
            uni = pd.read_csv(os.path.join(out, f"{asof}_blend50_universe_scores.csv"), dtype={"symbol": str})
            md = open(os.path.join(out, f"{asof}_blend50_band10_decision_book.md"), encoding="utf-8").read()
        finally:
            D.CACHE_DIR, P.BOOK_DIR, B.ROOT, B.PT_DIR = saved
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)
        return db, uni, md

    def row(self, x):
        asof = str(CAL[-1])[:10]
        return [x, "DATA_INTEGRITY_FAILURE", "WINDOW_CROSSES_DATA_GAP", asof, asof,
                E.CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW, False]

    def full_and_excluded(self, pick):
        """Build the 108/108-style full book, then exclude the name picked from
        its blend ranking. Returns (full db, full uni, excluded db, uni, md, x)."""
        import pipeline_gate as G
        syms = G.universe()                                   # full-size cross-section (name cap)
        dbf, unif, _ = self.build(syms, syms, [])
        x = pick(unif.sort_values("rank")["symbol"].tolist())
        db, uni, md = self.build(syms, [s for s in syms if s != x], [self.row(x)])
        return dbf, unif, db, uni, md, x

    @staticmethod
    def held(db):
        return db.loc[db["target_weight"] > 0].set_index("symbol")["target_weight"]

    def test_x1_A_all_valid_is_original_semantics(self):
        import paper_trading as P
        rng = np.random.default_rng(3)
        day = pd.DataFrame({"stock": [f"{1000 + i}" for i in range(108)], "score": rng.normal(size=108)})
        prev = list(day["stock"].sample(22, random_state=1))
        a = P._book_from_scores(day, prev_names=prev, band=P.BAND)
        b = P._book_from_scores(day, prev_names=prev, band=P.BAND, ref_n=108)
        pd.testing.assert_frame_equal(a, b)                   # ref_n == N: identical
        self.assertEqual(len(a), 22)

    def test_x2_B_excluded_in_book_slot_filled_by_next_valid_name(self):
        dbf, unif, db, uni, md, x = self.full_and_excluded(lambda r: r[10])
        hf, he = self.held(dbf), self.held(db)
        self.assertIn(x, hf.index)
        self.assertEqual(len(he), len(hf))                    # cardinality preserved (not N-1)
        nxt = unif.sort_values("rank")["symbol"].tolist()[len(hf)]   # first name outside the full book
        self.assertEqual(set(he.index), (set(hf.index) - {x}) | {nxt})
        self.assertNotIn(x, set(db["symbol"]))                # no score, rank or row of any kind
        self.assertNotIn(x, set(uni["symbol"]))
        self.assertEqual(sorted(uni["rank"]), list(range(1, len(uni) + 1)))
        self.assertIn(f"{x} — DATA_INTEGRITY_FAILURE / CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW", md)
        self.assertIn(f"reference eligible universe {len(unif)}, valid scored {len(uni)}", md)
        self.assertAlmostEqual(float(he.sum()), float(hf.sum()), places=4)   # gross exposure
        self.assertLessEqual(float(he.max()), 0.10 + 1e-4)                  # name cap

    def test_x3_E_would_be_selected_top_name_skipped(self):
        dbf, unif, db, uni, md, x = self.full_and_excluded(lambda r: r[0])
        hf, he = self.held(dbf), self.held(db)
        nxt = unif.sort_values("rank")["symbol"].tolist()[len(hf)]
        self.assertEqual(set(he.index), (set(hf.index) - {x}) | {nxt})
        self.assertEqual(len(he), len(hf))

    def test_x4_F_excluded_outside_all_bands_book_unchanged(self):
        dbf, unif, db, uni, md, x = self.full_and_excluded(lambda r: r[-1])
        hf, he = self.held(dbf), self.held(db)
        pd.testing.assert_series_equal(he.sort_index(), hf.sort_index())   # same names, same weights
        self.assertEqual(set(dbf["symbol"]) - {x}, set(db["symbol"]))       # same actions/watch list

    def test_x5_band10_retention_pool_sized_on_reference_universe(self):
        """The 2026-09-11 / 2308 case: an incumbent at valid rank 26 of 107 is
        retained with the 108-name rule (k=22, pool=26) and would be dropped by
        the old N-1 rule (k=21, pool=25)."""
        import paper_trading as P
        day = pd.DataFrame({"stock": [f"{1000 + i}" for i in range(107)],
                            "score": np.linspace(1, 0, 107)})
        inc = "1025"                                          # valid rank 26
        prev = [f"{1000 + i}" for i in range(21)] + [inc]
        new = P._book_from_scores(day, prev_names=prev, band=P.BAND, ref_n=108)
        old = P._book_from_scores(day, prev_names=prev, band=P.BAND)
        self.assertEqual(len(new), 22)
        self.assertIn(inc, set(new["stock"]))
        self.assertEqual(len(old), 21)
        self.assertNotIn(inc, set(old["stock"]))

    def test_x6_neural_target_book_uses_reference_universe(self):
        import inference_transformer_eod as I
        rng = np.random.default_rng(5)
        pred = pd.DataFrame({"stock": [f"{1100 + i}" for i in range(107)], "score": rng.normal(size=107),
                             "score_std": 0.1, "sector": "x", "vol_20": 0.2})
        full = I.make_decision_book(pred, None, 0.2, 0.05, 20, "t+1")
        ref = I.make_decision_book(pred, None, 0.2, 0.05, 20, "t+1", ref_n=108)
        self.assertEqual(int((full["target_weight"] > 0).sum()), 21)
        self.assertEqual(int((ref["target_weight"] > 0).sum()), 22)
        same = I.make_decision_book(pred, None, 0.2, 0.05, 20, "t+1", ref_n=107)
        pd.testing.assert_frame_equal(full, same)

    def test_x13_unavailable_sets_separate_sizing_from_coverage(self):
        tmp = tempfile.mkdtemp()
        try:
            pd.DataFrame([{"symbol": "a", "reason": "WINDOW_CROSSES_DATA_GAP"},
                          {"symbol": "b", "reason": "STALE_TAIL"}]).to_csv(
                os.path.join(tmp, "2026-01-02_data_integrity.csv"), index=False)
            self.assertEqual(E.integrity_window_excluded(tmp, "2026-01-02"), {"a"})     # coverage
            self.assertEqual(E.integrity_unavailable(tmp, "2026-01-02"), {"a", "b"})    # sizing
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_x14_stale_previous_book_name_exits_tagged(self):
        """A stale-tail name that was in the previous book leaves it as a tagged
        DATA_INTEGRITY_EXCLUSION, never an alpha-driven sell."""
        syms, dbf, unif = self.prev_book_full()
        prev = list(dbf.loc[dbf["target_weight"] > 0, "symbol"])
        x = prev[5]
        asof = str(CAL[-1])[:10]
        row = [x, "DATA_INTEGRITY_FAILURE", "STALE_TAIL", "2026-09-21", asof, "STALE_TAIL", False]
        db, uni, md = self.build(syms, [s for s in syms if s != x], [row], prev_book=prev)
        r = db.set_index("symbol").loc[x]
        self.assertEqual(r["action"], "SELL")
        self.assertTrue(E.is_integrity_exit(r["caveats"]))
        self.assertEqual(int((db["target_weight"] > 0).sum()),
                         int((dbf["target_weight"] > 0).sum()))            # cardinality preserved

    def test_x7_reference_universe_counts_unavailable_names(self):
        self.assertEqual(E.reference_universe_n(["a", "b"], {"c"}), 3)
        self.assertEqual(E.reference_universe_n(["a", "b"], {"c"}, eligible_pool=["a", "b"]), 2)
        self.assertEqual(E.reference_universe_n(["a", "b"], {"a"}), 2)
        tmp = tempfile.mkdtemp()
        try:
            pd.DataFrame([{"symbol": "c", "reason": "WINDOW_CROSSES_DATA_GAP"},
                          {"symbol": "d", "reason": "STALE_TAIL"}]).to_csv(
                os.path.join(tmp, "2026-01-02_data_integrity.csv"), index=False)
            self.assertEqual(E.integrity_window_excluded(tmp, "2026-01-02"), {"c"})
            self.assertEqual(E.integrity_window_excluded(tmp, "2026-01-05"), set())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def prev_book_full(self):
        """Full-universe book, then use its held names as the previous book."""
        import pipeline_gate as G
        syms = G.universe()
        dbf, unif, _ = self.build(syms, syms, [])
        return syms, dbf, unif

    def test_x8_E_book_only_invalid_inside_previous_book(self):
        """E: excluded, tagged DATA_INTEGRITY_EXCLUSION, next valid name fills the
        reference-universe slot, cardinality preserved."""
        syms, dbf, unif = self.prev_book_full()
        prev = list(dbf.loc[dbf["target_weight"] > 0, "symbol"])
        x = prev[3]
        full2, _, _ = self.build(syms, syms, [], prev_book=prev)          # prev book, all valid
        db, uni, md = self.build(syms, [s for s in syms if s != x], [self.row(x)], prev_book=prev)
        h0 = set(full2.loc[full2["target_weight"] > 0, "symbol"])
        h1 = set(db.loc[db["target_weight"] > 0, "symbol"])
        self.assertEqual(len(h1), len(h0))                                 # cardinality preserved
        self.assertNotIn(x, h1)
        self.assertEqual(len(h1 - h0), 1)                                  # one valid name fills the slot
        r = db.set_index("symbol").loc[x]
        self.assertEqual(r["action"], "SELL")                              # existing vocabulary
        self.assertTrue(E.is_integrity_exit(r["caveats"]))                 # machine-readable reason
        self.assertIn("signal_driven=false", r["caveats"])
        self.assertTrue(pd.isna(r["model_score"]) and pd.isna(r["rank"]))  # no stale score / rank
        self.assertNotIn(x, set(uni["symbol"]))
        self.assertIn("DATA_INTEGRITY_EXCLUSION — NOT AN ALPHA-DRIVEN SELL", md)
        self.assertIn("of which SELL = DATA_INTEGRITY_EXCLUSION", md)
        others = db[(db["action"] == "SELL") & (db["symbol"] != x)]
        self.assertFalse(others["caveats"].map(E.is_integrity_exit).any())  # alpha sells untouched

    def test_x9_D_book_only_invalid_outside_target_creates_no_exit(self):
        syms, dbf, unif = self.prev_book_full()
        prev = list(dbf.loc[dbf["target_weight"] > 0, "symbol"])
        x = unif.sort_values("rank")["symbol"].tolist()[-1]               # not in the previous book
        self.assertNotIn(x, prev)
        db, uni, md = self.build(syms, [s for s in syms if s != x], [self.row(x)], prev_book=prev)
        self.assertNotIn(x, set(db["symbol"]))                             # no synthetic exit record
        self.assertFalse(db["caveats"].map(E.is_integrity_exit).any())

    def test_x10_book_only_exit_never_a_user_sell_instruction(self):
        """A DATA_INTEGRITY_EXCLUSION SELL for a symbol absent from my_holdings.csv
        maps to NO_ACTION; only a real LONG could map to EXIT_LONG, and a real
        holding with invalid data blocks publication instead."""
        import holdings as H
        ua, pri, _ = H.map_user_action(position_side="NONE", model_action="SELL", model_target=0.0,
                                       in_universe=True, in_book=True, cmp_weight=np.nan)
        self.assertEqual(ua, "NO_ACTION")
        import user_holdings_overlay as O
        cls = O.classify({"symbol": "9999", "in_model_universe": True, "in_data_cache": True,
                          "model_action": "SELL", "model_integrity_exit": True, "weight_gap": np.nan,
                          "my_current_weight": 0.0, "my_shares": 1000.0, "both_sides": False,
                          "position_side": "LONG", "in_latest_decision_book": True}, 0.02, 0.05)
        self.assertIn("NOT AN ALPHA-DRIVEN SELL", cls[2])
        import simplified_reports as S
        self.assertEqual(S._model_status({"user_action": "NO_ACTION", "model_action": "SELL",
                                          "user_action_reason": "DATA_INTEGRITY_EXCLUSION — x"}),
                         "DATA_INTEGRITY_EXCLUSION")

    def test_x11_neural_target_book_tags_integrity_exit(self):
        import inference_transformer_eod as I
        self.assertEqual(I.E_INTEGRITY_EXIT_CAVEAT, E.INTEGRITY_EXIT_CAVEAT)
        rng = np.random.default_rng(7)
        pred = pd.DataFrame({"stock": [f"{1200 + i}" for i in range(107)], "score": rng.normal(size=107),
                             "score_std": 0.1, "sector": "x", "vol_20": 0.2})
        prev = pd.DataFrame({"symbol": ["9999", "1200"], "target_weight": [0.05, 0.05]})
        book = I.make_decision_book(pred, prev, 0.2, 0.05, 20, "t+1", ref_n=108,
                                    integrity_excluded={"9999", "8888"})
        r = book.set_index("symbol")
        self.assertEqual(r.loc["9999", "action"], "SELL")
        self.assertTrue(E.is_integrity_exit(r.loc["9999", "caveats"]))
        self.assertTrue(pd.isna(r.loc["9999", "prediction_score"]) and pd.isna(r.loc["9999", "rank"]))
        self.assertNotIn("8888", r.index)                                  # never in the previous book
        self.assertEqual(int((book["target_weight"] > 0).sum()), 22)

class TestInferencePolicy(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.cache = os.path.join(self.root, "research", "data_cache")
        os.makedirs(self.cache)
        import pipeline_gate as G
        for i, s in enumerate(G.universe()[:20]):
            write_cache(self.cache, s, CAL, i)
        write_cache(self.cache, "2883", CAL.delete(len(CAL) - 40), 3)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def integ(self):
        asof = str(CAL[-1])[:10]
        return pd.DataFrame([{"symbol": "2883", "status": "DATA_INTEGRITY_FAILURE",
                              "reason": "WINDOW_CROSSES_DATA_GAP", "last_cached_date": asof, "asof": asof}])

    def test_i1_held_symbol_invalid_blocks_inference_publication(self):
        import inference_transformer_eod as I
        pd.DataFrame({"symbol": ["2883"], "shares": [1000]}).to_csv(
            os.path.join(self.root, "my_holdings.csv"), index=False)
        df, pol = I.apply_integrity_policy(self.integ(), 107, str(CAL[-1])[:10], root=self.root)
        self.assertFalse(pol["publication_allowed"])
        self.assertTrue(bool(df["held"].iloc[0]))

    def test_x12_stale_tail_counts_for_sizing_not_for_coverage(self):
        """User decision 2026-09-24 (option 1), reproducing the 2026-09-22 run:
        106 valid + 2207 (invalid window) + 5903 (stale tail) -> sizing
        reference 108 (book 22), coverage ratio still 106/107."""
        import inference_transformer_eod as I
        asof = str(CAL[-1])[:10]
        integ = pd.DataFrame([
            {"symbol": "2883", "status": "DATA_INTEGRITY_FAILURE", "reason": "WINDOW_CROSSES_DATA_GAP",
             "last_cached_date": asof, "asof": asof},
            {"symbol": "5903", "status": "DATA_INTEGRITY_FAILURE", "reason": "STALE_TAIL",
             "last_cached_date": "2026-09-21", "asof": asof}])
        df, pol = I.apply_integrity_policy(integ, 106, asof, root=self.root)
        self.assertEqual(pol["status"], E.DEGRADED)
        self.assertEqual((pol["valid"], pol["eligible"]), (106, 107))       # coverage: window only
        self.assertAlmostEqual(pol["valid_ratio"], 106 / 107)
        self.assertEqual(pol["stale_tails"], ["5903"])
        self.assertEqual(pol["reference_eligible_universe_n"], 108)          # sizing: + stale tail
        rng = np.random.default_rng(11)
        pred = pd.DataFrame({"stock": [f"{1300 + i}" for i in range(106)], "score": rng.normal(size=106),
                             "score_std": 0.1, "sector": "x", "vol_20": 0.2})
        self.assertEqual(int((I.make_decision_book(pred, None, 0.2, 0.05, 20, "t+1",
                                                   ref_n=108)["target_weight"] > 0).sum()), 22)
        self.assertEqual(int((I.make_decision_book(pred, None, 0.2, 0.05, 20, "t+1",
                                                   ref_n=107)["target_weight"] > 0).sum()), 21)

    def test_i3_book_only_symbol_does_not_block_inference(self):
        import inference_transformer_eod as I
        pt = os.path.join(self.root, "reports", "paper_trading")
        os.makedirs(pt)
        pd.DataFrame({"symbol": ["2883"], "target_weight": [0.05]}).to_csv(
            os.path.join(pt, "2025-12-01_blend50_band10_decision_book.csv"), index=False)
        df, pol = I.apply_integrity_policy(self.integ(), 107, str(CAL[-1])[:10], root=self.root)
        self.assertTrue(pol["publication_allowed"])                        # book-only: DEGRADED
        self.assertFalse(bool(df["held"].iloc[0]))
        self.assertTrue(bool(df["in_previous_book"].iloc[0]))

    def test_i2_nonheld_symbol_degraded_with_detail(self):
        import inference_transformer_eod as I
        df, pol = I.apply_integrity_policy(self.integ(), 107, str(CAL[-1])[:10], root=self.root)
        self.assertEqual(pol["status"], E.DEGRADED)
        self.assertEqual((pol["eligible"], pol["valid"]), (108, 107))
        self.assertEqual(df["detail"].iloc[0], E.MISSING_SESSION_IN_REQUIRED_WINDOW)


class TestFrozenInputBook(unittest.TestCase):

    def test_17_frozen_input_decision_book_byte_identical(self):
        """Regenerate the latest runtime decision book from frozen inputs with
        THIS branch's code and compare bytes. Runtime files live in the
        production working tree (AIQUANT_RUNTIME_ROOT, default: this repo)."""
        rt = os.environ.get("AIQUANT_RUNTIME_ROOT", REPO)
        # the runtime cache is being repaired; the frozen comparison needs the
        # immutable pre-repair snapshot (AIQUANT_FROZEN_CACHE) when it exists
        frozen = os.environ.get("AIQUANT_FROZEN_CACHE", os.path.join(rt, "research", "data_cache"))
        pt = os.path.join(rt, "reports", "paper_trading")
        books = sorted(f for f in os.listdir(pt) if f.endswith("_blend50_band10_decision_book.csv")) \
            if os.path.isdir(pt) else []
        if not books or not os.path.isdir(frozen):
            self.skipTest("no runtime decision book / cache available")
        asof = books[-1][:10]
        import blended_decision_book as B
        import paper_trading as P
        saved = (D.CACHE_DIR, P.BOOK_DIR, B.ROOT, B.PT_DIR)
        out = tempfile.mkdtemp()
        try:
            D.CACHE_DIR = frozen
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
