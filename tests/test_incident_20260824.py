"""Regression tests for the 2026-08-24 daily-ops production incident:
name-cap infeasibility guard, pipeline fail-fast gates, stale-artifact
protection, refresh empty-payload detection.

Deterministic; no network, no GPU model loads (make_decision_book is
tested directly with synthetic prediction frames).

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_incident_20260824 -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

from inference_transformer_eod import make_decision_book  # noqa: E402
from transformer_portfolio import NAME_CAP  # noqa: E402
import pipeline_gate as pg  # noqa: E402


def pred_frame(n, seed=0, concentrated=False, dominant=False):
    rng = np.random.RandomState(seed)
    vol = np.full(n, 0.02)
    if concentrated:
        vol = rng.uniform(0.005, 0.08, n)
    if dominant:
        vol[0] = 1e-4                     # one name wants ~all the weight
    return pd.DataFrame({
        "stock": [f"9{i:03d}" for i in range(n)],
        "score": np.linspace(1, -1, n) + rng.normal(0, 1e-6, n),
        "score_std": rng.uniform(0.01, 0.05, n),
        "sector": ["other"] * n,
        "vol_20": vol,
    })


def build(pred, prev=None):
    return make_decision_book(pred, prev, top_frac=0.2, band=0.05,
                              horizon=20, exec_date="next")


class TestNameCapGuard(unittest.TestCase):

    # G1/G3: normal 108-name book obeys cap + sum invariant
    def test_normal_book(self):
        book = build(pred_frame(108))
        w = book["target_weight"]
        self.assertLessEqual(w.max(), NAME_CAP + 1e-4)
        self.assertAlmostEqual(w.sum(), 1.0, places=3)
        self.assertGreaterEqual(w.min(), 0.0)

    # G2: concentrated raw scores / vols still obey the cap
    def test_concentrated_vols(self):
        book = build(pred_frame(108, concentrated=True))
        self.assertLessEqual(book["target_weight"].max(), NAME_CAP + 1e-4)
        self.assertAlmostEqual(book["target_weight"].sum(), 1.0, places=3)

    # one dominant score/vol: waterfill caps it at exactly NAME_CAP
    def test_dominant_name_capped(self):
        book = build(pred_frame(108, dominant=True))
        self.assertLessEqual(book["target_weight"].max(), NAME_CAP + 1e-4)
        self.assertAlmostEqual(book["target_weight"].sum(), 1.0, places=3)

    # min-universe boundary: exactly 60 names (the research-standard
    # floor, k=12 -> cap feasible) builds a valid book
    def test_min_universe_boundary_feasible(self):
        book = build(pred_frame(60))
        w = book[book["target_weight"] > 0]["target_weight"]
        self.assertEqual(len(w), 12)
        self.assertLessEqual(w.max(), NAME_CAP + 1e-4)
        self.assertAlmostEqual(w.sum(), 1.0, places=3)

    # THE INCIDENT: 34 scored names -> thin-universe guard (the same
    # min_names=60 assumption as backtest_scores) fails clearly; the
    # cap-feasibility guard remains as defense in depth behind it
    def test_incident_34_names_fails_clearly(self):
        with self.assertRaises(RuntimeError) as cm:
            build(pred_frame(34))
        self.assertIn("too thin", str(cm.exception))
        self.assertIn("min_names=60", str(cm.exception))
        self.assertIn("partial EOD publication", str(cm.exception))

    def test_thin_universes_fail_clearly(self):
        for n in (3, 10, 20, 47, 50, 59):
            with self.assertRaises(RuntimeError, msg=n):
                build(pred_frame(n))

    # determinism: identical inputs -> identical books
    def test_deterministic(self):
        b1 = build(pred_frame(108, seed=5))
        b2 = build(pred_frame(108, seed=5))
        pd.testing.assert_frame_equal(b1, b2)


class TestPipelineGate(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="gate_test_")
        self.uni = pg.universe()
        for s in self.uni:
            self._cache(s, "2026-08-21")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _cache(self, sym, last_date):
        p = os.path.join(self.root, "research", "data_cache",
                         f"{sym}.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("date,open,high,low,close,volume\n"
                    f"2026-08-19,1,1,1,1,1\n{last_date},1,1,1,1,1\n")

    def _manifest(self, asof):
        p = os.path.join(self.root, "checkpoints", "transformer_eod",
                         "daily_manifest.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"asof": asof}, open(p, "w", encoding="utf-8"))

    def _artifact(self, rel):
        p = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write("x\n")

    # G6: full coverage passes; incident coverage (34/108) fails
    def test_refresh_gate(self):
        ok, _ = pg.check(self.root, "refresh")
        self.assertTrue(ok)
        # regress 74 names to 08-19 (the incident state)
        for s in self.uni[34:]:
            self._cache(s, "2026-08-19")
        ok, msg = pg.check(self.root, "refresh")
        self.assertFalse(ok)
        self.assertIn("partial publication", msg)

    # G5: stale manifest cannot masquerade as current
    def test_retrain_gate_stale_asof(self):
        self._manifest("2026-08-21")
        self.assertTrue(pg.check(self.root, "retrain")[0])
        self._manifest("2026-08-19")
        ok, msg = pg.check(self.root, "retrain")
        self.assertFalse(ok)
        self.assertIn("stale artifact", msg)

    # G4/G5: missing/stale inference artifacts fail the gate
    def test_inference_gate(self):
        ok, msg = pg.check(self.root, "inference")
        self.assertFalse(ok)
        self.assertIn("2026-08-21", msg)
        # a stale 08-19 artifact does NOT satisfy the 08-21 expectation
        for n in ("2026-08-19_predictions.csv",
                  "2026-08-19_target_book.csv",
                  "2026-08-19_metrics.json", "2026-08-19_report.md"):
            self._artifact(f"reports/transformer_gpu/{n}")
        self.assertFalse(pg.check(self.root, "inference")[0])
        for n in ("2026-08-21_predictions.csv",
                  "2026-08-21_target_book.csv",
                  "2026-08-21_metrics.json", "2026-08-21_report.md"):
            self._artifact(f"reports/transformer_gpu/{n}")
        self.assertTrue(pg.check(self.root, "inference")[0])

    def test_book_gate(self):
        self.assertFalse(pg.check(self.root, "book")[0])
        self._artifact("reports/paper_trading/"
                       "2026-08-21_blend50_band10_decision_book.csv")
        self.assertTrue(pg.check(self.root, "book")[0])

    # ---- review item 2: exit-code + artifact-freshness contract ----

    def _inference_artifacts(self, when_offset):
        import time as _t
        for n in ("2026-08-21_predictions.csv",
                  "2026-08-21_target_book.csv",
                  "2026-08-21_metrics.json", "2026-08-21_report.md"):
            p = os.path.join(self.root, "reports", "transformer_gpu", n)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", encoding="utf-8").write("x\n")
            os.utime(p, (
                _t.time() + when_offset, _t.time() + when_offset))

    def _marker(self):
        p = os.path.join(self.root, "step.marker")
        open(p, "w").close()
        return p

    def test_exitcode_valid_exit_fresh_artifacts_pass(self):
        m = self._marker()
        self._inference_artifacts(when_offset=+5)   # written after start
        ok, _ = pg.check(self.root, "inference", exit_code=0,
                         since_marker=m)
        self.assertTrue(ok)

    def test_exitcode_nonzero_with_stale_artifacts_aborts(self):
        # current-DATED artifacts exist but predate step start (earlier
        # run) + unexpected nonzero exit -> abort; a dated artifact
        # cannot excuse a crash
        m = self._marker()
        self._inference_artifacts(when_offset=-3600)
        ok, msg = pg.check(self.root, "inference", exit_code=134,
                           since_marker=m)
        self.assertFalse(ok)
        self.assertIn("cannot excuse a crash", msg)

    def test_exitcode_nonzero_missing_artifact_aborts(self):
        m = self._marker()
        self._inference_artifacts(when_offset=+5)
        os.remove(os.path.join(self.root, "reports", "transformer_gpu",
                               "2026-08-21_report.md"))
        ok, msg = pg.check(self.root, "inference", exit_code=1,
                           since_marker=m)
        self.assertFalse(ok)
        self.assertIn("missing", msg)

    def test_exitcode_zero_with_stale_mtime_aborts(self):
        # zero exit but no fresh output = silent no-op -> abort
        m = self._marker()
        self._inference_artifacts(when_offset=-3600)
        ok, msg = pg.check(self.root, "inference", exit_code=0,
                           since_marker=m)
        self.assertFalse(ok)
        self.assertIn("silent no-op", msg)

    def test_exitcode_teardown_whitelist(self):
        # the ONLY tolerated nonzero exit: every expected artifact
        # freshly written after step start (torch teardown case)
        m = self._marker()
        self._inference_artifacts(when_offset=+5)
        ok, _ = pg.check(self.root, "inference", exit_code=-1073740791,
                         since_marker=m)
        self.assertTrue(ok)

    def test_threshold_single_source(self):
        from user_next_session_plan import PARTIAL_COVERAGE_MIN
        self.assertEqual(pg.PARTIAL_COVERAGE_MIN, PARTIAL_COVERAGE_MIN)
        self.assertEqual(pg.PARTIAL_COVERAGE_MIN, 0.99)


class TestBatContract(unittest.TestCase):
    """G4/G7/G8: static contract — a failed step cannot reach steps 4-9,
    the standing plan survives, and no recovery flag is used."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REPO, "daily_ops.bat"),
                  encoding="utf-8") as f:
            cls.bat = f.read()

    def test_gates_wired_after_critical_steps(self):
        for stage in ("refresh", "retrain", "inference", "book"):
            self.assertIn(f"pipeline_gate.py {stage}", self.bat, stage)
        self.assertEqual(self.bat.count(
            r"python.exe research\pipeline_gate.py"), 4)
        # abort checks: 4 gate + snapshot/evaluate/diff plain ERRORLEVEL
        # + the book step's explicit STEP_RC abort
        self.assertEqual(
            self.bat.count("if errorlevel 1 goto :pipefail"), 7)
        self.assertIn('if not "%STEP_RC%"=="0" goto :pipefail', self.bat)
        # GPU steps pass exit code + step-start marker to the gate
        self.assertEqual(self.bat.count(
            "--exit-code %STEP_RC% --since-marker"), 3)
        self.assertEqual(self.bat.count('type nul > "%STEP_MARKER%"'), 3)

    def test_pipefail_block(self):
        self.assertIn(":pipefail", self.bat)
        self.assertIn("daily pipeline aborted", self.bat)
        self.assertIn("Previous standing plan remains untouched",
                      self.bat)
        self.assertIn("exit /b 1", self.bat)
        # pipefail must not fall through into overlay/step 9
        tail = self.bat.split(":pipefail")[1]
        self.assertNotIn("user_next_session_plan", tail)
        self.assertNotIn("user_holdings_overlay", tail)

    def test_no_deletion_no_recovery(self):
        import re
        # no deletion COMMAND at any line start ("model " contains the
        # raw substring "del ", so a blunt substring check is wrong)
        self.assertIsNone(re.search(r"(?mi)^\s*(del|erase|rd|rmdir)\s",
                                    self.bat))
        self.assertNotIn("--allow-current-book-recovery", self.bat)

    def test_ordering_gate_before_downstream(self):
        # the inference gate appears BEFORE the blended book invocation
        self.assertLess(self.bat.index("pipeline_gate.py inference"),
                        self.bat.index("blended_decision_book.py"))


class TestRefreshRetries(unittest.TestCase):
    """Review item 3: bounded rate-limit recovery."""

    def _setup(self, fetch_behavior):
        """fetch_behavior(call_no) -> list of day-objects or []."""
        import refresh_data as rd
        import data as data_mod

        calls = {"n": 0}

        class _Day:
            def __init__(self, date, px):
                self.date = date
                self.open = self.high = self.low = self.close = px
                self.capacity = 1000

        class _FakeStock:
            def __init__(self, sid):
                pass

            def fetch(self, y, m):
                calls["n"] += 1
                return fetch_behavior(calls["n"])

        import datetime as dtm
        self.day_rows = [_Day(dtm.date(2026, 8, 20), 1.0),
                         _Day(dtm.date(2026, 8, 21), 1.0)]
        fake = type(sys)("twstock")
        fake.Stock = _FakeStock
        self._old_tw = sys.modules.get("twstock")
        sys.modules["twstock"] = fake
        self.tmp = tempfile.mkdtemp(prefix="rdr_test_")
        self._old_dir = data_mod.CACHE_DIR
        data_mod.CACHE_DIR = self.tmp
        with open(os.path.join(self.tmp, "2330.csv"), "w",
                  encoding="utf-8") as f:
            f.write("date,open,high,low,close,volume\n"
                    "2026-08-19,1,1,1,1,1\n")
        self.rd = rd
        self.calls = calls

    def tearDown(self):
        import data as data_mod
        data_mod.CACHE_DIR = self._old_dir
        if self._old_tw is not None:
            sys.modules["twstock"] = self._old_tw
        else:
            sys.modules.pop("twstock", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_transient_empty_recovers(self):
        # empty on pass 1, data on the retry -> recovered, no suspects
        self._setup(lambda n: [] if n == 1 else self.day_rows)
        sleeps = []
        total, suspects = self.rd.refresh_with_retries(
            ["2330"], pd.Timestamp("2026-08-24"),
            cooldowns=(0.0, 0.0), sleep_fn=sleeps.append)
        self.assertEqual(total, 2)
        self.assertEqual(suspects, [])
        self.assertEqual(len(sleeps), 1)          # one cooldown used
        df = pd.read_csv(os.path.join(self.tmp, "2330.csv"))
        self.assertEqual(len(df), 3)              # 08-19/20/21, no dups
        self.assertEqual(df["date"].nunique(), 3)

    def test_persistent_empty_bounded_and_explicit(self):
        self._setup(lambda n: [])
        total, suspects = self.rd.refresh_with_retries(
            ["2330"], pd.Timestamp("2026-08-24"),
            cooldowns=(0.0, 0.0, 0.0), sleep_fn=lambda s: None)
        self.assertEqual(total, 0)
        self.assertEqual(suspects, ["2330"])      # explicit final failure
        self.assertEqual(self.calls["n"], 4)      # 1 pass + 3 bounded

    def test_no_duplicate_rows_on_repeat_fetch(self):
        # data returned on BOTH the first pass and a retry of another
        # symbol never duplicates rows (date>last + drop_duplicates)
        self._setup(lambda n: self.day_rows)
        self.rd.refresh_with_retries(["2330"], pd.Timestamp("2026-08-24"),
                                     cooldowns=(0.0,),
                                     sleep_fn=lambda s: None)
        added, susp = self.rd.refresh_stock(
            "2330", pd.Timestamp("2026-08-24"), throttle_s=0)
        self.assertEqual(added, 0)                # already current
        df = pd.read_csv(os.path.join(self.tmp, "2330.csv"))
        self.assertEqual(df["date"].nunique(), len(df))


class TestRefreshEmptyPayload(unittest.TestCase):
    """C-patch: empty twstock payloads are flagged suspect, not silent."""

    def test_refresh_stock_flags_empty(self):
        import refresh_data as rd

        class _FakeStock:
            def __init__(self, sid):
                pass

            def fetch(self, y, m):
                return []                 # rate-limit signature

        fake = type(sys)("twstock")
        fake.Stock = _FakeStock
        old = sys.modules.get("twstock")
        sys.modules["twstock"] = fake
        import data as data_mod
        tmp = tempfile.mkdtemp(prefix="rd_test_")
        try:
            old_dir = data_mod.CACHE_DIR
            data_mod.CACHE_DIR = tmp
            with open(os.path.join(tmp, "2330.csv"), "w",
                      encoding="utf-8") as f:
                f.write("date,open,high,low,close,volume\n"
                        "2026-08-19,1,1,1,1,1\n")
            added, suspect = rd.refresh_stock(
                "2330", pd.Timestamp("2026-08-24"), throttle_s=0)
            self.assertEqual(added, 0)
            self.assertGreaterEqual(suspect, 1)
            # up-to-date symbol is NOT suspect (no months needed)
            with open(os.path.join(tmp, "2330.csv"), "a",
                      encoding="utf-8") as f:
                f.write("2026-08-24,1,1,1,1,1\n")
        finally:
            data_mod.CACHE_DIR = old_dir
            if old is not None:
                sys.modules["twstock"] = old
            else:
                sys.modules.pop("twstock", None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
