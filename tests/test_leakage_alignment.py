"""v17 B1/B4/B5 — mechanical leakage / temporal-alignment proofs.

These are NOT code-reading checks: they perturb synthetic data and assert
that (a) every production feature at date T is invariant to prices after
T, (b) the target at T depends only on closes in (T+1 .. T+21], and (c)
the matured train/validation split never lets a training label overlap
a validation feature date (purge >= horizon + exec_lag).
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import dataset_transformer_eod as dte  # noqa: E402


def synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": dates, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c,
                         "volume": rng.integers(1000, 5000, n)})


class TestFeatureCausality(unittest.TestCase):

    def test_close_only_features_ignore_future(self):
        df = synth()
        base = dte._stock_features(df.copy(), "close_only")
        T = 250
        pert = df.copy()
        # scramble EVERYTHING after T (levels, not just noise)
        pert.loc[T + 1:, "close"] = pert.loc[T + 1:, "close"] * \
            np.linspace(1.5, 0.3, len(pert) - T - 1)
        pert.loc[T + 1:, "volume"] = 1
        feat = dte._stock_features(pert, "close_only")
        cols = list(dte.FEATURE_COLS["close_only"])
        a = base.loc[:T, cols].to_numpy()
        b = feat.loc[:T, cols].to_numpy()
        self.assertTrue(np.allclose(a, b, equal_nan=True),
                        "a close_only feature at t<=T changed when prices "
                        "after T changed — look-ahead leak")

    def test_features_do_change_with_past(self):
        # sanity: the invariance above is not because features are constant
        df = synth()
        base = dte._stock_features(df.copy(), "close_only")
        pert = df.copy()
        pert.loc[:100, "close"] *= 2.0
        feat = dte._stock_features(pert, "close_only")
        cols = list(dte.FEATURE_COLS["close_only"])
        self.assertFalse(np.allclose(base.loc[150:200, cols].to_numpy(),
                                     feat.loc[150:200, cols].to_numpy(),
                                     equal_nan=True))


class TestTargetAlignment(unittest.TestCase):

    def test_fwd_ret_window_is_t_plus_1_to_t_plus_21(self):
        c = np.arange(1.0, 101.0)           # close = index+1
        f = dte._fwd_ret(c, 20, 1)
        T = 10
        expect = c[T + 21] / c[T + 1] - 1.0
        self.assertAlmostEqual(f[T], expect)
        # the close at T itself must not appear in the label
        c2 = c.copy()
        c2[T] = 999.0
        self.assertAlmostEqual(dte._fwd_ret(c2, 20, 1)[T], expect)
        # and nothing beyond T+21 either
        c3 = c.copy()
        c3[T + 22:] = 0.5
        self.assertAlmostEqual(dte._fwd_ret(c3, 20, 1)[T], expect)
        # last 21 entries are NaN (immature)
        self.assertTrue(np.isnan(f[-21:]).all())
        self.assertTrue(np.isfinite(f[-22]))


class TestMaturedSplit(unittest.TestCase):

    def _data(self, n_dates=600, n_stocks=5, horizon=20, exec_lag=1):
        # minimal synthetic structure mirroring build_dataset outputs
        dr = np.repeat(np.arange(n_dates), n_stocks)
        y = np.random.default_rng(1).normal(size=len(dr))
        le = dr + exec_lag + horizon
        return {"date_rank": dr, "targets": {"tgt_rank_20": y},
                "label_end_rank": {horizon: le}, "exec_lag": exec_lag}

    def test_purge_and_maturity(self):
        data = self._data()
        refit = 500
        tr, va, w = dte.matured_train_val(data, "tgt_rank_20", refit, 20)
        dr, le = data["date_rank"], data["label_end_rank"][20]
        # every label used (train AND val) matured on/before refit
        self.assertLessEqual(int(le[tr].max()), refit)
        self.assertLessEqual(int(le[va].max()), refit)
        # no training label window reaches into validation feature dates
        self.assertLess(int(le[tr].max()), int(dr[va].min()) + 1)
        self.assertGreaterEqual(int(dr[va].min()) - int(dr[tr].max()), 21)
        # the OOS block (> refit) is untouched by both
        self.assertTrue((dr[tr] <= refit).all() and (dr[va] <= refit).all())
        self.assertEqual(float(w.min()), 1.0)   # production: equal weights

    def test_no_train_val_overlap(self):
        data = self._data()
        tr, va, _ = dte.matured_train_val(data, "tgt_rank_20", 450, 20)
        self.assertEqual(len(set(tr) & set(va)), 0)


if __name__ == "__main__":
    unittest.main()
