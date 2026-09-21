"""v17 H-RESIDUAL-SIGNAL / H-FACTOR-PREMIUM — pure-function tests on synthetic
data. No real prediction cache, label or production file is read."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import h_residual_factor as R  # noqa: E402


def synth_date(rng, n=108, nonlinear=0.0, linear=1.0, idle=0.6):
    F = rng.standard_normal((n, 10))
    z = rng.standard_normal(n)                                   # a signal outside the input span
    w = rng.standard_normal(n)                                   # a non-predictive idiosyncratic component
    y = linear * F[:, 4] + nonlinear * z + rng.standard_normal(n)
    score = linear * F[:, 4] + 0.3 * F[:, 6] + nonlinear * z + idle * w
    g = pd.DataFrame(F, columns=R.FEATURES)
    g["mom_12_1"] = F[:, 4] + 0.5 * rng.standard_normal(n)
    g["fwd"], g["score"] = y, score
    return g


class TestResidualisation(unittest.TestCase):

    def test_pure_factor_repackaging_leaves_no_residual_signal(self):
        rng = np.random.default_rng(0)
        beta = np.zeros(10)
        beta[4] = 1.0
        ics, res = [], []
        for _ in range(200):
            m = R.date_metrics(synth_date(rng, nonlinear=0.0), beta)
            ics.append(m["IC_TF"])
            res.append(m["RESIDUAL_IC_R10"])
        self.assertGreater(np.mean(ics), 0.3)                   # the raw score ranks well ...
        self.assertLess(abs(np.mean(res)), 0.02)                # ... but nothing survives beyond its inputs

    def test_near_exact_repackaging_floor_is_small_in_additive_terms(self):
        """When the score is almost exactly its inputs, the residual is mostly
        transform error: its Spearman IC can reach a few hundredths, but its
        additive contribution is negligible. This is why the real-data
        PLACEBO floor is reported next to the residual IC."""
        rng = np.random.default_rng(7)
        beta = np.zeros(10)
        beta[4] = 1.0
        rows = [R.date_metrics(synth_date(rng, idle=0.02), beta) for _ in range(200)]
        self.assertGreater(np.mean([r["R2_R10"] for r in rows]), 0.97)
        self.assertLess(abs(np.mean([r["cov_res_R10"] for r in rows])), 0.01)

    def test_nonlinear_signal_survives_residualisation(self):
        rng = np.random.default_rng(1)
        beta = np.zeros(10)
        beta[4] = 1.0
        res = [R.date_metrics(synth_date(rng, nonlinear=1.0), beta)["RESIDUAL_IC_R10"] for _ in range(200)]
        self.assertGreater(np.mean(res), 0.15)

    def test_additive_attribution_is_exact(self):
        rng = np.random.default_rng(2)
        m = R.date_metrics(synth_date(rng, nonlinear=0.5), np.ones(10) / 10)
        self.assertAlmostEqual(m["cov_TF"], m["cov_fit_R10"] + m["cov_res_R10"], places=10)
        self.assertAlmostEqual(m["cov_TF"], m["cov_fit_R_mom"] + m["cov_res_R_mom"], places=10)

    def test_residual_uses_no_forward_return(self):
        rng = np.random.default_rng(3)
        g = synth_date(rng, nonlinear=0.5)
        s = R.rankz(g["score"])
        F = np.column_stack([R.rankz(g[c]) for c in R.FEATURES])
        _, r1 = R.residualize(s, F)
        g2 = g.copy()
        g2["fwd"] = rng.permutation(g2["fwd"].to_numpy())       # scramble the future
        s2 = R.rankz(g2["score"])
        _, r2 = R.residualize(s2, np.column_stack([R.rankz(g2[c]) for c in R.FEATURES]))
        self.assertTrue(np.allclose(r1, r2))


class TestNoFutureRows(unittest.TestCase):

    def test_train_mask_requires_matured_labels(self):
        pos = np.arange(0, 200)
        block_start = 150
        m = R.train_mask(pos, block_start)
        self.assertTrue(np.all(pos[m] + R.MATURE <= block_start - 1))
        self.assertEqual(int(pos[m].max()), block_start - 1 - R.MATURE)

    def test_ridge_ignores_rows_after_cutoff(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2020-01-01", periods=120)
        idx = pd.MultiIndex.from_product([dates, range(40)], names=["date", "stock"])
        Z = pd.DataFrame(rng.standard_normal((len(idx), 10)), index=idx, columns=R.FEATURES)
        Z["y"] = Z["mom_126_5"] * 0.1 + rng.standard_normal(len(idx)) * 0.01
        Z["pos"] = np.repeat(np.arange(len(dates)), 40)
        b1, n1, last1 = R.fit_ridge(Z, 100)
        Z2 = Z.copy()
        late = Z2["pos"].to_numpy() + R.MATURE > 99
        Z2.loc[late, "y"] = 99.0                                  # poison every unmatured row
        b2, n2, last2 = R.fit_ridge(Z2, 100)
        self.assertTrue(np.allclose(b1, b2))
        self.assertEqual(n1, n2)
        self.assertLess(pd.Timestamp(last1), dates[100])


class TestStatsAndClassifiers(unittest.TestCase):

    def test_hac_close_to_naive_for_iid(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(2000)
        naive = x.std(ddof=1) / np.sqrt(len(x))
        self.assertLess(abs(R.hac_se(x, 4) / naive - 1), 0.15)

    def test_hac_larger_for_overlapping_series(self):
        rng = np.random.default_rng(6)
        e = rng.standard_normal(3000)
        x = np.convolve(e, np.ones(20) / 20, mode="valid")      # 20-step overlap
        naive = x.std(ddof=1) / np.sqrt(len(x))
        self.assertGreater(R.hac_se(x, 20) / naive, 2.5)

    def test_tcrit(self):
        self.assertAlmostEqual(R.tcrit(2), 4.303)
        self.assertAlmostEqual(R.tcrit(26), 2.056)

    def _st(self, mean, lo, hi, pos, loro):
        return {"mean": mean, "ci95": [lo, hi], "share_positive": pos, "loro_min": loro}

    def test_residual_classes(self):
        C = R.classify_residual
        self.assertEqual(C(self._st(0.03, 0.005, 0.055, 0.7, 0.02)), "RESIDUAL_SIGNAL_PRESENT")
        self.assertEqual(C(self._st(0.001, -0.006, 0.008, 0.5, -0.001)), "RESIDUAL_SIGNAL_ABSENT")
        self.assertEqual(C(self._st(0.012, -0.01, 0.034, 0.6, 0.005)), "RESIDUAL_SIGNAL_WEAK")
        self.assertEqual(C(self._st(-0.01, -0.04, 0.02, 0.4, -0.02)), "RESIDUAL_SIGNAL_INCONCLUSIVE")
        self.assertEqual(C(self._st(0.03, 0.005, 0.055, 0.7, -0.001)), "RESIDUAL_SIGNAL_WEAK")   # one block drives it

    def test_factor_classes(self):
        C = R.classify_factor
        st = {"mean": 0.08, "ci95": [0.01, 0.15]}
        self.assertEqual(C(st, 0.8, 0.07, 0.09, 0.02), "FACTOR_PREMIUM_ROBUST")
        self.assertEqual(C(st, 0.8, 0.07, 0.09, -0.2), "FACTOR_PREMIUM_WEAK")          # prospective collapse blocks ROBUST
        self.assertEqual(C({"mean": 0.04, "ci95": [-0.1, 0.18]}, 0.55, 0.15, -0.07, None), "FACTOR_PREMIUM_REGIME_DEPENDENT")
        self.assertEqual(C({"mean": -0.02, "ci95": [-0.1, 0.06]}, 0.4, -0.01, -0.03, None), "FACTOR_PREMIUM_ABSENT")
        self.assertEqual(C({"mean": 0.01, "ci95": [-0.1, 0.12]}, 0.62, 0.02, 0.0, None), "FACTOR_PREMIUM_WEAK")


if __name__ == "__main__":
    unittest.main()
