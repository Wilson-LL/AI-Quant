"""v17 P4-B — the research split policies never use post-refit
information and respect the purge; policy A maps to the production split.
Synthetic data; CPU only; no GPU, no production files."""

import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import run_p4_validation_policy as p4  # noqa: E402
import dataset_transformer_eod as dte  # noqa: E402


def synth(n_dates=900, n_stocks=6, horizon=20, exec_lag=1, seed=0):
    dr = np.repeat(np.arange(n_dates), n_stocks)
    y = np.random.default_rng(seed).normal(size=len(dr))
    le = dr + exec_lag + horizon
    return {"date_rank": dr, "targets": {"tgt_rank_20": y},
            "label_end_rank": {horizon: le}, "exec_lag": exec_lag}


class TestRecentSplits(unittest.TestCase):

    def _check(self, data, tr, va, refit, n_val):
        dr, le = data["date_rank"], data["label_end_rank"][20]
        self.assertLessEqual(int(le[tr].max()), refit)      # matured only
        self.assertLessEqual(int(le[va].max()), refit)
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertGreaterEqual(int(dr[va].min()) - int(dr[tr].max()), p4.PURGE)
        vd = np.unique(dr[va])
        self.assertEqual(len(vd), n_val)                     # exact block length
        self.assertTrue((np.diff(vd) == 1).all())            # contiguous
        self.assertEqual(int(vd.max()), refit - p4.PURGE)    # ends 21 before refit
        self.assertTrue((dr[tr] <= refit).all() and (dr[va] <= refit).all())

    def test_recent126_and_63(self):
        data = synth()
        for n_val in (126, 63):
            tr, va = p4.split_recent(data, 700, n_val)
            self._check(data, tr, va, 700, n_val)

    def test_policy_dispatch(self):
        data = synth()
        trB, vaB = p4.split_for("B_recent126", data, 700)
        trD, vaD = p4.split_for("D_calibrate_then_train_all", data, 700)
        self.assertTrue(np.array_equal(trB, trD) and np.array_equal(vaB, vaD))
        trC, vaC = p4.split_for("C_recent63", data, 700)
        self.assertEqual(len(np.unique(data["date_rank"][vaC])), 63)
        trA, vaA = p4.split_for("A_current", data, 700)
        trP, vaP, _ = dte.matured_train_val(data, "tgt_rank_20", 700, 20)
        self.assertTrue(np.array_equal(trA, trP) and np.array_equal(vaA, vaP))

    def test_train_plus_val_for_policy_D_has_no_future(self):
        data = synth()
        tr, va = p4.split_recent(data, 700, 126)
        both = np.concatenate([tr, va])
        self.assertLessEqual(int(data["label_end_rank"][20][both].max()), 700)

    def test_key_includes_policy_cadence_seed_target_hash(self):
        k = p4.key("B_recent126", "2026-01-05", 1, "abcd1234")
        self.assertEqual(k, "B_recent126_2026-01-05_c5_s1_tgt_rank_20_abcd1234.npz")


if __name__ == "__main__":
    unittest.main()
