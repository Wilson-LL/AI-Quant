"""v17 P4-B0-LONG amended evaluator (research/p4_long_eval.py) — pure-function
tests on synthetic data. No LONG artifact is read by these tests."""

import copy
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import p4_long_eval as E  # noqa: E402

TOL = {"tier1_epochs_1_3_every_cell": {"abs_train_loss": 1e-4, "abs_val_ic": 0.002,
                                       "abs_val_loss": 2e-4, "rel_oos_pred_std": 0.05},
       "tier2_epochs_4_15_every_cell": {"abs_train_loss": 5e-4, "abs_val_ic": 0.01,
                                        "abs_val_loss": 1e-3, "rel_oos_pred_std": 0.25},
       "tier2_pooled_epochs_4_15": {"mean_abs_val_ic": 0.005, "abs_mean_signed_val_ic": 0.003}}
REFITS = ("2026-01-05", "2026-04-20", "2026-07-23")


def fake_cells(horizon=15):
    b0, lg = [], []
    rng = np.random.default_rng(0)
    for rd in REFITS:
        for s in (0, 1, 2):
            curve = [{"epoch": e, "train_loss": 0.333 - 1e-4 * e, "val_ic": float(rng.normal(0.05, 0.02)),
                      "val_loss": 0.332 + 1e-4 * e, "oos_pred_std": 0.01 * e, "oos_ic": 0.1}
                     for e in range(1, horizon + 1)]
            meta = {"refit_date": rd, "seed": s, "horizon": horizon, "train_end": "x", "val_start": "a",
                    "val_end": "b", "oos_start": rd, "oos_end": rd}
            b0.append({**meta, "curve": copy.deepcopy(curve)})
            lg.append({**meta, "curve": [{k: v for k, v in c.items() if k != "oos_ic"} for c in curve]})
    return b0, lg


class TestClustered(unittest.TestCase):

    def test_refit_is_the_unit_and_seeds_do_not_add_n(self):
        df = pd.DataFrame({"refit_date": np.repeat(["a", "b", "c"], 3), "seed": [0, 1, 2] * 3,
                           "x": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]})
        r = E.clustered(df, "x")
        self.assertAlmostEqual(r["mean"], 0.2)
        self.assertAlmostEqual(r["se_refit"], 0.1 / np.sqrt(3))       # SD of 3 refit means / sqrt(3)
        self.assertEqual(r["n_refits"], 3)
        self.assertEqual(r["t_df"], 2)
        self.assertAlmostEqual(r["ci95_t"][1] - r["mean"], 4.303 * r["se_refit"])
        self.assertAlmostEqual(r["loro_range"][0], 0.15)
        self.assertAlmostEqual(r["loro_range"][1], 0.25)
        self.assertEqual(r["positive_refits"], 3)


class TestLateMaxNull(unittest.TestCase):

    def test_pure_noise_is_not_flagged(self):
        rng = np.random.default_rng(3)
        ser = [rng.normal(0.05, 0.05, 71) for _ in range(9)]
        res = E.late_max_null(ser, np.repeat(list(REFITS), 3), n_sim=4000, seed=1)
        self.assertLess(abs(res["observed_S_max_minus_mean"] - res["null_mean_S"]), 0.03)
        self.assertLess(res["null_percentile_of_observed"], 0.99)
        self.assertGreater(res["null_mean_S"], 0.08)                  # a raw max of 71 noisy epochs is biased up

    def test_real_late_spike_exceeds_null(self):
        rng = np.random.default_rng(4)
        ser = []
        for _ in range(9):
            x = rng.normal(0.0, 0.01, 5)
            x[-1] += 0.2
            ser.append(x)
        res = E.late_max_null(ser, np.repeat(list(REFITS), 3), n_sim=4000, seed=1)
        self.assertGreater(res["raw_mean_max"], res["window_mean"])
        self.assertIn("null_corrected_max", res)


class TestAnchor(unittest.TestCase):

    def test_identical_passes(self):
        b0, lg = fake_cells()
        res, _ = E.anchor_compare(lg, b0, TOL)
        self.assertEqual(res["outcome"], "PARITY_PASS")

    def test_tier1_violation_fails(self):
        b0, lg = fake_cells()
        lg[0]["curve"][1]["val_ic"] += 0.01                              # epoch 2, beyond 0.002
        res, _ = E.anchor_compare(lg, b0, TOL)
        self.assertEqual(res["outcome"], "LONG_HARNESS_PARITY_FAILURE")

    def test_small_late_drift_is_tolerated(self):
        b0, lg = fake_cells()
        lg[0]["curve"][13]["val_ic"] += 0.02                             # one cell-epoch at epoch 14
        res, _ = E.anchor_compare(lg, b0, TOL)
        self.assertEqual(res["outcome"], "PARITY_PASS_WITH_LATE_DRIFT")

    def test_split_metadata_mismatch_fails(self):
        b0, lg = fake_cells()
        lg[4]["train_end"] = "y"
        res, _ = E.anchor_compare(lg, b0, TOL)
        self.assertEqual(res["outcome"], "LONG_HARNESS_PARITY_FAILURE")

    def test_missing_cell_fails(self):
        b0, lg = fake_cells()
        res, _ = E.anchor_compare(lg[:8], b0, TOL)
        self.assertEqual(res["outcome"], "LONG_HARNESS_PARITY_FAILURE")


class TestLoaderDropsOutcome(unittest.TestCase):

    def test_oos_ic_never_returned_unless_whitelisted(self):
        with tempfile.TemporaryDirectory() as d:
            rec = {"refit_date": "2026-01-05", "seed": 0, "horizon": 2, "curve": [
                {"epoch": 1, "train_loss": 0.3, "val_ic": 0.1, "oos_ic": 0.9},
                {"epoch": 2, "train_loss": 0.29, "val_ic": 0.12, "oos_ic": 0.8}]}
            with open(os.path.join(d, "long_x.json"), "w") as f:
                json.dump(rec, f)
            got = E.load_long(("train_loss", "val_ic"), ldir=d)
            self.assertNotIn("oos_ic", json.dumps(got))
            self.assertEqual(got[0]["curve"][1]["val_ic"], 0.12)


def mech_frame(val_ic_late=0.03, disp_late=1.0, w_rate_late=1.0):
    rows = []
    for rd in REFITS:
        for e in E.CHECKPOINTS:
            late = e >= 50
            rows.append({"refit_date": rd, "epoch": e, "train_loss": 0.333 - 1e-4 * e,
                         "val_ic": (val_ic_late if late else 0.05 if e < 15 else 0.03),
                         "val_loss": 0.33, "oos_pred_std": 0.01 * min(e, 50) * (disp_late if late else 1),
                         "val_pred_sd": 0.01 * min(e, 50) * (disp_late if late else 1) + (0.001 * e if disp_late > 1 else 0),
                         "grad_norm_mean": 0.5, "grad_norm_max": 0.9,
                         "weight_norm": 20 + 0.1 * e * (w_rate_late if late else 1),
                         "churn": 0.05, "agree": 0.8})
    return pd.DataFrame(rows).set_index(["refit_date", "epoch"])


class TestMechanismClassifier(unittest.TestCase):

    def test_flat_late_regime_is_no_phase_change(self):
        R = mech_frame(val_ic_late=0.03)
        R["val_pred_sd"] = R["val_pred_sd"] + 0.002 * R.index.get_level_values(1)   # keeps expanding
        out = E.classify_mechanism(R)
        self.assertEqual(out["Q2"]["label"], "NO")
        self.assertIn(out["Q6"]["classification"], ("MECHANISM_NO_PHASE_CHANGE", "MECHANISM_WEAK_PHASE_CHANGE"))

    def test_val_recovery_with_plateau_is_clear(self):
        R = mech_frame(val_ic_late=0.08)            # +0.05 over epoch 15 on every refit
        out = E.classify_mechanism(R)
        self.assertEqual(out["Q2"]["label"], "YES")
        self.assertEqual(out["Q3"]["label"], "PLATEAU")
        self.assertEqual(out["Q6"]["classification"], "MECHANISM_CLEAR_PHASE_CHANGE")


class TestVerdict(unittest.TestCase):

    def test_rejected_needs_unanimous_decline_and_no_mechanism(self):
        v = E.long_verdict("A_CONTINUED_DECAY", False, "MECHANISM_NO_PHASE_CHANGE",
                           {r: "A_CONTINUED_DECAY" for r in REFITS}, 0, [-0.1] * 9,
                           {r: -0.1 for r in REFITS}, 0.02, -0.3)
        self.assertEqual(v, "REJECTED")

    def test_decline_with_weak_mechanism_is_weakly_contradicted(self):
        v = E.long_verdict("B_PLATEAU", False, "MECHANISM_WEAK_PHASE_CHANGE",
                           {r: "B_PLATEAU" for r in REFITS}, 0, [-0.05] * 9,
                           {r: -0.05 for r in REFITS}, 0.03, -0.3)
        self.assertEqual(v, "WEAKLY_CONTRADICTED")

    def test_unsupported_recovery_with_mechanism_is_inconclusive(self):
        v = E.long_verdict("D_FULL_RECOVERY", False, "MECHANISM_WEAK_PHASE_CHANGE",
                           {r: "D_FULL_RECOVERY" for r in REFITS}, 3, [0.01] * 9,
                           {r: 0.01 for r in REFITS}, 0.05, -0.2)
        self.assertEqual(v, "INCONCLUSIVE")

    def test_supported_needs_everything(self):
        args = ("E_LATE_IMPROVEMENT_BEYOND_EARLY_PEAK", True, "MECHANISM_CLEAR_PHASE_CHANGE",
                {r: "E_LATE_IMPROVEMENT_BEYOND_EARLY_PEAK" for r in REFITS}, 3, [0.1] * 9,
                {r: 0.1 for r in REFITS}, 0.01, 0.02)
        self.assertEqual(E.long_verdict(*args), "SUPPORTED")
        weaker = list(args)
        weaker[2] = "MECHANISM_WEAK_PHASE_CHANGE"
        self.assertEqual(E.long_verdict(*weaker), "WEAKLY_SUPPORTED")

    def test_trajectory_class_boundaries(self):
        self.assertEqual(E.trajectory_class(0.15, 0.13, 0.05, 0.00), "A_CONTINUED_DECAY")
        self.assertEqual(E.trajectory_class(0.15, 0.13, 0.05, 0.06), "B_PLATEAU")
        self.assertEqual(E.trajectory_class(0.15, 0.13, 0.05, 0.08), "C_PARTIAL_RECOVERY")
        self.assertEqual(E.trajectory_class(0.15, 0.13, 0.05, 0.12), "D_FULL_RECOVERY")
        self.assertEqual(E.trajectory_class(0.15, 0.13, 0.05, 0.18), "E_LATE_IMPROVEMENT_BEYOND_EARLY_PEAK")


class TestOOSGuard(unittest.TestCase):

    def test_untracked_file_is_not_frozen(self):
        with tempfile.NamedTemporaryFile(dir=REPO, suffix=".json", delete=False) as f:
            p = f.name
        try:
            self.assertFalse(E.git_frozen(p))
        finally:
            os.remove(p)

    def test_committed_file_is_frozen(self):
        self.assertTrue(E.git_frozen(os.path.join(REPO, "reports", "model_audit", "v17", "20_p4_b0_long_amendment.md")))


if __name__ == "__main__":
    unittest.main()
