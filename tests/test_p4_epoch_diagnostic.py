"""v17 P4-B0 / Policy D — mechanical guarantees (CPU, tiny synthetic
tensors, production fit_one unchanged):
1. Policy D returns the weights after EXACTLY epoch E (no restore);
2. the epoch diagnostic never early-stops and records every epoch;
3. the OOS block is future-only and separated from train/val;
4. OOS metrics do not influence optimisation / checkpoint choice;
5. refit-date selection is deterministic;
6. production Policy A remains the production split (already covered)."""

import copy
import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import train_transformer_eod as tte  # noqa: E402
import run_p4_validation_policy as p4  # noqa: E402

CFG = dict(hidden=8, trans_layers=1, lstm_layers=1, heads=2, ff=16, dropout=0.0,
           seq_len=6, max_epochs=25, patience=3, seeds=1)


def tiny(n=480, n_dates=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, CFG["seq_len"], 10, generator=g)
    y = torch.tanh(X[:, -1, 0] + 0.1 * torch.randn(n, generator=g))
    dr = np.repeat(np.arange(n_dates), n // n_dates)
    return X, y, dr


def snapshots_per_epoch(E, seed=1):
    """Run fit_fixed_epochs while snapshotting the weights after every epoch."""
    X, y, dr = tiny()
    tr = np.arange(0, 360)
    snaps = []
    orig = tte.predict_idx

    def hook(net, Xg, idx, batch=8192):
        snaps.append(copy.deepcopy(net.state_dict()))
        return orig(net, Xg, idx, batch)
    tte.predict_idx = hook
    try:
        net, info = p4.fit_fixed_epochs(tte, X, y, tr, dr, CFG, seed, E)
    finally:
        tte.predict_idx = orig
    return net, info, snaps


class TestPolicyDExact(unittest.TestCase):

    def test_returns_final_epoch_weights_not_best(self):
        E = 4
        net, info, snaps = snapshots_per_epoch(E)
        self.assertEqual(info["epochs_run"], E)
        self.assertEqual(len(snaps), E)                    # one snapshot per epoch
        final = net.state_dict()
        for k in final:                                    # == weights after epoch E
            self.assertTrue(torch.equal(final[k], snaps[-1][k]), k)
        self.assertTrue(any(not torch.equal(snaps[0][k], final[k]) for k in final))
        self.assertEqual(info["best_val_ic"], -1e9)        # never restored

    def test_exactly_E_epochs_no_early_stop(self):
        for E in (1, 3, 7):
            _, info, snaps = snapshots_per_epoch(E)
            self.assertEqual(info["epochs_run"], E)
            self.assertEqual(len(snaps), E)

    def test_same_seed_same_init_as_production_path(self):
        X, y, dr = tiny()
        tr = np.arange(0, 360)
        va = np.arange(360, 480)
        first = {}
        orig = tte.predict_idx

        def hook(net, Xg, idx, batch=8192):
            if "w" not in first:
                first["w"] = copy.deepcopy(net.state_dict())
            return orig(net, Xg, idx, batch)
        tte.predict_idx = hook
        try:
            tte.fit_one(X, y, tr, va, dr[va], CFG, seed=5, max_epochs=1, min_epochs=1)
            prod_after_ep1 = first.pop("w")
            p4.fit_fixed_epochs(tte, X, y, tr, dr, CFG, 5, 1)
            d_after_ep1 = first.pop("w")
        finally:
            tte.predict_idx = orig
        # same seed, same training rows -> identical epoch-1 weights (same
        # init, same batch permutation, same optimizer path)
        for k in prod_after_ep1:
            self.assertTrue(torch.equal(prod_after_ep1[k], d_after_ep1[k]), k)

    def test_train_rows_only(self):
        X, y, dr = tiny()
        tr = np.arange(0, 300)
        # rows >= 300 are never touched: perturbing them cannot change the result
        y2 = y.clone()
        y2[300:] = 99.0
        n1, _ = p4.fit_fixed_epochs(tte, X, y, tr, dr, CFG, 2, 2)
        n2, _ = p4.fit_fixed_epochs(tte, X, y2, tr, dr, CFG, 2, 2)
        for k, v in n1.state_dict().items():
            self.assertTrue(torch.equal(v, n2.state_dict()[k]), k)


class TestEpochDiagnostic(unittest.TestCase):

    def _run(self, H=5, mutate_oos=False):
        X, y, dr = tiny()
        tr, va = np.arange(0, 300), np.arange(300, 360)
        oos = np.arange(384, 480)                       # dates 32..39 > val
        oos_dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(dr[oos], "D")
        fwd = (y[oos].numpy() + 0.05 * np.random.default_rng(1).normal(size=len(oos)))
        if mutate_oos:
            fwd = -fwd
        return p4.epoch_curve(tte, X, y, tr, va, dr[va], CFG, 3, H, oos, oos_dates, fwd)

    def test_no_early_stop_and_every_epoch_recorded(self):
        H = 5
        curve = self._run(H)
        self.assertEqual([c["epoch"] for c in curve], list(range(1, H + 1)))
        for c in curve:
            for k in ("train_loss", "val_ic", "val_loss", "oos_ic", "oos_pred_std"):
                self.assertIn(k, c)
                self.assertTrue(np.isfinite(c[k]), k)

    def test_oos_never_influences_training(self):
        a = self._run(4)
        b = self._run(4, mutate_oos=True)      # flip OOS labels -> different oos_ic ...
        self.assertNotEqual(a[-1]["oos_ic"], b[-1]["oos_ic"])
        for x, z in zip(a, b):                   # ... but identical training trajectory
            self.assertEqual(x["train_loss"], z["train_loss"])
            self.assertEqual(x["val_ic"], z["val_ic"])
            self.assertEqual(x["val_loss"], z["val_loss"])

    def test_production_stop_simulation(self):
        self.assertEqual(p4.simulate_production_stop([0.1, 0.2, 0.15, 0.1, 0.05, 0.3]), 2)
        self.assertEqual(p4.simulate_production_stop([0.1, 0.2, 0.3, 0.4]), 4)
        self.assertEqual(p4.simulate_production_stop([float("nan")] * 5), 1)

    def test_deterministic_refit_dates(self):
        dates = np.array(pd.bdate_range("2025-11-01", periods=250), dtype="datetime64[ns]")
        a = p4.b0_refit_dates(dates, "2026-01-05", "2026-07-23", 5, 9)
        b = p4.b0_refit_dates(dates, "2026-01-05", "2026-07-23", 5, 9)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 9)
        refits, s, e = p4.plan_refits(dates, "2026-01-05", "2026-07-23", 5)
        self.assertEqual(a[0], refits[0])
        self.assertEqual(a[-1], refits[-1])
        self.assertTrue(all(x in refits for x in a))


class TestOOSSeparation(unittest.TestCase):

    def test_oos_block_future_only(self):
        # synthetic data dict with the production split: OOS block starts
        # at refit_rank + 1 and every train/val label ends <= refit_rank
        n_dates, n_stocks = 900, 6
        dr = np.repeat(np.arange(n_dates), n_stocks)
        data = {"date_rank": dr, "targets": {"tgt_rank_20": np.random.default_rng(0).normal(size=len(dr))},
                "label_end_rank": {20: dr + 21}, "exec_lag": 1}
        import dataset_transformer_eod as dte
        r0 = 700
        tr, va, _ = dte.matured_train_val(data, "tgt_rank_20", r0 - 1, 20)
        oos = np.nonzero((dr >= r0) & (dr <= r0 + 4))[0]
        self.assertLessEqual(int(data["label_end_rank"][20][tr].max()), r0 - 1)
        self.assertLessEqual(int(data["label_end_rank"][20][va].max()), r0 - 1)
        self.assertGreater(int(dr[oos].min()), int(dr[va].max()))
        self.assertGreaterEqual(int(dr[oos].min()) - int(dr[va].max()), 21)   # val ends 21 before refit


if __name__ == "__main__":
    unittest.main()
