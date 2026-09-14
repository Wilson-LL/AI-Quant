"""v17 0A — production seed behaviour, verified mechanically.

Production (`train_transformer_eod.mode_daily_retrain`) retrains the SAME
fixed seed IDs 0..6 from scratch every session; it does not draw new
seeds. Each fit calls torch.manual_seed(seed) + np.random.seed(seed), so
initialization is deterministic per seed; batch order comes from
torch.randperm under the same torch RNG. CUDA/cuDNN determinism flags
are NOT requested (AMP on GPU), so full-training bitwise reproducibility
is not guaranteed on CUDA — that is documented, not changed here.
"""

import inspect
import os
import sys
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research"))

import train_transformer_eod as tte  # noqa: E402


def _init(seed, cfg, input_dim=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = tte.build_net(input_dim, cfg)
    return [p.detach().cpu().clone() for p in net.parameters()]


class TestProductionSeeds(unittest.TestCase):

    def test_production_seed_ids_are_fixed_0_to_6(self):
        src = inspect.getsource(tte.mode_daily_retrain)
        self.assertIn("seeds = seeds or list(range(7))", src)
        self.assertNotIn("random.randint", src)
        self.assertNotIn("time.time()", src.split("seeds = seeds")[1].split("\n")[0])

    def test_fit_one_seeds_torch_and_numpy(self):
        src = inspect.getsource(tte.fit_one)
        self.assertIn("torch.manual_seed(seed)", src)
        self.assertIn("np.random.seed(seed)", src)
        # batch order is drawn from the (seeded) torch RNG, not Python random
        self.assertIn("torch.randperm", src)
        self.assertNotIn("random.shuffle", src)
        # cuDNN determinism is NOT requested (documented limitation)
        full = inspect.getsource(tte)
        self.assertNotIn("cudnn.deterministic", full)

    def test_same_seed_same_init_different_seed_different_init(self):
        cfg = tte.PRESETS["B"]
        a = _init(0, cfg)
        b = _init(0, cfg)
        c = _init(1, cfg)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(a, c)))

    def test_batch_order_deterministic_per_seed_on_cpu(self):
        torch.manual_seed(3)
        p1 = torch.randperm(1000)
        torch.manual_seed(3)
        p2 = torch.randperm(1000)
        self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
