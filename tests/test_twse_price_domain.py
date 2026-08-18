"""Unit tests for research/twse_price_domain.py (v16 Stage B2).

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_twse_price_domain -v
"""

import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "research"))

import twse_price_domain as tpd  # noqa: E402


class TestLegalTicks(unittest.TestCase):

    def test_official_anchor_40_60(self):
        d = tpd.build_domain("9999", 40.60)
        self.assertAlmostEqual(d.legal_limit_up, 44.65)
        self.assertAlmostEqual(d.legal_limit_down, 36.55)
        self.assertEqual(d.price_domain_status, "NORMAL_DAY_ASSUMPTION")
        self.assertEqual(d.reference_source, "PREVIOUS_CLOSE")
        self.assertAlmostEqual(d.tick_size_at_reference, 0.05)

    def test_tick_boundaries(self):
        cases = ((9.99, 0.01), (10.00, 0.05), (49.95, 0.05),
                 (50.00, 0.10), (99.9, 0.10), (100.0, 0.50),
                 (499.5, 0.50), (500.0, 1.00), (999.0, 1.00),
                 (1000.0, 5.00))
        for px, tick in cases:
            self.assertAlmostEqual(tpd.stock_tick(px), tick, msg=px)
            self.assertTrue(tpd.is_legal_tick(px), msg=px)

    def test_floor_ceil_never_illegal(self):
        rng = np.random.RandomState(0)
        for px in np.concatenate([rng.uniform(1, 1500, 500),
                                  [9.994, 10.02, 49.97, 50.04, 99.95,
                                   100.3, 499.7, 500.4, 999.5, 1001.0]]):
            lo, hi = tpd.legal_floor(px), tpd.legal_ceil(px)
            self.assertTrue(tpd.is_legal_tick(lo), msg=(px, lo))
            self.assertTrue(tpd.is_legal_tick(hi), msg=(px, hi))
            self.assertLessEqual(lo, px + 1e-9)
            self.assertGreaterEqual(hi, px - 1e-9)

    def test_limits_never_exceed_ten_percent(self):
        rng = np.random.RandomState(1)
        for ref in rng.uniform(5, 1200, 300):
            d = tpd.build_domain("9999", ref)
            self.assertLessEqual(d.legal_limit_up, ref * 1.10 + 1e-9)
            self.assertGreaterEqual(d.legal_limit_down, ref * 0.90 - 1e-9)
            self.assertTrue(tpd.is_legal_tick(d.legal_limit_up))
            self.assertTrue(tpd.is_legal_tick(d.legal_limit_down))


class TestDomainStatus(unittest.TestCase):

    def test_missing_reference_unknown(self):
        for bad in (None, float("nan"), 0.0, -5.0):
            d = tpd.build_domain("9999", bad)
            self.assertEqual(d.price_domain_status, "UNKNOWN")
            self.assertFalse(d.known())
            self.assertTrue(np.isnan(d.legal_limit_up))

    def test_etf_no_fabricated_limits(self):
        d = tpd.build_domain("0050", 106.45, security_type="etf")
        self.assertEqual(d.price_domain_status, "UNKNOWN")
        self.assertFalse(d.standard_limit_applicable)
        self.assertTrue(np.isnan(d.legal_limit_up))

    def test_never_confirmed_status(self):
        # the repo cannot confirm auction references -> never CONFIRMED
        d = tpd.build_domain("9999", 100.0)
        self.assertNotEqual(d.price_domain_status,
                            "CONFIRMED_STANDARD_LIMIT")


class TestClampValidate(unittest.TestCase):

    def setUp(self):
        self.d = tpd.build_domain("9999", 100.0)   # limits 90.0 / 110.0

    def test_clamp_near_limits(self):
        self.assertLessEqual(self.d.clamp(150.0, "sell"),
                             self.d.legal_limit_up)
        self.assertGreaterEqual(self.d.clamp(50.0, "buy"),
                                self.d.legal_limit_down)
        # inside-domain price projects to a legal tick
        v = self.d.clamp(100.37, "buy")
        self.assertTrue(tpd.is_legal_tick(v))
        self.assertLessEqual(v, 100.37)
        v = self.d.clamp(100.37, "sell")
        self.assertGreaterEqual(v, 100.37)

    def test_validate_raises_on_violation(self):
        with self.assertRaises(tpd.PriceDomainValidationError):
            self.d.validate(120.0, "test")           # outside domain
        with self.assertRaises(tpd.PriceDomainValidationError):
            self.d.validate(100.37, "test")          # illegal tick
        self.d.validate(104.5, "test")               # legal, inside
        self.d.validate(float("nan"), "test")        # NaN passes through

    def test_unknown_domain_validates_ticks_only(self):
        d = tpd.build_domain("9999", None)
        d.validate(104.5, "t")                       # tick check only
        with self.assertRaises(tpd.PriceDomainValidationError):
            d.validate(104.37, "t")

    def test_domain_position(self):
        self.assertAlmostEqual(self.d.position_of(90.0), 0.0)
        self.assertAlmostEqual(self.d.position_of(110.0), 1.0)
        self.assertAlmostEqual(self.d.position_of(100.0), 0.5)
        self.assertTrue(np.isnan(tpd.build_domain("x", None)
                                 .position_of(100.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
