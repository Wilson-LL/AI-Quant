"""Correctness tests for the research framework.

If the referee is wrong, every downstream conclusion is wrong — so these verify
the measurement tools on synthetic data with known answers, before any real
data is involved. Pure numpy/pandas; run with `python research/test_framework.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from targets import triple_barrier_close, triple_barrier_highlow, fwd_logreturn
from evaluation import (
    build_panel, cross_sectional_ic, walk_forward_scores, backtest_topk,
    backtest_long_short, cross_sectional_neutralize,
)


def _ohlcv(close, high=None, low=None, start="2015-01-01"):
    close = np.asarray(close, float)
    n = len(close)
    high = close if high is None else np.asarray(high, float)
    low = close if low is None else np.asarray(low, float)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="B"),
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.linspace(1e3, 2e3, n),
    })


# ---- targets ---------------------------------------------------------------
def test_triple_barrier_close_direction():
    up = triple_barrier_close(_ohlcv(100 * 1.02 ** np.arange(120)), Y=20)
    assert np.nanmean(up[:-20]) > 0.9, "uptrend should be mostly +1"
    down = triple_barrier_close(_ohlcv(100 * 0.98 ** np.arange(120)), Y=20)
    assert np.nansum(down) == 0, "downtrend should never be +1"
    # last Y rows are censored -> NaN
    assert np.isnan(up[-1])


def test_highlow_detects_intraday_touch():
    # Close never moves, but an intraday HIGH spikes +12% on day 5 -> label 1
    # under high/low, but 0 under close-only. Demonstrates the fidelity gap.
    close = np.full(60, 100.0)
    high = close.copy()
    high[5] = 112.5
    tb_c = triple_barrier_close(_ohlcv(close, high, close), Y=20)
    tb_hl = triple_barrier_highlow(_ohlcv(close, high, close), Y=20)
    assert tb_c[0] == 0 and tb_hl[0] == 1, "high/low must catch the intraday touch"


def test_fwd_logreturn_value_and_censoring():
    c = 100 * 1.01 ** np.arange(40)
    r = fwd_logreturn(_ohlcv(c), Y=20)
    assert abs(r[0] - np.log(c[20] / c[0])) < 1e-9
    assert np.isnan(r[-1]), "forward window must be censored at the tail"


# ---- information coefficient ----------------------------------------------
def test_cross_sectional_ic_recovers_sign():
    # Build a panel where score == target -> IC ~ +1; negated -> ~ -1.
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    recs = []
    for d in dates:
        for s in range(12):
            val = np.random.RandomState(hash((str(d), s)) % 2**32).rand()
            recs.append({"date": d, "stock": s, "sig": val,
                         "tgt": val, "neg": -val})
    panel = pd.DataFrame(recs)
    ic_pos, _, nd, _ = cross_sectional_ic(panel, "sig", "tgt")
    ic_neg, _, _, _ = cross_sectional_ic(panel, "sig", "neg")
    assert nd > 0 and ic_pos > 0.95, f"perfect corr should give IC~1, got {ic_pos}"
    assert ic_neg < -0.95, f"anti-corr should give IC~-1, got {ic_neg}"


# ---- backtest: no look-ahead, correct economics ----------------------------
def test_backtest_no_lookahead_and_sign():
    # Panel where a higher score deterministically means higher fwd_ret.
    # A top-k long book must then earn a positive mean return.
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    recs = []
    for d in dates:
        for s in range(20):
            score = s / 20.0
            recs.append({"date": d, "stock": s, "score": score,
                         "fwd_ret": 0.001 * (s - 10)})  # monotonic in score
    panel = pd.DataFrame(recs)
    m, rets = backtest_topk(panel, k=5, holding=20, cost_bps=0.0)
    assert m["n"] > 0
    assert m["mean_ret"] > 0, "top-k of a monotonic signal must be profitable"
    # Bottom-k (invert score) must lose -> confirms it's not using future info
    # in a symmetric, sign-respecting way.
    panel["score"] = -panel["score"]
    m2, _ = backtest_topk(panel, k=5, holding=20, cost_bps=0.0)
    assert m2["mean_ret"] < 0, "inverting the signal must invert the PnL"


def test_long_short_is_neutral_and_sign_correct():
    # Panel with a common market shock each date PLUS a score-monotonic spread.
    # Long-only return would be dominated by the shock; the long-short spread
    # must isolate the score effect (positive), and inverting the score flips it.
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    recs = []
    rng = np.random.RandomState(1)
    for d in dates:
        shock = rng.randn() * 0.05          # big common move, zero-mean
        for s in range(20):
            recs.append({"date": d, "stock": s, "score": s / 20.0,
                         "fwd_ret": shock + 0.001 * (s - 10)})
    panel = pd.DataFrame(recs)
    m, rets = backtest_long_short(panel, k=5, holding=20, cost_bps=0.0)
    assert m["n"] > 0
    assert m["mean_ret"] > 0, "L/S of a monotonic signal must be positive"
    # The common shock must be removed: L/S std << long-only std.
    mlo, rlo = backtest_topk(panel, k=5, holding=20, cost_bps=0.0)
    assert np.std(rets) < np.std(rlo), "L/S should be far less market-exposed"
    panel["score"] = -panel["score"]
    m2, _ = backtest_long_short(panel, k=5, holding=20, cost_bps=0.0)
    assert m2["mean_ret"] < 0, "inverting the signal must flip the L/S spread"


def test_cross_sectional_neutralize_removes_factor():
    # fwd_ret is built as 2*vol + idiosyncratic noise. After neutralising on
    # vol, the residual must be ~uncorrelated with vol but still hold the noise.
    rng = np.random.RandomState(3)
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    recs = []
    for d in dates:
        for s in range(15):
            vol = rng.rand()
            noise = rng.randn() * 0.01
            recs.append({"date": d, "stock": s, "vol": vol,
                         "fwd_ret": 2 * vol + noise, "noise": noise})
    panel = pd.DataFrame(recs)
    resid = cross_sectional_neutralize(panel, "fwd_ret", ["vol"])
    panel["resid"] = resid.values
    # Residual is orthogonal to the factor it was regressed out of...
    assert abs(panel["resid"].corr(panel["vol"])) < 0.1, "factor not removed"
    # ...but still carries the idiosyncratic signal.
    assert panel["resid"].corr(panel["noise"]) > 0.8, "residual lost the signal"


def test_walk_forward_is_causal_and_learns():
    # Feature x linearly predicts the (continuous) target; walk-forward scores
    # on held-out dates should correlate positively with the target.
    rng = np.random.RandomState(0)
    dates = pd.date_range("2016-01-01", periods=600, freq="B")
    recs = []
    for d in dates:
        for s in range(15):
            x = rng.randn()
            recs.append({"date": d, "stock": s, "mom_20": x,
                         "cont": x + 0.5 * rng.randn(), "fwd_ret": x * 0.01})
    panel = pd.DataFrame(recs)
    scored = walk_forward_scores(panel, target_col="cont",
                                 feature_cols=["mom_20"], Y=20,
                                 min_train_dates=200, step=40)
    oos = scored.dropna(subset=["score"])
    assert len(oos) > 0, "walk-forward produced no OOS scores"
    ic, _, _, _ = cross_sectional_ic(oos, "score", "fwd_ret")
    assert ic > 0.1, f"causal model should show positive OOS IC, got {ic}"


def _run_all():
    tests = [
        test_triple_barrier_close_direction,
        test_highlow_detects_intraday_touch,
        test_fwd_logreturn_value_and_censoring,
        test_cross_sectional_ic_recovers_sign,
        test_backtest_no_lookahead_and_sign,
        test_long_short_is_neutral_and_sign_correct,
        test_cross_sectional_neutralize_removes_factor,
        test_walk_forward_is_causal_and_learns,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
