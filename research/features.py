"""Point-in-time (causal) features.

Every feature at row t uses only data up to and including t. These are the
predictors the evaluation framework uses to measure how *learnable* each
candidate target is (via information coefficient), and as inputs to the simple
baseline models. Kept deliberately standard and cheap — the research question
is about the target and methodology, not about clever features.
"""

import numpy as np
import pandas as pd


def compute_features(df):
    """Return a DataFrame of causal features aligned to df's rows (date column
    preserved). Leading rows are NaN until each window fills."""
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    lr = np.log(c / c.shift(1))

    feats = pd.DataFrame({"date": df["date"].values})
    feats["mom_5"] = c.pct_change(5)
    feats["mom_10"] = c.pct_change(10)
    feats["mom_20"] = c.pct_change(20)
    feats["mom_60"] = c.pct_change(60)
    feats["vol_20"] = lr.rolling(20).std().values
    feats["vol_60"] = lr.rolling(60).std().values
    # distance below trailing 60-day high (negative = below high). A classic
    # "not overextended / basing" signal.
    feats["dist_hi_60"] = (c / c.rolling(60).max() - 1.0).values
    feats["dist_lo_60"] = (c / c.rolling(60).min() - 1.0).values
    feats["vol_z"] = ((v - v.rolling(20).mean())
                      / (v.rolling(20).std() + 1e-9)).values
    feats["rsi_14"] = _rsi(c, 14).values
    # trend: price vs its own moving averages
    feats["px_over_ma20"] = (c / c.rolling(20).mean() - 1.0).values
    feats["px_over_ma60"] = (c / c.rolling(60).mean() - 1.0).values
    return feats


def _rsi(close, window=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


FEATURE_COLS = [
    "mom_5", "mom_10", "mom_20", "mom_60",
    "vol_20", "vol_60",
    "dist_hi_60", "dist_lo_60",
    "vol_z", "rsi_14",
    "px_over_ma20", "px_over_ma60",
]
