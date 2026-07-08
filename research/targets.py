"""Candidate prediction targets.

The baseline system predicts a close-only triple-barrier label: "does +12%
happen before -6% within 20 trading days". This module implements that baseline
plus alternatives, so the evaluation framework can ask the primary research
question empirically: *is that target the right thing to predict at all?*

Each function takes a per-stock OHLCV DataFrame (sorted ascending by date) and
returns a numpy array aligned to its rows, with np.nan where the forward window
is incomplete (the last Y rows). All targets are strictly forward-looking and
carry no look-ahead into the features.

Design axes deliberately varied here:
  - binary vs continuous            (learnability, information content)
  - path-dependent vs endpoint      (matches a TP/SL exit vs a fixed hold)
  - close-only vs intraday high/low  (label fidelity)
  - raw vs volatility-adjusted       (comparability across names/regimes)
  - absolute vs cross-sectional      (matches the system's actual top-k use)
"""

import numpy as np


def _closes(df):
    return df["close"].to_numpy(dtype=np.float64)


def triple_barrier_close(df, Y=20, H=0.12, L=0.06):
    """BASELINE (as implemented in dataset.build_samples): +H before -L within
    Y days, judged on CLOSE only. 'No barrier touched' is labelled 0 — i.e. it
    is merged with the down case, which is a known weakness we want to measure.
    """
    c = _closes(df)
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(n - Y):
        base = c[t]
        label = 0
        for p in c[t + 1:t + Y + 1]:
            if p >= base * (1 + H):
                label = 1
                break
            if p <= base * (1 - L):
                label = 0
                break
        out[t] = label
    return out


def triple_barrier_highlow(df, Y=20, H=0.12, L=0.06):
    """Triple barrier judged on intraday HIGH/LOW (touch, not close). More
    faithful to a real TP/SL order. When both barriers are touched on the same
    day the pessimistic assumption is used (stop first)."""
    hi = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    c = _closes(df)
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(n - Y):
        base = c[t]
        up, dn = base * (1 + H), base * (1 - L)
        label = 0
        for j in range(t + 1, t + Y + 1):
            touched_dn = lo[j] <= dn
            touched_up = hi[j] >= up
            if touched_dn:          # pessimistic: stop wins a same-day tie
                label = 0
                break
            if touched_up:
                label = 1
                break
        out[t] = label
    return out


def fwd_logreturn(df, Y=20):
    """Continuous Y-day forward log return. Uses full magnitude, no thresholds."""
    c = _closes(df)
    n = len(c)
    out = np.full(n, np.nan)
    out[:n - Y] = np.log(c[Y:] / c[:n - Y])
    return out


def fwd_return_sign(df, Y=20):
    r = fwd_logreturn(df, Y)
    return np.where(np.isnan(r), np.nan, (r > 0).astype(float))


def fwd_return_voladj(df, Y=20, vol_window=20):
    """Forward return standardised by trailing realised vol scaled to Y days.
    Comparable across names and regimes (a 'forward Sharpe' of the trade)."""
    c = _closes(df)
    n = len(c)
    lr = np.zeros(n)
    lr[1:] = np.log(c[1:] / c[:-1])
    vol = _rolling_std(lr, vol_window)
    scale = vol * np.sqrt(Y)
    fwd = fwd_logreturn(df, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = fwd / scale
    out[scale <= 0] = np.nan
    return out


def mfe_mae_ratio(df, Y=20):
    """Max favourable excursion vs max adverse excursion over the next Y days
    (both from intraday high/low). A path-quality target: rewards names that
    run up cleanly with little drawdown, independent of a fixed barrier."""
    hi = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    c = _closes(df)
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(n - Y):
        base = c[t]
        window_hi = hi[t + 1:t + Y + 1].max()
        window_lo = lo[t + 1:t + Y + 1].min()
        mfe = window_hi / base - 1.0
        mae = base / window_lo - 1.0  # positive magnitude of drawdown
        out[t] = mfe - mae
    return out


def _rolling_std(a, w):
    s = np.full_like(a, np.nan, dtype=np.float64)
    for i in range(w - 1, len(a)):
        s[i] = a[i - w + 1:i + 1].std(ddof=1)
    return s


# Registry consumed by the evaluation framework.
TARGETS = {
    "baseline_tb_close": triple_barrier_close,   # the current target
    "tb_highlow": triple_barrier_highlow,
    "fwd_logreturn": fwd_logreturn,
    "fwd_return_sign": fwd_return_sign,
    "fwd_return_voladj": fwd_return_voladj,
    "mfe_mae": mfe_mae_ratio,
}

# Targets that are continuous (Spearman IC meaningful) vs binary (AUC/base-rate).
CONTINUOUS_TARGETS = {"fwd_logreturn", "fwd_return_voladj", "mfe_mae"}
BINARY_TARGETS = {"baseline_tb_close", "tb_highlow", "fwd_return_sign"}
