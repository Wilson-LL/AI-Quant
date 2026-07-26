"""Shared library for queue v9 — CPU-only deployment validation.

Signal is FROZEN: everything operates on the three existing 7-seed panels
(SCHED_A8_seeds7_full 2023->, SCHED_BEAR_A8_seeds7_full 2021->,
SCHED_W22_oos2022_start 2022->). No GPU, no training, cache is read-only.

Every experiment compares four books (rule 5 of the v9 approval):
  blend  — production default: 50/50 z-blend, band10, cap10
  d7b    — deployment variant:  50/50 z-blend, band15, cap7.5
  tf     — champion transformer standalone, band10, cap10
  d12    — D1.2 momentum standalone, band10, cap10
"""

import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import transformer_portfolio as tp  # noqa: E402
from transformer_hybrid import load_panel, merged, _cache_frames  # noqa: E402

V9_DIR = os.path.join(ROOT, "reports", "continuous_research", "queue_v9")
os.makedirs(V9_DIR, exist_ok=True)

PANELS = {"CH": "SCHED_A8_seeds7_full",        # champion window 2023->
          "BR": "SCHED_BEAR_A8_seeds7_full",   # bear window 2021->
          "W22": "SCHED_W22_oos2022_start"}    # crash-first window 2022->

SPECS = {
    "blend": {"score": "blend", "band": 0.10, "cap": 0.10},
    "d7b":   {"score": "blend", "band": 0.15, "cap": 0.075},
    "tf":    {"score": "z_tf",  "band": 0.10, "cap": 0.10},
    "d12":   {"score": "z_mom", "band": 0.10, "cap": 0.10},
}

_merged_cache = {}
_aux_lag_cache = None
_adv_cache = None
_regime_cache = None


def get_merged(win):
    """Merged panel (z_tf, z_mom, fwd_h, ...) for a window key, cached."""
    if win not in _merged_cache:
        panel, _ = load_panel(PANELS[win])
        _merged_cache[win] = merged(panel)
    return _merged_cache[win]


def with_score(m, spec_name):
    out = m.copy()
    s = SPECS[spec_name]["score"]
    out["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"] if s == "blend" else m[s]
    return out


@contextmanager
def caps(name_cap=0.10, sector_cap=0.20):
    """Patch tp.cap_weights so _leg_weights (call-time global lookup) applies
    non-default caps. Default args in the original are bound at def time, so
    this wrapper is the only clean route to cap 7.5% / hard sector caps."""
    orig = tp.cap_weights
    tp.cap_weights = lambda w, sectors=None, **kw: orig(
        w, sectors, name_cap=name_cap, sector_cap=sector_cap)
    try:
        yield
    finally:
        tp.cap_weights = orig


def run_book(win, spec_name, mode, ret_col="fwd_h", cost_bps=(0, 60, 100, 150),
             band=None, name_cap=None, sector_cap=0.20, exclude=None,
             score_override=None, keep_series=False):
    """Backtest one spec on one window. band/name_cap override the spec."""
    spec = SPECS[spec_name]
    m = get_merged(win)
    df = score_override(m) if score_override else with_score(m, spec_name)
    if exclude:
        df = df[~df["stock"].isin(set(exclude))]
    with caps(name_cap if name_cap is not None else spec["cap"], sector_cap):
        r = tp.backtest_scores(df, holding=20, mode=mode,
                               no_trade_band=band if band is not None else spec["band"],
                               cost_bps_list=tuple(cost_bps), ret_col=ret_col)
    if not keep_series:
        for k in ("gross", "turnover", "dates"):
            r.pop(k, None)
    return r


def slim(r):
    """Keep only headline fields for JSON output."""
    out = {"rank_ic": round(r["rank_ic"], 4), "avg_turnover": round(r["avg_turnover"], 3),
           "max_name_weight": r["max_name_weight"]}
    for k, v in r.items():
        if k.startswith("net") and isinstance(v, dict):
            out[k] = {kk: v[kk] for kk in ("sharpe", "ann_ret", "max_dd", "calmar", "n")}
    out["yearly_net60"] = r.get("yearly_net60")
    return out


def lag_returns(horizon=20, lags=(1, 2, 3)):
    """(date, stock) -> fwd_{horizon} at execution lag 1/2/3 from the frozen
    cache (read-only). Lag 1 duplicates the panel's own fwd_h and is used to
    VERIFY no same-bar execution exists anywhere in the pipeline."""
    global _aux_lag_cache
    if _aux_lag_cache is not None:
        return _aux_lag_cache
    rows = []
    for sid, df in _cache_frames().items():
        c = df["close"].to_numpy(np.float64)
        n = len(c)
        rec = {"date": df["date"].values, "stock": sid}
        for lag in lags:
            f = np.full(n, np.nan)
            stop = n - horizon - lag
            if stop > 0:
                f[:stop] = c[horizon + lag:] / c[lag:lag + stop] - 1.0
            rec[f"fwd_lag{lag}"] = f
        rows.append(pd.DataFrame(rec))
    _aux_lag_cache = pd.concat(rows, ignore_index=True)
    return _aux_lag_cache


def adv_frame():
    """(date, stock) -> adv20 = 20d median(close*volume) TWD, info through t."""
    global _adv_cache
    if _adv_cache is not None:
        return _adv_cache
    rows = []
    for sid, df in _cache_frames().items():
        dollar = df["close"].to_numpy(np.float64) * df["volume"].to_numpy(np.float64)
        adv = pd.Series(dollar).rolling(20).median().to_numpy()
        rows.append(pd.DataFrame({"date": df["date"].values, "stock": sid, "adv20": adv}))
    _adv_cache = pd.concat(rows, ignore_index=True)
    return _adv_cache


def hostile_dates():
    """Regime flag per date: equal-weight universe log-return index below its
    own 126d MA (cache-derived market proxy; info through t)."""
    global _regime_cache
    if _regime_cache is not None:
        return _regime_cache
    frames = _cache_frames()
    rets = []
    for sid, df in frames.items():
        c = df.set_index("date")["close"]
        rets.append(np.log(c / c.shift(1)).rename(sid))
    r = pd.concat(rets, axis=1).sort_index()
    proxy = r.mean(axis=1).cumsum()
    ma = proxy.rolling(126).mean()
    _regime_cache = (proxy < ma).fillna(False)
    return _regime_cache


def holdings_walk(df, mode="long_short", holding=20, band=0.0, name_cap=0.10):
    """Light replica of backtest_scores' selection loop that RETURNS holdings
    (for attribution / capacity accounting only — the stress re-runs always
    use the real backtest_scores). Yields (date, {stock: signed_weight}, day)."""
    from data import SECTOR_MAP
    df = (df.dropna(subset=["score", "fwd_h"])
            .drop_duplicates(["date", "stock"], keep="last"))
    dates = np.array(sorted(df["date"].unique()))
    by_date = dict(tuple(df.groupby("date")))
    prev_long = []
    for d in dates[::holding]:
        day = by_date.get(d)
        if day is None or day["stock"].nunique() < 60:
            continue
        n = day["stock"].nunique()
        kq = max(3, round(0.2 * n))
        o = day.sort_values("score")
        longs = list(o.tail(kq)["stock"])
        if band and prev_long:
            pool = set(o.tail(int(kq * (1 + 2 * band)))["stock"])
            inc = [s for s in prev_long if s in pool]
            longs = (inc + [s for s in longs if s not in inc])[:kq]
        w = {}
        with caps(name_cap):
            wl = tp._leg_weights(day, longs, "equal")
        w.update(dict(zip(longs, wl)))
        if mode == "long_short":
            shorts = list(o.head(kq)["stock"])
            with caps(name_cap):
                ws = tp._leg_weights(day, shorts, "equal")
            for s, x in zip(shorts, ws):
                w[s] = w.get(s, 0.0) - x
        prev_long = longs
        yield d, w, day
