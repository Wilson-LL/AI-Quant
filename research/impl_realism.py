"""Sprint Cycle 3 — implementation realism for D1.1 on TWSE.

Frictions tested (past-only, deterministic):
  1. Price-limit unfillability: TWSE has a +/-10% daily limit. A long name that is
     limit-UP at the signal close is not buyable at that price; a short name that
     is limit-DOWN is not shortable. Drop such names from the leg (proxy: entry-day
     return |close[t]/close[t-1]-1| >= 9.5%), renormalise weights over survivors.
  2. Cost stress: net Sharpe at 60/100/150 bps.
  3. No short side: long-only fallback (already carries beta).

Decision rule (pre-registered): survives iff price-limit-filtered net@60 Sharpe
>= 0.8x baseline AND positive at 100 bps.

Run: python -u research/impl_realism.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from momentum import momentum_signal, LOOKBACK, SKIP, HOLDING, QUINTILE
from d1_1_pername_cap import _cap_weights

LIMIT = 0.095
VOL = 60


def build(uni, holding=HOLDING):
    rows = []
    for sid, df in uni.items():
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].to_numpy(float); n = len(c)
        lr = np.zeros(n); lr[1:] = np.log(c[1:] / c[:-1])
        vol = pd.Series(lr).rolling(VOL).std().to_numpy()
        eret = np.zeros(n); eret[1:] = c[1:] / c[:-1] - 1.0     # entry-day return
        fwd = np.full(n, np.nan); fwd[:n - holding] = c[holding:] / c[:n - holding] - 1
        rows.append(pd.DataFrame({"date": df["date"].values, "stock": sid,
                                  "sector": SECTOR_MAP.get(sid, "other"),
                                  "mom": momentum_signal(c, LOOKBACK, SKIP),
                                  "fwd_ret": fwd, "vol": vol, "eret": eret}))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "stock"]).reset_index(drop=True)


def _w(sub, sc, nc):
    inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
    w = dict(zip(sub["stock"], inv / inv.sum()))
    sec = dict(zip(sub["stock"], sub["sector"]))
    return _cap_weights(w, sec, sc, nc), sec


def _tno(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def backtest(panel, k, holding=HOLDING, limit_filter=False, mode="long_short"):
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    dates = np.array(sorted(df["date"].unique())); rebal = dates[::holding]
    gross, turn = [], []
    pl, ps = {}, {}
    dropped = 0
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < 2 * k:
            continue
        o = day.sort_values("mom")
        longs, shorts = o.tail(k), o.head(k)
        if limit_filter:
            nl0, ns0 = len(longs), len(shorts)
            longs = longs[longs["eret"] < LIMIT]       # drop limit-up longs
            shorts = shorts[shorts["eret"] > -LIMIT]    # drop limit-down shorts
            dropped += (nl0 - len(longs)) + (ns0 - len(shorts))
        if len(longs) < 3 or (mode == "long_short" and len(shorts) < 3):
            continue
        wl, _ = _w(longs, 0.20, 0.10)
        lr = dict(zip(longs["stock"], longs["fwd_ret"]))
        g = sum(wl[x] * lr[x] for x in wl)
        if mode == "long_short":
            ws, _ = _w(shorts, 0.20, 0.10)
            sr = dict(zip(shorts["stock"], shorts["fwd_ret"]))
            g -= sum(ws[x] * sr[x] for x in ws)
            turn.append(0.5 * (_tno(wl, pl) + _tno(ws, ps))); ps = ws
        else:
            ws = {}
            turn.append(_tno(wl, pl))
        gross.append(g); pl = wl
    return np.array(gross), np.array(turn), dropped


def sharpe(gross, turn, cost, holding=HOLDING):
    net = gross - (cost / 1e4) * turn; net = net[~np.isnan(net)]
    return net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(252 / holding)


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    panel = build(load_universe(ids))
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    print(f"Cycle3 implementation realism. k={k}. LIMIT={LIMIT:.1%}. "
          f"survivorship-biased.\n")

    g0, t0, _ = backtest(panel, k)
    base = sharpe(g0, t0, 60)
    print(f"baseline D1.1 net@60 Sharpe = {base:.2f}")

    gl, tl, dropped = backtest(panel, k, limit_filter=True)
    filt = sharpe(gl, tl, 60)
    print(f"price-limit-filtered net@60 Sharpe = {filt:.2f} "
          f"({dropped} name-entries dropped over the sample)")

    print("\ncost stress (net Sharpe):")
    for c in (60, 100, 150):
        print(f"  baseline @{c}bps {sharpe(g0,t0,c):5.2f} | "
              f"limit-filtered @{c}bps {sharpe(gl,tl,c):5.2f}")

    glo, tlo, _ = backtest(panel, k, mode="long_only")
    print(f"\nlong-only (no short) net@60 Sharpe = {sharpe(glo,tlo,60):.2f} "
          f"(carries market beta)")

    passed = (filt >= 0.8 * base) and (sharpe(gl, tl, 100) > 0)
    print(f"\n=== VERDICT ===")
    print(f"  limit-filtered/baseline = {filt/base:.2f} (need >=0.80) | "
          f"limit-filtered @100bps = {sharpe(gl,tl,100):.2f} (need >0)")
    print(f"  -> {'PASS: survives implementation frictions' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
