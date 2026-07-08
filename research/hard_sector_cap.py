"""Sprint Cycle 5 — hard sector cap vs the soft-cap D1.1.

D1.1's sector cap is soft (floored at 1/n_sectors -> realized ~21%). This tests a
HARD 20% cap enforced by dropping the weakest-momentum name from an over-cap sector
(iteratively), then re-sizing survivors (inverse-vol + 10% name cap). Trade-off:
hard cap shrinks the leg when sectors are concentrated. No ML.

Decision rule (pre-registered): adopt hard cap iff realized max sector weight
<= 20.5% AND net@60 Sharpe drop <= 0.10 vs soft D1.1.

Run: python -u research/hard_sector_cap.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, QUINTILE, HOLDING
from walkforward_d1_1 import backtest as soft_backtest

SECTOR_CAP = 0.20
NAME_CAP = 0.10
MIN_NAMES = 12


def _name_cap_only(w, cap):
    w = dict(w)
    for _ in range(200):
        over = [x for x in w if w[x] > cap + 1e-9]
        if not over:
            break
        excess = sum(w[x] - cap for x in over)
        for x in over:
            w[x] = cap
        under = [x for x in w if w[x] < cap - 1e-12]
        tot = sum(w[x] for x in under)
        if tot <= 1e-12:
            break
        for x in under:
            w[x] += excess * w[x] / tot
    s = sum(w.values())
    return {x: v / s for x, v in w.items()}


def hard_leg(sub, weakest_low=True):
    """sub: rows with stock/vol/sector/mom. Drop weakest-momentum name from the
    largest-weight sector until max sector weight <= 20% (or leg hits MIN_NAMES)."""
    sub = sub.copy()
    for _ in range(50):
        inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
        w = _name_cap_only(dict(zip(sub["stock"], inv / inv.sum())), NAME_CAP)
        sec = dict(zip(sub["stock"], sub["sector"]))
        ssum = {}
        for x, wx in w.items():
            ssum[sec[x]] = ssum.get(sec[x], 0.0) + wx
        top = max(ssum, key=ssum.get)
        if ssum[top] <= SECTOR_CAP + 1e-3 or len(sub) <= MIN_NAMES:
            return w, sec
        cand = sub[sub["sector"] == top]
        drop = cand["mom"].idxmin() if weakest_low else cand["mom"].idxmax()
        sub = sub.drop(index=drop)
    return w, sec


def _tno(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def hard_backtest(panel, k, holding=HOLDING):
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    dates = np.array(sorted(df["date"].unique())); rebal = dates[::holding]
    gross, turn, msw, mnw = [], [], [], []
    pl, ps = {}, {}
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < 2 * k:
            continue
        o = day.sort_values("mom")
        wl, secl = hard_leg(o.tail(k), weakest_low=True)
        ws, secs = hard_leg(o.head(k), weakest_low=False)
        lr = dict(zip(o.tail(k)["stock"], o.tail(k)["fwd_ret"]))
        sr = dict(zip(o.head(k)["stock"], o.head(k)["fwd_ret"]))
        g = sum(wl[x] * lr[x] for x in wl) - sum(ws[x] * sr[x] for x in ws)
        gross.append(g)
        turn.append(0.5 * (_tno(wl, pl) + _tno(ws, ps))); pl, ps = wl, ws
        sw = {}
        for x, wx in wl.items():
            sw[secl[x]] = sw.get(secl[x], 0.0) + wx
        msw.append(max(sw.values())); mnw.append(max(list(wl.values()) + list(ws.values())))
    return np.array(gross), np.array(turn), np.array(msw), np.array(mnw)


def _m(gross, turn, cost=60, holding=HOLDING):
    net = gross - (cost / 1e4) * turn; net = net[~np.isnan(net)]
    ppy = 252 / holding
    sh = net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(ppy)
    eq = np.cumprod(1 + net); dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return sh, dd


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    panel = build_panel(load_universe(ids))
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    print(f"Cycle5 hard sector cap. k={k}. survivorship-biased.\n")

    sres = soft_backtest(panel, k, 0.20, 0.10)
    ssh, sdd = _m(sres["gross"], sres["turn"])
    print(f"soft-cap D1.1 : Sharpe {ssh:5.2f}  maxDD {sdd:6.1%}  "
          f"maxSecW {np.nanmean(sres['msw']):.1%}  maxNm {np.nanmean(sres['mnw']):.1%}")

    hg, ht, hmsw, hmnw = hard_backtest(panel, k)
    hsh, hdd = _m(hg, ht)
    print(f"hard-cap 20%  : Sharpe {hsh:5.2f}  maxDD {hdd:6.1%}  "
          f"maxSecW {np.nanmean(hmsw):.1%}  maxNm {np.nanmean(hmnw):.1%}")

    passed = (np.nanmean(hmsw) <= 0.205) and (ssh - hsh <= 0.10)
    print(f"\n=== VERDICT ===")
    print(f"  hard maxSecW {np.nanmean(hmsw):.1%} (need <=20.5%) | Sharpe drop "
          f"{ssh-hsh:+.2f} (need <=0.10)")
    print(f"  -> {'ADOPT hard cap' if passed else 'KEEP soft cap (hard cap not worth it)'}")


if __name__ == "__main__":
    main()
