"""Sprint Cycle 7 — universe-robustness bootstrap.

Does D1.1's edge depend on the specific 106-name universe? Draw 200 seeded random
subsets (each drops ~20% of names) and recompute the D1.1 net@60 Sharpe -> CI.
Complements the (unmeasurable) survivorship caveat: if the edge survives dropping
random names, it isn't riding a handful of specific survivors. No ML.

Pre-grouped panel (group-by-date once) so 200 backtests run fast.

Decision rule (pre-registered): robust iff 5th-pct Sharpe > 0 AND median within
~0.2 of full-universe Sharpe.

Run: python -u research/universe_bootstrap.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, QUINTILE, HOLDING
from d1_1_pername_cap import _cap_weights

SEED = 20260707
DRAWS = 200
DROP_FRAC = 0.20
PPY = 252 / HOLDING


def _w(sub):
    inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
    w = dict(zip(sub["stock"], inv / inv.sum()))
    sec = dict(zip(sub["stock"], sub["sector"]))
    return _cap_weights(w, sec, 0.20, 0.10)


def _tno(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def make_backtester(panel):
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    gb = {d: sub for d, sub in df.groupby("date")}
    rebal = sorted(gb)[::HOLDING]

    def bt(keep, cost=60):
        k = max(3, round(QUINTILE * len(keep)))
        pl, ps = {}, {}
        gross, turn = [], []
        for d in rebal:
            day = gb[d]
            if keep is not None:
                day = day[day["stock"].isin(keep)]
            if len(day) < 2 * k:
                continue
            o = day.sort_values("mom")
            longs, shorts = o.tail(k), o.head(k)
            wl = _w(longs); ws = _w(shorts)
            lr = dict(zip(longs["stock"], longs["fwd_ret"]))
            sr = dict(zip(shorts["stock"], shorts["fwd_ret"]))
            gross.append(sum(wl[x]*lr[x] for x in wl) - sum(ws[x]*sr[x] for x in ws))
            turn.append(0.5 * (_tno(wl, pl) + _tno(ws, ps))); pl, ps = wl, ws
        net = np.array(gross) - (cost/1e4) * np.array(turn)
        net = net[~np.isnan(net)]
        return net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(PPY) if len(net) > 1 else np.nan
    return bt


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    panel = build_panel(load_universe(ids))
    names = sorted(panel["stock"].unique())
    bt = make_backtester(panel)
    full = bt(set(names))
    print(f"Cycle7 universe bootstrap. {len(names)} names, drop {DROP_FRAC:.0%}, "
          f"{DRAWS} draws, seed={SEED}. survivorship-biased.")
    print(f"full-universe net@60 Sharpe = {full:.2f}\n")

    rng = np.random.RandomState(SEED)
    keepn = int(round(len(names) * (1 - DROP_FRAC)))
    sh = np.array([bt(set(rng.choice(names, keepn, replace=False))) for _ in range(DRAWS)])
    sh = sh[~np.isnan(sh)]
    p = np.percentile(sh, [5, 25, 50, 75, 95])
    print(f"subset Sharpe percentiles: 5% {p[0]:.2f} | 25% {p[1]:.2f} | "
          f"50% {p[2]:.2f} | 75% {p[3]:.2f} | 95% {p[4]:.2f}")
    print(f"  %positive = {np.mean(sh>0)*100:.0f}%  mean {sh.mean():.2f}  min {sh.min():.2f}")

    passed = p[0] > 0 and abs(p[2] - full) <= 0.2
    print(f"\n=== VERDICT ===")
    print(f"  5th-pct {p[0]:.2f} (need >0) | median {p[2]:.2f} vs full {full:.2f} "
          f"(|diff| need <=0.2)")
    print(f"  -> {'PASS: universe-robust' if passed else 'PARTIAL/FAIL'}")


if __name__ == "__main__":
    main()
