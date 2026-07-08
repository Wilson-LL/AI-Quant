"""Sprint Cycle 2 — parameter-sensitivity robustness of D1.1.

Is the edge robust to reasonable parameter choices, or overfit to
lookback=126 / skip=5 / holding=20 / quintile? Same D1.1 construction (inverse-vol
+ 20% sector + 10% name cap); we vary only the signal/selection parameters. No ML.
vol (60d, for sizing) is precomputed once; only mom (lookback,skip) and fwd
(holding) change per setting -> fast.

Decision rule (pre-registered): robust iff >=90% of the grid has positive net@60
Sharpe AND >=75% has Sharpe >= 0.7x baseline (~0.9).

Run: python -u research/param_sensitivity.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from momentum import momentum_signal
from walkforward_d1_1 import backtest

VOL = 60
QUINTILE = 0.20
LB, SK, HD, FR = 126, 5, 20, 0.20  # baseline


def _base(uni):
    b = {}
    for sid, df in uni.items():
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].to_numpy(float)
        lr = np.zeros(len(c)); lr[1:] = np.log(c[1:] / c[:-1])
        vol = pd.Series(lr).rolling(VOL).std().to_numpy()
        b[sid] = {"date": df["date"].values, "c": c, "vol": vol,
                  "sector": SECTOR_MAP.get(sid, "other")}
    return b


def panel_for(b, lookback, skip, holding):
    rows = []
    for sid, d in b.items():
        c = d["c"]; n = len(c)
        mom = momentum_signal(c, lookback, skip)
        fwd = np.full(n, np.nan); fwd[:n - holding] = c[holding:] / c[:n - holding] - 1
        rows.append(pd.DataFrame({"date": d["date"], "stock": sid,
                                  "sector": d["sector"], "mom": mom,
                                  "fwd_ret": fwd, "vol": d["vol"]}))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "stock"]).reset_index(drop=True)


def sharpe(b, lookback, skip, holding, frac, cost=60):
    panel = panel_for(b, lookback, skip, holding)
    k = max(3, round(frac * panel["stock"].nunique()))
    res = backtest(panel, k, 0.20, 0.10, holding=holding)
    net = res["gross"] - (cost / 1e4) * res["turn"]
    net = net[~np.isnan(net)]
    if len(net) < 2:
        return float("nan")
    return net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(252 / holding)


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    b = _base(load_universe(ids))
    base_sh = sharpe(b, LB, SK, HD, FR)
    thr = 0.7 * base_sh
    print(f"Cycle2 param sensitivity. baseline (lb{LB}/sk{SK}/hd{HD}/frac{FR}) "
          f"net@60 Sharpe = {base_sh:.2f}; robust threshold 0.7x = {thr:.2f}\n")

    grid = []
    print("=== one-axis sweeps (others at baseline) ===")
    for lb in [63, 126, 189, 252]:
        s = sharpe(b, lb, SK, HD, FR); grid.append(s)
        print(f"  lookback={lb:3d}: {s:5.2f}")
    for sk in [0, 5, 10]:
        s = sharpe(b, LB, sk, HD, FR); grid.append(s)
        print(f"  skip={sk:2d}:      {s:5.2f}")
    for hd in [10, 20, 40]:
        s = sharpe(b, LB, SK, hd, FR); grid.append(s)
        print(f"  holding={hd:2d}:  {s:5.2f}")
    for fr in [0.10, 0.20, 0.333]:
        s = sharpe(b, LB, SK, HD, fr); grid.append(s)
        print(f"  frac={fr:.3f}: {s:5.2f}")

    print("\n=== lookback x holding grid (interaction) ===")
    print("       hd=10  hd=20  hd=40")
    for lb in [63, 126, 189, 252]:
        cells = []
        for hd in [10, 20, 40]:
            s = sharpe(b, lb, SK, hd, FR); grid.append(s); cells.append(f"{s:5.2f}")
        print(f"  lb={lb:3d}  " + "  ".join(cells))

    g = np.array([x for x in grid if not np.isnan(x)])
    pos = np.mean(g > 0) * 100
    strong = np.mean(g >= thr) * 100
    passed = pos >= 90 and strong >= 75
    print(f"\n=== VERDICT ===")
    print(f"  {len(g)} settings | %positive {pos:.0f}% (need >=90) | "
          f"%>=0.7x-baseline {strong:.0f}% (need >=75) | "
          f"min {g.min():.2f} median {np.median(g):.2f} max {g.max():.2f}")
    print(f"  -> {'PASS: not overfit; robust to parameters' if passed else 'FAIL: parameter-fragile'}")


if __name__ == "__main__":
    main()
