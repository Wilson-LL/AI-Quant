"""Sprint Cycle 6 — alternative-momentum robustness.

Is D1.1's edge general cross-sectional momentum, or specific to the 6-1 (126/5)
signal? Compare 3-1 / 6-1 / 12-1 month definitions (skip 1 month = 21d), the D1.1
baseline (126/5, skip 1 week), and a rank-composite of the three. Same D1.1
construction. No ML.

Decision rule (pre-registered): robust iff all definitions positive AND the
composite is within ~0.10 of baseline (or better).

Run: python -u research/alt_momentum.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from momentum import momentum_signal
from walkforward_d1_1 import backtest

HOLDING = 20
VOL = 60
QUINTILE = 0.20
DEFS = {"3-1 (63/21)": (63, 21), "6-1 (126/21)": (126, 21),
        "12-1 (252/21)": (252, 21), "6-1 baseline (126/5)": (126, 5)}


def build(uni):
    rows = []
    for sid, df in uni.items():
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].to_numpy(float); n = len(c)
        lr = np.zeros(n); lr[1:] = np.log(c[1:] / c[:-1])
        vol = pd.Series(lr).rolling(VOL).std().to_numpy()
        fwd = np.full(n, np.nan); fwd[:n - HOLDING] = c[HOLDING:] / c[:n - HOLDING] - 1
        d = {"date": df["date"].values, "stock": sid,
             "sector": SECTOR_MAP.get(sid, "other"), "vol": vol, "fwd_ret": fwd}
        for name, (lb, sk) in DEFS.items():
            d[name] = momentum_signal(c, lb, sk)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "stock"]).reset_index(drop=True)


def sharpe(panel, k, cost=60):
    res = backtest(panel, k, 0.20, 0.10)
    net = res["gross"] - (cost / 1e4) * res["turn"]; net = net[~np.isnan(net)]
    return net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(252 / HOLDING)


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    panel = build(load_universe(ids))
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    print(f"Cycle6 alt-momentum. k={k}. survivorship-biased.\n")

    base = sharpe(panel.assign(mom=panel["6-1 baseline (126/5)"]), k)
    print(f"{'definition':22s} {'net@60 Sharpe':>13s}")
    res = {}
    for name in DEFS:
        s = sharpe(panel.assign(mom=panel[name]), k)
        res[name] = s
        print(f"{name:22s} {s:13.2f}")

    # rank-composite of the three X-1 defs (skip 1 month)
    comp_cols = ["3-1 (63/21)", "6-1 (126/21)", "12-1 (252/21)"]
    for c in comp_cols:
        panel[c + "_r"] = panel.groupby("date")[c].rank()
    panel["composite"] = panel[[c + "_r" for c in comp_cols]].mean(axis=1)
    comp = sharpe(panel.assign(mom=panel["composite"]), k)
    print(f"{'composite(3/6/12-1)':22s} {comp:13.2f}")

    all_pos = all(v > 0 for v in res.values()) and comp > 0
    passed = all_pos and (comp >= base - 0.10)
    print(f"\n=== VERDICT ===")
    print(f"  all defs positive: {all_pos} | composite {comp:.2f} vs baseline "
          f"{base:.2f} (need >= {base-0.10:.2f})")
    print(f"  -> {'PASS: general momentum, robust' if passed else 'PARTIAL/FAIL'}")


if __name__ == "__main__":
    main()
