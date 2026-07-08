"""Sprint Cycle 4 — regime / sub-period stability of D1.1.

Is the edge concentrated in one favourable regime, or does it hold across
bull/bear and high/low-vol sub-samples? Also characterise the worst drawdown
(depth, duration, recovery). No ML; D1.1 rule unchanged. net@60bps.

Decision rule (pre-registered): stable iff (a) Sharpe positive in bull AND bear
(bear not < -0.3), (b) positive in high-vol AND low-vol, (c) max drawdown recovers
within the sample.

Run: python -u research/regime_stability.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, QUINTILE, HOLDING
from walkforward_d1_1 import backtest
from regime_c1 import universe_index

PPY = 252 / HOLDING


def _sh(r):
    r = np.asarray(r); r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    return r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(PPY)


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    res = backtest(panel, k, 0.20, 0.10)
    net = res["gross"] - (60 / 1e4) * res["turn"]
    dates = pd.Series(res["dates"])
    reg = universe_index(uni)

    trend = np.array([reg["trend"].get(pd.Timestamp(d), np.nan) for d in res["dates"]])
    rvol = np.array([reg["rvol"].get(pd.Timestamp(d), np.nan) for d in res["dates"]])
    vmed = np.nanmedian(rvol)
    print(f"Cycle4 regime stability. k={k}. full net@60 Sharpe = {_sh(net):.2f}. "
          f"survivorship-biased.\n")

    bull = net[trend >= 0]; bear = net[trend < 0]
    hv = net[rvol > vmed]; lv = net[rvol <= vmed]
    print("=== regime-conditional net@60 Sharpe (mean%/rebal, n) ===")
    for tag, seg in [("BULL (trend>=0)", bull), ("BEAR (trend<0)", bear),
                     ("HIGH-vol", hv), ("LOW-vol", lv)]:
        seg = seg[~np.isnan(seg)]
        print(f"  {tag:16s} Sharpe {_sh(seg):5.2f}  mean {np.mean(seg)*100:6.2f}%  n {len(seg):3d}")

    # drawdown depth / duration / recovery
    x = net[~np.isnan(net)]
    eq = np.cumprod(1 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    trough = int(np.argmin(dd))
    peak_i = int(np.argmax(eq[:trough + 1]))
    rec = next((j for j in range(trough, len(eq)) if eq[j] >= peak[trough]), None)
    print("\n=== worst drawdown ===")
    print(f"  depth {dd[trough]:.1%} | peak->trough {trough-peak_i} rebalances "
          f"(~{(trough-peak_i)*HOLDING/21:.0f} mo) | "
          f"{'recovered in ' + str(rec-trough) + ' rebalances (~' + f'{(rec-trough)*HOLDING/21:.0f}' + ' mo)' if rec else 'NOT recovered in sample'}")

    # rolling 12m %positive (context)
    win = 13
    roll = np.array([_sh(x[i:i+win]) for i in range(len(x)-win+1)])
    print(f"  rolling-12m Sharpe %positive = {np.mean(roll>0)*100:.0f}%")

    passed = (_sh(bull) > 0 and _sh(bear) > -0.3 and _sh(hv) > 0 and _sh(lv) > 0
              and rec is not None)
    print(f"\n=== VERDICT -> {'PASS: regime-stable' if passed else 'PARTIAL/FAIL (see above)'} ===")


if __name__ == "__main__":
    main()
