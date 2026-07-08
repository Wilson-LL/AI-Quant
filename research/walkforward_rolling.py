"""Sprint Cycle 1 — stricter OOS: multi-fold walk-forward + block-bootstrap CI.

The D1.1 OOS validation used a single held-out split. This tests stability across
several sequential folds and gives a statistical confidence interval on the net
Sharpe (block bootstrap, seeded/reproducible). No ML; D1.1 rule unchanged.

Decision rule (pre-registered in RESEARCH_LOG): robust iff (a) >=60% of sequential
OOS folds have positive net@60 Sharpe, (b) bootstrap 90% CI lower bound on
full-sample annualized net@60 Sharpe > 0, (c) median fold Sharpe >= 0.5.

Run: python -u research/walkforward_rolling.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, QUINTILE, HOLDING
from walkforward_d1_1 import backtest

PPY = 252 / HOLDING
SEED = 20260707


def net_series(panel, k, sector_cap, name_cap, cost=60):
    res = backtest(panel, k, sector_cap, name_cap)
    net = res["gross"] - (cost / 1e4) * res["turn"]
    return net, pd.Series(res["dates"])


def ann_sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    return r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(PPY)


def block_bootstrap_sharpe(r, block=6, draws=5000, seed=SEED):
    r = r[~np.isnan(r)]
    n = len(r)
    rng = np.random.RandomState(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(draws)
    starts_max = n - block
    for i in range(draws):
        idx = rng.randint(0, max(1, starts_max + 1), size=nb)
        samp = np.concatenate([r[s:s + block] for s in idx])[:n]
        out[i] = ann_sharpe(samp)
    return np.nanpercentile(out, [5, 50, 95])


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    panel = build_panel(load_universe(ids))
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    print(f"Cycle1 stricter OOS. {panel['stock'].nunique()} names, k={k}, "
          f"seed={SEED}. net@60bps; survivorship-biased.\n")

    variants = {"D1.1 (sec20+name10)": (0.20, 0.10),
                "D1 (sec20)": (0.20, None),
                "uncapped": (None, None)}
    series = {name: net_series(panel, k, sc, nc)
              for name, (sc, nc) in variants.items()}

    # (a) sequential folds
    KF = 5
    print(f"=== (a) {KF} sequential OOS folds — net@60 Sharpe per fold ===")
    print(f"{'variant':22s} " + " ".join(f"f{j+1:>5d}" for j in range(KF)) +
          f" {'%pos':>6s} {'medn':>6s}")
    fold_ok = {}
    for name, (net, dates) in series.items():
        net = net[~np.isnan(net)]
        folds = np.array_split(net, KF)
        sh = [ann_sharpe(f) for f in folds]
        pos = np.mean([s > 0 for s in sh]) * 100
        med = np.nanmedian(sh)
        fold_ok[name] = (pos, med)
        print(f"{name:22s} " + " ".join(f"{s:6.2f}" for s in sh) +
              f" {pos:5.0f}% {med:6.2f}")

    # (b) rolling ~1yr Sharpe
    print("\n=== (b) rolling 1-yr (13-rebalance) net@60 Sharpe ===")
    win = 13
    for name, (net, dates) in series.items():
        net = net[~np.isnan(net)]
        roll = [ann_sharpe(net[i:i + win]) for i in range(len(net) - win + 1)]
        roll = np.array(roll)
        print(f"  {name:22s} min {np.nanmin(roll):5.2f}  median {np.nanmedian(roll):5.2f}"
              f"  max {np.nanmax(roll):5.2f}  %pos {np.mean(roll>0)*100:4.0f}%")

    # (c) block-bootstrap CI on full-sample annualized Sharpe
    print("\n=== (c) block-bootstrap 90% CI on full-sample net@60 Sharpe ===")
    boot = {}
    for name, (net, dates) in series.items():
        p5, p50, p95 = block_bootstrap_sharpe(net)
        boot[name] = (p5, p50, p95)
        print(f"  {name:22s} point {ann_sharpe(net):5.2f} | 5% {p5:5.2f}  "
              f"50% {p50:5.2f}  95% {p95:5.2f}")

    # verdict for D1.1
    pos, med = fold_ok["D1.1 (sec20+name10)"]
    lo = boot["D1.1 (sec20+name10)"][0]
    passed = (pos >= 60) and (lo > 0) and (med >= 0.5)
    print(f"\n=== VERDICT (D1.1) ===")
    print(f"  (a) folds %positive = {pos:.0f}% (need >=60)  median = {med:.2f} (need >=0.5)")
    print(f"  (b/c) bootstrap 5% CI lower bound = {lo:.2f} (need >0)")
    print(f"  -> {'PASS: robust under stricter OOS' if passed else 'WEAKENED/FAIL'}")


if __name__ == "__main__":
    main()
