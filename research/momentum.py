"""Phase A1 — the simplest possible momentum-factor product baseline.

No ML, no optimisation, no look-ahead. A single cross-sectional rank rule, so the
whole thing is a few lines of real logic; A/D/C must beat THIS.

Momentum signal (exact formula):
    mom_t = close[t - SKIP] / close[t - SKIP - LOOKBACK] - 1
    with LOOKBACK = 126 trading days (~6 months), SKIP = 5 trading days.
The skip drops the most recent week to avoid short-term reversal contamination —
standard academic "12-1"-style momentum, scaled to our 20-day horizon. Everything
is strictly point-in-time: mom_t uses only prices up to t-SKIP.

Backtest: rank the cross-section each rebalance, hold HOLDING (20) days.
  - long_short : long top quintile, short bottom quintile (beta/vol-neutral-ish).
  - long_only  : long top quintile (the deployable book; carries market beta).
Costs charged on turnover; sector attribution and yearly breakdown reported.

Run: python -u research/momentum.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP, DEFAULT_UNIVERSE

LOOKBACK = 126
SKIP = 5
HOLDING = 20
COSTS = [0, 10, 30, 60, 100]
QUINTILE = 0.20


def momentum_signal(close, lookback=LOOKBACK, skip=SKIP):
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    mom = np.full(n, np.nan)
    idx = np.arange(skip + lookback, n)
    mom[idx] = c[idx - skip] / c[idx - skip - lookback] - 1.0
    return mom


def build_mom_panel(universe, holding=HOLDING):
    rows = []
    for sid, df in universe.items():
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        fwd = np.full(n, np.nan)
        fwd[:n - holding] = c[holding:] / c[:n - holding] - 1.0
        rows.append(pd.DataFrame({
            "date": df["date"].values, "stock": sid,
            "sector": SECTOR_MAP.get(sid, "other"),
            "mom": momentum_signal(c), "fwd_ret": fwd,
        }))
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "stock"]).reset_index(drop=True)


def backtest(panel, k, holding=HOLDING, mode="long_short", signal="mom"):
    """Return per-rebalance gross return, turnover series, and sector P&L."""
    df = panel.dropna(subset=[signal, "fwd_ret"])
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]
    need = 2 * k if mode == "long_short" else k
    gross, turn, keep = [], [], []
    sector_pnl = {}
    prev = set()
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < need:
            continue
        o = day.sort_values(signal)
        longs = o.tail(k)
        if mode == "long_short":
            shorts = o.head(k)
            g = longs["fwd_ret"].mean() - shorts["fwd_ret"].mean()
            names = set(longs["stock"]) | set(shorts["stock"])
            for _, r in longs.iterrows():
                sector_pnl[r["sector"]] = sector_pnl.get(r["sector"], 0.0) + r["fwd_ret"] / k
            for _, r in shorts.iterrows():
                sector_pnl[r["sector"]] = sector_pnl.get(r["sector"], 0.0) - r["fwd_ret"] / k
        else:
            g = longs["fwd_ret"].mean()
            names = set(longs["stock"])
            for _, r in longs.iterrows():
                sector_pnl[r["sector"]] = sector_pnl.get(r["sector"], 0.0) + r["fwd_ret"] / k
        turn.append(len(names ^ prev) / max(len(names), 1))
        prev = names
        gross.append(g)
        keep.append(d)
    return (np.array(gross), np.array(turn), np.array(keep), sector_pnl)


def market_benchmark(panel, holding=HOLDING):
    """Equal-weight-all-names return each rebalance (the market proxy)."""
    df = panel.dropna(subset=["fwd_ret"])
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]
    r = [df[df["date"] == d]["fwd_ret"].mean() for d in rebal]
    return np.array(r), rebal


def perf(gross, turn, holding, cost_bps=60):
    net = gross - (cost_bps / 1e4) * (turn if turn is not None else 0.0)
    net = net[~np.isnan(net)]
    if len(net) < 2:
        return {k: float("nan") for k in
                ["ann_ret", "sharpe", "max_dd", "calmar", "turnover", "n"]}
    ppy = 252 / holding
    mean, std = net.mean(), net.std(ddof=1)
    sharpe = mean / (std + 1e-12) * np.sqrt(ppy)
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = (1 + mean) ** ppy - 1
    return {"ann_ret": ann, "sharpe": sharpe, "max_dd": dd,
            "calmar": ann / (abs(dd) + 1e-12),
            "turnover": float(np.nanmean(turn)) * ppy if turn is not None else 0.0,
            "n": int(len(net))}


def _ic(panel, signal="mom"):
    def one(g):
        g = g[[signal, "fwd_ret"]].dropna()
        if len(g) < 5:
            return np.nan
        return g[signal].rank().corr(g["fwd_ret"].rank())
    return panel.groupby("date", group_keys=False).apply(one).dropna().mean()


def _row(name, gross, turn, holding):
    m = perf(gross, turn, holding, 60)
    print(f"{name:30s} {m['ann_ret']:8.2%} {m['sharpe']:7.2f} {m['max_dd']:8.2%} "
          f"{m['calmar']:7.2f} {m['turnover']:7.1f}x {m['n']:5d}")
    return m


def main():
    all_ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(all_ids)
    panel = build_mom_panel(uni)
    n = panel["stock"].nunique()
    k = max(3, round(QUINTILE * n))
    print(f"Universe {n} names, {panel['sector'].nunique()} sectors; "
          f"quintile k={k}; mom = close[t-{SKIP}]/close[t-{SKIP+LOOKBACK}]-1; "
          f"holding={HOLDING}. Costs shown net@60bps. SURVIVORSHIP-biased (upper bound).")
    print(f"Cross-sectional IC(mom, fwd_ret) = {_ic(panel):.4f}\n")

    print(f"{'strategy':30s} {'annRet':>8s} {'Sharpe':>7s} {'maxDD':>8s} "
          f"{'Calmar':>7s} {'turnov':>8s} {'n':>5s}")
    ls = backtest(panel, k, mode="long_short")
    lo = backtest(panel, k, mode="long_only")
    m_ls = _row("A1 momentum L/S (multi)", ls[0], ls[1], HOLDING)
    m_lo = _row("A1 momentum long-only (multi)", lo[0], lo[1], HOLDING)
    mk, mkd = market_benchmark(panel)
    _row("equal-weight universe (market)", mk, np.zeros_like(mk), HOLDING)

    semis = {s: d for s, d in uni.items() if s in DEFAULT_UNIVERSE}
    sp = build_mom_panel(semis)
    ks = max(3, round(QUINTILE * sp["stock"].nunique()))
    sls = backtest(sp, ks, mode="long_short")
    _row(f"semis-only momentum L/S (k={ks})", sls[0], sls[1], HOLDING)

    print("\n-- Exp-3/4 linear ML model (cited from RESEARCH_LOG, same universe/protocol) --")
    print(f"{'Exp-4 linear L/S':30s} {'~':>8s} {0.62:7.2f} {'~':>8s} {'~':>7s} "
          f"{12.5:7.1f}x   net@30bps Sharpe; gross 0.82; 5/9 yrs +; concentrated")

    # cost sensitivity (main L/S)
    print("\n=== Cost sensitivity — A1 momentum L/S (multi) ===")
    print(f"{'cost_bps':>8s} {'net_Sharpe':>11s} {'net_annRet':>11s}")
    for cb in COSTS:
        m = perf(ls[0], ls[1], HOLDING, cb)
        print(f"{cb:8d} {m['sharpe']:11.2f} {m['ann_ret']:11.2%}")
    be = np.nanmean(ls[0]) / (np.nanmean(ls[1]) + 1e-12) * 1e4
    print(f"  breakeven ~= {be:.0f} bps/round-trip; ann. turnover "
          f"{np.nanmean(ls[1])*252/HOLDING:.1f}x")

    # yearly (main L/S, net@60)
    print("\n=== Yearly — A1 momentum L/S (multi), net@60bps ===")
    yrs = pd.Series(ls[2]).dt.year.to_numpy()
    net = ls[0] - (60 / 1e4) * ls[1]
    print(f"{'year':>5s} {'net_Sharpe':>11s} {'mean%':>8s} {'n':>4s}")
    for y in sorted(set(yrs)):
        r = net[yrs == y]
        s = (r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / HOLDING)
             if len(r) > 1 else float("nan"))
        print(f"{y:5d} {s:11.2f} {np.nanmean(r)*100:8.2f} {len(r):4d}")
    pos = sum(1 for y in sorted(set(yrs))
              if np.nanmean(net[yrs == y]) > 0)
    print(f"  positive years: {pos}/{len(set(yrs))}")

    # sector attribution (main L/S)
    print("\n=== Sector attribution — A1 momentum L/S (multi), cumulative L/S contribution ===")
    for sec, pnl in sorted(ls[3].items(), key=lambda x: -x[1]):
        print(f"  {sec:14s} {pnl*100:+7.1f}%")


if __name__ == "__main__":
    main()
