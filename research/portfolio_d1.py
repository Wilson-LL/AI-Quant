"""Phase D1 — portfolio construction on the A1 long-short momentum product.

Same signal, same selection (top/bottom momentum quintile) as A1 — we change only
the WEIGHTS, to isolate construction from signal. No ML, no aggressive tuning.

Constructions tested (all pre-declared):
  - equal (A1 baseline, reference)
  - sector cap ∈ {15, 20, 25, 30}%  (equal base weights, then cap+redistribute)
  - inverse-vol (weights ∝ 1/trailing_vol, past-only)
  - inverse-vol + sector cap 20%    (combined)

Sizing/vol use ONLY past data. Turnover & cost conventions match A1 exactly, so
rows are directly comparable. Costs net@60bps unless noted; survivorship-biased.

Run: python -u research/portfolio_d1.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from momentum import momentum_signal, LOOKBACK, SKIP, HOLDING, QUINTILE

COSTS = [0, 30, 60, 80, 100]
VOL_WINDOW = 60


def build_panel(universe, holding=HOLDING):
    rows = []
    for sid, df in universe.items():
        df = df.sort_values("date").reset_index(drop=True)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        fwd = np.full(n, np.nan)
        fwd[:n - holding] = c[holding:] / c[:n - holding] - 1.0
        lr = np.zeros(n)
        lr[1:] = np.log(c[1:] / c[:-1])
        vol = pd.Series(lr).rolling(VOL_WINDOW).std().to_numpy()  # past-only
        rows.append(pd.DataFrame({
            "date": df["date"].values, "stock": sid,
            "sector": SECTOR_MAP.get(sid, "other"),
            "mom": momentum_signal(c), "fwd_ret": fwd, "vol": vol,
        }))
    p = pd.concat(rows, ignore_index=True)
    return p.sort_values(["date", "stock"]).reset_index(drop=True)


def _apply_sector_caps(weights, sector_of, cap):
    """weights: {name: w>0} summing to 1. Cap each sector's total at `cap`,
    redistributing excess pro-rata to under-cap names. Cap is floored at
    1/n_sectors so it stays feasible (no aggressive tuning)."""
    w = dict(weights)
    secs = {sector_of[n] for n in w}
    eff = max(cap, 1.0 / len(secs) + 1e-12)
    for _ in range(200):
        ssum = {}
        for n, wn in w.items():
            ssum[sector_of[n]] = ssum.get(sector_of[n], 0.0) + wn
        over = [s for s, v in ssum.items() if v > eff + 1e-9]
        if not over:
            break
        excess = 0.0
        over_set = set(over)
        for s in over:
            scale = eff / ssum[s]
            for n in w:
                if sector_of[n] == s:
                    excess += w[n] * (1 - scale)
                    w[n] *= scale
        under = [n for n in w if sector_of[n] not in over_set]
        tot = sum(w[n] for n in under)
        if tot <= 1e-12:
            break
        for n in under:
            w[n] += excess * w[n] / tot
    s = sum(w.values())
    return {n: v / s for n, v in w.items()}


def _leg_weights(sub, scheme, cap):
    names = sub["stock"].tolist()
    sec = dict(zip(sub["stock"], sub["sector"]))
    if "invvol" in scheme:
        inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
        w = dict(zip(names, inv / inv.sum()))
    else:
        w = {n: 1.0 / len(names) for n in names}
    if "cap" in scheme:
        w = _apply_sector_caps(w, sec, cap)
    return w


def _leg_turnover(w_new, w_prev):
    keys = set(w_new) | set(w_prev)
    return 0.5 * sum(abs(w_new.get(k, 0.0) - w_prev.get(k, 0.0)) for k in keys)


def backtest(panel, k, scheme, cap=None, holding=HOLDING):
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]
    gross, turn, keep = [], [], []
    sector_pnl, max_secw = {}, []
    pl, ps = {}, {}
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < 2 * k:
            continue
        o = day.sort_values("mom")
        longs, shorts = o.tail(k), o.head(k)
        wl = _leg_weights(longs, scheme, cap)
        ws = _leg_weights(shorts, scheme, cap)
        lr = dict(zip(longs["stock"], longs["fwd_ret"]))
        sr = dict(zip(shorts["stock"], shorts["fwd_ret"]))
        secl = dict(zip(longs["stock"], longs["sector"]))
        secs = dict(zip(shorts["stock"], shorts["sector"]))
        g = sum(wl[n] * lr[n] for n in wl) - sum(ws[n] * sr[n] for n in ws)
        gross.append(g)
        turn.append(0.5 * (_leg_turnover(wl, pl) + _leg_turnover(ws, ps)))
        pl, ps = wl, ws
        # max sector weight in long leg
        sw = {}
        for n, wn in wl.items():
            sw[secl[n]] = sw.get(secl[n], 0.0) + wn
        max_secw.append(max(sw.values()))
        # sector P&L attribution (L/S contribution)
        for n in wl:
            sector_pnl[secl[n]] = sector_pnl.get(secl[n], 0.0) + wl[n] * lr[n]
        for n in ws:
            sector_pnl[secs[n]] = sector_pnl.get(secs[n], 0.0) - ws[n] * sr[n]
        keep.append(d)
    return {"gross": np.array(gross), "turn": np.array(turn),
            "dates": np.array(keep), "sector_pnl": sector_pnl,
            "max_secw": np.array(max_secw)}


def perf(res, holding, cost_bps=60):
    net = res["gross"] - (cost_bps / 1e4) * res["turn"]
    net = net[~np.isnan(net)]
    if len(net) < 2:
        return {}
    ppy = 252 / holding
    mean, std = net.mean(), net.std(ddof=1)
    sharpe = mean / (std + 1e-12) * np.sqrt(ppy)
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = (1 + mean) ** ppy - 1
    # max sector P&L share (of total gross cumulative)
    tot = sum(res["sector_pnl"].values())
    share = max(res["sector_pnl"].values()) / tot if tot > 0 else float("nan")
    return {"sharpe": sharpe, "ann": ann, "dd": dd,
            "calmar": ann / (abs(dd) + 1e-12),
            "turnover": float(np.nanmean(res["turn"])) * ppy,
            "max_secw": float(np.nanmean(res["max_secw"])),
            "max_pnl_share": share}


def _yearly_2022(res, holding):
    yrs = pd.Series(res["dates"]).dt.year.to_numpy()
    net = res["gross"] - (60 / 1e4) * res["turn"]
    r = net[yrs == 2022]
    if len(r) < 2:
        return float("nan"), float("nan")
    s = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / holding)
    return s, float(np.nanmean(r) * 100)


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    n = panel["stock"].nunique()
    k = max(3, round(QUINTILE * n))
    print(f"D1: {n} names, k={k}, holding={HOLDING}, vol={VOL_WINDOW}d. "
          f"net@60bps; survivorship-biased (upper bound). Same signal/selection "
          f"as A1; only weights change.\n")

    variants = [
        ("A1 equal (uncapped)", "equal", None),
        ("sector cap 15%", "sector_cap", 0.15),
        ("sector cap 20%", "sector_cap", 0.20),
        ("sector cap 25%", "sector_cap", 0.25),
        ("sector cap 30%", "sector_cap", 0.30),
        ("inverse-vol", "invvol", None),
        ("invvol + cap 20%", "invvol_cap", 0.20),
    ]
    results = {}
    print(f"{'variant':22s} {'Sharpe':>7s} {'annRet':>8s} {'maxDD':>8s} "
          f"{'Calmar':>7s} {'turn':>7s} {'maxSecW':>8s} {'maxPnLsh':>9s} "
          f"{'2022Shrp':>9s}")
    for name, scheme, cap in variants:
        res = backtest(panel, k, scheme, cap)
        m = perf(res, HOLDING, 60)
        s22, _ = _yearly_2022(res, HOLDING)
        results[name] = (res, m, s22)
        print(f"{name:22s} {m['sharpe']:7.2f} {m['ann']:8.2%} {m['dd']:8.2%} "
              f"{m['calmar']:7.2f} {m['turnover']:6.1f}x {m['max_secw']:8.1%} "
              f"{m['max_pnl_share']:9.1%} {s22:9.2f}")

    # cost sensitivity for the two most interesting variants + baseline
    print("\n=== Cost sensitivity (net Sharpe) ===")
    show = ["A1 equal (uncapped)", "sector cap 20%", "invvol + cap 20%"]
    print(f"{'variant':22s} " + " ".join(f"{c:>6d}bps" for c in COSTS))
    for name in show:
        res = results[name][0]
        cells = " ".join(f"{perf(res, HOLDING, c)['sharpe']:9.2f}" for c in COSTS)
        print(f"{name:22s} {cells}")

    # yearly for baseline vs recommended combined
    print("\n=== Yearly net@60bps Sharpe: A1 equal vs invvol+cap20 ===")
    base, comb = results["A1 equal (uncapped)"][0], results["invvol + cap 20%"][0]
    yb = pd.Series(base["dates"]).dt.year.to_numpy()
    yc = pd.Series(comb["dates"]).dt.year.to_numpy()
    nb = base["gross"] - (60 / 1e4) * base["turn"]
    nc = comb["gross"] - (60 / 1e4) * comb["turn"]
    print(f"{'year':>5s} {'equal':>8s} {'invvol+cap20':>13s}")
    for y in sorted(set(yb) | set(yc)):
        sb = nb[yb == y]; sc = nc[yc == y]
        f = lambda r: (r.mean()/(r.std(ddof=1)+1e-12)*np.sqrt(252/HOLDING)
                       if len(r) > 1 else float("nan"))
        print(f"{y:5d} {f(sb):8.2f} {f(sc):13.2f}")


if __name__ == "__main__":
    main()
