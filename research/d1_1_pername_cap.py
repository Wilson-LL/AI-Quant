"""D1.1 — does adding a per-NAME weight cap to the frozen D1 rule reduce
single-name concentration without materially hurting performance?

Frozen baseline (unchanged): D1 = momentum quintile L/S + inverse-vol sizing +
20% SECTOR cap + 20d hold. D1.1 candidate = D1 + a per-NAME cap. The "uncapped"
row here reuses the exact D1 sector-cap path (portfolio_d1._apply_sector_caps),
so it reproduces the validated D1 numbers — proof the baseline is untouched.

Pre-declared per-name caps: uncapped, 15%, 12.5%, 10%, 7.5%, 5%.
Not tuned for Sharpe: per the brief, prefer the simplest conservative cap if
several tie, not the highest-Sharpe one.

Run: python -u research/d1_1_pername_cap.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, _apply_sector_caps, HOLDING, QUINTILE

SECTOR_CAP = 0.20
NAME_CAPS = [None, 0.15, 0.125, 0.10, 0.075, 0.05]
COSTS = [0, 60, 80]


def _cap_weights(weights, sector_of, sector_cap, name_cap):
    """Enforce BOTH a per-name cap and a per-sector cap on positive weights that
    sum to 1, by iterative clip-and-redistribute. Caps are floored (name at
    1/n_names, sector at 1/n_sectors) to stay feasible."""
    w = dict(weights)
    n = len(w)
    eff_name = max(name_cap, 1.0 / n + 1e-12)
    secs = {sector_of[x] for x in w}
    eff_sec = max(sector_cap, 1.0 / len(secs) + 1e-12)
    for _ in range(500):
        changed = False
        # per-name cap
        over = [x for x in w if w[x] > eff_name + 1e-9]
        if over:
            excess = sum(w[x] - eff_name for x in over)
            for x in over:
                w[x] = eff_name
            under = [x for x in w if w[x] < eff_name - 1e-12]
            tot = sum(w[x] for x in under)
            if tot > 1e-12:
                for x in under:
                    w[x] += excess * w[x] / tot
            changed = True
        # per-sector cap
        ssum = {}
        for x, wx in w.items():
            ssum[sector_of[x]] = ssum.get(sector_of[x], 0.0) + wx
        secover = [s for s, v in ssum.items() if v > eff_sec + 1e-9]
        if secover:
            secset = set(secover)
            excess = 0.0
            for s in secover:
                scale = eff_sec / ssum[s]
                for x in w:
                    if sector_of[x] == s:
                        excess += w[x] * (1 - scale)
                        w[x] *= scale
            under = [x for x in w if sector_of[x] not in secset]
            tot = sum(w[x] for x in under)
            if tot > 1e-12:
                for x in under:
                    w[x] += excess * w[x] / tot
            changed = True
        if not changed:
            break
    s = sum(w.values())
    return {x: v / s for x, v in w.items()}


def _leg_weights(sub, name_cap):
    inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
    w = dict(zip(sub["stock"], inv / inv.sum()))
    sec = dict(zip(sub["stock"], sub["sector"]))
    if name_cap is None:                       # exact frozen-D1 path
        w = _apply_sector_caps(w, sec, SECTOR_CAP)
    else:
        w = _cap_weights(w, sec, SECTOR_CAP, name_cap)
    return w, sec


def _leg_turnover(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def backtest(panel, k, name_cap, holding=HOLDING):
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]
    gross, turn, keep, max_namew, max_secw = [], [], [], [], []
    sector_pnl = {}
    pl, ps = {}, {}
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < 2 * k:
            continue
        o = day.sort_values("mom")
        longs, shorts = o.tail(k), o.head(k)
        wl, secl = _leg_weights(longs, name_cap)
        ws, secs = _leg_weights(shorts, name_cap)
        lr = dict(zip(longs["stock"], longs["fwd_ret"]))
        sr = dict(zip(shorts["stock"], shorts["fwd_ret"]))
        gross.append(sum(wl[x] * lr[x] for x in wl) - sum(ws[x] * sr[x] for x in ws))
        turn.append(0.5 * (_leg_turnover(wl, pl) + _leg_turnover(ws, ps)))
        pl, ps = wl, ws
        max_namew.append(max(list(wl.values()) + list(ws.values())))
        sw = {}
        for x, wx in wl.items():
            sw[secl[x]] = sw.get(secl[x], 0.0) + wx
        max_secw.append(max(sw.values()))
        for x in wl:
            sector_pnl[secl[x]] = sector_pnl.get(secl[x], 0.0) + wl[x] * lr[x]
        for x in ws:
            sector_pnl[secs[x]] = sector_pnl.get(secs[x], 0.0) - ws[x] * sr[x]
        keep.append(d)
    return {"gross": np.array(gross), "turn": np.array(turn),
            "dates": np.array(keep), "sector_pnl": sector_pnl,
            "max_namew": np.array(max_namew), "max_secw": np.array(max_secw)}


def perf(res, cost_bps, holding=HOLDING):
    net = res["gross"] - (cost_bps / 1e4) * res["turn"]
    net = net[~np.isnan(net)]
    ppy = 252 / holding
    mean, std = net.mean(), net.std(ddof=1)
    sharpe = mean / (std + 1e-12) * np.sqrt(ppy)
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = (1 + mean) ** ppy - 1
    tot = sum(res["sector_pnl"].values())
    share = max(res["sector_pnl"].values()) / tot if tot > 0 else float("nan")
    yrs = pd.Series(res["dates"]).dt.year.to_numpy()
    net60 = res["gross"] - (60 / 1e4) * res["turn"]
    ymean = {y: float(np.nanmean(net60[yrs == y])) for y in sorted(set(yrs))}
    wy = min(ymean, key=ymean.get)
    return {"sharpe": sharpe, "ann": ann, "dd": dd, "calmar": ann / (abs(dd) + 1e-12),
            "turnover": float(np.nanmean(res["turn"])) * ppy,
            "max_namew": float(np.nanmean(res["max_namew"])),
            "max_secw": float(np.nanmean(res["max_secw"])),
            "max_pnl_share": share, "worst_year": wy, "worst_year_mean": ymean[wy],
            "y2022": ymean.get(2022, float("nan")), "ymean": ymean}


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    n = panel["stock"].nunique()
    k = max(3, round(QUINTILE * n))
    floor = 1.0 / k
    print(f"D1.1: {n} names, k={k}/leg (name-cap feasibility floor = {floor:.1%}). "
          f"Frozen base = D1 (sector cap {int(SECTOR_CAP*100)}%). net@60bps; "
          f"survivorship-biased.\n")

    R = {("uncapped (=D1)" if c is None else f"name cap {c:.1%}"):
         (c, backtest(panel, k, c)) for c in NAME_CAPS}

    print(f"{'variant':16s} {'Sh@0':>6s} {'Sh@60':>6s} {'Sh@80':>6s} {'annRet':>7s} "
          f"{'maxDD':>7s} {'Calmar':>7s} {'turn':>6s} {'maxNm':>6s} {'maxSec':>7s} "
          f"{'PnLsh':>6s} {'2022%':>6s} {'wYr%':>6s} {'feasible':>8s}")
    metr = {}
    for name, (c, res) in R.items():
        m = perf(res, 60)
        metr[name] = (c, m)
        feasible = "yes" if (c is None or c >= floor - 1e-9) else "FLOORED"
        print(f"{name:16s} {perf(res,0)['sharpe']:6.2f} {m['sharpe']:6.2f} "
              f"{perf(res,80)['sharpe']:6.2f} {m['ann']:7.1%} {m['dd']:7.1%} "
              f"{m['calmar']:7.2f} {m['turnover']:5.1f}x {m['max_namew']:6.1%} "
              f"{m['max_secw']:7.1%} {m['max_pnl_share']:6.1%} {m['y2022']*100:6.2f} "
              f"{m['worst_year_mean']*100:6.2f} {feasible:>8s}")

    base = metr["uncapped (=D1)"][1]
    print("\n=== Decision vs D1 (prefer cap only if it meaningfully cuts single-name "
          "concentration or DD, with net Sharpe drop <= 0.10) ===")
    for name, (c, m) in metr.items():
        if c is None:
            continue
        d_sh = base["sharpe"] - m["sharpe"]
        d_nm = (base["max_namew"] - m["max_namew"]) / base["max_namew"]
        # positive = drawdown improved (smaller magnitude)
        d_dd = (abs(base["dd"]) - abs(m["dd"])) / abs(base["dd"])
        ok = (d_sh <= 0.10) and (d_nm >= 0.15 or d_dd >= 0.15)
        print(f"  {name:14s} nameConc {d_nm:+.0%}  DD {d_dd:+.0%}  "
              f"SharpeDrop {d_sh:+.2f} -> {'PASS' if ok else 'weak/fail'}")

    print("\n=== Yearly net@60bps mean% — D1 vs name-cap 10% vs 12.5% ===")
    cols = ["uncapped (=D1)", "name cap 12.5%", "name cap 10.0%"]
    yset = sorted(set(pd.Series(R['uncapped (=D1)'][1]['dates']).dt.year))
    print("year   " + "  ".join(f"{cc[:12]:>12s}" for cc in cols))
    for y in yset:
        cells = "  ".join(f"{metr[cc][1]['ymean'].get(y, float('nan'))*100:12.2f}"
                          for cc in cols)
        print(f"{y:<6d} {cells}")


if __name__ == "__main__":
    main()
