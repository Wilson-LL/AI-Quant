"""OOS validation of the D1.1 momentum prototype.

Question: does D1.1 survive out-of-sample, and how much is the full-sample number
inflated by (a) selecting the 10% per-name cap on the full sample and (b)
survivorship bias? No ML. The D1.1 rule is NOT changed here — we only test it.

Two analyses:
  A. HELD-OUT split — select on 2018-2021 (IS), evaluate on 2022-2026 (OOS, which
     contains the 2022 momentum crash). Compare uncapped / D1 / D1.1-fixed-10 /
     IS-selected-cap. Answers: does fixed-10 hold OOS? would IS-only selection have
     even picked 10%?
  B. EXPANDING walk-forward — at each rebalance pick the name cap that maximised
     trailing net@60bps Sharpe over all PAST rebalances, apply to the next unseen
     rebalance. Compare adaptive vs fixed-10 vs D1 on the same OOS rebalances.

Plus survivorship-fragility proxies (task 7): long-only vs L/S, and drop-top-
contributor sensitivity. We cannot measure true survivorship bias (twstock lists
only current names) — this bounds fragility, and we state a reasoned haircut.

Run: python -u research/walkforward_d1_1.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, _apply_sector_caps, HOLDING, QUINTILE
from d1_1_pername_cap import _cap_weights

NAME_CANDS = [None, 0.15, 0.125, 0.10, 0.075, 0.05]  # sector cap fixed at 20%
SECTOR_CAP = 0.20
IS_END = pd.Timestamp("2021-12-31")   # IS 2018-2021 ; OOS 2022-2026 (crash in OOS)


def _wts(sub, sector_cap, name_cap):
    inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
    w = dict(zip(sub["stock"], inv / inv.sum()))
    sec = dict(zip(sub["stock"], sub["sector"]))
    if sector_cap is None and name_cap is None:
        pass
    elif name_cap is None:
        w = _apply_sector_caps(w, sec, sector_cap)
    else:
        w = _cap_weights(w, sec, sector_cap, name_cap)
    return w, sec


def _tno(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def backtest(panel, k, sector_cap, name_cap, dmin=None, dmax=None,
             mode="long_short", drop=None, holding=HOLDING):
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    if drop:
        df = df[~df["stock"].isin(drop)]
    if dmin is not None:
        df = df[df["date"] >= dmin]
    if dmax is not None:
        df = df[df["date"] <= dmax]
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]
    gross, turn, keep, mnw, msw = [], [], [], [], []
    sector_pnl, stock_pnl = {}, {}
    pl, ps = {}, {}
    need = 2 * k if mode == "long_short" else k
    for d in rebal:
        day = df[df["date"] == d]
        if len(day) < need:
            continue
        o = day.sort_values("mom")
        longs = o.tail(k)
        wl, secl = _wts(longs, sector_cap, name_cap)
        lr = dict(zip(longs["stock"], longs["fwd_ret"]))
        g = sum(wl[x] * lr[x] for x in wl)
        allw = list(wl.values())
        if mode == "long_short":
            shorts = o.head(k)
            ws, secs = _wts(shorts, sector_cap, name_cap)
            sr = dict(zip(shorts["stock"], shorts["fwd_ret"]))
            g -= sum(ws[x] * sr[x] for x in ws)
            allw += list(ws.values())
            for x in ws:
                sector_pnl[secs[x]] = sector_pnl.get(secs[x], 0.0) - ws[x] * sr[x]
                stock_pnl[x] = stock_pnl.get(x, 0.0) - ws[x] * sr[x]
        else:
            ws = {}
        gross.append(g)
        turn.append(0.5 * (_tno(wl, pl) + _tno(ws, ps)) if mode == "long_short"
                    else _tno(wl, pl))
        pl, ps = wl, ws
        mnw.append(max(allw))
        sw = {}
        for x, wx in wl.items():
            sw[secl[x]] = sw.get(secl[x], 0.0) + wx
        msw.append(max(sw.values()))
        for x in wl:
            sector_pnl[secl[x]] = sector_pnl.get(secl[x], 0.0) + wl[x] * lr[x]
            stock_pnl[x] = stock_pnl.get(x, 0.0) + wl[x] * lr[x]
        keep.append(d)
    return {"gross": np.array(gross), "turn": np.array(turn),
            "dates": np.array(keep), "mnw": np.array(mnw), "msw": np.array(msw),
            "sector_pnl": sector_pnl, "stock_pnl": stock_pnl}


def metrics(res, cost=60, holding=HOLDING):
    net = res["gross"] - (cost / 1e4) * res["turn"]
    net = net[~np.isnan(net)]
    if len(net) < 2:
        return {"sharpe": float("nan"), "n": len(net)}
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
    ym = {y: float(np.nanmean(net60[yrs == y])) for y in sorted(set(yrs))}
    wy = min(ym, key=ym.get)
    return {"sharpe": sharpe, "ann": ann, "dd": dd, "calmar": ann / (abs(dd) + 1e-12),
            "turnover": float(np.nanmean(res["turn"])) * ppy,
            "mnw": float(np.nanmean(res["mnw"])), "msw": float(np.nanmean(res["msw"])),
            "share": share, "worst": wy, "worst_mean": ym[wy],
            "y2022": ym.get(2022, float("nan")), "n": len(net)}


def _line(tag, m):
    print(f"{tag:22s} {m['sharpe']:6.2f} {m['ann']:7.1%} {m['dd']:7.1%} "
          f"{m['calmar']:6.2f} {m['turnover']:5.1f}x {m['mnw']:6.1%} {m['msw']:6.1%} "
          f"{m['share']:6.1%} {m.get('worst_mean',float('nan'))*100:6.2f} "
          f"{m.get('y2022',float('nan'))*100:6.2f} {m['n']:4d}")


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    print(f"OOS validation of D1.1. {panel['stock'].nunique()} names, k={k}. "
          f"IS=2018..2021, OOS=2022..2026 (crash in OOS). net@60bps; "
          f"survivorship-biased.\n")

    VAR = {"uncapped (no caps)": (None, None), "D1 (sector20)": (0.20, None),
           "D1.1 fixed 10%": (0.20, 0.10)}
    hdr = (f"{'variant':22s} {'Shrp':>6s} {'annRet':>7s} {'maxDD':>7s} {'Calmr':>6s} "
           f"{'turn':>6s} {'maxNm':>6s} {'maxSc':>6s} {'PnLsh':>6s} {'wrYr%':>6s} "
           f"{'2022%':>6s} {'n':>4s}")

    # ---------- Analysis A: held-out split ----------
    print("=== A. HELD-OUT: full-sample vs IS(2018-21) vs OOS(2022-26) ===")
    print(hdr)
    for tag, (sc, nc) in VAR.items():
        _line(tag + " [FULL]", metrics(backtest(panel, k, sc, nc)))
    print()
    for tag, (sc, nc) in VAR.items():
        _line(tag + " [IS]", metrics(backtest(panel, k, sc, nc, dmax=IS_END)))
    print()
    for tag, (sc, nc) in VAR.items():
        _line(tag + " [OOS]", metrics(backtest(panel, k, sc, nc, dmin=IS_END)))

    # IS-selected cap (only past data) -> apply OOS
    is_scores = {nc: metrics(backtest(panel, k, 0.20, nc, dmax=IS_END))["sharpe"]
                 for nc in NAME_CANDS}
    best_is = max(is_scores, key=is_scores.get)
    print(f"\nIS(2018-21) best name cap by Sharpe: "
          f"{'uncapped' if best_is is None else f'{best_is:.1%}'} "
          f"(IS scores: " + ", ".join(
              f"{'unc' if c is None else f'{int(c*100)}'}:{is_scores[c]:.2f}"
              for c in NAME_CANDS) + ")")
    print("Applied to OOS:")
    _line(f"IS-selected ({'unc' if best_is is None else f'{int(best_is*100)}%'}) [OOS]",
          metrics(backtest(panel, k, 0.20, best_is, dmin=IS_END)))
    _line("D1.1 fixed 10% [OOS]", metrics(backtest(panel, k, 0.20, 0.10, dmin=IS_END)))

    # ---------- Analysis B: expanding walk-forward ----------
    print("\n=== B. EXPANDING walk-forward (adaptive cap, past-only) ===")
    cand = {nc: backtest(panel, k, 0.20, nc) for nc in NAME_CANDS}
    dates = cand[0.10]["dates"]
    n = len(dates)
    MIN_TRAIN = 36  # ~3 yrs of rebalances before first OOS decision
    net = {nc: cand[nc]["gross"] - (60/1e4) * cand[nc]["turn"] for nc in NAME_CANDS}
    wf_ret, chosen = [], []
    for i in range(MIN_TRAIN, n):
        # pick cap with best trailing net@60 Sharpe over [0, i)
        best, bs = None, -1e9
        for nc in NAME_CANDS:
            r = net[nc][:i]
            r = r[~np.isnan(r)]
            s = r.mean() / (r.std(ddof=1) + 1e-12) if len(r) > 1 else -1e9
            if s > bs:
                bs, best = s, nc
        wf_ret.append(net[best][i])
        chosen.append(best)
    wf_ret = np.array(wf_ret)
    oos_dates = dates[MIN_TRAIN:]

    def _sr(r):
        r = r[~np.isnan(r)]
        return r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / HOLDING)

    print(f"WF OOS window: {pd.Timestamp(oos_dates[0]).date()}..{pd.Timestamp(oos_dates[-1]).date()} "
          f"({len(wf_ret)} rebalances)")
    print(f"  adaptive WF   net@60 Sharpe = {_sr(wf_ret):.2f}")
    print(f"  fixed 10%     net@60 Sharpe = {_sr(net[0.10][MIN_TRAIN:]):.2f}")
    print(f"  D1 sector-only net@60 Sharpe = {_sr(net[None][MIN_TRAIN:]):.2f}")
    cc = pd.Series([("unc" if c is None else f"{int(c*100)}%") for c in chosen]).value_counts()
    print("  cap chosen (count): " + ", ".join(f"{i}:{v}" for i, v in cc.items()))

    # ---------- Task 7: survivorship-fragility proxies ----------
    print("\n=== 7. Survivorship-fragility proxies (OOS 2022-26) ===")
    oos = backtest(panel, k, 0.20, 0.10, dmin=IS_END)
    lo = backtest(panel, k, 0.20, 0.10, dmin=IS_END, mode="long_only")
    print(f"  long-short OOS Sharpe {metrics(oos)['sharpe']:.2f} | "
          f"long-only OOS Sharpe {metrics(lo)['sharpe']:.2f} "
          f"(long-only carries beta/survivorship — gap indicates how much is the "
          f"survivor long book)")
    top = sorted(oos["stock_pnl"].items(), key=lambda x: -abs(x[1]))[:3]
    print("  top-3 |contribution| names (OOS): " +
          ", ".join(f"{s}({SECTOR_MAP.get(s,'?')}) {v*100:+.1f}%" for s, v in top))
    for j in (1, 3):
        drop = [s for s, _ in top[:j]]
        m = metrics(backtest(panel, k, 0.20, 0.10, dmin=IS_END, drop=drop))
        print(f"  drop top-{j} contributors -> OOS Sharpe {m['sharpe']:.2f} "
              f"(from {metrics(oos)['sharpe']:.2f})")


if __name__ == "__main__":
    main()
