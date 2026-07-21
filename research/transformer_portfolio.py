"""Portfolio construction + backtest for transformer (or any) score panels.

Input: a score panel DataFrame with columns [date, stock, score, fwd_h] where
fwd_h is the realized forward return over the holding horizon measured from
close[t+1] (execution lag already applied by dataset_transformer_eod).

Constructions (brief §9): long-only top-N / top-quintile, long-short quintiles,
equal / rank / inverse-vol weighting, hard 10% per-name cap, soft 20% sector cap,
no-trade band. Metrics: net Sharpe at multiple costs, ann ret, max DD, Calmar,
turnover, hit rate, yearly breakdown, IC / rank IC, max name weight.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SECTOR_MAP  # noqa: E402

NAME_CAP = 0.10
SECTOR_CAP = 0.20
CAP_TOL = 1e-6


def _waterfill_name_cap(base, name_cap):
    """Proportional weights under a HARD per-name cap. Saturated names are
    frozen at the cap and the remaining mass is re-shared among unsaturated
    names by base weight. Terminates in <= n passes (the saturated set only
    grows). If n*cap < 1 the cap is infeasible -> equal weights (upstream
    callers skip such thin books via min_names)."""
    n = len(base)
    if n * name_cap < 1.0 - 1e-12:
        return np.full(n, 1.0 / n)
    fixed = np.zeros(n, bool)
    w = base / base.sum()
    for _ in range(n + 1):
        over = (w > name_cap + 1e-12) & ~fixed
        if not over.any():
            break
        fixed |= over
        rem = 1.0 - fixed.sum() * name_cap
        w[fixed] = name_cap
        fb = np.where(fixed, 0.0, base)
        if fb.sum() <= 0:
            break
        w[~fixed] = rem * fb[~fixed] / fb.sum()
    return w


def cap_weights(w, sectors=None, name_cap=NAME_CAP, sector_cap=SECTOR_CAP,
                iters=20):
    """Enforce a HARD per-name cap and a SOFT per-sector cap on positive
    weights summing to 1 (D1.1 convention). Sector cap is floored at
    max(sector_cap, 1/n_sectors). The name cap always wins: the last pass is
    a name-cap water-fill, so no name exceeds it whenever n*cap >= 1.
    (The previous clip-and-redistribute loop could ping-pong: clipped names
    received excess back and stabilised ~11% > cap.)"""
    base = np.maximum(np.asarray(w, np.float64), 0)
    if base.sum() <= 0:
        return base
    ww = _waterfill_name_cap(base, name_cap)
    if sectors is None:
        return ww
    sec = list(sectors)
    n_sec = len(set(sec))
    eff_sec_cap = max(sector_cap, 1.0 / max(n_sec, 1)) + 1e-9
    for _ in range(iters):
        s = pd.Series(ww).groupby(sec).transform("sum").to_numpy()
        oversec = s > eff_sec_cap + 1e-9
        if not oversec.any():
            break
        scale = np.where(oversec, eff_sec_cap / s, 1.0)
        w2 = ww * scale
        excess = 1.0 - w2.sum()
        recv = ~oversec & (w2 < name_cap - 1e-12)
        if recv.any() and w2[recv].sum() > 0:
            w2[recv] += excess * w2[recv] / w2[recv].sum()
        ww = _waterfill_name_cap(np.maximum(w2, 0), name_cap)
    return ww


def _leg_weights(day, names, weighting, vol_col="vol_20"):
    """Weight vector for one leg (long or short), capped."""
    sub = day[day["stock"].isin(names)].set_index("stock").loc[list(names)]
    if weighting == "rank":
        base = sub["score"].rank().to_numpy()
    elif weighting == "invvol" and vol_col in sub:
        v = sub[vol_col].to_numpy(np.float64)
        v = np.where(np.isfinite(v) & (v > 0), v, np.nanmedian(v[np.isfinite(v)]) or 1.0)
        base = 1.0 / v
    else:
        base = np.ones(len(names))
    sectors = [SECTOR_MAP.get(s, "other") for s in names]
    return cap_weights(base, sectors)


def backtest_scores(panel, holding=20, quintile=0.2, k=None, mode="long_short",
                    weighting="equal", cost_bps_list=(0, 60, 100, 150),
                    no_trade_band=0.0, ret_col="fwd_h", min_names=60):
    """Backtest a score panel. Rebalance every `holding` panel dates.

    no_trade_band: keep an incumbent name if its rank percentile is within
    `band` of the selection threshold (reduces turnover).
    Returns dict of metrics + per-rebalance arrays.
    """
    df = (panel.dropna(subset=["score", ret_col])
                .drop_duplicates(["date", "stock"], keep="last").copy())
    dates = np.array(sorted(df["date"].unique()))
    rebal = dates[::holding]
    by_date = dict(tuple(df.groupby("date")))

    gross, weighted_turn, kept_dates = [], [], []
    max_name_w = 0.0
    prev_w = {}
    for d in rebal:
        day = by_date.get(d)
        if day is None:
            continue
        n = day["stock"].nunique()
        if n < min_names:
            continue  # thin boundary cross-section -> unrepresentative book
        kq = k or max(3, round(quintile * n))
        need = 2 * kq if mode == "long_short" else kq
        if n < need:
            continue
        o = day.sort_values("score")
        long_pool = set(o.tail(int(kq * (1 + no_trade_band * 2)))["stock"]) if no_trade_band else None
        longs = list(o.tail(kq)["stock"])
        if no_trade_band and prev_w:
            # keep incumbents still inside the widened pool
            incumbents = [s for s in prev_w.get("L", {}) if s in long_pool]
            fresh = [s for s in longs if s not in incumbents]
            longs = (incumbents + fresh)[:kq]
        wl = _leg_weights(day, longs, weighting)
        ret_map = day.set_index("stock")[ret_col]
        g = float(np.sum(wl * ret_map.loc[longs].to_numpy()))
        new_w = {"L": dict(zip(longs, wl))}
        if mode == "long_short":
            shorts = list(o.head(kq)["stock"])
            ws = _leg_weights(day, shorts, weighting)
            g -= float(np.sum(ws * ret_map.loc[shorts].to_numpy()))
            new_w["S"] = dict(zip(shorts, ws))
        # L1 turnover per leg (one-way), averaged over legs
        t = 0.0
        for leg in new_w:
            old = prev_w.get(leg, {})
            allnames = set(new_w[leg]) | set(old)
            t += sum(abs(new_w[leg].get(s, 0) - old.get(s, 0)) for s in allnames) / 2
        t /= len(new_w)
        max_name_w = max(max_name_w, max(max(w.values()) for w in new_w.values()))
        gross.append(g)
        weighted_turn.append(t)
        kept_dates.append(d)
        prev_w = new_w

    gross = np.array(gross)
    turn = np.array(weighted_turn)
    kept_dates = np.array(kept_dates)
    # cost = one-way bps * dollars traded; with per-leg-average turnover t,
    # dollars traded = n_legs * 2t, one-way = cb/2 => cost = n_legs*(cb/1e4)*t
    n_legs = 2 if mode == "long_short" else 1

    ic, ics = _panel_ic(df, ret_col)
    out = {"mode": mode, "weighting": weighting, "holding": holding,
           "n_rebal": len(gross), "rank_ic": ic,
           "ic_ir": float(ic / (ics.std() + 1e-12)) if len(ics) else float("nan"),
           "max_name_weight": round(float(max_name_w), 4),
           "avg_turnover": float(np.mean(turn)) if len(turn) else float("nan")}
    for cb in cost_bps_list:
        out[f"net{cb}"] = _metrics(gross - n_legs * (cb / 1e4) * turn, holding)
    # yearly at 60bps
    net60 = gross - n_legs * (60 / 1e4) * turn
    yr = pd.Series(kept_dates).astype("datetime64[ns]").dt.year.to_numpy() if len(kept_dates) else np.array([])
    yearly = {}
    for y in sorted(set(yr)):
        r = net60[yr == y]
        s = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / holding) if len(r) > 1 else float("nan")
        yearly[int(y)] = {"sharpe": round(float(s), 2), "mean_pct": round(float(np.mean(r)) * 100, 2), "n": int(len(r))}
    out["yearly_net60"] = yearly
    out["gross"] = gross.tolist()
    out["turnover"] = turn.tolist()
    out["dates"] = [str(d)[:10] for d in kept_dates]
    return out


def _panel_ic(df, ret_col):
    def one(g):
        g = g[["score", ret_col]].dropna()
        if len(g) < 5 or g["score"].nunique() < 2:
            return np.nan
        return g["score"].rank().corr(g[ret_col].rank())
    ics = df.groupby("date", group_keys=False).apply(one, include_groups=False).dropna()
    return (float(ics.mean()) if len(ics) else float("nan")), ics


def _metrics(net, holding):
    net = net[~np.isnan(net)]
    if len(net) < 2:
        return {k: float("nan") for k in ["sharpe", "ann_ret", "max_dd", "calmar", "hit", "n"]}
    ppy = 252 / holding
    mean, std = net.mean(), net.std(ddof=1)
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = (1 + mean) ** ppy - 1
    return {"sharpe": round(float(mean / (std + 1e-12) * np.sqrt(ppy)), 3),
            "ann_ret": round(float(ann), 4), "max_dd": round(dd, 4),
            "calmar": round(float(ann / (abs(dd) + 1e-12)), 2),
            "hit": round(float((net > 0).mean()), 3), "n": int(len(net))}


def summarize(result, label=""):
    n60, n100 = result["net60"], result["net100"]
    return (f"{label:34s} IC {result['rank_ic']:+.4f} | net60 Sharpe {n60['sharpe']:5.2f} "
            f"ann {n60['ann_ret']:+7.2%} dd {n60['max_dd']:7.2%} | net100 {n100['sharpe']:5.2f} "
            f"| turn {result['avg_turnover']:.2f} | maxW {result['max_name_weight']:.1%} "
            f"| n {n60['n']}")
