"""Experiment 4 — Is the Experiment-3 cross-sectional alpha real, stable, robust?

Goal: VALIDATE (or kill) the +0.75 vol-neutral long-short Sharpe found on the
multi-sector universe. No new architectures — this is pure robustness.

Pipeline: one walk-forward linear model trained to predict vol-neutral 20d
returns from standard features on the 100+ name universe, then interrogate the
out-of-sample scores seven ways:

  3. Rolling per-year OOS IC & L/S Sharpe        (is it persistent or one window?)
  4. Sector-by-sector P&L attribution            (broad or one lucky sector?)
  5. Alpha decay across forward horizons          (5/10/20/40/60d)
  6. Turnover                                      (is it tradeable?)
  7. Transaction-cost sensitivity + breakeven bps
  8. Factor-exposure regression of the L/S returns on market / volatility /
     momentum / size factors -> residual ALPHA and its t-stat (the real test)
  9. Persistence verdict.

Survivorship note: twstock exposes only currently-listed codes, so delisted
names cannot be included; the universe is therefore survivorship-biased upward.
We reduce (not remove) it by spanning sectors and size tiers incl. weaker mid/
small caps. This caveat is printed with the results.

Run: python -u research/exp4_robustness.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from features import compute_features, FEATURE_COLS
from evaluation import (
    cross_sectional_ic, walk_forward_scores, cross_sectional_neutralize,
)

HORIZONS = [5, 10, 20, 40, 60]
Y = 20
COST_SWEEP_BPS = [0, 5, 10, 20, 30, 50, 75, 100]


def build_full_panel(universe):
    rows = []
    for sid, df in universe.items():
        df = df.sort_values("date").reset_index(drop=True)
        feats = compute_features(df)
        c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        n = len(c)
        blk = feats.copy()
        blk["stock"] = sid
        blk["sector"] = SECTOR_MAP.get(sid, "other")
        for h in HORIZONS:
            fr = np.full(n, np.nan)
            fr[:n - h] = c[h:] / c[:n - h] - 1.0
            blk[f"fwd_{h}"] = fr
        # size proxy = log trailing 20d mean dollar volume (causal). Bigger =
        # more liquid ~ larger cap (a proxy, absent true market cap).
        dv = pd.Series(c * v).rolling(20).mean()
        blk["size"] = np.log(dv.to_numpy() + 1.0)
        rows.append(blk)
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["date", "stock"]).reset_index(drop=True)


def _ls_and_factors(oos, k, holding):
    """Single pass over rebalance dates: strategy L/S return, factor L/S returns
    (market/vol/mom/size), turnover, and per-sector P&L attribution. All aligned."""
    dates = np.array(sorted(oos["date"].unique()))
    rebal = dates[::holding]
    strat, mkt, volf, momf, sizf = [], [], [], [], []
    turns, keep_dates = [], []
    sector_pnl = {}
    prev = set()
    for d in rebal:
        day = oos[oos["date"] == d].dropna(subset=["score", "fwd_20"])
        if len(day) < 2 * k:
            continue
        o = day.sort_values("score")
        longs, shorts = o.tail(k), o.head(k)
        strat.append(longs["fwd_20"].mean() - shorts["fwd_20"].mean())
        mkt.append(day["fwd_20"].mean())
        for col, acc in [("vol_60", volf), ("mom_60", momf), ("size", sizf)]:
            of = day.dropna(subset=[col]).sort_values(col)
            acc.append(of["fwd_20"].tail(k).mean() - of["fwd_20"].head(k).mean()
                       if len(of) >= 2 * k else np.nan)
        names = set(longs["stock"]) | set(shorts["stock"])
        turns.append(len(names ^ prev) / max(len(names), 1))
        prev = names
        keep_dates.append(d)
        # sector attribution of the L/S spread
        for _, r in longs.iterrows():
            sector_pnl[r["sector"]] = sector_pnl.get(r["sector"], 0.0) + r["fwd_20"] / k
        for _, r in shorts.iterrows():
            sector_pnl[r["sector"]] = sector_pnl.get(r["sector"], 0.0) - r["fwd_20"] / k
    return {
        "dates": np.array(keep_dates),
        "strat": np.array(strat), "mkt": np.array(mkt),
        "vol": np.array(volf), "mom": np.array(momf), "size": np.array(sizf),
        "turnover": np.array(turns), "sector_pnl": sector_pnl,
    }


def _sharpe(rets, holding):
    rets = rets[~np.isnan(rets)]
    if len(rets) < 2:
        return float("nan")
    ppy = 252 / holding
    return rets.mean() / (rets.std(ddof=1) + 1e-12) * np.sqrt(ppy)


def _ols_tstats(y, X):
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(1, len(y) - X.shape[1])
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / (se + 1e-12)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / (ss_tot + 1e-12)
    return b, t, r2


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    universe = load_universe(ids)
    k = max(5, len(universe) // 5)
    print(f"Universe: {len(universe)} names, {len(set(SECTOR_MAP[s] for s in universe))} "
          f"sectors, k={k}, holding={Y}")
    print("SURVIVORSHIP: only currently-listed names (twstock); results are biased "
          "upward. Sectors/size tiers spread to reduce, not remove, the bias.\n")

    panel = build_full_panel(universe)
    panel["ret_vn"] = cross_sectional_neutralize(panel, "fwd_20", ["vol_60"]).values
    panel["fwd_ret"] = panel["fwd_20"]   # alias used by cross_sectional_ic (task 3)

    # Walk-forward model predicting vol-neutral 20d return.
    scored = walk_forward_scores(panel, target_col="ret_vn",
                                 feature_cols=FEATURE_COLS, Y=Y)
    oos = scored.dropna(subset=["score"]).copy()
    oos["year"] = oos["date"].dt.year
    print(f"OOS scored obs: {len(oos)}  ({oos['date'].min().date()}"
          f"..{oos['date'].max().date()})")

    F = _ls_and_factors(oos, k, Y)
    strat = F["strat"]
    yrs = pd.Series(F["dates"]).dt.year.to_numpy()

    # --- Task 3: rolling per-year ---
    print("\n=== 3. Rolling per-year (OOS) ===")
    print(f"{'year':>5} {'IC_vn':>8} {'IC_raw':>8} {'LS_Sharpe':>10} {'meanLS%':>8} {'rebals':>7}")
    for yr in sorted(oos["year"].unique()):
        g = oos[oos["year"] == yr]
        ic_vn, _, _, _ = cross_sectional_ic(g, "score", "ret_vn")
        ic_rw, _, _, _ = cross_sectional_ic(g, "score", "fwd_ret")
        r = strat[yrs == yr]
        print(f"{yr:>5} {ic_vn:8.4f} {ic_rw:8.4f} {_sharpe(r, Y):10.2f} "
              f"{np.nanmean(r)*100 if len(r) else float('nan'):8.2f} {len(r):7d}")

    # --- Task 4: sector attribution ---
    print("\n=== 4. Sector P&L attribution (sum of L/S spread contribution) ===")
    sp = sorted(F["sector_pnl"].items(), key=lambda x: -x[1])
    for sec, pnl in sp:
        print(f"  {sec:14s} {pnl*100:+7.1f}%  cumulative L/S contribution")

    # --- Task 5: alpha decay ---
    print("\n=== 5. Alpha decay: OOS IC(score, forward return) by horizon ===")
    for h in HORIZONS:
        ic_h, _, nd, _ = cross_sectional_ic(oos, "score", f"fwd_{h}")
        print(f"  h={h:>2}d: IC={ic_h:7.4f}")

    # --- Task 6: turnover ---
    print("\n=== 6. Turnover ===")
    print(f"  mean turnover/rebalance = {np.nanmean(F['turnover']):.2f} "
          f"(fraction of book replaced); ann. ~{np.nanmean(F['turnover'])*252/Y:.1f}x")

    # --- Task 7: cost sensitivity ---
    print("\n=== 7. Transaction-cost sensitivity (net L/S) ===")
    gross_mean = np.nanmean(strat)
    mean_turn = np.nanmean(F["turnover"])
    print(f"{'cost_bps':>8} {'net_Sharpe':>11} {'net_mean%':>10}")
    for cb in COST_SWEEP_BPS:
        net = strat - (cb / 1e4) * F["turnover"]
        print(f"{cb:>8} {_sharpe(net, Y):11.2f} {np.nanmean(net)*100:10.3f}")
    be = gross_mean / (mean_turn + 1e-12) * 1e4
    print(f"  breakeven cost ~= {be:.0f} bps/round-trip")

    # --- Task 8: factor exposure ---
    print("\n=== 8. Factor-exposure regression of L/S returns ===")
    m = ~np.isnan(F["vol"]) & ~np.isnan(F["mom"]) & ~np.isnan(F["size"])
    y = strat[m]
    X = np.column_stack([np.ones(m.sum()), F["mkt"][m], F["vol"][m],
                         F["mom"][m], F["size"][m]])
    b, t, r2 = _ols_tstats(y, X)
    names = ["alpha", "market", "volatility", "momentum", "size"]
    ppy = 252 / Y
    print(f"{'factor':>11} {'beta':>9} {'t-stat':>8}")
    for nm, bb, tt in zip(names, b, t):
        print(f"{nm:>11} {bb:9.4f} {tt:8.2f}")
    print(f"  regression R^2 = {r2:.2f}")
    print(f"  ANNUALISED alpha = {(1+b[0])**ppy-1:.2%}  (t={t[0]:.2f})  "
          f"<-- residual skill after market/vol/mom/size")

    # --- Task 9: persistence verdict ---
    print("\n=== 9. Persistence verdict ===")
    yearly = [_sharpe(strat[yrs == yr], Y) for yr in sorted(set(yrs))]
    pos = sum(1 for s in yearly if s > 0)
    print(f"  full-sample L/S Sharpe (gross)   = {_sharpe(strat, Y):.2f}")
    print(f"  net@30bps L/S Sharpe             = "
          f"{_sharpe(strat-(30/1e4)*F['turnover'], Y):.2f}")
    print(f"  positive years                   = {pos}/{len(yearly)}")
    print(f"  factor-neutral annualised alpha  = {(1+b[0])**ppy-1:.2%} (t={t[0]:.2f})")
    verdict = (t[0] > 2 and pos >= 0.6 * len(yearly)
               and _sharpe(strat-(30/1e4)*F['turnover'], Y) > 0.3)
    print(f"\n>>> Alpha is {'VALIDATED (persistent + factor-neutral)' if verdict else 'NOT robustly validated'} "
          f"on this survivorship-biased universe.")


if __name__ == "__main__":
    main()
