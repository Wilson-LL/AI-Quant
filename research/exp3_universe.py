"""Experiment 3 — Is the missing alpha a universe problem or a feature problem?

Experiments 1-2 found that in a semis-only universe the ONLY axis of
cross-sectional dispersion is volatility/beta — no standard feature had
vol-neutral alpha. Hypothesis: a multi-sector universe has genuine idiosyncratic
dispersion, so standard features may show vol-neutral (and vol+sector-neutral)
skill that was impossible when every name co-moved.

Decisive comparison — run the SAME analysis on two universes:
  SEMIS  = the original ~20 electronics/semis names.
  MULTI  = SEMIS + ~30 names across financials, telecom, materials, shipping,
           consumer, autos/industrials.

Three questions per universe:
  Q1. How vol-dominated is the cross-section? avg per-date R^2 of fwd_ret ~ vol_60.
  Q2. Does any feature have vol-neutral / vol+sector-neutral IC > 0?
  Q3. Does an OOS model trained on vol-neutral returns earn a positive
      vol-neutral long-short Sharpe?

If MULTI shows positive vol-neutral IC/Sharpe where SEMIS showed none, the
opportunity set was the binding constraint (fixable). If MULTI also shows none,
the standard FEATURES are the problem, not the universe.

Run: python -u research/exp3_universe.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, DEFAULT_UNIVERSE, SECTOR_MAP
from features import FEATURE_COLS
from evaluation import (
    build_panel, cross_sectional_ic, walk_forward_scores, backtest_long_short,
    cross_sectional_neutralize,
)


def _avg_r2_on_vol(panel):
    def _r2(g):
        d = g[["fwd_ret", "vol_60"]].dropna()
        if len(d) < 5 or d["vol_60"].nunique() < 2:
            return np.nan
        x = d["vol_60"].to_numpy(float)
        y = d["fwd_ret"].to_numpy(float)
        b1, b0 = np.polyfit(x, y, 1)
        pred = b0 + b1 * x
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / (ss_tot + 1e-12)
    return panel.groupby("date").apply(_r2).dropna().mean()


def _prep(panel):
    """Attach sector, and vol-neutral + vol&sector-neutral forward returns."""
    panel = panel.copy()
    panel["sector"] = panel["stock"].map(SECTOR_MAP).fillna("other")
    panel["ret_vn"] = cross_sectional_neutralize(panel, "fwd_ret", ["vol_60"]).values
    dummies = pd.get_dummies(panel["sector"], prefix="sec").astype(float)
    dcols = list(dummies.columns)[:-1]  # drop one to avoid collinearity w/ intercept
    panel = pd.concat([panel, dummies[dcols]], axis=1)
    panel["ret_vsn"] = cross_sectional_neutralize(
        panel, "fwd_ret", ["vol_60"] + dcols).values
    return panel


def analyze(universe, name, Y=20):
    # Exclude ETFs from stock-selection cross-section.
    universe = {s: df for s, df in universe.items()
                if SECTOR_MAP.get(s) != "etf"}
    panel = _prep(build_panel(universe, Y=Y))
    n_sectors = panel["sector"].nunique()
    print(f"\n########## {name}: {len(universe)} names, {n_sectors} sectors, "
          f"{panel['date'].nunique()} dates ##########")

    # Q1 -- vol domination
    r2 = _avg_r2_on_vol(panel)
    print(f"Q1. avg per-date R^2(fwd_ret ~ vol_60) = {r2:.3f}  "
          f"(higher = more vol-dominated, less idiosyncratic room)")

    # Q2 -- per-feature IC: raw / vol-neutral / vol+sector-neutral
    print("Q2. per-feature cross-sectional IC (mean over dates):")
    print(f"    {'feature':14s} {'raw':>8s} {'vol-neut':>9s} {'vol+sec-neut':>12s}")
    best = ("", 0.0)
    for f in FEATURE_COLS:
        ic_r, _, _, _ = cross_sectional_ic(panel, f, "fwd_ret")
        ic_v, _, _, _ = cross_sectional_ic(panel, f, "ret_vn")
        ic_s, _, _, _ = cross_sectional_ic(panel, f, "ret_vsn")
        if abs(ic_s) > abs(best[1]):
            best = (f, ic_s)
        print(f"    {f:14s} {ic_r:8.4f} {ic_v:9.4f} {ic_s:12.4f}")
    print(f"    -> strongest vol+sector-neutral feature: {best[0]} ({best[1]:.4f})")

    # Q3 -- OOS model predicting vol-neutral return
    scored = walk_forward_scores(panel, target_col="ret_vn",
                                 feature_cols=FEATURE_COLS, Y=Y)
    oos = scored.dropna(subset=["score"])
    k = max(2, len(universe) // 3)
    ic_oos, ir, _, _ = cross_sectional_ic(oos, "score", "ret_vn")
    m_raw, _ = backtest_long_short(oos, "score", "fwd_ret", k, Y, 30.0)
    m_vn, _ = backtest_long_short(oos, "score", "ret_vn", k, Y, 30.0)
    print(f"Q3. OOS model (predict vol-neutral return), k={k}:")
    print(f"    OOS IC vs vol-neutral ret = {ic_oos:.4f} (IR {ir:.3f})")
    print(f"    L/S Sharpe on RAW ret     = {m_raw['sharpe']:.2f}")
    print(f"    L/S Sharpe on VOL-NEUT ret= {m_vn['sharpe']:.2f}  <-- the honest number")
    return {"name": name, "r2": r2, "best_feat": best,
            "oos_ic_vn": ic_oos, "ls_vn": m_vn["sharpe"]}


def main(Y=20):
    all_ids = list(SECTOR_MAP.keys())
    loaded = load_universe(all_ids)
    print(f"Loaded {len(loaded)} cached names of {len(all_ids)} known.")

    semis = {s: df for s, df in loaded.items() if s in DEFAULT_UNIVERSE}
    multi = loaded

    res = []
    res.append(analyze(semis, "SEMIS-ONLY", Y))
    if len(multi) > len(semis) + 3:
        res.append(analyze(multi, "MULTI-SECTOR", Y))
    else:
        print("\n(Multi-sector cache not ready yet — only SEMIS analysed.)")

    if len(res) == 2:
        s, m = res
        print("\n===== VERDICT =====")
        print(f"vol-domination R^2:  semis {s['r2']:.3f}  ->  multi {m['r2']:.3f}")
        print(f"best vol+sec-neutral feature |IC|: semis {abs(s['best_feat'][1]):.4f}"
              f"  ->  multi {abs(m['best_feat'][1]):.4f}")
        print(f"OOS vol-neutral L/S Sharpe: semis {s['ls_vn']:.2f}"
              f"  ->  multi {m['ls_vn']:.2f}")
        better = m["ls_vn"] > s["ls_vn"] and m["oos_ic_vn"] > s["oos_ic_vn"]
        print(f"\n>>> Broadening the universe {'HELPED' if better else 'did NOT clearly help'} "
              f"vol-neutral alpha.")


if __name__ == "__main__":
    main()
