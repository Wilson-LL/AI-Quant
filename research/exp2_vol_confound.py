"""Experiment 2 — Is the triple-barrier target's edge real alpha, or a
volatility/beta bet in disguise?

Experiment 1 found the triple-barrier targets rank future returns far better than
return-based targets (OOS IC ~0.05, L/S Sharpe ~1.0), but they are learned almost
entirely through `vol_60`. In a 2018-2026 semis bull market, high-vol names had
high returns, so the "skill" may be regime-linked volatility exposure, not
idiosyncratic alpha. Four checks:

  T0. Naive volatility factor      — how much does simply ranking by vol_60 earn?
  T1. Skill WITHOUT vol features   — retrain the target dropping vol_20/vol_60.
  T2. Skill on VOL-NEUTRAL returns — score vs forward return residualised on vol.
  T3. Regime split                 — OOS IC by calendar year (2022 = drawdown).
  T4. Return overlap               — corr / R^2 of the barrier L/S vs the vol L/S.

If the barrier score keeps meaningful IC/Sharpe under T1 and T2, it has genuine
non-vol alpha. If both collapse, the target is a volatility proxy.

Run: python -u research/exp2_vol_confound.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe
from features import FEATURE_COLS
from evaluation import (
    build_panel, cross_sectional_ic, walk_forward_scores, backtest_long_short,
    cross_sectional_neutralize,
)

TARGET = "tb_highlow"          # the Experiment-1 pick
VOL_FEATURES = ["vol_20", "vol_60"]
NO_VOL_FEATURES = [c for c in FEATURE_COLS if c not in VOL_FEATURES]


def _score(panel, target, feats, Y):
    scored = walk_forward_scores(panel, target_col=target, feature_cols=feats, Y=Y)
    return scored.dropna(subset=["score"]).copy()


def main(Y=20, k=6):
    universe = load_universe()
    k = max(2, min(k, len(universe) // 3))
    panel = build_panel(universe, Y=Y)
    print(f"Universe {len(universe)} stocks, {len(panel)} obs, "
          f"{panel['date'].nunique()} dates, Y={Y}, k={k}, target={TARGET}\n")

    # Vol-neutral forward return (residual of fwd_ret on vol_60, per date).
    panel["fwd_ret_vn"] = cross_sectional_neutralize(
        panel, "fwd_ret", ["vol_60"]).values

    # --- T0: naive volatility factor (rank by vol_60, long-short) -------------
    full = _score(panel, TARGET, FEATURE_COLS, Y)   # also reused below
    vol_factor = full.assign(score=full["vol_60"])
    ic_vol, ir_vol, _, _ = cross_sectional_ic(vol_factor, "score", "fwd_ret")
    m_vol, r_vol = backtest_long_short(vol_factor, "score", "fwd_ret", k, Y, 30.0)
    print("=== T0. Naive vol_60 factor (no model) ===")
    print(f"  IC vs fwd_ret = {ic_vol:.4f} | L/S Sharpe = {m_vol['sharpe']:.2f} "
          f"| L/S ret = {m_vol['mean_ret']:.2%}")

    # --- barrier model, full features (Experiment-1 baseline) -----------------
    ic_full, ir_full, _, _ = cross_sectional_ic(full, "score", "fwd_ret")
    m_full, r_full = backtest_long_short(full, "score", "fwd_ret", k, Y, 30.0)
    ic_full_vn, _, _, _ = cross_sectional_ic(full, "score", "fwd_ret_vn")
    print(f"\n=== Barrier model ({TARGET}), FULL features ===")
    print(f"  IC vs fwd_ret       = {ic_full:.4f} | L/S Sharpe = {m_full['sharpe']:.2f}")
    print(f"  IC vs VOL-NEUTRAL   = {ic_full_vn:.4f}   (T2: skill beyond vol)")

    # --- T1: retrain WITHOUT vol features -------------------------------------
    novol = _score(panel, TARGET, NO_VOL_FEATURES, Y)
    ic_nv, _, _, _ = cross_sectional_ic(novol, "score", "fwd_ret")
    m_nv, _ = backtest_long_short(novol, "score", "fwd_ret", k, Y, 30.0)
    ic_nv_vn, _, _, _ = cross_sectional_ic(novol, "score", "fwd_ret_vn")
    print("\n=== T1. Barrier model WITHOUT vol features ===")
    print(f"  IC vs fwd_ret       = {ic_nv:.4f} | L/S Sharpe = {m_nv['sharpe']:.2f}")
    print(f"  IC vs VOL-NEUTRAL   = {ic_nv_vn:.4f}")

    # --- T3: regime split (OOS IC by year) ------------------------------------
    print("\n=== T3. Barrier model (full) OOS IC by year ===")
    full = full.copy()
    full["yr"] = full["date"].dt.year
    for yr, g in full.groupby("yr"):
        ic_y, _, nd, _ = cross_sectional_ic(g, "score", "fwd_ret")
        icv_y, _, _, _ = cross_sectional_ic(g, "score", "fwd_ret_vn")
        print(f"  {yr}: IC={ic_y:6.4f}  vol-neutral IC={icv_y:6.4f}  (dates={nd})")

    # --- T4: how much of the barrier L/S is the vol factor? -------------------
    print("\n=== T4. Barrier L/S returns vs vol-factor L/S returns ===")
    n = min(len(r_full), len(r_vol))
    if n > 5:
        a, b = r_full[:n], r_vol[:n]
        corr = np.corrcoef(a, b)[0, 1]
        beta = np.cov(a, b)[0, 1] / (np.var(b) + 1e-12)
        alpha = a.mean() - beta * b.mean()
        ss_res = np.sum((a - (alpha + beta * b)) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        print(f"  corr={corr:.2f}  beta_to_vol={beta:.2f}  R^2={r2:.2f}  "
              f"residual alpha/rebal={alpha:.2%}")

    print("\n--- Verdict inputs ---")
    print(f"  vol factor L/S Sharpe:        {m_vol['sharpe']:.2f}")
    print(f"  barrier full  L/S Sharpe:     {m_full['sharpe']:.2f}  IC {ic_full:.4f}")
    print(f"  barrier no-vol L/S Sharpe:    {m_nv['sharpe']:.2f}  IC {ic_nv:.4f}")
    print(f"  barrier IC on vol-neutral ret:{ic_full_vn:.4f} (full) / "
          f"{ic_nv_vn:.4f} (no-vol)")


if __name__ == "__main__":
    main()
