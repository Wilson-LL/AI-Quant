"""Diagnostics for a transformer OOS score panel: what is the signal made of?

Answers, per brief §7/§13:
  - is the score just momentum / volatility / beta in disguise?
  - does it predict anything after residualizing on those factors?
  - how stable is the IC across years?

Usage: python research/transformer_diagnostics.py [panel_name]
Writes reports/transformer_gpu/DIAG_<panel>.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_hybrid import load_panel, merged  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")


def _xs_corr(df, a, b):
    def one(g):
        g = g[[a, b]].dropna()
        if len(g) < 10:
            return np.nan
        return g[a].rank().corr(g[b].rank())
    s = df.groupby("date", group_keys=False).apply(one, include_groups=False).dropna()
    return float(s.mean()) if len(s) else float("nan")


def _resid_ic(df, ret_col="fwd_h"):
    """IC of score residualized per-date on mom + vol (rank-OLS residual)."""
    def one(g):
        g = g[["score", "mom", "vol_20", ret_col]].dropna()
        if len(g) < 15:
            return np.nan
        X = np.column_stack([np.ones(len(g)), g["mom"].rank(), g["vol_20"].rank()])
        y = g["score"].rank().to_numpy(float)
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ w
        return pd.Series(resid).corr(g[ret_col].rank().reset_index(drop=True))
    s = df.groupby("date", group_keys=False).apply(one, include_groups=False).dropna()
    return (float(s.mean()) if len(s) else float("nan"),
            float(s.mean() / (s.std() + 1e-12)) if len(s) else float("nan"))


def main(panel_name=None):
    panel, pname = load_panel(panel_name)
    m = merged(panel)

    out = {"panel": pname}
    out["corr_score_mom126"] = _xs_corr(m, "score", "mom")
    out["corr_score_vol20"] = _xs_corr(m, "score", "vol_20")
    out["ic_score"] = _xs_corr(m, "score", "fwd_h")
    out["ic_mom"] = _xs_corr(m, "mom", "fwd_h")
    ric, ric_ir = _resid_ic(m)
    out["ic_score_resid_mom_vol"] = ric
    out["ic_resid_ir"] = ric_ir

    m["year"] = m["date"].dt.year
    yearly = {}
    for y, g in m.groupby("year"):
        yearly[int(y)] = {
            "ic_score": _xs_corr(g, "score", "fwd_h"),
            "ic_mom": _xs_corr(g, "mom", "fwd_h"),
            "corr_score_mom": _xs_corr(g, "score", "mom"),
        }
    out["yearly"] = yearly

    # score autocorrelation day-to-day (book stability driver)
    piv = m.pivot_table(index="date", columns="stock", values="score")
    ac = piv.corrwith(piv.shift(1), axis=1).dropna()
    out["score_daily_rank_autocorr"] = float(ac.mean())

    path = os.path.join(REPORT_DIR, f"DIAG_{pname}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
