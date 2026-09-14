"""v17 P5 — simple-model baselines under the champion's exact protocol.

Features are cross-sectionally standardized per date (pooled raw-feature
ridge was tried first and is not a fair linear baseline: IC ~0.006).

Ridge regression (closed form, numpy — no new dependency) on the SAME 10
close_only features at the decision date (last timestep), the SAME target
(tgt_rank_20), the SAME chronological walk-forward (refit every 126
sessions, matured labels only, 21-session purge, val 10% unused by ridge
except for lambda choice on the FIRST refit only), the SAME OOS windows
(2021-01 / 2023-01 .. 2026-07-23) and the SAME portfolio protocol.
GBM is NOT run: sklearn/lightgbm are not installed in the production venv
and installing them is out of scope (dependency safety).

Outputs: reports/model_audit/v17/baseline_ridge_panel.csv.gz and
baseline_ridge.csv (standalone + blend results, correlations with the
transformer and momentum).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import dataset_transformer_eod as dte  # noqa: E402
from transformer_hybrid import _cache_frames, load_panel, merged, _z  # noqa: E402
from transformer_portfolio import backtest_scores  # noqa: E402

OUT = os.path.join(ROOT, "reports", "model_audit", "v17")
COLS = list(dte.FEATURE_COLS["close_only"])
REFIT = 126
PURGE = 21
LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def feature_panel():
    rows = []
    for sid, df in _cache_frames().items():
        f = dte._stock_features(df, "close_only")[COLS].clip(-10, 10)
        c = df["close"].to_numpy(np.float64)
        f["fwd_20"] = dte._fwd_ret(c, 20, 1)
        f["date"] = df["date"].to_numpy()
        f["stock"] = sid
        rows.append(f)
    p = pd.concat(rows, ignore_index=True).dropna(subset=COLS)
    # cross-sectional standardization per date (the fair linear baseline:
    # the target is a per-date rank, so features are compared within date)
    g = p.groupby("date")
    for c in COLS:
        p[c] = (p[c] - g[c].transform("mean")) / (g[c].transform("std") + 1e-9)
    p["tgt"] = 2.0 * (p.groupby("date")["fwd_20"].rank(pct=True) - 0.5)
    dates = np.array(sorted(p["date"].unique()))
    rank_of = {d: i for i, d in enumerate(dates)}
    p["dr"] = p["date"].map(rank_of).astype(int)
    p["label_end"] = p["dr"] + 21
    return p, dates


def ridge_fit(X, y, lam):
    Xm, ym = X.mean(0), y.mean()
    Xc, yc = X - Xm, y - ym
    sd = Xc.std(0) + 1e-12
    Xc = Xc / sd
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    return w, Xm, sd, ym


def ridge_pred(X, fit):
    w, Xm, sd, ym = fit
    return ((X - Xm) / sd) @ w + ym


def walkforward(p, dates, oos_start):
    oos0 = int(np.searchsorted(dates, np.datetime64(oos_start)))
    last = len(dates) - 1
    out = []
    lam = None
    for r0 in range(oos0, last + 1, REFIT):
        refit = r0 - 1
        end = min(r0 + REFIT - 1, last)
        usable = (p["label_end"] <= refit) & p["tgt"].notna()
        tr = p[usable]
        if lam is None:              # choose lambda ONCE on the first refit's val tail
            v0 = int(np.percentile(tr["dr"].unique(), 90))
            trn, val = tr[tr["dr"] <= v0 - PURGE], tr[tr["dr"] > v0]
            best = None
            for L in LAMBDAS:
                fit = ridge_fit(trn[COLS].to_numpy(), trn["tgt"].to_numpy(), L)
                s = ridge_pred(val[COLS].to_numpy(), fit)
                ic = pd.DataFrame({"d": val["dr"].to_numpy(), "s": s,
                                   "y": val["tgt"].to_numpy()}).groupby("d").apply(
                    lambda g: g["s"].rank().corr(g["y"].rank()), include_groups=False).mean()
                if best is None or ic > best[0]:
                    best = (ic, L)
            lam = best[1]
        fit = ridge_fit(tr[COLS].to_numpy(), tr["tgt"].to_numpy(), lam)
        blk = p[(p["dr"] >= r0) & (p["dr"] <= end)].copy()
        blk["score"] = ridge_pred(blk[COLS].to_numpy(), fit)
        out.append(blk[["date", "stock", "score", "fwd_20"]])
    res = pd.concat(out, ignore_index=True)
    res["fwd_h"] = res["fwd_20"]
    return res, lam


def evaluate(tag, panel_name, ridge_panel):
    tfp, _ = load_panel(panel_name)
    m = merged(tfp)                                   # z_tf, z_mom, mom, vol_20
    m = m.merge(ridge_panel[["date", "stock", "score"]].rename(
        columns={"score": "ridge"}), on=["date", "stock"], how="inner")
    m["z_ridge"] = m.groupby("date")["ridge"].transform(_z)
    m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    m["blend_ridge"] = 0.5 * m["z_ridge"] + 0.5 * m["z_mom"]
    m["blend3"] = (m["z_tf"] + m["z_mom"] + m["z_ridge"]) / 3
    rows, corr = [], {}
    for sig in ("z_tf", "z_ridge", "z_mom", "blend", "blend_ridge", "blend3"):
        pp = m[["date", "stock", sig, "fwd_h", "vol_20"]].rename(
            columns={sig: "score"}).dropna(subset=["score", "fwd_h"])
        r = backtest_scores(pp, mode="long_only", no_trade_band=0.10,
                            cost_bps_list=(60, 100))
        rows.append({"panel": tag, "signal": sig, "rank_ic": r["rank_ic"],
                     "turnover": r["avg_turnover"], "n": r["net60"]["n"],
                     "sharpe_net60": r["net60"]["sharpe"],
                     "sharpe_net100": r["net100"]["sharpe"],
                     "ann_net60": r["net60"]["ann_ret"],
                     "dd_net60": r["net60"]["max_dd"]})
    # per-date rank correlations between signals (prediction correlation)
    def rc(a, b):
        return m.groupby("date").apply(
            lambda g: g[a].rank().corr(g[b].rank()), include_groups=False).mean()
    corr = {"panel": tag, "corr_tf_ridge": rc("z_tf", "z_ridge"),
            "corr_tf_mom": rc("z_tf", "z_mom"), "corr_ridge_mom": rc("z_ridge", "z_mom")}
    # incremental IC: tf residual after ridge+mom (per-date OLS residual)
    def resid_ic(g):
        X = np.column_stack([np.ones(len(g)), g["z_ridge"], g["z_mom"]])
        beta, *_ = np.linalg.lstsq(X, g["z_tf"], rcond=None)
        res = g["z_tf"] - X @ beta
        return pd.Series(res).rank().corr(g["fwd_h"].rank())
    corr["ic_tf_residual_vs_ridge_mom"] = float(
        m.dropna(subset=["fwd_h"]).groupby("date").apply(resid_ic, include_groups=False).mean())
    return pd.DataFrame(rows), corr


def main():
    os.makedirs(OUT, exist_ok=True)
    p, dates = feature_panel()
    res, lam = walkforward(p, dates, "2021-01-01")
    res.to_csv(os.path.join(OUT, "baseline_ridge_panel.csv.gz"), index=False)
    print("ridge lambda (chosen once, first refit val):", lam)
    rows, corrs = [], []
    for tag, name in (("CH", "SCHED_A8_seeds7_full"),
                      ("BR", "SCHED_BEAR_A8_seeds7_full")):
        r, c = evaluate(tag, name, res)
        rows.append(r)
        corrs.append(c)
    pd.concat(rows).to_csv(os.path.join(OUT, "baseline_ridge.csv"), index=False)
    pd.DataFrame(corrs).to_csv(os.path.join(OUT, "baseline_correlations.csv"), index=False)
    pd.set_option("display.width", 200)
    print(pd.concat(rows).round(3).to_string(index=False))
    print(pd.DataFrame(corrs).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
