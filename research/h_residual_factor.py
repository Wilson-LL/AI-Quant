"""v17 H-RESIDUAL-SIGNAL and H-FACTOR-PREMIUM — zero-GPU burned mechanism
diagnostics. Implements reports/model_audit/v17/h_residual_factor_spec.json
(preregistered before any result was computed).

  python research/h_residual_factor.py residual   # H-RESIDUAL-SIGNAL + feature contribution
  python research/h_residual_factor.py factor     # H-FACTOR-PREMIUM + long-training attribution

Both stages refuse to run unless the spec is committed and unmodified. No
model is trained; the only fitted objects are the linear benchmarks. No GPU.
Production code and outputs are read, never written.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V17 = os.path.join(ROOT, "reports", "model_audit", "v17")
P0 = os.path.join(V17, "p0")
SPEC_P = os.path.join(V17, "h_residual_factor_spec.json")
OUT_DIR = os.path.join(V17, "h_rf")                      # untracked per-date artifacts
RES_JSON = os.path.join(V17, "h_residual_signal_result.json")
FAC_JSON = os.path.join(V17, "h_factor_premium_result.json")
FEATURES = ["log_ret_1", "mom_5", "mom_20", "mom_60", "mom_126_5",
            "vol_20", "vol_60", "dist_hi_60", "dist_lo_60", "px_over_ma20"]
H, LAG = 20, 1
MATURE = H + LAG                                          # label end = t + 21 sessions
RIDGE_ALPHA = 1.0
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


# ------------------------------------------------------------ statistics

def tcrit(df):
    df = int(max(1, df))
    return T975[df - 1] if df <= 30 else 1.96 + 2.4 / df


def hac_se(x, lag):
    """Newey-West SE of the mean of a time-ordered series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    d = x - x.mean()
    v = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return float(np.sqrt(max(v, 0.0) / n))


def summarize(series, lag, df=None):
    """series: pd.Series of block (or date) values in time order."""
    x = series.dropna()
    n = len(x)
    m = float(x.mean())
    se = hac_se(x.to_numpy(), lag)
    naive = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tc = tcrit(n - 1 if df is None else df)
    loro = [float(x.drop(k).mean()) for k in x.index] if n > 2 else []
    return {"mean": m, "hac_se": se, "naive_se": naive, "ci95": [m - tc * se, m + tc * se], "t_df": (n - 1 if df is None else df),
            "n": n, "share_positive": float((x > 0).mean()),
            "loro_min": min(loro) if loro else None, "loro_max": max(loro) if loro else None}


_INV = np.vectorize(__import__("statistics").NormalDist().inv_cdf)


def rankz(a):
    """Cross-sectional rank-Gaussian (normal) scores, standardised to mean 0 /
    SD 1 (ddof 0). Normal scores keep a linear mix of Gaussian-like inputs
    close to linear after the transform; plain uniform ranks do not (a purely
    linear score then leaves a spurious residual IC of ~0.026 in the
    synthetic calibration test)."""
    x = np.asarray(a, float)
    r = pd.Series(x).rank().to_numpy()
    q = _INV((r - 0.5) / len(r))
    s = q.std()
    return (q - q.mean()) / s if s > 0 else q * 0.0


def zwin(a, p=0.01):
    """Raw values winsorised at 1% / 99% and z-scored (used only by the placebo)."""
    x = np.asarray(a, float)
    lo, hi = np.quantile(x, [p, 1 - p])
    x = np.clip(x, lo, hi)
    return (x - x.mean()) / (x.std() + 1e-12)


def spearman(a, b):
    return float(np.corrcoef(rankz(a), rankz(b))[0, 1])


def residualize(s, X):
    """OLS of s on [1, X] across names of one date; returns (fitted, residual)."""
    X = np.column_stack([np.ones(len(s)), np.asarray(X, float).reshape(len(s), -1)])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    fit = X @ beta
    return fit, s - fit


def spread(score, fwd, q=0.2):
    n = len(score)
    k = max(1, int(round(q * n)))
    o = np.argsort(score)
    return float(np.mean(fwd[o[-k:]]) - np.mean(fwd[o[:k]]))


def classify_residual(st):
    lo, hi = st["ci95"]
    if st["mean"] > 0.01 and lo > 0 and st["share_positive"] > 0.5 and (st["loro_min"] or 0) > 0:
        return "RESIDUAL_SIGNAL_PRESENT"
    if hi < 0.01:
        return "RESIDUAL_SIGNAL_ABSENT"
    if st["mean"] > 0 and st["share_positive"] >= 0.5:
        return "RESIDUAL_SIGNAL_WEAK"
    return "RESIDUAL_SIGNAL_INCONCLUSIVE"


def classify_factor(st, share_pos_blocks, half1, half2, prosp_mean):
    lo, hi = st["ci95"]
    m = st["mean"]
    if (m >= 0.02 and lo > 0 and share_pos_blocks >= 2 / 3 and half1 > 0 and half2 > 0
            and (prosp_mean is None or not np.isfinite(prosp_mean) or prosp_mean >= -0.05)):
        return "FACTOR_PREMIUM_ROBUST"
    if hi < 0.01 or (m <= 0 and half1 <= 0 and half2 <= 0):
        return "FACTOR_PREMIUM_ABSENT"
    if ((np.sign(half1) != np.sign(half2) and max(abs(half1), abs(half2)) >= 0.05)
            or (m > 0 and share_pos_blocks < 0.6 and max(half1, half2) >= 0.05)):
        return "FACTOR_PREMIUM_REGIME_DEPENDENT"
    if m > 0:
        return "FACTOR_PREMIUM_WEAK"
    return "INCONCLUSIVE"


# ------------------------------------------------------------ data

def require_frozen_spec():
    from p4_long_eval import git_frozen
    if not git_frozen(SPEC_P):
        raise SystemExit("the preregistration spec must be committed and unmodified before any result is computed")
    with open(SPEC_P) as f:
        return json.load(f)


def wide_features(wide):
    """The 10 close_only inputs on the shared close calendar (formulas of
    dataset_transformer_eod._stock_features close block) plus true 12-1
    momentum. Every value at date t uses closes up to t only."""
    c = wide.astype(float)
    lr = np.log(c / c.shift(1))
    f = {"log_ret_1": lr,
         "mom_5": c / c.shift(5) - 1.0, "mom_20": c / c.shift(20) - 1.0, "mom_60": c / c.shift(60) - 1.0,
         "mom_126_5": c.shift(5) / c.shift(131) - 1.0,
         "vol_20": lr.rolling(20).std(), "vol_60": lr.rolling(60).std(),
         "dist_hi_60": c / c.rolling(60).max() - 1.0, "dist_lo_60": c / c.rolling(60).min() - 1.0,
         "px_over_ma20": c / c.rolling(20).mean() - 1.0,
         "mom_12_1": c.shift(21) / c.shift(252) - 1.0}
    return f


def long_panel():
    import audit_v17_signal as A
    wide = A.close_matrix()
    wide.columns = [int(c) for c in wide.columns]           # all codes are numeric; panels use int stock ids
    f = wide_features(wide)
    cols = {k: v.stack(future_stack=True) for k, v in f.items()}
    fwd = A.fwd_returns(wide, H)
    panel = pd.DataFrame(cols)
    panel["fwd"] = fwd
    panel.index.names = ["date", "stock"]
    return panel, wide.index


def train_mask(date_pos, block_start_pos):
    """Decision dates whose 20-session forward return had matured by the refit
    (label end t + 21 <= refit_rank = block_start - 1)."""
    return np.asarray(date_pos) + MATURE <= block_start_pos - 1


def ridge_training_frame(panel, cal_pos):
    """Rows with all 10 features and a matured forward return; per-date rank-z
    features and the per-date rank target 2*pct - 1. Each date is transformed
    using only its own cross-section, so the frame can be precomputed once and
    masked per refit without leaking any future row."""
    sub = panel.dropna(subset=FEATURES + ["fwd"])
    g = sub.groupby(level=0)
    n = g["fwd"].transform("size")
    sub = sub[n >= 20]
    g = sub.groupby(level=0)
    Z = g[FEATURES].transform(lambda x: rankz(x.to_numpy()))
    Z["y"] = 2 * g["fwd"].rank(pct=True) - 1
    Z["pos"] = cal_pos.reindex(Z.index.get_level_values(0)).to_numpy()
    return Z.dropna()


def fit_ridge(Z, block_start_pos):
    sub = Z[train_mask(Z["pos"].to_numpy(), block_start_pos)]
    X, y = sub[FEATURES].to_numpy(), sub["y"].to_numpy()
    beta = np.linalg.solve(X.T @ X + RIDGE_ALPHA * np.eye(X.shape[1]), X.T @ y)
    return beta, int(len(y)), str(sub.index.get_level_values(0).max())[:10]


def arm_b_blocks():
    """27 arm-B refit blocks as (block_start_date, block_end_date)."""
    p = pd.read_csv(os.path.join(P0, "panel_P0_A_armB.csv.gz"))
    r2d = dict(zip(p["date_rank"], p["date"]))
    out = []
    for f in sorted(glob.glob(os.path.join(P0, "fits", "*_c5_s0_*.npz"))):
        z = np.load(f)
        out.append((r2d[int(z["block_start"])], r2d[int(z["block_end"])]))
    return out


# ------------------------------------------------------------ per-date metrics

def date_metrics(g, ridge_beta):
    """g: one decision date, columns score, fwd, FEATURES, mom_12_1 (complete rows)."""
    s = rankz(g["score"])
    y = g["fwd"].to_numpy()
    yz = rankz(y)
    F = np.column_stack([rankz(g[c]) for c in FEATURES])
    rid = F @ ridge_beta
    m, m12, v60 = rankz(g["mom_126_5"]), rankz(g["mom_12_1"]), rankz(g["vol_60"])
    out = {"n": len(g), "IC_TF": spearman(s, y), "IC_mom_126_5": spearman(m, y), "IC_mom_12_1": spearman(m12, y),
           "IC_ridge10": spearman(rid, y), "IC_vol_60": spearman(v60, y),
           "rho_TF_mom_126_5": spearman(s, m), "rho_TF_mom_12_1": spearman(s, m12), "rho_TF_ridge10": spearman(s, rid),
           "spread_TF": spread(s, y), "spread_mom_126_5": spread(m, y), "spread_ridge10": spread(rid, y)}
    out["dIC_TF_minus_ridge"] = out["IC_TF"] - out["IC_ridge10"]
    sets = {"R10": F, "R_ridge": rankz(rid), "R_mom": m, "R_mom12_1": m12,
            "R10_plus_12_1": np.column_stack([F, m12])}
    for k, X in sets.items():
        fit, res = residualize(s, X)
        out[f"RESIDUAL_IC_{k}"] = spearman(res, y) if res.std() > 1e-12 else 0.0
        out[f"R2_{k}"] = float(1 - res.var() / s.var())
        if k in ("R10", "R_mom"):
            out[f"cov_fit_{k}"] = float(np.mean(fit * yz))
            out[f"cov_res_{k}"] = float(np.mean(res * yz))
        if k == "R10":
            out["spread_R10_residual"] = spread(res, y)
    out["cov_TF"] = float(np.mean(s * yz))
    # PLACEBO: a score that is purely a (raw-scale) function of two inputs.
    # Its R10 residual IC is the method's false-positive floor on real data.
    pl = rankz(zwin(g["mom_126_5"]) + zwin(g["vol_60"]))
    _, rp = residualize(pl, F)
    out["RESIDUAL_IC_PLACEBO"] = spearman(rp, y) if rp.std() > 1e-12 else 0.0
    return out


def feature_contribution(g):
    s = rankz(g["score"])
    F = np.column_stack([rankz(g[c]) for c in FEATURES])
    fit, res = residualize(s, F)
    full_r2 = 1 - res.var() / s.var()
    X1 = np.column_stack([np.ones(len(s)), F])
    beta, *_ = np.linalg.lstsq(X1, s, rcond=None)
    rows = {}
    for j, c in enumerate(FEATURES):
        _, r_drop = residualize(s, np.delete(F, j, axis=1))
        rows[c] = {"rho": spearman(s, F[:, j]), "uni_r2": spearman(s, F[:, j]) ** 2,
                   "drop_one_dR2": float(full_r2 - (1 - r_drop.var() / s.var())), "std_coef": float(beta[j + 1])}
    return rows


def evaluate_arm(scores, panel, blocks, cal_pos, betas):
    """scores: DataFrame date(str), stock(int), score. Returns per-date frame."""
    rows, contrib = [], []
    for (b0, b1), beta in zip(blocks, betas):
        for d in [d for d in sorted(scores["date"].unique()) if b0 <= d <= b1]:
            sc = scores[scores["date"] == d].set_index("stock")["score"]
            g = panel.xs(pd.Timestamp(d), level=0).join(sc, how="inner").dropna(subset=FEATURES + ["mom_12_1", "fwd", "score"])
            if len(g) < 30:
                continue
            rows.append({"date": d, "block": b0, **date_metrics(g, beta)})
            contrib.append(feature_contribution(g))
    return pd.DataFrame(rows), contrib


def block_summary(df, cols, lag=4):
    B = df.groupby("block")[cols].mean().sort_index()
    return {c: summarize(B[c], lag) for c in cols}, B


# ------------------------------------------------------------ stage: residual

MAIN_COLS = ["IC_TF", "IC_mom_126_5", "IC_mom_12_1", "IC_ridge10", "IC_vol_60",
             "RESIDUAL_IC_R10", "RESIDUAL_IC_R_ridge", "RESIDUAL_IC_R_mom", "RESIDUAL_IC_R_mom12_1",
             "RESIDUAL_IC_R10_plus_12_1", "RESIDUAL_IC_PLACEBO", "dIC_TF_minus_ridge", "rho_TF_mom_126_5", "rho_TF_mom_12_1",
             "rho_TF_ridge10", "R2_R10", "R2_R_mom", "R2_R_ridge", "spread_TF", "spread_R10_residual",
             "spread_mom_126_5", "spread_ridge10"]


def stage_residual():
    require_frozen_spec()
    os.makedirs(OUT_DIR, exist_ok=True)
    panel, cal = long_panel()
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    blocks = arm_b_blocks()
    Z = ridge_training_frame(panel, cal_pos)
    betas, ridge_meta = [], []
    for b0, _ in blocks:
        beta, n, last = fit_ridge(Z, int(cal_pos[pd.Timestamp(b0)]))
        betas.append(beta)
        ridge_meta.append({"block_start": b0, "n_train_rows": n, "last_train_date": last,
                           "coef": dict(zip(FEATURES, np.round(beta, 5).tolist()))})
        assert pd.Timestamp(last) < pd.Timestamp(b0)
    out = {"stage": "H-RESIDUAL-SIGNAL", "label": "BURNED MECHANISM DIAGNOSTIC", "ridge": ridge_meta, "arms": {}}
    for arm in ("B", "C", "A"):
        sc = pd.read_csv(os.path.join(P0, f"panel_P0_A_arm{arm}.csv.gz"))[["date", "stock", "score"]]
        df, contrib = evaluate_arm(sc, panel, blocks, cal_pos, betas)
        df.to_csv(os.path.join(OUT_DIR, f"residual_per_date_arm{arm}.csv"), index=False)
        stats, B = block_summary(df, MAIN_COLS)
        B.to_csv(os.path.join(OUT_DIR, f"residual_per_block_arm{arm}.csv"))
        share = {k: float(df[f"cov_fit_{k}"].sum() / df["cov_TF"].sum()) for k in ("R10", "R_mom")}
        fc = {c: {k: float(np.mean([r[c][k] for r in contrib])) for k in ("rho", "uni_r2", "drop_one_dR2", "std_coef")}
              for c in FEATURES}
        out["arms"][arm] = {"n_dates": int(len(df)), "n_blocks": int(df["block"].nunique()),
                            "stats": stats, "factor_share": share,
                            "classification_rule_applied": {k: classify_residual(stats[k]) for k in
                                                            ("RESIDUAL_IC_R10", "RESIDUAL_IC_R_ridge",
                                                             "RESIDUAL_IC_R_mom", "RESIDUAL_IC_R_mom12_1")},
                            "feature_contribution": fc}
    out["H_RESIDUAL_SIGNAL"] = out["arms"]["B"]["classification_rule_applied"]["RESIDUAL_IC_R10"]
    plc = out["arms"]["B"]["stats"]["RESIDUAL_IC_PLACEBO"]["mean"]
    out["placebo_floor_arm_B"] = plc
    out["METHOD_SENSITIVE"] = bool(abs(plc) >= 0.01)
    if (out["METHOD_SENSITIVE"] and out["H_RESIDUAL_SIGNAL"] == "RESIDUAL_SIGNAL_PRESENT"
            and out["arms"]["B"]["stats"]["RESIDUAL_IC_R10"]["mean"] - plc <= 0.01):
        # PRESENT disallowed (loro_min forced negative); ABSENT / WEAK / INCONCLUSIVE rules re-applied
        lab = classify_residual({**out["arms"]["B"]["stats"]["RESIDUAL_IC_R10"], "loro_min": -1.0})
        out["H_RESIDUAL_SIGNAL"] = lab
        out["placebo_rule_applied"] = "PRESENT did not exceed the placebo floor by 0.01; re-classified without PRESENT"
    # prospective (production 7-seed predictions; matured dates only; descriptive)
    pro = []
    last_beta = betas[-1]
    for p in sorted(glob.glob(os.path.join(ROOT, "reports", "transformer_gpu", "*_predictions.csv"))):
        d = os.path.basename(p)[:10]
        if d < "2026-07-24":
            continue
        sc = pd.read_csv(p)[["stock", "score"]]
        sc["stock"] = sc["stock"].astype(int)
        try:
            g = panel.xs(pd.Timestamp(d), level=0).join(sc.set_index("stock")["score"], how="inner")
        except KeyError:
            continue
        g = g.dropna(subset=FEATURES + ["mom_12_1", "fwd", "score"])
        if len(g) >= 30:
            pro.append({"date": d, **date_metrics(g, last_beta)})
    pro = pd.DataFrame(pro)
    if len(pro):
        pro.to_csv(os.path.join(OUT_DIR, "residual_prospective.csv"), index=False)
    out["prospective_descriptive"] = {"n_dates": int(len(pro)),
                                      "dates": [pro["date"].min(), pro["date"].max()] if len(pro) else None,
                                      "means": {c: float(pro[c].mean()) for c in MAIN_COLS} if len(pro) else None,
                                      "note": "production 7-seed scores; ridge = last arm-B refit; overlapping labels (~1 independent window); descriptive only"}
    with open(RES_JSON, "w") as f:
        json.dump(out, f, indent=1, default=float)
    st = out["arms"]["B"]["stats"]
    print("H_RESIDUAL_SIGNAL:", out["H_RESIDUAL_SIGNAL"])
    for c in MAIN_COLS:
        print(f"{c:28s} mean {st[c]['mean']:+.4f}  HAC {st[c]['hac_se']:.4f}  CI [{st[c]['ci95'][0]:+.4f},{st[c]['ci95'][1]:+.4f}]  pos {st[c]['share_positive']:.2f}")
    return out


# ------------------------------------------------------------ stage: factor premium + attribution

def factor_daily(panel, dates):
    rows = []
    for d in dates:
        try:
            g = panel.xs(pd.Timestamp(d), level=0)
        except KeyError:
            continue
        g = g.dropna(subset=["fwd"])
        if len(g) < 30:
            continue
        r = {"date": str(d)[:10]}
        for c in FEATURES + ["mom_12_1"]:
            h = g.dropna(subset=[c])
            r[f"IC_{c}"] = spearman(h[c], h["fwd"]) if len(h) >= 30 else np.nan
            r[f"SPR_{c}"] = spread(rankz(h[c]), h["fwd"].to_numpy()) if len(h) >= 30 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def perm_corr(x, y, n=20000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n)])
    return {"r": r, "p_one_sided_positive": float((null >= r).mean()), "n": int(len(x))}


def stage_factor():
    spec = require_frozen_spec()
    os.makedirs(OUT_DIR, exist_ok=True)
    panel, cal = long_panel()
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    blocks = arm_b_blocks()
    prim_dates = [d for d in cal if pd.Timestamp("2026-01-05") <= d <= pd.Timestamp("2026-07-23")]
    last_mature = cal[len(cal) - 1 - MATURE]
    pro_dates = [d for d in cal if pd.Timestamp("2026-07-24") <= d <= last_mature]
    D = factor_daily(panel, prim_dates)
    P = factor_daily(panel, pro_dates)
    D.to_csv(os.path.join(OUT_DIR, "factor_daily_primary.csv"), index=False)
    P.to_csv(os.path.join(OUT_DIR, "factor_daily_prospective.csv"), index=False)
    bmap = {}
    for b0, b1 in blocks:
        for d in D["date"]:
            if b0 <= d <= b1:
                bmap[d] = b0
    D["block"] = D["date"].map(bmap)
    median = sorted(D["date"])[len(D) // 2]
    b0_dates = spec_b0_blocks()
    df_hac = max(1, len(D) // MATURE - 1)
    res = {"stage": "H-FACTOR-PREMIUM", "label": "BURNED MECHANISM DIAGNOSTIC",
           "primary_dates": [D["date"].min(), D["date"].max(), int(len(D))],
           "prospective_dates": [P["date"].min() if len(P) else None, P["date"].max() if len(P) else None, int(len(P))],
           "half_split_date": median, "factors": {}}
    for c in ["mom_126_5", "mom_12_1", "vol_60"] + [f for f in FEATURES if f not in ("mom_126_5", "vol_60")]:
        ic = D.set_index("date")[f"IC_{c}"]
        st = summarize(ic, lag=H, df=df_hac)
        blk = D.groupby("block")[f"IC_{c}"].mean()
        h1 = float(ic[ic.index < median].mean())
        h2 = float(ic[ic.index >= median].mean())
        pm = float(P[f"IC_{c}"].mean()) if len(P) else None
        b0 = {k: float(ic[[d for d in ic.index if v[0] <= d <= v[1]]].mean()) for k, v in b0_dates.items()}
        res["factors"][c] = {
            "daily": st, "positive_daily_share": float((ic > 0).mean()),
            "blocks_positive_share": float((blk > 0).mean()), "n_blocks": int(len(blk)),
            "half1_mean": h1, "half2_mean": h2,
            "spread_top_minus_bottom_mean": float(D[f"SPR_{c}"].mean()),
            "by_month": {k: float(v) for k, v in ic.groupby(ic.index.str[:7]).mean().items()},
            "by_p4b0_block": b0,
            "prospective": {"mean_ic": pm, "positive_share": float((P[f"IC_{c}"] > 0).mean()) if len(P) else None,
                            "spread": float(P[f"SPR_{c}"].mean()) if len(P) else None},
            "classification": (classify_factor(st, float((blk > 0).mean()), h1, h2, pm)
                               if c in ("mom_126_5", "mom_12_1", "vol_60") else "descriptive only")}
    # context: calendar-year IC 2016-2025 (descriptive)
    ctx_dates = [d for d in cal if pd.Timestamp("2016-01-01") <= d <= pd.Timestamp("2025-12-31")][::5]
    C = factor_daily(panel, ctx_dates)
    res["context_by_year_every_5th_date"] = {c: {k: float(v) for k, v in C.groupby(C["date"].str[:4])[f"IC_{c}"].mean().items()}
                                             for c in ("mom_126_5", "mom_12_1", "vol_60")}
    res["attribution"] = long_training_attribution(panel, b0_dates, blocks)
    with open(FAC_JSON, "w") as f:
        json.dump(res, f, indent=1, default=float)
    for c in ("mom_126_5", "mom_12_1", "vol_60"):
        r = res["factors"][c]
        print(f"{c:10s} mean {r['daily']['mean']:+.4f} HAC {r['daily']['hac_se']:.4f} CI [{r['daily']['ci95'][0]:+.3f},{r['daily']['ci95'][1]:+.3f}] "
              f"blocks+ {r['blocks_positive_share']:.2f} halves {r['half1_mean']:+.3f}/{r['half2_mean']:+.3f} prosp {r['prospective']['mean_ic']} -> {r['classification']}")
    print(json.dumps(res["attribution"]["summary"], indent=1, default=float))
    return res


def spec_b0_blocks():
    """The nine P4-B0 refit blocks (from the frozen P4 spec and the B0 records)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(V17, "p4", "b0", "b0_*_s0_*.json"))):
        with open(p) as f:
            r = json.load(f)
        out[r["refit_date"]] = (r["oos_start"], r["oos_end"])
    return out


def long_training_attribution(panel, b0_blocks, blocks):
    from dataset_transformer_eod import build_dataset
    import train_transformer_eod as tte
    import audit_v17_signal as A
    data = build_dataset("close_only", seq_len=tte.PRESETS["B"]["seq_len"], horizons=(H,), verbose=False)
    dates, dr = data["dates"], data["date_rank"]
    stocks = np.asarray(data["stocks"]).astype(int)
    # (a) LONG decomposition, epoch 3 vs mean of 50/75/100
    ldir = os.path.join(V17, "p4", "b0_long")
    rows = []
    for p in sorted(glob.glob(os.path.join(ldir, "long_*_preds.npz"))):
        rd, seed = os.path.basename(p)[5:15], int(os.path.basename(p).split("_s")[1][0])
        with np.load(p) as z:
            op, idx = z["oos_preds"], z["oos_idx"]
        dd = pd.to_datetime(dates[dr[idx]])
        st = stocks[data["stock_idx"][idx]]
        for tag, pred in (("early", op[2]), ("late", np.mean([op[49], op[74], op[99]], axis=0))):
            for d in np.unique(dd):
                msk = np.asarray(dd == d)
                g = panel.xs(pd.Timestamp(d), level=0).reindex(st[msk])
                g["score"] = pred[msk]
                g = g.dropna(subset=FEATURES + ["fwd", "score"])
                if len(g) < 30:
                    continue
                s, yz = rankz(g["score"]), rankz(g["fwd"])
                F = np.column_stack([rankz(g[c]) for c in FEATURES])
                f10, r10 = residualize(s, F)
                fm, rm = residualize(s, rankz(g["mom_126_5"]))
                rows.append({"refit": rd, "seed": seed, "tag": tag, "date": str(d)[:10],
                             "cov_TF": float(np.mean(s * yz)), "cov_fit_R10": float(np.mean(f10 * yz)),
                             "cov_res_R10": float(np.mean(r10 * yz)), "cov_fit_mom": float(np.mean(fm * yz)),
                             "rho_mom": spearman(s, g["mom_126_5"])})
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(OUT_DIR, "long_attribution_per_date.csv"), index=False)
    per_fit = L.groupby(["refit", "seed", "tag"])[["cov_TF", "cov_fit_R10", "cov_res_R10", "cov_fit_mom", "rho_mom"]].mean()
    gap = per_fit.xs("early", level="tag") - per_fit.xs("late", level="tag")
    by_refit = gap.groupby(level="refit").mean()
    shares = (by_refit["cov_fit_R10"] / by_refit["cov_TF"]).to_dict()
    pooled = float(by_refit["cov_fit_R10"].mean() / by_refit["cov_TF"].mean())
    pooled_mom = float(by_refit["cov_fit_mom"].mean() / by_refit["cov_TF"].mean())
    big = [k for k in by_refit.index if abs(by_refit.loc[k, "cov_TF"]) >= 0.02]
    n_big_ok = sum(shares[k] >= 0.5 for k in big)
    if pooled >= 0.5 and n_big_ok >= min(2, len(big)):
        cls = "FACTOR_DEEXPOSURE_EXPLAINS_MOST"
    elif pooled >= 0.25:
        cls = "FACTOR_DEEXPOSURE_PARTIAL"
    else:
        cls = "FACTOR_DEEXPOSURE_NOT_EXPLAINED"
    exp = per_fit.groupby(level=["refit", "tag"])["rho_mom"].mean().unstack().to_dict()
    # (b) B0 cross-refit relationship (9 refits)
    b0c = pd.read_csv(os.path.join(V17, "p4", "b0_epoch_curves.csv"))
    piv = b0c.pivot_table(index=["refit_date", "seed"], columns="epoch", values="oos_ic")
    delta = (piv[3] - piv[15]).groupby(level=0).mean()
    armb = pd.read_csv(os.path.join(P0, "panel_P0_A_armB.csv.gz"))
    pay, expo = {}, {}
    for rd, (o0, o1) in b0_blocks.items():
        ics, rhos = [], []
        for d in [d for d in sorted(armb["date"].unique()) if o0 <= d <= o1]:
            g = panel.xs(pd.Timestamp(d), level=0).join(
                armb[armb["date"] == d].set_index("stock")["score"], how="inner").dropna(subset=["mom_126_5", "fwd", "score"])
            ics.append(spearman(g["mom_126_5"], g["fwd"]))
            rhos.append(spearman(g["score"], g["mom_126_5"]))
        pay[rd], expo[rd] = float(np.mean(ics)), float(np.mean(rhos))
    keys = sorted(delta.index)
    x_prod = [expo[k] * pay[k] for k in keys]
    b0 = {"delta_ep3_minus_ep15": {k: float(delta[k]) for k in keys}, "mom_payoff": pay, "exposure_armB": expo,
          "corr_delta_vs_exposure_x_payoff": perm_corr(x_prod, [delta[k] for k in keys]),
          "corr_delta_vs_payoff": perm_corr([pay[k] for k in keys], [delta[k] for k in keys])}
    return {"LONG": {"gap_by_refit": by_refit.to_dict(), "R10_share_by_refit": shares,
                     "refits_with_nontrivial_gap": big, "mom_exposure_early_late": exp},
            "B0_cross_refit": b0,
            "summary": {"pooled_R10_share": pooled, "pooled_mom_only_share": pooled_mom,
                        "R10_share_by_refit": shares, "classification": cls,
                        "B0_corr_delta_vs_exposure_x_payoff": b0["corr_delta_vs_exposure_x_payoff"],
                        "B0_corr_delta_vs_payoff": b0["corr_delta_vs_payoff"]}}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"residual": stage_residual, "factor": stage_factor}.get(stage, lambda: sys.exit(__doc__))()
