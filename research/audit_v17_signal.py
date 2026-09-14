"""v17 Track-B signal audit (CPU only, read-only on production artifacts).

Reproduces every number in reports/model_audit/v17/ from the FROZEN
champion OOS panels (SCHED_A8_seeds7_full = CH 2023-01..2026-07-23,
SCHED_BEAR_A8_seeds7_full = BR 2021-01..2026-07-23), the EOD cache, and
the dated production prediction artifacts (prospective period).

Sections (all written as CSV + one JSON summary):
  ic_by_horizon      rank IC of blend / tf / mom vs fwd 1..40d
  ic_by_year_quarter IC stability through time
  bucket_returns     forward returns by blend percentile bucket (alpha decay)
  regimes            IC / top-quintile return by objective regime
  sector             sector concentration + within/between-sector IC
  baselines          identical-protocol backtests: blend, tf, mom126_5,
                     mom20, mom60, equal-weight universe, 0050
  bootstrap          Sharpe CIs (per-rebalance net returns)
  construction_sweep top_frac x band x mode x cost (CHALLENGERS ONLY)
  prospective        production predictions 2026-07-07.. vs realized

Nothing here changes production. No GPU. Deterministic.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import SECTOR_MAP  # noqa: E402
from transformer_hybrid import load_panel, merged, _cache_frames  # noqa: E402
from transformer_portfolio import backtest_scores, _metrics  # noqa: E402

OUT = os.path.join(ROOT, "reports", "model_audit", "v17")
PANELS = {"CH": "SCHED_A8_seeds7_full", "BR": "SCHED_BEAR_A8_seeds7_full"}
HORIZONS = (1, 5, 10, 20, 30, 40)
BUCKETS = [(0.00, 0.05, "top5"), (0.05, 0.10, "5-10"), (0.10, 0.20, "10-20"),
           (0.20, 0.30, "20-30"), (0.30, 0.50, "30-50"), (0.50, 1.01, "bot50")]
EXEC_LAG = 1


# ------------------------------------------------------------ helpers

def close_matrix():
    frames = _cache_frames()
    wide = pd.concat({s: df.set_index("date")["close"] for s, df in
                      frames.items()}, axis=1).sort_index()
    return wide


def fwd_returns(wide, h, exec_lag=EXEC_LAG):
    """c[t+lag+h] / c[t+lag] - 1 on the shared calendar (audit convention:
    mirrors dataset_transformer_eod._fwd_ret up to halted-name drift)."""
    c1 = wide.shift(-exec_lag)
    return (wide.shift(-(exec_lag + h)) / c1 - 1.0).stack(
        future_stack=True).rename(f"fwd_{h}")


def spearman_by_date(df, a, b):
    def one(g):
        if len(g) < 5:
            return np.nan
        return g[a].rank().corr(g[b].rank())
    return df.groupby("date").apply(one, include_groups=False)


def ic_stats(s):
    s = s.dropna()
    return {"mean": float(s.mean()), "std": float(s.std()),
            "ir": float(s.mean() / s.std()) if s.std() > 0 else np.nan,
            "pos_frac": float((s > 0).mean()), "n_dates": int(len(s))}


def load_blend(name):
    panel, _ = load_panel(name)
    m = merged(panel)
    m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    m["sector"] = m["stock"].map(SECTOR_MAP).fillna("other")
    return m


def attach_fwd(m, wide):
    out = m
    for h in HORIZONS:
        f = fwd_returns(wide, h).reset_index()
        f.columns = ["date", "stock", f"fwdc_{h}"]
        out = out.merge(f, on=["date", "stock"], how="left")
    return out


# ------------------------------------------------------------ sections

def sec_ic(m, tag):
    rows = []
    for sig in ("blend", "z_tf", "z_mom"):
        for h in HORIZONS:
            s = spearman_by_date(m, sig, f"fwdc_{h}")
            rows.append({"panel": tag, "signal": sig, "horizon": h,
                         **ic_stats(s)})
    return pd.DataFrame(rows)


def sec_ic_time(m, tag):
    rows = []
    s = spearman_by_date(m, "blend", "fwdc_20").rename("ic").reset_index()
    s["year"] = s["date"].dt.year
    s["quarter"] = s["date"].dt.to_period("Q").astype(str)
    for key in ("year", "quarter"):
        for k, g in s.groupby(key):
            rows.append({"panel": tag, "by": key, "period": str(k),
                         **ic_stats(g["ic"])})
    return pd.DataFrame(rows)


def sec_buckets(m, tag):
    m = m.copy()
    m["pct"] = m.groupby("date")["blend"].rank(ascending=False, pct=True)
    rows = []
    for lo, hi, name in BUCKETS:
        sel = m[(m["pct"] > lo) & (m["pct"] <= hi)]
        rec = {"panel": tag, "bucket": name}
        for h in HORIZONS:
            per_date = sel.groupby("date")[f"fwdc_{h}"].mean()
            rec[f"mean_fwd_{h}"] = float(per_date.mean())
            rec[f"n_dates_{h}"] = int(per_date.notna().sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    # top-minus-bottom spread
    top = m[m["pct"] <= 0.20].groupby("date")
    bot = m[m["pct"] > 0.80].groupby("date")
    spread = {"panel": tag, "bucket": "top20_minus_bot20"}
    for h in HORIZONS:
        d = top[f"fwdc_{h}"].mean() - bot[f"fwdc_{h}"].mean()
        spread[f"mean_fwd_{h}"] = float(d.mean())
        spread[f"n_dates_{h}"] = int(d.notna().sum())
    return pd.concat([df, pd.DataFrame([spread])], ignore_index=True)


def market_regimes(wide):
    """Objective, trailing-only regime labels on the equal-weight index."""
    lr = np.log(wide / wide.shift(1))
    idx = lr.mean(axis=1)                       # equal-weight daily log ret
    level = idx.cumsum()
    ma126 = level.rolling(126).mean()
    vol20 = idx.rolling(20).std()
    above_ma60 = (wide > wide.rolling(60).mean()).mean(axis=1)
    reg = pd.DataFrame({"trend": np.where(level > ma126, "BULL", "BEAR"),
                        "vol20": vol20, "breadth": above_ma60}, index=wide.index)
    q = vol20.quantile([1 / 3, 2 / 3])
    reg["vol_regime"] = np.where(vol20 <= q.iloc[0], "LOW_VOL",
                                 np.where(vol20 <= q.iloc[1], "MID_VOL",
                                          "HIGH_VOL"))
    bq = above_ma60.quantile([1 / 3, 2 / 3])
    reg["breadth_regime"] = np.where(above_ma60 <= bq.iloc[0], "NARROW",
                                     np.where(above_ma60 <= bq.iloc[1],
                                              "MID_BREADTH", "BROAD"))
    reg["shock"] = np.where(idx.rolling(5).sum() < -0.05, "SHOCK_5D",
                            "NORMAL")
    reg.index.name = "date"
    return reg.reset_index()


def sec_regimes(m, reg, tag):
    ic = spearman_by_date(m, "blend", "fwdc_20").rename("ic").reset_index()
    m2 = m.copy()
    m2["pct"] = m2.groupby("date")["blend"].rank(ascending=False, pct=True)
    topq = m2[m2["pct"] <= 0.2].groupby("date")["fwdc_20"].mean().rename(
        "top_q_fwd20").reset_index()
    uni = m2.groupby("date")["fwdc_20"].mean().rename("uni_fwd20").reset_index()
    d = ic.merge(topq, on="date").merge(uni, on="date").merge(reg, on="date")
    rows = []
    for key in ("trend", "vol_regime", "breadth_regime", "shock"):
        for k, g in d.groupby(key):
            rows.append({"panel": tag, "regime_type": key, "regime": k,
                         "n_dates": int(len(g)),
                         "ic_mean": float(g["ic"].mean()),
                         "ic_pos_frac": float((g["ic"] > 0).mean()),
                         "top_q_fwd20": float(g["top_q_fwd20"].mean()),
                         "uni_fwd20": float(g["uni_fwd20"].mean()),
                         "top_q_excess": float((g["top_q_fwd20"]
                                                - g["uni_fwd20"]).mean())})
    return pd.DataFrame(rows)


def sec_sector(m, tag):
    m2 = m.copy()
    m2["pct"] = m2.groupby("date")["blend"].rank(ascending=False, pct=True)
    top = m2[m2["pct"] <= 0.2]
    share = top.groupby(["date", "sector"]).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    conc = {"panel": tag, "max_sector_share_mean": float(share.max(axis=1).mean()),
            "max_sector_share_p95": float(share.max(axis=1).quantile(0.95)),
            "semis_elec_share_mean": float(
                share.reindex(columns=["semis", "electronics"],
                              fill_value=0).sum(axis=1).mean())}
    # within-sector (sector-neutral) IC vs raw IC
    m2["blend_sn"] = m2["blend"] - m2.groupby(["date", "sector"])["blend"].transform("mean")
    m2["fwd_sn"] = m2["fwdc_20"] - m2.groupby(["date", "sector"])["fwdc_20"].transform("mean")
    raw = spearman_by_date(m2, "blend", "fwdc_20")
    sn = spearman_by_date(m2, "blend_sn", "fwd_sn")
    # between-sector: sector-mean score vs sector-mean return
    sm = m2.groupby(["date", "sector"])[["blend", "fwdc_20"]].mean().reset_index()
    bs = spearman_by_date(sm, "blend", "fwdc_20")
    conc.update({"ic_raw": ic_stats(raw)["mean"],
                 "ic_sector_neutral": ic_stats(sn)["mean"],
                 "ic_sector_neutral_ir": ic_stats(sn)["ir"],
                 "ic_between_sector": ic_stats(bs)["mean"],
                 "ic_between_sector_pos_frac": ic_stats(bs)["pos_frac"]})
    return pd.DataFrame([conc]), share.mean().rename("mean_share").reset_index().assign(panel=tag)


def sec_baselines(m, wide, tag):
    """Identical protocol (backtest_scores: 20d blocks, top quintile,
    band10, equal weight, cap10) for every candidate score column."""
    m2 = m.copy()
    c = wide
    mom20 = (c / c.shift(20) - 1).stack(future_stack=True).rename("mom20")
    mom60 = (c / c.shift(60) - 1).stack(future_stack=True).rename("mom60")
    for s in (mom20, mom60):
        f = s.reset_index()
        f.columns = ["date", "stock", s.name]
        m2 = m2.merge(f, on=["date", "stock"], how="left")
    rows = []
    for sig in ("blend", "score", "mom", "mom20", "mom60"):
        p = m2[["date", "stock", sig, "fwd_h", "vol_20"]].rename(
            columns={sig: "score"}).dropna(subset=["score"])
        for mode in ("long_only", "long_short"):
            r = backtest_scores(p, mode=mode, no_trade_band=0.10,
                                cost_bps_list=(0, 60, 100, 150))
            rec = {"panel": tag, "signal": sig, "mode": mode,
                   "rank_ic": r["rank_ic"], "turnover": r["avg_turnover"],
                   "n": r["net60"]["n"]}
            for cb in (0, 60, 100, 150):
                rec[f"sharpe_net{cb}"] = r[f"net{cb}"]["sharpe"]
                rec[f"ann_net{cb}"] = r[f"net{cb}"]["ann_ret"]
                rec[f"dd_net{cb}"] = r[f"net{cb}"]["max_dd"]
            rows.append(rec)
    # equal-weight universe and 0050 on the same rebalance blocks
    dates = np.array(sorted(m2["date"].unique()))
    rebal = dates[::20]
    ew = m2.groupby("date")["fwd_h"].mean()
    ew_blocks = ew.reindex(rebal).dropna()
    rec = {"panel": tag, "signal": "equal_weight_universe", "mode": "long_only",
           "n": int(len(ew_blocks))}
    for cb in (0, 60, 100, 150):
        mt = _metrics(ew_blocks.to_numpy(), 20)
        rec[f"sharpe_net{cb}"] = mt["sharpe"]
        rec[f"ann_net{cb}"] = mt["ann_ret"]
        rec[f"dd_net{cb}"] = mt["max_dd"]
    rows.append(rec)
    p50 = os.path.join(ROOT, "research", "data_cache", "0050.csv")
    if os.path.isfile(p50):
        s = pd.read_csv(p50, parse_dates=["date"]).drop_duplicates(
            "date", keep="last").set_index("date")["close"].sort_index()
        f = (s.shift(-21) / s.shift(-1) - 1).reindex(rebal).dropna()
        mt = _metrics(f.to_numpy(), 20)
        rows.append({"panel": tag, "signal": "0050_buy_hold_blocks",
                     "mode": "long_only", "n": int(len(f)),
                     **{f"sharpe_net{cb}": mt["sharpe"] for cb in (0, 60, 100, 150)},
                     **{f"ann_net{cb}": mt["ann_ret"] for cb in (0, 60, 100, 150)},
                     **{f"dd_net{cb}": mt["max_dd"] for cb in (0, 60, 100, 150)}})
    return pd.DataFrame(rows)


def block_returns(m, mode, band=0.10, quintile=0.2):
    p = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(
        columns={"blend": "score"})
    r = backtest_scores(p, mode=mode, no_trade_band=band, quintile=quintile,
                        cost_bps_list=(60,))
    return r


def sec_bootstrap(m, tag, n_boot=2000, seed=17):
    """Bootstrap the per-rebalance net60 return series to CI the Sharpe.
    Blocks are non-overlapping 20-session periods -> iid resampling."""
    rng = np.random.default_rng(seed)
    rows = []
    for mode in ("long_only", "long_short"):
        p = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(
            columns={"blend": "score"})
        r = backtest_scores(p, mode=mode, no_trade_band=0.10,
                            cost_bps_list=(60,))
        n_legs = 2 if mode == "long_short" else 1
        net = (np.asarray(r["gross"])
               - n_legs * (60 / 1e4) * np.asarray(r["turnover"]))
        base = _metrics(net, 20)["sharpe"]
        bs = np.array([_metrics(rng.choice(net, len(net), replace=True), 20)["sharpe"]
                       for _ in range(n_boot)])
        rows.append({"panel": tag, "mode": mode, "n_blocks": int(len(net)),
                     "sharpe": base, "p5": float(np.percentile(bs, 5)),
                     "p50": float(np.percentile(bs, 50)),
                     "p95": float(np.percentile(bs, 95)),
                     "pos_frac": float((bs > 0).mean()),
                     "analytic_se": float(np.sqrt((1 + (base / np.sqrt(252 / 20)) ** 2 / 2) / len(net)) * np.sqrt(252 / 20))})
    return pd.DataFrame(rows)


def sec_construction(m, tag):
    rows = []
    p = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(
        columns={"blend": "score"})
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        for band in (0.0, 0.10, 0.15, 0.20):
            for mode in ("long_only", "long_short"):
                r = backtest_scores(p, mode=mode, quintile=q,
                                    no_trade_band=band,
                                    cost_bps_list=(60, 100))
                rows.append({"panel": tag, "top_frac": q, "band": band,
                             "mode": mode, "turnover": r["avg_turnover"],
                             "sharpe_net60": r["net60"]["sharpe"],
                             "dd_net60": r["net60"]["max_dd"],
                             "sharpe_net100": r["net100"]["sharpe"],
                             "n": r["net60"]["n"]})
    return pd.DataFrame(rows)


def sec_prospective(wide):
    """Production predictions (dated artifacts) vs realized returns."""
    pdir = os.path.join(ROOT, "reports", "transformer_gpu")
    files = sorted(f for f in os.listdir(pdir) if f.endswith("_predictions.csv"))
    rows = []
    mom = (wide.shift(5) / wide.shift(131) - 1)
    for f in files:
        d = pd.Timestamp(f[:10])
        if d not in wide.index:
            continue
        pr = pd.read_csv(os.path.join(pdir, f), dtype={"stock": str})
        pr["date"] = d
        pr["mom"] = pr["stock"].map(mom.loc[d])
        pr = pr.dropna(subset=["mom"])
        pr["z_tf"] = (pr["score"] - pr["score"].mean()) / (pr["score"].std() + 1e-9)
        pr["z_mom"] = (pr["mom"] - pr["mom"].mean()) / (pr["mom"].std() + 1e-9)
        pr["blend"] = 0.5 * pr["z_tf"] + 0.5 * pr["z_mom"]
        pr["pct"] = pr["blend"].rank(ascending=False, pct=True)
        rec = {"date": d.date().isoformat(), "n": int(len(pr))}
        for h in (1, 5, 10, 20):
            fr = fwd_returns(wide, h).loc[d] if d in wide.index else None
            pr[f"fwd_{h}"] = pr["stock"].map(fr)
            if pr[f"fwd_{h}"].notna().sum() < 30:
                rec[f"ic_{h}"] = np.nan
                rec[f"topq_{h}"] = np.nan
                rec[f"uni_{h}"] = np.nan
                continue
            rec[f"ic_{h}"] = float(pr["blend"].rank().corr(pr[f"fwd_{h}"].rank()))
            rec[f"ic_tf_{h}"] = float(pr["score"].rank().corr(pr[f"fwd_{h}"].rank()))
            rec[f"topq_{h}"] = float(pr[pr["pct"] <= 0.2][f"fwd_{h}"].mean())
            rec[f"uni_{h}"] = float(pr[f"fwd_{h}"].mean())
            rec[f"botq_{h}"] = float(pr[pr["pct"] > 0.8][f"fwd_{h}"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ main

def main():
    os.makedirs(OUT, exist_ok=True)
    wide = close_matrix()
    reg = market_regimes(wide)
    summary = {"cache_last_date": str(wide.index.max().date()),
               "n_stocks": int(wide.shape[1])}
    parts = {k: [] for k in ("ic", "ic_time", "buckets", "regimes", "sector",
                             "sector_share", "baselines", "bootstrap",
                             "construction")}
    for tag, name in PANELS.items():
        m = attach_fwd(load_blend(name), wide)
        summary[f"{tag}_dates"] = [str(m["date"].min().date()),
                                   str(m["date"].max().date())]
        parts["ic"].append(sec_ic(m, tag))
        parts["ic_time"].append(sec_ic_time(m, tag))
        parts["buckets"].append(sec_buckets(m, tag))
        parts["regimes"].append(sec_regimes(m, reg, tag))
        conc, share = sec_sector(m, tag)
        parts["sector"].append(conc)
        parts["sector_share"].append(share)
        parts["baselines"].append(sec_baselines(m, wide, tag))
        parts["bootstrap"].append(sec_bootstrap(m, tag))
        parts["construction"].append(sec_construction(m, tag))
        print(f"[{tag}] done", flush=True)
    for k, lst in parts.items():
        pd.concat(lst, ignore_index=True).to_csv(
            os.path.join(OUT, f"{k}.csv"), index=False)
    pro = sec_prospective(wide)
    pro.to_csv(os.path.join(OUT, "prospective.csv"), index=False)
    reg.to_csv(os.path.join(OUT, "regime_labels.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("written to", OUT)


if __name__ == "__main__":
    main()
