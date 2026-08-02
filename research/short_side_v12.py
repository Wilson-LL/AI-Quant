"""v12 Task 7 — short-side / long-short research (CPU, panel-based).

REVIEW-ONLY research: never generates orders; the short candidate list is a
diagnostic, not a trade list. Real borrow/locate data is unavailable — the
squeeze/borrow screen is a clearly-labeled conservative PROXY:
  excluded from the short leg if any of
    - bottom liquidity tercile by adv20 (20d median close*volume) that day
    - 20d return > +30% (rally-squeeze proxy)
    - close < 10 TWD (price floor)
Costs: long leg charged `cost_bps` x one-way turnover per rebalance; short
leg charged 2 x cost_bps (borrow/locate margin) — sensitivity 60/100/150/200.

Evaluates signals tf / d12 / blend50 on the frozen 7-seed panels (CH 2023->,
BR 2021->) in configurations:
  L100 (long-only baseline) . L100_S50 . L100_S100 (net 0 = market-neutral
  construction) . S100 (short-only diagnostic).
Long leg: top quintile, equal weight, cap 10%/sector 20% (production caps).
Short leg: bottom quintile after proxy exclusions, equal weight, name cap
5%, sector cap 20%, rebalanced every 20 trading days (production cadence).

Usage:  python research/short_side_v12.py
Outputs (reports/continuous_research/v12_big_transformer/):
  short_side_metrics.csv . short_side_report.md . short_candidate_diagnostics.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import SECTOR_MAP  # noqa: E402
from transformer_hybrid import _cache_frames  # noqa: E402
from transformer_portfolio import cap_weights  # noqa: E402
from queue_v9_lib import get_merged, adv_frame  # noqa: E402

V12_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v12_big_transformer")
HOLD = 20
ANN = np.sqrt(252 / HOLD)
COSTS = (60, 100, 150, 200)          # bps per unit one-way turnover (long)
SHORT_COST_MULT = 2.0
SIGNALS = {"tf": "z_tf", "d12": "z_mom", "blend50": "blend"}
SQUEEZE_RET20 = 0.30
PRICE_FLOOR = 10.0
SHORT_NAME_CAP = 0.05
SHORT_SECTOR_CAP = 0.20


def _ret20_frame():
    rows = []
    for sid, df in _cache_frames().items():
        c = df["close"].to_numpy(np.float64)
        r = np.full(len(c), np.nan)
        r[20:] = c[20:] / c[:-20] - 1.0
        rows.append(pd.DataFrame({"date": df["date"].values, "stock": sid,
                                  "ret20": r, "close": c}))
    return pd.concat(rows, ignore_index=True)


def _leg(day, names, name_cap, sector_cap):
    sub = day[day["stock"].isin(names)]
    if sub.empty:
        return pd.Series(dtype=float)
    w = cap_weights(np.ones(len(sub)),
                    [SECTOR_MAP.get(s, "other") for s in sub["stock"]],
                    name_cap=name_cap, sector_cap=sector_cap)
    return pd.Series(w, index=sub["stock"].values)


def _turnover(prev, cur):
    if prev is None:
        return 1.0
    u = prev.index.union(cur.index)
    return 0.5 * float((prev.reindex(u, fill_value=0)
                        - cur.reindex(u, fill_value=0)).abs().sum())


def evaluate(win):
    m = get_merged(win).copy()
    m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    aux = _ret20_frame()
    m = m.merge(adv_frame(), on=["date", "stock"], how="left")
    m = m.merge(aux, on=["date", "stock"], how="left")
    dates = sorted(m["date"].unique())
    rebals = dates[::HOLD]

    out = {}
    for sig, col in SIGNALS.items():
        prev_l, prev_s = None, None
        periods = []
        short_rows = []
        for d in rebals:
            day = m[m["date"] == d].dropna(subset=["fwd_20", col])
            if len(day) < 30:
                continue
            day = day.sort_values(col, ascending=False)
            k = max(3, round(0.2 * len(day)))
            longs = list(day.head(k)["stock"])
            bot = day.tail(k).copy()
            # conservative squeeze/borrow PROXY exclusions (see module doc)
            adv_ter = bot["adv20"].quantile(1 / 3)
            excl = ((bot["adv20"] <= adv_ter) | (bot["ret20"] > SQUEEZE_RET20)
                    | (bot["close"] < PRICE_FLOOR))
            shorts = list(bot.loc[~excl, "stock"])
            wl = _leg(day, longs, 0.10, 0.20)
            ws = _leg(day, shorts, SHORT_NAME_CAP, SHORT_SECTOR_CAP)
            fwd = day.set_index("stock")["fwd_20"]
            rl = float((wl * fwd.reindex(wl.index)).sum())
            rs_raw = float((ws * fwd.reindex(ws.index)).sum()) if len(ws) else 0.0
            to_l, to_s = _turnover(prev_l, wl), _turnover(prev_s, ws)
            prev_l, prev_s = wl, ws
            sec_share = (pd.Series([SECTOR_MAP.get(s, "other")
                                    for s in ws.index], index=ws.index)
                         .groupby(lambda i: SECTOR_MAP.get(i, "other"))
                         .count())
            contrib = (-(ws * fwd.reindex(ws.index))).sort_values()
            periods.append({
                "date": d, "rl": rl, "rs_raw": rs_raw, "to_l": to_l,
                "to_s": to_s, "n_long": len(wl), "n_short": len(ws),
                "n_excluded": int(excl.sum()),
                "short_hit": float((fwd.reindex(ws.index) < 0).mean())
                if len(ws) else np.nan,
                "short_mae": float(fwd.reindex(ws.index).max())
                if len(ws) else np.nan,
                "short_max_sector_w": float(
                    ws.groupby([SECTOR_MAP.get(s, "other")
                                for s in ws.index]).sum().max())
                if len(ws) else np.nan})
            for stock, pnl in contrib.head(5).items():   # worst = most negative
                short_rows.append({"window": win, "signal": sig, "date": d,
                                   "stock": stock,
                                   "fwd_20": float(fwd.get(stock, np.nan)),
                                   "short_pnl_contrib": float(pnl)})
        out[sig] = (pd.DataFrame(periods), pd.DataFrame(short_rows))
    return out


def _series_stats(r, dates):
    r = np.asarray(r, float)
    sh = float(r.mean() / (r.std() + 1e-12) * ANN)
    cum = np.cumprod(1 + r)
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    yearly = {}
    ys = pd.Series(r, index=pd.to_datetime(list(dates))).groupby(
        lambda d: d.year)
    for y, g in ys:
        yearly[int(y)] = round(float(g.mean() / (g.std() + 1e-12) * ANN), 2)
    return sh, dd, yearly


def main():
    os.makedirs(V12_DIR, exist_ok=True)
    metric_rows, diag_frames = [], []
    report = ["# v12 short-side / long-short research", "",
              "Panel-based (frozen 7-seed A8 panels), 20d rebalance, "
              "production long caps; short leg: bottom quintile after the "
              "conservative squeeze/borrow PROXY (low-ADV tercile, 20d "
              "return > +30%, price < 10 excluded), name cap 5%, sector cap "
              "20%, costs 2x the long leg. REVIEW-ONLY — no orders.", ""]
    gate_results = {}
    for win in ("CH", "BR"):
        res = evaluate(win)
        report.append(f"## Window {win}\n")
        for sig, (p, worst) in res.items():
            diag_frames.append(worst)
            for cost in COSTS:
                c_l = cost / 1e4
                c_s = SHORT_COST_MULT * c_l
                r_l = p["rl"] - c_l * p["to_l"]
                r_s = -p["rs_raw"] - c_s * p["to_s"]
                configs = {
                    "L100": r_l,
                    "L100_S50": r_l + 0.5 * r_s,
                    "L100_S100_net0": r_l + r_s,
                    "S100_diagnostic": r_s,
                }
                for cfg, r in configs.items():
                    sh, dd, yearly = _series_stats(r, p["date"])
                    metric_rows.append({
                        "window": win, "signal": sig, "config": cfg,
                        "cost_bps": cost, "sharpe": round(sh, 3),
                        "max_dd": round(dd, 4),
                        "ann_ret": round(float(np.mean(r)) * 252 / HOLD, 4),
                        "yearly": json.dumps(yearly),
                        "short_hit_rate": round(float(p["short_hit"].mean()), 3),
                        "short_mae_mean": round(float(p["short_mae"].mean()), 4),
                        "turnover_long": round(float(p["to_l"].mean()), 3),
                        "turnover_short": round(float(p["to_s"].mean()), 3),
                        "avg_n_short": round(float(p["n_short"].mean()), 1),
                        "avg_n_excluded": round(float(p["n_excluded"].mean()), 1),
                        "max_short_sector_w": round(
                            float(p["short_max_sector_w"].max()), 3)})
            # gate check at the 60bps reference level
            c_l, c_s = 0.006, 0.012
            r_l = p["rl"] - c_l * p["to_l"]
            r_s = -p["rs_raw"] - c_s * p["to_s"]
            sh_l, dd_l, _ = _series_stats(r_l, p["date"])
            sh_50, dd_50, y50 = _series_stats(r_l + 0.5 * r_s, p["date"])
            sh_100, dd_100, y100 = _series_stats(r_l + r_s, p["date"])
            sh_s, dd_s, ys = _series_stats(r_s, p["date"])
            gate_results[(win, sig)] = dict(
                L100=sh_l, S50=sh_50, S100=sh_100, short_alone=sh_s,
                dd_L=dd_l, dd_50=dd_50, dd_100=dd_100, dd_S=dd_s, ys=ys)
            report += [
                f"### {sig} (net60 long / net120 short)",
                f"- L100 {sh_l:.2f} (DD {dd_l:.1%}) · L100_S50 {sh_50:.2f} "
                f"(DD {dd_50:.1%}) · L100_S100 {sh_100:.2f} (DD {dd_100:.1%})"
                f" · short leg alone {sh_s:.2f} (DD {dd_s:.1%})",
                f"- short hit rate {p['short_hit'].mean():.1%} · MAE(mean "
                f"worst 20d move among shorts) {p['short_mae'].mean():+.1%} "
                f"· avg shorts held {p['n_short'].mean():.0f} (excluded by "
                f"proxy {p['n_excluded'].mean():.0f}) · short turnover "
                f"{p['to_s'].mean():.2f}",
                f"- short-leg yearly Sharpe: "
                + ", ".join(f"{y} {v}" for y, v in ys.items()), ""]
    df = pd.DataFrame(metric_rows)
    df.to_csv(os.path.join(V12_DIR, "short_side_metrics.csv"), index=False)

    # ---- current short-candidate diagnostics (REVIEW ONLY)
    pred_dir = os.path.join(ROOT, "reports", "transformer_gpu")
    preds = sorted(f for f in os.listdir(pred_dir)
                   if f.endswith("_predictions.csv"))
    diag_p = None
    if preds:
        pr = pd.read_csv(os.path.join(pred_dir, preds[-1]),
                         dtype={"stock": str})
        aux = _ret20_frame()
        latest_aux = aux[aux["date"] == aux["date"].max()]
        adv = adv_frame()
        latest_adv = adv[adv["date"] == adv["date"].max()]
        rows = []
        cache = _cache_frames()
        mom = {sid: c["close"].to_numpy(np.float64)[-6]
               / c["close"].to_numpy(np.float64)[-132] - 1.0
               for sid, c in cache.items() if len(c) >= 132}
        pr = pr[pr["stock"].isin(mom)]
        pr["z_tf"] = (pr["score"] - pr["score"].mean()) / (pr["score"].std() + 1e-9)
        mm = pd.Series(mom).reindex(pr["stock"]).to_numpy()
        pr["z_mom"] = (mm - np.nanmean(mm)) / (np.nanstd(mm) + 1e-9)
        pr["blend"] = 0.5 * pr["z_tf"] + 0.5 * pr["z_mom"]
        bot = pr.nsmallest(max(3, round(0.2 * len(pr))), "blend")
        for _, r in bot.iterrows():
            a = latest_adv[latest_adv["stock"] == r["stock"]]["adv20"]
            x = latest_aux[latest_aux["stock"] == r["stock"]]
            adv_v = float(a.iloc[0]) if len(a) else np.nan
            ret20 = float(x["ret20"].iloc[0]) if len(x) else np.nan
            close = float(x["close"].iloc[0]) if len(x) else np.nan
            flags = []
            if np.isfinite(adv_v) and adv_v <= latest_adv["adv20"].quantile(1/3):
                flags.append("LOW_LIQUIDITY")
            if np.isfinite(ret20) and ret20 > SQUEEZE_RET20:
                flags.append("SQUEEZE_PROXY")
            if np.isfinite(close) and close < PRICE_FLOOR:
                flags.append("PRICE_FLOOR")
            rows.append({"asof": preds[-1][:10], "stock": r["stock"],
                         "blend_z": round(float(r["blend"]), 3),
                         "sector": SECTOR_MAP.get(r["stock"], "other"),
                         "close": close, "ret20": ret20,
                         "adv20_twd": adv_v,
                         "proxy_flags": "|".join(flags) or "none",
                         "eligible_under_proxy": not flags,
                         "note": "REVIEW ONLY - not a trade list; borrow/"
                                 "shortability NOT verified"})
        diag_p = os.path.join(V12_DIR, "short_candidate_diagnostics.csv")
        pd.DataFrame(rows).to_csv(diag_p, index=False)
    if diag_frames:
        pd.concat(diag_frames, ignore_index=True).to_csv(
            os.path.join(V12_DIR, "short_worst_trades.csv"), index=False)

    # ---- gates (plan §3/§5)
    report += ["## Gate evaluation (pre-registered)", ""]
    verdict_ok = True
    for (win, sig), g in gate_results.items():
        adds = g["S50"] > g["L100"] and g["S100"] > g["L100"]
        alone = g["short_alone"] > 0
        report.append(f"- {win}/{sig}: short adds value after 2x costs: "
                      f"**{'YES' if adds else 'NO'}** (L100 {g['L100']:.2f} "
                      f"-> S50 {g['S50']:.2f} / S100 {g['S100']:.2f}); "
                      f"short leg standalone positive: "
                      f"{'YES' if alone else 'NO'} ({g['short_alone']:.2f})")
        if sig == "blend50" and not (adds and alone):
            verdict_ok = False
    report += ["", "## Caveats", "",
               "- Panel universe is survivorship-biased; short-side results "
               "are OPTIMISTIC (delisted losers absent).",
               "- Borrow availability/fees are a proxy; TWSE short-sale "
               "rules (uptick, quota) NOT modeled.",
               "- This report never constitutes trading advice or orders."]
    with open(os.path.join(V12_DIR, "short_side_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"[short-side] metrics -> short_side_metrics.csv; report -> "
          f"short_side_report.md" + (f"; diagnostics -> {diag_p}" if diag_p else ""))
    print("[short-side] blend50 gate "
          + ("PASSES preliminary screens" if verdict_ok else
             "FAILS - short side does not add value after conservative costs"))


if __name__ == "__main__":
    main()
