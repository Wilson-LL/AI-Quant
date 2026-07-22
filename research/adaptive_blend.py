"""Cycle 1 (C1/C2): static blend frontier + walk-forward adaptive TF/D1.2 blends.

Operates on a cached OOS score panel (default the 2021-26 bear panel) — CPU only.
All adaptive weights use only matured trailing data (see C1_C2_preregistration.md):
sleeve returns up to rebalance t-2, IC dates <= t-22, regime index closes <= t.

Usage:
  python research/adaptive_blend.py [panel_name]
Outputs: reports/continuous_research/C1_C2_results.json + printed table.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_hybrid import load_panel, merged, _swap_score, _cache_frames  # noqa: E402
from transformer_portfolio import backtest_scores, _metrics, summarize  # noqa: E402

OUT_DIR = os.path.join(ROOT, "reports", "continuous_research")
HOLD = 20
W_FLOOR, W_CAP = 0.2, 0.8


def market_index():
    """Equal-weight universe daily return index from the frozen cache."""
    rets = []
    for sid, df in _cache_frames().items():
        c = df.set_index("date")["close"]
        rets.append(c.pct_change().rename(sid))
    r = pd.concat(rets, axis=1).mean(axis=1).dropna()
    idx = (1 + r).cumprod()
    dd = idx / idx.rolling(252, min_periods=60).max() - 1.0
    return idx, dd


def sleeve_arrays(m, col, mode="long_short"):
    r = backtest_scores(_swap_score(m, col), holding=HOLD, mode=mode)
    return (np.array(r["gross"]), np.array(r["turnover"]),
            pd.to_datetime(pd.Index(r["dates"])), r)


def combo_metrics(g, t, dates, n_legs=2, label=""):
    out = {}
    for cb in (0, 60, 100, 150):
        out[f"net{cb}"] = _metrics(g - n_legs * (cb / 1e4) * t, HOLD)
    net60 = g - n_legs * (60 / 1e4) * t
    yr = dates.year.to_numpy()
    out["yearly_net60"] = {}
    for y in sorted(set(yr)):
        r = net60[yr == y]
        s = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / HOLD) if len(r) > 1 else float("nan")
        out["yearly_net60"][int(y)] = round(float(s), 2)
    sub = dates >= pd.Timestamp("2023-01-01")
    out["sub2023_net60"] = _metrics(net60[sub], HOLD)
    out["avg_turnover"] = round(float(t.mean()), 3)
    print(f"{label:26s} full net60 {out['net60']['sharpe']:5.2f} dd {out['net60']['max_dd']:7.2%} "
          f"| 2022 {out['yearly_net60'].get(2022, float('nan')):5.2f} "
          f"| 2023+ {out['sub2023_net60']['sharpe']:5.2f} | net100 {out['net100']['sharpe']:5.2f} "
          f"| turn {out['avg_turnover']:.2f}")
    return out


def trailing_sharpe(rets):
    if len(rets) < 3:
        return None
    r = np.asarray(rets, float)
    return float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / HOLD))


def run(panel_name=None):
    panel, pname = load_panel(panel_name or "BEAR_presetB_2021")
    m = merged(panel)
    results = {"panel": pname, "protocol": "L/S quintile equal-weight, hold 20, exec-lag-1"}

    # --- sleeves (identical rebalance grid: same panel, same dates) ---
    g_tf, t_tf, dates, r_tf = sleeve_arrays(m, "z_tf")
    g_d12, t_d12, dates2, r_d12 = sleeve_arrays(m, "z_mom")
    assert list(dates) == list(dates2), "sleeve rebalance grids differ"
    print(f"[sleeves] {len(g_tf)} rebalances {dates[0].date()}..{dates[-1].date()}")
    results["sleeve_tf"] = combo_metrics(g_tf, t_tf, dates, label="TF standalone")
    results["sleeve_d12"] = combo_metrics(g_d12, t_d12, dates, label="D1.2 standalone")

    # per-sleeve net60 returns for adaptive signals (matured use only)
    net_tf = g_tf - 2 * 0.006 * t_tf
    net_d12 = g_d12 - 2 * 0.006 * t_d12

    # --- C2 static, score-level ---
    for w in (0.3, 0.5, 0.7):
        mm = m.copy()
        mm["score"] = w * m["z_tf"] + (1 - w) * m["z_mom"]
        for mode in ("long_short", "long_only"):
            r = backtest_scores(mm, holding=HOLD, mode=mode)
            gg, tt = np.array(r["gross"]), np.array(r["turnover"])
            dd = pd.to_datetime(pd.Index(r["dates"]))
            results[f"score_blend{int(w*100)}_{mode}"] = combo_metrics(
                gg, tt, dd, n_legs=2 if mode == "long_short" else 1,
                label=f"score-blend {int(w*100)} [{mode[:2]}]")

    # --- C2 static, return-level ---
    for w in (0.3, 0.5, 0.7):
        g = w * g_tf + (1 - w) * g_d12
        t = w * t_tf + (1 - w) * t_d12
        results[f"ret_blend{int(w*100)}"] = combo_metrics(
            g, t, dates, label=f"ret-blend {int(w*100)}")

    # --- C1 adaptive, return-level ---
    idx, mkt_dd = market_index()

    # daily IC per signal (matured at decision time via <= t-22 panel dates)
    def daily_ic(col):
        def one(g):
            g = g[[col, "fwd_h"]].dropna()
            if len(g) < 5:
                return np.nan
            return g[col].rank().corr(g["fwd_h"].rank())
        return m.groupby("date", group_keys=False).apply(one, include_groups=False)
    ic_tf, ic_d12 = daily_ic("z_tf"), daily_ic("z_mom")
    panel_dates = np.array(sorted(m["date"].unique()))

    def weights_for(variant):
        ws = []
        for i, d in enumerate(dates):
            w = 0.5
            past_tf, past_d12 = net_tf[:max(i - 1, 0)], net_d12[:max(i - 1, 0)]
            if variant in ("AD1", "AD2"):
                K = 6 if variant == "AD1" else 12
                s_t, s_d = trailing_sharpe(past_tf[-K:]), trailing_sharpe(past_d12[-K:])
                if s_t is not None and s_d is not None:
                    w = float(np.exp(s_t) / (np.exp(s_t) + np.exp(s_d)))
            elif variant == "AD3":
                pos = int(np.searchsorted(panel_dates, np.datetime64(d)))
                elig = panel_dates[:max(pos - 21, 0)]
                if len(elig) >= 40:
                    win = elig[-126:]
                    a = float(np.nanmean(ic_tf.reindex(win)))
                    b = float(np.nanmean(ic_d12.reindex(win)))
                    w = (max(a, 0) + 0.05) / (max(a, 0) + max(b, 0) + 0.10)
            elif variant == "AD4":
                dd_now = mkt_dd.loc[:d]
                w = 0.8 if (len(dd_now) and dd_now.iloc[-1] < -0.10) else 0.5
            elif variant == "AD5":
                w = 0.8 if (len(past_d12) >= 2 and np.mean(past_d12[-2:]) < -0.03) else 0.5
            ws.append(min(max(w, W_FLOOR), W_CAP))
        return np.array(ws)

    for variant in ("AD1", "AD2", "AD3", "AD4", "AD5"):
        w = weights_for(variant)
        g = w * g_tf + (1 - w) * g_d12
        t = w * t_tf + (1 - w) * t_d12 + 0.5 * np.abs(np.diff(w, prepend=w[0]))
        res = combo_metrics(g, t, dates, label=f"adaptive {variant}")
        res["weights"] = [round(float(x), 3) for x in w]
        res["mean_w_tf"] = round(float(w.mean()), 3)
        results[f"adaptive_{variant}"] = res

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "C1_C2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nsaved ->", out_path)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
