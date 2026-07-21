"""G6 (hybrid/ensemble with D1.2 momentum) and G7 (portfolio construction sweep).

Operates on saved OOS score panels (reports/transformer_gpu/panels/*.csv.gz) —
no GPU needed. The D1.2 momentum signal and alternative-horizon forward returns
are recomputed from the frozen cache with the same exec-lag-1 convention.

Usage:
  python research/transformer_hybrid.py G6 [panel_name]
  python research/transformer_hybrid.py G7 [panel_name]
(default panel: the champion G2 config if present, else G1 seq40)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import load_universe, SECTOR_MAP  # noqa: E402
from transformer_portfolio import backtest_scores, summarize  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")
PANEL_DIR = os.path.join(REPORT_DIR, "panels")


def load_panel(name=None):
    if name is None:
        cands = [f for f in sorted(os.listdir(PANEL_DIR)) if f.endswith(".csv.gz")]
        pref = [c for c in cands if "hl_126" in c] or [c for c in cands if "G1" in c] or cands
        name = pref[0]
    p = os.path.join(PANEL_DIR, name if name.endswith(".csv.gz") else name + ".csv.gz")
    panel = pd.read_csv(p, parse_dates=["date"], dtype={"stock": str})
    print(f"[panel] {os.path.basename(p)}: {len(panel)} rows, "
          f"{panel['date'].min().date()}..{panel['date'].max().date()}")
    return panel, os.path.basename(p).replace(".csv.gz", "")


def _cache_frames():
    uni = load_universe([s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"])
    out = {}
    for sid, df in uni.items():
        out[sid] = (df.sort_values("date").drop_duplicates("date", keep="last")
                      .reset_index(drop=True))
    return out


def aux_columns(horizons=(5, 10), exec_lag=1):
    """(date, stock) -> mom126_5, vol_20, fwd_5, fwd_10 from the cache."""
    rows = []
    for sid, df in _cache_frames().items():
        c = df["close"].to_numpy(np.float64)
        lr = np.log(c[1:] / c[:-1])
        vol20 = pd.Series(np.concatenate([[np.nan], lr])).rolling(20).std().to_numpy()
        rec = {"date": df["date"].values, "stock": sid,
               "mom": pd.Series(c).shift(5).to_numpy() / pd.Series(c).shift(131).to_numpy() - 1.0,
               "vol_20": vol20}
        n = len(c)
        for Y in horizons:
            f = np.full(n, np.nan)
            stop = n - Y - exec_lag
            if stop > 0:
                f[:stop] = c[Y + exec_lag:] / c[exec_lag:exec_lag + stop] - 1.0
            rec[f"fwd_{Y}"] = f
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def _z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def merged(panel):
    aux = aux_columns()
    m = panel.merge(aux, on=["date", "stock"], how="left")
    g = m.groupby("date")
    m["z_tf"] = g["score"].transform(_z)
    m["z_mom"] = g["mom"].transform(_z)
    return m


def _swap_score(df, col):
    out = df.copy()
    out["score"] = out[col]
    return out


def g6(panel_name=None):
    panel, pname = load_panel(panel_name)
    m = merged(panel)
    out_path = os.path.join(REPORT_DIR, "G6_results.json")
    results = {"panel": pname}

    def ev(df, label, **kw):
        res = {}
        for mode in ("long_short", "long_only"):
            r = backtest_scores(df, holding=20, mode=mode, **kw)
            print(summarize(r, f"{label} [{mode}]"))
            for kk in ("gross", "turnover", "dates"):
                r.pop(kk, None)
            res[mode] = r
        results[label] = res
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    ev(m, "tf_standalone")
    ev(_swap_score(m, "z_mom"), "d12_alone_same_panel")
    for a in (0.3, 0.5, 0.7):
        mm = m.copy()
        mm["score"] = a * m["z_tf"] + (1 - a) * m["z_mom"]
        ev(mm, f"blend_tf{int(a*100)}")

    # D1.2 filtered by transformer: momentum ranking, but drop names in the
    # bottom 30% of transformer score each date (tf as veto/filter)
    mm = m.copy()
    thresh = mm.groupby("date")["z_tf"].transform(lambda s: s.quantile(0.30))
    mm["score"] = np.where(mm["z_tf"] >= thresh, mm["z_mom"], -1e9)
    ev(mm, "d12_tf_filter")

    # transformer filtered by momentum (mom as veto)
    mm = m.copy()
    thresh = mm.groupby("date")["z_mom"].transform(lambda s: s.quantile(0.30))
    mm["score"] = np.where(mm["z_mom"] >= thresh, mm["z_tf"], -1e9)
    ev(mm, "tf_d12_filter")

    # sleeve: return-level 80% D1.2 + 20% transformer (L/S books)
    r_d = backtest_scores(_swap_score(m, "z_mom"), holding=20, mode="long_short")
    r_t = backtest_scores(m, holding=20, mode="long_short")
    if r_d["dates"] == r_t["dates"]:
        for w in (0.2,):
            g = (1 - w) * np.array(r_d["gross"]) + w * np.array(r_t["gross"])
            t = (1 - w) * np.array(r_d["turnover"]) + w * np.array(r_t["turnover"])
            from transformer_portfolio import _metrics
            sleeve = {f"net{cb}": _metrics(g - 2 * (cb / 1e4) * t, 20)
                      for cb in (0, 60, 100, 150)}
            sleeve["avg_turnover"] = float(t.mean())
            results[f"sleeve_d12_{int((1-w)*100)}_tf_{int(w*100)}"] = sleeve
            print(f"sleeve 80/20 net60 Sharpe {sleeve['net60']['sharpe']}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nG6 done ->", out_path)


def g7(panel_name=None):
    panel, pname = load_panel(panel_name)
    m = merged(panel)
    out_path = os.path.join(REPORT_DIR, "G7_results.json")
    results = {"panel": pname}

    def ev(label, **kw):
        r = backtest_scores(m, **kw)
        print(summarize(r, label))
        for kk in ("gross", "turnover", "dates"):
            r.pop(kk, None)
        results[label] = r
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    for holding, rc in [(5, "fwd_5"), (10, "fwd_10"), (20, "fwd_h")]:
        ev(f"LS_quintile_eq_h{holding}", holding=holding, mode="long_short",
           ret_col=rc)
        ev(f"LO_quintile_eq_h{holding}", holding=holding, mode="long_only",
           ret_col=rc)
    for weighting in ("rank", "invvol"):
        ev(f"LS_quintile_{weighting}_h20", holding=20, mode="long_short",
           weighting=weighting)
        ev(f"LO_quintile_{weighting}_h20", holding=20, mode="long_only",
           weighting=weighting)
    ev("LO_top10_eq_h20", holding=20, mode="long_only", k=10)
    for band in (0.05, 0.10):
        ev(f"LO_quintile_eq_band{int(band*100)}_h20", holding=20,
           mode="long_only", no_trade_band=band)
        ev(f"LS_quintile_eq_band{int(band*100)}_h20", holding=20,
           mode="long_short", no_trade_band=band)
    print("\nG7 done ->", out_path)


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else "G6"
    pname = sys.argv[2] if len(sys.argv) > 2 else None
    {"G6": g6, "G7": g7}[group](pname)
