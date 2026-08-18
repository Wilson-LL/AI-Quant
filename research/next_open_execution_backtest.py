"""v16 Task 2 — next-open execution timing audit (CPU-only).

Frozen 7-seed panels (CH 2023->, BR 2021->), production blend score,
untouched transformer_portfolio.backtest_scores engine; ONLY the
executable-price column changes between conventions (A..F, see
reports/continuous_research/v16_next_session_execution/
execution_cost_methodology.md — thresholds pre-registered there).

No training, no orders, cache read-only. open(T+1) is used strictly as
outcome data; signals are frozen bytes from pre-v16 runs.

Usage: python research/next_open_execution_backtest.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import transformer_portfolio as tp  # noqa: E402
from transformer_hybrid import load_panel, merged, _cache_frames  # noqa: E402
from data import SECTOR_MAP  # noqa: E402

OUT = os.path.join(ROOT, "reports", "continuous_research",
                   "v16_next_session_execution")
os.makedirs(OUT, exist_ok=True)

H = 20
PANELS = {"CH": "SCHED_A8_seeds7_full", "BR": "SCHED_BEAR_A8_seeds7_full"}
REFS = {"CH": 2.147, "BR": 1.443}
COST_LIST = (0, 60, 100, 150)
GAP_LIMIT = 0.05          # pre-registered corporate-action filter
CONVENTIONS = {           # ret column -> (label, entry description)
    "ret_c1": ("A/E: T+1 close -> T+21 close", 1),
    "ret_o1": ("B: T+1 open -> T+21 open", 1),
    "ret_o2": ("D: T+2 open -> T+22 open", 2),
    "ret_c0": ("F (diagnostic): T close -> T+20 close", 0),
    "ret_c2": ("decay: T+2 close -> T+22 close", 2),
}


def _fwd(px, lag, n):
    f = np.full(n, np.nan)
    stop = n - H - lag
    if stop > 0:
        f[:stop] = px[H + lag:] / px[lag:lag + stop] - 1.0
    return f


def exec_returns():
    """(date, stock) -> executable-outcome + gap columns from the raw cache."""
    rows = []
    for sid, df in _cache_frames().items():
        c = df["close"].to_numpy(np.float64)
        o = df["open"].to_numpy(np.float64)
        n = len(c)
        rec = {"date": df["date"].values, "stock": sid,
               "ret_c1": _fwd(c, 1, n), "ret_o1": _fwd(o, 1, n),
               "ret_o2": _fwd(o, 2, n), "ret_c0": _fwd(c, 0, n),
               "ret_c2": _fwd(c, 2, n)}
        g = np.full(n, np.nan)
        ia = np.full(n, np.nan)
        if n > 1:
            g[:-1] = o[1:] / c[:-1] - 1.0
            ia[:-1] = c[1:] / o[1:] - 1.0
        rec["gap1"], rec["intra1"] = g, ia
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def synthetic_check():
    """Hand-verifiable alignment test for _fwd (no-lookahead assertion)."""
    n = H + 5
    px = np.arange(1.0, n + 1)           # px[i] = i+1
    for lag in (0, 1, 2):
        f = _fwd(px, lag, n)
        # row 0: px[H+lag]/px[lag] - 1
        exp = px[H + lag] / px[lag] - 1.0
        assert abs(f[0] - exp) < 1e-12, f"synthetic fwd lag{lag} misaligned"
        assert np.isnan(f[n - H - lag]) if n - H - lag < n else True
    print("[check] synthetic forward-return alignment OK (lag 0/1/2)")


def blend(m):
    out = m.copy()
    out["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    return out


def engine(df, mode, ret_col):
    return tp.backtest_scores(df, holding=H, mode=mode, no_trade_band=0.10,
                              cost_bps_list=COST_LIST, ret_col=ret_col)


def replica(df, mode, ret_col, drop_gap=False):
    """Byte-faithful replica of backtest_scores' loop that can drop
    extreme-gap names AFTER selection (weights renormalized). With
    drop_gap=False it must match engine() exactly (asserted in main)."""
    d2 = (df.dropna(subset=["score", ret_col])
            .drop_duplicates(["date", "stock"], keep="last").copy())
    dates = np.array(sorted(d2["date"].unique()))
    by_date = dict(tuple(d2.groupby("date")))
    gross, turns, kept, dropped = [], [], [], 0
    prev_w = {}
    for d in dates[::H]:
        day = by_date.get(d)
        if day is None:
            continue
        n = day["stock"].nunique()
        if n < 60:
            continue
        kq = max(3, round(0.2 * n))
        if mode == "long_short" and n < 2 * kq:
            continue
        o = day.sort_values("score")
        pool = set(o.tail(int(kq * 1.2))["stock"])
        longs = list(o.tail(kq)["stock"])
        if prev_w:
            inc = [s for s in prev_w.get("L", {}) if s in pool]
            longs = (inc + [s for s in longs if s not in inc])[:kq]
        wl = tp._leg_weights(day, longs, "equal")
        legs = {"L": (longs, wl, 1.0)}
        if mode == "long_short":
            shorts = list(o.head(kq)["stock"])
            legs["S"] = (shorts, tp._leg_weights(day, shorts, "equal"), -1.0)
        gap_map = day.set_index("stock")["gap1"]
        ret_map = day.set_index("stock")[ret_col]
        g, new_w = 0.0, {}
        for leg, (names, w, sign) in legs.items():
            w = np.asarray(w, np.float64).copy()
            if drop_gap:
                bad = np.array([abs(gap_map.get(s, 0.0)) > GAP_LIMIT
                                if np.isfinite(gap_map.get(s, np.nan))
                                else False for s in names])
                dropped += int(bad.sum())
                if bad.any() and (~bad).any():
                    w[bad] = 0.0
                    w = w / w.sum()
                elif bad.all():
                    w[:] = 1.0 / len(w)   # degenerate: keep book, note it
            g += sign * float(np.sum(w * ret_map.loc[names].to_numpy()))
            new_w[leg] = dict(zip(names, w))
        t = 0.0
        for leg in new_w:
            old = prev_w.get(leg, {})
            alln = set(new_w[leg]) | set(old)
            t += sum(abs(new_w[leg].get(s, 0) - old.get(s, 0))
                     for s in alln) / 2
        t /= len(new_w)
        gross.append(g)
        turns.append(t)
        kept.append(d)
        prev_w = new_w
    return (np.array(gross), np.array(turns), np.array(kept), dropped)


def net_metrics(gross, turn, kept, mode, cb=60):
    n_legs = 2 if mode == "long_short" else 1
    net = gross - n_legs * (cb / 1e4) * turn
    return tp._metrics(net, H), net


def yearly(net, kept):
    yr = pd.Series(kept).astype("datetime64[ns]").dt.year.to_numpy()
    out = {}
    for y in sorted(set(yr)):
        r = net[yr == y]
        s = (r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / H)
             if len(r) > 1 else float("nan"))
        out[int(y)] = {"sharpe": round(float(s), 2),
                       "mean_pct": round(float(r.mean()) * 100, 2),
                       "n": int(len(r))}
    return out


def attribution(df):
    """Per-rebalance long-book roles + gap/intraday stats (long-only book,
    the deployment-relevant side)."""
    d2 = (df.dropna(subset=["score", "ret_o1"])
            .drop_duplicates(["date", "stock"], keep="last").copy())
    d2["quintile"] = d2.groupby("date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    dates = np.array(sorted(d2["date"].unique()))
    by_date = dict(tuple(d2.groupby("date")))
    recs = []
    prev = []
    for d in dates[::H]:
        day = by_date.get(d)
        if day is None or day["stock"].nunique() < 60:
            continue
        n = day["stock"].nunique()
        kq = max(3, round(0.2 * n))
        o = day.sort_values("score")
        pool = set(o.tail(int(kq * 1.2))["stock"])
        longs = list(o.tail(kq)["stock"])
        if prev:
            inc = [s for s in prev if s in pool]
            longs = (inc + [s for s in longs if s not in inc])[:kq]
        exits = [s for s in prev if s not in longs]
        idx = day.set_index("stock")
        for role, names in (("entrant", [s for s in longs if s not in prev]),
                            ("incumbent", [s for s in longs if s in prev]),
                            ("exit", exits)):
            for s in names:
                if s not in idx.index:
                    continue
                r = idx.loc[s]
                recs.append({"date": d, "stock": s, "role": role,
                             "gap1": r["gap1"], "intra1": r["intra1"],
                             "quintile": r["quintile"],
                             "sector": SECTOR_MAP.get(s, "other")})
        prev = longs
    return pd.DataFrame(recs)


def main():
    synthetic_check()
    ex = exec_returns()
    results = {}
    decay_rows, audit_rows = [], []
    attr_frames = []
    for win, pname in PANELS.items():
        panel, _ = load_panel(pname)
        m = blend(merged(panel))
        m = m.merge(ex, on=["date", "stock"], how="left")
        # --- verification 1: rebuilt close-lag-1 equals the panel's fwd_h
        both = m.dropna(subset=["fwd_h", "ret_c1"])
        dmax = float((both["fwd_h"] - both["ret_c1"]).abs().max())
        print(f"[check] {win}: |fwd_h - rebuilt ret_c1| max = {dmax:.2e} "
              f"on {len(both)} rows")
        assert dmax <= 1e-6, "MISMATCH: rebuilt close returns != panel fwd_h"
        # --- reproduction gate on the RAW panel (standing references)
        ref = engine(m, "long_short", "fwd_h")
        print(f"[gate] {win} raw-panel A (fwd_h) LS net60 sharpe = "
              f"{ref['net60']['sharpe']}  (expected {REFS[win]})")
        results[(win, "gate")] = ref["net60"]["sharpe"]
        # --- identity E: engine on rebuilt ret_c1 must match A. Restricted
        # to A's coverage: the live cache matures ~21 sessions of labels
        # that were still immature at panel freeze, so unrestricted E sees
        # one extra rebalance (diagnosed 2026-08-18; not a misalignment).
        e_chk = engine(m[m["fwd_h"].notna()], "long_short", "ret_c1")
        # --- common-coverage subset for paired comparisons
        need = ["fwd_h", "ret_c1", "ret_o1", "ret_o2", "ret_c0", "ret_c2"]
        sub = m.dropna(subset=["score"] + need).copy()
        print(f"[info] {win}: raw rows {len(m)}, common-coverage rows "
              f"{len(sub)}")
        for mode in ("long_short", "long_only"):
            base = engine(sub, mode, "ret_c1")
            for rc, (label, lag) in CONVENTIONS.items():
                r = engine(sub, mode, rc)
                row = {"window": win, "mode": mode, "convention": label,
                       "ret_col": rc, "n_rebal": r["n_rebal"],
                       "turnover": round(r["avg_turnover"], 3)}
                for cb in COST_LIST:
                    row[f"net{cb}_sharpe"] = r[f"net{cb}"]["sharpe"]
                nm = r["net60"]
                row.update({"net60_ann_ret": nm["ann_ret"],
                            "net60_max_dd": nm["max_dd"],
                            "net60_hit": nm["hit"],
                            "avg_rebal_ret_net60": round(float(np.mean(
                                np.array(r["gross"]) -
                                (2 if mode == "long_short" else 1) *
                                0.006 * np.array(r["turnover"]))), 5),
                            "yearly_net60": json.dumps(r["yearly_net60"])})
                audit_rows.append(row)
                decay_rows.append({"window": win, "mode": mode,
                                   "point": label, "lag": lag,
                                   "net0": r["net0"]["sharpe"],
                                   "net60": r["net60"]["sharpe"],
                                   "net100": r["net100"]["sharpe"],
                                   "net150": r["net150"]["sharpe"]})
                results[(win, mode, rc)] = r
            results[(win, mode, "base")] = base
        # identity check E vs A on the raw panel (block-return arrays)
        ga = np.array(ref["gross"])
        ge = np.array(e_chk["gross"])
        ok = (len(ga) == len(ge)
              and float(np.abs(ga - ge).max()) <= 1e-6)
        print(f"[check] {win}: identity E==A on raw panel: {ok}")
        results[(win, "identityE")] = bool(ok)
        # --- replica validation + corporate-action robustness (paired)
        for mode in ("long_short", "long_only"):
            for rc in ("ret_c1", "ret_o1"):
                er = results[(win, mode, rc)]
                g, t, k, _ = replica(sub, mode, rc, drop_gap=False)
                em, _ = net_metrics(g, t, k, mode)
                assert abs(em["sharpe"] - er["net60"]["sharpe"]) <= 0.001, (
                    f"replica mismatch {win} {mode} {rc}: "
                    f"{em['sharpe']} vs {er['net60']['sharpe']}")
                gf, tf_, kf, ndrop = replica(sub, mode, rc, drop_gap=True)
                fm, _ = net_metrics(gf, tf_, kf, mode)
                results[(win, mode, rc, "gapfilter")] = {
                    "net60_sharpe": fm["sharpe"], "dropped_obs": ndrop}
        print(f"[check] {win}: replica matches engine; gap-filter runs done")
        # --- attribution (long book)
        at = attribution(sub)
        at["window"] = win
        attr_frames.append(at)
    # ---------- outputs ----------
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(os.path.join(OUT, "next_open_execution_audit.csv"),
                 index=False)
    decay = pd.DataFrame(decay_rows)
    decay.to_csv(os.path.join(OUT, "signal_decay_report.csv"), index=False)
    attr = pd.concat(attr_frames, ignore_index=True)
    # aggregated attribution table
    aggs = []
    for (win, role), gdf in attr.groupby(["window", "role"]):
        aggs.append({"window": win, "group": f"role={role}",
                     "n": len(gdf),
                     "gap_mean_bps": round(gdf["gap1"].mean() * 1e4, 1),
                     "gap_median_bps": round(gdf["gap1"].median() * 1e4, 1),
                     "intra_mean_bps": round(gdf["intra1"].mean() * 1e4, 1),
                     "gap_p90_bps": round(gdf["gap1"].quantile(0.9) * 1e4, 1),
                     "gap_p10_bps": round(gdf["gap1"].quantile(0.1) * 1e4, 1)})
    for (win, q), gdf in attr.groupby(["window", "quintile"]):
        aggs.append({"window": win, "group": f"quintile={int(q)}",
                     "n": len(gdf),
                     "gap_mean_bps": round(gdf["gap1"].mean() * 1e4, 1),
                     "gap_median_bps": round(gdf["gap1"].median() * 1e4, 1),
                     "intra_mean_bps": round(gdf["intra1"].mean() * 1e4, 1),
                     "gap_p90_bps": round(gdf["gap1"].quantile(0.9) * 1e4, 1),
                     "gap_p10_bps": round(gdf["gap1"].quantile(0.1) * 1e4, 1)})
    for (win, sec), gdf in attr.groupby(["window", "sector"]):
        if len(gdf) >= 200:
            aggs.append({"window": win, "group": f"sector={sec}",
                         "n": len(gdf),
                         "gap_mean_bps": round(gdf["gap1"].mean() * 1e4, 1),
                         "gap_median_bps": round(gdf["gap1"].median() * 1e4,
                                                 1),
                         "intra_mean_bps": round(gdf["intra1"].mean() * 1e4,
                                                 1),
                         "gap_p90_bps": round(gdf["gap1"].quantile(0.9) * 1e4,
                                              1),
                         "gap_p10_bps": round(gdf["gap1"].quantile(0.1) * 1e4,
                                              1)})
    pd.DataFrame(aggs).to_csv(
        os.path.join(OUT, "next_open_gap_attribution.csv"), index=False)
    extremes = attr.reindex(attr["gap1"].abs().sort_values(
        ascending=False).index).head(20)
    extremes.to_csv(os.path.join(OUT, "extreme_gaps_top20.csv"), index=False)
    # machine-readable summary for the report writer
    summary = {}
    for win in PANELS:
        s = {"gate_sharpe": results[(win, "gate")],
             "identity_E": results[(win, "identityE")]}
        for mode in ("long_short", "long_only"):
            a = results[(win, mode, "ret_c1")]["net60"]
            b = results[(win, mode, "ret_o1")]["net60"]
            s[mode] = {
                "A_net60": a["sharpe"], "B_net60": b["sharpe"],
                "retention": round(b["sharpe"] / a["sharpe"], 3)
                if a["sharpe"] else float("nan"),
                "A_dd": a["max_dd"], "B_dd": b["max_dd"],
                "A_ann": a["ann_ret"], "B_ann": b["ann_ret"],
                "A_turn": results[(win, mode, "ret_c1")]["avg_turnover"],
                "B_turn": results[(win, mode, "ret_o1")]["avg_turnover"],
                "A_yearly": results[(win, mode, "ret_c1")]["yearly_net60"],
                "B_yearly": results[(win, mode, "ret_o1")]["yearly_net60"],
                "gapfilter_A": results[(win, mode, "ret_c1", "gapfilter")],
                "gapfilter_B": results[(win, mode, "ret_o1", "gapfilter")],
            }
        summary[win] = s
    with open(os.path.join(OUT, "task2_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("\n[done] wrote audit/decay/attribution CSVs + task2_summary.json")
    for win in PANELS:
        for mode in ("long_short", "long_only"):
            s = summary[win][mode]
            print(f"  {win} {mode:10s} A {s['A_net60']:.3f} -> "
                  f"B {s['B_net60']:.3f}  retention {s['retention']:.3f}")


if __name__ == "__main__":
    main()
