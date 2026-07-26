"""Queue v9 experiment implementations (CPU-only, frozen signal).

Each function returns a JSON-serializable dict with a `verdict` field.
Run via research/run_queue_v9.py — not directly.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import transformer_portfolio as tp  # noqa: E402
from queue_v9_lib import (ROOT, V9_DIR, PANELS, SPECS, get_merged, with_score,  # noqa: E402
                          run_book, slim, lag_returns, adv_frame, hostile_dates,
                          holdings_walk)

GATE_WINDOWS = ("CH", "BR")
ALL_WINDOWS = ("CH", "BR", "W22")
ALL_SPECS = ("blend", "d7b", "tf", "d12")


def _net60_series(r):
    n_legs = 2 if r["mode"] == "long_short" else 1
    g = np.array(r["gross"])
    t = np.array(r["turnover"])
    return g - n_legs * (60 / 1e4) * t, t, r["dates"]


# ---------------------------------------------------------------- Track 1

def c1():
    """Construction grid band{10,15} x cap{7.5,10} on the blend score,
    all four comparison books, LS+LO, three windows."""
    out = {"grid": {}, "refs": {}}
    cells = {"band10_cap10": (0.10, 0.10), "band15_cap75": (0.15, 0.075),
             "band10_cap75": (0.10, 0.075), "band15_cap10": (0.15, 0.10)}
    for win in ALL_WINDOWS:
        out["grid"][win] = {}
        for cell, (band, cap) in cells.items():
            out["grid"][win][cell] = {}
            for mode in ("long_short", "long_only"):
                r = run_book(win, "blend", mode, band=band, name_cap=cap)
                out["grid"][win][cell][mode] = slim(r)
        out["refs"][win] = {}
        for spec in ("tf", "d12"):
            out["refs"][win][spec] = {m: slim(run_book(win, spec, m))
                                      for m in ("long_short", "long_only")}
    # pre-registered dominance rule vs d7b cell (band15_cap75), LS net60
    d7b = {w: out["grid"][w]["band15_cap75"]["long_short"] for w in GATE_WINDOWS}
    challengers = {}
    for cell in cells:
        if cell == "band15_cap75":
            continue
        ok = True
        for w in GATE_WINDOWS:
            c = out["grid"][w][cell]["long_short"]
            if c["net60"]["sharpe"] < d7b[w]["net60"]["sharpe"] - 0.05:
                ok = False
            if c["avg_turnover"] > d7b[w]["avg_turnover"] + 1e-9:
                ok = False
        if out["grid"]["BR"][cell]["long_short"]["net60"]["max_dd"] > \
           d7b["BR"]["net60"]["max_dd"] + 0.01:
            pass  # dd better by >=1pp required
        else:
            ok = False
        challengers[cell] = ok
    dominating = [c for c, v in challengers.items() if v]
    out["verdict"] = ("DOMINATED-BY:" + ",".join(dominating)) if dominating else \
        "D7b (band15+cap7.5) stays recommended — no grid cell dominates on Sharpe+DD+turnover"
    out["chosen_spec"] = dominating[0] if dominating else "band15_cap75"
    return out


# ---------------------------------------------------------------- Track 2

def x1():
    """Cost curve 0/60/100/150/200 bps + linear break-even bps."""
    out = {"books": {}, "flags": []}
    for win in ALL_WINDOWS:
        out["books"][win] = {}
        for spec in ALL_SPECS:
            out["books"][win][spec] = {}
            for mode in ("long_short", "long_only"):
                r = run_book(win, spec, mode, cost_bps=(0, 60, 100, 150, 200),
                             keep_series=True)
                n_legs = 2 if mode == "long_short" else 1
                g, t = np.array(r["gross"]), np.array(r["turnover"])
                be = float(np.mean(g) / (n_legs * max(np.mean(t), 1e-9)) * 1e4)
                s = slim(r)
                s["breakeven_bps"] = round(be, 0)
                out["books"][win][spec][mode] = s
                if win in GATE_WINDOWS:
                    thr = 1.0 if win == "CH" else 0.7
                    if s["net150"]["sharpe"] < thr and spec in ("blend", "d7b"):
                        out["flags"].append(f"{win}/{spec}/{mode}: net150 "
                                            f"{s['net150']['sharpe']} < {thr}")
    out["verdict"] = "WARNING: " + "; ".join(out["flags"]) if out["flags"] else \
        "PASS — deployment specs clear net150 thresholds (CH>=1.0, BR>=0.7) both modes"
    return out


def x2():
    """ADV-proxy capacity stress. Participation = trade notional / adv20 at
    capital 1M/10M/100M TWD; first-order Sharpe impact estimated as
    capped-trade fraction x (lag1 - lag2 Sharpe decay from X3 logic)."""
    adv = adv_frame().set_index(["date", "stock"])["adv20"]
    lag = lag_returns()
    out = {"capacity": {}, "note": "ADV proxy = 20d median(close*volume); cache "
           "is OHLCV-only (no true turnover field until full-field data ~2027-01)"}
    P_LIMIT = 0.10  # max participation per rebalance day
    for win in GATE_WINDOWS:
        out["capacity"][win] = {}
        for spec in ALL_SPECS:
            df = with_score(get_merged(win), spec)
            band, cap = SPECS[spec]["band"], SPECS[spec]["cap"]
            # trades from the holdings replica (long leg; LO deployment view)
            prev = {}
            parts = []  # (participation, |trade_w|) rows across all rebalances
            for d, w, _day in holdings_walk(df, mode="long_only", band=band,
                                            name_cap=cap):
                for s in set(w) | set(prev):
                    dw = abs(w.get(s, 0) - prev.get(s, 0))
                    if dw < 1e-9:
                        continue
                    a = adv.get((d, s), np.nan)
                    if np.isfinite(a) and a > 0:
                        parts.append((dw / a, dw))  # participation per TWD capital
                prev = w
            parts = np.array(parts)
            spec_out = {}
            # Sharpe decay lag1->lag2 for impact estimate (delayed fills)
            ld = df.merge(lag, on=["date", "stock"], how="left")
            s1 = run_book(win, spec, "long_only")["net60"]["sharpe"]
            s2 = run_book(win, spec, "long_only", ret_col="fwd_lag2",
                          score_override=lambda m, _ld=ld: _ld)["net60"]["sharpe"]
            decay = max(0.0, s1 - s2)
            for cap_twd in (1e6, 1e7, 1e8):
                p = parts[:, 0] * cap_twd
                capped = p > P_LIMIT
                frac_capped_w = float(parts[capped, 1].sum() / max(parts[:, 1].sum(), 1e-12))
                spec_out[f"capital_{cap_twd:.0e}"] = {
                    "p95_participation": round(float(np.percentile(p, 95)), 4),
                    "max_participation": round(float(p.max()), 3),
                    "frac_trades_capped": round(float(capped.mean()), 4),
                    "frac_traded_weight_capped": round(frac_capped_w, 4),
                    "est_sharpe_impact": round(-frac_capped_w * decay, 3)}
            spec_out["lag1_lag2_decay"] = round(decay, 3)
            out["capacity"][win][spec] = spec_out
    flag = out["capacity"]["CH"]["d7b"]["capital_1e+07"]
    out["verdict"] = ("FLAG: 10M TWD impact " + str(flag["est_sharpe_impact"])
                      if flag["est_sharpe_impact"] < -0.1 else
                      "PASS — 10M TWD portfolio: est Sharpe impact "
                      f"{flag['est_sharpe_impact']} (>= -0.1) on deployment spec")
    return out


def x3():
    """Execution delay: verify no same-bar execution (panel fwd_h == cache
    lag-1 fwd20), then T+2 / T+3 decay for all books."""
    lag = lag_returns()
    out = {"same_bar_audit": {}, "delay": {}}
    for win in ALL_WINDOWS:
        m = get_merged(win)
        chk = m.merge(lag, on=["date", "stock"], how="left")
        both = chk.dropna(subset=["fwd_h", "fwd_lag1"])
        out["same_bar_audit"][win] = {
            "rows_compared": int(len(both)),
            "max_abs_diff": round(float((both["fwd_h"] - both["fwd_lag1"]).abs().max()), 8)}
    out["delay"] = {}
    for win in GATE_WINDOWS:
        out["delay"][win] = {}
        for spec in ALL_SPECS:
            df = with_score(get_merged(win), spec).merge(lag, on=["date", "stock"],
                                                         how="left")
            row = {}
            for mode in ("long_short", "long_only"):
                r1 = run_book(win, spec, mode)
                res = {"T+1": r1["net60"]["sharpe"]}
                for lg in (2, 3):
                    rl = run_book(win, spec, mode, ret_col=f"fwd_lag{lg}",
                                  score_override=lambda m, _df=df: _df)
                    res[f"T+{lg}"] = rl["net60"]["sharpe"]
                res["T2_cost"] = round(res["T+1"] - res["T+2"], 3)
                res["T3_cost"] = round(res["T+1"] - res["T+3"], 3)
                row[mode] = res
            out["delay"][win][spec] = row
    audit_ok = all(v["max_abs_diff"] < 1e-6 for v in out["same_bar_audit"].values())
    worst_t2 = max(out["delay"][w][s][m]["T2_cost"]
                   for w in GATE_WINDOWS for s in ("blend", "d7b")
                   for m in ("long_short", "long_only"))
    out["verdict"] = (("same-bar audit PASS; " if audit_ok else
                       "SAME-BAR AUDIT FAILED; ") +
                      (f"T+2 worst cost {worst_t2} <= 0.30 — delay-robust"
                       if worst_t2 <= 0.30 else
                       f"FRAGILE: T+2 costs {worst_t2} > 0.30 — deployment "
                       "requires guaranteed next-session execution"))
    return out


def x4():
    """LO sell-first / settled-cash worst case: new entries execute T+3
    instead of T+1. First-order book model: r = (1-t)*r_lag1 + t*r_lag3."""
    lag = lag_returns()
    out = {"books": {}}
    worst = 0.0
    for win in GATE_WINDOWS:
        out["books"][win] = {}
        for spec in ALL_SPECS:
            df = with_score(get_merged(win), spec).merge(lag, on=["date", "stock"],
                                                         how="left")
            r1 = run_book(win, spec, "long_only", keep_series=True)
            r3 = run_book(win, spec, "long_only", ret_col="fwd_lag3",
                          score_override=lambda m, _df=df: _df, keep_series=True)
            n1, t1, d1 = _net60_series(r1)
            n3, _, d3 = _net60_series(r3)
            k = min(len(n1), len(n3))
            if d1[:k] != d3[:k]:
                out["books"][win][spec] = {"error": "rebalance dates diverge"}
                continue
            mixed = (1 - t1[:k]) * n1[:k] + t1[:k] * n3[:k]
            m_base = tp._metrics(n1[:k], 20)
            m_mix = tp._metrics(mixed, 20)
            cost = round(m_base["sharpe"] - m_mix["sharpe"], 3)
            if spec in ("blend", "d7b"):
                worst = max(worst, cost)
            out["books"][win][spec] = {"base_sharpe": m_base["sharpe"],
                                       "cash_constrained_sharpe": m_mix["sharpe"],
                                       "sharpe_cost": cost,
                                       "avg_new_entry_frac": round(float(t1[:k].mean()), 3)}
    out["verdict"] = (f"PASS — worst-case settled-cash cost {worst} < 0.05; "
                      "idealized rebalance is safe for the paper book"
                      if worst < 0.05 else
                      f"ADOPT CONSTRAINED FILLS: worst-case cost {worst} >= 0.05 "
                      "— paper book generator must model T+3 buys")
    return out


# ---------------------------------------------------------------- Track 3

def p1():
    """Matured forward-return ledger: run paper_trading evaluate, then write
    LEDGER_STATUS.md with maturity accounting and the armed evidence gate."""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "research", "paper_trading.py"),
                        "evaluate"], capture_output=True, text=True, timeout=1800)
    led_path = os.path.join(ROOT, "reports", "paper_trading", "PAPER_LEDGER.csv")
    if r.returncode != 0 or not os.path.exists(led_path):
        return {"verdict": "FAIL — paper_trading evaluate errored",
                "stderr": (r.stderr or "")[-2000:]}
    led = pd.read_csv(led_path)
    out = {"n_rows": int(len(led)), "strategies": sorted(led["strategy"].unique().tolist()),
           "matured": {}}
    lines = ["# Paper-ledger status (queue v9 / P1)", "",
             f"Rows: {len(led)} · asof range {led['asof'].min()}..{led['asof'].max()}", "",
             "| strategy | matured 1d | 5d | 10d | 20d | mean 20d | ann Sharpe (gross) |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for strat, g in led.groupby("strategy"):
        counts = {h: int(g[f"ret_{h}d"].notna().sum()) for h in (1, 5, 10, 20)}
        m20 = g["ret_20d"].dropna()
        sh = float(m20.mean() / (m20.std() + 1e-12) * np.sqrt(252 / 20)) if len(m20) > 1 else np.nan
        out["matured"][strat] = {**{f"n_{h}d": counts[h] for h in counts},
                                 "mean_20d": round(float(m20.mean()), 5) if len(m20) else None,
                                 "ann_sharpe_gross": round(sh, 2) if sh == sh else None}
        lines.append(f"| {strat} | {counts[1]} | {counts[5]} | {counts[10]} | {counts[20]} "
                     f"| {m20.mean():+.2%} | {sh:.2f} |" if len(m20) else
                     f"| {strat} | {counts[1]} | {counts[5]} | {counts[10]} | {counts[20]} | — | — |")
    n20 = min(v["n_20d"] for v in out["matured"].values()) if out["matured"] else 0
    gate = (f"EVIDENCE GATE ARMED: {n20}/20 matured 20d observations — "
            "realized-vs-backtest comparison auto-reports at 20."
            if n20 < 20 else
            "EVIDENCE GATE ACTIVE: >=20 matured obs — compare ann Sharpe vs "
            "bootstrap CI (champ p5 1.61 / p50 1.92; bear p5 1.14 / p50 1.37).")
    lines += ["", gate, "",
              "Books tracked: d12 / tf / blend50 / blend50_band10 (gross, LO, "
              "next-session execution). NaN cells above = not yet matured."]
    status_path = os.path.join(ROOT, "reports", "paper_trading", "LEDGER_STATUS.md")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out["gate"] = gate
    out["status_file"] = os.path.relpath(status_path, ROOT)
    out["verdict"] = f"OPERATIONAL — ledger populated ({len(led)} rows), {gate.split(':')[0]}"
    return out


def p2():
    """Daily diff report: generate for the 3 most recent snapshot dates."""
    book_dir = os.path.join(ROOT, "reports", "paper_trading", "books")
    dates = sorted({f[:10] for f in os.listdir(book_dir)}) if os.path.isdir(book_dir) else []
    if len(dates) < 2:
        return {"verdict": "FAIL — <2 snapshot dates, nothing to diff"}
    targets = dates[-3:]
    outs = []
    for d in targets:
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "research", "daily_diff_report.py"),
                            "--asof", d], capture_output=True, text=True, timeout=600)
        outs.append({"asof": d, "rc": r.returncode,
                     "tail": (r.stdout or r.stderr or "").strip().splitlines()[-1:]})
    ok = all(o["rc"] == 0 for o in outs)
    return {"runs": outs,
            "verdict": ("OPERATIONAL — diff report generated for "
                        f"{len(outs)} snapshot dates" if ok else
                        "FAIL — diff report errored; see runs")}


# ---------------------------------------------------------------- Track 4

def _ex3_apply(net, n_legs, trigger, scale, reentry, cb=60):
    """Own-equity DD exposure state machine on a per-rebalance net series.
    Exposure switch costs n_legs*(cb/1e4)*|d_exp|/2 (one-way trade of the
    scaled fraction)."""
    eq, peak, exp = 1.0, 1.0, 1.0
    out, exps = [], []
    for r in net:
        rr = exp * r
        out.append(rr)
        exps.append(exp)
        eq *= (1 + rr)
        peak = max(peak, eq)
        dd = eq / peak - 1
        new_exp = exp
        if exp == 1.0 and dd <= trigger:
            new_exp = scale
        elif exp < 1.0 and dd >= reentry:
            new_exp = 1.0
        if new_exp != exp:
            out[-1] -= n_legs * (cb / 1e4) * abs(new_exp - exp) / 2
        exp = new_exp
    return np.array(out), exps


def b2():
    """EX3 own-equity DD exposure gate — monitor-only evaluation."""
    cells = {"primary_t10_s50": (-0.10, 0.5, -0.05),
             "sens_t8_s50": (-0.08, 0.5, -0.03),
             "sens_t12_s50": (-0.12, 0.5, -0.07),
             "sens_t10_s70": (-0.10, 0.7, -0.05)}
    out = {"cells": {}, "monitor_only": True}
    for win in ALL_WINDOWS:
        out["cells"][win] = {}
        for spec in ALL_SPECS:
            r = run_book(win, spec, "long_short", keep_series=True)
            net, _, _ = _net60_series(r)
            base = tp._metrics(net, 20)
            row = {"base": {"sharpe": base["sharpe"], "max_dd": base["max_dd"]}}
            for cell, (tr, sc, re) in cells.items():
                scaled, exps = _ex3_apply(net, 2, tr, sc, re)
                mm = tp._metrics(scaled, 20)
                row[cell] = {"sharpe": mm["sharpe"], "max_dd": mm["max_dd"],
                             "pct_time_scaled": round(float(np.mean(np.array(exps) < 1)), 3),
                             "switches": int(np.sum(np.diff(exps) != 0))}
            out["cells"][win][spec] = row
    # gate: primary cell on blend AND d7b — bear DD >=3pp better, CH cost <=0.05
    ok = True
    for spec in ("blend", "d7b"):
        b_br = out["cells"]["BR"][spec]
        b_ch = out["cells"]["CH"][spec]
        dd_gain = b_br["base"]["max_dd"] - b_br["primary_t10_s50"]["max_dd"]  # more negative base
        ch_cost = b_ch["base"]["sharpe"] - b_ch["primary_t10_s50"]["sharpe"]
        if not (dd_gain <= -0.03 and ch_cost <= 0.05):
            ok = False
    out["verdict"] = ("ADOPT AS ARMED MONITOR — primary cell cuts bear DD >=3pp "
                      "at <=0.05 CH Sharpe cost on both deployment books"
                      if ok else
                      "REJECT — EX3 primary cell fails the pre-registered bar; "
                      "DISARM EX3 (exposure-gate thread closed with numbers)")
    return out


def b3():
    """Transformer bottom-quintile short sleeve on the LO book (monitor-only).
    Basket return = LO(tf) - LS(tf) on identical rebalance dates."""
    SLEEVE = 0.15
    out = {"sleeve_gross": SLEEVE, "monitor_only": True, "books": {}}
    for win in ALL_WINDOWS:
        r_lo_tf = run_book(win, "tf", "long_only", keep_series=True)
        r_ls_tf = run_book(win, "tf", "long_short", keep_series=True)
        n_lo_tf, t_lo_tf, d_tf = _net60_series(r_lo_tf)
        n_ls_tf, t_ls_tf, d_tf2 = _net60_series(r_ls_tf)
        if d_tf != d_tf2:
            out["books"][win] = {"error": "tf LO/LS date mismatch"}
            continue
        basket = n_lo_tf - n_ls_tf  # short-basket return (positive = shorts rose)
        sleeve_cost = (60 / 1e4) * SLEEVE * np.array(r_ls_tf["turnover"])
        out["books"][win] = {}
        for spec in ALL_SPECS:
            r_lo = run_book(win, spec, "long_only", keep_series=True)
            n_lo, _, d_lo = _net60_series(r_lo)
            if d_lo != d_tf:
                out["books"][win][spec] = {"error": "date mismatch"}
                continue
            base = tp._metrics(n_lo, 20)
            always = n_lo - SLEEVE * basket - sleeve_cost
            m_always = tp._metrics(always, 20)
            # triggered: EX3 primary state machine on the base LO equity
            _, exps = _ex3_apply(n_lo, 1, -0.10, 0.5, -0.05, cb=0)
            trig = np.where(np.array(exps) < 1.0,
                            n_lo - SLEEVE * basket - sleeve_cost, n_lo)
            m_trig = tp._metrics(trig, 20)
            out["books"][win][spec] = {
                "base": {"sharpe": base["sharpe"], "max_dd": base["max_dd"]},
                "always_on": {"sharpe": m_always["sharpe"], "max_dd": m_always["max_dd"]},
                "triggered": {"sharpe": m_trig["sharpe"], "max_dd": m_trig["max_dd"],
                              "pct_time_active": round(float(np.mean(np.array(exps) < 1)), 3)}}
    ok = True
    for spec in ("blend", "d7b"):
        br, ch = out["books"]["BR"][spec], out["books"]["CH"][spec]
        dd_gain = br["base"]["max_dd"] - br["triggered"]["max_dd"]
        ch_cost = ch["base"]["sharpe"] - ch["triggered"]["sharpe"]
        if not (dd_gain <= -0.03 and ch_cost <= 0.10):
            ok = False
    out["verdict"] = ("CANDIDATE MONITOR — triggered sleeve clears bear-DD/-cost "
                      "bar on both deployment books (compare vs B2 before arming)"
                      if ok else
                      "REJECT — triggered tf hedge sleeve fails the DD/cost bar; "
                      "LO crash protection stays with (or without) B2 descaling")
    return out


def b4():
    """Regime-conditional D1.2 downweight — PRE-DECLARED EXPECTED REJECT.
    Single pre-registered rule: universe proxy < 126d MA -> blend 70/30 tf/mom."""
    hostile = hostile_dates()
    out = {"rule": "proxy<MA126 -> 0.7*z_tf+0.3*z_mom else 50/50",
           "expected": "REJECT", "books": {}}

    def cond_score(m):
        h = m["date"].map(hostile).fillna(False).to_numpy()
        s = np.where(h, 0.7 * m["z_tf"] + 0.3 * m["z_mom"],
                     0.5 * m["z_tf"] + 0.5 * m["z_mom"])
        return m.assign(score=s)

    for win in ALL_WINDOWS:
        out["books"][win] = {}
        for spec in ("blend", "d7b"):
            band, cap = SPECS[spec]["band"], SPECS[spec]["cap"]
            stat = run_book(win, spec, "long_short")
            dyn = run_book(win, spec, "long_short", band=band, name_cap=cap,
                           score_override=cond_score)
            out["books"][win][spec] = {
                "static": {"sharpe": stat["net60"]["sharpe"],
                           "yr2022": stat["yearly_net60"].get(2022, {}).get("sharpe")},
                "regime": {"sharpe": dyn["net60"]["sharpe"],
                           "yr2022": dyn["yearly_net60"].get(2022, {}).get("sharpe")}}
        for spec in ("tf", "d12"):  # references, weight rule not applicable
            r = run_book(win, spec, "long_short")
            out["books"][win][spec] = {"static": {"sharpe": r["net60"]["sharpe"],
                                                  "yr2022": r["yearly_net60"].get(2022, {}).get("sharpe")}}
    ok = True
    for spec in ("blend", "d7b"):
        for w in GATE_WINDOWS:
            b = out["books"][w][spec]
            if b["regime"]["sharpe"] < b["static"]["sharpe"] + 0.10:
                ok = False
        y_s = out["books"]["BR"][spec]["static"]["yr2022"] or 0
        y_r = out["books"]["BR"][spec]["regime"]["yr2022"] or 0
        if y_r < y_s + 0.20:
            ok = False
    out["verdict"] = ("SURPRISE PASS — regime downweight clears the high bar "
                      "(both windows +0.1, 2022 +0.2); escalate to user"
                      if ok else
                      "REJECT (as pre-declared) — adaptive-weight line stays "
                      "closed, now with book-level evidence")
    return out


# ---------------------------------------------------------------- Track 5

def r3():
    """Universe bootstrap (drop 20% of names) on all four books, three windows.
    200 draws for deployment books (blend, d7b); 100 for references (tf, d12)
    — reference cap logged, not silent."""
    rng = np.random.default_rng(0)
    out = {"draws": {"blend": 200, "d7b": 200, "tf": 100, "d12": 100},
           "note": "reference books capped at 100 draws (cost); same seed base",
           "windows": {}}
    for win in ALL_WINDOWS:
        m = get_merged(win)
        stocks = m["stock"].unique()
        keeps = [rng.choice(stocks, size=int(len(stocks) * 0.8), replace=False)
                 for _ in range(200)]
        out["windows"][win] = {}
        for spec in ALL_SPECS:
            df_full = with_score(m, spec)
            band, cap = SPECS[spec]["band"], SPECS[spec]["cap"]

            def s(df):
                from queue_v9_lib import caps
                with caps(cap):
                    return tp.backtest_scores(df, holding=20, mode="long_short",
                                              no_trade_band=band)["net60"]["sharpe"]
            base = s(df_full)
            draws = []
            for i in range(out["draws"][spec]):
                draws.append(s(df_full[df_full["stock"].isin(keeps[i])]))
            d = np.array(draws)
            out["windows"][win][spec] = {
                "base": base, "p5": round(float(np.percentile(d, 5)), 3),
                "p50": round(float(np.percentile(d, 50)), 3),
                "p95": round(float(np.percentile(d, 95)), 3),
                "positive_frac": round(float((d > 0).mean()), 3)}
            print(f"[r3] {win}/{spec}: base {base} p5 {out['windows'][win][spec]['p5']} "
                  f"p50 {out['windows'][win][spec]['p50']}", flush=True)
    d7b_ch, d7b_br = out["windows"]["CH"]["d7b"], out["windows"]["BR"]["d7b"]
    ok = (d7b_ch["positive_frac"] == 1.0 and d7b_br["positive_frac"] == 1.0
          and d7b_ch["p5"] >= 1.5 and d7b_br["p5"] >= 1.1)
    out["verdict"] = ("PASS — deployment spec bootstrap profile matches R2 "
                      "(100% positive, p5 thresholds met)" if ok else
                      "FLAG — deployment spec bootstrap weaker than R2 profile; "
                      "revisit spec choice vs grid")
    return out


def r4():
    """Sector concentration stress: hard sector caps 30%/20% + drop-largest."""
    from data import SECTOR_MAP
    out = {"cells": {}}
    m_ch = get_merged("CH")
    sec_counts = pd.Series([SECTOR_MAP.get(s, "other")
                            for s in m_ch["stock"].unique()]).value_counts()
    largest = sec_counts.index[0]
    out["largest_sector"] = {"name": str(largest), "n_names": int(sec_counts.iloc[0])}
    drop_names = [s for s in m_ch["stock"].unique()
                  if SECTOR_MAP.get(s, "other") == largest]
    for win in GATE_WINDOWS:
        out["cells"][win] = {}
        for spec in ALL_SPECS:
            row = {}
            for mode in ("long_short", "long_only"):
                base = run_book(win, spec, mode)
                cell = {"base": base["net60"]["sharpe"]}
                for sc in (0.30, 0.20):
                    r = run_book(win, spec, mode, sector_cap=sc)
                    cell[f"seccap{int(sc*100)}"] = r["net60"]["sharpe"]
                r = run_book(win, spec, mode, exclude=drop_names)
                cell["drop_largest"] = r["net60"]["sharpe"]
                row[mode] = cell
            out["cells"][win][spec] = row
    costs = [out["cells"][w][s][m]["base"] - out["cells"][w][s][m]["seccap30"]
             for w in GATE_WINDOWS for s in ("blend", "d7b")
             for m in ("long_short", "long_only")]
    worst = round(max(costs), 3)
    out["verdict"] = (f"concentration premium priced: worst 30%-cap cost {worst} "
                      + ("<= 0.3 — cap available if needed" if worst <= 0.3 else
                         "> 0.3 — PREMIUM IS LOAD-BEARING; P2 must track realized "
                         "sector shares"))
    return out


def r5():
    """Drop-top-N contributor stress (N in {1,3,5}, ex-post worst case)."""
    out = {"windows": {}}
    flags = []
    for win in GATE_WINDOWS:
        out["windows"][win] = {}
        for spec in ALL_SPECS:
            df = with_score(get_merged(win), spec)
            band, cap = SPECS[spec]["band"], SPECS[spec]["cap"]
            contrib = {}
            for _d, w, day in holdings_walk(df, mode="long_short", band=band,
                                            name_cap=cap):
                ret = day.set_index("stock")["fwd_h"]
                for s, x in w.items():
                    if s in ret.index and np.isfinite(ret[s]):
                        contrib[s] = contrib.get(s, 0.0) + x * float(ret[s])
            cs = pd.Series(contrib).sort_values(ascending=False)
            total_pos = float(cs[cs > 0].sum())
            top_share = float(cs.iloc[0] / max(total_pos, 1e-9))
            base = run_book(win, spec, "long_short")["net60"]["sharpe"]
            row = {"base_sharpe": base,
                   "top_name": str(cs.index[0]),
                   "top_name_share_of_positive_pnl": round(top_share, 3)}
            for n in (1, 3, 5):
                r = run_book(win, spec, "long_short", exclude=list(cs.index[:n]))
                row[f"drop_top{n}"] = r["net60"]["sharpe"]
            row["retention_top5"] = round(row["drop_top5"] / base, 3) if base else None
            out["windows"][win][spec] = row
            if spec in ("blend", "d7b"):
                if top_share > 0.15:
                    flags.append(f"{win}/{spec}: top name {row['top_name']} = "
                                 f"{top_share:.0%} of positive PnL")
                if row["retention_top5"] is not None and row["retention_top5"] < 0.70:
                    flags.append(f"{win}/{spec}: drop-top5 retention "
                                 f"{row['retention_top5']} < 0.70")
    out["verdict"] = ("CONCENTRATION-RISK: " + "; ".join(flags) if flags else
                      "PASS — no name >15% of PnL; drop-top5 retains >=70% Sharpe "
                      "on deployment books (broad cross-sectional edge)")
    return out


EXPERIMENTS = {"C1": c1, "X1": x1, "X2": x2, "X3": x3, "X4": x4,
               "P1": p1, "P2": p2, "B2": b2, "B3": b3, "B4": b4,
               "R3": r3, "R4": r4, "R5": r5}
