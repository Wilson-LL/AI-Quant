"""v14 daily-bar PROXY backtest — gap-bin x EOD-state -> open-to-close.

NON-DECISION-GRADE by construction (see daily_bar_proxy_limitations.md):
no time axis, path-ambiguous ranges, assumed costs, survivorship-biased
universe. Every output is labeled DAILY_BAR_PROXY_ONLY. Research only; no
orders.

Builds the day-frame (signal = previous session's blend z from the frozen
A8 BR panel 2021->2026; outcome = next session's gap and open-to-close),
writes cell statistics and risk envelopes.

Usage: python research/intraday_playbook/backtest_conditional_playbook.py
Outputs:
  reports/continuous_research/v14_intraday_playbook/backtest_results.csv
  reports/continuous_research/v14_intraday_playbook/backtest_results.md
  reports/intraday_playbook/day_frame.csv   (gitignored, reused by search)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V14_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v14_intraday_playbook")
PB_DIR = os.path.join(ROOT, "reports", "intraday_playbook")

GAP_EDGES = [-np.inf, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03,
             0.05, np.inf]
GAP_LABELS = ["<=-5", "-5..-3", "-3..-2", "-2..-1", "-1..0", "0..1",
              "1..2", "2..3", "3..5", ">=5"]
COST_LONG = 0.0030    # assumed 30bps day-trade round trip (NOT calibrated)
COST_SHORT = 0.0060   # 2x on the short-diagnostic side


def pseudo_action(pct):
    if pct >= 0.80:
        return "TOP_Q"
    if pct >= 0.70:
        return "WATCH_BAND"
    if pct <= 0.20:
        return "BOTTOM_Q"
    return "MID"


def build_day_frame():
    from queue_v9_lib import get_merged
    from transformer_hybrid import _cache_frames
    m = get_merged("BR").copy()
    m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    m["pct"] = m.groupby("date")["blend"].rank(pct=True)
    sig = m[["date", "stock", "blend", "pct"]]

    rows = []
    for sid, df in _cache_frames().items():
        df = (df.sort_values("date").drop_duplicates("date", keep="last")
                .reset_index(drop=True))
        o = df["open"].to_numpy(float)
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        pc = np.concatenate([[np.nan], c[:-1]])
        rows.append(pd.DataFrame({
            "date": df["date"].values, "stock": sid,
            "gap": o / pc - 1.0, "oc_ret": c / o - 1.0,
            "hi_ret": h / o - 1.0, "lo_ret": l / o - 1.0}))
    ohlc = pd.concat(rows, ignore_index=True)

    d = ohlc.merge(sig, on=["date", "stock"], how="inner").sort_values(
        ["stock", "date"])
    # decision info at day t's open = t-1 score/pct + t's gap (no lookahead)
    d["prev_pct"] = d.groupby("stock")["pct"].shift(1)
    d["prev_blend"] = d.groupby("stock")["blend"].shift(1)
    d = d.dropna(subset=["prev_pct", "gap", "oc_ret"]).copy()
    d["action"] = d["prev_pct"].map(pseudo_action)
    d["gap_bin"] = pd.cut(d["gap"], GAP_EDGES, labels=GAP_LABELS)
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d["net_long"] = d["oc_ret"] - COST_LONG
    d["net_short"] = -d["oc_ret"] - COST_SHORT
    return d


def cell_stats(d, net_col):
    def agg(g):
        r = g[net_col]
        wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
        return pd.Series({
            "n": len(g), "mean_net": r.mean(), "median_net": r.median(),
            "win_rate": (r > 0).mean(),
            "profit_factor": wins / losses if losses > 0 else np.inf,
            "t_stat": r.mean() / (r.std() / np.sqrt(len(g)) + 1e-12),
            "p5": r.quantile(0.05), "p95": r.quantile(0.95),
            "worst": r.min(),
            "mean_hi": g["hi_ret"].mean(), "mean_lo": g["lo_ret"].mean()})
    return (d.groupby(["action", "gap_bin"], observed=True)
             .apply(agg, include_groups=False).reset_index())


def main():
    os.makedirs(V14_DIR, exist_ok=True)
    os.makedirs(PB_DIR, exist_ok=True)
    d = build_day_frame()
    d.to_csv(os.path.join(PB_DIR, "day_frame.csv"), index=False)
    print(f"[proxy] day-frame: {len(d):,} symbol-days "
          f"{d['date'].min():%Y-%m-%d}..{d['date'].max():%Y-%m-%d}")

    long_cells = cell_stats(d, "net_long").assign(direction="long")
    short_cells = cell_stats(d[d["action"] == "BOTTOM_Q"], "net_short") \
        .assign(direction="short_diagnostic")
    cells = pd.concat([long_cells, short_cells], ignore_index=True)
    cells["data_quality"] = "DAILY_BAR_PROXY_ONLY"
    cells["live_trading_allowed"] = False
    cells.round(5).to_csv(os.path.join(V14_DIR, "backtest_results.csv"),
                          index=False)

    base = d["net_long"]
    md = ["# v14 daily-bar proxy backtest (NON-DECISION-GRADE)", "",
          "**DAILY_BAR_PROXY_ONLY — path-blind, assumed costs (30/60 bps "
          "round trip), survivorship-biased. Cannot validate time-of-day, "
          "stops, or fills. See daily_bar_proxy_limitations.md.**", "",
          f"Day-frame: {len(d):,} symbol-days, "
          f"{d['date'].min():%Y-%m-%d} → {d['date'].max():%Y-%m-%d}; "
          f"signal = previous session blend z (frozen OOS panel); outcome "
          f"= open→close net of assumed costs.", "",
          f"All-days long baseline: mean net {base.mean():+.4%}, win rate "
          f"{(base > 0).mean():.1%} (day-trading the open-to-close with "
          "costs is NEGATIVE on average — the proxy edge, if any, must "
          "come from conditioning).", "",
          "## Notable long cells (n>=100, |t|>=2)", ""]
    strong = long_cells[(long_cells["n"] >= 100)
                        & (long_cells["t_stat"].abs() >= 2)]
    md += ["| action | gap_bin | n | mean net | win | t | p5 | p95 |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in strong.sort_values("t_stat", ascending=False).iterrows():
        md.append(f"| {r['action']} | {r['gap_bin']} | {r['n']:.0f} | "
                  f"{r['mean_net']:+.3%} | {r['win_rate']:.0%} | "
                  f"{r['t_stat']:+.1f} | {r['p5']:+.2%} | {r['p95']:+.2%} |")
    md += ["", "## Short-diagnostic cells (BOTTOM_Q, 2x costs, n>=100, |t|>=2)",
           "", "| gap_bin | n | mean net | win | t |", "|---|---|---|---|---|"]
    ss = short_cells[(short_cells["n"] >= 100)
                     & (short_cells["t_stat"].abs() >= 2)]
    for _, r in ss.sort_values("t_stat", ascending=False).iterrows():
        md.append(f"| {r['gap_bin']} | {r['n']:.0f} | {r['mean_net']:+.3%} | "
                  f"{r['win_rate']:.0%} | {r['t_stat']:+.1f} |")
    md += ["", "## Risk envelopes by gap bin (descriptive only)", "",
           "| gap_bin | n | mean hi from open | mean lo from open | worst oc |",
           "|---|---|---|---|---|"]
    env = d.groupby("gap_bin", observed=True).agg(
        n=("oc_ret", "size"), hi=("hi_ret", "mean"), lo=("lo_ret", "mean"),
        worst=("oc_ret", "min"))
    for gb, r in env.iterrows():
        md.append(f"| {gb} | {r['n']:.0f} | {r['hi']:+.2%} | {r['lo']:+.2%} |"
                  f" {r['worst']:+.1%} |")
    md += ["", "Full cell table: backtest_results.csv. Rule selection with "
           "train/val/OOS splits: search_playbook_rules.py output."]
    with open(os.path.join(V14_DIR, "backtest_results.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[proxy] {len(strong)} notable long cells, {len(ss)} short cells "
          f"-> backtest_results.{{csv,md}}")


if __name__ == "__main__":
    main()
