"""Uniform re-evaluation of every saved OOS score panel for the final report.

All panels are re-backtested with identical settings (min_names=60 to exclude
the thin boundary cross-section, quintile books, costs 0/60/100/150), together
with the D1.2 momentum baseline evaluated on each panel's exact dates, and the
50/50 blend. Output: reports/transformer_gpu/CONSOLIDATED.json + .md table.

Usage: python research/consolidate_results.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_hybrid import merged, load_panel  # noqa: E402
from transformer_portfolio import backtest_scores  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")
PANEL_DIR = os.path.join(REPORT_DIR, "panels")


def holding_of(name):
    if "rank_5" in name:
        return 5
    if "rank_10" in name:
        return 10
    return 20


def row(panel_df, name, holding, score_col="score", ret_col=None):
    ret_col = ret_col or ("fwd_h" if "fwd_h" in panel_df else "fwd_20")
    df = panel_df.copy()
    if score_col != "score":
        df["score"] = df[score_col]
    ls = backtest_scores(df, holding=holding, mode="long_short", ret_col=ret_col)
    lo = backtest_scores(df, holding=holding, mode="long_only", ret_col=ret_col)
    return {
        "name": name, "holding": holding,
        "ic": round(ls["rank_ic"], 4), "ic_ir": round(ls["ic_ir"], 3),
        "ls_net0": ls["net0"]["sharpe"], "ls_net60": ls["net60"]["sharpe"],
        "ls_net100": ls["net100"]["sharpe"], "ls_net150": ls["net150"]["sharpe"],
        "ls_ann60": ls["net60"]["ann_ret"], "ls_dd60": ls["net60"]["max_dd"],
        "ls_calmar": ls["net60"]["calmar"],
        "lo_net60": lo["net60"]["sharpe"], "lo_net100": lo["net100"]["sharpe"],
        "lo_ann60": lo["net60"]["ann_ret"], "lo_dd60": lo["net60"]["max_dd"],
        "turnover": round(ls["avg_turnover"], 3),
        "max_w": ls["max_name_weight"],
        "hit60": ls["net60"]["hit"], "n_rebal": ls["net60"]["n"],
        "yearly_ls_net60": ls["yearly_net60"],
    }


def main():
    panels = sorted(f for f in os.listdir(PANEL_DIR) if f.endswith(".csv.gz"))
    rows = []
    champion = None
    for f in panels:
        name = f.replace(".csv.gz", "")
        panel, _ = load_panel(f)
        h = holding_of(name)
        m = merged(panel) if h == 20 else panel
        r = row(m, name, h)
        rows.append(r)
        print(f"{name:28s} IC {r['ic']:+.4f} LS60 {r['ls_net60']:5.2f} LO60 {r['lo_net60']:5.2f}")
        if name == "G2_equal_all":
            champion = m
    # baselines + blend on champion dates
    if champion is not None:
        rows.append(row(champion, "BASE_d12_mom126_5", 20, score_col="z_mom"))
        ch = champion.copy()
        ch["score"] = 0.5 * champion["z_tf"] + 0.5 * champion["z_mom"]
        rows.append(row(ch, "CHAMP_blend50_tf_d12", 20))
        b = ch.copy()
        rows.append({"name": "CHAMP_blend50_band5_LO",
                     **{k: v for k, v in row(b, "x", 20).items() if k != "name"}})
        # blend with band via backtest kwargs
        ls = backtest_scores(ch, holding=20, mode="long_short", no_trade_band=0.05)
        lo = backtest_scores(ch, holding=20, mode="long_only", no_trade_band=0.05)
        rows[-1] = {"name": "CHAMP_blend50_band5",
                    "holding": 20, "ic": round(ls["rank_ic"], 4),
                    "ic_ir": round(ls["ic_ir"], 3),
                    "ls_net0": ls["net0"]["sharpe"], "ls_net60": ls["net60"]["sharpe"],
                    "ls_net100": ls["net100"]["sharpe"], "ls_net150": ls["net150"]["sharpe"],
                    "ls_ann60": ls["net60"]["ann_ret"], "ls_dd60": ls["net60"]["max_dd"],
                    "ls_calmar": ls["net60"]["calmar"],
                    "lo_net60": lo["net60"]["sharpe"], "lo_net100": lo["net100"]["sharpe"],
                    "lo_ann60": lo["net60"]["ann_ret"], "lo_dd60": lo["net60"]["max_dd"],
                    "turnover": round(ls["avg_turnover"], 3),
                    "max_w": ls["max_name_weight"],
                    "hit60": ls["net60"]["hit"], "n_rebal": ls["net60"]["n"],
                    "yearly_ls_net60": ls["yearly_net60"]}
        print(f"blend50: LS60 {rows[-1]['ls_net60']} LO60 {rows[-1]['lo_net60']}")

    with open(os.path.join(REPORT_DIR, "CONSOLIDATED.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md = ["# Consolidated OOS results (uniform protocol, min_names=60, quintile books)",
          "", "OOS 2023-01→2026-07 unless the panel says otherwise (G3_* = 2024-07→).",
          "",
          "| config | hold | IC | IC-IR | L/S net60 | net100 | net150 | annRet60 | maxDD | LO net60 | LO net100 | turnover | maxW |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['holding']} | {r['ic']:+.4f} | {r['ic_ir']:.2f} | "
                  f"**{r['ls_net60']:.2f}** | {r['ls_net100']:.2f} | {r['ls_net150']:.2f} | "
                  f"{r['ls_ann60']:+.1%} | {r['ls_dd60']:.1%} | {r['lo_net60']:.2f} | "
                  f"{r['lo_net100']:.2f} | {r['turnover']:.2f} | {r['max_w']:.1%} |")
    with open(os.path.join(REPORT_DIR, "CONSOLIDATED.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n->", os.path.join(REPORT_DIR, "CONSOLIDATED.md"))


if __name__ == "__main__":
    main()
