"""v14 rule search over the daily-bar proxy day-frame (pre-registered in
rule_search_plan.md). DAILY_BAR_PROXY_ONLY — non-decision-grade.

Train 2021-2023 / Val 2024 / OOS 2025-2026-07; <=5 rules; n/t-stat/
concentration guards; OOS touched once after selection is frozen.

Usage: python research/intraday_playbook/search_playbook_rules.py
Outputs: reports/continuous_research/v14_intraday_playbook/selected_rules.json
         + section appended to backtest_results.md
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V14_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v14_intraday_playbook")
DF_P = os.path.join(ROOT, "reports", "intraday_playbook", "day_frame.csv")

MAX_RULES = 5


def _stats(r):
    n = len(r)
    if n == 0:
        return dict(n=0, mean=np.nan, t=np.nan, win=np.nan)
    return dict(n=n, mean=float(r.mean()),
                t=float(r.mean() / (r.std() / np.sqrt(n) + 1e-12)),
                win=float((r > 0).mean()))


def main():
    d = pd.read_csv(DF_P, parse_dates=["date"])
    d["gap_bin"] = d["gap_bin"].astype(str)
    tr = d[d["date"] <= "2023-12-31"]
    va = d[(d["date"] >= "2024-01-01") & (d["date"] <= "2024-12-31")]
    oo = d[d["date"] >= "2025-01-01"]
    print(f"[search] train {len(tr):,} / val {len(va):,} / oos {len(oo):,} "
          "symbol-days")

    cands = []
    for action in ("TOP_Q", "WATCH_BAND", "BOTTOM_Q"):
        col = "net_short" if action == "BOTTOM_Q" else "net_long"
        direction = "short_diagnostic" if action == "BOTTOM_Q" else "long"
        for gb in tr["gap_bin"].dropna().unique():
            trc = tr[(tr["action"] == action) & (tr["gap_bin"] == gb)]
            s = _stats(trc[col])
            if s["n"] < 100 or abs(s["t"]) < 2 or s["mean"] <= 0:
                continue
            # concentration guard: no symbol > 30% of train PnL
            pnl = trc.groupby("stock")[col].sum()
            pos = pnl[pnl > 0]
            if len(pos) and pos.max() / max(pnl.sum(), 1e-9) > 0.30:
                continue
            vac = va[(va["action"] == action) & (va["gap_bin"] == gb)]
            v = _stats(vac[col])
            if v["n"] < 30 or v["mean"] <= 0:
                continue
            cands.append({"action": action, "gap_bin": gb,
                          "direction": direction, "train": s, "val": v})
    cands = sorted(cands, key=lambda c: -abs(c["train"]["t"]))[:MAX_RULES]
    print(f"[search] {len(cands)} rule(s) survive train+val gates "
          f"(cap {MAX_RULES})")

    # OOS evaluated ONCE for the frozen selection
    for c in cands:
        col = "net_short" if c["action"] == "BOTTOM_Q" else "net_long"
        ooc = oo[(oo["action"] == c["action"])
                 & (oo["gap_bin"] == c["gap_bin"])]
        c["oos"] = _stats(ooc[col])
        c["oos_survives"] = bool(c["oos"]["n"] >= 20
                                 and (c["oos"]["mean"] or 0) > 0)

    out = {"data_quality": "DAILY_BAR_PROXY_ONLY",
           "live_trading_allowed": False,
           "plan": "rule_search_plan.md (pre-registered)",
           "splits": {"train": "2021..2023", "val": "2024",
                      "oos": "2025..2026-07"},
           "rules": cands}
    with open(os.path.join(V14_DIR, "selected_rules.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = ["", "## Rule search result (pre-registered splits; "
          "DAILY_BAR_PROXY_ONLY)", ""]
    if not cands:
        md.append("**No cell survived the train+validation gates** — the "
                  "conditional day-trade proxy shows no robust exploitable "
                  "open-to-close edge after costs at the pre-registered "
                  "bars.")
    else:
        md += ["| action | gap_bin | dir | train n/mean/t | val n/mean | "
               "OOS n/mean | OOS survives |", "|---|---|---|---|---|---|---|"]
        for c in cands:
            md.append(
                f"| {c['action']} | {c['gap_bin']} | {c['direction']} | "
                f"{c['train']['n']}/{c['train']['mean']:+.3%}/"
                f"{c['train']['t']:+.1f} | "
                f"{c['val']['n']}/{c['val']['mean']:+.3%} | "
                f"{c['oos']['n']}/{c['oos']['mean']:+.3%} | "
                f"{'YES' if c['oos_survives'] else 'NO'} |")
        md.append("")
        md.append("OOS survival here is proxy-grade evidence ONLY — it "
                  "cannot clear the Task-8 gates (fills/paths unmodeled).")
    with open(os.path.join(V14_DIR, "backtest_results.md"), "a",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    for c in cands:
        print(f"  {c['action']:10s} {c['gap_bin']:7s} {c['direction']:16s} "
              f"train t {c['train']['t']:+.1f} val {c['val']['mean']:+.3%} "
              f"oos {c['oos']['mean']:+.3%} survives={c['oos_survives']}")
    print("[search] -> selected_rules.json")


if __name__ == "__main__":
    main()
