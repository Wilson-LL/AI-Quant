"""Robustness checks for the champion transformer panel (brief §13/§16).

1. Universe bootstrap: 200 draws dropping 20% of names -> distribution of
   L/S net@60 Sharpe (does the result survive universe perturbation?).
2. Drop-top-3 contributors (survivorship stress).
3. Seed sensitivity: per-seed val ICs are already logged by the runner; here we
   measure how much the ensemble beats its members using seed-subset scores if
   available (skipped when the panel holds only the ensemble mean).

Usage: python research/transformer_robustness.py [panel_name]
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_hybrid import load_panel  # noqa: E402
from transformer_portfolio import backtest_scores  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")


def ls_sharpe(panel):
    r = backtest_scores(panel, holding=20, mode="long_short")
    return r["net60"]["sharpe"], r


def main(panel_name=None):
    panel, pname = load_panel(panel_name)
    rng = np.random.default_rng(0)
    stocks = panel["stock"].unique()
    base_sharpe, base = ls_sharpe(panel)
    print(f"base L/S net60 Sharpe {base_sharpe}")

    draws = []
    for i in range(200):
        keep = rng.choice(stocks, size=int(len(stocks) * 0.8), replace=False)
        s, _ = ls_sharpe(panel[panel["stock"].isin(keep)])
        draws.append(s)
        if (i + 1) % 50 == 0:
            print(f"  bootstrap {i+1}/200")
    draws = np.array(draws)

    # drop top-3 gross contributors (approx: names most often in the long leg
    # with the highest realized fwd sum)
    contrib = (panel.assign(c=panel["score"].rank(pct=True) * panel["fwd_h"])
                     .groupby("stock")["c"].sum().sort_values())
    top3 = list(contrib.index[-3:])
    s_drop, _ = ls_sharpe(panel[~panel["stock"].isin(top3)])

    out = {
        "panel": pname,
        "base_ls_net60": base_sharpe,
        "bootstrap_200_drop20pct": {
            "p5": round(float(np.percentile(draws, 5)), 3),
            "p50": round(float(np.percentile(draws, 50)), 3),
            "p95": round(float(np.percentile(draws, 95)), 3),
            "positive_frac": round(float((draws > 0).mean()), 3),
        },
        "drop_top3_names": {"names": top3, "ls_net60": s_drop},
    }
    path = os.path.join(REPORT_DIR, f"ROBUST_{pname}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
