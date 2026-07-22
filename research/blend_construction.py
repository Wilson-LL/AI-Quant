"""Cycle 2 (D1): construction sweep on the 50/50 TF+D1.2 score-blend book.

Grid pre-declared in A1_D1_preregistration.md. Runs on BOTH the champion panel
(2023-26) and the bear panel (2021-26); improvements must hold on both.
CPU-only. Output: reports/continuous_research/D1_results.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_hybrid import load_panel, merged  # noqa: E402
from transformer_portfolio import backtest_scores, summarize  # noqa: E402

OUT = os.path.join(ROOT, "reports", "continuous_research", "D1_results.json")


def sweep(panel_name, results):
    panel, pname = load_panel(panel_name)
    m = merged(panel)
    m["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    res = {}
    for mode in ("long_short", "long_only"):
        for weighting in ("equal", "invvol"):
            for band in (0.0, 0.05, 0.10):
                key = f"{mode}_{weighting}_band{int(band*100)}"
                r = backtest_scores(m, holding=20, mode=mode, weighting=weighting,
                                    no_trade_band=band)
                print(summarize(r, f"{pname[:14]} {key}"))
                for kk in ("gross", "turnover", "dates"):
                    r.pop(kk, None)
                res[key] = r
                results[pname] = res
                with open(OUT, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
    return res


if __name__ == "__main__":
    results = {}
    sweep("G9_presetB_equal_all", results)
    sweep("BEAR_presetB_2021", results)
    print("\nD1 done ->", OUT)
