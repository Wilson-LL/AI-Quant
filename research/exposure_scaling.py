"""Cycle 4 (C3): walk-forward exposure scaling on the blend50+band10 book.

Variants EX1/EX2/EX3 per C3_preregistration.md. CPU-only, both panels.
Output: reports/continuous_research/C3_results.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_hybrid import load_panel, merged  # noqa: E402
from transformer_portfolio import backtest_scores, _metrics  # noqa: E402
from adaptive_blend import market_index, combo_metrics  # noqa: E402

OUT = os.path.join(ROOT, "reports", "continuous_research", "C3_results.json")
HOLD = 20


def run_panel(panel_name, results):
    panel, pname = load_panel(panel_name)
    m = merged(panel)
    m["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    idx, mkt_dd = market_index()
    lr = np.log(idx / idx.shift(1))
    mkt_vol = (lr.rolling(60).std() * np.sqrt(252)).dropna()

    res = {}
    for mode in ("long_short", "long_only"):
        r = backtest_scores(m, holding=HOLD, mode=mode, no_trade_band=0.10)
        g, t = np.array(r["gross"]), np.array(r["turnover"])
        dates = pd.to_datetime(pd.Index(r["dates"]))
        n_legs = 2 if mode == "long_short" else 1
        net60 = g - n_legs * 0.006 * t

        res[f"{mode}_base"] = combo_metrics(g, t, dates, n_legs, f"{pname[:10]} {mode[:2]} base")

        def exposures(variant):
            es = []
            eq_high, eq = 1.0, 1.0
            for i, d in enumerate(dates):
                e = 1.0
                if variant == "EX1":
                    dd_now = mkt_dd.loc[:d]
                    e = 0.5 if (len(dd_now) and dd_now.iloc[-1] < -0.10) else 1.0
                elif variant == "EX2":
                    v = mkt_vol.loc[:d]
                    e = float(np.clip(0.15 / max(v.iloc[-1], 1e-9), 0.3, 1.0)) if len(v) else 1.0
                elif variant == "EX3":
                    # matured own equity through rebalance i-2 (unscaled base returns)
                    past = net60[:max(i - 1, 0)]
                    if len(past) >= 3:
                        eqc = np.cumprod(1 + past)
                        e = 0.5 if eqc[-1] / eqc.max() - 1 < -0.10 else 1.0
                es.append(e)
            return np.array(es)

        for variant in ("EX1", "EX2", "EX3"):
            e = exposures(variant)
            gg = e * g
            tt = e * t + 0.5 * np.abs(np.diff(e, prepend=e[0]))
            rr = combo_metrics(gg, tt, dates, n_legs, f"{pname[:10]} {mode[:2]} {variant}")
            rr["mean_exposure"] = round(float(e.mean()), 3)
            res[f"{mode}_{variant}"] = rr
    results[pname] = res
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    results = {}
    run_panel("G9_presetB_equal_all", results)
    run_panel("BEAR_presetB_2021", results)
    print("\nC3 done ->", OUT)
