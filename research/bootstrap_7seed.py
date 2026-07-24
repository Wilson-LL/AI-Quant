"""Universe bootstrap (200 draws, drop 20% of names) for the adopted 7-seed
blend spec on both reference panels. Output: ROBUST_7seed_blend.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research"))
from transformer_hybrid import load_panel, merged  # noqa: E402
from transformer_portfolio import backtest_scores  # noqa: E402

out = {}
for pn in ("SCHED_A8_seeds7_full", "SCHED_BEAR_A8_seeds7_full"):
    panel, pname = load_panel(pn)
    m = merged(panel)
    m["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]

    def s(df):
        return backtest_scores(df, holding=20, mode="long_short",
                               no_trade_band=0.10)["net60"]["sharpe"]

    base = s(m)
    rng = np.random.default_rng(0)
    stocks = m["stock"].unique()
    draws = []
    for i in range(200):
        keep = rng.choice(stocks, size=int(len(stocks) * 0.8), replace=False)
        draws.append(s(m[m["stock"].isin(keep)]))
        if (i + 1) % 50 == 0:
            print(pname, i + 1, "/200", flush=True)
    draws = np.array(draws)
    out[pname] = {"base": base, "p5": round(float(np.percentile(draws, 5)), 3),
                  "p50": round(float(np.percentile(draws, 50)), 3),
                  "p95": round(float(np.percentile(draws, 95)), 3),
                  "positive_frac": round(float((draws > 0).mean()), 3)}
    print(pname, out[pname], flush=True)

with open("reports/continuous_research/ROBUST_7seed_blend.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("bootstrap done")
