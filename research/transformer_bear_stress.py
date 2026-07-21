"""Bear-regime stress: champion config walk-forward with OOS extended back to
2021-01 so the OOS window contains the 2022 momentum crash (the known worst
regime of the D1.1/D1.2 lineage). Training data before the first refit is only
2018-2020 (thinner than the main run) — that is the honest price of the test.

Usage: python research/transformer_bear_stress.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from dataset_transformer_eod import build_dataset  # noqa: E402
from train_transformer_eod import walkforward, to_gpu, require_cuda  # noqa: E402
from transformer_experiments import eval_panel, PANEL_DIR  # noqa: E402
from transformer_hybrid import merged  # noqa: E402
from transformer_portfolio import backtest_scores  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")


def main():
    require_cuda()
    data = build_dataset("close_only", seq_len=60, horizons=(20,))
    Xg = to_gpu(data)
    panel, info = walkforward(data, Xg, target="tgt_rank_20", horizon=20,
                              preset="B", oos_start="2021-01-01",
                              refit_every=126, seeds=[0, 1, 2, 3, 4],
                              recency=None, log_prefix="[bear] ")
    panel.to_csv(os.path.join(PANEL_DIR, "BEAR_presetB_2021.csv.gz"),
                 index=False, compression="gzip")
    m = merged(panel)
    out = {"run_info": {k: v for k, v in info.items() if k != "fits"}}
    out["tf"] = {}
    out["d12"] = {}
    for label, col in [("tf", "score"), ("d12", "z_mom")]:
        df = m.copy()
        df["score"] = df[col]
        for mode in ("long_short", "long_only"):
            r = backtest_scores(df, holding=20, mode=mode)
            out[label][mode] = {
                "ic": round(r["rank_ic"], 4),
                "net60": r["net60"], "net100": r["net100"],
                "yearly_net60": r["yearly_net60"],
                "turnover": round(r["avg_turnover"], 3),
                "max_w": r["max_name_weight"],
            }
    # 50/50 blend
    df = m.copy()
    df["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    out["blend50"] = {}
    for mode in ("long_short", "long_only"):
        r = backtest_scores(df, holding=20, mode=mode)
        out["blend50"][mode] = {"ic": round(r["rank_ic"], 4), "net60": r["net60"],
                                "yearly_net60": r["yearly_net60"]}
    with open(os.path.join(REPORT_DIR, "BEAR_stress_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for label in ("tf", "d12", "blend50"):
        ls = out[label]["long_short"]
        print(f"{label:8s} LS net60 {ls['net60']['sharpe']:5.2f} "
              f"yearly {json.dumps(ls['yearly_net60'])}")
    print("done")


if __name__ == "__main__":
    main()
