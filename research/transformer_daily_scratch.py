"""G3 supplement: daily retrain FROM SCRATCH (fresh init every trading day),
sampled over the most recent ~6 OOS months (full-history OOS at daily-scratch
cost would be ~3 h GPU for statistically weak Sharpe estimates; this sample
gives an IC-level and timing comparison vs frozen/monthly/daily-warm).

Usage: python research/transformer_daily_scratch.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from dataset_transformer_eod import build_dataset  # noqa: E402
from train_transformer_eod import walkforward, to_gpu, require_cuda, PRESETS  # noqa: E402
from transformer_experiments import eval_panel, PANEL_DIR  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")

# champion architecture but capped epochs so a from-scratch fit per day stays ~30s
PRESETS["Bd"] = dict(PRESETS["B"], max_epochs=10, patience=2)


def main():
    require_cuda()
    data = build_dataset("close_only", seq_len=60, horizons=(20,))
    Xg = to_gpu(data)
    panel, info = walkforward(data, Xg, target="tgt_rank_20", horizon=20,
                              preset="Bd", oos_start="2026-01-01",
                              refit_every=1, seeds=[0], recency=None,
                              cadence="refit", log_prefix="[scratch] ")
    panel.to_csv(os.path.join(PANEL_DIR, "G3_daily_scratch_2026.csv.gz"),
                 index=False, compression="gzip")
    perf = eval_panel(panel, 20, "G3_daily_scratch_2026")
    fits = info["fits"]
    out = {
        "run_info": {k: v for k, v in info.items() if k != "fits"},
        "mean_fit_s": round(float(np.mean([f["train_s"] for f in fits])), 1),
        "n_fits": len(fits),
        "mean_val_ic": info["mean_val_ic"],
        "perf": perf,
    }
    with open(os.path.join(REPORT_DIR, "G3_daily_scratch_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
