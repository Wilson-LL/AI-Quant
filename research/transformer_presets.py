"""Preset comparison (brief §3): A (h64/seq40), B (h64/seq60/5 seeds),
C (h128/seq60) at the champion configuration (close-only, equal-weight history,
20d rank target). Reports params, train/inference time, VRAM, OOS perf, and
whether each fits the 12h daily budget.

Usage: python research/transformer_presets.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from dataset_transformer_eod import build_dataset  # noqa: E402
from train_transformer_eod import (walkforward, to_gpu, require_cuda, build_net,
                                   param_count, PRESETS, DEVICE)  # noqa: E402
from transformer_experiments import eval_panel, PANEL_DIR  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")


def main():
    require_cuda()
    out_path = os.path.join(REPORT_DIR, "G9_presets_results.json")
    results = {}
    for preset in ("A", "B", "C"):
        cfg = PRESETS[preset]
        data = build_dataset("close_only", seq_len=cfg["seq_len"], horizons=(20,))
        Xg = to_gpu(data)
        torch.cuda.reset_peak_memory_stats()
        name = f"G9_preset{preset}_equal_all"
        panel, info = walkforward(data, Xg, target="tgt_rank_20", horizon=20,
                                  preset=preset, oos_start="2023-01-01",
                                  refit_every=126, seeds=list(range(cfg["seeds"])),
                                  recency=None, log_prefix=f"[{preset}] ")
        panel.to_csv(os.path.join(PANEL_DIR, f"{name}.csv.gz"), index=False,
                     compression="gzip")
        perf = eval_panel(panel, 20, name)
        net = build_net(Xg.shape[2], cfg).to(DEVICE)
        # inference timing on a full cross-section
        n_stocks = len(data["stocks"])
        xb = Xg[:n_stocks]
        net.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                with torch.amp.autocast("cuda"):
                    net(xb)
            torch.cuda.synchronize()
        infer_ms = (time.time() - t0) / 50 * 1000
        fits = info["fits"]
        mean_fit_s = float(np.mean([f["train_s"] for f in fits]))
        daily_train_s = mean_fit_s * cfg["seeds"]
        results[name] = {
            "params": param_count(net),
            "seq_len": cfg["seq_len"], "hidden": cfg["hidden"],
            "seeds": cfg["seeds"],
            "mean_fit_s": round(mean_fit_s, 1),
            "daily_retrain_est_s": round(daily_train_s, 1),
            "inference_full_universe_ms": round(infer_ms, 1),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
            "fits_12h_budget": daily_train_s < 12 * 3600,
            "mean_val_ic": info["mean_val_ic"],
            "perf": perf,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        del Xg
        torch.cuda.empty_cache()
    print("\nG9 presets done ->", out_path)


if __name__ == "__main__":
    main()
