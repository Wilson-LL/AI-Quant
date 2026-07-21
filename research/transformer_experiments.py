"""Sprint experiment driver (G1..G5) for the RTX 4060 Ti transformer system.

Run one experiment group per invocation (background-friendly, crash-safe:
results JSON is rewritten after every config):

    python research/transformer_experiments.py G1
    python research/transformer_experiments.py G2
    ...

Outputs: reports/transformer_gpu/<group>_results.json (+ console log).

Config-selection discipline: downstream defaults are chosen by mean VALIDATION
rank IC (never by OOS results); OOS numbers are reported for all configs.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from dataset_transformer_eod import build_dataset  # noqa: E402
from train_transformer_eod import walkforward, to_gpu, require_cuda, DEVICE  # noqa: E402
from transformer_portfolio import backtest_scores, summarize  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")
os.makedirs(REPORT_DIR, exist_ok=True)

OOS_START = "2023-01-01"
SCREEN_SEEDS = [0, 1]
CHAMP_SEEDS = [0, 1, 2]

_data_cache = {}


def get_data(feature_set, seq_len, include_barrier=False):
    key = (feature_set, seq_len, include_barrier)
    if key not in _data_cache:
        _data_cache.clear()  # keep at most one big dataset in RAM/VRAM
        torch.cuda.empty_cache()
        d = build_dataset(feature_set, seq_len=seq_len, horizons=(5, 10, 20),
                          include_barrier=include_barrier)
        _data_cache[key] = (d, to_gpu(d))
    return _data_cache[key]


def eval_panel(panel, holding, label, weighting="equal"):
    res = {}
    for mode in ("long_short", "long_only"):
        r = backtest_scores(panel, holding=holding, mode=mode, weighting=weighting)
        print(summarize(r, f"{label} [{mode}]"))
        for kk in ("gross", "turnover", "dates"):
            r.pop(kk, None)
        res[mode] = r
    return res


def baseline_panel(data, col, oos_start=OOS_START):
    """Score panel for a feature-based baseline (score = feature at window end)."""
    ci = data["feature_cols"].index(col)
    dates = data["dates"]
    oos0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(oos_start))))
    m = data["date_rank"] >= oos0
    idx = np.nonzero(m)[0]
    return pd.DataFrame({
        "date": dates[data["date_rank"][idx]],
        "stock": np.asarray(data["stocks"])[data["stock_idx"][idx]],
        "score": data["X"][idx, -1, ci],
        "fwd_h": data["targets"]["fwd_20"][idx],
        "fwd_20": data["targets"]["fwd_20"][idx],
    })


PANEL_DIR = os.path.join(REPORT_DIR, "panels")
os.makedirs(PANEL_DIR, exist_ok=True)


def run_config(data, Xg, name, results, out_path, holding=20, **wf_kwargs):
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    panel, info = walkforward(data, Xg, oos_start=wf_kwargs.pop("oos_start", OOS_START),
                              log_prefix=f"[{name}] ", **wf_kwargs)
    panel.to_csv(os.path.join(PANEL_DIR, f"{name}.csv.gz"), index=False,
                 compression="gzip")
    perf = eval_panel(panel, holding, name)
    results[name] = {"run_info": {k: v for k, v in info.items() if k != "fits"},
                     "fit_summary": {
                         "mean_val_ic": info["mean_val_ic"],
                         "mean_train_s": round(float(np.mean([f["train_s"] for f in info["fits"]])), 1),
                         "mean_epochs": round(float(np.mean([f["epochs"] for f in info["fits"]])), 1),
                     },
                     "perf": perf,
                     "wall_s": round(time.time() - t0, 1)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return panel


# ---------------------------------------------------------------------------
def g1():
    """Full-scale close-only transformer, seq 40 vs 60, vs D1.2/mom20 baselines."""
    out = os.path.join(REPORT_DIR, "G1_results.json")
    results = {}
    for seq, preset in [(40, "A"), (60, "B")]:
        data, Xg = get_data("close_only", seq)
        if f"baseline_mom126_5_seq{seq}" not in results:
            for col, label in [("mom_126_5", "mom126_5(D1.2-style)"), ("mom_20", "mom20")]:
                bp = baseline_panel(data, col)
                results[f"baseline_{col}_seq{seq}"] = {"perf": eval_panel(bp, 20, f"{label} seq{seq}")}
            # equal-weight universe benchmark: score=const -> use mom panel fwd mean
            bp = baseline_panel(data, "mom_20")
            ew = bp.groupby("date")["fwd_20"].mean().to_numpy()[::20]
            ppy = 252 / 20
            s = ew.mean() / (ew.std(ddof=1) + 1e-12) * np.sqrt(ppy)
            results[f"baseline_equal_weight_seq{seq}"] = {
                "sharpe_gross": round(float(s), 3),
                "ann_ret": round(float((1 + ew.mean()) ** ppy - 1), 4)}
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        run_config(data, Xg, f"G1_close_seq{seq}_hl126", results, out,
                   target="tgt_rank_20", horizon=20, preset=preset,
                   refit_every=126, seeds=CHAMP_SEEDS,
                   recency={"halflife": 126})
    print("\nG1 done")


def g2():
    """Recency confirmation on close-only (best seq from G1 by VAL ic)."""
    out = os.path.join(REPORT_DIR, "G2_results.json")
    g1p = os.path.join(REPORT_DIR, "G1_results.json")
    seq = 40
    if os.path.exists(g1p):
        with open(g1p, encoding="utf-8") as f:
            g1r = json.load(f)
        vals = {s: g1r.get(f"G1_close_seq{s}_hl126", {}).get("fit_summary", {}).get("mean_val_ic")
                for s in (40, 60)}
        vals = {s: v for s, v in vals.items() if v is not None}
        if vals:
            seq = max(vals, key=vals.get)
    print(f"[G2] seq_len={seq} (chosen by val IC)")
    preset = "A" if seq == 40 else "B"
    data, Xg = get_data("close_only", seq)
    configs = [
        ("equal_all", None),
        ("roll_3y", {"window": 756}),
        ("roll_1p5y", {"window": 378}),
        ("roll_1y", {"window": 252}),
        ("hl_63", {"halflife": 63}),
        ("hl_126", {"halflife": 126}),
        ("hl_252", {"halflife": 252}),
        ("hl_378", {"halflife": 378}),
        ("cap5y_hl126", {"halflife": 126, "window": 1260}),
    ]
    results = {"chosen_seq": seq}
    for name, rec in configs:
        run_config(data, Xg, f"G2_{name}", results, out,
                   target="tgt_rank_20", horizon=20, preset=preset,
                   refit_every=126, seeds=SCREEN_SEEDS, recency=rec)
    print("\nG2 done")


def _best_recency():
    """Best recency by mean VAL ic from G2 (fallback hl126)."""
    p = os.path.join(REPORT_DIR, "G2_results.json")
    default = {"halflife": 126}
    if not os.path.exists(p):
        return default, "hl_126(default)"
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    best, bic = None, -1e9
    mapping = {"equal_all": None, "roll_3y": {"window": 756}, "roll_1p5y": {"window": 378},
               "roll_1y": {"window": 252}, "hl_63": {"halflife": 63},
               "hl_126": {"halflife": 126}, "hl_252": {"halflife": 252},
               "hl_378": {"halflife": 378}, "cap5y_hl126": {"halflife": 126, "window": 1260}}
    for name, rec in mapping.items():
        v = r.get(f"G2_{name}", {}).get("fit_summary", {}).get("mean_val_ic")
        if v is not None and v > bic:
            best, bic = name, v
    return (mapping.get(best, default), best or "hl_126(default)")


def _best_seq():
    p = os.path.join(REPORT_DIR, "G2_results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return int(json.load(f).get("chosen_seq", 40))
    return 40


def g4():
    """Target comparison at the champion recency/seq."""
    out = os.path.join(REPORT_DIR, "G4_results.json")
    seq = _best_seq()
    preset = "A" if seq == 40 else "B"
    rec, rec_name = _best_recency()
    print(f"[G4] seq={seq} recency={rec_name}")
    data, Xg = get_data("close_only", seq, include_barrier=True)
    results = {"recency": rec_name, "seq": seq}
    for target, horizon in [("tgt_rank_5", 5), ("tgt_rank_10", 10), ("tgt_rank_20", 20),
                            ("tgt_exc_uni_20", 20), ("tgt_exc_sec_20", 20),
                            ("tgt_barrier", 20)]:
        run_config(data, Xg, f"G4_{target}", results, out, holding=horizon,
                   target=target, horizon=horizon, preset=preset,
                   refit_every=126, seeds=SCREEN_SEEDS, recency=rec)
    print("\nG4 done")


def g5():
    """Feature-set fair test at champion seq/recency, 20d rank target."""
    out = os.path.join(REPORT_DIR, "G5_results.json")
    seq = _best_seq()
    preset = "A" if seq == 40 else "B"
    rec, rec_name = _best_recency()
    print(f"[G5] seq={seq} recency={rec_name}")
    results = {"recency": rec_name, "seq": seq}
    for fs in ["close_only", "close_d12", "ohlc_range", "volume_block",
               "sector_rel", "curated_full", "full_d12"]:
        data, Xg = get_data(fs, seq)
        run_config(data, Xg, f"G5_{fs}", results, out,
                   target="tgt_rank_20", horizon=20, preset=preset,
                   refit_every=126, seeds=SCREEN_SEEDS, recency=rec)
    print("\nG5 done")


def g3():
    """Retrain cadence comparison over the last ~2 OOS years, champion config."""
    out = os.path.join(REPORT_DIR, "G3_results.json")
    seq = _best_seq()
    preset = "A" if seq == 40 else "B"
    rec, rec_name = _best_recency()
    print(f"[G3] seq={seq} recency={rec_name}")
    data, Xg = get_data("close_only", seq)
    results = {"recency": rec_name, "seq": seq}
    oos = "2024-07-01"
    common = dict(target="tgt_rank_20", horizon=20, preset=preset, recency=rec)

    def run(name, **kw):
        print(f"\n===== G3_{name} =====", flush=True)
        t0 = time.time()
        panel, info = walkforward(data, Xg, oos_start=oos, log_prefix=f"[G3_{name}] ", **kw)
        panel.to_csv(os.path.join(PANEL_DIR, f"G3_{name}.csv.gz"), index=False,
                     compression="gzip")
        perf = eval_panel(panel, 20, f"G3_{name}")
        results[f"G3_{name}"] = {
            "run_info": {k: v for k, v in info.items() if k != "fits"},
            "fit_summary": {"mean_val_ic": info["mean_val_ic"],
                            "total_train_s": round(sum(f["train_s"] for f in info["fits"]), 1),
                            "n_fits": len(info["fits"])},
            "perf": perf, "wall_s": round(time.time() - t0, 1)}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    run("frozen", refit_every=10**6, seeds=CHAMP_SEEDS, **common)
    run("quarterly", refit_every=63, seeds=CHAMP_SEEDS, **common)
    run("monthly", refit_every=20, seeds=CHAMP_SEEDS, **common)
    run("weekly_warm", refit_every=5, seeds=SCREEN_SEEDS, cadence="warm",
        warm_epochs=2, **common)
    run("daily_warm", refit_every=1, seeds=[0], cadence="warm",
        warm_epochs=1, **common)
    print("\nG3 done")


if __name__ == "__main__":
    require_cuda()
    group = sys.argv[1] if len(sys.argv) > 1 else "G1"
    t0 = time.time()
    {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}[group]()
    print(f"\n[{group}] total wall {time.time() - t0:.0f}s")
