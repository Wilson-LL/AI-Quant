"""v13 Phase 1 — feature construction smoke + leakage/quality battery.

Builds F0 (close_only control) and v13_f1/f2/f3, runs the 13-point quality
battery (leakage, NaN/inf, drift, extremes, missingness, correlation,
duplicates, constants), and does a tiny 2-epoch plumbing fit on f1 and f3
to verify the training/inference path accepts the wider inputs. No long
training; nothing written outside the v13 report dir.

Usage:  python research/v13_feature_quality.py [--sets v13_f1,v13_f2,v13_f3]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V13_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v13_fixed_xl_feature_rich")


def truncation_leakage_test(sets, n_stocks=5, cut=20):
    """Features at t must be identical whether or not the last `cut` days
    exist -> proves no future dependence (covers checks 1/2/6)."""
    import dataset_transformer_eod as D
    from data import load_universe, SECTOR_MAP
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"][:n_stocks]
    uni = load_universe(ids)
    out = {}
    for fs in sets:
        worst = 0.0
        for sid, df in uni.items():
            df = (df.sort_values("date").drop_duplicates("date", keep="last")
                    .reset_index(drop=True))
            full = D._stock_features(df, fs)
            part = D._stock_features(df.iloc[:-cut].reset_index(drop=True), fs)
            n = len(part)
            a, b = full.iloc[:n], part
            common = [c for c in a.columns if c in b.columns]
            d = (a[common] - b[common]).abs()
            worst = max(worst, float(np.nanmax(d.to_numpy())))
        out[fs] = worst
    return out


def battery(fs, data):
    X = data["X"]                       # (n, seq, F) float32
    F = np.asarray(X[:, -1, :], np.float64)   # features at prediction date
    cols = data["feature_cols"]
    dr = np.asarray(data["date_rank"])
    stocks = np.asarray(data["stocks"])[np.asarray(data["stock_idx"])]
    years = pd.to_datetime(np.asarray(data["dates"])[dr]).year
    dfF = pd.DataFrame(F, columns=cols)

    # 7. NaN / inf (post all-finite-window rule this must be zero)
    n_bad = int((~np.isfinite(F)).sum())
    # 13. constants
    stds = dfF.std()
    constants = list(stds[stds < 1e-10].index)
    # 11/12. correlation + duplicates (sampled for speed)
    samp = dfF.iloc[::5]
    corr = samp.corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.90:
                pairs.append({"feature_set": fs, "f1": a, "f2": b,
                              "corr": round(float(r), 4),
                              "duplicate": bool(abs(r) > 0.999)})
    # 9. extremes / clipping
    ext = pd.DataFrame({
        "p001": dfF.quantile(0.001), "p999": dfF.quantile(0.999),
        "frac_abs_gt10": (dfF.abs() > 10).mean()})
    heavy = list(ext[ext["frac_abs_gt10"] > 0.001].index)
    # 8. yearly drift: max |yearly mean - overall mean| / overall std
    drift = {}
    for c in cols:
        ym = pd.Series(dfF[c].values, index=years).groupby(level=0).mean()
        drift[c] = float(((ym - dfF[c].mean()).abs() / (stds[c] + 1e-12)).max())
    top_drift = sorted(drift.items(), key=lambda kv: -kv[1])[:5]
    # 10. missingness = share of raw rows excluded by the finite-window rule
    #     (approximated by sample count vs the close_only reference)
    miss_rows = [{"feature_set": fs, "feature": c,
                  "nan_frac_in_dataset": 0.0,
                  "note": "all-finite-window rule excludes NaNs upstream"}
                 for c in cols]
    # 5. target must not be an input
    assert not any(c.startswith(("tgt_", "fwd_")) for c in cols), \
        "target leaked into feature columns"
    per_symbol = pd.Series(1, index=stocks).groupby(level=0).sum()
    return {
        "n_samples": int(len(F)), "n_features": len(cols),
        "n_nonfinite": n_bad, "constants": constants,
        "n_corr_gt90": len(pairs), "pairs": pairs,
        "heavy_tailed(|x|>10 >0.1%)": heavy,
        "top_yearly_drift": [(c, round(v, 2)) for c, v in top_drift],
        "extremes": ext.round(3).to_dict("index"),
        "min_symbol_samples": int(per_symbol.min()),
    }, miss_rows, pairs


def plumbing_fit(fs, data):
    """2-epoch small-model fit + reload-and-score: plumbing only."""
    import torch
    import train_transformer_eod as T
    T.require_cuda()
    Xg = T.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]),
                         device=Xg.device)
    rr = int(data["date_rank"].max())
    tr, va, w = T.matured_train_val(data, "tgt_rank_20", rr, 20)
    cfg = dict(T.PRESETS["B"], max_epochs=2)
    t0 = time.time()
    net, vic, info = T.fit_one(Xg, yg, tr, va, data["date_rank"][va], cfg,
                               seed=0, weights=w)
    idx = torch.as_tensor(np.nonzero(np.asarray(data["date_rank"]) == rr)[0],
                          device=Xg.device)
    sc = T.predict_idx(net, Xg, idx).cpu().numpy()
    ok = bool(np.isfinite(sc).all()) and info["epochs_run"] == 2
    r = {"feature_set": fs, "input_dim": int(Xg.shape[2]), "ok": ok,
         "val_ic_2ep": round(vic, 5), "train_s": info["train_s"],
         "n_scored": int(len(sc)),
         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
         "total_s": round(time.time() - t0, 1)}
    del net, Xg, yg
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="v13_f1,v13_f2,v13_f3")
    ap.add_argument("--fit-sets", default="v13_f1,v13_f3")
    a = ap.parse_args()
    sets = a.sets.split(",")
    os.makedirs(V13_DIR, exist_ok=True)
    from dataset_transformer_eod import build_dataset

    print("[quality] truncation leakage test (features must not change when "
          "future days are removed)")
    leak = truncation_leakage_test(sets)
    for fs, worst in leak.items():
        print(f"  {fs}: max |diff| = {worst:.2e} "
              f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    results, miss_all, pairs_all, fits = {}, [], [], []
    ref_n = None
    for fs in ["close_only"] + sets:
        t0 = time.time()
        data = build_dataset(fs, seq_len=60, horizons=(20,))
        r, miss, pairs = battery(fs, data)
        r["build_s"] = round(time.time() - t0, 1)
        r["leakage_max_diff"] = leak.get(fs, 0.0)
        if fs == "close_only":
            ref_n = r["n_samples"]
        r["samples_vs_close_only"] = round(r["n_samples"] / ref_n, 4)
        results[fs] = r
        miss_all += miss
        pairs_all += pairs
        print(f"[quality] {fs}: {r['n_features']} features, "
              f"{r['n_samples']:,} samples ({r['samples_vs_close_only']:.1%} "
              f"of close_only), nonfinite {r['n_nonfinite']}, "
              f"constants {r['constants']}, |corr|>0.90 pairs "
              f"{r['n_corr_gt90']}")
        if fs in a.fit_sets.split(","):
            fr = plumbing_fit(fs, data)
            fits.append(fr)
            print(f"[plumbing] {fs}: input_dim {fr['input_dim']} ok={fr['ok']} "
                  f"val_ic(2ep) {fr['val_ic_2ep']} vram {fr['peak_vram_gb']}GB")
        del data

    pd.DataFrame(miss_all).to_csv(
        os.path.join(V13_DIR, "feature_missingness.csv"), index=False)
    pd.DataFrame(pairs_all).to_csv(
        os.path.join(V13_DIR, "feature_correlation_summary.csv"), index=False)

    md = ["# v13 feature quality report (Phase 1)", "",
          "Leakage: truncation test — per-stock features recomputed with the "
          "last 20 days removed must be identical on overlapping dates "
          "(covers future-price leakage, rolling-window causality, and "
          "timestamp<=prediction). Cross-sectional transforms are per-date "
          "by construction; targets asserted absent from inputs.", ""]
    for fs, r in results.items():
        md += [f"## {fs}", "",
               f"- leakage max diff: {r['leakage_max_diff']:.2e} "
               f"({'PASS' if r['leakage_max_diff'] < 1e-9 else 'FAIL'})"
               if fs != "close_only" else
               "- leakage: production control (unchanged path)",
               f"- features {r['n_features']} · samples {r['n_samples']:,} "
               f"({r['samples_vs_close_only']:.1%} of close_only — the "
               f"252d lookbacks cost warmup history)",
               f"- non-finite values: {r['n_nonfinite']} · constant features: "
               f"{r['constants'] or 'none'}",
               f"- |corr|>0.90 pairs: {r['n_corr_gt90']} (full list in "
               "feature_correlation_summary.csv)",
               f"- heavy-tailed (|x|>10 on >0.1% rows): "
               f"{r['heavy_tailed(|x|>10 >0.1%)'] or 'none'}",
               f"- top yearly drift (|year mean − overall|/std): "
               + ", ".join(f"{c} {v}" for c, v in r["top_yearly_drift"]), ""]
    if fits:
        md += ["## Plumbing fits (2 epochs, small model — NOT signal)", ""]
        for f in fits:
            md.append(f"- {f['feature_set']}: input_dim {f['input_dim']}, "
                      f"ok={f['ok']}, val_ic(2ep) {f['val_ic_2ep']}, "
                      f"train {f['train_s']}s, VRAM {f['peak_vram_gb']} GB")
    with open(os.path.join(V13_DIR, "feature_quality_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(V13_DIR, "phase1_battery.json"), "w",
              encoding="utf-8") as f:
        json.dump({"results": results, "fits": fits, "leakage": leak},
                  f, indent=2, default=str)
    print(f"[quality] -> feature_quality_report.md, feature_missingness.csv, "
          f"feature_correlation_summary.csv")


if __name__ == "__main__":
    main()
