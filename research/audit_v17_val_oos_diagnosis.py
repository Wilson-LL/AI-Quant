"""v17 P4-A (part 5) — why is own-validation IC anti-predictive of later
OOS IC across refits? Uses ONLY the existing P0 fits (arms B=5-session,
C=daily; seeds {0,1,2}) on the burned diagnostic interval. CPU only.

Per refit (and per seed where the per-seed predictions allow) assembles:
validation_IC, subsequent block OOS IC, training_end, validation range,
regimes (market at refit, majority regime of the validation block,
regime of the OOS block), epochs run. Tests hypotheses A-E descriptively.
Outputs reports/model_audit/v17/p4/val_oos_diagnosis.{csv,json}.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import dataset_transformer_eod as dte  # noqa: E402
import train_transformer_eod as tte  # noqa: E402
import audit_v17_signal as A  # noqa: E402

V17 = os.path.join(ROOT, "reports", "model_audit", "v17")
FITS = os.path.join(V17, "p0", "fits")
OUT = os.path.join(V17, "p4")


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(a, b, n=2000, seed=3):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        vals.append(spearman(a[i], b[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = tte.PRESETS["B"]
    data = dte.build_dataset("close_only", seq_len=cfg["seq_len"], horizons=(20,), verbose=False)
    dates, dr, stocks = data["dates"], data["date_rank"], np.asarray(data["stocks"])
    wide = A.close_matrix()
    f20 = A.fwd_returns(wide, 20).rename("fwd").reset_index()
    f20.columns = ["date", "stock", "fwd"]
    fwd_map = f20.set_index(["date", "stock"])["fwd"]
    reg = pd.read_csv(os.path.join(V17, "regime_labels.csv"), parse_dates=["date"]).set_index("date")

    rows = []
    for cad in (5, 1):
        for p in sorted(glob.glob(os.path.join(FITS, f"*_c{cad}_s*_tgt_rank_20_*.npz"))):
            z = np.load(p)
            seed, r0 = int(z["seed"]), int(z["block_start"])
            b_end, refit_rank = int(z["block_end"]), int(z["refit_rank"])
            blk = (z["date_rank"] >= r0) & (z["date_rank"] <= b_end)
            df = pd.DataFrame({"date": dates[z["date_rank"][blk]],
                               "stock": stocks[z["stock_idx"][blk]],
                               "pred": z["pred"][blk]})
            df["fwd"] = [fwd_map.get((d, s), np.nan) for d, s in zip(df["date"], df["stock"])]
            df = df.dropna()
            oos = df.groupby("date").apply(lambda g: g["pred"].rank().corr(g["fwd"].rank()),
                                           include_groups=False).mean()
            # validation window for this refit (production rule)
            tr, va, _ = dte.matured_train_val(data, "tgt_rank_20", refit_rank, 20)
            v_first, v_last, t_last = dates[int(dr[va].min())], dates[int(dr[va].max())], dates[int(dr[tr].max())]
            def maj(a, b, col):
                s = reg.loc[a:b, col]
                return s.mode().iloc[0] if len(s) else ""
            rd = pd.Timestamp(dates[r0])
            rows.append({
                "cadence": cad, "seed": seed, "refit_date": str(rd.date()),
                "quarter": f"{rd.year}Q{(rd.month - 1) // 3 + 1}",
                "train_end": str(pd.Timestamp(t_last).date()),
                "val_start": str(pd.Timestamp(v_first).date()), "val_end": str(pd.Timestamp(v_last).date()),
                "val_ic": float(z["val_ic"]), "epochs": int(z["epochs"]),
                "oos_ic_seed": float(oos), "n_block_dates": int(df["date"].nunique()),
                "regime_at_refit": reg.loc[:rd, "trend"].iloc[-1] if rd in reg.index else "",
                "vol_at_refit": reg.loc[:rd, "vol_regime"].iloc[-1] if rd in reg.index else "",
                "val_regime": maj(v_first, v_last, "trend"),
                "oos_regime": maj(dates[r0], dates[b_end], "trend"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "val_oos_diagnosis.csv"), index=False)

    out = {}
    for cad in (5, 1):
        s = df[df["cadence"] == cad]
        ens = s.groupby("refit_date").agg(val_ic=("val_ic", "mean"), oos=("oos_ic_seed", "mean"),
                                          epochs=("epochs", "mean"), quarter=("quarter", "first"),
                                          regime_at_refit=("regime_at_refit", "first"),
                                          val_regime=("val_regime", "first"),
                                          oos_regime=("oos_regime", "first")).reset_index()
        rec = {"n_refits": int(len(ens)), "n_seed_fits": int(len(s)),
               "ensemble_level": {"spearman": spearman(ens["val_ic"], ens["oos"]),
                                  "pearson": pearson(ens["val_ic"], ens["oos"]),
                                  "spearman_boot95": boot_ci(ens["val_ic"], ens["oos"])},
               "seed_level": {"spearman": spearman(s["val_ic"], s["oos_ic_seed"]),
                              "pearson": pearson(s["val_ic"], s["oos_ic_seed"])},
               "per_seed": {int(k): spearman(g["val_ic"], g["oos_ic_seed"]) for k, g in s.groupby("seed")}}
        # trimmed: drop 10% most extreme oos and val values
        lo, hi = ens["oos"].quantile([0.05, 0.95])
        lv, hv = ens["val_ic"].quantile([0.05, 0.95])
        t = ens[(ens["oos"].between(lo, hi)) & (ens["val_ic"].between(lv, hv))]
        rec["trimmed_spearman"] = spearman(t["val_ic"], t["oos"]) if len(t) >= 8 else None
        rec["by_quarter"] = {q: {"n": int(len(g)), "spearman": spearman(g["val_ic"], g["oos"]) if len(g) >= 6 else None,
                                 "val_ic_mean": float(g["val_ic"].mean()), "oos_mean": float(g["oos"].mean())}
                             for q, g in ens.groupby("quarter")}
        # E: epochs vs OOS / val
        rec["epochs"] = {"mean": float(ens["epochs"].mean()),
                         "spearman_epochs_vs_oos": spearman(ens["epochs"], ens["oos"]),
                         "spearman_epochs_vs_val": spearman(ens["epochs"], ens["val_ic"]),
                         "seed_level_spearman_epochs_vs_oos": spearman(s["epochs"], s["oos_ic_seed"])}
        # A/B: regime structure
        rec["regimes"] = {"val_regime_counts": ens["val_regime"].value_counts().to_dict(),
                          "oos_regime_counts": ens["oos_regime"].value_counts().to_dict(),
                          "same_regime_frac": float((ens["val_regime"] == ens["oos_regime"]).mean())}
        for name, g in (("same_regime", ens[ens["val_regime"] == ens["oos_regime"]]),
                        ("different_regime", ens[ens["val_regime"] != ens["oos_regime"]])):
            rec["regimes"][name] = {"n": int(len(g)), "spearman": spearman(g["val_ic"], g["oos"]) if len(g) >= 6 else None,
                                    "oos_mean": float(g["oos"].mean()) if len(g) else None}
        # D: how much of the val-IC variance is seed vs refit
        piv = s.pivot(index="refit_date", columns="seed", values="val_ic")
        rec["val_ic_variance_decomposition"] = {
            "between_refit_std": float(piv.mean(axis=1).std()),
            "within_refit_seed_std_mean": float(piv.std(axis=1).mean())}
        # val-IC vs OOS gap by val-IC tercile (is the top tercile the worst?)
        ens["val_tercile"] = pd.qcut(ens["val_ic"], 3, labels=["low", "mid", "high"])
        rec["oos_by_val_tercile"] = {str(k): float(v) for k, v in ens.groupby("val_tercile", observed=True)["oos"].mean().items()}
        out[f"cadence_{cad}"] = rec
    with open(os.path.join(OUT, "val_oos_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
