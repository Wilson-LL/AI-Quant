"""v17 P0 — refit-cadence parity runner (paired seeds; resumable; isolated).

Implements reports/model_audit/v17/p0_refit_parity_spec.json. Both arms
use the SAME seed IDs, decision dates, dataset, training rules and
downstream construction; only `refit_every` differs.

  python research/run_p0_refit_parity.py --stage P0_A_prime --dry-run
  python research/run_p0_refit_parity.py --stage P0_A_prime            # fits
  python research/run_p0_refit_parity.py --stage P0_A_prime --evaluate # metrics

Artifacts live ONLY under reports/model_audit/v17/p0/. Production
checkpoints, predictions, books, plans, holdings and the intraday DB are
never written.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

SPEC_P = os.path.join(ROOT, "reports", "model_audit", "v17",
                      "p0_refit_parity_spec.json")
OUT = os.path.join(ROOT, "reports", "model_audit", "v17", "p0")
FITS = os.path.join(OUT, "fits")
FEATURE_SET, TARGET, HORIZON, PRESET = "close_only", "tgt_rank_20", 20, "B"


def load_spec():
    with open(SPEC_P) as f:
        return json.load(f)


def cfg_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def plan_refits(dates, start, end, cadence):
    """Refit block starts (date ranks) covering [start, end] every `cadence`."""
    s = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(start))))
    e = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(end)), side="right")) - 1
    return list(range(s, e + 1, cadence)), s, e


def fit_key(refit_date, cadence, seed, chash):
    return f"{refit_date}_c{cadence}_s{seed}_{TARGET}_{chash}.npz"


def run_fits(stage, dry_run=False):
    import torch
    import train_transformer_eod as tte
    from dataset_transformer_eod import build_dataset, matured_train_val

    spec = load_spec()
    arms = spec["arms"][stage]
    start, end = spec["evaluation_interval"]["start"], spec["evaluation_interval"]["end"]
    cfg = tte.PRESETS[PRESET]
    chash = cfg_hash(cfg)
    os.makedirs(FITS, exist_ok=True)

    print("[p0] building dataset (features causal; labels masked by maturity)...")
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,),
                         verbose=False)
    dates = data["dates"]
    dr = data["date_rank"]
    # decision dates (identical for both arms) — frozen to disk first
    _, s_rank, e_rank = plan_refits(dates, start, end, 1)
    dec_dates = [str(dates[r])[:10] for r in range(s_rank, e_rank + 1)]
    with open(os.path.join(OUT, "decision_dates.json"), "w") as f:
        json.dump({"stage": stage, "dates": dec_dates}, f, indent=1)

    todo, total = [], 0
    for arm, a in arms.items():
        refits, _, _ = plan_refits(dates, start, end, a["refit_every_sessions"])
        for r0 in refits:
            for seed in a["seeds"]:
                total += 1
                k = fit_key(str(dates[r0])[:10], a["refit_every_sessions"], seed, chash)
                if not os.path.isfile(os.path.join(FITS, k)):
                    todo.append((arm, r0, a["refit_every_sessions"], seed, k))
    sec = spec["compute"]["measured_seconds_per_seed_fit"]
    print(f"[p0] {stage}: decision dates {len(dec_dates)} ({dec_dates[0]}..{dec_dates[-1]})")
    print(f"[p0] seed-fits total {total}, missing {len(todo)}, "
          f"estimated runtime {len(todo) * sec / 3600:.2f} h at {sec}s/fit")
    if dry_run:
        return
    if len(todo) * sec / 3600 > 2.5 and stage == "P0_A_prime":
        sys.exit("STOP: estimated runtime exceeds the pre-registered 2.5 h stop rule")

    tte.require_cuda()
    Xg = tte.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][TARGET], -1, 1)),
                         device=Xg.device)
    t_all = time.time()
    failed = []
    for i, (arm, r0, cad, seed, k) in enumerate(todo, 1):
        refit_rank = r0 - 1
        block_end = min(r0 + cad - 1, e_rank)
        try:
            tr, va, _ = matured_train_val(data, TARGET, refit_rank, HORIZON)
            # production-parity assertion: no label used matures after refit
            assert int(data["label_end_rank"][HORIZON][tr].max()) <= refit_rank
            assert int(data["label_end_rank"][HORIZON][va].max()) <= refit_rank
            net, vic, info = tte.fit_one(Xg, yg, tr, va, dr[va], cfg, seed=seed)
            # score the block AND the next block's first date (stability overlap)
            score_ranks = list(range(r0, block_end + 1))
            overlap_rank = block_end + 1 if block_end + 1 <= e_rank else None
            if overlap_rank is not None:
                score_ranks.append(overlap_rank)
            idx = np.nonzero(np.isin(dr, score_ranks))[0]
            preds = tte.predict_idx(net, Xg, idx).cpu().numpy()
            np.savez_compressed(
                os.path.join(FITS, k), idx=idx, pred=preds.astype(np.float32),
                date_rank=dr[idx], stock_idx=data["stock_idx"][idx],
                overlap_rank=-1 if overlap_rank is None else overlap_rank,
                block_start=r0, block_end=block_end, refit_rank=refit_rank,
                seed=seed, cadence=cad, val_ic=float(vic),
                epochs=int(info["epochs_run"]), train_s=float(info["train_s"]),
                n_train=int(len(tr)))
            print(f"[p0] {i}/{len(todo)} {arm} refit {str(dates[r0])[:10]} c{cad} "
                  f"s{seed}: val_ic {vic:+.4f} ep {info['epochs_run']} "
                  f"{info['train_s']:.0f}s  (elapsed {(time.time()-t_all)/60:.1f} min)",
                  flush=True)
            del net
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — record and continue
            failed.append({"key": k, "error": repr(e)})
            print(f"[p0] FAILED {k}: {e!r}", flush=True)
    with open(os.path.join(OUT, f"run_{stage}_log.json"), "w") as f:
        json.dump({"stage": stage, "fits_planned": total, "fits_run": len(todo),
                   "failed": failed, "elapsed_s": round(time.time() - t_all, 1),
                   "cfg_hash": chash}, f, indent=1)
    print(f"[p0] done: {len(todo) - len(failed)} ok, {len(failed)} failed, "
          f"{(time.time()-t_all)/3600:.2f} h")


# ------------------------------------------------------------ evaluation

def assemble_panel(stage, arm, spec, data_dates, stocks):
    """Ensemble mean/std per (date, stock) from the arm's fits, plus the
    overlap-date predictions from the PREVIOUS refit (for stability)."""
    import train_transformer_eod as tte
    cfg = tte.PRESETS[PRESET]
    chash = cfg_hash(cfg)
    a = spec["arms"][stage][arm]
    start, end = spec["evaluation_interval"]["start"], spec["evaluation_interval"]["end"]
    refits, _, e_rank = plan_refits(data_dates, start, end, a["refit_every_sessions"])
    rows, overlaps, fitmeta = [], [], []
    for r0 in refits:
        per_seed = []
        for seed in a["seeds"]:
            k = fit_key(str(data_dates[r0])[:10], a["refit_every_sessions"], seed, chash)
            z = np.load(os.path.join(FITS, k))
            per_seed.append(z)
            fitmeta.append({"refit_date": str(data_dates[r0])[:10], "seed": seed,
                            "val_ic": float(z["val_ic"]), "epochs": int(z["epochs"]),
                            "train_s": float(z["train_s"])})
        preds = np.stack([z["pred"] for z in per_seed])
        z0 = per_seed[0]
        df = pd.DataFrame({"date_rank": z0["date_rank"], "stock": stocks[z0["stock_idx"]],
                           "score": preds.mean(0), "score_std": preds.std(0)})
        ov = int(z0["overlap_rank"])
        blk = df[df["date_rank"] <= int(z0["block_end"])]
        rows.append(blk)
        if ov >= 0:
            overlaps.append(df[df["date_rank"] == ov].assign(from_refit=int(r0)))
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = data_dates[panel["date_rank"].to_numpy()]
    ovl = pd.concat(overlaps, ignore_index=True) if overlaps else pd.DataFrame()
    if len(ovl):
        ovl["date"] = data_dates[ovl["date_rank"].to_numpy()]
    return panel, ovl, pd.DataFrame(fitmeta)


def hac_se(x, lag=20):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    d = x - x.mean()
    g = [float(np.mean(d[:n - k] * d[k:])) for k in range(0, min(lag, n - 1) + 1)]
    v = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, len(g)))
    return math.sqrt(max(v, 1e-12) / n)


def evaluate(stage):
    from dataset_transformer_eod import build_dataset
    from transformer_hybrid import aux_columns, _z
    from transformer_portfolio import backtest_scores
    import audit_v17_signal as A
    import train_transformer_eod as tte

    spec = load_spec()
    cfg = tte.PRESETS[PRESET]
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,),
                         verbose=False)
    dates, stocks = data["dates"], np.asarray(data["stocks"])
    wide = A.close_matrix()
    f20 = A.fwd_returns(wide, 20).reset_index()
    f20.columns = ["date", "stock", "fwd_h"]
    aux = aux_columns()
    res, per_date = {}, {}
    for arm in ("A", "B"):
        panel, ovl, fm = assemble_panel(stage, arm, spec, dates, stocks)
        m = panel.merge(f20, on=["date", "stock"], how="left").merge(
            aux, on=["date", "stock"], how="left")
        g = m.groupby("date")
        m["z_tf"] = g["score"].transform(_z)
        m["z_mom"] = g["mom"].transform(_z)
        m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
        ic = A.spearman_by_date(m.dropna(subset=["fwd_h"]), "score", "fwd_h")
        m["pct"] = g["score"].rank(ascending=False, pct=True)
        top = m[m["pct"] <= 0.2].groupby("date")["fwd_h"].mean()
        bot = m[m["pct"] > 0.8].groupby("date")["fwd_h"].mean()
        spread = (top - bot)
        per_date[arm] = pd.DataFrame({"ic": ic, "spread20": spread})
        pp = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(
            columns={"blend": "score"}).dropna(subset=["score", "fwd_h"])
        bt = backtest_scores(pp, mode="long_only", no_trade_band=0.10,
                             cost_bps_list=(0, 60, 100), min_names=60)
        k = max(3, round(0.2 * m.groupby("date")["stock"].nunique().median()))
        # stability at consecutive refits (same date scored by old and new fit)
        stab = []
        for _, o in ovl.groupby("from_refit"):
            d = o["date"].iloc[0]
            new = m[m["date"] == d][["stock", "score"]].set_index("stock")["score"]
            old = o.set_index("stock")["score"]
            j = new.index.intersection(old.index)
            if len(j) < 10:
                continue
            kk = max(3, round(0.2 * len(j)))
            tn = set(new.loc[j].nlargest(kk).index)
            to = set(old.loc[j].nlargest(kk).index)
            stab.append({"date": str(pd.Timestamp(d).date()),
                         "score_corr": float(np.corrcoef(new.loc[j], old.loc[j])[0, 1]),
                         "rank_corr": float(new.loc[j].rank().corr(old.loc[j].rank())),
                         "topq_overlap": len(tn & to) / max(len(tn | to), 1)})
        stab = pd.DataFrame(stab)
        res[arm] = {
            "cadence": spec["arms"][stage][arm]["refit_every_sessions"],
            "seeds": spec["arms"][stage][arm]["seeds"],
            "n_decision_dates": int(ic.notna().sum()),
            "rank_ic_mean": float(ic.mean()), "rank_ic_std": float(ic.std()),
            "positive_ic_frequency": float((ic > 0).mean()),
            "top20_minus_bottom20_fwd20": float(spread.mean()),
            "gross_return_per_block_mean": float(np.mean(bt["gross"])),
            "n_blocks": int(bt["net60"]["n"]),
            "net0_sharpe_like": bt["net0"]["sharpe"],
            "net60_sharpe_like": bt["net60"]["sharpe"],
            "net100_sharpe_like": bt["net100"]["sharpe"],
            "max_drawdown_net60": bt["net60"]["max_dd"],
            "avg_one_way_turnover": bt["avg_turnover"],
            "avg_names_changed_per_rebalance": float(bt["avg_turnover"] * k * 2),
            "consecutive_refit_score_corr": float(stab["score_corr"].mean()) if len(stab) else float("nan"),
            "consecutive_refit_rank_corr": float(stab["rank_corr"].mean()) if len(stab) else float("nan"),
            "top_quintile_overlap_consecutive": float(stab["topq_overlap"].mean()) if len(stab) else float("nan"),
            "n_refit_boundaries": int(len(stab)),
            "seed_dispersion_mean_score_std": float(panel["score_std"].mean()),
            "fits": {"n": int(len(fm)), "val_ic_mean": float(fm["val_ic"].mean()),
                     "epochs_mean": float(fm["epochs"].mean()),
                     "train_s_total": float(fm["train_s"].sum())},
        }
        panel.to_csv(os.path.join(OUT, f"panel_{stage}_arm{arm}.csv.gz"), index=False)
        stab.to_csv(os.path.join(OUT, f"stability_{stage}_arm{arm}.csv"), index=False)
    # paired differences on the same dates
    j = per_date["A"].join(per_date["B"], lsuffix="_A", rsuffix="_B", how="inner").dropna()
    d_ic = (j["ic_B"] - j["ic_A"]).to_numpy()
    d_sp = (j["spread20_B"] - j["spread20_A"]).to_numpy()
    paired = {"n_dates": int(len(j)),
              "d_ic_mean": float(d_ic.mean()), "d_ic_hac_se": hac_se(d_ic),
              "d_ic_frac_positive": float((d_ic > 0).mean()),
              "d_spread20_mean": float(d_sp.mean()), "d_spread20_hac_se": hac_se(d_sp),
              "d_pos_frac": res["B"]["positive_ic_frequency"] - res["A"]["positive_ic_frequency"],
              "d_turnover": res["B"]["avg_one_way_turnover"] - res["A"]["avg_one_way_turnover"],
              "d_max_dd_pp": 100 * (res["B"]["max_drawdown_net60"] - res["A"]["max_drawdown_net60"]),
              "d_net60_sharpe": res["B"]["net60_sharpe_like"] - res["A"]["net60_sharpe_like"],
              "d_net100_sharpe": res["B"]["net100_sharpe_like"] - res["A"]["net100_sharpe_like"]}
    # pre-registered classification (spec.decision_thresholds)
    dic, se = paired["d_ic_mean"], paired["d_ic_hac_se"]
    rc, ov = res["B"]["consecutive_refit_rank_corr"], res["B"]["top_quintile_overlap_consecutive"]
    ok = ((dic >= -0.005 or (dic < 0 and abs(dic) <= se)) and paired["d_pos_frac"] >= -0.05
          and rc >= 0.90 and ov >= 0.70 and paired["d_turnover"] <= 0.05
          and paired["d_max_dd_pp"] >= -3.0)   # max_dd is negative; B worse = more negative
    risk_votes = sum([paired["d_pos_frac"] <= -0.05, paired["d_spread20_mean"] <= -0.005,
                      (rc < 0.80 or ov < 0.50), paired["d_turnover"] > 0.10])
    risk = (dic <= -0.010 and dic < -se and risk_votes >= 2)
    cls = "PARITY_OK" if ok else ("PRODUCTION_RESEARCH_PARITY_RISK" if risk else "PARITY_WARNING")
    out = {"stage": stage, "arms": res, "paired": paired, "classification": cls,
           "interval": spec["evaluation_interval"], "note": spec["evaluation_interval"]["note"]}
    with open(os.path.join(OUT, f"result_{stage}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="P0_A_prime",
                    choices=["P0_A_prime", "P0_A", "P0_B"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.evaluate:
        evaluate(a.stage)
    else:
        run_fits(a.stage, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
