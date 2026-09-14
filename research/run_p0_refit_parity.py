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


def run_fits(stage, dry_run=False, deadline=None):
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
    dl = None
    if deadline:
        import datetime as _dt
        hh, mm = map(int, deadline.split(":"))
        dl = _dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        print(f"[p0] wall-clock deadline {dl:%H:%M}: no new fit starts after it (resumable)")

    tte.require_cuda()
    Xg = tte.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][TARGET], -1, 1)),
                         device=Xg.device)
    t_all = time.time()
    failed = []
    stopped_early = False
    for i, (arm, r0, cad, seed, k) in enumerate(todo, 1):
        if dl is not None and __import__("datetime").datetime.now() >= dl:
            print(f"[p0] deadline reached after {i-1} fits - stopping (resume later)")
            stopped_early = True
            break
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
        json.dump({"stage": stage, "fits_planned": total, "fits_missing_at_start": len(todo),
                   "failed": failed, "elapsed_s": round(time.time() - t_all, 1),
                   "stopped_at_deadline": stopped_early, "cfg_hash": chash}, f, indent=1)
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


def _arm_metrics(stage, arm, spec, dates, stocks, f20, aux):
    from transformer_hybrid import _z
    from transformer_portfolio import backtest_scores
    import audit_v17_signal as A
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
    per_date = pd.DataFrame({"ic": ic, "spread20": spread})
    pp = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(
        columns={"blend": "score"}).dropna(subset=["score", "fwd_h"])
    bt = backtest_scores(pp, mode="long_only", no_trade_band=0.10,
                         cost_bps_list=(0, 60, 100), min_names=60)
    k = max(3, round(0.2 * m.groupby("date")["stock"].nunique().median()))
    # refit shock: previous model (stored overlap prediction) vs new model on
    # the SAME date and IDENTICAL feature windows
    shock = []
    for _, o in ovl.groupby("from_refit"):
        d = o["date"].iloc[0]
        new = m[m["date"] == d][["stock", "score"]].set_index("stock")["score"]
        old = o.set_index("stock")["score"]
        j = new.index.intersection(old.index)
        if len(j) < 10:
            continue
        kk = max(3, round(0.2 * len(j)))
        rn, ro = new.loc[j].rank(ascending=False), old.loc[j].rank(ascending=False)
        tn, to = set(rn.nsmallest(kk).index), set(ro.nsmallest(kk).index)
        shock.append({"date": str(pd.Timestamp(d).date()),
                      "score_corr": float(np.corrcoef(new.loc[j], old.loc[j])[0, 1]),
                      "rank_corr": float(rn.corr(ro)),
                      "topq_overlap": len(tn & to) / max(len(tn | to), 1),
                      "topq_entering": len(tn - to), "topq_leaving": len(to - tn),
                      "mean_abs_rank_move": float((rn - ro).abs().mean()),
                      "max_abs_rank_move": float((rn - ro).abs().max()),
                      "n_names": int(len(j))})
    shock = pd.DataFrame(shock)
    # validation IC vs subsequent OOS IC per refit
    blocks = []
    a = spec["arms"][stage][arm]
    refits, _, e_rank = plan_refits(dates, spec["evaluation_interval"]["start"],
                                    spec["evaluation_interval"]["end"],
                                    a["refit_every_sessions"])
    for r0 in refits:
        b_end = min(r0 + a["refit_every_sessions"] - 1, e_rank)
        bd = [dates[r] for r in range(r0, b_end + 1)]
        oos = ic.reindex(pd.DatetimeIndex(bd)).mean()
        v = fm[fm["refit_date"] == str(dates[r0])[:10]]["val_ic"].mean()
        blocks.append({"refit_date": str(dates[r0])[:10], "val_ic": float(v),
                       "oos_block_ic": float(oos) if pd.notna(oos) else np.nan})
    blocks = pd.DataFrame(blocks).dropna()
    val_oos_corr = (float(blocks["val_ic"].rank().corr(blocks["oos_block_ic"].rank()))
                    if len(blocks) >= 5 else float("nan"))
    res = {
        "cadence": a["refit_every_sessions"], "seeds": a["seeds"],
        "n_decision_dates": int(ic.notna().sum()),
        "rank_ic_mean": float(ic.mean()), "rank_ic_std": float(ic.std()),
        "rank_ic_hac_se": hac_se(ic.dropna().to_numpy()),
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
        "consecutive_refit_score_corr": float(shock["score_corr"].mean()) if len(shock) else float("nan"),
        "consecutive_refit_rank_corr": float(shock["rank_corr"].mean()) if len(shock) else float("nan"),
        "top_quintile_overlap_consecutive": float(shock["topq_overlap"].mean()) if len(shock) else float("nan"),
        "refit_shock": ({"n_boundaries": int(len(shock)),
                         "same_date_score_corr": float(shock["score_corr"].mean()),
                         "same_date_rank_corr": float(shock["rank_corr"].mean()),
                         "same_date_topq_overlap": float(shock["topq_overlap"].mean()),
                         "topq_entering_mean": float(shock["topq_entering"].mean()),
                         "topq_leaving_mean": float(shock["topq_leaving"].mean()),
                         "mean_abs_rank_move": float(shock["mean_abs_rank_move"].mean()),
                         "max_abs_rank_move_mean": float(shock["max_abs_rank_move"].mean()),
                         "max_abs_rank_move_max": float(shock["max_abs_rank_move"].max())}
                        if len(shock) else "REFIT_SHOCK_DIAGNOSTIC_DEFERRED"),
        "seed_dispersion_mean_score_std": float(panel["score_std"].mean()),
        "training": {"n_fits": int(len(fm)), "val_ic_mean": float(fm["val_ic"].mean()),
                     "val_ic_std": float(fm["val_ic"].std()),
                     "val_ic_p10": float(fm["val_ic"].quantile(0.1)),
                     "val_ic_p90": float(fm["val_ic"].quantile(0.9)),
                     "epochs_mean": float(fm["epochs"].mean()),
                     "train_s_total": float(fm["train_s"].sum()),
                     "n_refits": int(len(blocks)),
                     "oos_block_ic_mean": float(blocks["oos_block_ic"].mean()) if len(blocks) else float("nan"),
                     "val_ic_vs_oos_block_ic_spearman": val_oos_corr},
    }
    panel.to_csv(os.path.join(OUT, f"panel_{stage}_arm{arm}.csv.gz"), index=False)
    shock.to_csv(os.path.join(OUT, f"refit_shock_{stage}_arm{arm}.csv"), index=False)
    blocks.to_csv(os.path.join(OUT, f"val_vs_oos_{stage}_arm{arm}.csv"), index=False)
    return res, per_date


def _paired(pa, pb, ra, rb):
    j = pa.join(pb, lsuffix="_A", rsuffix="_B", how="inner").dropna()
    d_ic = (j["ic_B"] - j["ic_A"]).to_numpy()
    d_sp = (j["spread20_B"] - j["spread20_A"]).to_numpy()
    return {"n_dates": int(len(j)),
            "d_ic_mean": float(d_ic.mean()), "d_ic_hac_se": hac_se(d_ic),
            "d_ic_frac_positive": float((d_ic > 0).mean()),
            "d_spread20_mean": float(d_sp.mean()), "d_spread20_hac_se": hac_se(d_sp),
            "d_pos_frac": rb["positive_ic_frequency"] - ra["positive_ic_frequency"],
            "d_rank_corr": rb["consecutive_refit_rank_corr"] - ra["consecutive_refit_rank_corr"],
            "d_turnover": rb["avg_one_way_turnover"] - ra["avg_one_way_turnover"],
            "d_max_dd_pp": 100 * (rb["max_drawdown_net60"] - ra["max_drawdown_net60"]),
            "d_net60_sharpe": rb["net60_sharpe_like"] - ra["net60_sharpe_like"],
            "d_net100_sharpe": rb["net100_sharpe_like"] - ra["net100_sharpe_like"],
            "d_val_ic": rb["training"]["val_ic_mean"] - ra["training"]["val_ic_mean"]}


def _classify(paired, rb):
    dic, se = paired["d_ic_mean"], paired["d_ic_hac_se"]
    rc, ov = rb["consecutive_refit_rank_corr"], rb["top_quintile_overlap_consecutive"]
    ok = ((dic >= -0.005 or (dic < 0 and abs(dic) <= se)) and paired["d_pos_frac"] >= -0.05
          and rc >= 0.90 and ov >= 0.70 and paired["d_turnover"] <= 0.05
          and paired["d_max_dd_pp"] >= -3.0)
    votes = sum([paired["d_pos_frac"] <= -0.05, paired["d_spread20_mean"] <= -0.005,
                 (rc < 0.80 or ov < 0.50), paired["d_turnover"] > 0.10])
    risk = (dic <= -0.010 and dic < -se and votes >= 2)
    return ("PARITY_OK" if ok else "PRODUCTION_RESEARCH_PARITY_RISK" if risk
            else "PARITY_WARNING"), ok, votes


def evaluate(stage):
    from dataset_transformer_eod import build_dataset
    from transformer_hybrid import aux_columns
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
    arms = list(spec["arms"][stage].keys())
    res, per_date = {}, {}
    for arm in arms:
        res[arm], per_date[arm] = _arm_metrics(stage, arm, spec, dates, stocks, f20, aux)
    pairs = {}
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            pairs[f"{y}_minus_{x}"] = _paired(per_date[x], per_date[y], res[x], res[y])
    out = {"stage": stage, "arms": res, "paired": pairs,
           "interval": spec["evaluation_interval"],
           "interval_status": spec["evaluation_interval"].get("status", "")}
    if stage == "P0_A_prime":
        out["classification"], _, _ = _classify(pairs["B_minus_A"], res["B"])
    elif "C" in arms:
        cls_c, ok_c, _ = _classify(pairs["C_minus_A"], res["C"])
        cls_b, ok_b, _ = _classify(pairs["B_minus_A"], res["B"])
        out["P0_A_DAILY"] = cls_c
        out["P0_A_5session_reclassified"] = cls_b
        pa, pc = pairs["B_minus_A"], pairs["C_minus_A"]
        worse = [res["C"]["rank_ic_mean"] < res["B"]["rank_ic_mean"],
                 res["C"]["positive_ic_frequency"] < res["B"]["positive_ic_frequency"],
                 res["C"]["top20_minus_bottom20_fwd20"] < res["B"]["top20_minus_bottom20_fwd20"],
                 res["C"]["consecutive_refit_rank_corr"] < res["B"]["consecutive_refit_rank_corr"],
                 res["C"]["avg_one_way_turnover"] > res["B"]["avg_one_way_turnover"],
                 res["C"]["max_drawdown_net60"] < res["B"]["max_drawdown_net60"],
                 res["C"]["net100_sharpe_like"] < res["B"]["net100_sharpe_like"]]
        n_worse = int(sum(worse))
        if ok_c:
            dens = "NONE"
        elif pc["d_ic_mean"] <= pa["d_ic_mean"] - 0.005 and n_worse >= 4:
            dens = "DENSITY_RELATED"
        elif (not ok_b) and (pc["d_ic_mean"] >= pa["d_ic_mean"] + 0.005 or ok_c):
            dens = "NON_MONOTONIC"
        else:
            dens = "INCONCLUSIVE"
        out["REFIT_DENSITY_EFFECT"] = dens
        out["density_components_C_worse_than_B"] = n_worse
        dn, dd, dt = abs(pc["d_net100_sharpe"]), -pc["d_max_dd_pp"], pc["d_turnover"]
        if dn <= 0.15 and dd <= 1.0 and dt <= 0.05:
            band = "ABSORBS_DIFFERENCE"
        elif dn <= 0.30 and dd <= 2.0 and dt <= 0.10:
            band = "PARTIALLY_ABSORBS_DIFFERENCE"
        else:
            band = "FAILS_TO_ABSORB_DIFFERENCE"
        out["BAND10_ROBUSTNESS"] = band
        out["val_vs_oos_dissociation"] = {
            arm: {"val_ic_mean": res[arm]["training"]["val_ic_mean"],
                  "oos_ic_mean": res[arm]["rank_ic_mean"]} for arm in arms}
    with open(os.path.join(OUT, f"result_{stage}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    for arm in arms:
        r = res[arm]
        print(f"[{arm}] c{r['cadence']} IC {r['rank_ic_mean']:.4f}+-{r['rank_ic_hac_se']:.4f} "
              f"pos {r['positive_ic_frequency']:.3f} spread {r['top20_minus_bottom20_fwd20']:.4f} "
              f"net100 {r['net100_sharpe_like']:.3f} DD {r['max_drawdown_net60']:.4f} "
              f"turn {r['avg_one_way_turnover']:.3f} rankcorr {r['consecutive_refit_rank_corr']:.3f} "
              f"valIC {r['training']['val_ic_mean']:.4f} val~oos rho "
              f"{r['training']['val_ic_vs_oos_block_ic_spearman']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="P0_A_prime",
                    choices=["P0_A_prime", "P0_A", "P0_B"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--deadline", default=None, help="HH:MM wall clock; stop launching fits after")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.evaluate:
        evaluate(a.stage)
    else:
        run_fits(a.stage, dry_run=a.dry_run, deadline=a.deadline)


if __name__ == "__main__":
    main()
