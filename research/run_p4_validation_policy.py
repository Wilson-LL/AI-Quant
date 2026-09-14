"""v17 P4-B — validation / final-training policy screen (paired; 5-session
cadence; seeds {0,1,2}; resumable; isolated under reports/model_audit/v17/p4/).

Implements reports/model_audit/v17/p4_validation_policy_spec.json.
Policies: A_current (reuses P0 arm-B fits), B_recent126, C_recent63,
D_calibrate_then_train_all. Plus --stage epoch_diag (P4-B0).

  python research/run_p4_validation_policy.py --dry-run
  python research/run_p4_validation_policy.py               # fits
  python research/run_p4_validation_policy.py --stage epoch_diag
  python research/run_p4_validation_policy.py --evaluate

Production code is never modified; the epoch diagnostic wraps
train_transformer_eod.predict_idx inside this process only.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V17 = os.path.join(ROOT, "reports", "model_audit", "v17")
SPEC_P = os.path.join(V17, "p4_validation_policy_spec.json")
OUT = os.path.join(V17, "p4")
FITS = os.path.join(OUT, "fits")
P0_FITS = os.path.join(V17, "p0", "fits")
FEATURE_SET, TARGET, HORIZON, PRESET, CADENCE = "close_only", "tgt_rank_20", 20, "B", 5
SEEDS = (0, 1, 2)
POLICIES = ("A_current", "B_recent126", "C_recent63", "D_calibrate_then_train_all")
PURGE = HORIZON + 1


def load_spec():
    with open(SPEC_P) as f:
        return json.load(f)


def cfg_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def plan_refits(dates, start, end, cadence):
    s = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(start))))
    e = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(end)), side="right")) - 1
    return list(range(s, e + 1, cadence)), s, e


def key(policy, refit_date, seed, chash):
    return f"{policy}_{refit_date}_c{CADENCE}_s{seed}_{TARGET}_{chash}.npz"


# ------------------------------------------------------------ split policies

def split_recent(data, refit_rank, n_val, horizon=HORIZON):
    """Validation = most recent n_val matured sessions (contiguous, ending
    PURGE-1 sessions before refit_rank), train = matured dates ending
    PURGE sessions before the validation block. No row after refit_rank
    contributes a label (label_end <= refit_rank)."""
    dr = data["date_rank"]
    le = data["label_end_rank"][horizon]
    y = data["targets"][TARGET]
    usable = (le <= refit_rank) & np.isfinite(y)
    idx = np.nonzero(usable)[0]
    d_sorted = np.array(sorted(np.unique(dr[idx])))
    if len(d_sorted) < n_val + 200:
        raise ValueError("not enough matured history")
    v_lo = d_sorted[-n_val]
    va = idx[dr[idx] >= v_lo]
    tr = idx[dr[idx] <= v_lo - PURGE]
    assert int(le[tr].max()) <= refit_rank and int(le[va].max()) <= refit_rank
    assert int(dr[va].min()) - int(dr[tr].max()) >= PURGE
    return tr, va


def split_for(policy, data, refit_rank):
    from dataset_transformer_eod import matured_train_val
    if policy == "A_current":
        tr, va, _ = matured_train_val(data, TARGET, refit_rank, HORIZON)
        return tr, va
    if policy in ("B_recent126", "D_calibrate_then_train_all"):
        return split_recent(data, refit_rank, 126)
    if policy == "C_recent63":
        return split_recent(data, refit_rank, 63)
    raise ValueError(policy)


# ------------------------------------------------------------ fits

def run_fits(dry_run=False, deadline=None):
    import torch
    import train_transformer_eod as tte
    from dataset_transformer_eod import build_dataset

    spec = load_spec()
    iv = spec["evaluation_interval"]
    cfg = tte.PRESETS[PRESET]
    chash = cfg_hash(cfg)
    os.makedirs(FITS, exist_ok=True)
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,), verbose=False)
    dates, dr = data["dates"], data["date_rank"]
    refits, s_rank, e_rank = plan_refits(dates, iv["start"], iv["end"], CADENCE)

    # policy A: reuse the semantically identical P0 arm-B fits (copy under the P4 key)
    reused = 0
    for r0 in refits:
        for seed in SEEDS:
            src = os.path.join(P0_FITS, f"{str(dates[r0])[:10]}_c{CADENCE}_s{seed}_{TARGET}_{chash}.npz")
            dst = os.path.join(FITS, key("A_current", str(dates[r0])[:10], seed, chash))
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copyfile(src, dst)
                reused += 1
    todo = []
    for pol in ("B_recent126", "C_recent63", "D_calibrate_then_train_all"):
        for r0 in refits:
            for seed in SEEDS:
                k = key(pol, str(dates[r0])[:10], seed, chash)
                if not os.path.isfile(os.path.join(FITS, k)):
                    todo.append((pol, r0, seed, k))
    # D depends on B: order B before D (list order already does)
    sec = spec["compute"]["measured_seconds_per_fit"]
    print(f"[p4] refits {len(refits)} ({str(dates[refits[0]])[:10]}..{str(dates[refits[-1]])[:10]}), "
          f"policy-A fits reused from P0: {reused}, missing fits {len(todo)}, "
          f"estimated {len(todo) * sec / 3600:.2f} h at {sec}s/fit")
    if dry_run:
        return
    dl = None
    if deadline:
        import datetime as _dt
        hh, mm = map(int, deadline.split(":"))
        dl = _dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    tte.require_cuda()
    Xg = tte.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][TARGET], -1, 1)), device=Xg.device)
    t_all, failed, done = time.time(), [], 0
    for i, (pol, r0, seed, k) in enumerate(todo, 1):
        if dl is not None and __import__("datetime").datetime.now() >= dl:
            print(f"[p4] deadline reached after {i-1} fits - stopping (resumable)")
            break
        refit_rank = r0 - 1
        block_end = min(r0 + CADENCE - 1, e_rank)
        try:
            if pol == "D_calibrate_then_train_all":
                bk = os.path.join(FITS, key("B_recent126", str(dates[r0])[:10], seed, chash))
                if not os.path.isfile(bk):
                    raise RuntimeError("policy B fit missing for calibration")
                zb = np.load(bk, allow_pickle=True)
                hist = json.loads(str(zb["history"]))
                E = 1 + int(np.argmax([h["val_ic"] for h in hist]))
                tr, va = split_recent(data, refit_rank, 126)
                tr_all = np.concatenate([tr, va])
                cfg_d = dict(cfg, patience=10 ** 6)
                net, vic, info = tte.fit_one(Xg, yg, tr_all, va, dr[va], cfg_d, seed=seed,
                                             max_epochs=E, min_epochs=E)
                extra = {"calibrated_epochs": E}
            else:
                tr, va = split_for(pol, data, refit_rank)
                net, vic, info = tte.fit_one(Xg, yg, tr, va, dr[va], cfg, seed=seed)
                extra = {}
            score_ranks = list(range(r0, block_end + 1))
            overlap = block_end + 1 if block_end + 1 <= e_rank else None
            if overlap is not None:
                score_ranks.append(overlap)
            idx = np.nonzero(np.isin(dr, score_ranks))[0]
            preds = tte.predict_idx(net, Xg, idx).cpu().numpy()
            np.savez_compressed(
                os.path.join(FITS, k), idx=idx, pred=preds.astype(np.float32),
                date_rank=dr[idx], stock_idx=data["stock_idx"][idx],
                overlap_rank=-1 if overlap is None else overlap,
                block_start=r0, block_end=block_end, refit_rank=refit_rank,
                seed=seed, cadence=CADENCE, policy=pol, val_ic=float(vic),
                epochs=int(info["epochs_run"]), train_s=float(info["train_s"]),
                n_train=int(len(tr)), n_val=int(len(va)),
                val_first=int(dr[va].min()), val_last=int(dr[va].max()),
                train_last=int(dr[tr].max()),
                history=json.dumps(info["history"]), **extra)
            done += 1
            print(f"[p4] {i}/{len(todo)} {pol} {str(dates[r0])[:10]} s{seed}: val_ic {vic:+.4f} "
                  f"ep {info['epochs_run']} {info['train_s']:.0f}s (elapsed {(time.time()-t_all)/60:.1f} min)",
                  flush=True)
            del net
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            failed.append({"key": k, "error": repr(e)})
            print(f"[p4] FAILED {k}: {e!r}", flush=True)
    with open(os.path.join(OUT, "run_P4_B_log.json"), "w") as f:
        json.dump({"fits_missing_at_start": len(todo), "done": done, "failed": failed,
                   "policy_A_reused": reused, "elapsed_s": round(time.time() - t_all, 1),
                   "cfg_hash": chash}, f, indent=1)
    print(f"[p4] done {done}, failed {len(failed)}, {(time.time()-t_all)/3600:.2f} h")


# ------------------------------------------------------------ epoch diagnostic (P4-B0)

def run_epoch_diag(n_refits=9, max_epochs=15, dry_run=False):
    """Policy-A split; early stopping disabled; after every epoch record
    val IC, train loss and the 5-session OOS block IC (research-only)."""
    import torch
    import train_transformer_eod as tte
    from dataset_transformer_eod import build_dataset, matured_train_val
    import audit_v17_signal as A

    spec = load_spec()
    iv = spec["evaluation_interval"]
    cfg = dict(tte.PRESETS[PRESET], patience=10 ** 6)
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,), verbose=False)
    dates, dr, stocks = data["dates"], data["date_rank"], np.asarray(data["stocks"])
    refits, s_rank, e_rank = plan_refits(dates, iv["start"], iv["end"], CADENCE)
    pick = [refits[int(round(i))] for i in np.linspace(0, len(refits) - 1, n_refits)]
    out_p = os.path.join(OUT, "epoch_diag.csv")
    done = set()
    if os.path.isfile(out_p):
        prev = pd.read_csv(out_p)
        done = set(zip(prev["refit_date"], prev["seed"]))
    todo = [(r0, s) for r0 in pick for s in SEEDS if (str(dates[r0])[:10], s) not in done]
    print(f"[p4-epoch] refits {len(pick)}, fits missing {len(todo)}, ~{len(todo) * 130 / 3600:.2f} h")
    if dry_run:
        return
    tte.require_cuda()
    Xg = tte.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][TARGET], -1, 1)), device=Xg.device)
    wide = A.close_matrix()
    f20 = A.fwd_returns(wide, 20).rename("fwd").reset_index()
    f20.columns = ["date", "stock", "fwd"]
    fwd_map = f20.set_index(["date", "stock"])["fwd"]
    rows = []
    for r0, seed in todo:
        refit_rank = r0 - 1
        block_end = min(r0 + CADENCE - 1, e_rank)
        tr, va, _ = matured_train_val(data, TARGET, refit_rank, HORIZON)
        idx = np.nonzero((dr >= r0) & (dr <= block_end))[0]
        blk = pd.DataFrame({"date": dates[dr[idx]], "stock": stocks[data["stock_idx"][idx]]})
        blk["fwd"] = [fwd_map.get((d, s), np.nan) for d, s in zip(blk["date"], blk["stock"])]
        va_t = torch.as_tensor(va, device=Xg.device)
        epoch_oos = []
        orig = tte.predict_idx

        def hooked(net, X, ii, batch=8192, _orig=orig, _va=va_t, _idx=idx, _blk=blk, _eo=epoch_oos):
            res = _orig(net, X, ii, batch)
            if torch.is_tensor(ii) and len(ii) == len(_va) and bool((ii == _va).all()):
                p = _orig(net, X, _idx, batch).cpu().numpy()
                b = _blk.assign(pred=p).dropna()
                _eo.append(float(b.groupby("date").apply(
                    lambda g: g["pred"].rank().corr(g["fwd"].rank()), include_groups=False).mean()))
            return res
        tte.predict_idx = hooked
        try:
            net, vic, info = tte.fit_one(Xg, yg, tr, va, dr[va], cfg, seed=seed,
                                         max_epochs=max_epochs, min_epochs=max_epochs)
        finally:
            tte.predict_idx = orig
        for h, oos in zip(info["history"], epoch_oos):
            rows.append({"refit_date": str(dates[r0])[:10], "seed": seed, "epoch": h["epoch"] + 1,
                         "train_loss": h["train_loss"], "val_ic": h["val_ic"], "oos_block_ic": oos})
        pd.DataFrame(rows).to_csv(out_p, index=False, mode="a", header=not os.path.isfile(out_p))
        rows = []
        print(f"[p4-epoch] {str(dates[r0])[:10]} s{seed} done ({info['train_s']:.0f}s)", flush=True)
        del net
        torch.cuda.empty_cache()


# ------------------------------------------------------------ evaluation

def evaluate():
    from dataset_transformer_eod import build_dataset
    from transformer_hybrid import aux_columns, _z
    from transformer_portfolio import backtest_scores
    import audit_v17_signal as A
    import train_transformer_eod as tte
    from run_p0_refit_parity import hac_se

    spec = load_spec()
    iv = spec["evaluation_interval"]
    cfg = tte.PRESETS[PRESET]
    chash = cfg_hash(cfg)
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,), verbose=False)
    dates, stocks = data["dates"], np.asarray(data["stocks"])
    refits, _, e_rank = plan_refits(dates, iv["start"], iv["end"], CADENCE)
    wide = A.close_matrix()
    f20 = A.fwd_returns(wide, 20).reset_index()
    f20.columns = ["date", "stock", "fwd_h"]
    aux = aux_columns()
    res, per_date = {}, {}
    for pol in POLICIES:
        rows, ovl, fm = [], [], []
        for r0 in refits:
            zs = [np.load(os.path.join(FITS, key(pol, str(dates[r0])[:10], s, chash)), allow_pickle=True)
                  for s in SEEDS]
            preds = np.stack([z["pred"] for z in zs])
            z0 = zs[0]
            df = pd.DataFrame({"date_rank": z0["date_rank"], "stock": stocks[z0["stock_idx"]],
                               "score": preds.mean(0), "score_std": preds.std(0)})
            rows.append(df[df["date_rank"] <= int(z0["block_end"])])
            ov = int(z0["overlap_rank"])
            if ov >= 0:
                ovl.append(df[df["date_rank"] == ov].assign(from_refit=int(r0)))
            for z in zs:
                fm.append({"refit_date": str(dates[r0])[:10], "seed": int(z["seed"]),
                           "val_ic": float(z["val_ic"]), "epochs": int(z["epochs"])})
        panel = pd.concat(rows, ignore_index=True)
        panel["date"] = dates[panel["date_rank"].to_numpy()]
        fm = pd.DataFrame(fm)
        m = panel.merge(f20, on=["date", "stock"], how="left").merge(aux, on=["date", "stock"], how="left")
        g = m.groupby("date")
        m["z_tf"], m["z_mom"] = g["score"].transform(_z), g["mom"].transform(_z)
        m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
        ic = A.spearman_by_date(m.dropna(subset=["fwd_h"]), "score", "fwd_h")
        m["pct"] = g["score"].rank(ascending=False, pct=True)
        spread = m[m["pct"] <= 0.2].groupby("date")["fwd_h"].mean() - m[m["pct"] > 0.8].groupby("date")["fwd_h"].mean()
        per_date[pol] = pd.DataFrame({"ic": ic, "spread20": spread})
        pp = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(columns={"blend": "score"}).dropna()
        bt = backtest_scores(pp, mode="long_only", no_trade_band=0.10, cost_bps_list=(0, 60, 100), min_names=60)
        # refit shock
        sh = []
        for _, o in pd.concat(ovl).groupby("from_refit") if ovl else []:
            d = o["date_rank"].iloc[0]
            new = m[m["date_rank"] == d].set_index("stock")["score"]
            old = o.set_index("stock")["score"]
            j = new.index.intersection(old.index)
            if len(j) >= 10:
                rn, ro = new.loc[j].rank(), old.loc[j].rank()
                kk = max(3, round(0.2 * len(j)))
                tn, to = set(new.loc[j].nlargest(kk).index), set(old.loc[j].nlargest(kk).index)
                sh.append({"rank_corr": rn.corr(ro), "topq": len(tn & to) / len(tn | to)})
        sh = pd.DataFrame(sh)
        # val vs subsequent OOS (raw and time-detrended)
        blocks = []
        for r0 in refits:
            bd = [dates[r] for r in range(r0, min(r0 + CADENCE - 1, e_rank) + 1)]
            oos = ic.reindex(pd.DatetimeIndex(bd)).mean()
            v = fm[fm["refit_date"] == str(dates[r0])[:10]]["val_ic"].mean()
            blocks.append({"val": float(v), "oos": float(oos) if pd.notna(oos) else np.nan})
        b = pd.DataFrame(blocks).dropna().reset_index(drop=True)
        b["t"] = np.arange(len(b))
        def resid(y, x):
            X = np.column_stack([np.ones(len(x)), x, x ** 2])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        raw = float(b["val"].rank().corr(b["oos"].rank())) if len(b) >= 5 else np.nan
        det = (float(pd.Series(resid(b["val"].to_numpy(), b["t"].to_numpy())).rank().corr(
            pd.Series(resid(b["oos"].to_numpy(), b["t"].to_numpy())).rank())) if len(b) >= 8 else np.nan)
        res[pol] = {"rank_ic_mean": float(ic.mean()), "rank_ic_hac_se": hac_se(ic.dropna().to_numpy()),
                    "rank_ic_std": float(ic.std()), "positive_ic_frequency": float((ic > 0).mean()),
                    "spread20": float(spread.mean()), "val_oos_spearman_raw": raw,
                    "val_oos_spearman_detrended": det,
                    "val_ic_mean": float(fm["val_ic"].mean()), "epochs_mean": float(fm["epochs"].mean()),
                    "seed_dispersion": float(panel["score_std"].mean()),
                    "refit_rank_corr": float(sh["rank_corr"].mean()) if len(sh) else np.nan,
                    "refit_topq_overlap": float(sh["topq"].mean()) if len(sh) else np.nan,
                    "net60_sharpe_like": bt["net60"]["sharpe"], "net100_sharpe_like": bt["net100"]["sharpe"],
                    "max_drawdown_net60": bt["net60"]["max_dd"], "avg_one_way_turnover": bt["avg_turnover"],
                    "n_blocks": int(bt["net60"]["n"]),
                    "per_seed_ic": {}}
        # per-seed IC (for the all-seeds-agree criterion)
        for s_i, s in enumerate(SEEDS):
            ps = []
            for r0 in refits:
                z = np.load(os.path.join(FITS, key(pol, str(dates[r0])[:10], s, chash)), allow_pickle=True)
                blk = z["date_rank"] <= int(z["block_end"])
                ps.append(pd.DataFrame({"date": dates[z["date_rank"][blk]], "stock": stocks[z["stock_idx"][blk]],
                                        "score": z["pred"][blk]}))
            ps = pd.concat(ps).merge(f20, on=["date", "stock"], how="left").dropna()
            res[pol]["per_seed_ic"][str(s)] = float(A.spearman_by_date(ps, "score", "fwd_h").mean())
        panel.to_csv(os.path.join(OUT, f"panel_{pol}.csv.gz"), index=False)
    pairs = {}
    for pol in POLICIES[1:]:
        j = per_date["A_current"].join(per_date[pol], lsuffix="_A", rsuffix="_P", how="inner").dropna()
        d_ic = (j["ic_P"] - j["ic_A"]).to_numpy()
        pairs[pol] = {"n_dates": int(len(j)), "d_ic_mean": float(d_ic.mean()), "d_ic_hac_se": hac_se(d_ic),
                      "d_pos_frac": res[pol]["positive_ic_frequency"] - res["A_current"]["positive_ic_frequency"],
                      "d_spread20": res[pol]["spread20"] - res["A_current"]["spread20"],
                      "d_turnover": res[pol]["avg_one_way_turnover"] - res["A_current"]["avg_one_way_turnover"],
                      "d_max_dd_pp": 100 * (res[pol]["max_drawdown_net60"] - res["A_current"]["max_drawdown_net60"]),
                      "d_net100": res[pol]["net100_sharpe_like"] - res["A_current"]["net100_sharpe_like"],
                      "d_val_oos_detrended": (res[pol]["val_oos_spearman_detrended"]
                                              - res["A_current"]["val_oos_spearman_detrended"])}
        p, a = pairs[pol], res["A_current"]
        seeds_ok = all(res[pol]["per_seed_ic"][k] >= a["per_seed_ic"][k] for k in a["per_seed_ic"])
        promising = (p["d_ic_mean"] >= 0.005 and p["d_ic_mean"] > p["d_ic_hac_se"]
                     and p["d_pos_frac"] >= -0.02 and p["d_spread20"] >= -0.002
                     and (p["d_val_oos_detrended"] >= 0.30 or res[pol]["val_oos_spearman_detrended"] >= 0)
                     and p["d_turnover"] <= 0.05 and p["d_max_dd_pp"] >= -1.0 and seeds_ok)
        fails = sum([p["d_pos_frac"] < -0.02, p["d_spread20"] < -0.002, p["d_turnover"] > 0.05,
                     p["d_max_dd_pp"] < -1.0, not seeds_ok])
        pairs[pol]["classification"] = ("PROMISING" if promising else
                                        "NOT_PROMISING" if (p["d_ic_mean"] <= -0.005 or fails >= 2)
                                        else "VALIDATION_POLICY_INCONCLUSIVE")
    out = {"label": spec["label"], "policies": res, "paired_vs_A": pairs}
    with open(os.path.join(OUT, "result_P4_B.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="fits", choices=["fits", "epoch_diag"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--deadline", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.evaluate:
        evaluate()
    elif a.stage == "epoch_diag":
        run_epoch_diag(dry_run=a.dry_run)
    else:
        run_fits(dry_run=a.dry_run, deadline=a.deadline)


if __name__ == "__main__":
    main()
