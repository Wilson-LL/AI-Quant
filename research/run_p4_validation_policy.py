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
                # EXACT: E optimizer epochs on train+val, final-epoch weights,
                # no early stopping, no best-checkpoint restore (see fit_fixed_epochs)
                net, info = fit_fixed_epochs(tte, Xg, yg, tr_all, dr, cfg, seed, E)
                vic = float("nan")
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


# ------------------------------------------------------------ exact Policy D (research-only)

def fit_fixed_epochs(tte, Xg, yg, tr_all, date_ranks, cfg, seed, E):
    """Train from scratch on tr_all for EXACTLY E optimizer epochs and return
    the weights after epoch E — no early stopping, no best-checkpoint
    restore — WITHOUT modifying the production trainer.

    Mechanism: fit_one restores the best-validation epoch only when a
    finite validation IC was ever observed. Passing a 4-row sentinel
    (rows taken from tr_all itself, so no extra information) makes
    _rank_ic return NaN every epoch (groups with < 5 rows are skipped);
    best_state therefore stays None, no restore happens, and with
    patience set to 1e9 and min_epochs == max_epochs == E the loop runs
    exactly E epochs. Seeding, batching, AMP and the optimizer are the
    production code paths (same fit_one)."""
    import numpy as np
    sentinel = np.asarray(tr_all)[:4]
    cfg_d = dict(cfg, patience=10 ** 9)
    net, vic, info = tte.fit_one(Xg, yg, tr_all, sentinel, date_ranks[sentinel],
                                 cfg_d, seed=seed, max_epochs=E, min_epochs=E)
    assert info["epochs_run"] == E, f"expected {E} epochs, ran {info['epochs_run']}"
    assert vic == -1e9, "a finite validation IC was observed — restore may have happened"
    return net, info


# ------------------------------------------------------------ P4-B0 per-epoch curve

def epoch_curve(tte, Xg, yg, tr, va, va_dates, cfg, seed, horizon_epochs,
                oos_idx, oos_dates, oos_fwd):
    """Policy-A split, early stopping disabled, `horizon_epochs` epochs.
    After EVERY epoch record train loss, validation IC, validation MSE,
    the subsequent OOS block IC and OOS prediction dispersion. The OOS
    numbers are computed inside a research-only wrapper of predict_idx
    and never flow into optimisation or checkpoint choice (the returned
    net is discarded)."""
    import numpy as np
    import pandas as pd
    import torch
    cfg_b = dict(cfg, patience=10 ** 9)
    va_t = torch.as_tensor(np.asarray(va), device=Xg.device)
    oos_t = torch.as_tensor(np.asarray(oos_idx), device=Xg.device)
    yv = yg[va_t]
    recs = []
    orig = tte.predict_idx

    def hooked(net, X, ii, batch=8192):
        res = orig(net, X, ii, batch)
        if torch.is_tensor(ii) and len(ii) == len(va_t) and bool((ii == va_t).all()):
            val_loss = float(((res - yv) ** 2).mean())
            p = orig(net, X, oos_t, batch).cpu().numpy()
            b = pd.DataFrame({"date": oos_dates, "pred": p, "fwd": oos_fwd}).dropna()
            ic = float(b.groupby("date").apply(
                lambda g: g["pred"].rank().corr(g["fwd"].rank()), include_groups=False).mean())
            recs.append({"val_loss": val_loss, "oos_ic": ic, "oos_pred_std": float(np.std(p))})
        return res
    tte.predict_idx = hooked
    try:
        net, vic, info = tte.fit_one(Xg, yg, tr, va, va_dates, cfg_b, seed=seed,
                                     max_epochs=horizon_epochs, min_epochs=horizon_epochs)
    finally:
        tte.predict_idx = orig
    assert info["epochs_run"] == horizon_epochs == len(recs), "an epoch was not recorded"
    out = []
    for h, r in zip(info["history"], recs):
        out.append({"epoch": h["epoch"] + 1, "train_loss": h["train_loss"],
                    "val_ic": h["val_ic"], **r})
    del net
    return out


def simulate_production_stop(val_ics, patience=3, min_epochs=2):
    """Epoch (1-based) the production early-stopping rule would have kept."""
    best, best_ep, no_imp = -1e9, 1, 0
    for i, v in enumerate(val_ics):
        if v == v and v > best:
            best, best_ep, no_imp = v, i + 1, 0
        else:
            no_imp += 1
            if no_imp >= patience and i + 1 >= min_epochs:
                break
    return best_ep


def b0_refit_dates(dates, start, end, cadence, n):
    """Deterministic: n evenly spaced refit block-starts across the interval."""
    import numpy as np
    refits, _, _ = plan_refits(dates, start, end, cadence)
    pick = [refits[int(round(i))] for i in np.linspace(0, len(refits) - 1, n)]
    return pick


def run_epoch_diag(dry_run=False):
    import torch
    import train_transformer_eod as tte
    from dataset_transformer_eod import build_dataset, matured_train_val
    import audit_v17_signal as A

    spec = load_spec()
    b0 = spec["P4_B0"]
    iv = spec["evaluation_interval"]
    H = int(b0["epoch_horizon"])
    cfg = tte.PRESETS[PRESET]
    chash = cfg_hash(cfg)
    b0_dir = os.path.join(OUT, "b0")
    os.makedirs(b0_dir, exist_ok=True)
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,), verbose=False)
    dates, dr, stocks = data["dates"], data["date_rank"], np.asarray(data["stocks"])
    pick = b0_refit_dates(dates, iv["start"], iv["end"], CADENCE, int(b0["n_refits"]))
    pick_dates = [str(dates[r])[:10] for r in pick]
    assert pick_dates == b0["refit_dates"], f"refit dates differ from the frozen spec: {pick_dates}"
    _, _, e_rank = plan_refits(dates, iv["start"], iv["end"], CADENCE)
    todo = [(r0, s) for r0 in pick for s in SEEDS
            if not os.path.isfile(os.path.join(b0_dir, f"b0_{str(dates[r0])[:10]}_c{CADENCE}_s{s}_{TARGET}_H{H}_{chash}.json"))]
    print(f"[p4-b0] refits {len(pick)} x seeds {len(SEEDS)} x epochs {H}; fits missing {len(todo)}; "
          f"~{len(todo) * b0['seconds_per_fit_estimate'] / 3600:.2f} h")
    if dry_run:
        return
    tte.require_cuda()
    Xg = tte.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][TARGET], -1, 1)), device=Xg.device)
    wide = A.close_matrix()
    f20 = A.fwd_returns(wide, 20).rename("fwd").reset_index()
    f20.columns = ["date", "stock", "fwd"]
    fwd_map = f20.set_index(["date", "stock"])["fwd"]
    t_all = time.time()
    for r0, seed in todo:
        refit_rank = r0 - 1
        block_end = min(r0 + CADENCE - 1, e_rank)
        tr, va, _ = matured_train_val(data, TARGET, refit_rank, HORIZON)
        oos_idx = np.nonzero((dr >= r0) & (dr <= block_end))[0]
        oos_dates = dates[dr[oos_idx]]
        oos_fwd = np.array([fwd_map.get((d, s), np.nan)
                            for d, s in zip(oos_dates, stocks[data["stock_idx"][oos_idx]])])
        # separation guarantees (also unit-tested)
        assert int(data["label_end_rank"][HORIZON][tr].max()) <= refit_rank
        assert int(data["label_end_rank"][HORIZON][va].max()) <= refit_rank
        assert int(dr[oos_idx].min()) > refit_rank
        t0 = time.time()
        curve = epoch_curve(tte, Xg, yg, tr, va, dr[va], cfg, seed, H, oos_idx, oos_dates, oos_fwd)
        rec = {"refit_date": str(dates[r0])[:10], "seed": seed, "horizon": H,
               "train_end": str(dates[int(dr[tr].max())])[:10],
               "val_start": str(dates[int(dr[va].min())])[:10],
               "val_end": str(dates[int(dr[va].max())])[:10],
               "oos_start": str(dates[r0])[:10], "oos_end": str(dates[block_end])[:10],
               "purge_sessions": int(dr[va].min()) - int(dr[tr].max()) - 1,
               "production_selected_epoch": simulate_production_stop([c["val_ic"] for c in curve]),
               "train_s": round(time.time() - t0, 1), "curve": curve}
        with open(os.path.join(b0_dir, f"b0_{rec['refit_date']}_c{CADENCE}_s{seed}_{TARGET}_H{H}_{chash}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[p4-b0] {rec['refit_date']} s{seed}: {H} epochs in {rec['train_s']:.0f}s "
              f"(elapsed {(time.time()-t_all)/60:.1f} min)", flush=True)
        torch.cuda.empty_cache()
    print(f"[p4-b0] done in {(time.time()-t_all)/3600:.2f} h")


def evaluate_b0():
    import glob
    from run_p0_refit_parity import hac_se  # noqa: F401  (paired SE helper reuse)
    spec = load_spec()
    b0 = spec["P4_B0"]
    comps = [int(k) for k in b0["fixed_epoch_comparators"]]
    kstar = int(b0["primary_fixed_epoch"])
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "b0", "b0_*.json")))]
    rows = []
    for r in recs:
        c = pd.DataFrame(r["curve"])
        vb = int(c.loc[c["val_ic"].idxmax(), "epoch"])
        ob = int(c.loc[c["oos_ic"].idxmax(), "epoch"])
        ps = int(r["production_selected_epoch"])
        oos = c.set_index("epoch")["oos_ic"]
        row = {"refit_date": r["refit_date"], "seed": r["seed"],
               "val_best_epoch": vb, "oos_best_epoch": ob, "prod_selected_epoch": ps,
               "oos_val_best": float(oos[vb]), "oos_prod_selected": float(oos[ps]),
               "oos_oracle": float(oos[ob]),
               "within_refit_spearman": float(c["val_ic"].rank().corr(c["oos_ic"].rank())),
               "within_refit_pearson": float(c["val_ic"].corr(c["oos_ic"]))}
        for k in comps:
            row[f"oos_fixed_{k}"] = float(oos[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "b0_summary.csv"), index=False)
    allc = pd.concat([pd.DataFrame(r["curve"]).assign(refit_date=r["refit_date"], seed=r["seed"]) for r in recs])
    allc.to_csv(os.path.join(OUT, "b0_epoch_curves.csv"), index=False)

    def paired(a, b):
        d = (df[a] - df[b])
        by_refit = d.groupby(df["refit_date"]).mean()      # cluster by refit (fits share the OOS block)
        se = float(by_refit.std(ddof=1) / np.sqrt(len(by_refit))) if len(by_refit) > 1 else np.nan
        return {"mean": float(d.mean()), "se_refit_clustered": se, "frac_positive": float((d > 0).mean()),
                "n_pairs": int(len(d)), "n_refits": int(len(by_refit))}
    out = {"n_fits": int(len(df)), "horizon": b0["epoch_horizon"],
           "aggregate": {"spearman_val_oos_all_epochs": float(allc["val_ic"].rank().corr(allc["oos_ic"].rank())),
                         "pearson_val_oos_all_epochs": float(allc["val_ic"].corr(allc["oos_ic"]))},
           "within_refit_spearman": {"median": float(df["within_refit_spearman"].median()),
                                     "mean": float(df["within_refit_spearman"].mean()),
                                     "iqr": [float(df["within_refit_spearman"].quantile(.25)), float(df["within_refit_spearman"].quantile(.75))],
                                     "frac_positive": float((df["within_refit_spearman"] > 0).mean()),
                                     "by_seed": {str(k): float(v) for k, v in df.groupby("seed")["within_refit_spearman"].median().items()},
                                     "by_refit": {k: float(v) for k, v in df.groupby("refit_date")["within_refit_spearman"].mean().items()}},
           "selected_epochs": {"val_best": df["val_best_epoch"].describe().to_dict(),
                               "prod_selected": df["prod_selected_epoch"].describe().to_dict(),
                               "oos_oracle": df["oos_best_epoch"].describe().to_dict(),
                               "abs_gap_val_best_vs_oracle_median": float((df["val_best_epoch"] - df["oos_best_epoch"]).abs().median()),
                               "frac_val_best_within_2_of_oracle": float(((df["val_best_epoch"] - df["oos_best_epoch"]).abs() <= 2).mean())},
           "oos_ic": {"val_best": float(df["oos_val_best"].mean()), "prod_selected": float(df["oos_prod_selected"].mean()),
                      "oracle": float(df["oos_oracle"].mean()),
                      "fixed": {str(k): {"mean": float(df[f"oos_fixed_{k}"].mean()),
                                         "pos_frac": float((df[f"oos_fixed_{k}"] > 0).mean()),
                                         "std_across_fits": float(df[f"oos_fixed_{k}"].std())} for k in comps}},
           "regret_to_oracle": {"val_best": float((df["oos_oracle"] - df["oos_val_best"]).mean()),
                                "prod_selected": float((df["oos_oracle"] - df["oos_prod_selected"]).mean()),
                                **{f"fixed_{k}": float((df["oos_oracle"] - df[f"oos_fixed_{k}"]).mean()) for k in comps}},
           "paired_val_best_minus_fixed": {str(k): paired("oos_val_best", f"oos_fixed_{k}") for k in comps},
           "paired_prod_selected_minus_fixed": {str(k): paired("oos_prod_selected", f"oos_fixed_{k}") for k in comps},
           "by_seed_val_best_minus_primary": {str(s): float((g["oos_val_best"] - g[f"oos_fixed_{kstar}"]).mean()) for s, g in df.groupby("seed")},
           "by_refit_val_best_minus_primary": {k: float((g["oos_val_best"] - g[f"oos_fixed_{kstar}"]).mean()) for k, g in df.groupby("refit_date")}}
    # frozen classification (spec.P4_B0.classification)
    pk = out["paired_prod_selected_minus_fixed"][str(kstar)]
    med = out["within_refit_spearman"]["median"]
    if pk["mean"] >= 0.010 and pk["mean"] > pk["se_refit_clustered"] and pk["frac_positive"] >= 0.60 and med >= 0.30:
        cls = "EARLY_STOPPING_INFORMATIVE"
    elif pk["mean"] <= 0.003 and med <= 0.10:
        cls = "EARLY_STOPPING_UNINFORMATIVE"
    else:
        cls = "EARLY_STOPPING_WEAK"
    out["classification"] = cls
    out["primary_comparator"] = kstar
    with open(os.path.join(OUT, "result_P4_B0.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


# ------------------------------------------------------------ P4-B0-LONG (training-dynamics diagnostic)

def epoch_curve_long(tte, Xg, yg, tr, va, va_dates, cfg, seed, horizon_epochs,
                     oos_idx, oos_dates, oos_fwd):
    """Like epoch_curve but also records per-epoch gradient norm (mean of
    the per-step total norms returned by clip_grad_norm_, wrapped in this
    process only), weight norm, and RETAINS the validation and OOS
    predictions of every epoch. Early stopping disabled; no restore; the
    OOS block never influences optimisation."""
    import numpy as np
    import pandas as pd
    import torch
    cfg_b = dict(cfg, patience=10 ** 9)
    va_t = torch.as_tensor(np.asarray(va), device=Xg.device)
    oos_t = torch.as_tensor(np.asarray(oos_idx), device=Xg.device)
    yv = yg[va_t]
    recs, val_preds, oos_preds = [], [], []
    grad_norms = []
    orig_pred = tte.predict_idx
    orig_clip = torch.nn.utils.clip_grad_norm_

    def clip_hook(params, max_norm, *a, **k):
        gn = orig_clip(params, max_norm, *a, **k)
        grad_norms.append(float(gn))
        return gn

    def pred_hook(net, X, ii, batch=8192):
        res = orig_pred(net, X, ii, batch)
        if torch.is_tensor(ii) and len(ii) == len(va_t) and bool((ii == va_t).all()):
            val_loss = float(((res - yv) ** 2).mean())
            p = orig_pred(net, X, oos_t, batch).cpu().numpy()
            b = pd.DataFrame({"date": oos_dates, "pred": p, "fwd": oos_fwd}).dropna()
            ic = float(b.groupby("date").apply(
                lambda g: g["pred"].rank().corr(g["fwd"].rank()), include_groups=False).mean())
            wn = float(torch.sqrt(sum((q.detach().float() ** 2).sum() for q in net.parameters())))
            recs.append({"val_loss": val_loss, "oos_ic": ic, "oos_pred_std": float(np.std(p)),
                         "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                         "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
                         "weight_norm": wn})
            grad_norms.clear()
            val_preds.append(res.cpu().numpy().astype(np.float32))
            oos_preds.append(p.astype(np.float32))
        return res
    tte.predict_idx = pred_hook
    torch.nn.utils.clip_grad_norm_ = clip_hook
    try:
        net, vic, info = tte.fit_one(Xg, yg, tr, va, va_dates, cfg_b, seed=seed,
                                     max_epochs=horizon_epochs, min_epochs=horizon_epochs)
    finally:
        tte.predict_idx = orig_pred
        torch.nn.utils.clip_grad_norm_ = orig_clip
    assert info["epochs_run"] == horizon_epochs == len(recs), "an epoch was not recorded"
    curve = [{"epoch": h["epoch"] + 1, "train_loss": h["train_loss"], "val_ic": h["val_ic"], **r}
             for h, r in zip(info["history"], recs)]
    del net
    return curve, np.stack(val_preds), np.stack(oos_preds)


def run_long_epoch(dry_run=False):
    import torch
    import train_transformer_eod as tte
    from dataset_transformer_eod import build_dataset, matured_train_val
    import audit_v17_signal as A

    spec = load_spec()
    L = spec["P4_B0_LONG"]
    iv = spec["evaluation_interval"]
    H = int(L["epoch_horizon"])
    cfg = tte.PRESETS[PRESET]
    chash = cfg_hash(cfg)
    ldir = os.path.join(OUT, "b0_long")
    os.makedirs(ldir, exist_ok=True)
    data = build_dataset(FEATURE_SET, seq_len=cfg["seq_len"], horizons=(HORIZON,), verbose=False)
    dates, dr, stocks = data["dates"], data["date_rank"], np.asarray(data["stocks"])
    refits, _, e_rank = plan_refits(dates, iv["start"], iv["end"], CADENCE)
    by_date = {str(dates[r])[:10]: r for r in refits}
    pick = [by_date[d] for d in L["refit_dates"]]
    todo = [(r0, s) for r0 in pick for s in SEEDS
            if not os.path.isfile(os.path.join(ldir, f"long_{str(dates[r0])[:10]}_c{CADENCE}_s{s}_{TARGET}_H{H}_{chash}.json"))]
    sec = float(L["seconds_per_epoch_estimate"]) * H
    print(f"[p4-long] refits {len(pick)} x seeds {len(SEEDS)} x epochs {H}; fits missing {len(todo)}; "
          f"~{sec:.0f} s/fit -> ~{len(todo) * sec / 3600:.2f} h")
    if dry_run:
        return
    tte.require_cuda()
    Xg = tte.to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][TARGET], -1, 1)), device=Xg.device)
    wide = A.close_matrix()
    f20 = A.fwd_returns(wide, 20).rename("fwd").reset_index()
    f20.columns = ["date", "stock", "fwd"]
    fwd_map = f20.set_index(["date", "stock"])["fwd"]
    t_all = time.time()
    for r0, seed in todo:
        refit_rank = r0 - 1
        block_end = min(r0 + CADENCE - 1, e_rank)
        tr, va, _ = matured_train_val(data, TARGET, refit_rank, HORIZON)
        oos_idx = np.nonzero((dr >= r0) & (dr <= block_end))[0]
        oos_dates = dates[dr[oos_idx]]
        oos_fwd = np.array([fwd_map.get((d, s), np.nan)
                            for d, s in zip(oos_dates, stocks[data["stock_idx"][oos_idx]])])
        assert int(data["label_end_rank"][HORIZON][tr].max()) <= refit_rank
        assert int(data["label_end_rank"][HORIZON][va].max()) <= refit_rank
        assert int(dr[oos_idx].min()) > refit_rank
        t0 = time.time()
        curve, vp, op = epoch_curve_long(tte, Xg, yg, tr, va, dr[va], cfg, seed, H,
                                         oos_idx, oos_dates, oos_fwd)
        stem = f"long_{str(dates[r0])[:10]}_c{CADENCE}_s{seed}_{TARGET}_H{H}_{chash}"
        rec = {"refit_date": str(dates[r0])[:10], "seed": seed, "horizon": H,
               "train_end": str(dates[int(dr[tr].max())])[:10],
               "val_start": str(dates[int(dr[va].min())])[:10], "val_end": str(dates[int(dr[va].max())])[:10],
               "oos_start": str(dates[r0])[:10], "oos_end": str(dates[block_end])[:10],
               "production_selected_epoch": simulate_production_stop([c["val_ic"] for c in curve]),
               "train_s": round(time.time() - t0, 1), "curve": curve}
        with open(os.path.join(ldir, stem + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        np.savez_compressed(os.path.join(ldir, stem + "_preds.npz"), val_preds=vp, oos_preds=op,
                            oos_idx=oos_idx, va_idx=np.asarray(va))
        print(f"[p4-long] {rec['refit_date']} s{seed}: {H} epochs in {rec['train_s']:.0f}s "
              f"(elapsed {(time.time()-t_all)/60:.1f} min)", flush=True)
        torch.cuda.empty_cache()
    print(f"[p4-long] done in {(time.time()-t_all)/3600:.2f} h")


def evaluate_long():
    import glob
    spec = load_spec()
    L = spec["P4_B0_LONG"]
    cps = [int(k) for k in L["checkpoint_epochs"]]
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "b0_long", "long_*.json")))]
    allc = pd.concat([pd.DataFrame(r["curve"]).assign(refit_date=r["refit_date"], seed=r["seed"]) for r in recs])
    allc.to_csv(os.path.join(OUT, "b0_long_epoch_curves.csv"), index=False)
    mean = allc.groupby("epoch")[["train_loss", "val_ic", "val_loss", "oos_ic", "oos_pred_std",
                                  "grad_norm_mean", "weight_norm"]].mean()
    rows = []
    for r in recs:
        c = pd.DataFrame(r["curve"]).set_index("epoch")
        ps = int(r["production_selected_epoch"])
        row = {"refit_date": r["refit_date"], "seed": r["seed"], "prod_selected_epoch": ps,
               "oos_prod_selected": float(c.loc[ps, "oos_ic"]),
               "oos_early_regime_2_5": float(c.loc[2:5, "oos_ic"].mean()),
               "oos_late_50_100": float(c.loc[[50, 75, 100], "oos_ic"].mean()),
               "oos_oracle": float(c["oos_ic"].max()), "oracle_epoch": int(c["oos_ic"].idxmax()),
               "oos_late_max": float(c.loc[30:, "oos_ic"].max()), "oos_late_max_epoch": int(c.loc[30:, "oos_ic"].idxmax())}
        for k in cps:
            row[f"oos_{k}"] = float(c.loc[k, "oos_ic"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "b0_long_summary.csv"), index=False)
    early = float(df["oos_early_regime_2_5"].mean())
    late = float(df["oos_late_50_100"].mean())
    e3, prod = float(df["oos_3"].mean()), float(df["oos_prod_selected"].mean())
    # frozen classification (spec.P4_B0_LONG.classification)
    per_fit_late_beats_e3 = int((df["oos_late_50_100"] >= df["oos_3"] + 0.010).sum())
    per_fit_late_beats_prod = int((df["oos_late_50_100"] >= df["oos_prod_selected"] + 0.010).sum())
    late_cps = [float(df[f"oos_{k}"].mean()) for k in (50, 75, 100)]
    if late >= max(e3, prod) + 0.010 and per_fit_late_beats_e3 >= 7 and per_fit_late_beats_prod >= 7 \
            and all(df.groupby("seed")["oos_late_50_100"].mean() >= df.groupby("seed")["oos_3"].mean() + 0.010) \
            and all(df.groupby("refit_date")["oos_late_50_100"].mean() >= df.groupby("refit_date")["oos_3"].mean() + 0.010):
        cls = "LONG_EPOCH_PROMISING"
    elif all(v <= early - 0.020 for v in late_cps) and float(df["oos_late_max"].mean()) < early - 0.010:
        cls = "LONG_EPOCH_NO_BENEFIT"
    else:
        cls = "LONG_EPOCH_POSSIBLE_RECOVERY"
    out = {"n_fits": int(len(df)), "horizon": L["epoch_horizon"], "refit_dates": L["refit_dates"],
           "checkpoint_mean_oos_ic": {str(k): float(df[f"oos_{k}"].mean()) for k in cps},
           "checkpoint_mean_val_ic": {str(k): float(mean.loc[k, "val_ic"]) for k in cps},
           "checkpoint_mean_train_loss": {str(k): float(mean.loc[k, "train_loss"]) for k in cps},
           "checkpoint_mean_pred_std": {str(k): float(mean.loc[k, "oos_pred_std"]) for k in cps},
           "early_regime_2_5_mean": early, "late_50_100_mean": late, "epoch3_mean": e3,
           "prod_selected_mean": prod, "prod_selected_epochs": df["prod_selected_epoch"].tolist(),
           "oracle_epochs": df["oracle_epoch"].tolist(), "late_max_epochs": df["oos_late_max_epoch"].tolist(),
           "per_fit_late_beats_epoch3_by_0.01": per_fit_late_beats_e3,
           "per_fit_late_beats_prod_by_0.01": per_fit_late_beats_prod,
           "by_seed": {str(s): {"epoch3": float(g["oos_3"].mean()), "late": float(g["oos_late_50_100"].mean()),
                                "e100": float(g["oos_100"].mean())} for s, g in df.groupby("seed")},
           "by_refit": {k: {"epoch3": float(g["oos_3"].mean()), "late": float(g["oos_late_50_100"].mean()),
                            "e100": float(g["oos_100"].mean())} for k, g in df.groupby("refit_date")},
           "classification": cls}
    with open(os.path.join(OUT, "result_P4_B0_LONG.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(mean.loc[cps].round(4).to_string())


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
    ap.add_argument("--stage", default="fits", choices=["fits", "epoch_diag", "long_epoch"])
    ap.add_argument("--evaluate-b0", action="store_true")
    ap.add_argument("--evaluate-long", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--deadline", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.evaluate_b0:
        evaluate_b0()
    elif a.evaluate_long:
        evaluate_long()
    elif a.evaluate:
        evaluate()
    elif a.stage == "long_epoch":
        run_long_epoch(dry_run=a.dry_run)
    elif a.stage == "epoch_diag":
        run_epoch_diag(dry_run=a.dry_run)
    else:
        run_fits(dry_run=a.dry_run, deadline=a.deadline)


if __name__ == "__main__":
    main()
