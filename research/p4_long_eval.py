"""v17 P4-B0-LONG — amended, staged, refit-clustered evaluator (research-only).

Implements reports/model_audit/v17/p4_validation_policy_spec.json ->
P4_B0_LONG_AMENDMENT_1 (recorded before unblinding, commit c6e6a6d).

Stages must run in order; each refuses to run early:

  python research/p4_long_eval.py anchor     # LONG epochs 1-15 vs P4-B0, non-outcome fields only
  python research/p4_long_eval.py mechanism  # mechanism channels only; OOS IC is never loaded
  python research/p4_long_eval.py oos        # only after the mechanism summary is committed

The original frozen classifier (run_p4_validation_policy.evaluate_long) is
called last by the oos stage and reported as ORIGINAL_PREREGISTERED_CLASSIFIER
/ STRUCTURALLY_FLAWED / NOT_PRIMARY_EVIDENCE.

Independent unit = REFIT DATE. Seeds are replicate models inside a block and
are reported only as descriptive diagnostics. Production code is not touched.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V17 = os.path.join(ROOT, "reports", "model_audit", "v17")
P4 = os.path.join(V17, "p4")
LONG_DIR = os.path.join(P4, "b0_long")
B0_DIR = os.path.join(P4, "b0")
SPEC_P = os.path.join(V17, "p4_validation_policy_spec.json")
ANCHOR_SUMMARY = os.path.join(V17, "p4_b0_long_anchor_summary.json")
MECH_SUMMARY = os.path.join(V17, "p4_b0_long_mechanism_summary.json")
OOS_SUMMARY = os.path.join(V17, "p4_b0_long_oos_summary.json")

CHECKPOINTS = (1, 3, 5, 10, 15, 20, 30, 50, 75, 100)
META_KEYS = ("refit_date", "seed", "horizon", "train_end", "val_start", "val_end",
             "oos_start", "oos_end", "production_selected_epoch", "train_s")
SPLIT_KEYS = ("train_end", "val_start", "val_end", "oos_start", "oos_end")
ANCHOR_FIELDS = ("train_loss", "val_ic", "val_loss", "oos_pred_std")
MECH_FIELDS = ("train_loss", "val_ic", "val_loss", "oos_pred_std",
               "grad_norm_mean", "grad_norm_max", "weight_norm")
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
TARGET_SD = (1.0 / 3.0) ** 0.5
PASSING = ("PARITY_PASS", "PARITY_PASS_WITH_LATE_DRIFT")
BAND = 0.02          # materiality band (amendment trajectory classes)
SEL_BAR = 0.023      # selection-aware bar vs epoch 3 (0.010 + 0.0128 premium)


# ------------------------------------------------------------ loading

def amendment(spec_p=SPEC_P):
    with open(spec_p) as f:
        return json.load(f)["P4_B0_LONG_AMENDMENT_1"]


def verify_checksums(am, ldir=LONG_DIR):
    """Every LONG file must match the SHA-256 recorded before unblinding."""
    want = am["long_artifact_sha256"]
    bad = []
    for f, h in want.items():
        with open(os.path.join(ldir, f), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != h:
                bad.append(f)
    extra = sorted(set(os.listdir(ldir)) - set(want))
    if bad or extra:
        raise RuntimeError(f"LONG artifacts differ from the pre-unblinding checksums: changed={bad} extra={extra}")
    return len(want)


def load_long(fields, ldir=LONG_DIR):
    """LONG records with metadata plus ONLY the whitelisted per-epoch fields.
    Any other field (in particular oos_ic before stage 3) is discarded right
    after parsing and never returned, printed or written."""
    out = []
    for p in sorted(glob.glob(os.path.join(ldir, "long_*.json"))):
        with open(p) as f:
            raw = json.load(f)
        rec = {k: raw[k] for k in META_KEYS if k in raw}
        rec["stem"] = os.path.basename(p)[:-5]
        rec["curve"] = [{"epoch": int(c["epoch"]), **{k: float(c[k]) for k in fields}} for c in raw["curve"]]
        del raw
        out.append(rec)
    return out


def load_b0(b0_dir=B0_DIR):
    recs = []
    for p in sorted(glob.glob(os.path.join(b0_dir, "b0_*.json"))):
        with open(p) as f:
            recs.append(json.load(f))
    return recs


def load_val_preds(stem, ldir=LONG_DIR):
    """Validation-block predictions only. The OOS prediction array in the
    same NPZ is never accessed (NpzFile decompresses members lazily)."""
    with np.load(os.path.join(ldir, stem + "_preds.npz")) as z:
        return z["val_preds"], z["va_idx"]


def load_oos_preds(stem, ldir=LONG_DIR):
    with np.load(os.path.join(ldir, stem + "_preds.npz")) as z:
        return z["oos_preds"], z["oos_idx"]


def git_frozen(path):
    """True if `path` is tracked and identical to HEAD (committed, unmodified)."""
    rel = os.path.relpath(path, ROOT)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    return tracked and clean


# ------------------------------------------------------------ statistics

def spearman(a, b):
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def clustered(df, col, cluster="refit_date"):
    """Refit value = mean over its seeds; estimate = equal-weight mean of refit
    values; SE = SD(refit values)/sqrt(n); 95% CI with t(n-1)."""
    r = df.groupby(cluster)[col].mean()
    n = len(r)
    m = float(r.mean())
    se = float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tc = T_975.get(n - 1, 1.96)
    loro = [float(r.drop(k).mean()) for k in r.index] if n > 1 else []
    return {"mean": m, "se_refit": se, "ci95_t": [m - tc * se, m + tc * se], "t_df": n - 1,
            "n_refits": n, "positive_refits": int((r > 0).sum()),
            "by_refit": {str(k): float(v) for k, v in r.items()},
            "loro_range": [min(loro), max(loro)] if loro else None,
            "by_seed_descriptive": {str(s): float(v) for s, v in df.groupby("seed")[col].mean().items()},
            "n_fits": int(len(df)), "positive_fits": int((df[col] > 0).sum())}


def ar1_params(x):
    x = np.asarray(x, float)
    d = x - x.mean()
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    rho = 0.0
    if len(x) > 2 and d[:-1].std() > 0 and d[1:].std() > 0:
        rho = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    return sd, float(np.clip(np.nan_to_num(rho), -0.95, 0.95))


def simulate_ar1(n, sd, rho, n_sim, rng):
    z = rng.standard_normal((n_sim, n))
    e = np.empty_like(z)
    e[:, 0] = z[:, 0]
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        e[:, t] = rho * e[:, t - 1] + s * z[:, t]
    return sd * e


def late_max_null(series, refits, n_sim=20000, seed=0):
    """series: per-fit 1-D arrays of a window; refits: cluster label per fit.
    Statistic S = max - mean of the window, averaged to refit, then overall.
    Null: no systematic epoch effect inside the window (per-fit AR(1) with the
    fit's own SD and lag-1 autocorrelation, demeaned not detrended)."""
    rng = np.random.default_rng(seed)
    refits = np.asarray(refits)
    s_obs = pd.Series([float(np.max(x) - np.mean(x)) for x in series]).groupby(refits).mean().mean()
    max_obs = pd.Series([float(np.max(x)) for x in series]).groupby(refits).mean().mean()
    mean_obs = pd.Series([float(np.mean(x)) for x in series]).groupby(refits).mean().mean()
    sims = np.empty((len(series), n_sim))
    for i, x in enumerate(series):
        sd, rho = ar1_params(x)
        e = simulate_ar1(len(x), sd, rho, n_sim, rng)
        sims[i] = e.max(1) - e.mean(1)
    agg = pd.DataFrame(sims).groupby(refits).mean().mean(axis=0).to_numpy()
    return {"observed_S_max_minus_mean": float(s_obs), "null_mean_S": float(agg.mean()),
            "null_p95_S": float(np.quantile(agg, 0.95)),
            "null_percentile_of_observed": float((agg < s_obs).mean()),
            "raw_mean_max": float(max_obs), "window_mean": float(mean_obs),
            "null_corrected_max": float(max_obs - agg.mean()), "n_sim": n_sim}


# ------------------------------------------------------------ stage 1: anchor

def anchor_compare(long_recs, b0_recs, tol, expected_cells=9):
    b0 = {(r["refit_date"], int(r["seed"])): r for r in b0_recs}
    meta_bad, rows = [], []
    if len(long_recs) != expected_cells:
        meta_bad.append(f"expected {expected_cells} LONG cells, found {len(long_recs)}")
    for L in long_recs:
        key = (L["refit_date"], int(L["seed"]))
        B = b0.get(key)
        if B is None:
            meta_bad.append(f"{key}: no P4-B0 cell")
            continue
        for k in SPLIT_KEYS:
            if L.get(k) != B.get(k):
                meta_bad.append(f"{key}: {k} {L.get(k)} != {B.get(k)}")
        if int(L.get("horizon", 0)) < 15:
            meta_bad.append(f"{key}: horizon < 15")
        lc = {c["epoch"]: c for c in L["curve"]}
        bc = {int(c["epoch"]): c for c in B["curve"]}
        for e in range(1, 16):
            if e not in lc or e not in bc:
                meta_bad.append(f"{key}: epoch {e} missing")
                continue
            row = {"refit_date": key[0], "seed": key[1], "epoch": e}
            for f in ANCHOR_FIELDS:
                a, b = float(lc[e][f]), float(bc[e][f])
                row[f"d_{f}"] = a - b
                row[f"rel_{f}"] = abs(a - b) / abs(b) if b else (0.0 if a == b else float("inf"))
            rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return {"outcome": "LONG_HARNESS_PARITY_FAILURE", "metadata_problems": meta_bad}, d
    t1, t2 = tol["tier1_epochs_1_3_every_cell"], tol["tier2_epochs_4_15_every_cell"]
    pooled_tol = tol["tier2_pooled_epochs_4_15"]

    def violates(r, t):
        return (abs(r["d_train_loss"]) > t["abs_train_loss"] or abs(r["d_val_ic"]) > t["abs_val_ic"]
                or abs(r["d_val_loss"]) > t["abs_val_loss"] or r["rel_oos_pred_std"] > t["rel_oos_pred_std"])
    d["tier"] = np.where(d["epoch"] <= 3, 1, 2)
    d["violation"] = [violates(r, t1 if r["tier"] == 1 else t2) for _, r in d.iterrows()]
    early, late = d[d["tier"] == 1], d[d["tier"] == 2]
    pooled = {"mean_abs_val_ic": float(late["d_val_ic"].abs().mean()),
              "abs_mean_signed_val_ic": float(abs(late["d_val_ic"].mean()))}
    pooled_ok = (pooled["mean_abs_val_ic"] <= pooled_tol["mean_abs_val_ic"]
                 and pooled["abs_mean_signed_val_ic"] <= pooled_tol["abs_mean_signed_val_ic"])
    t1_ok = not bool(early["violation"].any())
    late_viol = late[late["violation"]]
    if meta_bad or not t1_ok or not pooled_ok:
        outcome = "LONG_HARNESS_PARITY_FAILURE"
    elif late_viol.empty:
        outcome = "PARITY_PASS"
    elif len(late_viol) <= 0.10 * len(late) and int(late_viol["epoch"].min()) >= 6:
        outcome = "PARITY_PASS_WITH_LATE_DRIFT"
    else:
        outcome = "LONG_HARNESS_PARITY_FAILURE"
    maxabs = {f"tier{t}": {f: float(g[f"d_{f}"].abs().max()) if f != "oos_pred_std"
                           else float(g[f"rel_{f}"].max()) for f in ANCHOR_FIELDS}
              for t, g in d.groupby("tier")}
    return {"outcome": outcome, "metadata_problems": meta_bad, "tier1_ok": t1_ok,
            "pooled_tier2": pooled, "pooled_ok": pooled_ok,
            "tier2_cell_epoch_violations": int(len(late_viol)), "tier2_cell_epochs": int(len(late)),
            "max_abs_diff_by_tier (oos_pred_std relative)": maxabs,
            "exact_match_rate": {f: float((d[f"d_{f}"] == 0).mean()) for f in ANCHOR_FIELDS},
            "cells_compared": int(d.groupby(["refit_date", "seed"]).ngroups)}, d


def stage_anchor():
    am = amendment()
    n = verify_checksums(am)
    long_recs = load_long(ANCHOR_FIELDS)
    res, d = anchor_compare(long_recs, load_b0(), am["anchor_tolerance"])
    res.update({"stage": "ANCHOR", "checksums_verified": n, "oos_ic_accessed": False,
                "tolerance": am["anchor_tolerance"]})
    d.to_csv(os.path.join(P4, "b0_long_anchor_diffs.csv"), index=False)
    with open(ANCHOR_SUMMARY, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "tolerance"}, indent=1, default=float))
    return res


# ------------------------------------------------------------ stage 2: mechanism

def mechanism_frame(long_recs, val_pred_fn):
    """Per-fit checkpoint table of mechanism channels plus validation-only
    prediction diagnostics (dispersion, epoch-to-epoch churn, cross-seed
    agreement)."""
    rows, vp_by = [], {}
    for r in long_recs:
        c = {x["epoch"]: x for x in r["curve"]}
        vp, va = val_pred_fn(r["stem"])
        vp_by[(r["refit_date"], int(r["seed"]))] = (vp, va)
        for e in CHECKPOINTS:
            row = {"refit_date": r["refit_date"], "seed": int(r["seed"]), "epoch": e,
                   **{f: c[e][f] for f in MECH_FIELDS if f in c[e]}}
            row["val_pred_sd"] = float(np.std(vp[e - 1]))
            row["churn"] = 1.0 - spearman(vp[e - 1], vp[e - 2]) if e >= 2 else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    agree = []
    for rd in sorted(df["refit_date"].unique()):
        seeds = sorted(s for (d, s) in vp_by if d == rd)
        vas = [vp_by[(rd, s)][1] for s in seeds]
        assert all(np.array_equal(vas[0], v) for v in vas), "seeds of one refit must share the validation rows"
        for e in CHECKPOINTS:
            pairs = [spearman(vp_by[(rd, a)][0][e - 1], vp_by[(rd, b)][0][e - 1])
                     for i, a in enumerate(seeds) for b in seeds[i + 1:]]
            agree.append({"refit_date": rd, "epoch": e, "agree": float(np.mean(pairs)) if pairs else float("nan")})
    return df, pd.DataFrame(agree)


def classify_mechanism(R):
    """R: per-refit (seed-averaged) checkpoint values, index (refit_date, epoch),
    columns train_loss, val_ic, oos_pred_std, val_pred_sd, grad_norm_mean,
    grad_norm_max, weight_norm, churn, agree. Frozen rules: amendment
    mechanism_rules."""
    refits = sorted(R.index.get_level_values(0).unique())

    def v(rd, e, col):
        return float(R.loc[(rd, e), col])
    out = {}
    # Q1 train loss
    drop = {rd: v(rd, 15, "train_loss") - v(rd, 100, "train_loss") for rd in refits}
    accel = {}
    for rd in refits:
        s1 = (v(rd, 30, "train_loss") - v(rd, 15, "train_loss")) / 15
        s2 = (v(rd, 100, "train_loss") - v(rd, 50, "train_loss")) / 50
        accel[rd] = bool(s2 < 0 and (s1 >= 0 or s2 <= 2 * s1))
    out["Q1"] = {"drop_15_to_100": drop, "continues": all(x >= 0.005 for x in drop.values()),
                 "label": "CONTINUES" if all(x >= 0.005 for x in drop.values()) else "SLOWS",
                 "acceleration_by_refit": accel, "ACCELERATION": sum(accel.values()) >= 2,
                 "in_sample_r2_at": {str(e): float(np.mean([1 - v(rd, e, "train_loss") / (1 / 3) for rd in refits]))
                                     for e in (3, 15, 30, 50, 100)}}
    # Q2 validation rank IC
    late = {rd: np.mean([v(rd, e, "val_ic") for e in (50, 75, 100)]) for rd in refits}
    r_late = {rd: late[rd] - v(rd, 15, "val_ic") for rd in refits}
    r_trough = {rd: late[rd] - min(v(rd, e, "val_ic") for e in (15, 20, 30)) for rd in refits}
    mean_at = {e: np.mean([v(rd, e, "val_ic") for rd in refits]) for e in CHECKPOINTS}
    yes = (np.mean(list(r_late.values())) >= BAND and all(x > 0 for x in r_late.values())
           and all(mean_at[e] > mean_at[15] for e in (50, 75, 100)))
    partial = sum(x >= BAND for x in r_trough.values()) >= 2
    out["Q2"] = {"R_late": {k: float(x) for k, x in r_late.items()},
                 "R_trough": {k: float(x) for k, x in r_trough.items()},
                 "mean_val_ic_at": {str(e): float(x) for e, x in mean_at.items()},
                 "label": "YES" if yes else ("PARTIAL" if partial else "NO")}

    # Q3 dispersion
    def disp(col):
        ratio = {rd: v(rd, 100, col) / v(rd, 50, col) for rd in refits}
        n_exp = sum(x >= 1.10 for x in ratio.values())
        n_con = sum(x <= 0.90 for x in ratio.values())
        n_pla = sum(0.90 < x < 1.10 for x in ratio.values())
        lab = ("EXPANDING" if n_exp >= 2 else "CONTRACTING" if n_con >= 2 else "PLATEAU" if n_pla >= 2 else "MIXED")
        return {"ratio_100_over_50": {k: float(x) for k, x in ratio.items()}, "label": lab,
                "fraction_of_target_sd_at": {str(e): float(np.mean([v(rd, e, col) for rd in refits]) / TARGET_SD)
                                             for e in (3, 15, 50, 100)}}
    out["Q3"] = {"validation_primary": disp("val_pred_sd"), "oos_inputs_label_free": disp("oos_pred_std")}
    out["Q3"]["label"] = out["Q3"]["validation_primary"]["label"]
    # Q4 gradients / prediction movement
    q4 = {}
    for rd in refits:
        gm = [v(rd, e, "grad_norm_mean") for e in CHECKPOINTS]
        gx = [v(rd, e, "grad_norm_max") for e in CHECKPOINTS]
        finite = bool(np.isfinite(R.loc[rd][["train_loss", "val_ic", "val_loss", "grad_norm_mean",
                                              "grad_norm_max", "weight_norm"]].to_numpy(float)).all())
        inst = (not all(np.isfinite(gm + gx))) or any(
            v(rd, e, "grad_norm_max") > 10 * v(rd, e, "grad_norm_mean") for e in CHECKPOINTS if e >= 20)
        a15, a100 = v(rd, 15, "agree"), v(rd, 100, "agree")
        c15, c100 = v(rd, 15, "churn"), v(rd, 100, "churn")
        g15, g100 = v(rd, 15, "grad_norm_mean"), v(rd, 100, "grad_norm_mean")
        if inst or not finite:
            lab = "INSTABILITY"
        elif c100 < 0.25 * c15 and g100 < g15:
            lab = "CONVERGED"
        elif a100 >= a15 - 0.05 and c100 < c15:
            lab = "STRUCTURED"
        elif a100 < a15 - 0.10 and c100 >= c15:
            lab = "NOISY"
        else:
            lab = "MIXED"
        q4[rd] = {"label": lab, "agree_15": a15, "agree_100": a100, "churn_15": c15, "churn_100": c100,
                  "grad_norm_mean_15": g15, "grad_norm_mean_100": g100,
                  "clipping_binding_any_checkpoint": any(v(rd, e, "grad_norm_mean") >= 1.0 for e in CHECKPOINTS)}
    labs = [x["label"] for x in q4.values()]
    maj = next((lab for lab in ("INSTABILITY", "CONVERGED", "STRUCTURED", "NOISY", "MIXED")
                if labs.count(lab) >= 2), "MIXED")
    out["Q4"] = {"by_refit": q4, "label": maj, "instability_any_refit": "INSTABILITY" in labs}
    # Q5 weight norm
    tr5 = {}
    for rd in refits:
        s1 = (v(rd, 30, "weight_norm") - v(rd, 15, "weight_norm")) / 15
        s2 = (v(rd, 100, "weight_norm") - v(rd, 50, "weight_norm")) / 50
        ratio = s2 / s1 if s1 != 0 else float("inf")
        tr5[rd] = {"rate_15_30": s1, "rate_50_100": s2, "ratio": ratio,
                   "transition": bool(np.sign(s1) != np.sign(s2) or ratio > 2 or ratio < 0.5)}
    out["Q5"] = {"by_refit": tr5, "label": "TRANSITION" if sum(x["transition"] for x in tr5.values()) >= 2 else "SMOOTH"}
    # Q6
    nonic = {"Q1_ACCELERATION": out["Q1"]["ACCELERATION"],
             "Q3_PLATEAU_OR_CONTRACTING": out["Q3"]["label"] in ("PLATEAU", "CONTRACTING"),
             "Q4_CONVERGED": out["Q4"]["label"] == "CONVERGED",
             "Q5_TRANSITION": out["Q5"]["label"] == "TRANSITION"}
    k = sum(nonic.values())
    q2 = out["Q2"]["label"]
    if q2 == "YES" and k >= 1:
        cls = "MECHANISM_CLEAR_PHASE_CHANGE"
    elif q2 == "PARTIAL" or (q2 == "YES" and k == 0) or (q2 == "NO" and k >= 2):
        cls = "MECHANISM_WEAK_PHASE_CHANGE"
    else:
        cls = "MECHANISM_NO_PHASE_CHANGE"
    out["Q6"] = {"non_ic_changes": nonic, "n_non_ic_changes": k, "classification": cls}
    return out


def stage_mechanism():
    am = amendment()
    if not os.path.isfile(ANCHOR_SUMMARY):
        raise SystemExit("run the anchor stage first")
    with open(ANCHOR_SUMMARY) as f:
        anchor = json.load(f)
    if anchor["outcome"] not in PASSING:
        raise SystemExit(f"anchor outcome {anchor['outcome']}: STOP (no mechanism or OOS reading)")
    n = verify_checksums(am)
    recs = load_long(MECH_FIELDS)
    df, agree = mechanism_frame(recs, load_val_preds)
    R = df.groupby(["refit_date", "epoch"]).mean(numeric_only=True).drop(columns=["seed"])
    R = R.join(agree.set_index(["refit_date", "epoch"]))
    res = classify_mechanism(R)
    # full 100-epoch mechanism curves (no OOS IC) for plots
    curves = pd.concat([pd.DataFrame(r["curve"]).assign(refit_date=r["refit_date"], seed=r["seed"]) for r in recs])
    curves.to_csv(os.path.join(P4, "b0_long_mechanism_curves.csv"), index=False)
    overall = R.groupby("epoch").mean(numeric_only=True)
    summary = {"stage": "MECHANISM", "frozen": True, "oos_ic_accessed": False, "oos_preds_accessed": False,
               "anchor_outcome": anchor["outcome"], "checksums_verified": n,
               "classification": res["Q6"]["classification"], "answers": res,
               "checkpoint_mean_over_refits": {str(e): {k: float(x) for k, x in row.items()}
                                               for e, row in overall.iterrows()},
               "checkpoint_by_refit": {f"{rd}|{e}": {k: float(x) for k, x in row.items()}
                                       for (rd, e), row in R.iterrows()},
               "per_fit_production_selected_epoch": {r["stem"]: int(r["production_selected_epoch"]) for r in recs}}
    with open(MECH_SUMMARY, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(overall.round(5).to_string())
    print(json.dumps(res, indent=1, default=float))
    return summary


# ------------------------------------------------------------ stage 3: OOS

def trajectory_class(E3, E2_5, T15, L):
    if L >= E3 + SEL_BAR:
        return "E_LATE_IMPROVEMENT_BEYOND_EARLY_PEAK"
    if L >= E2_5 - BAND:
        return "D_FULL_RECOVERY"
    if L >= T15 + BAND:
        return "C_PARTIAL_RECOVERY"
    if L >= T15 - BAND:
        return "B_PLATEAU"
    return "A_CONTINUED_DECAY"


def long_verdict(cls, supported, mech, refit_classes, n_recovering_refits, fits_L_minus_E25,
                 refit_L_minus_E25, se_L_minus_E25, ci_low_L_minus_E3):
    """Frozen verdict rules (amendment verdict_rules_H_EPOCH_LONG), evaluated in
    the frozen precedence REJECTED, SUPPORTED, WEAKLY_SUPPORTED, INCONCLUSIVE,
    WEAKLY_CONTRADICTED, else INCONCLUSIVE.

    Operationalisation (fixed before unblinding):
    - "n/3 refits" for a class = refits whose OWN refit-level class is at least
      that class in the order A < B < C < D < E;
    - "refit directions split" = 1 or 2 of 3 refits have L > T15 (recovering).
    """
    order = "ABCDE"
    base = cls[0]
    eff = base if (base in "AB" or supported) else "B"
    mech_ge_weak = mech in ("MECHANISM_WEAK_PHASE_CHANGE", "MECHANISM_CLEAR_PHASE_CHANGE")
    m_d = float(np.mean(list(refit_L_minus_E25.values())))
    n_consistent = sum(order.index(c[0]) >= order.index(base) for c in refit_classes.values())
    if (eff in "AB" and mech == "MECHANISM_NO_PHASE_CHANGE"
            and all(x < 0 for x in refit_L_minus_E25.values())
            and sum(x < 0 for x in fits_L_minus_E25) >= 7 and m_d + se_L_minus_E25 < 0):
        return "REJECTED"
    if (base == "E" and supported and ci_low_L_minus_E3 > 0 and n_consistent == 3
            and mech == "MECHANISM_CLEAR_PHASE_CHANGE"):
        return "SUPPORTED"
    if base in "DE" and supported and n_consistent >= 2 and mech_ge_weak:
        return "WEAKLY_SUPPORTED"
    if ((base == "C" and supported) or (base in "CDE" and not supported and mech_ge_weak)
            or (0 < n_recovering_refits < 3 and mech_ge_weak)):
        return "INCONCLUSIVE"
    if eff in "AB" and mech in ("MECHANISM_NO_PHASE_CHANGE", "MECHANISM_WEAK_PHASE_CHANGE") and m_d < 0:
        return "WEAKLY_CONTRADICTED"
    return "INCONCLUSIVE"


def topq_overlap(recs, frac=0.2):
    """Label-free: mean per-date overlap of the top-quintile of seed-averaged
    OOS scores at epoch k vs epoch 3. Descriptive only (not a performance claim)."""
    from dataset_transformer_eod import build_dataset
    import train_transformer_eod as tte
    data = build_dataset("close_only", seq_len=tte.PRESETS["B"]["seq_len"], horizons=(20,), verbose=False)
    dates, dr = data["dates"], data["date_rank"]
    out = {}
    for rd in sorted({r["refit_date"] for r in recs}):
        rs = [r for r in recs if r["refit_date"] == rd]
        arrs = [load_oos_preds(r["stem"]) for r in rs]
        idx = arrs[0][1]
        assert all(np.array_equal(idx, a[1]) for a in arrs)
        dd = pd.to_datetime(dates[dr[idx]])
        lo, hi = pd.Timestamp(rs[0]["oos_start"]), pd.Timestamp(rs[0]["oos_end"])
        if not ((dd >= lo) & (dd <= hi)).all():
            out[rd] = "SKIPPED: dataset row indexing no longer matches the LONG OOS block"
            continue
        mean_p = np.mean([a[0] for a in arrs], axis=0)          # (epochs, rows)
        res = {}
        for k in CHECKPOINTS:
            ov = []
            for d in np.unique(dd):
                m = np.asarray(dd == d)
                n = int(m.sum())
                q = max(1, int(round(frac * n)))
                top3 = set(np.argsort(-mean_p[2][m])[:q])
                topk = set(np.argsort(-mean_p[k - 1][m])[:q])
                ov.append(len(top3 & topk) / q)
            res[str(k)] = float(np.mean(ov))
        out[rd] = res
    return out


def stage_oos():
    if not os.path.isfile(MECH_SUMMARY) or not git_frozen(MECH_SUMMARY):
        raise SystemExit("the mechanism summary must be committed and unmodified before OOS unblinding")
    with open(MECH_SUMMARY) as f:
        mech = json.load(f)
    am = amendment()
    verify_checksums(am)
    recs = load_long(MECH_FIELDS + ("oos_ic",))
    rows = []
    for r in recs:
        c = {x["epoch"]: x["oos_ic"] for x in r["curve"]}
        rows.append({"refit_date": r["refit_date"], "seed": int(r["seed"]),
                     **{f"oos_{e}": c[e] for e in range(1, 101)}})
    F = pd.DataFrame(rows)
    F["E3"] = F["oos_3"]
    F["E2_5"] = F[[f"oos_{e}" for e in range(2, 6)]].mean(1)
    F["T15"] = F["oos_15"]
    F["L"] = F[["oos_50", "oos_75", "oos_100"]].mean(1)
    traj = {str(k): clustered(F, f"oos_{k}") for k in CHECKPOINTS}
    for k in CHECKPOINTS:
        F[f"d3_{k}"] = F[f"oos_{k}"] - F["E3"]
        F[f"d25_{k}"] = F[f"oos_{k}"] - F["E2_5"]
    paired3 = {str(k): clustered(F, f"d3_{k}") for k in CHECKPOINTS if k != 3}
    paired25 = {str(k): clustered(F, f"d25_{k}") for k in CHECKPOINTS}
    for a, b in (("L", "E3"), ("L", "E2_5"), ("L", "T15")):
        F[f"{a}_minus_{b}"] = F[a] - F[b]
    summ = {c: clustered(F, c) for c in ("E3", "E2_5", "T15", "L", "L_minus_E3", "L_minus_E2_5", "L_minus_T15")}
    E3, E25, T15, L = (summ[c]["mean"] for c in ("E3", "E2_5", "T15", "L"))
    cls = trajectory_class(E3, E25, T15, L)
    mcls = mech["classification"]
    refit_L = summ["L"]["by_refit"]
    refit_T = summ["T15"]["by_refit"]
    same_dir = sum((refit_L[k] - refit_T[k]) > 0 for k in refit_L)
    neigh = all(traj[str(e)]["mean"] > T15 for e in (50, 75, 100))
    two_se = summ["L_minus_T15"]["mean"] > 2 * summ["L_minus_T15"]["se_refit"]
    support = {"neighbouring_checkpoints_above_T15": neigh, "refits_same_direction": int(same_dir),
               "mechanism_not_no_phase_change": mcls != "MECHANISM_NO_PHASE_CHANGE",
               "L_minus_T15_gt_2SE": bool(two_se)}
    supported = cls[0] in "CDE" and neigh and same_dir >= 2 and support["mechanism_not_no_phase_change"] and two_se
    reported_cls = cls if (cls[0] in "AB" or supported) else f"B_PLATEAU (point estimate suggests {cls}; unsupported)"
    refit_classes = {rd: trajectory_class(summ["E3"]["by_refit"][rd], summ["E2_5"]["by_refit"][rd],
                                          refit_T[rd], refit_L[rd]) for rd in refit_L}
    verdict = long_verdict(cls, supported, mcls, refit_classes, int(same_dir), list(F["L_minus_E2_5"]),
                           summ["L_minus_E2_5"]["by_refit"], summ["L_minus_E2_5"]["se_refit"],
                           summ["L_minus_E3"]["ci95_t"][0])
    nulls = {}
    for name, win in (("W1_checkpoints_20_30_50_75_100", (20, 30, 50, 75, 100)),
                      ("W2_epochs_30_to_100", tuple(range(30, 101)))):
        ser = [F.loc[i, [f"oos_{e}" for e in win]].to_numpy(float) for i in F.index]
        nl = late_max_null(ser, F["refit_date"].to_numpy())
        nl["null_corrected_max_minus_E3"] = nl["null_corrected_max"] - E3
        nl["null_corrected_max_minus_E2_5"] = nl["null_corrected_max"] - E25
        nl["raw_max_minus_E3"] = nl["raw_mean_max"] - E3
        nulls[name] = nl
    b0 = {(r["refit_date"], int(r["seed"])): {int(c["epoch"]): c["oos_ic"] for c in r["curve"]} for r in load_b0()}
    par = [abs(F.loc[i, f"oos_{e}"] - b0[(F.loc[i, "refit_date"], int(F.loc[i, "seed"]))][e])
           for i in F.index for e in range(1, 16)]
    try:
        overlap = topq_overlap(recs)
    except Exception as exc:            # secondary diagnostic only
        overlap = f"SKIPPED: {type(exc).__name__}: {exc}"
    import run_p4_validation_policy as p4
    orig = p4.evaluate_long()
    out = {"stage": "OOS", "mechanism_classification_frozen": mcls,
           "epoch3_label": "SELECTED_COMPARATOR_ON_BURNED_INTERVAL", "selection_aware_bar": SEL_BAR,
           "trajectory_by_checkpoint": traj, "paired_minus_epoch3": paired3, "paired_minus_E2_5": paired25,
           "summary_contrasts": summ, "trajectory_class_point": cls, "trajectory_class_reported": reported_cls,
           "support_checks": support, "trajectory_class_by_refit": refit_classes, "late_max_null": nulls,
           "H_EPOCH_LONG_verdict": verdict,
           "verdict_scope": "more optimizer steps under the frozen recipe (batch 1024, constant LR 3e-4, AdamW wd 1e-4, no schedule; ~22.6k-23.9k steps)",
           "secondary_oos_ic_parity_epochs_1_15_vs_B0": {"max_abs": float(np.max(par)), "mean_abs": float(np.mean(par))},
           "secondary_topq_overlap_vs_epoch3": overlap,
           "ORIGINAL_PREREGISTERED_CLASSIFIER": {"label": orig["classification"],
                                                 "status": "STRUCTURALLY_FLAWED / NOT_PRIMARY_EVIDENCE"},
           "per_fit": F[["refit_date", "seed", "E3", "E2_5", "T15", "L"]].to_dict("records")}
    with open(OOS_SUMMARY, "w") as f:
        json.dump(out, f, indent=1, default=float)
    F.to_csv(os.path.join(P4, "b0_long_oos_per_fit.csv"), index=False)
    print(json.dumps({k: out[k] for k in ("trajectory_class_point", "trajectory_class_reported",
                                          "support_checks", "H_EPOCH_LONG_verdict",
                                          "ORIGINAL_PREREGISTERED_CLASSIFIER")}, indent=1, default=float))
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"anchor": stage_anchor, "mechanism": stage_mechanism, "oos": stage_oos}.get(
        stage, lambda: sys.exit(__doc__))()
