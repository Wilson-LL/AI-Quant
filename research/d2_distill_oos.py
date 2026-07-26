"""D2 distillation OOS walkforward check (user-approved, focused).

At each walkforward refit: train the 7 teacher seeds (0-6, champion config,
byte-identical protocol to A8), train ONE student on a 50/50 blend of the
true target and the teacher-mean scores, then score the OOS block with BOTH
the student and the ensemble mean. Produces paired panels for an exact
apples-to-apples comparison; the ensemble panel also cross-checks
determinism vs the frozen SCHED_A8 panels.

No changes to train_transformer_eod.py / model.py. Cache read-only.

Usage:
  python research/d2_distill_oos.py gpu      # both windows (~2.5 h GPU)
  python research/d2_distill_oos.py compare  # CPU books + gates + JSON out
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

from train_transformer_eod import (PRESETS, fit_one, to_gpu, require_cuda,  # noqa: E402
                                   predict_idx)
from dataset_transformer_eod import build_dataset, matured_train_val  # noqa: E402

OUT_DIR = os.path.join(ROOT, "reports", "continuous_research", "d2_distillation_oos")
PANEL_DIR = os.path.join(ROOT, "reports", "transformer_gpu", "panels")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5, 6]
WINDOWS = {"champ_2023": "2023-01-01", "bear_2021": "2021-01-01"}


def distill_walkforward(oos_start, tag):
    require_cuda()
    cfg = PRESETS["B"]
    data = build_dataset("close_only", seq_len=60, horizons=(5, 10, 20))
    Xg = to_gpu(data)
    y_np = np.nan_to_num(np.clip(data["targets"]["tgt_rank_20"], -1, 1))
    yg = torch.as_tensor(y_np, device=Xg.device)
    dates, dr = data["dates"], data["date_rank"]
    oos0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(oos_start))))
    last = int(dr.max())
    refit_ranks = list(range(oos0, last + 1, 126))

    rows_s, rows_e = [], []
    log = {"tag": tag, "refits": [], "teacher_s": 0.0, "student_s": 0.0}
    t_all = time.time()
    for ri, r0 in enumerate(refit_ranks):
        refit_rank = r0 - 1
        block_end = min(r0 + 126 - 1, last)
        try:
            tr, va, _ = matured_train_val(data, "tgt_rank_20", refit_rank, 20,
                                          recency=None)
        except ValueError as e:
            print(f"[{tag}] skip refit@{str(dates[r0])[:10]}: {e}", flush=True)
            continue
        va_dates = dr[va]
        # --- teachers (identical protocol to the A8 ensemble runs)
        nets, t0 = [], time.time()
        for s in SEEDS:
            net, vic, _ = fit_one(Xg, yg, tr, va, va_dates, cfg, seed=s,
                                  loss="mse", date_ranks=dr)
            nets.append(net)
        log["teacher_s"] += time.time() - t0
        # --- student: 50/50 true target + teacher-mean on the TRAIN set
        tr_np = np.asarray(tr)
        with torch.no_grad():
            t_mean_tr = torch.stack(
                [predict_idx(n, Xg, tr_np) for n in nets]).mean(0).cpu().numpy()
        y_soft = y_np.copy()
        y_soft[tr_np] = 0.5 * y_np[tr_np] + 0.5 * t_mean_tr
        yg_soft = torch.as_tensor(y_soft, device=Xg.device)
        t0 = time.time()
        student, s_vic, s_info = fit_one(Xg, yg_soft, tr, va, va_dates, cfg,
                                         seed=0, loss="mse", date_ranks=dr)
        stu_s = time.time() - t0
        log["student_s"] += stu_s
        # --- score the OOS block with student AND ensemble
        in_block = np.nonzero((dr >= r0) & (dr <= block_end))[0]
        if len(in_block):
            p_stu = predict_idx(student, Xg, in_block).cpu().numpy()
            p_ens = torch.stack(
                [predict_idx(n, Xg, in_block) for n in nets]).mean(0).cpu().numpy()
            stocks = np.asarray(data["stocks"])[data["stock_idx"][in_block]]
            base = {"date": dates[dr[in_block]], "stock": stocks,
                    "target": data["targets"]["tgt_rank_20"][in_block],
                    "fwd_h": data["targets"]["fwd_20"][in_block],
                    "fwd_20": data["targets"]["fwd_20"][in_block]}
            rows_s.append(pd.DataFrame({**base, "score": p_stu}))
            rows_e.append(pd.DataFrame({**base, "score": p_ens}))
        del nets, student
        torch.cuda.empty_cache()
        log["refits"].append({"refit_date": str(dates[r0])[:10],
                              "student_val_ic": round(s_vic, 5),
                              "student_train_s": round(stu_s, 1)})
        print(f"[{tag}] [wf {ri+1}/{len(refit_ranks)}] {str(dates[r0])[:10]} "
              f"student val_ic {s_vic:+.4f} ({time.time()-t_all:.0f}s elapsed)",
              flush=True)
    for rows, kind in ((rows_s, "student"), (rows_e, "ensemble")):
        panel = pd.concat(rows, ignore_index=True)
        panel.to_csv(os.path.join(PANEL_DIR, f"D2S_{tag}_{kind}.csv.gz"),
                     index=False, compression="gzip")
    log["total_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(OUT_DIR, f"gpu_log_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[{tag}] done in {log['total_s']}s "
          f"(teachers {log['teacher_s']:.0f}s, student {log['student_s']:.0f}s)",
          flush=True)


def compare():
    import transformer_portfolio as tp
    from transformer_hybrid import load_panel, merged
    from queue_v9_lib import caps, holdings_walk

    out = {"windows": {}, "runtime": {}}
    for tag in WINDOWS:
        glog = json.load(open(os.path.join(OUT_DIR, f"gpu_log_{tag}.json")))
        out["runtime"][tag] = {
            "teacher_7seed_s": round(glog["teacher_s"], 0),
            "student_1seed_s": round(glog["student_s"], 0),
            "daily_retrain_ratio": round(glog["teacher_s"] / max(glog["student_s"], 1), 1)}
        w = {}
        books = {}
        for kind in ("student", "ensemble"):
            panel, _ = load_panel(f"D2S_{tag}_{kind}")
            m = merged(panel)
            m["score_blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
            r_std = tp.backtest_scores(m.assign(score=m["z_tf"]), holding=20,
                                       mode="long_short", no_trade_band=0.10)
            r_bl = tp.backtest_scores(m.assign(score=m["score_blend"]), holding=20,
                                      mode="long_short", no_trade_band=0.10)
            with caps(0.075):
                r_d7b = tp.backtest_scores(m.assign(score=m["score_blend"]),
                                           holding=20, mode="long_short",
                                           no_trade_band=0.15)
            books[kind] = m

            def pack(r):
                return {"net60": r["net60"]["sharpe"], "net100": r["net100"]["sharpe"],
                        "max_dd": r["net60"]["max_dd"],
                        "yr2022": r["yearly_net60"].get(2022, {}).get("sharpe"),
                        "yearly": r["yearly_net60"],
                        "turnover": round(r["avg_turnover"], 3),
                        "rank_ic": round(r["rank_ic"], 4)}
            w[kind] = {"standalone": pack(r_std), "blend": pack(r_bl),
                       "d7b": pack(r_d7b)}
        # book overlap + weight diff (blend construction, band10)
        hs = {k: dict((str(d)[:10], wts) for d, wts, _ in holdings_walk(
            books[k].assign(score=books[k]["score_blend"]), mode="long_short",
            band=0.10)) for k in books}
        common = sorted(set(hs["student"]) & set(hs["ensemble"]))
        ovl, wdiff = [], []
        for d in common:
            a, b = hs["student"][d], hs["ensemble"][d]
            la, lb = {s for s, x in a.items() if x > 0}, {s for s, x in b.items() if x > 0}
            ovl.append(len(la & lb) / max(len(la | lb), 1))
            alln = set(a) | set(b)
            wdiff.append(sum(abs(a.get(s, 0) - b.get(s, 0)) for s in alln) / 2)
        w["book_overlap_jaccard_mean"] = round(float(np.mean(ovl)), 3)
        w["book_overlap_jaccard_min"] = round(float(np.min(ovl)), 3)
        w["weight_diff_L1_mean"] = round(float(np.mean(wdiff)), 3)
        out["windows"][tag] = w

    # ---- adoption gates (pre-registered from the user's approval)
    gates = {}
    for tag in WINDOWS:
        s, e = out["windows"][tag]["student"]["blend"], out["windows"][tag]["ensemble"]["blend"]
        gates[f"{tag}_sharpe_95pct"] = {
            "student": s["net60"], "ensemble": e["net60"],
            "ratio": round(s["net60"] / max(e["net60"], 1e-9), 3),
            "pass": s["net60"] >= 0.95 * e["net60"]}
        gates[f"{tag}_dd_within_1pp"] = {
            "student": s["max_dd"], "ensemble": e["max_dd"],
            "pass": s["max_dd"] >= e["max_dd"] - 0.01}
    s22 = out["windows"]["bear_2021"]["student"]["blend"]["yr2022"]
    e22 = out["windows"]["bear_2021"]["ensemble"]["blend"]["yr2022"]
    gates["yr2022_not_materially_worse"] = {
        "student": s22, "ensemble": e22,
        "pass": (s22 is not None and e22 is not None and s22 >= e22 - 0.15)}
    gates["book_overlap_trust"] = {
        "mean_jaccard": {t: out["windows"][t]["book_overlap_jaccard_mean"]
                         for t in WINDOWS},
        "pass": all(out["windows"][t]["book_overlap_jaccard_mean"] >= 0.80
                    for t in WINDOWS)}
    out["gates"] = gates
    n_pass = sum(1 for g in gates.values() if g["pass"])
    out["verdict"] = (
        f"PASS ({n_pass}/{len(gates)} gates) — deployment-efficiency candidate "
        "ONLY; adoption requires user approval" if n_pass == len(gates) else
        f"REJECT ({n_pass}/{len(gates)} gates) — 7-seed production spec stays; "
        "distillation recorded as "
        + ("near-miss research-only" if n_pass >= len(gates) - 1 else "rejected"))
    with open(os.path.join(OUT_DIR, "D2_OOS_RESULT.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out["gates"], indent=1, default=str))
    print(out["verdict"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    if mode == "gpu":
        for tag, oos in WINDOWS.items():
            distill_walkforward(oos, tag)
    else:
        compare()
