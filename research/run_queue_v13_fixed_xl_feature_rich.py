"""Queue v13 runner — fixed-XL / feature-rich research line.

Research-only. Plan docs: reports/continuous_research/v13_fixed_xl_feature_rich/.
Phases: 0 inventory/plans (done) · 1 feature quality battery (done, see
v13_feature_quality.py) · 2 small-model feature screen · 3 fixed-XL feature
screen · 4 confirmation. Phases 3/4 are materialized held and unlock only on
user approval. Never touches production paths.

Commands:
  python research/run_queue_v13_fixed_xl_feature_rich.py inspect|feature-plan
  python research/run_queue_v13_fixed_xl_feature_rich.py materialize [--force]
  python research/run_queue_v13_fixed_xl_feature_rich.py dry-run
  python research/run_queue_v13_fixed_xl_feature_rich.py smoke-features
  python research/run_queue_v13_fixed_xl_feature_rich.py run --phase 2
  python research/run_queue_v13_fixed_xl_feature_rich.py unlock --phase 3|4
  python research/run_queue_v13_fixed_xl_feature_rich.py report|status
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V13_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v13_fixed_xl_feature_rich")
LOG_DIR = os.path.join(V13_DIR, "logs")
QUEUE = os.path.join(V13_DIR, "queue_v13.json")
PANEL_DIR = os.path.join(ROOT, "reports", "transformer_gpu",
                         "v13_fixed_xl_feature_rich", "panels")
V12_QUEUE = os.path.join(ROOT, "reports", "continuous_research",
                         "v12_big_transformer", "queue_v12.json")
V12_PANEL = os.path.join(ROOT, "reports", "transformer_gpu",
                         "v12_big_transformer", "panels",
                         "V12_P2_recipe_Wall_baseline.csv.gz")

SCREEN_SETS = ["v13_f1", "v13_f2", "v13_f3", "v13_f4", "v13_f5", "v13_f6"]
P2_SEEDS = [0, 1, 2]
XL_SETS = ["close_only", "BEST_FROM_P2", "v13_f6"]   # Phase-3 slots


def _is_oom(e):
    return "out of memory" in str(e).lower()


def _load():
    with open(QUEUE, encoding="utf-8") as f:
        return json.load(f)


def _save(q):
    with open(QUEUE, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2, default=str)


def _f0_reference():
    """Import v12's Wall_baseline screen (identical protocol/cache/code —
    anchor-verified) as the F0 close_only control."""
    with open(V12_QUEUE, encoding="utf-8") as f:
        v12 = json.load(f)
    w = next(c for c in v12 if c["id"] == "P2_recipe_Wall_baseline")
    return {"id": "P2_screen_close_only", "phase": "P2",
            "feature_set": "close_only", "status": "done",
            "result": w["result"], "runtime_s": w.get("runtime_s"),
            "verdict": w.get("verdict", "") + "  [REUSED from v12 "
            "Wall_baseline: same protocol, same cache 2026-07-30, "
            "anchor-verified identical code path]"}


def materialize(force=False):
    os.makedirs(V13_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(QUEUE) and not force:
        print(f"queue exists: {QUEUE}")
        return
    q = [_f0_reference()]
    for fs in SCREEN_SETS:
        q.append({"id": f"P2_screen_{fs}", "phase": "P2", "feature_set": fs,
                  "status": "pending", "seeds": P2_SEEDS})
    for slot in XL_SETS:
        q.append({"id": f"P3_xl_{slot}", "phase": "P3", "feature_set": slot,
                  "status": "held",
                  "hold_reason": "Phase 3 requires user approval"})
    _save(q)
    print(f"materialized {len(q)} items -> {QUEUE} "
          "(F0 reference imported from v12)")


def _blend_ls(item):
    return (item.get("result", {}).get("blend_band10", {})
            .get("long_short", {}).get("net60", {}))


def run_phase2():
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data, _eval_books, _eval_blend
    T.require_cuda()
    os.makedirs(PANEL_DIR, exist_ok=True)
    q = _load()
    for item in [c for c in q if c["phase"] == "P2"
                 and c.get("status") == "pending"]:
        fs = item["feature_set"]
        item["status"] = "running"
        _save(q)
        print(f"\n[P2] feature screen {fs}", flush=True)
        t0 = time.time()
        try:
            data, Xg = get_data(fs, 60)
            panel, info = T.walkforward(
                data, Xg, target="tgt_rank_20", horizon=20, preset="B",
                oos_start="2023-01-01", refit_every=126, seeds=P2_SEEDS,
                recency=None, log_prefix=f"[P2:{fs}] ")
            panel.to_csv(os.path.join(PANEL_DIR, f"V13_P2_{fs}.csv.gz"),
                         index=False, compression="gzip")
            res = {"mean_val_ic": info["mean_val_ic"],
                   "total_train_s": info["total_s"],
                   "peak_vram_mb": info["peak_vram_mb"],
                   "books": {m: {k: v[k] for k in
                                 ("rank_ic", "avg_turnover", "net60",
                                  "net100", "yearly_net60")}
                             for m, v in _eval_books(panel, 20).items()},
                   "blend_band10": {m: {k: v[k] for k in
                                        ("rank_ic", "avg_turnover", "net60",
                                         "net100", "yearly_net60")}
                                    for m, v in _eval_blend(panel, 20).items()}}
            item["result"] = res
            bl = _blend_ls(item)
            item["verdict"] = (f"blend LS net60 {bl.get('sharpe')} dd "
                               f"{bl.get('max_dd')} val_ic "
                               f"{info['mean_val_ic']}")
            item["runtime_s"] = round(time.time() - t0, 1)
            item["status"] = "done"
            print(f"[P2] {fs}: {item['verdict']}", flush=True)
        except Exception as e:  # noqa: BLE001
            item.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            if "CUDA" in str(e) and not _is_oom(e):
                _save(q)
                print("[P2] CUDA error — stopping (hardware-note rule)")
                return
        _save(q)
    report()
    print("\n[P2] feature screen complete", flush=True)


def report():
    from run_queue_v11 import _quintile_overlap
    q = _load()
    base_panel = (pd.read_csv(V12_PANEL, dtype={"stock": str},
                              parse_dates=["date"])
                  if os.path.exists(V12_PANEL) else None)
    rows = []
    for c in q:
        bl = _blend_ls(c)
        tf = (c.get("result", {}).get("books", {})
              .get("long_short", {}).get("net60", {}))
        row = {"id": c["id"], "phase": c["phase"],
               "feature_set": c.get("feature_set"),
               "status": c.get("status"),
               "n_features": None,
               "val_ic": c.get("result", {}).get("mean_val_ic"),
               "blend_LS_net60": bl.get("sharpe"),
               "blend_LS_dd": bl.get("max_dd"),
               "blend_turnover": (c.get("result", {}).get("blend_band10", {})
                                  .get("long_short", {}).get("avg_turnover")),
               "tf_LS_net60": tf.get("sharpe"),
               "train_s": c.get("result", {}).get("total_train_s"),
               "error": c.get("error", "")}
        try:
            from dataset_transformer_eod import FEATURE_COLS
            row["n_features"] = len(FEATURE_COLS.get(c.get("feature_set"), []))
        except Exception:
            pass
        pp = os.path.join(PANEL_DIR, f"V13_P2_{c.get('feature_set')}.csv.gz")
        if (base_panel is not None and c.get("status") == "done"
                and os.path.exists(pp)):
            p = pd.read_csv(pp, dtype={"stock": str}, parse_dates=["date"])
            row["q5_overlap_vs_close_only"] = _quintile_overlap(base_panel, p)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V13_DIR, "small_model_feature_screen.csv"),
              index=False)
    df.to_csv(os.path.join(V13_DIR, "queue_v13_results.csv"), index=False)
    md = ["# v13 small-model feature screen (Phase 2)", "",
          "Bar = close_only (F0) blend LS net60 **1.969** (v12 Wall_baseline "
          "reuse; same protocol/cache/code). Reject rules (Task 6): material "
          "Sharpe drop, worse DD, turnover explosion, overlap collapse "
          "without return gain, val-IC-only wins. Prior: richer sets were "
          "REJECTED at this model size in v1-v7 — this screen is also a "
          "replication control; the XL hypothesis is tested in Phase 3.", "",
          df.to_string(index=False), ""]
    with open(os.path.join(V13_DIR, "small_model_feature_screen.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(V13_DIR, "queue_v13_results.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(df.to_string(index=False))


# ------------------------------------------------- Phase 2B: f2 confirmation

FULL_SEEDS = [0, 1, 2, 3, 4, 5, 6]
BATTERY_SEEDS = [10, 11, 12, 13, 14, 15, 16]
WINDOWS = {"CH": "2023-01-01", "BR": "2021-01-01"}
REFS_7S = {"CH": 2.147, "BR": 1.443}   # close_only 7-seed standing refs


def materialize_p2b(fs="v13_f2"):
    q = _load()
    added = 0
    for tag, seeds in (("7s", FULL_SEEDS), ("bat", BATTERY_SEEDS)):
        for win in ("CH", "BR"):
            iid = f"P2B_{fs}_{win}_{tag}"
            if not any(c["id"] == iid for c in q):
                q.append({"id": iid, "phase": "P2B", "feature_set": fs,
                          "window": win, "seeds": seeds, "status": "pending"})
                added += 1
    _save(q)
    print(f"[P2B] added {added} confirmation item(s) for {fs}")


def run_phase2b():
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data, _eval_books, _eval_blend
    T.require_cuda()
    os.makedirs(PANEL_DIR, exist_ok=True)
    q = _load()
    for item in [c for c in q if c["phase"] == "P2B"
                 and c.get("status") == "pending"]:
        fs, win = item["feature_set"], item["window"]
        item["status"] = "running"
        _save(q)
        print(f"\n[P2B] {item['id']} ({len(item['seeds'])} seeds, {win})",
              flush=True)
        t0 = time.time()
        try:
            data, Xg = get_data(fs, 60)
            panel, info = T.walkforward(
                data, Xg, target="tgt_rank_20", horizon=20, preset="B",
                oos_start=WINDOWS[win], refit_every=126,
                seeds=item["seeds"], recency=None,
                log_prefix=f"[{item['id']}] ")
            panel.to_csv(os.path.join(PANEL_DIR, f"V13_{item['id']}.csv.gz"),
                         index=False, compression="gzip")
            res = {"mean_val_ic": info["mean_val_ic"],
                   "total_train_s": info["total_s"],
                   "peak_vram_mb": info["peak_vram_mb"],
                   "books": {m: {k: v[k] for k in
                                 ("rank_ic", "avg_turnover", "net60",
                                  "net100", "yearly_net60")}
                             for m, v in _eval_books(panel, 20).items()},
                   "blend_band10": {m: {k: v[k] for k in
                                        ("rank_ic", "avg_turnover", "net60",
                                         "net100", "yearly_net60")}
                                    for m, v in _eval_blend(panel, 20).items()}}
            item["result"] = res
            bl = _blend_ls(item)
            yr = (res["blend_band10"]["long_short"].get("yearly_net60")) or {}
            y22 = ((yr.get("2022") or yr.get(2022)) or {}).get("sharpe")
            item["verdict"] = (f"blend LS net60 {bl.get('sharpe')} "
                               f"(ref {REFS_7S[win]}) dd {bl.get('max_dd')} "
                               f"2022 {y22} val_ic {info['mean_val_ic']}")
            item["runtime_s"] = round(time.time() - t0, 1)
            item["status"] = "done"
            print(f"[P2B] {item['id']}: {item['verdict']}", flush=True)
        except Exception as e:  # noqa: BLE001
            item.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            if "CUDA" in str(e) and not _is_oom(e):
                _save(q)
                print("[P2B] CUDA error — stopping (hardware-note rule)")
                return
        _save(q)
    print("\n[P2B] confirmation batch complete", flush=True)


# ------------------------------------------------- Phase 3: fixed-XL screen

XL_CFG = dict(hidden=1024, lstm_layers=1, trans_layers=24, heads=16,
              ff=4096, dropout=0.2, seq_len=60, max_epochs=25, patience=3)
XL_RUN = dict(micro_batch=64, accum_steps=16, grad_ckpt=True, epochs=8,
              max_seconds=4.2 * 3600)
XL_REFIT = "2023-01-01"
CKPT13 = os.path.join(ROOT, "checkpoints", "v13_fixed_xl_feature_rich")


def run_xl_child(fs):
    """CHILD: XL2 + matched small S_ref, single refit, fixed 8 epochs, on
    feature set `fs`. Writes xl_<fs>.json. Collapse diagnostics included."""
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data, _eval_books, _eval_blend
    from run_queue_v12_big_transformer import fit_big, _oos_panel_from_net
    out_p = os.path.join(V13_DIR, f"xl_{fs}.json")
    t0 = time.time()
    try:
        T.require_cuda()
        os.makedirs(CKPT13, exist_ok=True)
        os.makedirs(PANEL_DIR, exist_ok=True)
        data, Xg = get_data(fs, 60)
        yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]),
                             device=Xg.device)
        r0 = int(np.searchsorted(np.asarray(data["dates"]),
                                 np.datetime64(pd.Timestamp(XL_REFIT))))
        tr, va, _ = T.matured_train_val(data, "tgt_rank_20", r0, 20)
        va_dr = np.asarray(data["date_rank"])[va]
        oos = np.nonzero(np.asarray(data["date_rank"]) > r0)[0]
        oos = oos[np.isfinite(data["targets"]["fwd_20"][oos])]
        res = {"feature_set": fs, "n_features": int(Xg.shape[2])}
        for tag, cfg, kw in (
                ("S_ref", dict(T.PRESETS["B"]),
                 dict(micro_batch=1024, accum_steps=1, grad_ckpt=False,
                      epochs=XL_RUN["epochs"])),
                ("XL", XL_CFG,
                 {k: XL_RUN[k] for k in ("micro_batch", "accum_steps",
                                         "grad_ckpt", "epochs",
                                         "max_seconds")})):
            print(f"[XL:{fs}] training {tag}", flush=True)
            net, vic, curves = fit_big(Xg, yg, tr, va, va_dr, cfg, seed=0,
                                       **kw)
            panel = _oos_panel_from_net(net, data, Xg, oos,
                                        batch=min(512, kw["micro_batch"] * 4))
            panel.to_csv(os.path.join(PANEL_DIR,
                                      f"V13_XL_{fs}_{tag}.csv.gz"),
                         index=False, compression="gzip")
            if tag == "XL":
                ck = os.path.join(CKPT13, f"xl_{fs}_seed0.pt")
                torch.save({"model_state_dict": net.state_dict(), "cfg": cfg,
                            "feature_set": fs, "target": "tgt_rank_20",
                            "horizon": 20, "seed": 0, "val_ic": vic,
                            "refit": XL_REFIT}, ck)
                res["ckpt_mb"] = round(os.path.getsize(ck) / 2**20, 1)
            del net
            torch.cuda.empty_cache()
            # collapse diagnostics
            date_std = panel.groupby("date")["score"].std()
            losses = [c["train_loss"] for c in curves]
            vics = [c["val_ic"] for c in curves]
            collapsed = (float(np.nanmean(date_std)) < 1e-3
                         or all(not np.isfinite(v) for v in vics))
            res[tag] = {
                "val_ic_best": None if not np.isfinite(vic) else round(vic, 5),
                "curves": curves,
                "train_loss_last": losses[-1],
                "loss_plateau_at_target_var": bool(
                    abs(losses[-1] - 0.3332) < 0.002),
                "mean_per_date_score_std": round(float(np.nanmean(date_std)), 5),
                "n_nan_val_epochs": int(sum(1 for v in vics
                                            if not np.isfinite(v))),
                "COLLAPSED": collapsed,
                "books": {m: {k: v[k] for k in ("rank_ic", "avg_turnover",
                                                "net60", "net100")}
                          for m, v in _eval_books(panel, 20).items()},
                "blend_band10": {m: {k: v[k] for k in
                                     ("rank_ic", "avg_turnover", "net60",
                                      "net100")}
                                 for m, v in _eval_blend(panel, 20).items()}}
            bl = res[tag]["blend_band10"]["long_short"]["net60"]
            print(f"[XL:{fs}] {tag}: collapsed={collapsed} blend "
                  f"{bl['sharpe']} dd {bl['max_dd']} score_std "
                  f"{res[tag]['mean_per_date_score_std']}", flush=True)
        res["total_s"] = round(time.time() - t0, 1)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        res = {"feature_set": fs, "ok": False,
               "error": f"{type(e).__name__}: {e}", "is_oom": _is_oom(e)}
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    return 0 if res.get("ok") else 1


def run_phase3():
    """PARENT: one subprocess per XL feature run (context isolation)."""
    q = _load()
    cl = next((c for c in q if c["id"] == "P3_xl_close_only"), None)
    if cl is not None and cl.get("status") != "done":
        cl.update(status="done", verdict=(
            "REUSED from v12: XL2 on close_only COLLAPSED to mean (loss "
            "pinned at 0.3332, val IC NaN); S_ref single-refit blend 1.418. "
            "Same protocol/cache; serves as the collapse control."))
    for c in q:
        if c["id"] == "P3_xl_BEST_FROM_P2" and c.get("feature_set") == "BEST_FROM_P2":
            c["feature_set"] = "v13_f2"
            c["note"] = "best-of-P2 slot resolved to v13_f2 (screen 2.101)"
    _save(q)
    for item in [c for c in q if c["phase"] == "P3"
                 and c.get("status") == "pending"]:
        fs = item["feature_set"]
        item["status"] = "running"
        _save(q)
        print(f"\n[P3] spawning XL screen on {fs} (fresh process)", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "run-xl", "--feature-set", fs], cwd=ROOT)
        rp = os.path.join(V13_DIR, f"xl_{fs}.json")
        r = (json.load(open(rp, encoding="utf-8")) if os.path.exists(rp)
             else {"ok": False, "error": "child wrote no result"})
        if r.get("ok"):
            xl = r["XL"]
            item.update(status="done", result={k: v for k, v in r.items()
                                               if k not in ("S_ref", "XL")},
                        xl_summary={t: {k: v for k, v in r[t].items()
                                        if k != "curves"}
                                    for t in ("S_ref", "XL")},
                        verdict=(f"XL collapsed={xl['COLLAPSED']}, blend "
                                 f"{xl['blend_band10']['long_short']['net60']['sharpe']}"
                                 f" vs S_ref "
                                 f"{r['S_ref']['blend_band10']['long_short']['net60']['sharpe']}"
                                 f", ckpt {r.get('ckpt_mb')}MB"))
        else:
            item.update(status="failed", error=r.get("error"))
            if "CUDA" in str(r.get("error", "")) and not r.get("is_oom"):
                _save(q)
                print("[P3] non-OOM CUDA error — stopping", flush=True)
                return
        _save(q)
    _write_xl_reports(q)
    print("\n[P3] XL feature screen complete", flush=True)


def _write_xl_reports(q):
    rows, curve_rows = [], []
    for c in q:
        if c["phase"] != "P3" or c.get("status") != "done":
            continue
        rp = os.path.join(V13_DIR, f"xl_{c.get('feature_set')}.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp, encoding="utf-8"))
        for tag in ("S_ref", "XL"):
            t = r.get(tag, {})
            bl = (t.get("blend_band10", {}).get("long_short", {})
                  .get("net60", {}))
            rows.append({"feature_set": r["feature_set"], "model": tag,
                         "n_features": r.get("n_features"),
                         "collapsed": t.get("COLLAPSED"),
                         "blend_LS_net60": bl.get("sharpe"),
                         "blend_LS_dd": bl.get("max_dd"),
                         "val_ic_best": t.get("val_ic_best"),
                         "score_std": t.get("mean_per_date_score_std"),
                         "train_loss_last": t.get("train_loss_last"),
                         "ckpt_mb": r.get("ckpt_mb") if tag == "XL" else None})
            for e in t.get("curves", []):
                curve_rows.append({"feature_set": r["feature_set"],
                                   "model": tag, **e})
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(V13_DIR, "xl_feature_screen.csv"), index=False)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(
            os.path.join(V13_DIR, "xl_learning_curves.csv"), index=False)
    md = ["# v13 fixed-XL feature screen (Phase 3)", "",
          "Single refit 2023-01-01, fixed 8 epochs, 1 seed — EXPLORATORY "
          "(not decision-grade). Control: v12 XL2-on-close_only collapsed "
          "to the mean under this exact protocol (S_ref 1.418). Primary "
          "question: does richer input prevent XL collapse?", ""]
    if rows:
        md += [pd.DataFrame(rows).to_string(index=False), ""]
    with open(os.path.join(V13_DIR, "xl_feature_screen.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("[P3] -> xl_feature_screen.{csv,md}, xl_learning_curves.csv")


def status():
    for c in _load():
        print(f"{c['id']:28s} {c['phase']}  {c.get('status', ''):8s} "
              f"{c.get('verdict', c.get('hold_reason', ''))[:70]}")


def dry_run():
    q = _load()
    for c in q:
        print(f"{c['id']:28s} {c['phase']}  {c.get('status')}")
    print("\n[dry-run] pending P2 screens: ~14-25 min each (13->44 features),"
          " total ~2h. Nothing executed.")


def unlock(phase):
    q = _load()
    n = 0
    for c in q:
        if c["phase"] == f"P{phase}" and c.get("status") == "held":
            c["status"] = "pending"
            c.pop("hold_reason", None)
            n += 1
    _save(q)
    print(f"[unlock] {n} P{phase} item(s) pending")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["inspect", "feature-plan",
                                       "materialize", "materialize-p2b",
                                       "dry-run", "smoke-features", "run",
                                       "run-xl", "unlock", "report",
                                       "status"])
    ap.add_argument("--phase", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--feature-set", default=None)
    a = ap.parse_args()
    if a.command == "run-xl":
        sys.exit(run_xl_child(a.feature_set))
    if a.command == "materialize-p2b":
        materialize_p2b()
        sys.exit(0)
    if a.command == "run" and a.phase == "2b":
        run_phase2b()
        sys.exit(0)
    if a.command == "run" and a.phase == "3":
        run_phase3()
        sys.exit(0)
    a.phase = int(a.phase) if a.phase is not None else None
    if a.command == "inspect":
        print(open(os.path.join(V13_DIR, "current_feature_inventory.md"),
                   encoding="utf-8").read())
    elif a.command == "feature-plan":
        print(open(os.path.join(V13_DIR, "feature_family_plan.md"),
                   encoding="utf-8").read())
    elif a.command == "materialize":
        materialize(a.force)
    elif a.command == "dry-run":
        dry_run()
    elif a.command == "smoke-features":
        subprocess.run([sys.executable,
                        os.path.join(ROOT, "research",
                                     "v13_feature_quality.py")], cwd=ROOT)
    elif a.command == "run" and a.phase == 2:
        run_phase2()
    elif a.command == "run":
        sys.exit(f"phase {a.phase} execution path lands with its approval")
    elif a.command == "unlock":
        unlock(a.phase)
    elif a.command == "report":
        report()
    else:
        status()
