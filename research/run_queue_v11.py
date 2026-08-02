"""Queue v11 runner — GPU model-size sweep + VRAM/throughput instrumentation.

Research-only. Reuses gpu_research_scheduler.run_one (derived presets via
`overrides`, OOM seed-halving, panel save, book+blend eval) with NO
promote()/spawn_bear()/daily_ops() — nothing auto-promotes, nothing touches
production defaults, checkpoints/transformer_eod, data_cache, or the paper
ledger. Plan: reports/continuous_research/queue_v11_gpu_model_size_sweep_plan.md

Commands (all from repo root, venv python):
  python research/run_queue_v11.py materialize          # write queue + manifest
  python research/run_queue_v11.py smoke                # tiny end-to-end check
  python research/run_queue_v11.py probe [--target-vram-gb 10]   # batch ladder
  python research/run_queue_v11.py run [--phase P0|P1|P2] [--max-candidates N]
                                       [--dry-run]
  python research/run_queue_v11.py select               # P1 -> P2 promotion
  python research/run_queue_v11.py report               # results/verdict files
  python research/run_queue_v11.py status
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V11_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v11_gpu_model_size_sweep")
LOG_DIR = os.path.join(V11_DIR, "logs")
QUEUE = os.path.join(V11_DIR, "queue_v11.json")
GPU_CSV = os.path.join(V11_DIR, "queue_v11_gpu_usage.csv")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "v11_sweep")  # gitignored
PANEL_DIR = os.path.join(ROOT, "reports", "transformer_gpu", "panels")

# ---- pre-registered candidates (see plan §2) ------------------------------
CANDS = {
    "V11_A_baseline":        dict(seq_len=60, overrides=None),
    "V11_B_wider96":         dict(seq_len=60, overrides={"hidden": 96, "ff": 192}),
    "V11_C_wider128":        dict(seq_len=60, overrides={"hidden": 128, "ff": 256,
                                                         "heads": 8}),
    "V11_D_deeper64":        dict(seq_len=60, overrides={"trans_layers": 3}),
    "V11_E_longer90":        dict(seq_len=90, overrides=None),
    "V11_F_wider96_longer90": dict(seq_len=90, overrides={"hidden": 96, "ff": 192}),
    "V11_G_wider128_deeper3": dict(seq_len=60, overrides={"hidden": 128, "ff": 256,
                                                          "heads": 8,
                                                          "trans_layers": 3}),
}
SCREEN_SEEDS = [0, 1, 2]
FULL_SEEDS = [0, 1, 2, 3, 4, 5, 6]
BATTERY_SEEDS = [10, 11, 12, 13, 14, 15, 16]
WINDOWS = {"CH": "2023-01-01", "BR": "2021-01-01"}
# baseline env-drift check bounds (plan §3, Phase 0)
P0_RANGES = {"CH": (1.85, 2.30), "BR": (1.30, 1.55)}
# rough wall-clock multipliers vs baseline (refined by probe/smoke timing)
EST_MULT = {"V11_A_baseline": 1.0, "V11_B_wider96": 1.4, "V11_C_wider128": 1.8,
            "V11_D_deeper64": 1.3, "V11_E_longer90": 1.6,
            "V11_F_wider96_longer90": 2.2, "V11_G_wider128_deeper3": 2.4}
BASE_S = {"CH": 1950, "BR": 4800}  # 7-seed h64 calibration from queue v8


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        sys.stdout = self.stdout
        self.f.close()


def _load():
    with open(QUEUE, encoding="utf-8") as f:
        return json.load(f)


def _save(queue):
    with open(QUEUE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, default=str)


def _cfg(item_id, cand_id, seeds, window, phase):
    c = CANDS[cand_id]
    return {"id": item_id, "candidate": cand_id, "phase": phase,
            "window": window, "feature_set": "close_only",
            "target": "tgt_rank_20", "seq_len": c["seq_len"],
            "preset_base": "B", "overrides": c["overrides"],
            "seeds": list(seeds), "oos_start": WINDOWS[window],
            "eval_blend": True, "batch": 1024, "status": "pending"}


def _gpu_row(item_id, batch, info, extra=None):
    import torch
    row = {"ts": time.strftime("%F %T"), "id": item_id,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
           if torch.cuda.is_available() else 0,
           "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
           if torch.cuda.is_available() else 0,
           "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)
           if torch.cuda.is_available() else 0,
           "batch": batch, "amp": True,
           "n_seeds": len(info.get("seeds", [])) if info else "",
           "total_train_s": info.get("total_s") if info else "",
           "mean_seed_train_s": (round(np.mean([f.get("train_s", np.nan)
                                                for f in info.get("fits", [])]), 1)
                                 if info and info.get("fits") else "")}
    row.update(extra or {})
    hdr = not os.path.exists(GPU_CSV)
    pd.DataFrame([row]).to_csv(GPU_CSV, mode="a", header=hdr, index=False)
    return row


# ------------------------------------------------------------- materialize

def materialize(force=False):
    os.makedirs(V11_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(QUEUE) and not force:
        print(f"queue exists: {QUEUE} (use --force to rebuild)")
        return
    queue = [
        _cfg("P0_A_CH_7s", "V11_A_baseline", FULL_SEEDS, "CH", "P0"),
        _cfg("P0_A_BR_7s", "V11_A_baseline", FULL_SEEDS, "BR", "P0"),
    ]
    for cid in ("V11_A_baseline", "V11_B_wider96", "V11_C_wider128",
                "V11_D_deeper64", "V11_E_longer90", "V11_F_wider96_longer90"):
        queue.append(_cfg(f"P1_{cid}_CH_3s", cid, SCREEN_SEEDS, "CH", "P1"))
    g = _cfg("P1_V11_G_wider128_deeper3_CH_3s", "V11_G_wider128_deeper3",
             SCREEN_SEEDS, "CH", "P1")
    g["status"] = "held"          # unlocked by `select` iff B..F trained stably
    g["hold_reason"] = "optional candidate; runs only if B-F stable (plan §2)"
    queue.append(g)
    _save(queue)

    import subprocess
    manifest = {
        "queue": "v11_gpu_model_size_sweep", "created": time.strftime("%F %T"),
        "branch": subprocess.run(["git", "branch", "--show-current"],
                                 capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "protocol": {"feature_set": "close_only", "target": "tgt_rank_20",
                     "horizon": 20, "refit_every": 126, "loss": "mse",
                     "lr": 3e-4, "weight_decay": 1e-4, "batch": 1024,
                     "windows": WINDOWS, "screen_seeds": SCREEN_SEEDS,
                     "full_seeds": FULL_SEEDS, "battery_seeds": BATTERY_SEEDS},
        "phase0_ranges": P0_RANGES,
        "candidates": CANDS,
        "gates": "see queue_v11_gpu_model_size_sweep_plan.md §6 (pre-registered)",
        "baseline_refs": {"blend_CH": 2.147, "blend_BR": 1.443,
                          "ranges_CH": [1.85, 2.15], "ranges_BR": [1.30, 1.45]},
    }
    with open(os.path.join(V11_DIR, "queue_v11_run_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"materialized {len(queue)} items -> {QUEUE}")


# -------------------------------------------------------------------- run

def run(phase=None, max_candidates=None, dry_run=False):
    queue = _load()
    todo = [c for c in queue if c.get("status") == "pending"
            and (phase is None or c["phase"] == phase)]
    if max_candidates:
        todo = todo[:max_candidates]
    if dry_run:
        tot = 0
        print(f"[dry-run] {len(todo)} pending item(s)"
              + (f" in phase {phase}" if phase else "") + ":")
        for c in todo:
            est = BASE_S[c["window"]] * EST_MULT[c["candidate"]] * len(c["seeds"]) / 7
            tot += est
            print(f"  {c['id']:34s} seeds={len(c['seeds'])} window={c['window']}"
                  f" est~{est/60:.0f}min")
        print(f"[dry-run] total estimate ~{tot/3600:.1f}h; outputs -> {V11_DIR}")
        print("[dry-run] nothing executed.")
        return
    from gpu_research_scheduler import run_one
    import torch
    for cfg in todo:
        cfg["status"] = "running"
        _save(queue)
        tee = _Tee(os.path.join(LOG_DIR, f"{cfg['id']}.log"))
        sys.stdout = tee
        t0 = time.time()
        print(f"\n===== {cfg['id']} ({cfg['candidate']}, {cfg['window']}, "
              f"{len(cfg['seeds'])} seeds) {time.strftime('%F %T')}", flush=True)
        try:
            # route the batch key through the derived-preset mechanism
            if cfg.get("batch") and cfg["batch"] != 1024:
                cfg["overrides"] = dict(cfg.get("overrides") or {},
                                        batch=cfg["batch"])
            run_one(cfg)
            _gpu_row(cfg["id"], cfg.get("batch", 1024),
                     {"seeds": cfg["seeds"],
                      "total_s": cfg["result"]["total_train_s"],
                      "fits": []})
            cfg["status"] = "done"
            bl = cfg["result"].get("blend_band10", {}).get("long_short", {})
            cfg["verdict"] = (f"blend LS net60 {bl.get('net60', {}).get('sharpe')}"
                              f" val_ic {cfg['result']['mean_val_ic']}")
            # Phase-0 environment-drift stop rule (plan §3)
            if cfg["phase"] == "P0":
                sh = bl.get("net60", {}).get("sharpe")
                lo, hi = P0_RANGES[cfg["window"]]
                if sh is None or not (lo <= sh <= hi):
                    cfg["verdict"] += f" — OUTSIDE {lo}-{hi}: STOP AND INVESTIGATE"
                    cfg["p0_drift"] = True
                    _save(queue)
                    print(f"[v11] P0 drift check FAILED ({sh} outside "
                          f"[{lo},{hi}]) — stopping queue", flush=True)
                    return
            cfg["runtime_s"] = round(time.time() - t0, 1)
            print(f"[{cfg['id']}] done {cfg['runtime_s']}s — {cfg['verdict']}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            cfg["status"] = "failed"
            cfg["error"] = f"{type(e).__name__}: {e}"
            print(f"[{cfg['id']}] FAILED: {cfg['error']}", flush=True)
            traceback.print_exc()
            if "CUDA" in cfg["error"] or "cuda" in cfg["error"]:
                print("[v11] CUDA error — stopping queue, write a hardware "
                      "note before retrying (standing rule)", flush=True)
                _save(queue)
                tee.close()
                return
        finally:
            if sys.stdout is tee:
                tee.close()
        _save(queue)
    done = sum(1 for c in queue if c["status"] == "done")
    fail = sum(1 for c in queue if c["status"] == "failed")
    print(f"[v11] batch complete: {done} done, {fail} failed "
          f"(of {len(queue)} items)")


# ------------------------------------------------------------------ select

def select():
    """Pre-registered P1 -> P2 promotion (plan §3): unlock G if B-F stable;
    promote top<=2 screens beating the A 3-seed screen to 7-seed dual-window."""
    queue = _load()
    by_id = {c["id"]: c for c in queue}

    bf = [c for c in queue if c["phase"] == "P1" and "V11_G" not in c["id"]
          and c["candidate"] != "V11_A_baseline"]
    g = by_id.get("P1_V11_G_wider128_deeper3_CH_3s")
    if g and g["status"] == "held":
        stable = all(c["status"] == "done" and not c.get("oom_retried")
                     for c in bf)
        if stable:
            g["status"] = "pending"
            print("[select] B-F stable -> V11_G unlocked (pending)")
        else:
            print("[select] B-F not all stable/done -> V11_G stays held")

    p1 = [c for c in queue if c["phase"] == "P1" and c["status"] == "done"]
    a = next((c for c in p1 if c["candidate"] == "V11_A_baseline"), None)
    if a is None:
        _save(queue)
        print("[select] A 3-seed screen not done yet — no P2 promotion")
        return

    def _sh(c):
        return (c["result"]["blend_band10"]["long_short"]["net60"]["sharpe"])
    base = _sh(a)
    beat = sorted([c for c in p1 if c is not a and _sh(c) >= base],
                  key=_sh, reverse=True)[:2]
    print(f"[select] A screen blend LS net60 = {base}; "
          f"{len(beat)} candidate(s) beat/match it")
    added = 0
    for c in beat:
        for w in ("CH", "BR"):
            nid = f"P2_{c['candidate']}_{w}_7s"
            if nid not in by_id:
                queue.append(_cfg(nid, c["candidate"], FULL_SEEDS, w, "P2"))
                added += 1
    _save(queue)
    print(f"[select] added {added} P2 item(s)" if added else
          "[select] no P2 items added"
          + ("" if beat else " — screens do not justify confirmation runs"))


# ------------------------------------------------------------------- probe

def probe(target_gb=10.0, cand_id="V11_A_baseline", ladder=None):
    """Batch ladder toward ~target_gb peak VRAM; OOM-safe (plan §4)."""
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data
    T.require_cuda()
    c = CANDS[cand_id]
    data, Xg = get_data("close_only", c["seq_len"])
    ladder = ladder or [1024, 2048, 4096, 8192, 12288, 16384]
    print(f"[probe] {cand_id} target ~{target_gb} GB "
          f"(allowed 9-11), ladder {ladder}")
    results, last_ok = [], None
    for b in ladder:
        name = f"__v11_probe_{cand_id}_{b}"
        T.PRESETS[name] = dict(T.PRESETS["B"], **(c["overrides"] or {}),
                               seq_len=c["seq_len"], max_epochs=1, batch=b)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            _, info = T.walkforward(data, Xg, target="tgt_rank_20", horizon=20,
                                    preset=name, oos_start="2026-04-01",
                                    refit_every=126, seeds=[0], verbose=False)
            dt = time.time() - t0
            row = _gpu_row(f"probe_{cand_id}", b, info,
                           {"probe_epoch_s": round(dt, 1), "oom": False})
            print(f"[probe] batch {b:6d}: alloc {row['max_alloc_gb']:.2f} GB, "
                  f"reserved {row['max_reserved_gb']:.2f} GB, "
                  f"1-epoch fit {dt:.0f}s, val_ic {info['mean_val_ic']}")
            results.append(row)
            last_ok = b
            if row["max_alloc_gb"] >= target_gb:
                print(f"[probe] reached target at batch {b}")
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _gpu_row(f"probe_{cand_id}", b, None, {"oom": True})
            print(f"[probe] batch {b}: OOM — backing off (last ok {last_ok})")
            break
    print(f"[probe] done; rows appended to {GPU_CSV}")
    return results


# ------------------------------------------------------------------- smoke

def smoke():
    """Tiny end-to-end: 1 seed, 2 epochs, checkpoint save/load + inference
    compat + GPU logging. No production files touched."""
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data
    T.require_cuda()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(V11_DIR, exist_ok=True)
    t0 = time.time()
    data, Xg = get_data("close_only", 60)
    yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]),
                         device=Xg.device)
    refit_rank = int(data["date_rank"].max())
    tr, va, w = T.matured_train_val(data, "tgt_rank_20", refit_rank, 20,
                                    recency=None)
    cfg = dict(T.PRESETS["B"], max_epochs=2)
    torch.cuda.reset_peak_memory_stats()
    net, vic, info = T.fit_one(Xg, yg, tr, va, data["date_rank"][va], cfg,
                               seed=0, weights=w)
    ck_p = os.path.join(CKPT_DIR, "smoke_V11_A_seed0.pt")
    torch.save({"model_state_dict": net.state_dict(), "cfg": cfg,
                "feature_set": "close_only", "target": "tgt_rank_20",
                "horizon": 20, "recency": None, "seed": 0, "val_ic": vic,
                "asof": str(data["dates"][refit_rank])[:10]}, ck_p)
    # inference-compat: rebuild exactly the way inference_transformer_eod does
    from inference_transformer_eod import build_net_from_ck
    ck = torch.load(ck_p, map_location=Xg.device, weights_only=False)
    net2 = build_net_from_ck(ck).to(Xg.device)
    net2.load_state_dict(ck["model_state_dict"])
    net2.eval()
    idx = torch.as_tensor(np.nonzero(data["date_rank"] == refit_rank)[0],
                          device=Xg.device)
    scores = T.predict_idx(net2, Xg, idx)
    scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
    finite = bool(np.isfinite(scores).all())
    out = {
        "ok": finite and info["epochs_run"] == 2,
        "val_ic": vic, "epochs_run": info["epochs_run"],
        "train_s": info["train_s"],
        "amp_enabled": bool(Xg.is_cuda),
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "checkpoint": ck_p,
        "checkpoint_mb": round(os.path.getsize(ck_p) / 2**20, 2),
        "reload_and_score_ok": finite,
        "n_scored_latest_date": int(len(scores)),
        "total_s": round(time.time() - t0, 1),
    }
    _gpu_row("smoke_V11_A", 1024, {"seeds": [0], "total_s": out["total_s"],
                                   "fits": [info]})
    with open(os.path.join(V11_DIR, "smoke_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[smoke] {'PASS' if out['ok'] else 'FAIL'}")
    return out


# ------------------------------------------------------------------ report

def _quintile_overlap(panel_a, panel_b):
    """Mean per-date Jaccard of top-quintile sets between two score panels."""
    j = []
    b_by_date = {d: g for d, g in panel_b.groupby("date")}
    for d, ga in panel_a.groupby("date"):
        gb = b_by_date.get(d)
        if gb is None or len(ga) < 10 or len(gb) < 10:
            continue
        ka, kb = max(1, len(ga) // 5), max(1, len(gb) // 5)
        sa = set(ga.nlargest(ka, "score")["stock"])
        sb = set(gb.nlargest(kb, "score")["stock"])
        j.append(len(sa & sb) / len(sa | sb))
    return round(float(np.mean(j)), 3) if j else None


def report():
    queue = _load()
    rows, book_rows = [], []
    panels = {}

    def _panel(cfg):
        p = os.path.join(PANEL_DIR, f"SCHED_{cfg['id']}.csv.gz")
        if cfg["id"] not in panels and os.path.exists(p):
            panels[cfg["id"]] = pd.read_csv(p, dtype={"stock": str},
                                            parse_dates=["date"])
        return panels.get(cfg["id"])

    base_ids = {"CH": "P0_A_CH_7s", "BR": "P0_A_BR_7s"}
    by_id = {c["id"]: c for c in queue}
    for cfg in queue:
        r = cfg.get("result", {})
        bl = r.get("blend_band10", {}).get("long_short", {})
        n60 = bl.get("net60", {})
        row = {"id": cfg["id"], "candidate": cfg["candidate"],
               "phase": cfg["phase"], "window": cfg["window"],
               "n_seeds": len(cfg["seeds"]), "status": cfg.get("status"),
               "val_ic": r.get("mean_val_ic"),
               "blend_LS_net60_sharpe": n60.get("sharpe"),
               "blend_LS_max_dd": n60.get("max_dd"),
               "blend_turnover": bl.get("avg_turnover"),
               "blend_2022": ((bl.get("yearly_net60") or {}).get("2022")
                              or (bl.get("yearly_net60") or {}).get(2022)
                              or {}).get("sharpe") if bl.get("yearly_net60") else None,
               "train_s": r.get("total_train_s"),
               "peak_vram_mb": r.get("peak_vram_mb"),
               "runtime_s": cfg.get("runtime_s"), "error": cfg.get("error", "")}
        # top-quintile overlap vs the 7-seed baseline panel of the same window
        base = by_id.get(base_ids.get(cfg["window"], ""))
        if (cfg.get("status") == "done" and base and base is not cfg
                and base.get("status") == "done"):
            pa, pb = _panel(base), _panel(cfg)
            if pa is not None and pb is not None:
                row["q5_overlap_vs_baseline"] = _quintile_overlap(pa, pb)
        rows.append(row)
        for book_key in ("books", "blend_band10"):
            for mode, m in (r.get(book_key) or {}).items():
                book_rows.append({"id": cfg["id"], "book": book_key,
                                  "mode": mode, **{k: (m.get(k, {}) or {}).get("sharpe")
                                                   if isinstance(m.get(k), dict) else m.get(k)
                                                   for k in ("rank_ic", "avg_turnover",
                                                             "net0", "net60", "net100",
                                                             "net150")},
                                  "yearly_net60": json.dumps(m.get("yearly_net60"))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(V11_DIR, "queue_v11_results.csv"), index=False)
    pd.DataFrame(book_rows).to_csv(
        os.path.join(V11_DIR, "queue_v11_book_metrics.csv"), index=False)

    done = res[res["status"] == "done"]
    all_done = (res["status"].isin(("done", "held"))).all() and len(done) > 0
    md = ["# Queue v11 results — GPU model-size sweep", "",
          f"Generated {time.strftime('%F %T')}. "
          f"{len(done)}/{len(res)} items done.", "",
          res.to_string(index=False), ""]
    with open(os.path.join(V11_DIR, "queue_v11_results.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    verdict = ["# Queue v11 verdict", ""]
    if not all_done:
        verdict += ["**IN PROGRESS** — verdict is only issued when the "
                    "pre-registered phases complete. No challenger is "
                    "promoted automatically at any point."]
    else:
        verdict += ["Fill against plan §6 gates; allowed values:",
                    "KEEP_CURRENT_PRODUCTION · "
                    "PROMOTE_CHALLENGER_FOR_FURTHER_VALIDATION · "
                    "PROMOTE_CHALLENGER_TO_PAPER_ONLY · REJECT_ALL_CHALLENGERS",
                    "", "Adoption is always a separate user decision."]
    with open(os.path.join(V11_DIR, "queue_v11_verdict.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(verdict) + "\n")
    print(res.to_string(index=False))
    print(f"\n[report] wrote results/book_metrics/verdict under {V11_DIR}")


def status():
    queue = _load()
    for c in queue:
        print(f"{c['id']:36s} {c.get('status','pending'):8s} "
              f"{str(c.get('runtime_s','')):>9s}  {c.get('verdict','')[:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["materialize", "smoke", "probe", "run",
                                        "select", "report", "status"])
    ap.add_argument("--phase", default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target-vram-gb", type=float, default=10.0)
    ap.add_argument("--probe-candidate", default="V11_A_baseline")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--amp", action="store_true",
                    help="informational: AMP is always ON for CUDA (champion "
                         "protocol); disabling it is not offered")
    ap.add_argument("--auto-batch", action="store_true",
                    help="run the VRAM probe before `run` (research batch "
                         "stays 1024 for candidate comparability)")
    a = ap.parse_args()
    if a.command == "materialize":
        materialize(force=a.force)
    elif a.command == "smoke":
        smoke()
    elif a.command == "probe":
        probe(a.target_vram_gb, a.probe_candidate)
    elif a.command == "run":
        if a.auto_batch and not a.dry_run:
            probe(a.target_vram_gb, a.probe_candidate)
        run(a.phase, a.max_candidates, a.dry_run)
    elif a.command == "select":
        select()
    elif a.command == "report":
        report()
    else:
        status()
