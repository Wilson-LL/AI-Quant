"""GPU experiment scheduler for the continuous research loop (queue v5+).

Runs a JSON queue of experiment configs sequentially on the RTX 4060 Ti
(max_concurrent_gpu_jobs = 1 by design: one CUDA training process, datasets
GPU-resident, CPU preprocessing inside build_dataset is vectorized). Each
config is crash-safe: queue state + results rewritten after every job; a crash
resumes at the first non-done config.

Features:
- dataset reuse across configs grouped by (feature_set, seq_len)
- preset overrides (layers / dropout / hidden) injected as derived presets
- OOM handling: empty_cache -> retry once with half the seeds -> mark failed
  (never silently skipped; failures stay in the queue file with status=failed)
- auto-promotion: after all 'screen' configs finish, top-K by mean VAL IC are
  cloned to 5-seed 'promoted' configs (2023 window), and any promoted config
  whose blend book beats the standing reference spawns a bear-window run
- per-job log line in gpu_scheduler/scheduler_log.jsonl (times, VRAM, params)
- daily TWSE ops run only at queue completion (never during GPU jobs)

Usage:
  python research/gpu_research_scheduler.py run   [queue.json]
  python research/gpu_research_scheduler.py status [queue.json]
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from dataset_transformer_eod import build_dataset  # noqa: E402
from train_transformer_eod import walkforward, to_gpu, require_cuda, PRESETS  # noqa: E402
from transformer_portfolio import backtest_scores, summarize  # noqa: E402

SCHED_DIR = os.path.join(ROOT, "reports", "continuous_research", "gpu_scheduler")
PANEL_DIR = os.path.join(ROOT, "reports", "transformer_gpu", "panels")
LOG_PATH = os.path.join(SCHED_DIR, "scheduler_log.jsonl")
DEFAULT_QUEUE = os.path.join(SCHED_DIR, "queue_v5.json")
os.makedirs(SCHED_DIR, exist_ok=True)

PROMOTE_TOP_K = 2
FULL_SEEDS = [0, 1, 2, 3, 4]

_data_cache = {}


def get_data(feature_set, seq_len):
    key = (feature_set, seq_len)
    if key not in _data_cache:
        _data_cache.clear()
        torch.cuda.empty_cache()
        d = build_dataset(feature_set, seq_len=seq_len, horizons=(5, 10, 20))
        _data_cache[key] = (d, to_gpu(d))
    return _data_cache[key]


def _log(rec):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _save(queue, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def _preset_for(cfg):
    base = cfg.get("preset_base", "B")
    ov = dict(cfg.get("overrides") or {})
    # the model's positional sizing must match the dataset's sequence length
    if cfg["seq_len"] != PRESETS[base]["seq_len"]:
        ov["seq_len"] = cfg["seq_len"]
    if not ov:
        return base
    name = f"__{cfg['id']}"
    PRESETS[name] = dict(PRESETS[base], **ov)
    return name


def _eval_books(panel, holding):
    out = {}
    for mode in ("long_short", "long_only"):
        r = backtest_scores(panel, holding=holding, mode=mode, no_trade_band=0.10)
        print(summarize(r, f"  [{mode[:2]} band10]"))
        out[mode] = {k: r[k] for k in
                     ("rank_ic", "ic_ir", "avg_turnover", "max_name_weight",
                      "net0", "net60", "net100", "net150", "yearly_net60")}
    return out


def _eval_blend(panel, holding):
    from transformer_hybrid import merged
    m = merged(panel)
    m["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    return _eval_books(m, holding)


def run_one(cfg):
    t0 = time.time()
    data, Xg = get_data(cfg["feature_set"], cfg["seq_len"])
    preset = _preset_for(cfg)
    seeds = cfg["seeds"]
    kw = dict(target=cfg["target"], horizon=cfg.get("horizon", 20),
              preset=preset, oos_start=cfg.get("oos_start", "2023-01-01"),
              refit_every=cfg.get("refit_every", 126), seeds=seeds,
              recency=None, loss=cfg.get("loss", "mse"),
              weight_decay=cfg.get("weight_decay", 1e-4))
    torch.cuda.reset_peak_memory_stats()
    try:
        panel, info = walkforward(data, Xg, log_prefix=f"[{cfg['id']}] ", **kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        retry_seeds = seeds[:max(1, len(seeds) // 2)]
        print(f"[{cfg['id']}] OOM -> retry with seeds {retry_seeds}")
        _log({"id": cfg["id"], "event": "oom_retry", "seeds": retry_seeds,
              "ts": time.strftime("%F %T")})
        kw["seeds"] = retry_seeds
        panel, info = walkforward(data, Xg, log_prefix=f"[{cfg['id']}][retry] ", **kw)
        cfg["oom_retried"] = True
    pp = os.path.join(PANEL_DIR, f"SCHED_{cfg['id']}.csv.gz")
    panel.to_csv(pp, index=False, compression="gzip")
    res = {"mean_val_ic": info["mean_val_ic"],
           "n_refits": info["n_refits"], "total_train_s": info["total_s"],
           "peak_vram_mb": info["peak_vram_mb"],
           "books": _eval_books(panel, cfg.get("holding", 20))}
    if cfg.get("eval_blend"):
        res["blend_band10"] = _eval_blend(panel, cfg.get("holding", 20))
    cfg["result"] = res
    cfg["panel"] = os.path.basename(pp)
    cfg["runtime_s"] = round(time.time() - t0, 1)
    _log({"id": cfg["id"], "start": time.strftime("%F %T", time.localtime(t0)),
          "elapsed_s": cfg["runtime_s"], "peak_vram_mb": info["peak_vram_mb"],
          "val_ic": info["mean_val_ic"], "seeds": len(kw["seeds"]),
          "gpu": torch.cuda.get_device_name(0)})
    return cfg


def promote(queue):
    """Clone top-K finished screens (by VAL IC — selection metric, no OOS
    peeking) to 5-seed promoted configs; skip if already present."""
    screens = [c for c in queue if c.get("phase") == "screen" and c.get("status") == "done"]
    pending_screens = [c for c in queue if c.get("phase") == "screen"
                       and c.get("status") not in ("done", "failed")]
    if pending_screens or not screens:
        return
    ranked = sorted(screens, key=lambda c: c["result"]["mean_val_ic"], reverse=True)
    for c in ranked[:PROMOTE_TOP_K]:
        pid = f"P5_{c['id']}"
        if any(q["id"] == pid for q in queue):
            continue
        queue.append({
            "id": pid, "phase": "promoted", "status": "pending",
            "hypothesis": f"5-seed confirmation of screen winner {c['id']} "
                          f"(val IC {c['result']['mean_val_ic']})",
            "feature_set": c["feature_set"], "target": c["target"],
            "seq_len": c["seq_len"], "preset_base": c.get("preset_base", "B"),
            "overrides": c.get("overrides"), "seeds": FULL_SEEDS,
            "weight_decay": c.get("weight_decay", 1e-4),
            "loss": c.get("loss", "mse"), "holding": c.get("holding", 20),
            "oos_start": "2023-01-01", "eval_blend": True,
            "gate": "blend50+band10 books vs refs 2.06/-10.7% (2023) — if beaten, add bear run",
        })
        print(f"[promote] queued {pid}")


def spawn_bear(queue):
    """Any promoted config whose blend L/S net60 >= champion-window ref gets a
    bear-window validation run."""
    REF_LS = 2.06
    for c in [q for q in queue if q.get("phase") == "promoted" and q.get("status") == "done"]:
        bl = c.get("result", {}).get("blend_band10", {}).get("long_short", {})
        if bl.get("net60", {}).get("sharpe", -9) >= REF_LS - 0.05:
            bid = f"BEAR_{c['id']}"
            if any(q["id"] == bid for q in queue):
                continue
            inherit = {k: c.get(k) for k in
                       ("feature_set", "target", "seq_len", "preset_base",
                        "overrides", "seeds", "weight_decay", "loss", "holding")}
            inherit = {k: v for k, v in inherit.items() if v is not None}
            queue.append({**inherit,
                          "id": bid, "phase": "bear", "status": "pending",
                          "hypothesis": f"bear-window validation of {c['id']}",
                          "oos_start": "2021-01-01", "eval_blend": True,
                          "gate": "blend books vs bear refs 1.47/-18.7%"})
            print(f"[promote] queued {bid}")


def daily_ops():
    """Refresh + daily inference AFTER the GPU queue completes (never during)."""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "research", "refresh_data.py")],
                       capture_output=True, text=True, timeout=1800)
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    print("[daily]", tail[0])
    import re
    mrows = re.search(r"\+(\d+) rows", tail[0])
    if mrows and int(mrows.group(1)) >= 50:
        for script in ("inference_transformer_eod.py",):
            subprocess.run([sys.executable, os.path.join(ROOT, script)], timeout=3600)
        for args in (["research/paper_trading.py", "snapshot"],
                     ["research/blended_decision_book.py"],
                     ["research/paper_trading.py", "evaluate"]):
            subprocess.run([sys.executable, os.path.join(ROOT, *args[0].split("/"))] + args[1:],
                           timeout=1800)
        print("[daily] new trading day processed")


def run(queue_path=DEFAULT_QUEUE):
    require_cuda()
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    # group order: explicit 'order' field minimizes dataset rebuilds
    def order_key(c):
        return (c.get("order", 99), c["feature_set"], c["seq_len"], c["id"])
    while True:
        pending = [c for c in queue if c.get("status", "pending") == "pending"]
        if not pending:
            promote(queue)
            spawn_bear(queue)
            _save(queue, queue_path)
            pending = [c for c in queue if c.get("status", "pending") == "pending"]
            if not pending:
                break
        cfg = sorted(pending, key=order_key)[0]
        cfg["status"] = "running"
        _save(queue, queue_path)
        print(f"\n===== {cfg['id']} ({cfg.get('phase')}) — {cfg.get('hypothesis','')[:80]}")
        try:
            run_one(cfg)
            cfg["status"] = "done"
        except Exception as e:  # noqa: BLE001
            cfg["status"] = "failed"
            cfg["error"] = f"{type(e).__name__}: {e}"
            _log({"id": cfg["id"], "event": "failed", "error": cfg["error"],
                  "ts": time.strftime("%F %T")})
            print(f"[{cfg['id']}] FAILED: {cfg['error']}")
        _save(queue, queue_path)
    n_done = sum(1 for c in queue if c["status"] == "done")
    n_fail = sum(1 for c in queue if c["status"] == "failed")
    print(f"\n[scheduler] queue complete: {n_done} done, {n_fail} failed")
    try:
        daily_ops()
    except Exception as e:  # noqa: BLE001
        print(f"[daily] skipped: {type(e).__name__}: {e}")


def status(queue_path=DEFAULT_QUEUE):
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    for c in queue:
        r = c.get("result", {})
        ls = r.get("books", {}).get("long_short", {}).get("net60", {}).get("sharpe")
        print(f"{c['id']:24s} {c.get('phase','?'):9s} {c.get('status','pending'):8s} "
              f"val_ic {r.get('mean_val_ic', '')} LS60 {ls if ls is not None else ''}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    qp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUEUE
    {"run": run, "status": status}[mode](qp)
