"""Queue v9 runner — CPU-only deployment validation queue.

Crash-safe like the GPU scheduler: queue state rewritten after every
experiment; a crash resumes at the first non-done item. NO GPU training,
NO cache mutation (all loads read-only), signal frozen to 7-seed panels.

Usage:
  python research/run_queue_v9.py run    [queue.json]
  python research/run_queue_v9.py status [queue.json]
"""

import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

V9_DIR = os.path.join(ROOT, "reports", "continuous_research", "queue_v9")
DEFAULT_QUEUE = os.path.join(V9_DIR, "queue_v9.json")
LOG = os.path.join(V9_DIR, "runner_log.jsonl")


def _save(queue, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def _log(rec):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run(queue_path=DEFAULT_QUEUE):
    from queue_v9_experiments import EXPERIMENTS
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    for cfg in queue:
        if cfg.get("status", "pending") != "pending":
            continue
        cfg["status"] = "running"
        _save(queue, queue_path)
        t0 = time.time()
        print(f"\n===== {cfg['id']} ({cfg['track']}) — {cfg['hypothesis'][:90]}",
              flush=True)
        try:
            result = EXPERIMENTS[cfg["id"]]()
            out_path = os.path.join(V9_DIR, f"{cfg['id']}_result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            cfg["status"] = "done"
            cfg["verdict"] = result.get("verdict", "")
            cfg["result_file"] = os.path.basename(out_path)
            cfg["runtime_s"] = round(time.time() - t0, 1)
            _log({"id": cfg["id"], "elapsed_s": cfg["runtime_s"],
                  "verdict": cfg["verdict"], "ts": time.strftime("%F %T")})
            print(f"[{cfg['id']}] done in {cfg['runtime_s']}s — {cfg['verdict']}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            cfg["status"] = "failed"
            cfg["error"] = f"{type(e).__name__}: {e}"
            _log({"id": cfg["id"], "event": "failed", "error": cfg["error"],
                  "ts": time.strftime("%F %T")})
            print(f"[{cfg['id']}] FAILED: {cfg['error']}", flush=True)
            traceback.print_exc()
        _save(queue, queue_path)
    n_done = sum(1 for c in queue if c["status"] == "done")
    n_fail = sum(1 for c in queue if c["status"] == "failed")
    print(f"\n[v9] queue complete: {n_done} done, {n_fail} failed", flush=True)


def status(queue_path=DEFAULT_QUEUE):
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    for c in queue:
        print(f"{c['id']:4s} {c['track']:8s} {c.get('status','pending'):8s} "
              f"{c.get('runtime_s','')!s:>8s}  {c.get('verdict','')[:90]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    qp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUEUE
    {"run": run, "status": status}[mode](qp)
