"""Queue v10-short runner (user-approved Option B).

GPU screens run through gpu_research_scheduler.run_one (dataset reuse, OOM
handling, panel+books+blend eval) but WITHOUT promote()/spawn_bear()/
daily_ops() — no auto-promotion, no data_cache mutation. The single
promotion slot (PROMOTE_TOP_K=1) follows the pre-registered rule stored in
the queue file. C1 is a CPU pre-screen on the frozen 7-seed panels.

Usage:
  python research/run_queue_v10_short.py run
  python research/run_queue_v10_short.py status
"""

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

V10_DIR = os.path.join(ROOT, "reports", "continuous_research", "queue_v10_short")
QUEUE = os.path.join(V10_DIR, "queue_v10_short.json")

VAL_IC_MARGIN = 0.002  # E3 lesson: marginal val-IC edges are not decision-grade
CANDIDATES = ("A1_CS3", "B1_MT3")


def _save(queue):
    with open(QUEUE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def _dump(name, obj):
    with open(os.path.join(V10_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _slim_result(cfg):
    r = cfg.get("result", {})
    out = {"mean_val_ic": r.get("mean_val_ic"), "n_refits": r.get("n_refits"),
           "total_train_s": r.get("total_train_s"), "peak_vram_mb": r.get("peak_vram_mb")}
    for book in ("books", "blend_band10"):
        if book in r:
            out[book] = {mode: {k: r[book][mode][k]
                                for k in ("rank_ic", "avg_turnover", "net60", "net100",
                                          "yearly_net60")}
                         for mode in r[book]}
    return out


def c1_rev_cpu():
    """5d-reversal third-leg CPU pre-screen on the frozen panels."""
    import transformer_portfolio as tp
    from transformer_hybrid import _cache_frames
    from queue_v9_lib import get_merged

    rows = []
    for sid, df in _cache_frames().items():
        c = df["close"].to_numpy(np.float64)
        r5 = np.full(len(c), np.nan)
        r5[5:] = c[5:] / c[:-5] - 1.0
        rows.append(pd.DataFrame({"date": df["date"].values, "stock": sid,
                                  "rev": -r5}))
    rev = pd.concat(rows, ignore_index=True)

    out = {"cells": {}, "note": "band10+cap10 construction throughout; "
           "pre-registered weights 45/45/10 and 40/40/20"}
    for win in ("CH", "BR"):
        m = get_merged(win).merge(rev, on=["date", "stock"], how="left")
        m["z_rev"] = m.groupby("date")["rev"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9))
        cells = {
            "two_way_50_50": 0.5 * m["z_tf"] + 0.5 * m["z_mom"],
            "rev10_45_45_10": 0.45 * m["z_tf"] + 0.45 * m["z_mom"] + 0.10 * m["z_rev"],
            "rev20_40_40_20": 0.40 * m["z_tf"] + 0.40 * m["z_mom"] + 0.20 * m["z_rev"],
            "rev_alone": m["z_rev"],
            "tf_alone": m["z_tf"],
            "d12_alone": m["z_mom"],
        }
        out["cells"][win] = {}
        for name, sc in cells.items():
            r = tp.backtest_scores(m.assign(score=sc), holding=20,
                                   mode="long_short", no_trade_band=0.10)
            out["cells"][win][name] = {
                "net60_sharpe": r["net60"]["sharpe"], "max_dd": r["net60"]["max_dd"],
                "rank_ic": round(r["rank_ic"], 4), "turnover": round(r["avg_turnover"], 3)}
    wins = {c: all(out["cells"][w][c]["net60_sharpe"] >
                   out["cells"][w]["two_way_50_50"]["net60_sharpe"]
                   for w in ("CH", "BR"))
            for c in ("rev10_45_45_10", "rev20_40_40_20")}
    passing = [c for c, v in wins.items() if v]
    out["verdict"] = (f"GPU BRANCH JUSTIFIED (v10b): {','.join(passing)} beats "
                      "two-way on both windows" if passing else
                      "REJECT — no reversal cell beats the two-way blend on both "
                      "windows; GPU branch not justified")
    return out


def promotion(queue):
    """PROMOTE_TOP_K=1 per the pre-registered rule in the queue file."""
    from gpu_research_scheduler import run_one
    by_id = {c["id"]: c for c in queue}
    base = by_id["BASE3_s012"]["result"]["mean_val_ic"]
    cands = {cid: by_id[cid]["result"]["mean_val_ic"] for cid in CANDIDATES
             if by_id.get(cid, {}).get("status") == "done"}
    out = {"baseline_val_ic": base, "candidates": cands,
           "rule": f"argmax val_ic must exceed baseline + {VAL_IC_MARGIN}"}
    if not cands:
        out["verdict"] = "NO CANDIDATES COMPLETED — no promotion"
        return out
    winner = max(cands, key=cands.get)
    if cands[winner] < base + VAL_IC_MARGIN:
        out["verdict"] = (f"NO PROMOTION — best candidate {winner} "
                          f"({cands[winner]}) does not clear baseline {base} "
                          f"+ {VAL_IC_MARGIN}; Short v10 closes cleanly (rule 11)")
        return out
    out["winner"] = winner
    win_cfg = by_id[winner]
    runs = {}
    for tag, oos in (("champ_2023", None), ("bear_2021", "2021-01-01")):
        cfg = {"id": f"P7_{winner}_{tag}", "phase": "v10promoted",
               "feature_set": win_cfg["feature_set"], "target": win_cfg["target"],
               "seq_len": win_cfg["seq_len"], "preset_base": win_cfg["preset_base"],
               "overrides": win_cfg.get("overrides"),
               "seeds": [0, 1, 2, 3, 4, 5, 6], "eval_blend": True}
        if oos:
            cfg["oos_start"] = oos
        print(f"[promote] running {cfg['id']} (7 seeds)", flush=True)
        run_one(cfg)
        cfg["status"] = "done"
        runs[tag] = cfg
        _dump(f"PROMOTE_{tag}_config.json",
              {k: v for k, v in cfg.items() if k != "result"})
        _dump(f"PROMOTE_{tag}_metrics.json", _slim_result(cfg))
    # dual-window comparison vs the four standing books (from v9 C1)
    BASELINES = {"blend_default": {"champ_2023": 2.147, "bear_2021": 1.443},
                 "d7b": {"champ_2023": 2.155, "bear_2021": 1.441},
                 "tf": {"champ_2023": 1.672, "bear_2021": 1.193},
                 "d12": {"champ_2023": 1.570, "bear_2021": 1.361}}
    comp = {}
    for tag, cfg in runs.items():
        bl = cfg["result"]["blend_band10"]["long_short"]["net60"]
        comp[tag] = {"candidate_blend_LS_net60": bl["sharpe"],
                     "candidate_max_dd": bl["max_dd"],
                     "vs": {k: round(bl["sharpe"] - v[tag], 3)
                            for k, v in BASELINES.items()}}
    out["dual_window"] = comp
    ch, br = comp["champ_2023"], comp["bear_2021"]
    in_range = (1.85 <= ch["candidate_blend_LS_net60"] and
                1.30 <= br["candidate_blend_LS_net60"])
    out["verdict"] = (f"PROMOTED {winner}: blend {ch['candidate_blend_LS_net60']} / "
                      f"{br['candidate_blend_LS_net60']} — "
                      + ("within/above seed-robust ranges; ADOPTION DECISION TO USER"
                         if in_range else "BELOW range on at least one window — REJECT"))
    return out


def run():
    from gpu_research_scheduler import run_one  # noqa: F401 (also warms imports)
    with open(QUEUE, encoding="utf-8") as f:
        queue = json.load(f)
    for cfg in queue:
        if cfg.get("status", "pending") != "pending":
            continue
        cfg["status"] = "running"
        _save(queue)
        t0 = time.time()
        print(f"\n===== {cfg['id']} ({cfg['exp']}) — {cfg['hypothesis'][:90]}",
              flush=True)
        try:
            if cfg["phase"] == "cpu":
                result = c1_rev_cpu()
                _dump(f"{cfg['id']}_metrics.json", result)
                cfg["verdict"] = result["verdict"]
            elif cfg["phase"] == "promotion":
                result = promotion(queue)
                _dump("PROMOTE_metrics.json", result)
                cfg["verdict"] = result["verdict"]
            else:
                run_one(cfg)
                _dump(f"{cfg['id']}_config.json",
                      {k: v for k, v in cfg.items() if k != "result"})
                _dump(f"{cfg['id']}_metrics.json", _slim_result(cfg))
                cfg["verdict"] = f"val_ic {cfg['result']['mean_val_ic']}"
            cfg["status"] = "done"
            cfg["runtime_s"] = round(time.time() - t0, 1)
            print(f"[{cfg['id']}] done {cfg['runtime_s']}s — {cfg['verdict']}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            cfg["status"] = "failed"
            cfg["error"] = f"{type(e).__name__}: {e}"
            print(f"[{cfg['id']}] FAILED: {cfg['error']}", flush=True)
            traceback.print_exc()
            if "CUDA" in cfg["error"]:
                print("[v10] CUDA error — stopping queue (no retries per rule 12)",
                      flush=True)
                _save(queue)
                return
        _save(queue)
    n_done = sum(1 for c in queue if c["status"] == "done")
    n_fail = sum(1 for c in queue if c["status"] == "failed")
    print(f"\n[v10-short] queue complete: {n_done} done, {n_fail} failed", flush=True)


def status():
    with open(QUEUE, encoding="utf-8") as f:
        queue = json.load(f)
    for c in queue:
        print(f"{c['id']:18s} {c.get('status','pending'):8s} "
              f"{c.get('runtime_s','')!s:>8s}  {c.get('verdict','')[:80]}")


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
