"""Queue v10b runner — phase 1: MT validation battery (user-approved).

Same discipline as run_queue_v10_short: run_one for GPU items (no
promote/spawn/daily_ops — no cache mutation), crash-safe queue rewrites,
CUDA error = stop immediately (no retries). The battery verdict applies the
pre-registered gates stored in the queue file.

Usage:
  python research/run_queue_v10b.py run
  python research/run_queue_v10b.py status
"""

import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

V10B_DIR = os.path.join(ROOT, "reports", "continuous_research", "queue_v10b")
QUEUE = os.path.join(V10B_DIR, "queue_v10b.json")


def _save(queue, path=None):
    with open(path or QUEUE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def _dump(name, obj):
    with open(os.path.join(V10B_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _slim(cfg):
    r = cfg.get("result", {})
    out = {"mean_val_ic": r.get("mean_val_ic"), "n_refits": r.get("n_refits"),
           "total_train_s": r.get("total_train_s"),
           "peak_vram_mb": r.get("peak_vram_mb")}
    for book in ("books", "blend_band10"):
        if book in r:
            out[book] = {m: {k: r[book][m][k]
                             for k in ("rank_ic", "avg_turnover", "net60",
                                       "net100", "yearly_net60")}
                         for m in r[book]}
    return out


def battery_verdict(queue):
    by_id = {c["id"]: c for c in queue}

    def blend(cid):
        return by_id[cid]["result"]["blend_band10"]["long_short"]

    out = {"gates": {}, "reference": {"original_draw": {"champ": 2.011, "bear": 1.515,
                                                        "bear_dd": -0.1301, "yr2022": 0.23}}}
    g = {}
    b1 = blend("SR_MT1_disjoint_2023")["net60"]
    g["SR_MT1_champ_in_range"] = {"value": b1["sharpe"], "dd": b1["max_dd"],
                                  "pass": 1.85 <= b1["sharpe"] <= 2.15}
    b2 = blend("SR_MT2_disjoint_2021")
    yr = b2["yearly_net60"]
    y22 = (yr.get(2022) or yr.get("2022") or {}).get("sharpe")
    g["SR_MT2_bear_holds"] = {"value": b2["net60"]["sharpe"], "dd": b2["net60"]["max_dd"],
                              "yr2022": y22,
                              "pass": (b2["net60"]["sharpe"] >= 1.44
                                       and (y22 is not None and y22 >= 0.0)
                                       and b2["net60"]["max_dd"] >= -0.16)}
    b3 = blend("RF_MT_refit63_2023")["net60"]
    g["RF_MT_protocol_robust"] = {"value": b3["sharpe"], "dd": b3["max_dd"],
                                  "pass": 1.85 <= b3["sharpe"] <= 2.15}
    out["gates"] = g
    n_pass = sum(1 for v in g.values() if v["pass"])
    if n_pass == 3:
        out["verdict"] = ("MT VALIDATED (3/3 gates) — formal challenger / "
                          "defensive-spec candidate; ADOPTION DECISION TO USER")
    else:
        failed = [k for k, v in g.items() if not v["pass"]]
        out["verdict"] = (f"MT NOT VALIDATED ({n_pass}/3 gates; failed: "
                          f"{', '.join(failed)}) — seed-luck component recorded, "
                          "line closed, champion stands")
    return out


def c2_residual_target(data):
    """v10b-C2: inject tgt_rank_res20 (rank of market-beta-residualized fwd20)
    into data['targets']. Beta = 126d rolling cov/var of stock vs equal-weight
    market proxy, info through t. Cache read-only; in-memory only."""
    import numpy as np
    import pandas as pd
    from transformer_hybrid import _cache_frames
    frames = _cache_frames()
    rets = {}
    for sid, df in frames.items():
        c = df.set_index("date")["close"]
        rets[sid] = np.log(c / c.shift(1))
    r = pd.DataFrame(rets).sort_index()
    mkt = r.mean(axis=1)
    var = mkt.rolling(126).var()
    beta = r.rolling(126).cov(mkt).div(var, axis=0)
    # fwd20 (exec lag 1) per stock and for the market proxy
    from queue_v9_lib import lag_returns
    lag = lag_returns()[["date", "stock", "fwd_lag1"]].rename(
        columns={"fwd_lag1": "fwd20"})
    mkt_fwd = lag.groupby("date")["fwd20"].mean().rename("mkt_fwd20")
    b_long = beta.stack().rename("beta").reset_index()
    b_long.columns = ["date", "stock", "beta"]
    m = lag.merge(b_long, on=["date", "stock"], how="left").merge(
        mkt_fwd, on="date", how="left")
    m["res"] = m["fwd20"] - m["beta"].fillna(1.0) * m["mkt_fwd20"]
    m["tgt"] = m.groupby("date")["res"].rank(pct=True)
    key = m.set_index(["date", "stock"])["tgt"]
    dates = pd.to_datetime(data["dates"][data["date_rank"]])
    stocks = np.asarray(data["stocks"])[data["stock_idx"]]
    idx = pd.MultiIndex.from_arrays([dates, stocks])
    data["targets"]["tgt_rank_res20"] = key.reindex(idx).to_numpy()
    return data


def d2_distill():
    """v10b-D2: distill the 7 production daily checkpoints (teachers, asof
    2026-07-24 — no new data since) into one student at the same refit.
    Gate: student val IC >= 0.95x ensemble val IC. Teachers are read-only."""
    import numpy as np
    import torch
    from train_transformer_eod import (PRESETS, build_net, fit_one, to_gpu,
                                       predict_idx, _rank_ic)
    from dataset_transformer_eod import build_dataset, matured_train_val
    ck_dir = os.path.join(ROOT, "checkpoints", "transformer_eod")
    data = build_dataset("close_only", seq_len=60, horizons=(20,))
    Xg = to_gpu(data)
    y_np = np.nan_to_num(np.clip(data["targets"]["tgt_rank_20"], -1, 1))
    yg = torch.as_tensor(y_np, device=Xg.device)
    refit_rank = int(data["date_rank"].max())
    tr, va, _ = matured_train_val(data, "tgt_rank_20", refit_rank, 20)
    cfg = PRESETS["B"]
    preds = []
    for s in range(7):
        ck = torch.load(os.path.join(ck_dir, f"daily_seed{s}.pt"),
                        map_location=Xg.device, weights_only=False)
        net = build_net(Xg.shape[2], ck.get("cfg", cfg)).to(Xg.device)
        net.load_state_dict(ck["model_state_dict"])
        all_idx = np.concatenate([np.asarray(tr), np.asarray(va)])
        preds.append(predict_idx(net, Xg, all_idx).cpu().numpy())
        del net
        torch.cuda.empty_cache()
    teacher = np.mean(preds, axis=0)
    n_tr = len(np.asarray(tr))
    ens_ic, _ = _rank_ic(teacher[n_tr:], y_np[np.asarray(va)],
                         data["date_rank"][np.asarray(va)])
    # student: 50/50 target + teacher soft scores (blended regression target)
    y_soft = y_np.copy()
    y_soft[np.asarray(tr)] = 0.5 * y_np[np.asarray(tr)] + 0.5 * teacher[:n_tr]
    yg_soft = torch.as_tensor(y_soft, device=Xg.device)
    net, _, info = fit_one(Xg, yg_soft, tr, va, data["date_rank"][va], cfg,
                           seed=0)
    sv = predict_idx(net, Xg, np.asarray(va)).cpu().numpy()
    stu_ic, _ = _rank_ic(sv, y_np[np.asarray(va)], data["date_rank"][np.asarray(va)])
    out = {"ensemble_val_ic": round(float(ens_ic), 5),
           "student_val_ic": round(float(stu_ic), 5),
           "ratio": round(float(stu_ic / max(ens_ic, 1e-9)), 3),
           "student_train_s": info["train_s"],
           "teachers": "checkpoints/transformer_eod/daily_seed0-6.pt (read-only)"}
    out["verdict"] = ("PASS — student recovers "
                      f"{out['ratio']:.0%} of ensemble val IC (bar 95%); daily "
                      "retrain could drop 7x -> 1x" if out["ratio"] >= 0.95 else
                      f"REJECT — student {out['ratio']:.0%} < 95% of ensemble; "
                      "the 7-seed ensemble earns its cost")
    return out


def run(queue_path=QUEUE):
    from gpu_research_scheduler import run_one, get_data
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    for cfg in queue:
        if cfg.get("status", "pending") != "pending":
            continue
        cfg["status"] = "running"
        _save(queue, queue_path)
        t0 = time.time()
        print(f"\n===== {cfg['id']} — {cfg['hypothesis'][:90]}", flush=True)
        try:
            if cfg["item"] == "battery_verdict":
                result = battery_verdict(queue)
                _dump("MT_BATTERY_VERDICT.json", result)
                cfg["verdict"] = result["verdict"]
            elif cfg["item"] == "c2_gpu":
                data, _ = get_data(cfg["feature_set"], cfg["seq_len"])
                c2_residual_target(data)
                run_one(cfg)
                _dump(f"{cfg['id']}_config.json",
                      {k: v for k, v in cfg.items() if k != "result"})
                _dump(f"{cfg['id']}_metrics.json", _slim(cfg))
                vic = cfg["result"]["mean_val_ic"]
                oos_ic = cfg["result"]["books"]["long_short"]["rank_ic"]
                trip = vic is not None and vic >= 0.055 and oos_ic < 0.02
                cfg["verdict"] = (f"val_ic {vic} oos_ic {round(oos_ic,4)}"
                                  + (" — INVERSION #3 TRIPWIRE" if trip else ""))
            elif cfg["item"] == "d2_distill":
                result = d2_distill()
                _dump("D2_DISTILL_metrics.json", result)
                cfg["verdict"] = result["verdict"]
            else:
                run_one(cfg)
                _dump(f"{cfg['id']}_config.json",
                      {k: v for k, v in cfg.items() if k != "result"})
                _dump(f"{cfg['id']}_metrics.json", _slim(cfg))
                bl = cfg["result"]["blend_band10"]["long_short"]["net60"]
                cfg["verdict"] = (f"val_ic {cfg['result']['mean_val_ic']} "
                                  f"blend LS {bl['sharpe']} dd {bl['max_dd']}")
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
                print("[v10b] CUDA error — stopping queue (rule: no retries)",
                      flush=True)
                _save(queue, queue_path)
                return
        _save(queue, queue_path)
    n_done = sum(1 for c in queue if c["status"] == "done")
    n_fail = sum(1 for c in queue if c["status"] == "failed")
    print(f"\n[v10b] phase complete: {n_done} done, {n_fail} failed", flush=True)


def status(queue_path=QUEUE):
    with open(queue_path, encoding="utf-8") as f:
        queue = json.load(f)
    for c in queue:
        print(f"{c['id']:22s} {c.get('status','pending'):8s} "
              f"{c.get('runtime_s','')!s:>8s}  {c.get('verdict','')[:80]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    qp = sys.argv[2] if len(sys.argv) > 2 else QUEUE
    {"run": run, "status": status}[mode](qp)
