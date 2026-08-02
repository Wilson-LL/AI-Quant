"""Queue v12 runner — big LSTM+Transformer / recency / epochs / long-short.

Research-only. Plan: reports/continuous_research/v12_big_transformer/
queue_v12_plan.md (pre-registered; every phase launch is user-gated).

Never touches: checkpoints/transformer_eod/, daily_manifest, data_cache,
paper ledger, holdings overlay, production presets. v12 checkpoints go to
checkpoints/v12_big_transformer/ (gitignored).

Commands:
  python research/run_queue_v12_big_transformer.py inspect|estimate
  python research/run_queue_v12_big_transformer.py materialize [--force]
  python research/run_queue_v12_big_transformer.py dry-run
  python research/run_queue_v12_big_transformer.py smoke        # 1 band quick
  python research/run_queue_v12_big_transformer.py run --phase 0|1
  python research/run_queue_v12_big_transformer.py unlock --phase 2|3|4
  python research/run_queue_v12_big_transformer.py report|status

Phase 2/3/4 items are materialized as status=held; `unlock` flips them to
pending ONLY after the user approves that phase (their execution paths are
added with the same approval).
"""

import argparse
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

V12_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v12_big_transformer")
LOG_DIR = os.path.join(V12_DIR, "logs")
QUEUE = os.path.join(V12_DIR, "queue_v12.json")
GPU_CSV = os.path.join(V12_DIR, "queue_v12_gpu_usage.csv")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "v12_big_transformer")
PROD_CKPT = os.path.join(ROOT, "checkpoints", "transformer_eod")

from model_capacity_estimator import BANDS  # noqa: E402  (exact configs)

# Phase-1 feasibility bands with their large-training settings.
# Micro-batches set from MEASURED Phase-1 VRAM (L1 hit 9.2 GB at micro 512 —
# analytic estimates were low), not from the estimator.
P1_BANDS = {
    "M2_h256L4":   dict(micro_batch=1024, accum_steps=1, grad_ckpt=False),
    "L1_h512L8":   dict(micro_batch=512,  accum_steps=2, grad_ckpt=False),
    "XL1_h1024L16": dict(micro_batch=128, accum_steps=8, grad_ckpt=True),
    "XL2_h1024L24": dict(micro_batch=64,  accum_steps=16, grad_ckpt=True),
}


def _is_oom(e):
    """torch nightly raises AcceleratorError (not cuda.OutOfMemoryError) for
    some OOM paths — classify by message, not type."""
    return "out of memory" in str(e).lower()
SMOKE_WINDOW_DAYS = 504   # feasibility fits train on the last ~2y only (speed)
SMOKE_EPOCHS = 2
ANCHOR = {"date": "2026-07-29", "CH": 2.147, "BR": 1.443,
          "source": "v11 P0 (exact reproduction of standing refs, 2026-07-30)"}

# Phase-2 recipe grid (S-band model; execution path lands with P2 approval)
P2_RECIPES = ["Wall_baseline", "W1y", "W2y", "W3y",
              "HL63", "HL126", "HL252", "HL504",
              "HY3y126", "HY3y252", "CAL_bands"]
# Phase-3 model screens (recipe chosen by P2; XL = reduced protocol)
P3_MODELS = ["M2_h256L4", "M3_h256L6", "L1_h512L8", "L3_h768L8",
             "XL1_h1024L16", "XL2_h1024L24"]


def _load():
    with open(QUEUE, encoding="utf-8") as f:
        return json.load(f)


def _save(q):
    with open(QUEUE, "w", encoding="utf-8") as f:
        json.dump(q, f, indent=2, default=str)


def _gpu_row(row):
    hdr = not os.path.exists(GPU_CSV)
    pd.DataFrame([row]).to_csv(GPU_CSV, mode="a", header=hdr, index=False)


def _prod_snapshot():
    """Fingerprint of the production checkpoint dir (must never change)."""
    if not os.path.isdir(PROD_CKPT):
        return {}
    return {f: (os.path.getsize(os.path.join(PROD_CKPT, f)),
                os.path.getmtime(os.path.join(PROD_CKPT, f)))
            for f in sorted(os.listdir(PROD_CKPT))}


def _band_cfg(band):
    b = BANDS[band]
    return {"hidden": b["hidden"], "lstm_layers": 1,
            "trans_layers": b["trans_layers"], "heads": b["heads"],
            "ff": b["ff"], "dropout": 0.2, "seq_len": b["seq_len"],
            "max_epochs": 25, "patience": 3}


# ------------------------------------------------------ big-model fit loop

def _forward_big(net, x, grad_ckpt):
    """LSTM_CondTransformer forward composed from its own submodules, with
    optional per-layer gradient checkpointing (model.py untouched)."""
    import torch
    lstm_out, _ = net.lstm(x)
    h = net.trans_input_proj(lstm_out) + net.pos_emb
    kv = net.cross_kv_proj(x)
    attn_out, _ = net.cross_attn(query=h, key=kv, value=kv)
    h = h + attn_out
    if grad_ckpt and net.training:
        from torch.utils.checkpoint import checkpoint
        for layer in net.transformer.layers:
            h = checkpoint(layer, h, use_reentrant=False)
    else:
        h = net.transformer(h)
    return net.fc(h[:, -1, :]).squeeze(-1)


def fit_big(Xg, yg, tr_idx, va_idx, va_dranks, cfg, seed, micro_batch,
            accum_steps, grad_ckpt, epochs, lr=3e-4, weight_decay=1e-4,
            weights=None, log=print, max_seconds=None):
    """Research fit loop: AMP + micro-batch + gradient accumulation +
    optional gradient checkpointing + OOM backoff. Returns
    (net, best_val_ic, curves list). Never writes production files."""
    import torch
    import train_transformer_eod as T
    attempt, mb = 0, micro_batch
    while True:
        attempt += 1
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = T.build_net(Xg.shape[2], cfg).to(Xg.device)
            opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                    weight_decay=weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=Xg.is_cuda)
            tr = torch.as_tensor(np.asarray(tr_idx), device=Xg.device)
            va = torch.as_tensor(np.asarray(va_idx), device=Xg.device)
            w_full = None
            if weights is not None:
                w_full = torch.zeros(Xg.shape[0], device=Xg.device)
                w_full[tr] = torch.as_tensor(np.asarray(weights, np.float32),
                                             device=Xg.device)
                w_full[tr] /= w_full[tr].mean().clamp_min(1e-8)
            best_ic, best_state, curves = -1e9, None, []
            eff_batch = mb * accum_steps
            for ep in range(epochs):
                torch.cuda.reset_peak_memory_stats() if Xg.is_cuda else None
                t_ep = time.time()
                net.train()
                perm = tr[torch.randperm(len(tr), device=Xg.device)]
                tot, ns, gnorms = 0.0, 0, []
                opt.zero_grad(set_to_none=True)
                for i in range(0, len(perm), mb):
                    b = perm[i:i + mb]
                    with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                        pred = _forward_big(net, Xg[b], grad_ckpt)
                        err = (pred - yg[b]) ** 2
                        if w_full is not None:
                            err = err * w_full[b]
                        loss = err.mean() / accum_steps
                    scaler.scale(loss).backward()
                    tot += float(loss.detach()) * accum_steps * len(b)
                    ns += len(b)
                    if ((i // mb) + 1) % accum_steps == 0 or i + mb >= len(perm):
                        scaler.unscale_(opt)
                        g = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        gnorms.append(float(g))
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                # validation rank IC — eval batch scales with micro_batch
                # (fixed 8192 OOMed at h1024: ~3GB just for packed QKV)
                eb = min(2048, max(64, mb * 4))
                net.eval()
                with torch.no_grad():
                    vp = torch.empty(len(va), device=Xg.device)
                    for i in range(0, len(va), eb):
                        b = va[i:i + eb]
                        with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                            vp[i:i + len(b)] = _forward_big(net, Xg[b], False).float()
                vic, _ = T._rank_ic(vp.cpu().numpy(),
                                    yg[va].cpu().numpy(), va_dranks)
                curves.append({
                    "epoch": ep, "train_loss": round(tot / max(ns, 1), 5),
                    "val_ic": round(vic, 5),
                    "grad_norm_mean": round(float(np.mean(gnorms)), 3) if gnorms else None,
                    "epoch_s": round(time.time() - t_ep, 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    if Xg.is_cuda else 0,
                    "micro_batch": mb, "eff_batch": eff_batch, "lr": lr})
                log(f"    ep{ep} loss {curves[-1]['train_loss']} "
                    f"val_ic {vic:+.4f} {curves[-1]['epoch_s']}s "
                    f"vram {curves[-1]['peak_vram_gb']}GB")
                if vic > best_ic:
                    best_ic = vic
                    best_state = {k: v.detach().clone()
                                  for k, v in net.state_dict().items()}
                if max_seconds and sum(c["epoch_s"] for c in curves) > max_seconds:
                    log(f"    wall-clock budget reached after epoch {ep} — "
                        "stopping (best-by-val state kept)")
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            return net, best_ic, curves
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                raise
            try:
                del net, opt, scaler
            except NameError:
                pass
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if attempt >= 4 or mb <= 16:
                raise
            mb = max(16, mb // 2)
            log(f"    OOM -> retry with micro_batch {mb} (attempt {attempt + 1})")


# ------------------------------------------------------------- materialize

def materialize(force=False):
    os.makedirs(V12_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(QUEUE) and not force:
        print(f"queue exists: {QUEUE} (use --force to rebuild)")
        return
    q = [{"id": "P0_anchor_check", "phase": "P0", "status": "pending",
          "anchor": ANCHOR}]
    for band in P1_BANDS:
        q.append({"id": f"P1_feas_{band}", "phase": "P1", "band": band,
                  "status": "pending", **P1_BANDS[band],
                  "epochs": SMOKE_EPOCHS, "window_days": SMOKE_WINDOW_DAYS})
    for r in P2_RECIPES:
        q.append({"id": f"P2_recipe_{r}", "phase": "P2", "recipe": r,
                  "status": "held",
                  "hold_reason": "Phase 2 requires user approval"})
    for m in P3_MODELS:
        q.append({"id": f"P3_model_{m}", "phase": "P3", "band": m,
                  "status": "held",
                  "hold_reason": "Phase 3 requires user approval",
                  "protocol": ("reduced (1 seed, single refit, 5h budget)"
                               if m.startswith("XL") else
                               "walkforward screen (3 seeds, CH)")})
    _save(q)
    import subprocess
    man = {"queue": "v12_big_transformer", "created": time.strftime("%F %T"),
           "branch": subprocess.run(["git", "branch", "--show-current"],
                                    capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip(),
           "plan": "queue_v12_plan.md (pre-registered)",
           "anchor": ANCHOR, "phase1_bands": P1_BANDS,
           "phase2_recipes": P2_RECIPES, "phase3_models": P3_MODELS,
           "gates": "plan §5; +1% relative book Sharpe minimum; "
                    "no val-IC-only adoption"}
    with open(os.path.join(V12_DIR, "queue_v12_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(man, f, indent=2, default=str)
    print(f"materialized {len(q)} items -> {QUEUE}")


# ------------------------------------------------------------------ phases

def _latest_cache_date():
    p = os.path.join(ROOT, "research", "data_cache", "2330.csv")
    df = pd.read_csv(p, usecols=["date"]).drop_duplicates()
    return str(df["date"].iloc[-1])[:10]


def run_phase0():
    q = _load()
    item = next(c for c in q if c["id"] == "P0_anchor_check")
    item["status"] = "running"
    _save(q)
    cache_date = _latest_cache_date()
    if cache_date == ANCHOR["date"]:
        item["status"] = "done"
        item["verdict"] = (f"ANCHOR REUSED: cache unchanged ({cache_date}); "
                           f"v11 P0 panels stand — CH {ANCHOR['CH']} / "
                           f"BR {ANCHOR['BR']} (exact standing refs)")
    else:
        item["status"] = "stale"
        item["verdict"] = (f"cache advanced to {cache_date} > anchor "
                           f"{ANCHOR['date']} — baseline re-run required "
                           "before candidate comparisons (ask user; ~2h GPU)")
    _save(q)
    print(f"[P0] {item['verdict']}")
    return item["status"] == "done"


def run_band(band, micro_batch, accum_steps, grad_ckpt, epochs=SMOKE_EPOCHS):
    """CHILD-PROCESS feasibility run for one band. A hard OOM
    (AcceleratorError) poisons the CUDA context, so each band gets a fresh
    process; the result lands in feas_<band>.json for the parent."""
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data
    from inference_transformer_eod import build_net_from_ck
    out_p = os.path.join(V12_DIR, f"feas_{band}.json")
    t0 = time.time()
    try:
        T.require_cuda()
        os.makedirs(CKPT_DIR, exist_ok=True)
        data, Xg = get_data("close_only", 60)
        yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]),
                             device=Xg.device)
        refit_rank = int(data["date_rank"].max())
        tr, va, _ = T.matured_train_val(data, "tgt_rank_20", refit_rank, 20,
                                        recency=None)
        cutoff = refit_rank - SMOKE_WINDOW_DAYS
        tr_sub = np.asarray(tr)[np.asarray(data["date_rank"])[tr] >= cutoff]
        va_dranks = np.asarray(data["date_rank"])[va]
        print(f"[P1:{band}] subset {len(tr_sub):,} samples; micro "
              f"{micro_batch} × accum {accum_steps} grad_ckpt={grad_ckpt}",
              flush=True)
        cfg = _band_cfg(band)
        net, vic, curves = fit_big(Xg, yg, tr_sub, va, va_dranks, cfg, seed=0,
                                   micro_batch=micro_batch,
                                   accum_steps=accum_steps,
                                   grad_ckpt=grad_ckpt, epochs=epochs)
        ck_p = os.path.join(CKPT_DIR, f"feas_{band}_seed0.pt")
        torch.save({"model_state_dict": net.state_dict(), "cfg": cfg,
                    "feature_set": "close_only", "target": "tgt_rank_20",
                    "horizon": 20, "recency": None, "seed": 0,
                    "val_ic": vic, "band": band,
                    "asof": str(data["dates"][refit_rank])[:10]}, ck_p)
        ck_mb = os.path.getsize(ck_p) / 2**20
        del net
        torch.cuda.empty_cache()
        # inference-compat: reload via the production loader path; small
        # predict batch — eval activations at h1024 are the proven OOM risk
        ck = torch.load(ck_p, map_location=Xg.device, weights_only=False)
        net2 = build_net_from_ck(ck).to(Xg.device)
        net2.load_state_dict(ck["model_state_dict"])
        net2.eval()
        idx = torch.as_tensor(
            np.nonzero(np.asarray(data["date_rank"]) == refit_rank)[0],
            device=Xg.device)
        t_inf = time.time()
        scores = T.predict_idx(net2, Xg, idx, batch=min(512, micro_batch * 4))
        infer_s = time.time() - t_inf
        scores = scores.cpu().numpy()
        ok = bool(np.isfinite(scores).all())
        r = {"band": band, "ok": ok, "params": BANDS_PARAMS.get(band),
             "val_ic_2ep": round(vic, 5), "ckpt_mb": round(ck_mb, 1),
             "epoch_s": [c["epoch_s"] for c in curves],
             "peak_vram_gb": max(c["peak_vram_gb"] for c in curves),
             "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
             "micro_batch_final": curves[-1]["micro_batch"],
             "accum_steps": accum_steps, "grad_ckpt": grad_ckpt,
             "grad_norm": curves[-1]["grad_norm_mean"],
             "infer_s_latest_date": round(infer_s, 2),
             "n_scored": int(len(scores)),
             "total_s": round(time.time() - t0, 1),
             "gpu": torch.cuda.get_device_name(0), "curves": curves}
        print(f"[P1:{band}] ckpt {r['ckpt_mb']}MB epoch {r['epoch_s'][-1]}s "
              f"vram {r['peak_vram_gb']}GB ok={ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        r = {"band": band, "ok": False, "error": f"{type(e).__name__}: {e}",
             "is_oom": _is_oom(e), "total_s": round(time.time() - t0, 1)}
        print(f"[P1:{band}] FAILED: {r['error']}", flush=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, default=str)
    return 0 if r.get("ok") else 1


def run_phase1():
    """PARENT: one subprocess per band — a hard OOM only kills the child."""
    import subprocess
    q = _load()
    prod_before = _prod_snapshot()
    feas = {}
    for cfg_item in [c for c in q if c["phase"] == "P1"
                     and c.get("status") == "pending"]:
        band = cfg_item["band"]
        cfg_item["status"] = "running"
        _save(q)
        print(f"\n[P1] spawning {band} (fresh process)", flush=True)
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "run-band",
             "--band", band,
             "--micro-batch", str(cfg_item["micro_batch"]),
             "--accum-steps", str(cfg_item["accum_steps"])]
            + (["--grad-ckpt"] if cfg_item["grad_ckpt"] else []),
            cwd=ROOT)
        rp = os.path.join(V12_DIR, f"feas_{band}.json")
        r = (json.load(open(rp, encoding="utf-8"))
             if os.path.exists(rp) else
             {"band": band, "ok": False, "error": "child wrote no result"})
        feas[band] = r
        if r.get("ok"):
            cfg_item.update(status="done", result={k: v for k, v in r.items()
                                                   if k != "curves"},
                            verdict=f"ckpt {r['ckpt_mb']}MB, epoch "
                                    f"{r['epoch_s'][-1]}s, vram "
                                    f"{r['peak_vram_gb']}GB, ok=True")
            _gpu_row({"ts": time.strftime("%F %T"), "id": f"P1_{band}",
                      "gpu": r.get("gpu"), "total_vram_gb": 16.0,
                      "max_alloc_gb": r["peak_vram_gb"],
                      "max_reserved_gb": r.get("max_reserved_gb"),
                      "micro_batch": r["micro_batch_final"],
                      "accum_steps": r["accum_steps"],
                      "grad_ckpt": r["grad_ckpt"], "amp": True,
                      "epoch_s_last": r["epoch_s"][-1],
                      "ckpt_mb": r["ckpt_mb"],
                      "infer_s": r["infer_s_latest_date"]})
        else:
            cfg_item.update(status="failed", error=r.get("error", "unknown"))
            if "CUDA" in r.get("error", "") and not r.get("is_oom"):
                _save(q)
                print("[P1] non-OOM CUDA error in child — stopping "
                      "(hardware-note rule)", flush=True)
                return
        print(f"[P1] {band}: {cfg_item.get('verdict', cfg_item.get('error'))}",
              flush=True)
        _save(q)

    prod_after = _prod_snapshot()
    untouched = prod_before == prod_after
    summary = {"bands": feas, "production_ckpt_untouched": untouched,
               "note": "feasibility fits on 2y subset — NOT signal evidence"}
    with open(os.path.join(V12_DIR, "phase1_feasibility.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[P1] complete; production checkpoints untouched: {untouched}")
    if not untouched:
        print("[P1] WARNING: production fingerprint changed — investigate NOW")


BANDS_PARAMS = {"M2_h256L4": 3_788_865, "L1_h512L8": 27_658_305,
                "XL1_h1024L16": 211_136_577, "XL2_h1024L24": 311_906_369}


# ---------------------------------------------------------------- phase 2

# recipe -> recency spec for matured_train_val/recency_weights (None = equal)
RECIPE_SPECS = {
    "Wall_baseline": None,
    "W1y": {"window": 252}, "W2y": {"window": 504}, "W3y": {"window": 756},
    "HL63": {"halflife": 63}, "HL126": {"halflife": 126},
    "HL252": {"halflife": 252}, "HL504": {"halflife": 504},
    "HY3y126": {"window": 756, "halflife": 126},
    "HY3y252": {"window": 756, "halflife": 252},
    "CAL_bands": {"bands": [[126, 1.0], [252, 0.75], [504, 0.5], [756, 0.25]]},
}
P2_EPOCH_CELLS = {
    "E50p5":     dict(max_epochs=50, patience=5),
    "E50p10m10": dict(max_epochs=50, patience=10, min_epochs=10),
    "E100p10":   dict(max_epochs=100, patience=10),
    "E100p20m25": dict(max_epochs=100, patience=20, min_epochs=25),
}
P2_SEEDS = [0, 1, 2]
P2_PANEL_DIR = os.path.join(ROOT, "reports", "transformer_gpu",
                            "v12_big_transformer", "panels")


def _blend_ls(item):
    return (item.get("result", {}).get("blend_band10", {})
            .get("long_short", {}).get("net60", {}))


def _slim(res):
    out = {"mean_val_ic": res.get("mean_val_ic"),
           "total_train_s": res.get("total_train_s"),
           "peak_vram_mb": res.get("peak_vram_mb")}
    for k in ("books", "blend_band10"):
        if k in res:
            out[k] = {m: {kk: res[k][m][kk] for kk in
                          ("rank_ic", "avg_turnover", "net60", "net100",
                           "yearly_net60")} for m in res[k]}
    return out


def _weight_diagnostics(data):
    """ESS + weight share by year per recipe (CPU only, no training)."""
    import dataset_transformer_eod as D
    le_all = data["label_end_rank"][20]
    y = data["targets"]["tgt_rank_20"]
    refit_rank = int(data["date_rank"].max())
    usable = (le_all <= refit_rank) & np.isfinite(y)
    le = le_all[np.nonzero(usable)[0]]
    ref = int(le.max())
    years = pd.to_datetime(np.asarray(data["dates"])[le]).year
    rows = []
    for rec, spec in RECIPE_SPECS.items():
        w = (np.ones(len(le), np.float32) if not spec else
             D.recency_weights(le, ref, spec.get("halflife"),
                               spec.get("window"), spec.get("bands")))
        kept = w > 1e-4
        wk = w[kept]
        ess = float(wk.sum() ** 2 / (wk ** 2).sum()) if wk.sum() else 0.0
        share = pd.Series(wk, index=years[kept]).groupby(level=0).sum()
        share = (share / share.sum()).round(4).to_dict()
        rows.append({"recipe": rec, "n_total": int(len(le)),
                     "n_kept": int(kept.sum()),
                     "effective_sample_size": round(ess, 0),
                     "weight_share_by_year": json.dumps(share)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V12_DIR, "sample_weight_diagnostics.csv"),
              index=False)
    with open(os.path.join(V12_DIR, "train_window_diagnostics.md"), "w",
              encoding="utf-8") as f:
        f.write("# v12 train-window / weight diagnostics (latest refit)\n\n"
                "Weights are relative to the latest matured label; ESS = "
                "(Σw)²/Σw². Equal-weight all-history is the production "
                "baseline.\n\n" + df.to_string(index=False) + "\n")
    print(f"[P2] weight diagnostics -> sample_weight_diagnostics.csv")


def _run_p2_item(item, data, Xg, preset, recency, label):
    import train_transformer_eod as T
    from gpu_research_scheduler import _eval_books, _eval_blend
    t0 = time.time()
    panel, info = T.walkforward(data, Xg, target="tgt_rank_20", horizon=20,
                                preset=preset, oos_start="2023-01-01",
                                refit_every=126, seeds=P2_SEEDS,
                                recency=recency, log_prefix=f"[{label}] ")
    os.makedirs(P2_PANEL_DIR, exist_ok=True)
    panel.to_csv(os.path.join(P2_PANEL_DIR, f"V12_{item['id']}.csv.gz"),
                 index=False, compression="gzip")
    res = {"mean_val_ic": info["mean_val_ic"],
           "total_train_s": info["total_s"],
           "peak_vram_mb": info["peak_vram_mb"],
           "books": _eval_books(panel, 20),
           "blend_band10": _eval_blend(panel, 20)}
    item["result"] = _slim(res)
    item["fits_meta"] = [{k: f.get(k) for k in
                          ("refit_date", "seed", "val_ic", "epochs")}
                         for f in info["fits"]]
    if any("history" in f for f in info["fits"]):
        item["_histories"] = [{"refit_date": f["refit_date"],
                               "seed": f["seed"], "history": f["history"]}
                              for f in info["fits"] if "history" in f]
    bl = item["result"]["blend_band10"]["long_short"]["net60"]
    yr = item["result"]["blend_band10"]["long_short"].get("yearly_net60") or {}
    y22 = ((yr.get("2022") or yr.get(2022)) or {}).get("sharpe")
    item["verdict"] = (f"blend LS net60 {bl['sharpe']} dd {bl['max_dd']} "
                       f"2022 {y22} val_ic {info['mean_val_ic']}")
    item["runtime_s"] = round(time.time() - t0, 1)
    _gpu_row({"ts": time.strftime("%F %T"), "id": item["id"],
              "max_alloc_gb": round(info["peak_vram_mb"] / 1024, 2),
              "batch": 1024, "amp": True, "n_seeds": len(P2_SEEDS),
              "total_train_s": info["total_s"]})


def run_phase2():
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data
    T.require_cuda()
    q = _load()
    data, Xg = get_data("close_only", 60)
    _weight_diagnostics(data)

    # -- recipe screens (Wall_baseline first: it is the comparison bar)
    order = ["Wall_baseline"] + [r for r in P2_RECIPES if r != "Wall_baseline"]
    for rec in order:
        item = next((c for c in q if c["id"] == f"P2_recipe_{rec}"), None)
        if item is None or item.get("status") != "pending":
            continue
        item["status"] = "running"
        _save(q)
        print(f"\n[P2] recipe {rec} ({RECIPE_SPECS[rec]})", flush=True)
        try:
            _run_p2_item(item, data, Xg, "B", RECIPE_SPECS[rec], f"P2:{rec}")
            item["status"] = "done"
            print(f"[P2] {rec}: {item['verdict']}", flush=True)
        except Exception as e:  # noqa: BLE001
            item.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            if "CUDA" in str(e) and not _is_oom(e):
                _save(q)
                print("[P2] CUDA error — stopping (hardware-note rule)")
                return
        _save(q)
    _write_recency_comparison(q)

    # -- epoch cells on Wall_baseline + top-2 recipes that beat/match it
    base = next(c for c in q if c["id"] == "P2_recipe_Wall_baseline")
    if base.get("status") != "done":
        print("[P2] baseline recipe not done — skipping epoch cells")
        return
    base_sh = _blend_ls(base).get("sharpe")
    winners = sorted([c for c in q if c["phase"] == "P2"
                      and c.get("status") == "done" and c is not base
                      and c.get("recipe") in RECIPE_SPECS
                      and (_blend_ls(c).get("sharpe") or -9) >= base_sh],
                     key=lambda c: _blend_ls(c)["sharpe"], reverse=True)[:2]
    chosen = ["Wall_baseline"] + [c["recipe"] for c in winners]
    print(f"\n[P2] epoch cells on: {chosen} (bar {base_sh})", flush=True)
    for rec in chosen:
        for cell, ck in P2_EPOCH_CELLS.items():
            iid = f"P2E_{rec}_{cell}"
            if not any(c["id"] == iid for c in q):
                q.append({"id": iid, "phase": "P2", "recipe": rec,
                          "cell": cell, "status": "pending"})
    _save(q)
    for item in [c for c in q if c["id"].startswith("P2E_")
                 and c.get("status") == "pending"]:
        rec, cell = item["recipe"], item["cell"]
        name = f"__v12_{item['id']}"
        T.PRESETS[name] = dict(T.PRESETS["B"], **P2_EPOCH_CELLS[cell],
                               keep_history=True)
        item["status"] = "running"
        _save(q)
        print(f"\n[P2E] {rec} × {cell}", flush=True)
        try:
            _run_p2_item(item, data, Xg, name, RECIPE_SPECS[rec],
                         f"P2E:{rec}:{cell}")
            item["status"] = "done"
            print(f"[P2E] {item['id']}: {item['verdict']}", flush=True)
        except Exception as e:  # noqa: BLE001
            item.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            if "CUDA" in str(e) and not _is_oom(e):
                _save(q)
                print("[P2E] CUDA error — stopping")
                return
        _save(q)
    _write_epoch_reports(q)
    print("\n[P2] phase 2 complete", flush=True)


def _write_recency_comparison(q):
    from run_queue_v11 import _quintile_overlap
    base = next((c for c in q if c["id"] == "P2_recipe_Wall_baseline"), None)
    bp = os.path.join(P2_PANEL_DIR, "V12_P2_recipe_Wall_baseline.csv.gz")
    base_panel = (pd.read_csv(bp, dtype={"stock": str}, parse_dates=["date"])
                  if os.path.exists(bp) else None)
    rows = []
    for c in q:
        if c.get("recipe") not in RECIPE_SPECS or "cell" in c:
            continue
        bl = _blend_ls(c)
        lo = (c.get("result", {}).get("blend_band10", {})
              .get("long_only", {}).get("net60", {}))
        tf = (c.get("result", {}).get("books", {})
              .get("long_short", {}).get("net60", {}))
        yr = (c.get("result", {}).get("blend_band10", {})
              .get("long_short", {}).get("yearly_net60")) or {}
        row = {"recipe": c["recipe"], "status": c.get("status"),
               "val_ic": c.get("result", {}).get("mean_val_ic"),
               "blend_LS_net60": bl.get("sharpe"),
               "blend_LS_dd": bl.get("max_dd"),
               "blend_LO_net60": lo.get("sharpe"),
               "tf_LS_net60": tf.get("sharpe"),
               "blend_2022": ((yr.get("2022") or yr.get(2022)) or {}).get("sharpe"),
               "turnover": (c.get("result", {}).get("blend_band10", {})
                            .get("long_short", {}).get("avg_turnover")),
               "train_s": c.get("result", {}).get("total_train_s")}
        pp = os.path.join(P2_PANEL_DIR, f"V12_{c['id']}.csv.gz")
        if base_panel is not None and c["recipe"] != "Wall_baseline" \
                and os.path.exists(pp):
            p = pd.read_csv(pp, dtype={"stock": str}, parse_dates=["date"])
            row["q5_overlap_vs_baseline"] = _quintile_overlap(base_panel, p)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V12_DIR, "recency_window_comparison.csv"),
              index=False)
    md = ["# v12 recency / training-window comparison (3 seeds, CH window)",
          "", "Bar = Wall_baseline (equal-weight all-history, the production "
          "recipe). Pre-registered reject rules: val-IC-only wins, worse "
          "book/bear/turnover, unstable books (plan §5).", "",
          df.to_string(index=False), ""]
    with open(os.path.join(V12_DIR, "recency_window_comparison.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("[P2] -> recency_window_comparison.{csv,md}")


# ---------------------------------------------------------------- phase 3

P3_OVERRIDES = {
    "M2_h256L4":  dict(hidden=256, ff=1024, heads=8, trans_layers=4),
    "M3_h256L6":  dict(hidden=256, ff=1024, heads=8, trans_layers=6),
    "L1_h512L8":  dict(hidden=512, ff=2048, heads=8, trans_layers=8),
    "L3_h768L8":  dict(hidden=768, ff=3072, heads=12, trans_layers=8),
}
XL2_SETTINGS = dict(micro_batch=64, accum_steps=16, grad_ckpt=True,
                    epochs=8, max_seconds=4.2 * 3600)  # 5h budget w/ margin
XL_REFIT = "2023-01-01"   # single-refit point for the reduced protocol


def _oos_panel_from_net(net, data, Xg, idx, batch=256):
    """Score OOS samples with one net; panel matches walkforward's schema."""
    import torch
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = torch.as_tensor(np.asarray(idx[i:i + batch]), device=Xg.device)
            with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                outs.append(_forward_big(net, Xg[b], False).float().cpu().numpy())
    sc = np.concatenate(outs)
    dr = np.asarray(data["date_rank"])[idx]
    stocks = np.asarray(data["stocks"])[np.asarray(data["stock_idx"])[idx]]
    t = data["targets"]
    return pd.DataFrame({
        "date": np.asarray(data["dates"])[dr], "stock": stocks, "score": sc,
        "score_std": 0.0, "target": t["tgt_rank_20"][idx],
        "fwd_h": t["fwd_20"][idx], "fwd_20": t["fwd_20"][idx]})


def run_xl2_reduced(item, data, Xg, q):
    """XL2 (1.19 GB) single-refit protocol + matched S-band companion.
    Pre-registered: both train at the 2023-01-01 refit, fixed epoch budget,
    best-by-val state, then score the whole 2023-> OOS block. This is an
    EXPLORATORY read — not decision-grade walkforward evidence (plan §1)."""
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import _eval_books, _eval_blend
    t0 = time.time()
    r0 = int(np.searchsorted(np.asarray(data["dates"]),
                             np.datetime64(pd.Timestamp(XL_REFIT))))
    yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]),
                         device=Xg.device)
    tr, va, _ = T.matured_train_val(data, "tgt_rank_20", r0, 20, recency=None)
    va_dranks = np.asarray(data["date_rank"])[va]
    oos_idx = np.nonzero(np.asarray(data["date_rank"]) > r0)[0]
    oos_idx = oos_idx[np.isfinite(data["targets"]["fwd_20"][oos_idx])]
    print(f"[P3:XL2] single refit {XL_REFIT}: {len(tr):,} train, "
          f"{len(oos_idx):,} OOS samples", flush=True)

    results = {}
    for tag, cfg, kw in (
            ("S_ref", _band_cfg("S_baseline") if "S_baseline" in BANDS
             else dict(_band_cfg("M2_h256L4"), hidden=64, ff=128, heads=4,
                       trans_layers=2),
             dict(micro_batch=1024, accum_steps=1, grad_ckpt=False,
                  epochs=XL2_SETTINGS["epochs"])),
            ("XL2", _band_cfg("XL2_h1024L24"),
             {k: XL2_SETTINGS[k] for k in
              ("micro_batch", "accum_steps", "grad_ckpt", "epochs",
               "max_seconds")})):
        print(f"[P3:XL2] training {tag} ({cfg['hidden']}h/"
              f"{cfg['trans_layers']}L, fixed {kw['epochs']} epochs)",
              flush=True)
        net, vic, curves = fit_big(Xg, yg, tr, va, va_dranks, cfg, seed=0,
                                   **kw)
        panel = _oos_panel_from_net(net, data, Xg, oos_idx,
                                    batch=min(512, kw["micro_batch"] * 4))
        os.makedirs(P2_PANEL_DIR, exist_ok=True)
        panel.to_csv(os.path.join(P2_PANEL_DIR, f"V12_P3_XL2red_{tag}.csv.gz"),
                     index=False, compression="gzip")
        if tag == "XL2":
            ck_p = os.path.join(CKPT_DIR, "xl2_refit2023_seed0.pt")
            torch.save({"model_state_dict": net.state_dict(), "cfg": cfg,
                        "feature_set": "close_only", "target": "tgt_rank_20",
                        "horizon": 20, "seed": 0, "val_ic": vic,
                        "refit": XL_REFIT}, ck_p)
            print(f"[P3:XL2] saved {os.path.getsize(ck_p)/2**20:.0f}MB "
                  "checkpoint", flush=True)
        del net
        torch.cuda.empty_cache()
        results[tag] = {
            "val_ic": round(vic, 5), "curves": curves,
            "books": _eval_books(panel, 20),
            "blend_band10": _eval_blend(panel, 20)}
        bl = results[tag]["blend_band10"]["long_short"]["net60"]
        print(f"[P3:XL2] {tag}: single-refit blend LS net60 {bl['sharpe']} "
              f"dd {bl['max_dd']} val_ic {vic:+.4f}", flush=True)

    item["result"] = {t: {"val_ic": r["val_ic"],
                          "blend_band10": {m: {k: r["blend_band10"][m][k]
                                               for k in ("net60", "net100",
                                                         "avg_turnover")}
                                           for m in r["blend_band10"]},
                          "curves": r["curves"]}
                      for t, r in results.items()}
    xb = results["XL2"]["blend_band10"]["long_short"]["net60"]
    sb = results["S_ref"]["blend_band10"]["long_short"]["net60"]
    item["verdict"] = (f"single-refit blend: XL2 {xb['sharpe']} vs S_ref "
                       f"{sb['sharpe']} (dd {xb['max_dd']} vs {sb['max_dd']})"
                       " — reduced protocol, exploratory only")
    item["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(V12_DIR, "phase3_xl2_reduced.json"), "w",
              encoding="utf-8") as f:
        json.dump(item["result"], f, indent=2, default=str)


def run_phase3():
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data
    T.require_cuda()
    q = _load()
    data, Xg = get_data("close_only", 60)
    for band in ("M2_h256L4", "M3_h256L6", "L1_h512L8", "L3_h768L8"):
        item = next((c for c in q if c["id"] == f"P3_model_{band}"), None)
        if item is None or item.get("status") != "pending":
            continue
        item["status"] = "running"
        _save(q)
        name = f"__v12_P3_{band}"
        T.PRESETS[name] = dict(T.PRESETS["B"], **P3_OVERRIDES[band])
        print(f"\n[P3] model screen {band}", flush=True)
        try:
            _run_p2_item(item, data, Xg, name, None, f"P3:{band}")
            item["status"] = "done"
            print(f"[P3] {band}: {item['verdict']}", flush=True)
        except Exception as e:  # noqa: BLE001
            item.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            if "CUDA" in str(e) and not _is_oom(e):
                _save(q)
                print("[P3] CUDA error — stopping (hardware-note rule)")
                return
        _save(q)
    xl2 = next((c for c in q if c["id"] == "P3_model_XL2_h1024L24"), None)
    if xl2 is not None and xl2.get("status") == "pending":
        xl2["status"] = "running"
        _save(q)
        try:
            run_xl2_reduced(xl2, data, Xg, q)
            xl2["status"] = "done"
        except Exception as e:  # noqa: BLE001
            xl2.update(status="failed", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        _save(q)
    print("\n[P3] phase 3 batch complete", flush=True)


def _write_epoch_reports(q):
    rows = []
    for c in q:
        if not c["id"].startswith("P2E_"):
            continue
        for h in c.get("_histories", []):
            for e in h["history"]:
                rows.append({"item": c["id"], "recipe": c["recipe"],
                             "cell": c["cell"], "refit_date": h["refit_date"],
                             "seed": h["seed"], **e})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(V12_DIR, "epoch_curves.csv"),
                                  index=False)
    md = ["# v12 epoch/training-depth report", "",
          "| item | blend LS net60 | dd | val_ic | mean epochs run | "
          "mean val-IC-peak epoch |", "|---|---|---|---|---|---|"]
    for c in q:
        if not (c["id"].startswith("P2E_") or c.get("recipe")):
            continue
        if c.get("status") != "done" or "cell" not in c and \
                c.get("recipe") != "Wall_baseline":
            continue
        bl = _blend_ls(c)
        eps = [f["epochs"] for f in c.get("fits_meta", [])]
        peaks = []
        for h in c.get("_histories", []):
            vics = [e["val_ic"] for e in h["history"]]
            if vics:
                peaks.append(int(np.argmax(vics)))
        md.append(f"| {c['id']} | {bl.get('sharpe')} | {bl.get('max_dd')} | "
                  f"{c.get('result', {}).get('mean_val_ic')} | "
                  f"{round(np.mean(eps), 1) if eps else ''} | "
                  f"{round(np.mean(peaks), 1) if peaks else 'n/a (25-ep run)'} |")
    md += ["", "Baseline row = P2_recipe_Wall_baseline (max_epochs 25 / "
           "patience 3, no per-epoch curves). A cell is only interesting if "
           "it beats the baseline at BOOK level — longer training is never "
           "evidence by itself (plan §5)."]
    with open(os.path.join(V12_DIR, "epoch_depth_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("[P2] -> epoch_curves.csv, epoch_depth_report.md")


# --------------------------------------------------------------- commands

def dry_run():
    q = _load()
    for c in q:
        extra = ""
        if c["phase"] == "P1":
            extra = (f" micro={c['micro_batch']} accum={c['accum_steps']} "
                     f"ckpt_est={BANDS_PARAMS.get(c.get('band'), 0) * 4 / 2**20:.0f}MB")
        print(f"{c['id']:32s} {c['phase']}  {c.get('status'):8s}{extra}")
    print("\n[dry-run] P1 runs 2-epoch feasibility fits on a 2y subset; "
          "est ~5-50 min/band, ~1-1.5h total. Nothing executed.")


def run_phase1_single(band):
    """Quick single-band feasibility check (queue untouched)."""
    import torch
    import train_transformer_eod as T
    from gpu_research_scheduler import get_data
    T.require_cuda()
    os.makedirs(CKPT_DIR, exist_ok=True)
    data, Xg = get_data("close_only", 60)
    yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]),
                         device=Xg.device)
    refit_rank = int(data["date_rank"].max())
    tr, va, _ = T.matured_train_val(data, "tgt_rank_20", refit_rank, 20,
                                    recency=None)
    cutoff = refit_rank - SMOKE_WINDOW_DAYS
    tr_sub = np.asarray(tr)[np.asarray(data["date_rank"])[tr] >= cutoff]
    s = P1_BANDS[band]
    net, vic, curves = fit_big(Xg, yg, tr_sub, va,
                               np.asarray(data["date_rank"])[va],
                               _band_cfg(band), 0, s["micro_batch"],
                               s["accum_steps"], s["grad_ckpt"], SMOKE_EPOCHS)
    print(f"[smoke] {band} val_ic {vic:+.4f} epochs {[c['epoch_s'] for c in curves]}s")


def report():
    q = _load()
    rows = [{"id": c["id"], "phase": c["phase"], "status": c.get("status"),
             "verdict": c.get("verdict", ""), "error": c.get("error", "")}
            for c in q]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V12_DIR, "queue_v12_results.csv"), index=False)
    with open(os.path.join(V12_DIR, "queue_v12_results.md"), "w",
              encoding="utf-8") as f:
        f.write("# Queue v12 results\n\n" + df.to_string(index=False) + "\n")
    print(df.to_string(index=False))


def status():
    for c in _load():
        print(f"{c['id']:32s} {c['phase']}  {c.get('status', ''):8s} "
              f"{c.get('verdict', c.get('hold_reason', ''))[:70]}")


def unlock(phase):
    q = _load()
    n = 0
    for c in q:
        if c["phase"] == f"P{phase}" and c.get("status") == "held":
            c["status"] = "pending"
            c.pop("hold_reason", None)
            n += 1
    _save(q)
    print(f"[unlock] {n} P{phase} item(s) now pending "
          "(execution path must exist before `run`)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["inspect", "estimate", "materialize",
                                        "dry-run", "smoke", "run", "run-band",
                                        "unlock", "report", "status"])
    ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--band", default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
    if a.command == "run-band":
        sys.exit(run_band(a.band, a.micro_batch, a.accum_steps, a.grad_ckpt))
    if a.command == "inspect":
        print(open(os.path.join(V12_DIR, "current_system_capacity_report.md"),
                   encoding="utf-8").read())
    elif a.command == "estimate":
        os.system(f'"{sys.executable}" '
                  f'"{os.path.join(ROOT, "research", "model_capacity_estimator.py")}"')
    elif a.command == "materialize":
        materialize(a.force)
    elif a.command == "dry-run":
        dry_run()
    elif a.command == "smoke":
        run_phase1_single("M2_h256L4")
    elif a.command == "run" and a.phase == 0:
        run_phase0()
    elif a.command == "run" and a.phase == 1:
        run_phase1()
    elif a.command == "run" and a.phase == 2:
        run_phase2()
    elif a.command == "run" and a.phase == 3:
        run_phase3()
    elif a.command == "run":
        sys.exit(f"phase {a.phase} execution path lands with its approval "
                 "(plan §2); nothing run")
    elif a.command == "unlock":
        unlock(a.phase)
    elif a.command == "report":
        report()
    else:
        status()
