"""GPU trainer for the TWSE EOD LSTM-Transformer daily-retrain system.

Reuses the production `LSTM_CondTransformer` (model.py) and the training
technology of train.py (AdamW, gradient clipping, early stopping, checkpoints),
extended per the sprint brief with:

  * AMP mixed precision (10.6x on the RTX 4060 Ti vs fp32)
  * per-sample weighted loss (recency weighting, brief §5)
  * cross-sectional targets with matured-label-only training (brief §4)
  * validation-rank-IC early stopping (never OOS-driven)
  * seed ensembles, walk-forward runner with retrain cadence, warm-start
  * device / time / VRAM logging

Library + CLI. CLI:
  python train_transformer_eod.py --mode smoke
  python train_transformer_eod.py --mode daily-retrain [--feature-set close_only]
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from model import LSTM_CondTransformer
from dataset_transformer_eod import build_dataset, matured_train_val

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints", "transformer_eod")
REPORT_DIR = os.path.join(ROOT, "reports", "transformer_gpu")

PRESETS = {
    # lstm_layers=1 + trans_layers=2 mirrors the production architecture's split
    "A": dict(hidden=64, trans_layers=2, lstm_layers=1, heads=4, ff=128,
              dropout=0.2, seq_len=40, max_epochs=20, patience=3, seeds=3),
    "B": dict(hidden=64, trans_layers=2, lstm_layers=1, heads=4, ff=128,
              dropout=0.2, seq_len=60, max_epochs=25, patience=3, seeds=5),
    "C": dict(hidden=128, trans_layers=2, lstm_layers=1, heads=4, ff=256,
              dropout=0.2, seq_len=60, max_epochs=20, patience=3, seeds=3),
}


def require_cuda(allow_cpu_smoke=False):
    if DEVICE != "cuda" and not allow_cpu_smoke:
        raise RuntimeError("CUDA unavailable — hard rule: no CPU full-scale training. "
                           "Fix the environment first (see RTX4060TI_ENVIRONMENT_CHECK.md).")


def build_net(input_dim, cfg):
    return LSTM_CondTransformer(
        input_dim=input_dim, lstm_hidden=cfg["hidden"], lstm_layers=cfg["lstm_layers"],
        trans_hidden=cfg["hidden"], trans_heads=cfg["heads"],
        trans_layers=cfg["trans_layers"], trans_ff=cfg["ff"],
        dropout=cfg["dropout"], seq_len=cfg["seq_len"])


def param_count(net):
    return sum(p.numel() for p in net.parameters())


def _rank_ic(scores, targets, date_ranks):
    """Mean per-date Spearman IC. Returns (mean_ic, per-date series)."""
    df = pd.DataFrame({"d": date_ranks, "s": scores, "t": targets}).dropna()
    def one(g):
        if len(g) < 5 or g["s"].nunique() < 2 or g["t"].nunique() < 2:
            return np.nan
        return g["s"].rank().corr(g["t"].rank())
    ics = df.groupby("d", group_keys=False).apply(one, include_groups=False).dropna()
    return (float(ics.mean()) if len(ics) else float("nan")), ics


@torch.no_grad()
def predict_idx(net, Xg, idx, batch=8192):
    net.eval()
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(np.asarray(idx), device=Xg.device)
    out = torch.empty(len(idx), device=Xg.device)
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
            out[i:i + len(b)] = net(Xg[b]).float()
    return out


def fit_one(Xg, yg, tr_idx, va_idx, va_dates, cfg, seed, weights=None,
            lr=3e-4, weight_decay=1e-4, batch=1024, warm_state=None,
            max_epochs=None, min_epochs=2, verbose=False):
    """Train one model on GPU-resident tensors; early-stop on val rank IC.

    weights: per-sample weights aligned with tr_idx (recency weighting), or None.
    Returns (net, best_val_ic, info).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_net(Xg.shape[2], cfg).to(Xg.device)
    if warm_state is not None:
        net.load_state_dict(warm_state)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=Xg.is_cuda)

    tr = torch.as_tensor(np.asarray(tr_idx), device=Xg.device)
    va = torch.as_tensor(np.asarray(va_idx), device=Xg.device)
    # full-length weight vector so batches can gather weights by global index
    w_full = None
    if weights is not None:
        w_full = torch.zeros(Xg.shape[0], device=Xg.device)
        w_full[tr] = torch.as_tensor(np.asarray(weights, np.float32), device=Xg.device)
        # normalize so the mean train weight is 1 (keeps lr scale comparable)
        w_full[tr] /= w_full[tr].mean().clamp_min(1e-8)

    best_ic, best_state, no_improve = -1e9, None, 0
    epochs = max_epochs or cfg["max_epochs"]
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), device=Xg.device)]
        tot, ns = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                pred = net(Xg[b])
                per = (pred - yg[b]) ** 2
                loss = (per * w_full[b]).mean() if w_full is not None else per.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(b)
            ns += len(b)
        vs = predict_idx(net, Xg, va).cpu().numpy()
        vic, _ = _rank_ic(vs, yg[va].cpu().numpy(), va_dates)
        hist.append({"epoch": ep, "train_loss": round(tot / max(ns, 1), 5),
                     "val_ic": round(vic, 5)})
        if verbose:
            print(f"  ep{ep} loss {tot/max(ns,1):.4f} val_ic {vic:+.4f}")
        if np.isfinite(vic) and vic > best_ic:
            best_ic, best_state, no_improve = vic, copy.deepcopy(net.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"] and ep + 1 >= min_epochs:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    info = {"epochs_run": len(hist), "train_s": round(time.time() - t0, 1),
            "best_val_ic": round(best_ic, 5), "history": hist,
            "params": param_count(net)}
    return net, best_ic, info


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------
def to_gpu(data):
    Xg = torch.as_tensor(data["X"], device=DEVICE)
    return Xg


def walkforward(data, Xg, target="tgt_rank_20", horizon=20, preset="A",
                oos_start="2023-01-01", refit_every=126, seeds=None,
                recency=None, cadence="refit", warm_epochs=2, lr=3e-4,
                weight_decay=1e-4, verbose=True, log_prefix=""):
    """Walk-forward OOS scores with a given retrain cadence.

    cadence="refit": full retrain (fresh init) every `refit_every` trading days.
    cadence="warm":  first fit full, then warm-start fine-tune (`warm_epochs`
                     epochs, recent data emphasized by `recency`) every
                     `refit_every` days.
    Returns (score_panel DataFrame, run_info dict).
    """
    cfg = PRESETS[preset]
    seeds = seeds if seeds is not None else list(range(cfg["seeds"]))
    dates = data["dates"]
    dr = data["date_rank"]
    oos0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(oos_start))))
    last = int(dr.max())

    rows = []
    fit_infos = []
    warm_states = None
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
    t_start = time.time()

    refit_ranks = list(range(oos0, last + 1, refit_every))
    for ri, r0 in enumerate(refit_ranks):
        refit_rank = r0 - 1          # last observable close before the block
        block_end = min(r0 + refit_every - 1, last)
        try:
            tr, va, w = matured_train_val(data, target, refit_rank, horizon,
                                          recency=recency)
        except ValueError as e:
            if verbose:
                print(f"{log_prefix}[wf] skip refit@{str(dates[r0])[:10]}: {e}")
            continue
        va_dates = dr[va]
        # clip so fat-tailed excess-return targets can't dominate the MSE
        yg = torch.as_tensor(np.nan_to_num(np.clip(data["targets"][target], -1, 1)),
                             device=Xg.device)

        nets = []
        for s in seeds:
            warm = None
            max_ep = None
            if cadence == "warm" and warm_states is not None:
                warm, max_ep = warm_states[s], warm_epochs
            net, vic, info = fit_one(Xg, yg, tr, va, va_dates, cfg, seed=s,
                                     weights=w if recency else None,
                                     lr=lr if warm is None else lr * 0.3,
                                     weight_decay=weight_decay,
                                     warm_state=warm, max_epochs=max_ep)
            nets.append(net)
            fit_infos.append({"refit_date": str(dates[r0])[:10], "seed": s,
                              "val_ic": info["best_val_ic"],
                              "epochs": info["epochs_run"],
                              "train_s": info["train_s"],
                              "n_train": int(len(tr)), "warm": warm is not None})
        if cadence == "warm":
            warm_states = {s: copy.deepcopy(n.state_dict()) for s, n in zip(seeds, nets)}

        # score every sample in the OOS block (ensemble mean)
        in_block = np.nonzero((dr >= r0) & (dr <= block_end))[0]
        if len(in_block):
            pstack = torch.stack([predict_idx(n, Xg, in_block) for n in nets])
            preds = pstack.mean(0).cpu().numpy()
            pstd = (pstack.std(0, unbiased=False) if len(nets) > 1
                    else torch.zeros_like(pstack[0])).cpu().numpy()
            stocks = np.asarray(data["stocks"])[data["stock_idx"][in_block]]
            rows.append(pd.DataFrame({
                "date": dates[dr[in_block]],
                "stock": stocks,
                "score": preds,
                "score_std": pstd,
                "target": data["targets"][target][in_block],
                "fwd_h": data["targets"][f"fwd_{horizon}"][in_block],
                "fwd_20": data["targets"].get("fwd_20", data["targets"][f"fwd_{horizon}"])[in_block],
            }))
        if verbose:
            vics = [fi["val_ic"] for fi in fit_infos[-len(seeds):]]
            print(f"{log_prefix}[wf {ri+1}/{len(refit_ranks)}] {str(dates[r0])[:10]} "
                  f"train {len(tr)} val_ic {np.mean(vics):+.4f} "
                  f"({time.time()-t_start:.0f}s elapsed)")

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    run_info = {
        "target": target, "horizon": horizon, "preset": preset,
        "cadence": cadence, "refit_every": refit_every,
        "recency": recency, "seeds": list(seeds),
        "oos_start": str(dates[oos0])[:10],
        "n_refits": len(refit_ranks),
        "total_s": round(time.time() - t_start, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEVICE == "cuda" else 0,
        "mean_val_ic": round(float(np.mean([f["val_ic"] for f in fit_infos])), 5) if fit_infos else None,
        "fits": fit_infos,
    }
    return panel, run_info


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------
def mode_smoke():
    """G0: tiny end-to-end train + predict on real data, CUDA required."""
    require_cuda()
    data = build_dataset("close_only", seq_len=40, horizons=(20,))
    Xg = to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(data["targets"]["tgt_rank_20"]), device=DEVICE)
    refit_rank = int(data["date_rank"].max())
    tr, va, w = matured_train_val(data, "tgt_rank_20", refit_rank, 20,
                                  recency={"halflife": 126})
    cfg = dict(PRESETS["A"], max_epochs=3, patience=3)
    net, vic, info = fit_one(Xg, yg, tr, va, data["date_rank"][va], cfg, seed=0,
                             weights=w, verbose=True)
    print(json.dumps({"val_ic": vic, "params": info["params"],
                      "train_s": info["train_s"],
                      "peak_vram_mb": round(torch.cuda.max_memory_allocated()/1e6, 1)},
                     indent=2))
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"model_state_dict": net.state_dict(), "cfg": cfg,
                "feature_set": "close_only", "val_ic": vic},
               os.path.join(CKPT_DIR, "smoke.pt"))
    print("smoke checkpoint saved")


def mode_daily_retrain(feature_set="close_only", preset="B", target="tgt_rank_20",
                       horizon=20, recency=None, seeds=None):
    """Production daily retrain: train on all matured labels as of the latest
    cached date, save the seed ensemble + a train log. Feeds inference_transformer_eod.

    Champion defaults per this sprint's G2/G5/G9: close-only features,
    EQUAL-WEIGHT full history (recency weighting hurt OOS), preset B
    (h64/seq60/5 seeds). Pass recency explicitly to override."""
    require_cuda()
    t0 = time.time()
    data = build_dataset(feature_set, seq_len=PRESETS[preset]["seq_len"],
                         horizons=(horizon,))
    Xg = to_gpu(data)
    yg = torch.as_tensor(np.nan_to_num(data["targets"][target]), device=DEVICE)
    refit_rank = int(data["date_rank"].max())
    tr, va, w = matured_train_val(data, target, refit_rank, horizon, recency=recency)
    cfg = PRESETS[preset]
    seeds = seeds or list(range(cfg["seeds"]))
    asof = str(data["dates"][refit_rank])[:10]

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    fit_logs = []
    for s in seeds:
        net, vic, info = fit_one(Xg, yg, tr, va, data["date_rank"][va], cfg,
                                 seed=s, weights=w)
        torch.save({"model_state_dict": net.state_dict(), "cfg": cfg,
                    "feature_set": feature_set, "target": target,
                    "horizon": horizon, "recency": recency, "seed": s,
                    "val_ic": vic, "asof": asof},
                   os.path.join(CKPT_DIR, f"daily_seed{s}.pt"))
        fit_logs.append({"seed": s, "val_ic": vic,
                         "epochs": info["epochs_run"], "train_s": info["train_s"]})
        print(f"[daily] seed {s}: val_ic {vic:+.4f} in {info['train_s']}s")

    log = {
        "asof": asof, "feature_set": feature_set, "preset": preset,
        "target": target, "horizon": horizon, "recency": recency,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "fits": fit_logs,
        "mean_val_ic": round(float(np.mean([f["val_ic"] for f in fit_logs])), 5),
        "total_s": round(time.time() - t0, 1),
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }
    with open(os.path.join(REPORT_DIR, f"{asof}_train_log.md"), "w", encoding="utf-8") as f:
        f.write(f"# Daily retrain log — {asof}\n\n```json\n"
                + json.dumps(log, indent=2) + "\n```\n")
    with open(os.path.join(CKPT_DIR, "daily_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(json.dumps({k: v for k, v in log.items() if k != "fits"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    choices=["smoke", "daily-retrain"])
    ap.add_argument("--feature-set", default="close_only")
    ap.add_argument("--preset", default="B")
    ap.add_argument("--target", default="tgt_rank_20")
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()
    if args.mode == "smoke":
        mode_smoke()
    else:
        mode_daily_retrain(args.feature_set, args.preset, args.target, args.horizon)
