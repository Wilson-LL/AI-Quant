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


# ---------------------------------------------------------------------------
# Queue v10 flag-gated extensions (default OFF — absent from PRESETS A/B/C).
# model.py is untouched: wrappers replicate its forward minus the head to
# reach the pre-head embedding. Activated only via preset keys
# cs_attn / mt_aux / aug_noise / aug_datedrop; with none set, fit_one and
# walkforward take the byte-identical champion path (verified by anchor).
# ---------------------------------------------------------------------------
def _embed(base, x):
    """LSTM_CondTransformer forward, minus the fc head -> (B, trans_hidden)."""
    lstm_out, _ = base.lstm(x)
    h = base.trans_input_proj(lstm_out) + base.pos_emb
    kv = base.cross_kv_proj(x)
    attn_out, _ = base.cross_attn(query=h, key=kv, value=kv)
    h = h + attn_out
    h = base.transformer(h)
    return h[:, -1, :]


class CSAttnNet(torch.nn.Module):
    """v10-A1: one cross-sectional attention block over the names of a single
    date, applied to the pre-head embeddings, residual + LayerNorm, then the
    base head. Callers MUST pass same-date batches (train: date-grouped
    chunks; inference: predict_idx_cs)."""

    def __init__(self, base, hidden, heads):
        super().__init__()
        self.base = base
        self.cs_attn = torch.nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cs_norm = torch.nn.LayerNorm(hidden)

    def forward(self, x):
        e = _embed(self.base, x)                      # (B, H), B = one date
        a, _ = self.cs_attn(e.unsqueeze(0), e.unsqueeze(0), e.unsqueeze(0))
        e = self.cs_norm(e + a.squeeze(0))
        return self.base.fc(e).squeeze(-1)


class MTNet(torch.nn.Module):
    """v10-B1: shared trunk + aux heads for tgt_rank_5 / tgt_rank_10.
    Default forward returns the main head only, so prediction/inference
    paths are unchanged."""

    def __init__(self, base, hidden, dropout):
        super().__init__()
        self.base = base

        def head():
            return torch.nn.Sequential(
                torch.nn.Linear(hidden, 32), torch.nn.ReLU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(32, 1))
        self.aux5, self.aux10 = head(), head()

    def forward(self, x, aux=False):
        e = _embed(self.base, x)
        main = self.base.fc(e).squeeze(-1)
        if aux:
            return main, self.aux5(e).squeeze(-1), self.aux10(e).squeeze(-1)
        return main


class QNet(torch.nn.Module):
    """v10b-B2: quantile head (q10/q50/q90, pinball loss). Default forward
    returns q50, so every prediction/inference path is unchanged."""

    def __init__(self, base, hidden, dropout):
        super().__init__()
        self.base = base
        self.qhead = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(32, 3))

    def forward(self, x, quantiles=False):
        q = self.qhead(_embed(self.base, x))
        return q if quantiles else q[:, 1]


def _pinball(q, y, taus=(0.1, 0.5, 0.9)):
    diff = y.unsqueeze(1) - q
    t = torch.as_tensor(taus, device=q.device, dtype=q.dtype).unsqueeze(0)
    return torch.maximum(t * diff, (t - 1) * diff).mean()


class TCNNet(torch.nn.Module):
    """v10b-A3: causal dilated temporal-convolution baseline (~matched params
    to preset B). Anchor experiment: is attention earning its complexity?"""

    def __init__(self, input_dim, hidden, dropout, blocks=4, k=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        ch = input_dim
        for i in range(blocks):
            self.blocks.append(torch.nn.Sequential(
                torch.nn.Conv1d(ch, hidden, k, dilation=2 ** i,
                                padding=(k - 1) * 2 ** i),
                torch.nn.ReLU(), torch.nn.Dropout(dropout)))
            ch = hidden
        self.pad = [(k - 1) * 2 ** i for i in range(blocks)]
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(32, 1))

    def forward(self, x):
        h = x.transpose(1, 2)                       # (B, feat, seq)
        for blk, p in zip(self.blocks, self.pad):
            out = blk(h)[..., :h.shape[-1]]         # causal: trim right pad
            h = out + h if out.shape == h.shape else out
        return self.fc(h[..., -1]).squeeze(-1)


@torch.no_grad()
def predict_idx_cs(net, Xg, idx, dranks):
    """Date-grouped prediction for CSAttnNet: attention must see exactly the
    names of one date. Returns scores aligned with idx order."""
    net.eval()
    idx = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
    dranks = dranks.cpu().numpy() if torch.is_tensor(dranks) else np.asarray(dranks)
    out = torch.empty(len(idx), device=Xg.device)
    order = np.argsort(dranks, kind="stable")
    bounds = np.flatnonzero(np.diff(dranks[order])) + 1
    for g in np.split(order, bounds):
        b = torch.as_tensor(idx[g], device=Xg.device)
        with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
            out[torch.as_tensor(g, device=Xg.device)] = net(Xg[b]).float()
    return out


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
            max_epochs=None, min_epochs=2, verbose=False,
            loss="mse", date_ranks=None, aux_targets=None):
    """Train one model on GPU-resident tensors; early-stop on val rank IC.

    weights: per-sample weights aligned with tr_idx (recency weighting), or None.
    loss: "mse" (default, unchanged path) or "pairwise" — same-date pairwise
    logistic ranking loss; requires date_ranks (per-sample date rank array).
    aux_targets: {horizon: tensor} for mt_aux (v10-B1), else None.
    Returns (net, best_val_ic, info).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # v10/v10b flags (all default OFF; extra RNG consumed only when active)
    cs_mode = bool(cfg.get("cs_attn"))
    mt_mode = bool(cfg.get("mt_aux")) and aux_targets is not None
    q_mode = bool(cfg.get("quantile"))
    aug_sigma = float(cfg.get("aug_noise") or 0.0)
    aug_drop = float(cfg.get("aug_datedrop") or 0.0)
    if cfg.get("tcn"):
        net = TCNNet(Xg.shape[2], cfg["hidden"], cfg["dropout"]).to(Xg.device)
    else:
        net = build_net(Xg.shape[2], cfg).to(Xg.device)
    if warm_state is not None:
        net.load_state_dict(warm_state)
    if cs_mode:
        net = CSAttnNet(net, cfg["hidden"], cfg["heads"]).to(Xg.device)
    elif mt_mode:
        net = MTNet(net, cfg["hidden"], cfg["dropout"]).to(Xg.device)
    elif q_mode:
        net = QNet(net, cfg["hidden"], cfg["dropout"]).to(Xg.device)
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

    # pairwise/listwise modes (and v10 cs_attn) pre-group train indices by date
    date_groups = None
    if loss in ("pairwise", "listwise") or cs_mode:
        assert date_ranks is not None, f"{loss}/cs_attn needs date_ranks"
        drt = np.asarray(date_ranks)[np.asarray(tr_idx)]
        order = np.argsort(drt, kind="stable")
        tr_np = np.asarray(tr_idx)[order]
        bounds = np.flatnonzero(np.diff(drt[order])) + 1
        date_groups = [torch.as_tensor(g, device=Xg.device)
                       for g in np.split(tr_np, bounds) if len(g) >= 10]

    def _pairwise_loss(pred, y):
        dp = pred.unsqueeze(0) - pred.unsqueeze(1)
        dy = y.unsqueeze(0) - y.unsqueeze(1)
        mask = dy.abs() > 0.1
        if not mask.any():
            return pred.sum() * 0.0
        return torch.nn.functional.softplus(-dp * dy.sign())[mask].mean()

    def _listwise_loss(pred, y, tau=0.5):
        # ListNet top-1: CE between target and prediction softmax over the date
        p_tgt = torch.softmax(y.float() / tau, dim=0)
        return -(p_tgt * torch.log_softmax(pred.float(), dim=0)).sum()

    _group_loss = _listwise_loss if loss == "listwise" else _pairwise_loss

    best_ic, best_state, no_improve = -1e9, None, 0
    epochs = max_epochs or cfg["max_epochs"]
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        net.train()
        tot, ns = 0.0, 0
        if date_groups is not None:
            gperm = np.random.permutation(len(date_groups))
            i = 0
            while i < len(gperm):
                chunk = []
                while i < len(gperm) and sum(len(c) for c in chunk) < batch:
                    chunk.append(date_groups[gperm[i]])
                    i += 1
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                    b_all = torch.cat(chunk)
                    if cs_mode:
                        # per-date forwards (attention within one date only),
                        # plain MSE over the concatenated chunk
                        pred = torch.cat([net(Xg[c]) for c in chunk])
                        loss_t = ((pred - yg[b_all]) ** 2).mean()
                    else:
                        pred = net(Xg[b_all])
                        parts, off = [], 0
                        for c in chunk:
                            parts.append(_group_loss(pred[off:off + len(c)], yg[c]))
                            off += len(c)
                        loss_t = torch.stack(parts).mean()
                scaler.scale(loss_t).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss_t.item() * len(b_all)
                ns += len(b_all)
        else:
            ep_tr = tr
            if aug_drop > 0:
                # v10-D1: drop a fraction of distinct train DATES this epoch
                dr_tr = np.asarray(date_ranks)[tr.cpu().numpy()]
                uniq = np.unique(dr_tr)
                kept = np.random.choice(uniq, size=int(len(uniq) * (1 - aug_drop)),
                                        replace=False)
                mask = np.isin(dr_tr, kept)
                ep_tr = tr[torch.as_tensor(mask, device=Xg.device)]
            perm = ep_tr[torch.randperm(len(ep_tr), device=Xg.device)]
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=Xg.is_cuda):
                    xb = Xg[b]
                    if aug_sigma > 0:
                        xb = xb + aug_sigma * torch.randn_like(xb)  # v10-D1
                    if mt_mode:
                        m_out, a5, a10 = net(xb, aux=True)  # v10-B1
                        loss_t = (((m_out - yg[b]) ** 2).mean()
                                  + 0.3 * ((a5 - aux_targets[5][b]) ** 2).mean()
                                  + 0.3 * ((a10 - aux_targets[10][b]) ** 2).mean())
                    elif q_mode:
                        loss_t = _pinball(net(xb, quantiles=True), yg[b])  # v10b-B2
                    else:
                        pred = net(xb)
                        per = (pred - yg[b]) ** 2
                        loss_t = (per * w_full[b]).mean() if w_full is not None else per.mean()
                scaler.scale(loss_t).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot += loss_t.item() * len(b)
                ns += len(b)
        vs = (predict_idx_cs(net, Xg, va, va_dates) if cs_mode
              else predict_idx(net, Xg, va)).cpu().numpy()
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
                weight_decay=1e-4, verbose=True, log_prefix="", loss="mse"):
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
        aux_t = None
        if cfg.get("mt_aux"):  # v10-B1: joint aux heads on rank_5 / rank_10
            aux_t = {h: torch.as_tensor(
                np.nan_to_num(np.clip(data["targets"][f"tgt_rank_{h}"], -1, 1)),
                device=Xg.device) for h in (5, 10)}

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
                                     warm_state=warm, max_epochs=max_ep,
                                     loss=loss, date_ranks=data["date_rank"],
                                     aux_targets=aux_t)
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
            pstack = torch.stack([
                predict_idx_cs(n, Xg, in_block, dr[in_block])
                if isinstance(n, CSAttnNet) else predict_idx(n, Xg, in_block)
                for n in nets])
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
        "recency": recency, "seeds": list(seeds), "loss": loss,
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
    # production spec (2026-07-24, A8/A9): 7-seed ensemble — the seed curve
    # saturates at 7 (see RESEARCH_SCOREBOARD.md B4e/B4f)
    seeds = seeds or list(range(7))
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
