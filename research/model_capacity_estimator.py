"""v12 research-only model capacity / checkpoint-size / VRAM estimator.

Instantiates the REAL `model.LSTM_CondTransformer` on CPU for exact
parameter counts (no guessing), then estimates memory and runtime
feasibility on the current GPU analytically. Nothing here trains, touches
production checkpoints, or writes outside the v12 report directory.

Usage:
  python research/model_capacity_estimator.py            # all bands -> csv/md
  python research/model_capacity_estimator.py --cfg hidden=1024,trans_layers=16,heads=16,ff=4096,seq_len=60
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import torch  # noqa: E402

from model import LSTM_CondTransformer  # noqa: E402

OUT_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v12_big_transformer")

INPUT_DIM = 10          # close_only feature set
N_SAMPLES = 276_571     # current dataset rows (seq60, 2015->2026-07)
GPU_VRAM_GB = 16.0      # RTX 4060 Ti
# sustained fp16 throughput assumption for runtime estimates (measured range
# on this card for transformer-ish workloads; deliberately conservative)
EFF_TFLOPS = (6.0, 12.0)

# candidate bands (v12 plan §2). All use seq60 / close_only unless noted.
BANDS = {
    "S_baseline":  dict(hidden=64,   trans_layers=2,  heads=4,  ff=128,  seq_len=60),
    "M1_h128L4":   dict(hidden=128,  trans_layers=4,  heads=8,  ff=512,  seq_len=60),
    "M2_h256L4":   dict(hidden=256,  trans_layers=4,  heads=8,  ff=1024, seq_len=60),
    "M3_h256L6":   dict(hidden=256,  trans_layers=6,  heads=8,  ff=1024, seq_len=60),
    "L1_h512L8":   dict(hidden=512,  trans_layers=8,  heads=8,  ff=2048, seq_len=60),
    "L2_h512L12":  dict(hidden=512,  trans_layers=12, heads=8,  ff=2048, seq_len=60),
    "L3_h768L8":   dict(hidden=768,  trans_layers=8,  heads=12, ff=3072, seq_len=60),
    "XL1_h1024L16": dict(hidden=1024, trans_layers=16, heads=16, ff=4096, seq_len=60),
    "XL2_h1024L24": dict(hidden=1024, trans_layers=24, heads=16, ff=4096, seq_len=60),
    "XL3_h768L24": dict(hidden=768,  trans_layers=24, heads=12, ff=3072, seq_len=60),
    "XL4_h1536L12": dict(hidden=1536, trans_layers=12, heads=16, ff=6144, seq_len=60),
}


def build(cfg):
    return LSTM_CondTransformer(
        input_dim=INPUT_DIM, lstm_hidden=cfg["hidden"], lstm_layers=1,
        trans_hidden=cfg["hidden"], trans_heads=cfg["heads"],
        trans_layers=cfg["trans_layers"], trans_ff=cfg["ff"],
        dropout=0.2, seq_len=cfg["seq_len"])


def module_breakdown(net):
    groups = {"lstm": 0, "input_proj": 0, "pos_emb": 0, "cross_attn": 0,
              "encoder": 0, "head": 0}
    for name, p in net.named_parameters():
        n = p.numel()
        if name.startswith("lstm."):
            groups["lstm"] += n
        elif name.startswith("trans_input_proj"):
            groups["input_proj"] += n
        elif name.startswith("pos_emb"):
            groups["pos_emb"] += n
        elif name.startswith(("cross_attn", "cross_kv_proj")):
            groups["cross_attn"] += n
        elif name.startswith("transformer."):
            groups["encoder"] += n
        elif name.startswith("fc."):
            groups["head"] += n
    return groups


def estimate(cfg, micro_batch=256):
    t0 = time.time()
    net = build(cfg)
    params = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    groups = module_breakdown(net)
    del net

    fp32_gb = params * 4 / 2**30
    fp16_gb = params * 2 / 2**30
    # AdamW: exp_avg + exp_avg_sq in fp32 (+ master weights already fp32)
    opt_gb = params * 8 / 2**30
    grad_gb = params * 4 / 2**30  # grads kept fp32 by GradScaler.unscale_
    # rough activation memory per micro-batch, AMP fp16 (2 bytes), per layer
    # ~ (attn qkv/out + 2 LN + 2 ff tensors) ≈ 6*D + 2*FF floats per token
    B, S, D, F, L = micro_batch, cfg["seq_len"], cfg["hidden"], cfg["ff"], cfg["trans_layers"]
    act_gb = B * S * (L * (6 * D + 2 * F) + 12 * D + 8 * INPUT_DIM) * 2 / 2**30
    train_gb = fp32_gb + opt_gb + grad_gb + act_gb + 0.9  # +cuda ctx/data margin
    infer_gb = fp16_gb + B * S * (6 * D + 2 * F) * 2 / 2**30 + 0.9

    # training FLOPs/epoch ~= 6 * params * tokens (fwd 2P, bwd 4P per token)
    tokens = N_SAMPLES * S
    flops_epoch = 6.0 * params * tokens
    ep_min_lo = flops_epoch / (EFF_TFLOPS[1] * 1e12) / 60
    ep_min_hi = flops_epoch / (EFF_TFLOPS[0] * 1e12) / 60

    return {
        "config": json.dumps(cfg), "params": params, "trainable": trainable,
        **{f"p_{k}": v for k, v in groups.items()},
        "ckpt_fp32_mb": round(fp32_gb * 1024, 1),
        "ckpt_fp16_mb": round(fp16_gb * 1024, 1),
        "optimizer_gb": round(opt_gb, 2), "grads_gb": round(grad_gb, 2),
        "act_gb_at_micro256": round(act_gb, 2),
        "train_vram_gb_est": round(train_gb, 2),
        "infer_vram_gb_est": round(infer_gb, 2),
        "fits_16gb_train": bool(train_gb < GPU_VRAM_GB - 2),
        "epoch_min_est": f"{ep_min_lo:.0f}-{ep_min_hi:.0f}",
        "estimate_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None,
                    help="single config: hidden=..,trans_layers=..,heads=..,ff=..,seq_len=..")
    ap.add_argument("--micro-batch", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if a.cfg:
        cfg = {k: int(v) for k, v in
               (kv.split("=") for kv in a.cfg.split(","))}
        print(json.dumps(estimate(cfg, a.micro_batch), indent=2))
        return

    import pandas as pd
    rows = []
    for name, cfg in BANDS.items():
        r = {"band": name, **estimate(cfg, a.micro_batch)}
        rows.append(r)
        print(f"{name:14s} params {r['params']:>12,}  ckpt {r['ckpt_fp32_mb']:>8.1f} MB"
              f"  trainVRAM~{r['train_vram_gb_est']:>5.1f} GB"
              f"  fits16GB={r['fits_16gb_train']}  epoch~{r['epoch_min_est']}min")
    df = pd.DataFrame(rows)
    csv_p = os.path.join(OUT_DIR, "model_capacity_estimates.csv")
    df.to_csv(csv_p, index=False)

    md = ["# v12 model capacity estimates", "",
          f"Exact parameter counts from instantiating `model.LSTM_CondTransformer` "
          f"(input_dim={INPUT_DIM}, close_only). Memory/runtime columns are "
          f"analytic estimates (AdamW fp32 states, AMP fp16 activations at "
          f"micro-batch {a.micro_batch}, {EFF_TFLOPS[0]}-{EFF_TFLOPS[1]} "
          "effective TFLOPS on the RTX 4060 Ti 16 GB); the v12 Phase-1 "
          "feasibility smoke measures them for real before any long run.", "",
          "| band | params | ckpt fp32 | ckpt fp16 | opt (GB) | act (GB) | train VRAM est | fits 16GB | min/epoch est |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['band']} | {r['params']:,} | {r['ckpt_fp32_mb']} MB | "
                  f"{r['ckpt_fp16_mb']} MB | {r['optimizer_gb']} | "
                  f"{r['act_gb_at_micro256']} | {r['train_vram_gb_est']} GB | "
                  f"{'yes' if r['fits_16gb_train'] else 'NO'} | {r['epoch_min_est']} |")
    md += ["", "1 GiB fp32 checkpoint = 268,435,456 parameters (2^30 / 4).",
           "", "Module split of the baseline (exact): "
           + ", ".join(f"{k} {v:,}" for k, v in
                       module_breakdown(build(BANDS["S_baseline"])).items())]
    with open(os.path.join(OUT_DIR, "model_capacity_estimates.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n-> {csv_p} and .md")


if __name__ == "__main__":
    main()
