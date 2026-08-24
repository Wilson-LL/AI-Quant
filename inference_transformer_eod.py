"""Daily EOD inference + decision book for the LSTM-Transformer system.

Loads the seed ensemble saved by `train_transformer_eod.py --mode daily-retrain`,
scores every stock in the universe at the latest cached date, builds the target
book (hard 10% per-name cap, soft 20% sector cap), and writes the daily outputs:

  reports/transformer_gpu/YYYY-MM-DD_predictions.csv
  reports/transformer_gpu/YYYY-MM-DD_target_book.csv
  reports/transformer_gpu/YYYY-MM-DD_metrics.json
  reports/transformer_gpu/YYYY-MM-DD_report.md

Predictions from features at date t are intended for execution on the NEXT
trading day (t+1) — printed on every book row.

Usage:  python inference_transformer_eod.py [--top-frac 0.2] [--band 0.05]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "research"))

from dataset_transformer_eod import build_dataset  # noqa: E402
from train_transformer_eod import build_net, predict_idx, DEVICE, CKPT_DIR, REPORT_DIR  # noqa: E402
from data import SECTOR_MAP  # noqa: E402
from transformer_portfolio import cap_weights, NAME_CAP  # noqa: E402


def load_ensemble():
    manifest_path = os.path.join(CKPT_DIR, "daily_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "No daily ensemble found — run `python train_transformer_eod.py "
            "--mode daily-retrain` first.")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "daily_seed*.pt")))
    nets, cfg = [], None
    for p in paths:
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        cfg = ck["cfg"]
        net = build_net_from_ck(ck).to(DEVICE)
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        nets.append(net)
    return nets, cfg, manifest


def build_net_from_ck(ck):
    from dataset_transformer_eod import FEATURE_COLS
    input_dim = len(FEATURE_COLS[ck["feature_set"]])
    return build_net(input_dim, ck["cfg"])


def previous_book():
    books = sorted(glob.glob(os.path.join(REPORT_DIR, "*_target_book.csv")))
    if not books:
        return None
    return pd.read_csv(books[-1], dtype={"symbol": str})


def make_decision_book(pred, prev, top_frac, band, horizon, exec_date):
    """pred: DataFrame [stock, score, score_std, sector, vol_20]."""
    pred = pred.sort_values("score", ascending=False).reset_index(drop=True)
    pred["rank"] = np.arange(1, len(pred) + 1)
    n = len(pred)
    k = max(3, round(top_frac * n))
    band_k = max(1, round(band * n))
    # Thin-universe guard (2026-08-24 incident review): reuse the SAME
    # research assumption the validated backtests have always used —
    # transformer_portfolio.backtest_scores(min_names=60) skips thin
    # cross-sections as unrepresentative. Inference had no such guard,
    # so a 34-name partial-publication cross-section reached book
    # construction. Not a new tuned threshold; the documented existing
    # one.
    MIN_UNIVERSE_NAMES = 60
    if n < MIN_UNIVERSE_NAMES:
        raise RuntimeError(
            f"scored universe too thin: {n} names < min_names="
            f"{MIN_UNIVERSE_NAMES} (the research-standard thin-cross-"
            "section floor used by backtest_scores) — likely partial "
            "EOD publication. Inference aborted; no artifacts written. "
            "Re-run refresh_data and retry.")
    # Cap-feasibility guard (defense in depth; unreachable while
    # MIN_UNIVERSE_NAMES=60 keeps k >= 12, kept in case selection
    # parameters ever change): with k names in the book, the hard
    # NAME_CAP is satisfiable only if k * NAME_CAP >= 1. cap_weights'
    # documented infeasible-cap fallback returns EQUAL weights ABOVE the
    # cap and relies on callers to guard. Never silently violate; never
    # weaken the assertion below.
    if k * NAME_CAP < 1.0 - 1e-9:
        raise RuntimeError(
            f"decision-book name cap infeasible: only {n} names scored "
            f"-> book size k={k}, and k*NAME_CAP = {k * NAME_CAP:.2f} < 1 "
            "(likely partial EOD publication). Inference aborted; no "
            "artifacts written. Re-run refresh_data and retry.")

    prev_w = {}
    if prev is not None:
        prev_w = dict(zip(prev["symbol"].astype(str), prev["target_weight"]))

    core = list(pred.head(k)["stock"])
    # no-trade band: incumbents ranked within k+band_k stay held
    held = [s for s in prev_w if prev_w.get(s, 0) > 0]
    incumbents_ok = [s for s in held
                     if s in set(pred.head(k + band_k)["stock"])]
    book_names = list(dict.fromkeys([s for s in core if s not in incumbents_ok][:max(0, k - len(incumbents_ok))] + incumbents_ok))
    if len(book_names) < k:
        extra = [s for s in core if s not in book_names]
        book_names += extra[:k - len(book_names)]

    sub = pred[pred["stock"].isin(book_names)].copy()
    # inverse-vol sizing with hard caps
    v = sub["vol_20"].to_numpy(np.float64)
    v = np.where(np.isfinite(v) & (v > 0), v, np.nanmedian(v) or 1.0)
    w = cap_weights(1.0 / v, [SECTOR_MAP.get(s, "other") for s in sub["stock"]])
    sub["target_weight"] = w
    wmap = dict(zip(sub["stock"], sub["target_weight"]))

    # confidence from seed disagreement (lower std = higher confidence)
    std_terciles = pred["score_std"].quantile([0.33, 0.66]).to_numpy()
    def conf(s):
        return "high" if s <= std_terciles[0] else ("medium" if s <= std_terciles[1] else "low")

    rows = []
    for _, r in pred.iterrows():
        sid = r["stock"]
        tw = float(wmap.get(sid, 0.0))
        pw = float(prev_w.get(sid, 0.0))
        if tw > 0 and pw == 0:
            action = "BUY"
        elif tw > 0 and pw > 0:
            action = "HOLD" if abs(tw - pw) <= 0.02 else ("REDUCE" if tw < pw else "BUY")
        elif tw == 0 and pw > 0:
            action = "SELL"
        elif r["rank"] <= k + band_k:
            action = "WATCH"
        else:
            action = ""
        if not action:
            continue
        rows.append({
            "symbol": sid, "prediction_score": round(float(r["score"]), 5),
            "rank": int(r["rank"]), "action": action,
            "target_weight": round(tw, 5), "previous_weight": round(pw, 5),
            "weight_change": round(tw - pw, 5),
            "sector": r["sector"], "confidence": conf(r["score_std"]),
            "execution_date": exec_date, "holding_horizon_days": horizon,
            "caveats": "survivorship-biased research universe; EOD close-to-close model; "
                       "execute next session; not investment advice",
        })
    book = pd.DataFrame(rows)
    assert book.empty or book["target_weight"].max() <= NAME_CAP + 1e-4, \
        "hard 10% name cap violated"
    assert book.empty or abs(book["target_weight"].sum() - 1.0) < 1e-3, \
        "book weights do not sum to 1"
    return book


def main(top_frac=0.2, band=0.05):
    t0 = time.time()
    nets, cfg, manifest = load_ensemble()
    fs, horizon = manifest["feature_set"], manifest["horizon"]
    data = build_dataset(fs, seq_len=cfg["seq_len"], horizons=(horizon,))
    asof_rank = int(data["date_rank"].max())
    asof = str(data["dates"][asof_rank])[:10]
    idx = np.nonzero(data["date_rank"] == asof_rank)[0]
    if len(idx) == 0:
        raise RuntimeError("no samples at latest date")
    Xg = torch.as_tensor(data["X"][idx], device=DEVICE)
    all_idx = torch.arange(len(idx), device=DEVICE)
    preds = torch.stack([predict_idx(n, Xg, all_idx) for n in nets])
    mean, std = preds.mean(0).cpu().numpy(), preds.std(0).cpu().numpy()

    stocks = np.asarray(data["stocks"])[data["stock_idx"][idx]]
    vol_col = data["feature_cols"].index("vol_20") if "vol_20" in data["feature_cols"] else None
    vol20 = data["X"][idx, -1, vol_col] if vol_col is not None else np.full(len(idx), np.nan)
    pred = pd.DataFrame({
        "stock": stocks, "score": mean, "score_std": std,
        "sector": [SECTOR_MAP.get(s, "other") for s in stocks],
        "vol_20": vol20,
    })
    infer_s = time.time() - t0

    prev = previous_book()
    exec_date = f"next trading day after {asof}"
    book = make_decision_book(pred, prev, top_frac, band, horizon, exec_date)

    os.makedirs(REPORT_DIR, exist_ok=True)
    pred_out = pred.sort_values("score", ascending=False)
    pred_out.to_csv(os.path.join(REPORT_DIR, f"{asof}_predictions.csv"), index=False)
    book.to_csv(os.path.join(REPORT_DIR, f"{asof}_target_book.csv"), index=False)

    metrics = {
        "asof": asof, "n_stocks_scored": int(len(pred)),
        "n_book": int((book["target_weight"] > 0).sum()) if not book.empty else 0,
        "max_name_weight": float(book["target_weight"].max()) if not book.empty else 0,
        "train_manifest": {k: manifest[k] for k in
                           ("asof", "feature_set", "target", "horizon", "recency",
                            "mean_val_ic", "total_s") if k in manifest},
        "inference_s": round(infer_s, 2),
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "seeds": len(nets),
    }
    with open(os.path.join(REPORT_DIR, f"{asof}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    top = book[book["target_weight"] > 0].head(30) if not book.empty else book
    lines = [f"# Daily decision report — {asof}", "",
             f"Model: LSTM-Transformer (cross-attention), feature set `{manifest['feature_set']}`, "
             f"target `{manifest['target']}`, {len(nets)}-seed ensemble, "
             f"val rank IC {manifest.get('mean_val_ic')}",
             f"Universe scored: {len(pred)} stocks · inference {infer_s:.1f}s on "
             f"{metrics['device']}", "",
             f"**Execution: {exec_date}, holding horizon {horizon} trading days.**", "",
             "## Target book (long, inverse-vol, 10% name cap, 20% sector cap)", "",
             "| symbol | sector | score | rank | action | weight | Δweight | conf |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        lines.append(f"| {r['symbol']} | {r['sector']} | {r['prediction_score']:+.4f} | "
                     f"{r['rank']} | {r['action']} | {r['target_weight']:.2%} | "
                     f"{r['weight_change']:+.2%} | {r['confidence']} |")
    sells = book[book["action"].isin(["SELL", "REDUCE"])] if not book.empty else book
    if not sells.empty:
        lines += ["", "## Exits / reductions", ""]
        for _, r in sells.iterrows():
            lines.append(f"- {r['action']} {r['symbol']} ({r['sector']}), "
                         f"weight {r['previous_weight']:.2%} → {r['target_weight']:.2%}")
    lines += ["", "## Caveats",
              "- Research system on a survivorship-biased cached universe; not investment advice.",
              "- Scores are cross-sectional ranks, not return forecasts.",
              "- Data snapshot ends at the as-of date; refresh the cache before live use."]
    with open(os.path.join(REPORT_DIR, f"{asof}_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {asof}_predictions.csv / _target_book.csv / _metrics.json / _report.md "
          f"in {REPORT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--band", type=float, default=0.05)
    a = ap.parse_args()
    main(a.top_frac, a.band)
