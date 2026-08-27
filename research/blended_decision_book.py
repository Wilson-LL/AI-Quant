"""F2 — blended decision book for the promoted candidate (blend50 + band10).

Combines champion transformer daily predictions with the D1.2 momentum score
(z-avg 50/50), applies top-quintile selection with the 10% no-trade band vs the
previous blended book, equal weight, hard 10% name cap, and emits a full
decision book (Section-9 columns) + markdown summary.

Usage:
  python research/blended_decision_book.py [--asof YYYY-MM-DD]
Inputs:  reports/transformer_gpu/<asof>_predictions.csv (tf scores + score_std)
Outputs: reports/paper_trading/<asof>_blend50_band10_decision_book.{csv,md}
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import SECTOR_MAP  # noqa: E402
from transformer_hybrid import _cache_frames  # noqa: E402
from transformer_portfolio import cap_weights, NAME_CAP, CAP_TOL  # noqa: E402
from paper_trading import _book_from_scores, _prev_book, BAND  # noqa: E402

PT_DIR = os.path.join(ROOT, "reports", "paper_trading")
CAVEAT = ("survivorship-biased research universe; EOD close-to-close model; "
          "execute next session; not investment advice")


def build(asof=None):
    pred_dir = os.path.join(ROOT, "reports", "transformer_gpu")
    if asof is None:
        cands = sorted(f[:10] for f in os.listdir(pred_dir) if f.endswith("_predictions.csv"))
        if not cands:
            sys.exit("no predictions; run inference_transformer_eod.py first")
        asof = cands[-1]
    preds = pd.read_csv(os.path.join(pred_dir, f"{asof}_predictions.csv"), dtype={"stock": str})

    rows = []
    for sid, df in _cache_frames().items():
        df = df[df["date"] <= pd.Timestamp(asof)]
        if len(df) < 132:
            continue
        c = df["close"].to_numpy(np.float64)
        rows.append({"stock": sid, "mom": c[-6] / c[-132] - 1.0})
    mm = preds.merge(pd.DataFrame(rows), on="stock", how="inner")
    mm["z_tf"] = (mm["score"] - mm["score"].mean()) / (mm["score"].std() + 1e-9)
    mm["z_mom"] = (mm["mom"] - mm["mom"].mean()) / (mm["mom"].std() + 1e-9)
    mm["blend"] = 0.5 * mm["z_tf"] + 0.5 * mm["z_mom"]
    # confidence: seed disagreement (lower std = higher confidence), terciles
    if "score_std" in mm:
        q = mm["score_std"].rank(pct=True)
        mm["confidence"] = np.where(q <= 1 / 3, "high", np.where(q <= 2 / 3, "medium", "low"))
    else:
        mm["confidence"] = "n/a"

    prev = _prev_book("blend50_band10", asof)
    prev_w = (prev.set_index("stock")["weight"] if prev is not None
              else pd.Series(dtype=float))
    book = _book_from_scores(mm[["stock", "blend"]].rename(columns={"blend": "score"}),
                             prev_names=list(prev_w.index) if len(prev_w) else None,
                             band=BAND)
    tgt = book.set_index("stock")["weight"]
    assert float(tgt.max()) <= NAME_CAP + CAP_TOL, "name cap violated"

    ranks = mm.set_index("stock")["blend"].rank(ascending=False).astype(int)
    n = len(mm)
    k_watch = int(0.3 * n)
    out = []
    for sid in sorted(set(tgt.index) | set(prev_w.index) | set(ranks.nsmallest(0).index)):
        w1, w0 = float(tgt.get(sid, 0.0)), float(prev_w.get(sid, 0.0))
        if w1 > 0 and w0 == 0:
            action = "BUY"
        elif w1 > 0 and w1 >= w0 - 1e-6:
            action = "HOLD"
        elif w1 > 0:
            action = "REDUCE"
        else:
            action = "SELL"
        out.append((sid, action, w1, w0))
    # WATCH: top-30% blend names not in book and not being sold
    held = {s for s, *_ in out}
    for sid, rk in ranks.items():
        if rk <= k_watch and sid not in held:
            out.append((sid, "WATCH", 0.0, 0.0))

    db = pd.DataFrame(out, columns=["symbol", "action", "target_weight", "previous_weight"])
    db["weight_change"] = db["target_weight"] - db["previous_weight"]
    db["model_score"] = db["symbol"].map(mm.set_index("stock")["blend"]).round(4)
    db["rank"] = db["symbol"].map(ranks)
    db["sector"] = db["symbol"].map(lambda s: SECTOR_MAP.get(s, "other"))
    db["confidence"] = db["symbol"].map(mm.set_index("stock")["confidence"])
    db["intended_execution_date"] = f"next trading day after {asof}"
    db["holding_horizon_days"] = 20
    db["caveats"] = CAVEAT
    db = db.sort_values(["action", "rank"]).reset_index(drop=True)
    db = db[["symbol", "model_score", "rank", "action", "target_weight",
             "previous_weight", "weight_change", "sector", "confidence",
             "intended_execution_date", "holding_horizon_days", "caveats"]]
    for c in ("target_weight", "previous_weight", "weight_change"):
        db[c] = db[c].round(5)

    os.makedirs(PT_DIR, exist_ok=True)

    # Additive full-universe export (decision-support only): the complete
    # blend cross-section that the top-quintile filter below hides. Never
    # feeds back into book construction — the book above is already built.
    uni = mm[["stock", "score", "score_std", "mom", "z_tf", "z_mom",
              "blend", "confidence"]].copy()
    uni.columns = ["symbol", "tf_score", "seed_score_std", "momentum",
                   "z_tf", "z_momentum", "blend_score", "confidence"]
    uni["rank"] = uni["symbol"].map(ranks)
    uni["sector"] = uni["symbol"].map(lambda s: SECTOR_MAP.get(s, "other"))
    uni["signal_date"] = asof
    uni = uni.sort_values("rank").reset_index(drop=True)
    uni.to_csv(os.path.join(
        PT_DIR, f"{asof}_blend50_universe_scores.csv"), index=False)

    csv_p = os.path.join(PT_DIR, f"{asof}_blend50_band10_decision_book.csv")
    db.to_csv(csv_p, index=False)
    sec = db[db["target_weight"] > 0].groupby("sector")["target_weight"].sum().sort_values(ascending=False)
    md = ["# Blended decision book (blend50 + band10) — " + asof, "",
          f"names in book: {(db['target_weight'] > 0).sum()} · max weight "
          f"{db['target_weight'].max():.1%} · actions: " +
          ", ".join(f"{a}:{c}" for a, c in db["action"].value_counts().items()), "",
          "Sector exposure: " + ", ".join(f"{s} {w:.0%}" for s, w in sec.items()), "",
          "```",
          db[db["action"] != "WATCH"].drop(columns=["caveats", "intended_execution_date"]).to_string(index=False),
          "```", "", f"Caveats: {CAVEAT}"]
    with open(csv_p.replace(".csv", ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[decision book {asof}] {(db['target_weight'] > 0).sum()} held, "
          f"maxW {db['target_weight'].max():.1%}, actions "
          f"{dict(db['action'].value_counts())} -> {csv_p}")
    return db


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    build(ap.parse_args().asof)
