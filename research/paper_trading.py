"""F1 — paper-trading scaffold (shadow validation; no brokerage, no real trades).

Tracks strategies: d12 (mom126/5), tf (champion transformer), blend50 (z-avg),
blend50_band10 (promoted candidate). Long-only top-quintile, equal-weight,
hard 10% name cap; execution next session (lag 1).

Modes:
  backfill  — seed books from the cached walk-forward OOS panel (scores were
              produced walk-forward, so this is legitimate shadow history),
              every 20 panel days from --start (default 2025-01-01).
  snapshot  — build today's books from reports/transformer_gpu/<asof>_predictions.csv
              (champion daily inference output) + D1.2 scores from cache.
  evaluate  — for every saved book, compute matured 1/5/10/20d forward returns,
              turnover, hit rate; write PAPER_LEDGER.csv + PAPER_REPORT.md.

Usage:
  python research/paper_trading.py backfill [--start 2025-01-01]
  python research/paper_trading.py snapshot [--asof 2026-07-07]
  python research/paper_trading.py evaluate
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import SECTOR_MAP  # noqa: E402
from transformer_hybrid import load_panel, merged, _cache_frames  # noqa: E402
from transformer_portfolio import cap_weights  # noqa: E402

PT_DIR = os.path.join(ROOT, "reports", "paper_trading")
BOOK_DIR = os.path.join(PT_DIR, "books")
STRATS = ("d12", "tf", "blend50", "blend50_band10")
QUINTILE = 0.2
BAND = 0.10


def _book_from_scores(day, prev_names=None, band=0.0):
    """day: DataFrame[stock, score]. Returns DataFrame[stock, weight, score, rank]."""
    day = day.dropna(subset=["score"]).drop_duplicates("stock").sort_values(
        "score", ascending=False).reset_index(drop=True)
    n = len(day)
    k = max(3, round(QUINTILE * n))
    longs = list(day["stock"].head(k))
    if band and prev_names:
        pool = set(day["stock"].head(int(k * (1 + 2 * band))))
        incumbents = [s for s in prev_names if s in pool]
        fresh = [s for s in longs if s not in incumbents]
        longs = (incumbents + fresh)[:k]
    sub = day[day["stock"].isin(longs)].copy()
    sectors = [SECTOR_MAP.get(s, "other") for s in sub["stock"]]
    sub["weight"] = cap_weights(np.ones(len(sub)), sectors)
    sub["rank"] = sub["score"].rank(ascending=False).astype(int)
    return sub[["stock", "weight", "score", "rank"]]


def _save_book(asof, strat, book):
    os.makedirs(BOOK_DIR, exist_ok=True)
    p = os.path.join(BOOK_DIR, f"{asof}_{strat}.csv")
    book.to_csv(p, index=False)
    return p


def _prev_book(strat, before_asof):
    if not os.path.isdir(BOOK_DIR):
        return None
    cands = sorted(f for f in os.listdir(BOOK_DIR)
                   if f.endswith(f"_{strat}.csv") and f[:10] < before_asof)
    if not cands:
        return None
    return pd.read_csv(os.path.join(BOOK_DIR, cands[-1]), dtype={"stock": str})


def backfill(start="2025-01-01"):
    panel, pname = load_panel("G9_presetB_equal_all")
    m = merged(panel)
    dates = np.array(sorted(m["date"].unique()))
    rebal = [d for d in dates[::20] if d >= np.datetime64(start)]
    print(f"[backfill] {len(rebal)} rebalance dates from {pname}")
    for d in rebal:
        day = m[m["date"] == d]
        asof = str(d)[:10]
        scores = {"tf": day.rename(columns={"z_tf": "s"})[["stock", "s"]],
                  "d12": day.rename(columns={"z_mom": "s"})[["stock", "s"]]}
        scores["blend50"] = day.assign(s=0.5 * day["z_tf"] + 0.5 * day["z_mom"])[["stock", "s"]]
        for strat in STRATS:
            base = "blend50" if strat == "blend50_band10" else strat
            sd = scores[base].rename(columns={"s": "score"})
            prev = _prev_book(strat, asof)
            book = _book_from_scores(
                sd, prev_names=list(prev["stock"]) if prev is not None else None,
                band=BAND if strat.endswith("band10") else 0.0)
            _save_book(asof, strat, book)
        print(f"  {asof}: books written ({day['stock'].nunique()} names scored)")


def snapshot(asof=None):
    pred_dir = os.path.join(ROOT, "reports", "transformer_gpu")
    if asof is None:
        cands = sorted(f[:10] for f in os.listdir(pred_dir) if f.endswith("_predictions.csv"))
        if not cands:
            sys.exit("no predictions file; run inference_transformer_eod.py first")
        asof = cands[-1]
    preds = pd.read_csv(os.path.join(pred_dir, f"{asof}_predictions.csv"), dtype={"stock": str})
    # D1.2 score from cache at asof
    rows = []
    for sid, df in _cache_frames().items():
        df = df[df["date"] <= pd.Timestamp(asof)]
        if len(df) < 132:
            continue
        c = df["close"].to_numpy(np.float64)
        rows.append({"stock": sid, "mom": c[-6] / c[-132] - 1.0})
    d12 = pd.DataFrame(rows)
    mm = preds[["stock", "score"]].merge(d12, on="stock", how="inner")
    mm["z_tf"] = (mm["score"] - mm["score"].mean()) / (mm["score"].std() + 1e-9)
    mm["z_mom"] = (mm["mom"] - mm["mom"].mean()) / (mm["mom"].std() + 1e-9)
    scores = {"tf": mm.assign(s=mm["z_tf"]), "d12": mm.assign(s=mm["z_mom"]),
              "blend50": mm.assign(s=0.5 * mm["z_tf"] + 0.5 * mm["z_mom"])}
    for strat in STRATS:
        base = "blend50" if strat == "blend50_band10" else strat
        sd = scores[base][["stock", "s"]].rename(columns={"s": "score"})
        prev = _prev_book(strat, asof)
        book = _book_from_scores(
            sd, prev_names=list(prev["stock"]) if prev is not None else None,
            band=BAND if strat.endswith("band10") else 0.0)
        p = _save_book(asof, strat, book)
        print(f"[snapshot {asof}] {strat}: {len(book)} names -> {p}")


def evaluate():
    frames = _cache_frames()
    close = {sid: df.set_index("date")["close"] for sid, df in frames.items()}
    all_dates = sorted(set().union(*[set(c.index) for c in close.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    def fwd(sid, asof, h):
        c = close.get(sid)
        if c is None:
            return np.nan
        pos = all_dates.searchsorted(pd.Timestamp(asof))
        # exec at close of next session; matured h sessions later
        if pos + 1 + h >= len(all_dates):
            return np.nan
        d0, d1 = all_dates[pos + 1], all_dates[pos + 1 + h]
        if d0 not in c.index or d1 not in c.index:
            return np.nan
        return float(c.loc[d1] / c.loc[d0] - 1.0)

    rows = []
    books = sorted(os.listdir(BOOK_DIR)) if os.path.isdir(BOOK_DIR) else []
    for f in books:
        asof, strat = f[:10], f[11:-4]
        book = pd.read_csv(os.path.join(BOOK_DIR, f), dtype={"stock": str})
        prev = _prev_book(strat, asof)
        turn = np.nan
        if prev is not None:
            w0 = prev.set_index("stock")["weight"]
            w1 = book.set_index("stock")["weight"]
            alln = w0.index.union(w1.index)
            turn = float((w1.reindex(alln, fill_value=0) - w0.reindex(alln, fill_value=0)).abs().sum() / 2)
        rec = {"asof": asof, "strategy": strat, "n_names": len(book),
               "max_w": round(float(book["weight"].max()), 4), "turnover": round(turn, 4) if turn == turn else np.nan}
        for h in (1, 5, 10, 20):
            r = np.array([fwd(s, asof, h) for s in book["stock"]])
            w = book["weight"].to_numpy()
            ok = np.isfinite(r)
            rec[f"ret_{h}d"] = round(float(np.sum(w[ok] * r[ok]) / max(w[ok].sum(), 1e-9)), 5) if ok.any() else np.nan
            rec[f"hit_{h}d"] = round(float((r[ok] > 0).mean()), 3) if ok.any() else np.nan
        rows.append(rec)
    led = pd.DataFrame(rows)
    os.makedirs(PT_DIR, exist_ok=True)
    led.to_csv(os.path.join(PT_DIR, "PAPER_LEDGER.csv"), index=False)

    lines = ["# Paper-trading report", "",
             f"Books: {len(led)} · strategies: {', '.join(STRATS)}", "",
             "## Matured 20d-return summary by strategy (per rebalance, gross)", ""]
    if len(led):
        g = led.dropna(subset=["ret_20d"]).groupby("strategy")["ret_20d"]
        summ = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()})
        summ["sharpe_ann"] = summ["mean"] / (summ["std"] + 1e-12) * np.sqrt(252 / 20)
        lines += ["| strategy | n matured | mean 20d | ann Sharpe (gross) | avg turn |",
                  "|---|--:|--:|--:|--:|"]
        for s_name, r in summ.iterrows():
            at = led[led["strategy"] == s_name]["turnover"].mean()
            lines.append(f"| {s_name} | {int(r['n'])} | {r['mean']:+.2%} | {r['sharpe_ann']:.2f} | {at:.2f} |")
    with open(os.path.join(PT_DIR, "PAPER_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(led.to_string(index=False) if len(led) else "no books yet")
    print("\n->", os.path.join(PT_DIR, "PAPER_LEDGER.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backfill", "snapshot", "evaluate"])
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()
    if a.mode == "backfill":
        backfill(a.start)
    elif a.mode == "snapshot":
        snapshot(a.asof)
    else:
        evaluate()
