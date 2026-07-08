"""D1.1 prototype — clean, deterministic momentum-factor product pipeline.

Deliverable of the A -> D -> C arc, refined by D1.1 (RESEARCH_LOG.md /
ROADMAP_A_D_C.md). NOT machine learning; does NOT touch production
train.py / inference.py. Reuses vetted, unit-consistent logic so the prototype
and the research backtest share one code path.

THE RULE (D1.1, fixed & explainable):
  1. Signal   : 6-month momentum, skip last week: mom_t = close[t-5]/close[t-131]-1
  2. Select   : rank cross-sectionally; long top quintile, short bottom.
  3. Size     : inverse-volatility (1 / trailing 60d return vol), past-only.
  4. Caps     : 20% per SECTOR and 10% per NAME (redistribute; both enforced).
  5. Hold     : 20 trading days, then rebalance.

D1.1 = the frozen D1 rule + a 10% per-name cap (validated in
d1_1_pername_cap.py: cuts max name weight 13.5%->11.0% and sector P&L share
37%->35% at preserved net Sharpe and flat drawdown). The ORIGINAL D1 (sector cap
only) remains frozen and reproducible in portfolio_d1.py.

Honest framing: this harvests the cross-sectional MOMENTUM premium — a known
factor, NOT proprietary alpha (Experiment 4). Historical numbers are
survivorship-biased upward -> UPPER BOUND. Shorting TWSE is hard, so a long-only
top-quintile book is also emitted (it carries market beta).

Usage:
  python research/prototype.py                 # today's book + strategy card
  from prototype import generate_book, backtest_summary
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from momentum import LOOKBACK, SKIP, HOLDING, QUINTILE
from portfolio_d1 import build_panel                     # frozen D1 panel builder
from d1_1_pername_cap import backtest, perf, _leg_weights, SECTOR_CAP

NAME_CAP = 0.10


def _load_panel():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    if len(uni) < 10:
        raise SystemExit("Universe cache too small — run research/data.py first.")
    return build_panel(uni)


def generate_book(panel=None, mode="long_short"):
    """Current target portfolio as a DataFrame [side, stock, sector, weight, mom,
    asof], using each name's latest valid (past-only) signal, D1.1 weighting."""
    if panel is None:
        panel = _load_panel()
    snap = (panel.dropna(subset=["mom", "vol"])
                 .sort_values("date").groupby("stock").tail(1)
                 .reset_index(drop=True))
    n = len(snap)
    k = max(3, round(QUINTILE * n))
    o = snap.sort_values("mom")
    longs = o.tail(k)
    wl, _ = _leg_weights(longs, NAME_CAP)

    def _leg(df, w, side):
        return pd.DataFrame({
            "side": side, "stock": df["stock"].values, "sector": df["sector"].values,
            "weight": [w[s] * (1 if side == "LONG" else -1) for s in df["stock"]],
            "mom": df["mom"].values,
            "asof": df["date"].dt.date.astype(str).values,
        })

    book = _leg(longs, wl, "LONG")
    if mode == "long_short":
        shorts = o.head(k)
        ws, _ = _leg_weights(shorts, NAME_CAP)
        book = pd.concat([book, _leg(shorts, ws, "SHORT")], ignore_index=True)
    return book.sort_values(
        ["side", "weight"],
        key=lambda s: s.abs() if s.name == "weight" else s,
        ascending=[True, False]).reset_index(drop=True)


def backtest_summary(panel=None):
    if panel is None:
        panel = _load_panel()
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    res = backtest(panel, k, NAME_CAP)
    return {c: perf(res, c) for c in (0, 60, 80)}, res


def _sector_report(leg):
    return (leg.assign(aw=leg["weight"].abs()).groupby("sector")["aw"].sum()
               .sort_values(ascending=False))


def main():
    panel = _load_panel()
    print("=" * 74)
    print("  AI-Quant D1.1 momentum product — prototype pipeline")
    print("=" * 74)
    print(f"Rule: mom=close[t-{SKIP}]/close[t-{SKIP+LOOKBACK}]-1 | top/bottom quintile "
          f"| inverse-vol sizing | {int(SECTOR_CAP*100)}% sector + {int(NAME_CAP*100)}% "
          f"name cap | {HOLDING}d hold")
    print("Type: cross-sectional MOMENTUM factor product (known premium, not alpha).")

    summ, res = backtest_summary(panel)
    m = summ[60]
    print("\n-- Historical validation (net@60bps; SURVIVORSHIP-BIASED upper bound) --")
    print(f"  net Sharpe   : {m['sharpe']:.2f}   (net@0bps {summ[0]['sharpe']:.2f}, "
          f"net@80bps {summ[80]['sharpe']:.2f})")
    print(f"  ann. return  : {m['ann']:.1%}")
    print(f"  max drawdown : {m['dd']:.1%}   Calmar {m['calmar']:.2f}")
    print(f"  turnover     : {m['turnover']:.1f}x/yr")
    print(f"  concentration: max name {m['max_namew']:.0%}, max sector "
          f"{m['max_secw']:.0%}, max sector P&L share {m['max_pnl_share']:.0%}")
    print(f"  rebalances   : {len(res['gross'])}")

    for mode in ("long_short", "long_only"):
        book = generate_book(panel, mode=mode)
        asof = book["asof"].max()
        print(f"\n-- TARGET BOOK ({mode}, as of ~{asof}) --")
        for side in ["LONG", "SHORT"]:
            leg = book[book["side"] == side]
            if leg.empty:
                continue
            print(f"  {side} leg: {len(leg)} names, gross weight "
                  f"{leg['weight'].abs().sum():.0%}, max name "
                  f"{leg['weight'].abs().max():.1%}")
            top = leg.reindex(leg["weight"].abs().sort_values(ascending=False).index).head(8)
            for _, r in top.iterrows():
                print(f"      {r['stock']:>5s} {r['sector']:<12s} {r['weight']:+6.1%}"
                      f"  mom={r['mom']:+.1%}")
            if len(leg) > 8:
                print(f"      … +{len(leg)-8} more")
            sr = _sector_report(leg)
            print("      sector wts: " + ", ".join(f"{s} {w:.0%}" for s, w in sr.head(5).items()))

    print("\nNotes: D1.1 = frozen D1 + 10% per-name cap (D1 itself stays reproducible "
          "in portfolio_d1.py). Long-only carries market beta. No regime overlay "
          "(Phase C1 found none worthwhile). Deterministic & reproducible; not "
          "committed; production train.py/inference.py untouched.")


if __name__ == "__main__":
    main()
