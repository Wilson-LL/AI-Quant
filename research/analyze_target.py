"""Experiment 1 — Is the prediction target correct?

Compares the baseline target (close-only triple barrier, +12%/-6%/20d) against
alternatives on real cached TWSE data, using the fair framework. Three lenses:

  A. Label diagnostics   — balance, censoring, information destroyed.
  B. Learnability        — best single-feature cross-sectional IC vs each target.
  C. Economic value      — train a ridge model on each target (walk-forward),
                           then judge every resulting score on the SAME ground
                           truth: OOS rank-IC vs realised forward return, and a
                           net top-k backtest. This is model-controlled, so the
                           only thing varying is the target.

Run: python research/analyze_target.py   (after research/data.py has cached).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe
from features import FEATURE_COLS
from targets import CONTINUOUS_TARGETS, BINARY_TARGETS
from evaluation import (
    build_panel, cross_sectional_ic, walk_forward_scores, backtest_topk,
    backtest_long_short,
)


def label_diagnostics(panel):
    print("\n=== A. Label diagnostics ===")
    print(f"{'target':22s} {'kind':6s} {'n':>7s} {'pos/mean':>10s} "
          f"{'std':>8s} {'censored%':>10s}")
    for name in list(BINARY_TARGETS) + list(CONTINUOUS_TARGETS):
        col = panel[name]
        n = col.notna().sum()
        censored = col.isna().mean() * 100
        if name in BINARY_TARGETS:
            print(f"{name:22s} {'bin':6s} {n:7d} {col.mean():10.3f} "
                  f"{col.std():8.3f} {censored:9.1f}%")
        else:
            print(f"{name:22s} {'cont':6s} {n:7d} {col.mean():10.4f} "
                  f"{col.std():8.4f} {censored:9.1f}%")

    # Special: what fraction of baseline '0' labels are "no barrier touched"
    # (i.e. flat, wrongly merged with the down case)?
    base = panel["baseline_tb_close"]
    fwd = panel["fwd_ret"]
    flat_negs = ((base == 0) & (fwd > 0)).sum()
    total_negs = (base == 0).sum()
    if total_negs:
        print(f"\nBaseline pathology: {flat_negs}/{total_negs} "
              f"({100*flat_negs/total_negs:.1f}%) of the '0' (negative) class "
              f"actually had a POSITIVE 20d forward return — the close-only "
              f"'no-touch=0' rule mislabels flat/up names as losers.")


def learnability(panel):
    print("\n=== B. Learnability: best single-feature |IC| vs each target ===")
    print(f"{'target':22s} {'best_feat':14s} {'IC':>7s} {'|IC|':>7s}")
    for name in list(BINARY_TARGETS) + list(CONTINUOUS_TARGETS):
        best = (None, 0.0)
        for f in FEATURE_COLS:
            ic, _, nd, _ = cross_sectional_ic(panel, f, name)
            if nd > 20 and abs(ic) > abs(best[1]):
                best = (f, ic)
        print(f"{name:22s} {str(best[0]):14s} {best[1]:7.4f} {abs(best[1]):7.4f}")


def economic_value(panel, Y=20, k=10, cost_bps=30.0):
    print("\n=== C. Economic value (model-controlled, walk-forward, net) ===")
    print("Train ridge on each target -> score -> judge on realised fwd_ret.")
    print("IC & LS(long-short) are beta-neutral skill; long-only Sharpe carries "
          "market/sector beta and is shown only for reference.")
    print(f"{'train_target':22s} {'OOS_IC':>8s} {'IC_IR':>7s} "
          f"{'LS_shrp':>7s} {'LS_ret':>7s} {'LO_shrp':>7s} {'LO_DD':>7s} {'n':>4s}")
    results = []
    for name in list(BINARY_TARGETS) + list(CONTINUOUS_TARGETS):
        scored = walk_forward_scores(panel, target_col=name,
                                     feature_cols=FEATURE_COLS, Y=Y)
        oos = scored.dropna(subset=["score"])
        if len(oos) == 0:
            print(f"{name:22s}  (no OOS scores — not enough history)")
            continue
        ic, ir, nd, _ = cross_sectional_ic(oos, "score", "fwd_ret")
        lo, _ = backtest_topk(oos, "score", "fwd_ret", k=k,
                              holding=Y, cost_bps=cost_bps)
        ls, _ = backtest_long_short(oos, "score", "fwd_ret", k=k,
                                    holding=Y, cost_bps=cost_bps)
        results.append((name, ic, ir, lo, ls))
        print(f"{name:22s} {ic:8.4f} {ir:7.3f} {ls.get('sharpe', float('nan')):7.2f} "
              f"{ls.get('mean_ret', float('nan')):7.2%} {lo.get('sharpe', float('nan')):7.2f} "
              f"{lo.get('max_dd', float('nan')):7.2%} {lo.get('n', 0):4d}")

    def _val(x, default=-9.0):
        return default if x is None or (isinstance(x, float) and np.isnan(x)) else x

    # Rank by beta-neutral skill: long-short Sharpe (index 4), falling back to
    # OOS IC. Long-only Sharpe is deliberately NOT used to choose — it is beta.
    tradable = [r for r in results if r[4].get("n", 0) > 0]
    if tradable:
        best = max(tradable, key=lambda r: _val(r[4].get("sharpe")))
        print(f"\n>>> Best target by beta-neutral long-short Sharpe: {best[0]} "
              f"(LS Sharpe {best[4]['sharpe']:.2f}, OOS IC {best[1]:.4f})")
    elif results:
        best = max(results, key=lambda r: _val(r[1]))
        print(f"\n>>> No tradable L/S book (need >= 2k stocks per date). "
              f"Best target by OOS IC: {best[0]} (IC {best[1]:.4f})")
    return results


def main(Y=20, k=10):
    universe = load_universe()
    if len(universe) < 3:
        print(f"Only {len(universe)} stocks cached — run research/data.py first "
              f"(needs >=3 for cross-sectional work).")
        return
    # Cross-sectional top-k needs k < universe size; cap it so the backtest can
    # actually form a book (with a small cache, use ~a third of the names).
    k = max(2, min(k, len(universe) // 3))
    total_rows = sum(len(v) for v in universe.values())
    dmin = min(v["date"].min() for v in universe.values())
    dmax = max(v["date"].max() for v in universe.values())
    print(f"Universe: {len(universe)} stocks, {total_rows} rows, "
          f"{dmin.date()}..{dmax.date()}  (Y={Y}, k={k})")

    panel = build_panel(universe, Y=Y)
    print(f"Panel: {len(panel)} (date,stock) obs, "
          f"{panel['date'].nunique()} dates.")

    label_diagnostics(panel)
    learnability(panel)
    economic_value(panel, Y=Y, k=k)


if __name__ == "__main__":
    main()
