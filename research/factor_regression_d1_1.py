"""Sprint Cycle 8 — factor-regression of the D1.1 product.

Is the deployable D1.1 book proprietary alpha or a known factor? Regress the D1.1
L/S returns on market/vol/momentum/size factor L/S series (same rebalances), report
betas + residual alpha (annualized) + alpha t-stat. Confirmatory of Exp-4 (which
found the ML model was ~momentum with zero residual alpha), now on the actual
product. No ML.

Decision rule (pre-registered): expect alpha t-stat < 2 and a dominant, significant
momentum beta -> known-factor product, not alpha.

Run: python -u research/factor_regression_d1_1.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, QUINTILE, HOLDING
from d1_1_pername_cap import _cap_weights
from exp4_robustness import _ols_tstats

PPY = 252 / HOLDING


def _w(sub):
    inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
    return _cap_weights(dict(zip(sub["stock"], inv / inv.sum())),
                        dict(zip(sub["stock"], sub["sector"])), 0.20, 0.10)


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    # size proxy = log trailing-20d mean dollar volume (causal)
    size_rows = []
    for sid, df in uni.items():
        df = df.sort_values("date").reset_index(drop=True)
        dv = (df["close"] * df["volume"]).rolling(20).mean()
        size_rows.append(pd.DataFrame({"date": df["date"].values, "stock": sid,
                                       "size": np.log(dv.to_numpy() + 1.0)}))
    panel = panel.merge(pd.concat(size_rows, ignore_index=True), on=["date", "stock"], how="left")

    df = panel.dropna(subset=["mom", "fwd_ret", "vol", "size"])
    gb = {d: sub for d, sub in df.groupby("date")}
    rebal = sorted(gb)[::HOLDING]
    k = max(3, round(QUINTILE * panel["stock"].nunique()))

    strat, mkt, volf, momf, sizef = [], [], [], [], []
    for d in rebal:
        day = gb[d]
        if len(day) < 2 * k:
            continue
        o = day.sort_values("mom")
        L, S = o.tail(k), o.head(k)
        wl, ws = _w(L), _w(S)
        lr = dict(zip(L["stock"], L["fwd_ret"])); sr = dict(zip(S["stock"], S["fwd_ret"]))
        strat.append(sum(wl[x]*lr[x] for x in wl) - sum(ws[x]*sr[x] for x in ws))
        mkt.append(day["fwd_ret"].mean())
        momf.append(L["fwd_ret"].mean() - S["fwd_ret"].mean())          # equal-wt momentum factor
        ov = day.sort_values("vol"); volf.append(ov.tail(k)["fwd_ret"].mean() - ov.head(k)["fwd_ret"].mean())
        oz = day.sort_values("size"); sizef.append(oz.tail(k)["fwd_ret"].mean() - oz.head(k)["fwd_ret"].mean())

    strat = np.array(strat)
    X = np.column_stack([np.ones(len(strat)), np.array(mkt), np.array(volf),
                         np.array(momf), np.array(sizef)])
    b, t, r2 = _ols_tstats(strat, X)
    names = ["alpha", "market", "volatility", "momentum", "size"]
    print(f"Cycle8 factor regression of D1.1 book. {len(strat)} rebalances, k={k}. "
          f"survivorship-biased.\n")
    print(f"{'factor':>11s} {'beta':>9s} {'t-stat':>8s}")
    for nm, bb, tt in zip(names, b, t):
        print(f"{nm:>11s} {bb:9.4f} {tt:8.2f}")
    ann_alpha = (1 + b[0]) ** PPY - 1
    print(f"  R^2 = {r2:.2f}")
    print(f"  annualized residual ALPHA = {ann_alpha:.2%} (t={t[0]:.2f})")
    print(f"\n=== VERDICT ===")
    verdict = "CONFIRMED: known-factor (momentum) product, ~zero residual alpha" \
        if (abs(t[0]) < 2 and abs(t[3]) > 2) else "UNEXPECTED — see numbers"
    print(f"  alpha t={t[0]:.2f} (expect <2) | momentum t={t[3]:.2f} (expect >2) -> {verdict}")


if __name__ == "__main__":
    main()
