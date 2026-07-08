"""Sprint Cycles 9-12 — deeper characterization of the D1.1 book (one block).

C9 calendar decay/crowding | C10 diversification vs market/0050 | C11 rebalance-
timing robustness | C12 beta-hedged bear-regime check. Characterization only; the
D1.1 rule is unchanged; not Sharpe-optimising. No ML.

Run: python -u research/deeper_characterization.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP, load_cached
from portfolio_d1 import build_panel, QUINTILE, HOLDING
from d1_1_pername_cap import _cap_weights
from regime_c1 import universe_index

PPY = 252 / HOLDING
IS_END = pd.Timestamp("2021-12-31")


def _w(sub):
    inv = 1.0 / (sub["vol"].to_numpy() + 1e-6)
    return _cap_weights(dict(zip(sub["stock"], inv / inv.sum())),
                        dict(zip(sub["stock"], sub["sector"])), 0.20, 0.10)


def _tno(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _sh(r):
    r = np.asarray(r); r = r[~np.isnan(r)]
    return r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(PPY) if len(r) > 1 else np.nan


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    df = panel.dropna(subset=["mom", "fwd_ret", "vol"])
    gb = {d: sub for d, sub in df.groupby("date")}
    alldates = sorted(gb)
    k = max(3, round(QUINTILE * panel["stock"].nunique()))

    def run(offset=0):
        strat, market, turn, ic, dates = [], [], [], [], []
        pl, ps = {}, {}
        for d in alldates[offset::HOLDING]:
            day = gb[d]
            if len(day) < 2 * k:
                continue
            o = day.sort_values("mom"); L, S = o.tail(k), o.head(k)
            wl, ws = _w(L), _w(S)
            lr = dict(zip(L["stock"], L["fwd_ret"])); sr = dict(zip(S["stock"], S["fwd_ret"]))
            strat.append(sum(wl[x]*lr[x] for x in wl) - sum(ws[x]*sr[x] for x in ws))
            market.append(day["fwd_ret"].mean())
            turn.append(0.5 * (_tno(wl, pl) + _tno(ws, ps))); pl, ps = wl, ws
            ic.append(day["mom"].rank().corr(day["fwd_ret"].rank()))
            dates.append(d)
        net = np.array(strat) - (60/1e4)*np.array(turn)
        return (net, np.array(strat), np.array(market), np.array(ic),
                pd.Series(dates))

    net, strat, market, ic, dates = run(0)
    print(f"Deeper characterization. k={k}. full net@60 Sharpe {_sh(net):.2f}. "
          f"survivorship-biased.\n")

    # C9 calendar decay
    yb = dates.dt.year.to_numpy() <= 2021
    print("=== C9 calendar decay/crowding ===")
    print(f"  first-half 2018-21: net@60 Sharpe {_sh(net[yb]):.2f}  meanIC {np.nanmean(ic[yb]):.4f}  n {yb.sum()}")
    print(f"  second-half 2022-26: net@60 Sharpe {_sh(net[~yb]):.2f}  meanIC {np.nanmean(ic[~yb]):.4f}  n {(~yb).sum()}")
    print(f"  -> {'2nd half NOT weaker (no decay/crowding evident)' if _sh(net[~yb]) >= _sh(net[yb]) - 0.2 else '2nd half materially weaker (possible decay)'}")

    # C10 diversification
    etf = load_cached("0050")
    e = etf.sort_values("date").reset_index(drop=True)
    ec = e["close"].to_numpy(float)
    efwd = pd.Series(np.concatenate([ec[HOLDING:] / ec[:len(ec)-HOLDING] - 1,
                                     np.full(HOLDING, np.nan)]), index=e["date"].values)
    e0050 = np.array([efwd.get(pd.Timestamp(d), np.nan) for d in dates])
    m = ~np.isnan(strat) & ~np.isnan(e0050)
    print("\n=== C10 diversification (corr of D1.1 L/S returns) ===")
    print(f"  vs equal-weight market: {np.corrcoef(strat, market)[0,1]:+.2f}")
    print(f"  vs 0050 ETF (20d fwd) : {np.corrcoef(strat[m], e0050[m])[0,1]:+.2f}  "
          f"(low = good diversifier)")

    # C11 rebalance-timing robustness
    print("\n=== C11 rebalance-timing robustness (grid offset) ===")
    for off in (0, 5, 10, 15):
        no, _, _, _, _ = run(off)
        print(f"  offset {off:2d}d: net@60 Sharpe {_sh(no):.2f}")

    # C12 beta-hedged bear-regime
    reg = universe_index(uni)
    trend = np.array([reg["trend"].get(pd.Timestamp(d), np.nan) for d in dates])
    beta = np.cov(strat, market)[0, 1] / (np.var(market) + 1e-12)
    hedged = strat - beta * market   # gross returns, market-beta removed
    bear = trend < 0
    print("\n=== C12 beta-hedged (market beta subtracted) ===")
    print(f"  market beta = {beta:.2f}")
    print(f"  FULL: raw gross Sharpe {_sh(strat):.2f} -> hedged {_sh(hedged):.2f}")
    print(f"  BEAR: raw gross Sharpe {_sh(strat[bear]):.2f} -> hedged {_sh(hedged[bear]):.2f}")
    print(f"  -> {'hedging improves bear regime' if _sh(hedged[bear]) > _sh(strat[bear]) else 'hedging does not rescue bear regime'}")


if __name__ == "__main__":
    main()
