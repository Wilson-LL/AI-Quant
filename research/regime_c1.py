"""Phase C1 — simple, rule-based regime/drawdown overlay on the D1 portfolio.

Base = D1 selected rule: A1 momentum signal + inverse-vol sizing + 20% sector cap.
Purpose: cut momentum-crash left tail (2022-like), NOT maximise Sharpe. No ML, no
optimisation. Exposure scalers are fixed {1.0, 0.5, 0.0}. Every rule + threshold is
PRE-DECLARED below (round, explainable numbers — no tuning to 2022) and uses only
information available at the rebalance date.

PRE-DECLARED OVERLAY RULES (fixed before looking at any result):
  O1 Universe trend : scaler 0.5 if the equal-weight universe index's trailing
                      120-trading-day return < 0, else 1.0.
  O2 Realized vol   : scaler 0.5 if the universe index's 20-day realized vol is
                      > 1.5x its own trailing 252-day median, else 1.0.
  O3 Momentum crash : scaler 0.5 if the sum of the base strategy's last 3
                      COMPLETED rebalance returns < -3%, else 1.0. (Own recent
                      reversal; uses only realized past returns.)
  O4 Combined       : 0.0 if (trend<0 AND vol elevated)  [full risk-off, crisis],
                      0.5 if (trend<0  OR vol elevated),
                      1.0 otherwise.

Robustness (NOT tuning): O1/O2 also shown at one neighbouring threshold to confirm
the result is not a knife-edge; the DECISION uses the pre-declared values only.

Run: python -u research/regime_c1.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import load_universe, SECTOR_MAP
from portfolio_d1 import build_panel, backtest, HOLDING, QUINTILE

# --- pre-declared thresholds ---
TREND_WIN = 120
TREND_THR = 0.0
VOL_WIN = 20
VOL_MED_WIN = 252
VOL_MULT = 1.5
CRASH_LOOKBACK = 3      # completed rebalances
CRASH_THR = -0.03
COSTS = [0, 30, 60, 80, 100]


def universe_index(uni):
    """Equal-weight universe daily index + trend & vol regime series (causal)."""
    rets = []
    for sid, df in uni.items():
        df = df.sort_values("date")
        r = df["close"].pct_change()
        rets.append(pd.DataFrame({"date": df["date"].values, "r": r.values}))
    daily = (pd.concat(rets).groupby("date")["r"].mean().sort_index())
    idx = (1 + daily).cumprod()
    trend = idx / idx.shift(TREND_WIN) - 1.0
    rvol = daily.rolling(VOL_WIN).std()
    rvol_med = rvol.rolling(VOL_MED_WIN).median()
    return pd.DataFrame({"trend": trend, "rvol": rvol, "rvol_med": rvol_med})


def scalers(dates, base_gross, reg, overlay, vol_mult=VOL_MULT, trend_thr=TREND_THR):
    """Return the exposure scaler per rebalance for a given overlay (causal)."""
    s = np.ones(len(dates))
    for i, d in enumerate(dates):
        row = reg.reindex([pd.Timestamp(d)]).iloc[0] if pd.Timestamp(d) in reg.index else None
        trend_neg = row is not None and pd.notna(row["trend"]) and row["trend"] < trend_thr
        elevated = (row is not None and pd.notna(row["rvol"]) and pd.notna(row["rvol_med"])
                    and row["rvol"] > vol_mult * row["rvol_med"])
        # crash: last CRASH_LOOKBACK completed rebalances (indices < i)
        past = base_gross[max(0, i - CRASH_LOOKBACK):i]
        crash = len(past) >= 1 and float(np.nansum(past)) < CRASH_THR
        if overlay == "O1_trend":
            s[i] = 0.5 if trend_neg else 1.0
        elif overlay == "O2_vol":
            s[i] = 0.5 if elevated else 1.0
        elif overlay == "O3_crash":
            s[i] = 0.5 if crash else 1.0
        elif overlay == "O4_combined":
            s[i] = 0.0 if (trend_neg and elevated) else (0.5 if (trend_neg or elevated) else 1.0)
    return s


def apply_overlay(base, scaler, cost_bps):
    gross = base["gross"] * scaler
    ds = np.abs(np.diff(np.concatenate([[1.0], scaler])))  # exposure-change trading
    turn = base["turn"] + ds
    return gross - (cost_bps / 1e4) * turn, turn


def metrics(net, turn, dates, holding=HOLDING):
    net = np.asarray(net)
    ppy = 252 / holding
    mean, std = np.nanmean(net), np.nanstd(net, ddof=1)
    sharpe = mean / (std + 1e-12) * np.sqrt(ppy)
    eq = np.cumprod(1 + np.nan_to_num(net))
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    ann = (1 + mean) ** ppy - 1
    yrs = pd.Series(dates).dt.year.to_numpy()
    ymean = {y: float(np.nanmean(net[yrs == y])) for y in sorted(set(yrs))}
    ysharpe = {y: (np.nanmean(net[yrs == y]) /
                   (np.nanstd(net[yrs == y], ddof=1) + 1e-12) * np.sqrt(ppy)
                   if (yrs == y).sum() > 1 else float("nan")) for y in ymean}
    worst_y = min(ymean, key=ymean.get)
    return {"sharpe": sharpe, "ann": ann, "dd": dd,
            "calmar": ann / (abs(dd) + 1e-12),
            "turnover": float(np.nanmean(turn)) * ppy,
            "worst_year": worst_y, "worst_year_mean": ymean[worst_y],
            "y2022": ymean.get(2022, float("nan")),
            "ymean": ymean, "ysharpe": ysharpe}


def main():
    ids = [s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf"]
    uni = load_universe(ids)
    panel = build_panel(uni)
    k = max(3, round(QUINTILE * panel["stock"].nunique()))
    base = backtest(panel, k, "invvol_cap", 0.20)   # D1 selected rule
    reg = universe_index(uni)
    dates = base["dates"]

    print(f"C1 base = D1 (invvol + 20% cap), {panel['stock'].nunique()} names, k={k}. "
          f"net@60bps; survivorship-biased. Scalers fixed {{1.0,0.5,0.0}}.")
    print("Pre-declared: O1 trend120<0; O2 rvol20>1.5x med252; O3 last-3-rebal sum<-3%; "
          "O4 both->0, either->0.5.\n")

    base_net = base["gross"] - (60 / 1e4) * base["turn"]
    m0 = metrics(base_net, base["turn"], dates)

    overlays = ["O1_trend", "O2_vol", "O3_crash", "O4_combined"]
    rows = {"D1 base (no overlay)": (base_net, base["turn"], np.ones(len(dates)), m0)}
    for ov in overlays:
        sc = scalers(dates, base["gross"], reg, ov)
        net, turn = apply_overlay(base, sc, 60)
        rows[ov] = (net, turn, sc, metrics(net, turn, dates))

    # main comparison table
    hdr = (f"{'variant':20s} {'Sharpe':>7s} {'annRet':>8s} {'maxDD':>8s} {'Calmar':>7s} "
           f"{'worstY':>7s} {'2022%':>7s} {'turn':>6s} {'%off':>6s} {'#red':>5s}")
    print(hdr)
    for name, (net, turn, sc, m) in rows.items():
        pct_off = np.mean(sc < 1.0) * 100
        n_red = int(np.sum(sc < 1.0))
        print(f"{name:20s} {m['sharpe']:7.2f} {m['ann']:8.2%} {m['dd']:8.2%} "
              f"{m['calmar']:7.2f} {m['worst_year_mean']*100:6.2f}% {m['y2022']*100:6.2f}% "
              f"{m['turnover']:5.1f}x {pct_off:5.0f}% {n_red:5d}")

    # decision-rule check vs base
    print("\n=== Decision-rule check vs D1 base (need: maxDD -20% | Calmar +20% | "
          "worstYr -30%; AND Sharpe drop <= 0.15) ===")
    for name in overlays:
        m = rows[name][3]
        dd_imp = (m0["dd"] - m["dd"]) / abs(m0["dd"])  # positive = smaller DD
        cal_imp = (m["calmar"] - m0["calmar"]) / abs(m0["calmar"])
        wy_imp = (m["worst_year_mean"] - m0["worst_year_mean"]) / abs(m0["worst_year_mean"])
        sh_drop = m0["sharpe"] - m["sharpe"]
        passes = ((dd_imp >= 0.20 or cal_imp >= 0.20 or wy_imp >= 0.30)
                  and sh_drop <= 0.15)
        print(f"  {name:14s} dd_imp={dd_imp:+.0%} calmar_imp={cal_imp:+.0%} "
              f"worstY_imp={wy_imp:+.0%} sharpe_drop={sh_drop:+.2f} -> "
              f"{'PASS' if passes else 'fail'}")

    # non-crash-year edge preservation (exclude 2022)
    print("\n=== Edge preservation: Sharpe EXCLUDING 2022 (crash year) ===")
    yrs = pd.Series(dates).dt.year.to_numpy()
    keep = yrs != 2022
    for name, (net, turn, sc, m) in rows.items():
        r = np.asarray(net)[keep]
        s = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(252 / HOLDING)
        print(f"  {name:20s} ex-2022 Sharpe = {s:.2f}")

    # yearly for base vs best-looking overlay(s)
    print("\n=== Yearly net@60bps mean% (2018-2026) ===")
    show = ["D1 base (no overlay)"] + overlays
    yset = sorted(set(pd.Series(dates).dt.year))
    print("year   " + "  ".join(f"{n.split('_')[0][:8]:>8s}" for n in show))
    for y in yset:
        cells = "  ".join(f"{rows[n][3]['ymean'].get(y, float('nan'))*100:8.2f}" for n in show)
        print(f"{y:<6d} {cells}")

    # cost sensitivity for base + O1 + O4
    print("\n=== Cost sensitivity net Sharpe (base, O1, O4) ===")
    print("variant              " + " ".join(f"{c:>5d}bps" for c in COSTS))
    for name in ["D1 base (no overlay)", "O1_trend", "O4_combined"]:
        sc = rows[name][2]
        cells = []
        for c in COSTS:
            net_c, turn_c = apply_overlay(base, sc, c)
            cells.append(f"{metrics(net_c, turn_c, dates)['sharpe']:8.2f}")
        print(f"{name:20s} {' '.join(cells)}")

    # robustness neighbours (NOT for decision)
    print("\n=== Robustness (neighbouring thresholds; decision uses pre-declared) ===")
    for vm in (1.3, 1.5, 2.0):
        sc = scalers(dates, base["gross"], reg, "O2_vol", vol_mult=vm)
        net, turn = apply_overlay(base, sc, 60)
        m = metrics(net, turn, dates)
        print(f"  O2 vol_mult={vm}: Sharpe {m['sharpe']:.2f} maxDD {m['dd']:.2%} "
              f"2022 {m['y2022']*100:.2f}% %off {np.mean(sc<1)*100:.0f}%")


if __name__ == "__main__":
    main()
