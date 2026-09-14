"""v17 Parts 3-4 — prospective independent-sample analysis and a
selection-adjusted (deflated) Sharpe sanity check for the champion.

Prospective: daily IC rows from prospective.csv are OVERLAPPING (20-session
labels). We report calendar span, daily n, greedy non-overlapping block
ICs (every start date), and an HAC-style effective-n so the mean IC is
never mistaken for an independent sample.

Deflated Sharpe (Bailey & Lopez de Prado 2014): given N materially
distinct trials on the same window with trial-Sharpe std V, the expected
max of null trials is SR0 = sqrt(V) * ((1-g) Z(1-1/N) + g Z(1-1/(N e))),
g = Euler-Mascheroni. DSR = P(true SR > SR0) using the per-block Sharpe,
its n, skew and kurtosis. We report a RANGE over defensible (N, V)
assumptions rather than one number. Outputs statistics_summary.json.
"""

import json
import os
import sys
from math import erf, sqrt, log, exp

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from transformer_hybrid import load_panel, merged  # noqa: E402
from transformer_portfolio import backtest_scores  # noqa: E402

OUT = os.path.join(ROOT, "reports", "model_audit", "v17")
EULER = 0.5772156649


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def Zinv(p):
    # Acklam-free: bisection on Phi (good enough here)
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def prospective():
    p = pd.read_csv(os.path.join(OUT, "prospective.csv"), parse_dates=["date"])
    out = {"first_date": str(p["date"].min().date()),
           "last_date": str(p["date"].max().date()),
           "calendar_days": int((p["date"].max() - p["date"].min()).days),
           "n_prediction_dates": int(len(p))}
    for h, step in ((5, 6), (10, 11), (20, 21)):
        s = p.dropna(subset=[f"ic_{h}"]).reset_index(drop=True)
        n = len(s)
        blocks = []
        # greedy non-overlapping selections for every possible start
        for start in range(min(step, n)):
            sel = s.iloc[start::step]
            blocks.append(float(sel[f"ic_{h}"].mean()))
        ics = s[f"ic_{h}"].to_numpy()
        # HAC (Bartlett, lag = overlap) variance of the mean
        lag = step - 1
        dm = ics - ics.mean()
        gam = [float(np.mean(dm[:n - k] * dm[k:])) if n - k > 1 else 0.0
               for k in range(0, min(lag, n - 1) + 1)]
        var = gam[0] + 2 * sum((1 - k / (lag + 1)) * gam[k] for k in range(1, len(gam)))
        se = sqrt(max(var, 1e-12) / n)
        out[f"h{h}"] = {
            "n_daily_obs": n, "mean_ic": round(float(ics.mean()), 4),
            "pos_frac": round(float((ics > 0).mean()), 3),
            "approx_nonoverlap_blocks": int(np.ceil(n / step)),
            "block_mean_ic_min": round(min(blocks), 4),
            "block_mean_ic_max": round(max(blocks), 4),
            "hac_se_of_mean": round(se, 4),
            "hac_95ci": [round(float(ics.mean() - 1.96 * se), 4),
                         round(float(ics.mean() + 1.96 * se), 4)],
            "n_eff_from_hac": round(float(ics.var() / max(var, 1e-12)) if var > 0 else 0, 2),
        }
    return out


def deflated(tag, name):
    panel, _ = load_panel(name)
    m = merged(panel)
    m["blend"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    pp = m[["date", "stock", "blend", "fwd_h", "vol_20"]].rename(columns={"blend": "score"})
    r = backtest_scores(pp, mode="long_only", no_trade_band=0.10, cost_bps_list=(60,))
    net = np.asarray(r["gross"]) - (60 / 1e4) * np.asarray(r["turnover"])
    n = len(net)
    sr = net.mean() / net.std(ddof=1)                # per-block Sharpe
    d = net - net.mean()
    skew = float((d ** 3).mean() / net.std() ** 3)
    kurt = float((d ** 4).mean() / net.std() ** 4)
    ann = sqrt(252 / 20)
    res = {"panel": tag, "n_blocks": n, "sharpe_annual": round(sr * ann, 3),
           "sharpe_per_block": round(sr, 4), "skew": round(skew, 3),
           "kurtosis": round(kurt, 3), "scenarios": []}
    denom = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-9))
    for N in (10, 30, 70):
        for V_ann in (0.20, 0.35, 0.50):       # std of trial Sharpes (annualized)
            V = (V_ann / ann) ** 2
            sr0 = sqrt(V) * ((1 - EULER) * Zinv(1 - 1 / N) + EULER * Zinv(1 - 1 / (N * exp(1))))
            z = (sr - sr0) * sqrt(n - 1) / denom
            res["scenarios"].append({"N_trials": N, "trial_sharpe_std_annual": V_ann,
                                     "expected_max_null_sharpe_annual": round(sr0 * ann, 3),
                                     "haircut_annual": round(sr0 * ann, 3),
                                     "DSR_prob_true_sr_exceeds_null_max": round(Phi(z), 3)})
    # plain (non-deflated) probabilistic Sharpe vs 0 and vs 1.0 annual
    for bench in (0.0, 1.0):
        b = bench / ann
        z = (sr - b) * sqrt(n - 1) / denom
        res[f"PSR_vs_{bench}"] = round(Phi(z), 3)
    return res


def main():
    out = {"prospective": prospective(),
           "deflated": [deflated("CH", "SCHED_A8_seeds7_full"),
                        deflated("BR", "SCHED_BEAR_A8_seeds7_full")]}
    with open(os.path.join(OUT, "statistics_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
