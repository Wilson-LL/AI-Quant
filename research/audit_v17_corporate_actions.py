"""v17 P1 — corporate-action distortion scan on the UNADJUSTED cache.

No dividend table exists locally, so events are DETECTED from price
behaviour: a symbol-day whose close-to-close return is far below the
equal-weight universe return that day (market-residual < -RES_THRESH)
is a candidate ex-dividend / ex-rights / capital-action print. TWSE cash
dividends cluster Jun-Aug, so the seasonal excess of such prints over the
rest of the year is the evidence that they are corporate actions rather
than ordinary crashes. Outputs reports/model_audit/v17/corporate_actions.csv
+ a summary dict. Read-only; production data untouched.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from transformer_hybrid import _cache_frames  # noqa: E402

OUT = os.path.join(ROOT, "reports", "model_audit", "v17")
RES_THRESH = 0.04      # residual drop that a normal day rarely produces
LABEL_LEN = 21         # sessions spanned by fwd_20 (T+1..T+21)


def main():
    frames = _cache_frames()
    wide = pd.concat({s: df.set_index("date")["close"] for s, df in
                      frames.items()}, axis=1).sort_index()
    ret = wide.pct_change()
    mkt = ret.mean(axis=1)
    resid = ret.sub(mkt, axis=0)
    cand = (resid < -RES_THRESH)
    ev = cand.stack()
    ev = ev[ev]
    ev = ev.reset_index()
    ev.columns = ["date", "stock", "flag"]
    ev["resid"] = [resid.at[d, s] for d, s in zip(ev["date"], ev["stock"])]
    ev["month"] = ev["date"].dt.month
    ev["year"] = ev["date"].dt.year
    by_month = ev.groupby("month").size()
    n_days_by_month = wide.index.to_series().groupby(wide.index.month).size()
    rate = (by_month / n_days_by_month).rename("events_per_session")
    base = rate[[1, 2, 3, 4, 5, 10, 11, 12]].mean()
    peak = rate[[6, 7, 8, 9]]
    excess = (peak - base).clip(lower=0) * n_days_by_month[[6, 7, 8, 9]]
    n_years = wide.index.year.nunique()
    est_ca_events = float(excess.sum())          # seasonal excess = CA proxy
    # label windows affected: a symbol-date T whose label window T+1..T+21
    # contains a candidate event for that symbol
    ev_mat = cand.astype(int)
    win = ev_mat.iloc[::-1].rolling(LABEL_LEN, min_periods=1).sum().iloc[::-1].shift(-1)
    affected = (win > 0)
    valid = wide.notna() & wide.shift(-LABEL_LEN).notna()
    frac_windows = float(affected[valid].sum().sum() / valid.sum().sum())
    # seasonal version (Jun-Sep only)
    sel = wide.index.month.isin([6, 7, 8, 9])
    frac_windows_summer = float(affected[valid][sel].sum().sum() / valid[sel].sum().sum())
    # typical size of the residual drop on candidate days
    summer_ev = ev[ev["month"].isin([6, 7, 8, 9])]
    summary = {
        "n_sessions": int(len(wide)), "n_stocks": int(wide.shape[1]),
        "n_years": int(n_years),
        "candidate_events_total": int(len(ev)),
        "events_per_session_by_month": {int(k): round(float(v), 4)
                                        for k, v in rate.items()},
        "seasonal_excess_events_est": round(est_ca_events, 1),
        "seasonal_excess_events_per_stock_year": round(
            est_ca_events / wide.shape[1] / n_years, 3),
        "median_resid_drop_summer": round(float(summer_ev["resid"].median()), 4),
        "p25_resid_drop_summer": round(float(summer_ev["resid"].quantile(0.25)), 4),
        "frac_label_windows_containing_candidate": round(frac_windows, 4),
        "frac_label_windows_containing_candidate_JunSep": round(
            frac_windows_summer, 4),
        "fwd20_cross_sectional_std": round(float(
            (wide.shift(-21) / wide.shift(-1) - 1).std(axis=1).mean()), 4),
        "adjustment_source_local": "NONE (twstock raw; data_cache_full has "
                                   "no dividend/adjust fields)",
    }
    os.makedirs(OUT, exist_ok=True)
    ev.to_csv(os.path.join(OUT, "corporate_action_candidates.csv"), index=False)
    with open(os.path.join(OUT, "corporate_actions_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
