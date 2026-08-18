"""v16 Stage B — historical OOS validation of the night-before price bands.

Walk-forward evaluation on the frozen BR 7-seed panel (2021->, OOS
signals): at each 20-session rebalance date T the long book is
reconstructed with the production selection rule; entrants get FRESH
entry bands, incumbents get EXISTING-target bands, exits get SELL bands —
all calibrated from observations STRICTLY BEFORE T (expanding window;
epb.BandCalibrator strict=True). T+1 open/high/low are outcomes only.

Explicitly NOT evaluated: any "stop hit before target" sequencing —
daily OHLC is path-ambiguous (v14 finding); no code here interprets
intraday ordering.

Metrics are pre-registered in price_band_methodology.md. Descriptive
coverage only; no profitability claim.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import execution_price_bands as epb  # noqa: E402
from transformer_hybrid import load_panel, merged, _cache_frames  # noqa: E402

OUT = os.path.join(ROOT, "reports", "continuous_research",
                   "v16_next_session_execution")
PANEL = "SCHED_BEAR_A8_seeds7_full"
H = 20


def role_rows(m):
    """Reconstruct long-book roles per rebalance (production selection:
    top quintile, band10 incumbent retention). Yields dicts with role,
    rank_pct, per-date info."""
    d2 = (m.dropna(subset=["score"])
           .drop_duplicates(["date", "stock"], keep="last"))
    dates = np.array(sorted(d2["date"].unique()))
    by_date = dict(tuple(d2.groupby("date")))
    prev = []
    out = []
    for d in dates[::H]:
        day = by_date.get(d)
        if day is None or day["stock"].nunique() < 60:
            continue
        n = day["stock"].nunique()
        kq = max(3, round(0.2 * n))
        o = day.sort_values("score")
        pool = set(o.tail(int(kq * 1.2))["stock"])
        longs = list(o.tail(kq)["stock"])
        if prev:
            inc = [s for s in prev if s in pool]
            longs = (inc + [s for s in longs if s not in inc])[:kq]
        rk = day.set_index("stock")["score"].rank(ascending=False, pct=True)
        for role, names in (
                ("entrant", [s for s in longs if s not in prev]),
                ("incumbent", [s for s in longs if s in prev]),
                ("exit", [s for s in prev if s not in longs])):
            for s in names:
                if s in rk.index:
                    out.append({"date": d, "stock": s, "role": role,
                                "rank_pct": float(rk[s])})
        prev = longs
    return pd.DataFrame(out)


def main():
    panel, _ = load_panel(PANEL)
    m = merged(panel)
    m["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    frames = _cache_frames()
    hist = epb.build_history(frames,
                             score_panel=m[["date", "stock", "score"]])
    cal = epb.BandCalibrator(hist)
    hidx = hist.set_index(["date", "stock"])
    roles = role_rows(m)
    print(f"[val] {len(roles)} role-rows over "
          f"{roles['date'].nunique()} rebalances")

    memo = {}
    recs, skipped = [], 0
    for _, r in roles.iterrows():
        key = (r["date"], r["stock"])
        if key not in hidx.index:
            skipped += 1
            continue
        hrow = hidx.loc[key]
        if isinstance(hrow, pd.DataFrame):
            hrow = hrow.iloc[0]
        if not np.isfinite(hrow["next_open_gap"]):
            skipped += 1
            continue
        rb = epb.rank_bucket_of(r["rank_pct"])
        vb = hrow["vol_bucket"] if isinstance(hrow["vol_bucket"], str) \
            else None
        ck = (r["date"], rb, vb)
        if ck not in memo:
            memo[ck] = cal.cell(r["date"], rb, vb, strict=True)
        q, n_obs, fb = memo[ck]
        if q is None:
            skipped += 1
            continue
        P = float(hrow["close"])
        A = float(hrow["atr20_pct"])
        op = P * (1 + hrow["next_open_gap"])
        rec = {"date": r["date"], "year": pd.Timestamp(r["date"]).year,
               "stock": r["stock"], "role": r["role"], "vol_bucket": vb,
               "fallback": fb, "n_obs": n_obs,
               "next_open": op,
               "next_high": P * (1 + hrow["next_high_from_close"]),
               "next_low": P * (1 + hrow["next_low_from_close"]),
               "prev_close": P}
        if r["role"] in ("entrant", "incumbent"):
            fresh = r["role"] == "entrant"
            b = epb.entry_bands(P, A, q, fresh=fresh)
            b2 = epb.entry_bands(P, A, q, fresh=True)
            # pre-registered invariant: existing <= fresh aggressiveness
            if not fresh:
                assert b["acceptable_ceiling"] <= b2["acceptable_ceiling"] \
                    + 1e-9 and b["do_not_chase_above"] <= \
                    b2["do_not_chase_above"] + 1e-9
            rec.update(b, band_kind="fresh" if fresh else "existing")
            rec["zone_pos"] = (
                "below_ideal" if op < b["ideal_zone_low"] else
                "inside_ideal" if op <= b["ideal_zone_high"] else
                "to_ceiling" if op <= b["acceptable_ceiling"] else
                "above_ceiling" if op <= b["do_not_chase_above"] else
                "above_chase")
            rec["ref_err_bps"] = abs(op - b["reference"]) / P * 1e4
        else:
            b = epb.sell_bands(P, A, q)
            rec.update(b, band_kind="sell")
            rec["zone_pos"] = (
                "below_floor" if op < b["acceptable_sell_floor"] else
                "floor_to_zone" if op < b["ideal_sell_zone_low"] else
                "inside_ideal" if op <= b["ideal_sell_zone_high"] else
                "above_ideal")
            rec["below_panic"] = op < b["do_not_panic_sell_below"]
            rec["ref_err_bps"] = abs(op - b["sell_reference"]) / P * 1e4
        recs.append(rec)
    ev = pd.DataFrame(recs)
    print(f"[val] evaluated {len(ev)}, skipped {skipped}")

    out_rows = []

    def coverage(sub, label):
        n = len(sub)
        if not n:
            return
        row = {"group": label, "n": n,
               "median_ref_err_bps": round(sub["ref_err_bps"].median(), 1),
               "q75_ref_err_bps": round(sub["ref_err_bps"].quantile(.75), 1),
               "q90_ref_err_bps": round(sub["ref_err_bps"].quantile(.90), 1)}
        for z in ("below_ideal", "inside_ideal", "to_ceiling",
                  "above_ceiling", "above_chase", "below_floor",
                  "floor_to_zone", "above_ideal"):
            pct = (sub["zone_pos"] == z).mean()
            if pct > 0:
                row[f"pct_{z}"] = round(pct, 4)
        if "below_panic" in sub and sub["band_kind"].iloc[0] == "sell":
            row["pct_below_panic"] = round(sub["below_panic"].mean(), 4)
        out_rows.append(row)

    for kind in ("fresh", "existing", "sell"):
        sub = ev[ev["band_kind"] == kind]
        coverage(sub, f"{kind}/ALL")
        for y in sorted(sub["year"].unique()):
            coverage(sub[sub["year"] == y], f"{kind}/{y}")
        for vb in ("LOW", "MED", "HIGH"):
            coverage(sub[sub["vol_bucket"] == vb], f"{kind}/vol_{vb}")
    # conditional next-day high/low after inside-zone vs chase opens
    for kind in ("fresh", "existing"):
        sub = ev[ev["band_kind"] == kind]
        for zp, label in (("inside_ideal", "inside"),
                          ("above_chase", "chase")):
            s = sub[sub["zone_pos"] == zp]
            if len(s):
                out_rows.append({
                    "group": f"{kind}/after_{label}_open", "n": len(s),
                    "med_high_vs_open_bps": round(((s["next_high"] /
                        s["next_open"] - 1) * 1e4).median(), 1),
                    "med_low_vs_open_bps": round(((s["next_low"] /
                        s["next_open"] - 1) * 1e4).median(), 1)})
    res = pd.DataFrame(out_rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "price_band_validation.csv"), index=False)
    print(res.head(30).to_string(index=False))
    print(f"[done] -> {os.path.join(OUT, 'price_band_validation.csv')}")
    return res



# ===================================================== Stage B2 extension
# Legal-domain hard gate, next-open distribution coverage, range-reach
# calibration, do-not-chase outcome analysis, waiting tradeoff. All
# pre-registered in price_domain_methodology.md BEFORE this ran. Same
# expanding-window (< T) rule; daily OHLC path order never interpreted.

import twse_price_domain as tpd  # noqa: E402


def _fwd_lookup(frames, horizon=20):
    out = {}
    for sid, df in frames.items():
        out[sid] = (pd.DatetimeIndex(df["date"].values),
                    df["open"].to_numpy(np.float64),
                    df["low"].to_numpy(np.float64))
    def stats(sid, T):
        if sid not in out:
            return None
        idx, o, lo = out[sid]
        pos = idx.searchsorted(pd.Timestamp(T))
        if pos >= len(idx) or idx[pos] != pd.Timestamp(T) \
                or pos + 1 + horizon >= len(idx):
            return None
        entry = o[pos + 1]
        if not np.isfinite(entry) or entry <= 0:
            return None
        return {"entry_open": entry,
                "fwd_ret": o[pos + 1 + horizon] / entry - 1.0,
                "max_adverse": lo[pos + 1:pos + 2 + horizon].min()
                / entry - 1.0}
    return stats


def b2_main():
    panel, _ = load_panel(PANEL)
    m = merged(panel)
    m["score"] = 0.5 * m["z_tf"] + 0.5 * m["z_mom"]
    frames = _cache_frames()
    hist = epb.build_history(frames,
                             score_panel=m[["date", "stock", "score"]])
    cal = epb.BandCalibrator(hist)
    hidx = hist.set_index(["date", "stock"])
    roles = role_rows(m)
    fwd_stats = _fwd_lookup(frames)
    memo = {}
    violations = 0
    cov_rows, reach_pairs, chase_rows, wait_rows = [], [], [], []
    OPEN_QS = (0.10, 0.25, 0.50, 0.75, 0.90)
    DISCOUNTS = (0.0, 0.005, 0.010, 0.015, 0.020)

    for _, r in roles.iterrows():
        key = (r["date"], r["stock"])
        if key not in hidx.index:
            continue
        hrow = hidx.loc[key]
        if isinstance(hrow, pd.DataFrame):
            hrow = hrow.iloc[0]
        if not np.isfinite(hrow["next_open_gap"]):
            continue
        rb = epb.rank_bucket_of(r["rank_pct"])
        vb = hrow["vol_bucket"] if isinstance(hrow["vol_bucket"], str) \
            else None
        ck = (r["date"], rb, vb)
        if ck not in memo:
            memo[ck] = cal.cell_full(r["date"], rb, vb, strict=True)
        q, n_obs, fb, samples = memo[ck]
        if q is None:
            continue
        P = float(hrow["close"])
        A = float(hrow["atr20_pct"])
        gap = float(hrow["next_open_gap"])
        low_r = float(hrow["next_low_from_close"])
        high_r = float(hrow["next_high_from_close"])
        year = pd.Timestamp(r["date"]).year
        dom = tpd.build_domain(r["stock"], P)
        # ---- open-distribution coverage (actual gap <= q-quantile?)
        cov = {"year": year, "vol_bucket": vb, "role": r["role"],
               "abs_p50_err_bps": abs(gap - q["g0.50"]) * 1e4}
        for p in OPEN_QS:
            cov[f"le_p{int(p * 100)}"] = float(gap <= q[f"g{p:.2f}"])
        cov_rows.append(cov)
        # ---- bands + legal-domain hard gate on every emitted level
        fresh = r["role"] == "entrant"
        if r["role"] in ("entrant", "incumbent"):
            raw = epb.entry_bands(P, A, q, fresh=fresh)
        else:
            raw = epb.sell_bands(P, A, q)
        try:
            lv = epb.clamp_levels(epb.round_levels(raw), dom)
            for v in epb.expected_price_quantiles(q, P, dom).values():
                dom.validate(v, "dist")
        except tpd.PriceDomainValidationError:
            violations += 1
            continue
        # ---- reach calibration pairs (predicted vs realized)
        if r["role"] in ("entrant", "incumbent"):
            for lname in ("reference", "ideal_zone_low",
                          "ideal_zone_high"):
                pred = epb.reach_prob_buy(samples, P, lv[lname])
                real = float(low_r <= lv[lname] / P - 1 + 1e-12)
                if np.isfinite(pred):
                    reach_pairs.append({"side": "BUY", "year": year,
                                        "vol_bucket": vb, "pred": pred,
                                        "real": real})
            # waiting tradeoff at fixed discounts vs reference
            for d in DISCOUNTS:
                level = dom.clamp(lv["reference"] * (1 - d), "buy")
                pred = epb.reach_prob_buy(samples, P, level)
                real = float(low_r <= level / P - 1 + 1e-12)
                wait_rows.append({"discount": d, "year": year,
                                  "pred": pred, "real": real})
            # do-not-chase outcome buckets
            fs = fwd_stats(r["stock"], r["date"])
            op = P * (1 + gap)
            if fs is not None:
                bucket = ("open<=ceiling" if op <= lv["acceptable_ceiling"]
                          else "ceiling<open<=chase"
                          if op <= lv["do_not_chase_above"]
                          else "open>chase")
                chase_rows.append({"bucket": bucket, "year": year,
                                   "kind": "fresh" if fresh else
                                   "existing", **fs})
        else:
            for lname in ("sell_reference", "ideal_sell_zone_high",
                          "acceptable_sell_floor"):
                pred = epb.reach_prob_sell(samples, P, lv[lname])
                real = float(high_r >= lv[lname] / P - 1 - 1e-12)
                if np.isfinite(pred):
                    reach_pairs.append({"side": "SELL", "year": year,
                                        "vol_bucket": vb, "pred": pred,
                                        "real": real})

    # ---------------- outputs
    cov = pd.DataFrame(cov_rows)
    dom_rows = [{"group": "ALL", "n": len(cov), "violations": violations,
                 "med_abs_p50_err_bps":
                 round(cov["abs_p50_err_bps"].median(), 1),
                 **{f"cov_p{int(p*100)}":
                    round(cov[f"le_p{int(p*100)}"].mean(), 4)
                    for p in OPEN_QS}}]
    for gcol, pre in (("year", "y"), ("vol_bucket", "vol")):
        for gv, sub in cov.groupby(gcol):
            dom_rows.append(
                {"group": f"{pre}_{gv}", "n": len(sub),
                 "violations": np.nan,
                 "med_abs_p50_err_bps":
                 round(sub["abs_p50_err_bps"].median(), 1),
                 **{f"cov_p{int(p*100)}":
                    round(sub[f"le_p{int(p*100)}"].mean(), 4)
                    for p in OPEN_QS}})
    pd.DataFrame(dom_rows).to_csv(
        os.path.join(OUT, "price_domain_validation.csv"), index=False)

    rp = pd.DataFrame(reach_pairs)
    rr_rows = []
    for (side,), sub in rp.groupby(["side"]):
        brier = float(((sub["pred"] - sub["real"]) ** 2).mean())
        rr_rows.append({"group": f"{side}/ALL", "n": len(sub),
                        "brier": round(brier, 4),
                        "mean_pred": round(sub["pred"].mean(), 4),
                        "mean_real": round(sub["real"].mean(), 4)})
        sub = sub.copy()
        sub["bucket"] = np.clip((sub["pred"] * 10).astype(int), 0, 9)
        for b, s2 in sub.groupby("bucket"):
            rr_rows.append({"group": f"{side}/decile_{b}", "n": len(s2),
                            "brier": np.nan,
                            "mean_pred": round(s2["pred"].mean(), 4),
                            "mean_real": round(s2["real"].mean(), 4)})
        for y, s2 in sub.groupby("year"):
            rr_rows.append({"group": f"{side}/{y}", "n": len(s2),
                            "brier": round(float(((s2["pred"] -
                                s2["real"]) ** 2).mean()), 4),
                            "mean_pred": round(s2["pred"].mean(), 4),
                            "mean_real": round(s2["real"].mean(), 4)})
        for vb, s2 in sub.groupby("vol_bucket"):
            rr_rows.append({"group": f"{side}/vol_{vb}", "n": len(s2),
                            "brier": round(float(((s2["pred"] -
                                s2["real"]) ** 2).mean()), 4),
                            "mean_pred": round(s2["pred"].mean(), 4),
                            "mean_real": round(s2["real"].mean(), 4)})
    wt = pd.DataFrame(wait_rows)
    for d, s2 in wt.groupby("discount"):
        rr_rows.append({"group": f"WAIT/discount_{d:.1%}", "n": len(s2),
                        "brier": np.nan,
                        "mean_pred": round(s2["pred"].mean(), 4),
                        "mean_real": round(s2["real"].mean(), 4)})
    pd.DataFrame(rr_rows).to_csv(
        os.path.join(OUT, "range_reach_validation.csv"), index=False)

    ch = pd.DataFrame(chase_rows)
    ch_rows = []
    for (kind, bucket), sub in ch.groupby(["kind", "bucket"]):
        f = sub["fwd_ret"]
        sh = (f.mean() / (f.std(ddof=1) + 1e-12) * np.sqrt(252 / 20)
              if len(f) > 5 else np.nan)
        ch_rows.append({"kind": kind, "bucket": bucket, "n": len(sub),
                        "mean_fwd_pct": round(f.mean() * 100, 2),
                        "median_fwd_pct": round(f.median() * 100, 2),
                        "hit_rate": round(float((f > 0).mean()), 3),
                        "sharpe_ann_approx": round(float(sh), 2)
                        if np.isfinite(sh) else np.nan,
                        "median_max_adverse_pct":
                        round(sub["max_adverse"].median() * 100, 2)})
    pd.DataFrame(ch_rows).to_csv(
        os.path.join(OUT, "do_not_chase_outcome_analysis.csv"),
        index=False)
    print(f"[b2] domain violations: {violations} (hard gate: 0 required)")
    print(pd.DataFrame(dom_rows).head(10).to_string(index=False))
    print(pd.DataFrame(rr_rows).head(20).to_string(index=False))
    print(pd.DataFrame(ch_rows).to_string(index=False))
    return violations


if __name__ == "__main__":
    main()
    b2_main()
