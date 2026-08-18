"""v16 Stage B — USER NEXT-SESSION ACTION PLAN (the nightly human report).

Separate operational layer on top of (never replacing) the paper-trading
decision book:

    MODEL DECISION -> ACTUAL HOLDINGS -> USER ACTION -> NIGHT-BEFORE
    PRICE BAND -> NEXT-SESSION ACTION PLAN -> manual review

- model_action is preserved verbatim; user_action uses the Stage-A
  vocabulary from research/holdings.py (OPEN_SHORT does not exist).
- Price bands implement the pre-registered methodology
  (price_band_methodology.md) via research/execution_price_bands.py;
  they are conditional research estimates, not guaranteed fills.
- Reads everything READ-ONLY; writes only under reports/user_actions/
  (gitignored). No orders, no broker APIs, no scheduling.

Usage:
  python research/user_next_session_plan.py
  python research/user_next_session_plan.py --date YYYY-MM-DD
  python research/user_next_session_plan.py --holdings my_holdings.csv
"""

import argparse
import datetime as _dt
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import holdings as hold  # noqa: E402
import execution_price_bands as epb  # noqa: E402
import twse_price_domain as tpd  # noqa: E402
import user_holdings_overlay as uho  # noqa: E402

try:
    from data import SECTOR_MAP
except Exception:
    SECTOR_MAP = {}

OUT_DIR = os.path.join(ROOT, "reports", "user_actions")
TIMING_VALIDATION = "NEXT_OPEN_TIMING_VALIDATED"   # f6d629b, Tasks 1-2
STRATEGY = "blend50_band10"
BAND_COLS = [
    "reference", "ideal_zone_low", "ideal_zone_high", "acceptable_ceiling",
    "do_not_chase_above", "risk_review_below", "sell_reference",
    "ideal_sell_zone_low", "ideal_sell_zone_high", "acceptable_sell_floor",
    "do_not_panic_sell_below", "urgent_risk_review_below",
    "no_action_zone_low", "no_action_zone_high", "review_below",
    "review_above", "cover_reference", "cover_zone_low", "cover_zone_high",
    "risk_review_above",
]
DOMAIN_COLS = [
    "auction_reference_price", "auction_reference_source",
    "auction_reference_confidence", "legal_limit_down", "legal_limit_up",
    "price_domain_status", "distance_to_limit_up_pct",
    "distance_to_limit_down_pct",
    "expected_open_p10", "expected_open_p25", "expected_open_p50",
    "expected_open_p75", "expected_open_p90",
    "expected_low_p25", "expected_low_p50", "expected_low_p75",
    "expected_high_p25", "expected_high_p50", "expected_high_p75",
    "buy_reference_reach_probability", "ideal_low_reach_probability",
    "sell_reference_reach_probability", "p_open_above_do_not_chase",
    "p_open_below_panic_level", "range_reach_data_quality",
    "range_reach_confidence", "reach_drift_recent",
]
CSV_COLS = [
    "symbol", "sector", "signal_date", "intended_execution_date",
    "model_action", "model_rank", "model_score", "model_target_weight",
    "position_side", "position_qty", "avg_cost", "market_value_abs",
    "my_long_cmp_weight", "user_action", "user_action_priority",
    "signal_freshness", "model_position_age_sessions", "execution_posture",
    "previous_close", "atr20_pct",
    *BAND_COLS, *DOMAIN_COLS,
    "band_sample_count", "band_fallback_level", "user_action_reason",
    "book_stale",
]
BUY_SIDE_ACTIONS = ("OPEN_LONG_NEW_SIGNAL", "OPEN_LONG_EXISTING_TARGET",
                    "ADD_LONG", "WATCH_LONG")
SELL_SIDE_ACTIONS = ("REDUCE_LONG", "EXIT_LONG")


def next_twse_session(date_str):
    """Next weekday after date_str. LIMITATION (documented): no TWSE
    holiday calendar exists in this repo — a holiday shifts the true next
    session later; the plan header states the intended session as
    NEXT_TWSE_SESSION and this date as the best weekday estimate."""
    d = _dt.date.fromisoformat(str(date_str)[:10])
    d += _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return d.isoformat()


def model_position_age(root, sym, date):
    """Consecutive sessions (dated paper books <= date) the symbol has
    been in the model book, counting back from `date`. 0 = not in the
    dated book; count reflects available book history only."""
    bdir = os.path.join(root, "reports", "paper_trading", "books")
    dates = sorted((os.path.basename(p)[:10] for p in
                    glob.glob(os.path.join(bdir, f"*_{STRATEGY}.csv"))
                    if os.path.basename(p)[:10] <= date), reverse=True)
    age = 0
    for d in dates:
        try:
            bk = pd.read_csv(os.path.join(bdir, f"{d}_{STRATEGY}.csv"),
                             dtype={"stock": str})
        except Exception:
            break
        if sym in set(bk["stock"]):
            age += 1
        else:
            break
    return age


def signal_freshness(model_action, position_side, in_universe):
    if not in_universe:
        return "NO_MODEL_OPINION"
    if position_side in ("LONG", "SHORT"):
        return "CURRENT_USER_POSITION"
    if model_action == "BUY":
        return "FRESH_ENTRY"
    if model_action in ("HOLD", "REDUCE"):
        return "EXISTING_MODEL_POSITION"
    if model_action == "WATCH":
        return "WATCH_ONLY"
    return "WATCH_ONLY" if model_action == "" else "EXISTING_MODEL_POSITION"


def build_plan(root, holdings_path, date=None, use_panel=True):
    books = uho.available_books(root)
    if STRATEGY not in books:
        sys.exit(f"no {STRATEGY} decision books found")
    if date is None:
        date = books[STRATEGY][-1]
    elif date not in books[STRATEGY]:
        sys.exit(f"no {STRATEGY} book for {date}; latest "
                 f"{books[STRATEGY][-1]}")
    book = uho.load_book(root, STRATEGY, date)
    universe, universe_src, rank_pct_map = uho.model_universe(root)

    # holdings (absent file -> model-only plan, USER_POSITION_UNKNOWN)
    warnings, positions = [], pd.DataFrame(
        columns=["symbol", "position_side", "position_qty", "avg_cost",
                 "current_price", "current_value", "account", "notes",
                 "n_lots", "both_sides"])
    user_position_known = os.path.isfile(holdings_path)
    if user_position_known:
        lots, w1 = hold.load_lots(holdings_path)
        positions, w2 = hold.aggregate_positions(lots)
        warnings = w1 + w2

    # cache frames (root-relative, read-only, date-deduped) for the whole
    # universe (calibration) + held symbols outside it (e.g. 0050)
    frames = {}
    for sym in sorted(universe | set(positions["symbol"])):
        p = os.path.join(root, "research", "data_cache", f"{sym}.csv")
        if not os.path.isfile(p):
            continue
        try:
            frames[sym] = (pd.read_csv(p, parse_dates=["date"])
                           .sort_values("date")
                           .drop_duplicates("date", keep="last")
                           .reset_index(drop=True))
        except Exception:
            pass

    # freshness metadata
    all_dates = sorted({d for df in frames.values()
                        for d in df["date"].dt.strftime("%Y-%m-%d")})
    newer = [d for d in all_dates if d > date]
    book_age = len(newer)
    book_stale = book_age > 0

    # calibration history: score panel (frozen BR walkforward) gives
    # historical rank buckets where covered; try/skip if absent (tests /
    # fixture repos fall back to vol/global cells).
    panel = None
    if use_panel:
        try:
            from transformer_hybrid import load_panel
            panel, _ = load_panel("SCHED_BEAR_A8_seeds7_full")
        except Exception:
            pass
    hist = epb.build_history(frames, score_panel=panel)
    cal = epb.BandCalibrator(hist)
    feat_asof = {}
    for sid, df in frames.items():
        f = epb.symbol_features(df)
        f = f[f["date"] <= pd.Timestamp(date)]
        if len(f):
            feat_asof[sid] = f.iloc[-1]
    # cross-sectional vol terciles on the signal date
    vols = pd.Series({s: f["vol20"] for s, f in feat_asof.items()}).dropna()
    v1, v2 = vols.quantile(1 / 3), vols.quantile(2 / 3)

    def vol_bucket(sym):
        v = feat_asof.get(sym, {}).get("vol20", np.nan) \
            if sym in feat_asof else np.nan
        if pd.isna(v):
            return None
        return "LOW" if v <= v1 else "MED" if v <= v2 else "HIGH"

    # exposure denominators (Stage-A conventions)
    pos = positions.copy()
    vals = []
    for _, h in pos.iterrows():
        price = h["current_price"]
        if pd.isna(price):
            fa = feat_asof.get(h["symbol"])
            price = float(fa["close"]) if fa is not None else np.nan
        v = h["current_value"]
        if pd.isna(v) and pd.notna(h["position_qty"]) and pd.notna(price):
            v = h["position_qty"] * price
        vals.append(v)
    pos["market_value_abs"] = vals
    exp = hold.exposure_metrics(pos) if len(pos) else {
        "gross_long_value": 0.0, "gross_short_value": 0.0,
        "gross_exposure": 0.0, "net_exposure": 0.0}
    gl = exp["gross_long_value"]
    pos_by_symside = {(r["symbol"], r["position_side"]): r
                      for _, r in pos.iterrows()}

    # -------- symbol union: positions + book rows (all actions)
    syms = list(dict.fromkeys(list(pos["symbol"]) + list(book.index)))
    n_uni = max(len(universe), 1)
    rows, curve_rows = [], []
    for sym in syms:
        b = book.loc[sym] if sym in book.index else None
        act = str(b["action"]) if b is not None else ""
        tgt = float(b["target_weight"]) if b is not None else np.nan
        in_uni = sym in universe and sym in frames
        sides = [s for (s2, s) in pos_by_symside if s2 == sym] or ["NONE"]
        for side in sides:
            p = pos_by_symside.get((sym, side))
            qty = p["position_qty"] if p is not None else np.nan
            mv = p["market_value_abs"] if p is not None else np.nan
            cmp_w = (mv / gl if p is not None and side == "LONG"
                     and pd.notna(mv) and gl > 0 else np.nan)
            conflict = bool(p["both_sides"]) if p is not None else False
            rp = None
            if b is not None and pd.notna(b.get("rank")):
                rp = float(b["rank"]) / n_uni
            elif sym in rank_pct_map:
                rp = float(rank_pct_map[sym])
            if side == "UNKNOWN":
                ua, pri, reason = ("NO_ACTION", "INFO",
                                   "shares unparseable — review manually")
            else:
                ua, pri, reason = hold.map_user_action(
                    position_side=side if side in ("LONG", "SHORT")
                    else "NONE",
                    model_action=act, model_target=tgt, in_universe=in_uni,
                    in_book=b is not None, cmp_weight=cmp_w,
                    universe_rank_pct=rp, conflict=conflict,
                    material=(pd.notna(mv) and exp["gross_exposure"] > 0
                              and mv / exp["gross_exposure"] > 0.05))
            fresh = signal_freshness(act, side, in_uni)
            fa = feat_asof.get(sym)
            prev_close = float(fa["close"]) if fa is not None else np.nan
            atr = float(fa["atr20_pct"]) if fa is not None else np.nan
            q = n_obs = samples = None
            fb = "N/A"
            if in_uni and fa is not None:
                q, n_obs, fb, samples = cal.cell_full(
                    date, epb.rank_bucket_of(rp), vol_bucket(sym),
                    strict=False)
            bands = epb.bands_for_action(ua, prev_close, atr, q) \
                if q is not None else {}
            # B2: legal price domain (normal-day assumption; ETFs and
            # unknown types get UNKNOWN — no fabricated limits) + hard
            # clamp/validation of every emitted level
            sec_type = ("etf" if SECTOR_MAP.get(sym) == "etf" else
                        "stock" if in_uni else
                        SECTOR_MAP.get(sym, "unknown"))
            dom = tpd.build_domain(sym, prev_close, security_type=sec_type)
            if bands:
                bands = epb.clamp_levels(bands, dom)
            dq = ("NA" if q is None else
                  "DEGRADED" if fb == "GLOBAL" else "OK")
            dist = (epb.expected_price_quantiles(q, prev_close, dom)
                    if q is not None and np.isfinite(prev_close)
                    and ua not in ("NO_MODEL_OPINION",) else {})
            for v_ in dist.values():
                dom.validate(v_, context="expected_quantile")
            reach_buy_ref = reach_ideal_low = reach_sell_ref = np.nan
            p_chase = p_panic = reach_drift = np.nan
            reach_conf = "NA"
            if samples is not None and np.isfinite(prev_close):
                if ua in BUY_SIDE_ACTIONS and bands:
                    reach_buy_ref = epb.reach_prob_buy(
                        samples, prev_close, bands.get("reference"))
                    reach_ideal_low = epb.reach_prob_buy(
                        samples, prev_close, bands.get("ideal_zone_low"))
                    p_chase = epb.prob_open_beyond(
                        samples, prev_close,
                        bands.get("do_not_chase_above"), "above")
                    reach_conf, reach_drift = epb.reach_confidence(
                        samples, prev_close, bands.get("reference"),
                        "buy", fb)
                if ua in SELL_SIDE_ACTIONS and bands:
                    reach_sell_ref = epb.reach_prob_sell(
                        samples, prev_close, bands.get("sell_reference"))
                    p_panic = epb.prob_open_beyond(
                        samples, prev_close,
                        bands.get("do_not_panic_sell_below"), "below")
                    reach_conf, reach_drift = epb.reach_confidence(
                        samples, prev_close, bands.get("sell_reference"),
                        "sell", fb)
            row = {
                "symbol": sym, "sector": (b["sector"] if b is not None
                                          else ""),
                "signal_date": date,
                "intended_execution_date": next_twse_session(date),
                "model_action": act,
                "model_rank": b["rank"] if b is not None else np.nan,
                "model_score": b["score"] if b is not None else np.nan,
                "model_target_weight": tgt,
                "position_side": side if side != "NONE" else "",
                "position_qty": qty,
                "avg_cost": p["avg_cost"] if p is not None else np.nan,
                "market_value_abs": mv,
                "my_long_cmp_weight": cmp_w,
                "user_action": ua, "user_action_priority": pri,
                "signal_freshness": fresh,
                "model_position_age_sessions": (
                    model_position_age(root, sym, date)
                    if fresh in ("EXISTING_MODEL_POSITION",
                                 "CURRENT_USER_POSITION") and in_uni
                    else np.nan),
                "execution_posture": epb.posture_for(ua),
                "previous_close": prev_close, "atr20_pct": atr,
                "band_sample_count": n_obs if n_obs else np.nan,
                "band_fallback_level": fb,
                "user_action_reason": reason,
                "book_stale": book_stale,
                "auction_reference_price": dom.reference_price,
                "auction_reference_source": dom.reference_source,
                "auction_reference_confidence": dom.reference_confidence,
                "legal_limit_down": dom.legal_limit_down,
                "legal_limit_up": dom.legal_limit_up,
                "price_domain_status": dom.price_domain_status,
                "distance_to_limit_up_pct": (
                    dom.legal_limit_up / prev_close - 1
                    if dom.known() and np.isfinite(prev_close) else np.nan),
                "distance_to_limit_down_pct": (
                    prev_close / dom.legal_limit_down - 1
                    if dom.known() and np.isfinite(prev_close) else np.nan),
                "buy_reference_reach_probability": reach_buy_ref,
                "ideal_low_reach_probability": reach_ideal_low,
                "sell_reference_reach_probability": reach_sell_ref,
                "p_open_above_do_not_chase": p_chase,
                "p_open_below_panic_level": p_panic,
                "range_reach_data_quality": dq,
                "range_reach_confidence": reach_conf,
                "reach_drift_recent": reach_drift,
            }
            for c in BAND_COLS:
                row[c] = bands.get(c, np.nan)
            for c in [c for c in DOMAIN_COLS if c.startswith("expected_")]:
                row[c] = dist.get(c, np.nan)
            # reach curve rows (full curve -> gitignored CSV; md shows 3-5)
            if samples is not None and bands and np.isfinite(prev_close):
                if ua in BUY_SIDE_ACTIONS:
                    cons = prev_close * (1 + (q["g0.10"] if q else np.nan))
                    cons = dom.clamp(cons, "buy")
                    for lname, lv in (
                            ("ideal_zone_high", bands.get("ideal_zone_high")),
                            ("buy_reference", bands.get("reference")),
                            ("ideal_zone_low", bands.get("ideal_zone_low")),
                            ("conservative_g10", cons)):
                        pr = epb.reach_prob_buy(samples, prev_close, lv)
                        curve_rows.append(
                            {"symbol": sym, "side": "BUY", "level": lname,
                             "price": lv,
                             "discount_vs_reference_pct": (
                                 lv / bands.get("reference") - 1
                                 if bands.get("reference") else np.nan),
                             "range_reach_probability": pr,
                             "non_reach_probability": (1 - pr if
                                                       np.isfinite(pr)
                                                       else np.nan)})
                if ua in SELL_SIDE_ACTIONS:
                    for lname, lv in (
                            ("sell_reference", bands.get("sell_reference")),
                            ("ideal_sell_zone_high",
                             bands.get("ideal_sell_zone_high")),
                            ("acceptable_sell_floor",
                             bands.get("acceptable_sell_floor"))):
                        pr = epb.reach_prob_sell(samples, prev_close, lv)
                        curve_rows.append(
                            {"symbol": sym, "side": "SELL", "level": lname,
                             "price": lv,
                             "discount_vs_reference_pct": np.nan,
                             "range_reach_probability": pr,
                             "non_reach_probability": (1 - pr if
                                                       np.isfinite(pr)
                                                       else np.nan)})
            rows.append(row)
    plan = pd.DataFrame(rows)
    assert not set(plan["user_action"]) - set(hold.USER_ACTIONS), \
        "user_action outside the Stage-A vocabulary"
    assert not {"OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"} & \
        set(plan["user_action"]), "short-creation action emitted — forbidden"
    meta = {
        "signal_date": date, "data_asof": f"{date} CLOSE",
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "intended_execution_date": next_twse_session(date),
        "intended_execution_session": "NEXT_TWSE_SESSION",
        "timing_validation": TIMING_VALIDATION,
        "book_age_sessions": book_age, "book_stale": book_stale,
        "universe_source": universe_src,
        "user_position_known": user_position_known,
        "exposure": exp, "warnings": warnings,
        "curve_rows": curve_rows,
    }
    return plan, meta


# ------------------------------------------------------------------ report

def _fmt(v, pat="{:.2f}"):
    return "n/a" if v is None or (isinstance(v, float) and not
                                  np.isfinite(v)) else pat.format(v)


def _pct(v):
    return "n/a" if v is None or (isinstance(v, float) and not
                                  np.isfinite(v)) else f"{v:.0%}"


def _domain_lines(r):
    if r["price_domain_status"] == "NORMAL_DAY_ASSUMPTION" and \
            np.isfinite(r["legal_limit_up"]):
        return [f"  auction reference: "
                f"{_fmt(r['auction_reference_price'])} "
                f"({r['auction_reference_source']}, confidence "
                f"{r['auction_reference_confidence']}; normal-day "
                "assumption — ex-date/special references not detectable)",
                f"  legal range: {_fmt(r['legal_limit_down'])} – "
                f"{_fmt(r['legal_limit_up'])}"]
    return ["  standard-limit estimate unavailable — special reference "
            "may apply"]


def _open_dist_line(r):
    if np.isfinite(r.get("expected_open_p25", np.nan)):
        return [f"  expected next open: {_fmt(r['expected_open_p25'])} – "
                f"{_fmt(r['expected_open_p75'])} (median "
                f"{_fmt(r['expected_open_p50'])}; p10/p90 "
                f"{_fmt(r['expected_open_p10'])}/"
                f"{_fmt(r['expected_open_p90'])})"]
    return []


def _entry_block(r):
    lines = [f"  previous close: {_fmt(r['previous_close'])}"]
    lines += _domain_lines(r) + _open_dist_line(r)
    if np.isfinite(r.get("expected_low_p25", np.nan)):
        lines.append(f"  expected next-session low: p25 "
                     f"{_fmt(r['expected_low_p25'])} / p50 "
                     f"{_fmt(r['expected_low_p50'])} / p75 "
                     f"{_fmt(r['expected_low_p75'])}")
    lines += [f"  provisional buy reference: {_fmt(r['reference'])}",
              f"  ideal buy zone: {_fmt(r['ideal_zone_low'])}"
              f"–{_fmt(r['ideal_zone_high'])}",
              f"  acceptable ceiling: {_fmt(r['acceptable_ceiling'])}",
              f"  above preferred execution range: "
              f"{_fmt(r['do_not_chase_above'])} (execution-quality "
              "threshold: an unusually expensive entry vs the historical "
              "next-open distribution — B2 analysis did NOT show the "
              "validated 20-session signal fails above it)",
              f"  risk review below: {_fmt(r['risk_review_below'])}"]
    if np.isfinite(r.get("buy_reference_reach_probability", np.nan)):
        nr = (1 - r["ideal_low_reach_probability"]
              if np.isfinite(r["ideal_low_reach_probability"]) else np.nan)
        lines += [
            "  historical next-session range reach (daily LOW at or "
            "below the level — NOT a fill probability):",
            f"    buy reference or lower: "
            f"{_pct(r['buy_reference_reach_probability'])}",
            f"    ideal-zone low or lower: "
            f"{_pct(r['ideal_low_reach_probability'])} "
            f"(T+1_RANGE_DID_NOT_REACH_LEVEL: {_pct(nr)})",
            f"  historical probability next open exceeds the preferred "
            f"execution range: {_pct(r['p_open_above_do_not_chase'])}"]
        if r.get("range_reach_confidence") == "DEGRADED":
            d = r.get("reach_drift_recent", np.nan)
            word = ("under-realized" if np.isfinite(d) and d < 0
                    else "over-realized" if np.isfinite(d)
                    else "drifted from")
            lines.append(
                f"  reach confidence DEGRADED: recent comparable "
                f"observations have {word} the full-sample range-reach "
                "estimate; treat the percentages as lower confidence")
    lines.append(f"  band sample: "
                 f"{_fmt(r['band_sample_count'], '{:.0f}')} obs, "
                 f"fallback {r['band_fallback_level']} "
                 f"(reach data quality {r['range_reach_data_quality']}, "
                 f"confidence {r.get('range_reach_confidence', 'NA')})")
    return lines


def _sell_block(r):
    lines = [f"  previous close: {_fmt(r['previous_close'])}"]
    lines += _domain_lines(r) + _open_dist_line(r)
    if np.isfinite(r.get("expected_high_p25", np.nan)):
        lines.append(f"  expected next-session high: p25 "
                     f"{_fmt(r['expected_high_p25'])} / p50 "
                     f"{_fmt(r['expected_high_p50'])} / p75 "
                     f"{_fmt(r['expected_high_p75'])}")
    lines += [f"  provisional sell reference: {_fmt(r['sell_reference'])}",
              f"  ideal sell zone: {_fmt(r['ideal_sell_zone_low'])}"
              f"–{_fmt(r['ideal_sell_zone_high'])}",
              f"  acceptable floor: {_fmt(r['acceptable_sell_floor'])}",
              f"  do not panic-sell below: "
              f"{_fmt(r['do_not_panic_sell_below'])}",
              f"  URGENT risk review below: "
              f"{_fmt(r['urgent_risk_review_below'])}"]
    if np.isfinite(r.get("sell_reference_reach_probability", np.nan)):
        lines += [
            "  historical next-session range reach (daily HIGH at or "
            "above the level — NOT a fill probability):",
            f"    sell reference or higher: "
            f"{_pct(r['sell_reference_reach_probability'])}",
            f"  historical probability next open gaps below the panic "
            f"level: {_pct(r['p_open_below_panic_level'])}",
            "  note: daily bars cannot show whether a rebound happened "
            "before further downside — no path order is implied"]
        if r.get("range_reach_confidence") == "DEGRADED":
            d = r.get("reach_drift_recent", np.nan)
            word = ("under-realized" if np.isfinite(d) and d < 0
                    else "over-realized" if np.isfinite(d)
                    else "drifted from")
            lines.append(
                f"  reach confidence DEGRADED: recent comparable "
                f"observations have {word} the full-sample sell-side "
                "range-reach estimate; treat the percentage as lower "
                "confidence")
    lines.append(f"  band sample: "
                 f"{_fmt(r['band_sample_count'], '{:.0f}')} obs, "
                 f"fallback {r['band_fallback_level']} "
                 f"(reach data quality {r['range_reach_data_quality']}, "
                 f"confidence {r.get('range_reach_confidence', 'NA')})")
    return lines


def write_report(plan, meta, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    date = meta["signal_date"]
    csv_p = os.path.join(out_dir, f"{date}_next_session_action_plan.csv")
    md_p = os.path.join(out_dir, f"{date}_next_session_action_plan.md")
    out = plan.copy()
    out["my_long_cmp_weight"] = out["my_long_cmp_weight"].round(5)
    out[CSV_COLS].to_csv(csv_p, index=False)

    fresh_txt = ("STALE" if meta["book_stale"] else "FRESH")
    md = [f"# AI-Quant Next Session Action Plan — {date}", "",
          f"Signal based on: **{date} close** · Generated: "
          f"{meta['generated_at']} · Intended execution: "
          f"**{meta['intended_execution_date']} (NEXT TWSE SESSION)**",
          f"Timing validation: **{meta['timing_validation']}** · "
          f"Book freshness: **{fresh_txt}**",
          "", "Reference prices are conditional research estimates, not "
          "guaranteed fills. This plan never generates orders. "
          "NO VALIDATED OPEN-SHORT MODEL EXISTS — no short entries are "
          "ever suggested. TWSE tick rounding is implemented from public "
          "documentation and unverified against a live feed. All "
          "range-reach percentages are DAILY-RANGE statistics (the "
          "next-day low/high reached the level), never fill "
          "probabilities. Legal ranges use the normal-day assumption "
          "(reference = previous close); ex-right/ex-dividend or other "
          "special-reference days are not detectable from this repo's "
          "data.", ""]
    if (plan["range_reach_confidence"] == "DEGRADED").any():
        md += ["> **NOTE — reach-calibration confidence DEGRADED on some "
               "rows**: recent comparable observations drifted from the "
               "full-sample range-reach estimates (regime effect); the "
               "affected rows say so inline — treat their percentages as "
               "lower confidence. Thresholds are unchanged.", ""]
    if meta["book_stale"]:
        md += ["> **WARNING — STALE BOOK**: the newest decision book "
               f"({date}) is {meta['book_age_sessions']} session(s) older "
               "than the newest cached EOD data. This plan repeats an old "
               "signal; run daily_ops before trusting it.", ""]
    if not meta["user_position_known"]:
        md += ["> **USER_POSITION_UNKNOWN**: my_holdings.csv not found — "
               "model-side plan only; position-aware actions unavailable.",
               ""]
    for w in meta["warnings"]:
        md.append(f"> holdings warning: {w}")
    if meta["warnings"]:
        md.append("")
    e = meta["exposure"]
    if e["gross_exposure"]:
        md += [f"Portfolio: gross {e['gross_exposure']:,.0f} (long "
               f"{e['gross_long_value']:,.0f} / short "
               f"{e['gross_short_value']:,.0f}, net "
               f"{e['net_exposure']:,.0f})", ""]

    def sect(title, rows_md):
        md.extend(["", f"## {title}", ""])
        md.extend(rows_md if rows_md else ["None."])

    hi = plan[(plan["user_action_priority"] == "HIGH")
              & (plan["user_action"] != "NO_MODEL_OPINION")]
    rows_md = []
    for _, r in hi.iterrows():
        rows_md.append(f"- **{r['symbol']}** → {r['user_action']} "
                       f"[{r['user_action_priority']}] "
                       f"({r['execution_posture']}): "
                       f"{r['user_action_reason']}")
    sect("1. HIGH PRIORITY", rows_md)

    WAIT_NOTE = ("Waiting for a lower price increases price quality but "
                 "increases T+1_RANGE_DID_NOT_REACH_LEVEL probability — "
                 "the next session's range may not return to the level; "
                 "entry later in the 20-session holding horizon remains "
                 "possible. 'Above preferred execution range' is an "
                 "execution-quality threshold, not an alpha-validity "
                 "level.")

    rows_md = []
    for _, r in plan[plan["user_action"] == "OPEN_LONG_NEW_SIGNAL"
                     ].iterrows():
        rows_md += [f"### {r['symbol']} — OPEN_LONG_NEW_SIGNAL", "",
                    f"  model: BUY · target "
                    f"{_fmt(r['model_target_weight'], '{:.1%}')} · rank "
                    f"{_fmt(r['model_rank'], '{:.0f}')} · freshness "
                    f"FRESH_ENTRY · posture {r['execution_posture']}",
                    "  actual: no position",
                    *_entry_block(r), ""]
    if rows_md:
        rows_md.insert(0, WAIT_NOTE + "\n")
    sect("2. NEW LONG SIGNALS (fresh model entries)", rows_md)

    rows_md = []
    for _, r in plan[plan["user_action"] == "OPEN_LONG_EXISTING_TARGET"
                     ].iterrows():
        age = _fmt(r["model_position_age_sessions"], "{:.0f}")
        rows_md += [f"### {r['symbol']} — OPEN_LONG_EXISTING_TARGET", "",
                    f"  model: {r['model_action']} · target "
                    f"{_fmt(r['model_target_weight'], '{:.1%}')} · model "
                    f"position age {age} session(s) · priority "
                    f"{r['user_action_priority']}",
                    "  actual: no position — the model already holds this; "
                    "entry rules are deliberately MORE conservative than a "
                    "fresh BUY",
                    *_entry_block(r), ""]
    if rows_md:
        rows_md.insert(0, WAIT_NOTE + "\n")
    sect("3. MODEL POSITIONS I DO NOT OWN (standing targets)", rows_md)

    held_long = plan[(plan["position_side"] == "LONG")
                     & plan["user_action"].isin(("HOLD_LONG", "ADD_LONG"))]
    rows_md = []
    for _, r in held_long.iterrows():
        rows_md += [f"### {r['symbol']} — {r['user_action']}", "",
                    f"  LONG {_fmt(r['position_qty'], '{:.0f}')} · avg cost "
                    f"{_fmt(r['avg_cost'])} · latest close "
                    f"{_fmt(r['previous_close'])} · my long weight "
                    f"{_fmt(r['my_long_cmp_weight'], '{:.1%}')} vs target "
                    f"{_fmt(r['model_target_weight'], '{:.1%}')}"]
        if r["user_action"] == "HOLD_LONG":
            rows_md += _domain_lines(r) + _open_dist_line(r)
            rows_md += [f"  no-action zone: "
                        f"{_fmt(r['no_action_zone_low'])}–"
                        f"{_fmt(r['no_action_zone_high'])}",
                        f"  review below: {_fmt(r['review_below'])} · "
                        f"review above: {_fmt(r['review_above'])}",
                        "  (no buy/sell price manufactured — no "
                        "transaction is required while aligned)"]
        else:
            rows_md += _entry_block(r)
        rows_md.append("")
    sect("4. ACTUAL LONG POSITIONS (hold / add)", rows_md)

    rows_md = []
    for _, r in plan[(plan["position_side"] == "LONG") & plan["user_action"]
                     .isin(("REDUCE_LONG", "EXIT_LONG"))].iterrows():
        kind = ("RISK EXIT" if r["user_action"] == "EXIT_LONG"
                else "SELL INTO STRENGTH")
        rows_md += [f"### {r['symbol']} — {r['user_action']} ({kind})", "",
                    f"  LONG {_fmt(r['position_qty'], '{:.0f}')} · avg cost "
                    f"{_fmt(r['avg_cost'])} · "
                    f"{r['user_action_reason']}",
                    *_sell_block(r), ""]
    sect("5. REDUCE / EXIT (actual positions only)", rows_md)

    rows_md = []
    for _, r in plan[plan["position_side"] == "SHORT"].iterrows():
        rows_md += [f"### {r['symbol']} — {r['user_action']}", "",
                    f"  SHORT {_fmt(r['position_qty'], '{:.0f}')} · avg "
                    f"short price {_fmt(r['avg_cost'])} · "
                    f"{r['user_action_reason']}",
                    f"  cover reference: {_fmt(r['cover_reference'])} "
                    f"(zone {_fmt(r['cover_zone_low'])}–"
                    f"{_fmt(r['cover_zone_high'])})",
                    f"  risk review above: {_fmt(r['risk_review_above'])}",
                    ""]
    if rows_md:
        rows_md.insert(0, "**NO VALIDATED OPEN-SHORT MODEL EXISTS** — "
                          "these are risk-tracked existing positions "
                          "only.\n")
    sect("6. ACTUAL SHORT POSITIONS", rows_md)

    rows_md = []
    for _, r in plan[plan["user_action"].isin(("WATCH_LONG",
                                               "WATCH_NEUTRAL"))
                     ].iterrows():
        rows_md += [f"### {r['symbol']} — {r['user_action']}", "",
                    *_domain_lines(r), *_open_dist_line(r),
                    f"  watch entry reference: {_fmt(r['reference'])} · "
                    f"watch zone {_fmt(r['ideal_zone_low'])}–"
                    f"{_fmt(r['ideal_zone_high'])}",
                    f"  above preferred execution range: "
                    f"{_fmt(r['do_not_chase_above'])} · invalidation/"
                    f"review below: {_fmt(r['risk_review_below'])}"]
        if np.isfinite(r.get("buy_reference_reach_probability", np.nan)):
            rows_md.append(
                f"  historical comparable sessions reached the watch "
                f"reference or lower "
                f"{_pct(r['buy_reference_reach_probability'])} of the "
                "time (daily-range reach, not a fill probability)")
        rows_md.append("")
    sect("7. WATCHLIST (every row has a price answer)", rows_md)

    rows_md = []
    for _, r in plan[plan["user_action"] == "NO_MODEL_OPINION"].iterrows():
        held = (f"{r['position_side']} "
                f"{_fmt(r['position_qty'], '{:.0f}')} · avg cost "
                f"{_fmt(r['avg_cost'])} · value "
                f"{_fmt(r['market_value_abs'], '{:,.0f}')}"
                if r["position_side"] else "no position")
        rows_md += [f"- **{r['symbol']}**: {held} · latest close "
                    f"{_fmt(r['previous_close'])} — outside the scored "
                    "model universe. No AI-Quant target / action / "
                    "execution band available."]
    sect("8. NO MODEL OPINION", rows_md)

    n_noact = int((plan["user_action"] == "NO_ACTION").sum())
    md += ["", f"(NO_ACTION rows: {n_noact} — nothing to do; see CSV for "
           "the full row dump.)", ""]
    with open(md_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    if meta.get("curve_rows"):
        pd.DataFrame(meta["curve_rows"]).to_csv(
            os.path.join(out_dir, f"{date}_price_reach_curve.csv"),
            index=False)
    shutil.copyfile(csv_p, os.path.join(out_dir,
                    "latest_next_session_action_plan.csv"))
    shutil.copyfile(md_p, os.path.join(out_dir,
                    "latest_next_session_action_plan.md"))
    return csv_p, md_p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--holdings", default="my_holdings.csv")
    ap.add_argument("--date", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    hp = a.holdings if os.path.isabs(a.holdings) \
        else os.path.join(ROOT, a.holdings)
    try:
        plan, meta = build_plan(ROOT, hp, a.date)
    except hold.HoldingsError as e:
        print(f"HOLDINGS VALIDATION ERROR: {e}")
        return 2
    csv_p, md_p = write_report(plan, meta, a.out_dir)
    ua = plan["user_action"].value_counts().to_dict()
    print(f"[plan {meta['signal_date']} -> "
          f"{meta['intended_execution_date']}] "
          f"{len(plan)} rows, stale={meta['book_stale']}, "
          f"actions={ua} -> {md_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
