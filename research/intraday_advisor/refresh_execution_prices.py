"""v16 Stage C1 — LIVE EXECUTION REFRESH of the nightly action plan.

Reads (all READ-ONLY): the latest next-session action plan CSV, the v15
intraday SQLite, and the session date. Writes only gitignored
reports/user_actions/ outputs. No orders, no broker APIs, no model
inference, no portfolio construction, no re-calibration — fast CPU only.

Separates three concepts the B2 review made explicit:
  signal_validity        — the validated model signal (price never
                           invalidates it)
  live_execution_state   — where the actionable price sits vs the
                           night bands
  execution_quality      — how expensive the entry/exit is vs the
                           historical next-open distribution
`ABOVE_PREFERRED_EXECUTION_RANGE` means unusually expensive execution,
NOT model-signal invalidity. `BELOW_RISK_BAND` means GAPPED_THROUGH_
RISK_REVIEW (manual review), never "bargain — buy".

Usage:
  python research/intraday_advisor/refresh_execution_prices.py
      [--plan PATH] [--db PATH] [--session-date YYYY-MM-DD]
      [--now "YYYY-MM-DD HH:MM:SS"] [--diagnostic] [--out-dir PATH]
"""

import argparse
import datetime as _dt
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, HERE)

import twse_price_domain as tpd  # noqa: E402
import live_market_state as lms  # noqa: E402

OUT_DIR = os.path.join(ROOT, "reports", "user_actions")
PLAN_DEFAULT = os.path.join(OUT_DIR, "latest_next_session_action_plan.csv")

BUY_ACTIONS = ("OPEN_LONG_NEW_SIGNAL", "OPEN_LONG_EXISTING_TARGET",
               "ADD_LONG", "WATCH_LONG", "BUY_TO_COVER")
SELL_ACTIONS = ("REDUCE_LONG", "EXIT_LONG")
NO_TXN_ACTIONS = ("HOLD_LONG", "HOLD_SHORT", "REDUCE_SHORT", "NO_ACTION",
                  "NO_MODEL_OPINION", "POSITION_CONFLICT_REVIEW",
                  "WATCH_NEUTRAL")

CSV_COLS = [
    "symbol", "user_action", "model_action", "signal_validity",
    "signal_freshness", "quote_freshness", "quote_age_seconds",
    "quote_exchange_timestamp", "quote_collected_at",
    "live_price", "live_price_source", "execution_reference_confidence",
    "bid", "ask", "last_trade_price", "actual_open",
    "open_vs_auction_reference_pct", "open_vs_expected_p50_pct",
    "open_percentile_approx", "live_execution_state", "execution_quality",
    "action_valid_now", "suggested_limit_reference",
    "suggested_limit_reason", "night_reference", "night_ideal_low",
    "night_ideal_high", "night_ceiling", "night_above_preferred_range",
    "night_risk_below", "night_sell_reference", "night_sell_floor",
    "night_panic_below", "buy_reference_reach_probability",
    "range_reach_confidence", "legal_limit_down", "legal_limit_up",
    "price_domain_status", "domain_validation_status", "errors",
]


def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def open_percentile_bucket(actual_open, row):
    qs = [(_f(row.get(f"expected_open_p{p}")), p)
          for p in (10, 25, 50, 75, 90)]
    if not np.isfinite(actual_open) or any(not np.isfinite(v)
                                           for v, _ in qs):
        return ""
    v10, v25, v50, v75, v90 = [v for v, _ in qs]
    if actual_open < v10:
        return "BELOW_P10"
    if actual_open < v25:
        return "P10_P25"
    if actual_open < v50:
        return "P25_P50"
    if actual_open < v75:
        return "P50_P75"
    if actual_open <= v90:
        return "P75_P90"
    return "ABOVE_P90"


def buy_state(price, row):
    if not np.isfinite(price):
        return ""
    if np.isfinite(_f(row.get("risk_review_below"))) and \
            price < _f(row.get("risk_review_below")):
        return "BELOW_RISK_BAND"
    if price < _f(row.get("ideal_zone_low")):
        return "BELOW_IDEAL_ZONE"
    if price <= _f(row.get("ideal_zone_high")):
        return "IN_IDEAL_ZONE"
    if price <= _f(row.get("acceptable_ceiling")):
        return "ABOVE_IDEAL_WITHIN_LIMIT"
    if price <= _f(row.get("do_not_chase_above")):
        return "ABOVE_ACCEPTABLE_LIMIT"
    return "ABOVE_PREFERRED_EXECUTION_RANGE"


def sell_state(price, row):
    if not np.isfinite(price):
        return ""
    if np.isfinite(_f(row.get("do_not_panic_sell_below"))) and \
            price < _f(row.get("do_not_panic_sell_below")):
        return "BELOW_PANIC_REVIEW_LEVEL"
    if price < _f(row.get("acceptable_sell_floor")):
        return "BELOW_ACCEPTABLE_SELL_FLOOR"
    if price < _f(row.get("ideal_sell_zone_low")):
        return "BELOW_IDEAL_WITHIN_FLOOR"
    if price <= _f(row.get("ideal_sell_zone_high")):
        return "IN_IDEAL_SELL_ZONE"
    return "ABOVE_IDEAL_SELL_ZONE"


def hold_state(price, row):
    if not np.isfinite(price):
        return ""
    if np.isfinite(_f(row.get("review_below"))) and \
            price < _f(row.get("review_below")):
        return "REVIEW_BELOW"
    if np.isfinite(_f(row.get("review_above"))) and \
            price > _f(row.get("review_above")):
        return "REVIEW_ABOVE"
    return "NO_ACTION_IN_RANGE"


EXEC_QUALITY = {
    "BELOW_IDEAL_ZONE": "GOOD", "IN_IDEAL_ZONE": "GOOD",
    "ABOVE_IDEAL_WITHIN_LIMIT": "ACCEPTABLE",
    "ABOVE_ACCEPTABLE_LIMIT": "EXPENSIVE",
    "ABOVE_PREFERRED_EXECUTION_RANGE": "EXPENSIVE",
    "BELOW_RISK_BAND": "RISK_REVIEW",
    "ABOVE_IDEAL_SELL_ZONE": "GOOD", "IN_IDEAL_SELL_ZONE": "GOOD",
    "BELOW_IDEAL_WITHIN_FLOOR": "ACCEPTABLE",
    "BELOW_ACCEPTABLE_SELL_FLOOR": "POOR",
    "BELOW_PANIC_REVIEW_LEVEL": "RISK_REVIEW",
}


def suggest_limit(side, state_label, price, source, row, domain):
    """(suggested_limit_reference, reason). Only when actionable; never
    chases beyond the night bands; tick-aligned and domain-clamped
    (suggestions are OURS to clamp — observed prices never are)."""
    if source in ("MIDQUOTE_STATE_PROXY", "STALE_TRADE_STATE_PROXY",
                  "NONE") or not np.isfinite(price):
        return np.nan, "STALE_QUOTE" if source == "NONE" else \
            "STATE_PROXY_ONLY"
    if side == "buy":
        if state_label == "BELOW_RISK_BAND":
            return np.nan, "RISK_REVIEW_REQUIRED"
        if state_label in ("BELOW_IDEAL_ZONE", "IN_IDEAL_ZONE"):
            # presentation-only reason split (2026-08-19): a below-zone
            # price must not be described as "inside" the ideal zone
            inside = state_label == "IN_IDEAL_ZONE"
            src_tag = "ASK" if source == "BEST_ASK" else "TRADE"
            ref, why = price, (f"{src_tag}_INSIDE_IDEAL_ZONE" if inside
                               else f"{src_tag}_BELOW_IDEAL_ZONE")
        elif state_label == "ABOVE_IDEAL_WITHIN_LIMIT":
            ref, why = min(price, _f(row.get("acceptable_ceiling"))), \
                "ASK_WITHIN_ACCEPTABLE_LIMIT"
        else:
            # expensive: do not chase; keep the night zone as reference
            return np.nan, "CURRENT_PRICE_ABOVE_PREFERRED_RANGE"
        ref = domain.clamp(tpd.legal_floor(ref), "buy") if domain.known() \
            else tpd.legal_floor(ref)
        return ref, why
    if side == "sell":
        if state_label == "BELOW_PANIC_REVIEW_LEVEL":
            return np.nan, "RISK_REVIEW_REQUIRED"
        if state_label in ("ABOVE_IDEAL_SELL_ZONE", "IN_IDEAL_SELL_ZONE"):
            ref, why = price, ("BID_INSIDE_IDEAL_SELL_ZONE"
                               if source == "BEST_BID"
                               else "TRADE_INSIDE_IDEAL_SELL_ZONE")
        elif state_label == "BELOW_IDEAL_WITHIN_FLOOR":
            ref, why = max(price, _f(row.get("acceptable_sell_floor"))), \
                "BID_ABOVE_ACCEPTABLE_FLOOR"
        else:
            return np.nan, "RISK_REVIEW_REQUIRED"
        ref = domain.clamp(tpd.legal_ceil(ref), "sell") if domain.known() \
            else tpd.legal_ceil(ref)
        return ref, why
    return np.nan, "NO_ACTION_REQUIRED"


def refresh(plan_path, db_path, session_date, now=None, diagnostic=False):
    now = now or _dt.datetime.now()
    plan = pd.read_csv(plan_path, dtype={"symbol": str})
    signal_date = str(plan["signal_date"].iloc[0])
    intended = str(plan["intended_execution_date"].iloc[0])
    book_stale = bool(plan["book_stale"].iloc[0])
    session_date = str(session_date)[:10]

    session_ok = session_date == intended
    plan_ok = not book_stale
    mode = "LIVE"
    if not session_ok or not plan_ok:
        if not diagnostic:
            mode = "REJECTED"
        else:
            mode = "HISTORICAL_SESSION_DIAGNOSTIC"
    elif now.date().isoformat() != session_date:
        mode = "HISTORICAL_SESSION_DIAGNOSTIC"

    states = lms.load_session_state(db_path, session_date, now=now) \
        if mode != "REJECTED" else {}
    rows = []
    for _, r in plan.iterrows():
        sym = r["symbol"]
        ua = r["user_action"]
        st = states.get(sym)
        errors = []
        # ---- signal validity (never price-derived)
        if not session_ok:
            sv = "SESSION_MISMATCH"
        elif book_stale:
            sv = "STALE_PLAN"
        elif ua == "NO_MODEL_OPINION":
            sv = "NO_MODEL_OPINION"
        elif ua == "POSITION_CONFLICT_REVIEW":
            sv = "POSITION_CONFLICT"
        else:
            sv = "VALIDATED_MODEL_SIGNAL"
        side = ("buy" if ua in BUY_ACTIONS else
                "sell" if ua in SELL_ACTIONS else "none")
        fresh = st["quote_freshness"] if st else "MISSING"
        live_price, src = lms.actionable_price(
            st, side if side != "none" else "state")
        exec_conf = ("DEGRADED" if src in ("MIDQUOTE_STATE_PROXY",
                                           "STALE_TRADE_STATE_PROXY")
                     else "NA" if src == "NONE" else "NORMAL")
        # ---- domain validation of OBSERVED prices (never clamped,
        # never rewritten). Four states (C1 correctness patch):
        #   DOMAIN_OK                        inside the known domain
        #   PRICE_DOMAIN_ASSUMPTION_CONFLICT trusted quote outside a
        #       NORMAL_DAY_ASSUMPTION domain — the previous-close
        #       reference assumption may not apply (special auction
        #       reference); NOT an illegal market price
        #   LIVE_PRICE_DOMAIN_ERROR          violation of a genuinely
        #       CONFIRMED domain (currently unreachable — the repo never
        #       confirms references; branch kept for the future)
        #   UNKNOWN_DOMAIN                   no bound validation possible
        # Structurally impossible data (non-positive / off the stock
        # tick grid) is DATA_VALIDATION_ERROR — corruption, kept distinct
        # from reference-assumption uncertainty.
        night_dom = str(r.get("price_domain_status") or "")
        domain = tpd.build_domain(
            sym, _f(r.get("auction_reference_price")),
            security_type="stock"
            if night_dom in ("NORMAL_DAY_ASSUMPTION",
                             "CONFIRMED_STANDARD_LIMIT")
            else "unknown")
        domain_status = "UNKNOWN_DOMAIN"
        dom_gate = False
        if st and domain.known():
            domain_status = "DOMAIN_OK"
            confirmed = night_dom == "CONFIRMED_STANDARD_LIMIT"
            for label, v in (("bid", st.get("bid")),
                             ("ask", st.get("ask")),
                             ("last_trade", st.get("last_trade_price")),
                             ("open", st.get("open"))):
                if v is None:
                    continue
                if not np.isfinite(v) or v <= 0 or \
                        not tpd.is_legal_tick(v):
                    domain_status = "DATA_VALIDATION_ERROR"
                    errors.append(f"DATA_VALIDATION_ERROR:{label}")
                    dom_gate = True
                elif not (domain.legal_limit_down - 1e-9 <= v
                          <= domain.legal_limit_up + 1e-9):
                    dom_gate = True
                    if confirmed:
                        domain_status = "LIVE_PRICE_DOMAIN_ERROR"
                        errors.append(f"LIVE_PRICE_DOMAIN_ERROR:{label}")
                    else:
                        if domain_status == "DOMAIN_OK":
                            domain_status = \
                                "PRICE_DOMAIN_ASSUMPTION_CONFLICT"
                        errors.append(
                            f"PRICE_DOMAIN_ASSUMPTION_CONFLICT:{label}")
        # ---- actual open context
        a_open = _f(st.get("open")) if st else np.nan
        ref = _f(r.get("auction_reference_price"))
        p50 = _f(r.get("expected_open_p50"))
        open_bucket = open_percentile_bucket(a_open, r)
        # ---- live execution state
        price_for_state = live_price if live_price is not None else np.nan
        if side == "buy" and ua != "BUY_TO_COVER":
            lstate = buy_state(price_for_state, r)
        elif ua == "BUY_TO_COVER":
            lstate = buy_state(price_for_state, {
                "risk_review_below": np.nan,
                "ideal_zone_low": r.get("cover_zone_low"),
                "ideal_zone_high": r.get("cover_zone_high"),
                "acceptable_ceiling": r.get("cover_zone_high"),
                "do_not_chase_above": r.get("risk_review_above")})
        elif side == "sell":
            lstate = sell_state(price_for_state, r)
        elif ua == "HOLD_LONG":
            lstate = hold_state(price_for_state, r)
        else:
            lstate = ""
        equality = EXEC_QUALITY.get(lstate, "NA")
        if lstate == "BELOW_RISK_BAND":
            lstate_display = "GAPPED_THROUGH_RISK_REVIEW"
        elif lstate == "BELOW_PANIC_REVIEW_LEVEL":
            lstate_display = "URGENT_RISK_REVIEW"
        else:
            lstate_display = lstate
        # ---- action_valid_now (hard gates)
        valid = True
        if sv in ("SESSION_MISMATCH", "STALE_PLAN", "POSITION_CONFLICT"):
            valid = False
        if fresh in ("STALE", "MISSING") and ua not in ("NO_ACTION",):
            valid = False
        if src in ("MIDQUOTE_STATE_PROXY", "STALE_TRADE_STATE_PROXY",
                   "NONE") and side != "none":
            valid = False
        if dom_gate:
            valid = False       # nightly domain context can't be trusted;
            # signal_validity is deliberately NOT touched by price
        if lstate in ("BELOW_RISK_BAND", "BELOW_PANIC_REVIEW_LEVEL"):
            valid = False
        if ua in ("NO_MODEL_OPINION",):
            valid = False                # nothing to act on
        if mode == "HISTORICAL_SESSION_DIAGNOSTIC":
            valid = False                # never actionable after the fact
        # NOTE: ABOVE_PREFERRED_EXECUTION_RANGE alone does NOT gate:
        # execution_quality=EXPENSIVE, signal stays validated, decision
        # remains manual.
        # ---- suggested limit
        if valid and side in ("buy", "sell") and ua not in NO_TXN_ACTIONS:
            sug, why = suggest_limit(side, lstate, price_for_state, src,
                                     r, domain)
        elif ua in NO_TXN_ACTIONS or side == "none":
            sug, why = np.nan, "NO_ACTION_REQUIRED"
        else:
            sug, why = np.nan, ("STALE_QUOTE"
                                if fresh in ("STALE", "MISSING")
                                else "RISK_REVIEW_REQUIRED"
                                if lstate in ("BELOW_RISK_BAND",
                                              "BELOW_PANIC_REVIEW_LEVEL")
                                else "NOT_ACTIONABLE")
        rows.append({
            "symbol": sym, "user_action": ua,
            "model_action": r.get("model_action"),
            "signal_validity": sv,
            "signal_freshness": r.get("signal_freshness"),
            "quote_freshness": fresh,
            "quote_age_seconds": (round(st["quote_age_seconds"])
                                  if st and st.get("quote_age_seconds")
                                  is not None else np.nan),
            "quote_exchange_timestamp": st.get("exchange_ts") if st else "",
            "quote_collected_at": st.get("collected_at") if st else "",
            "live_price": live_price if live_price is not None else np.nan,
            "live_price_source": src,
            "execution_reference_confidence": exec_conf,
            "bid": st.get("bid") if st else np.nan,
            "ask": st.get("ask") if st else np.nan,
            "last_trade_price": (st.get("last_trade_price") if st
                                 else np.nan),
            "actual_open": a_open,
            "open_vs_auction_reference_pct": (
                a_open / ref - 1 if np.isfinite(a_open)
                and np.isfinite(ref) else np.nan),
            "open_vs_expected_p50_pct": (
                a_open / p50 - 1 if np.isfinite(a_open)
                and np.isfinite(p50) else np.nan),
            "open_percentile_approx": open_bucket,
            "live_execution_state": lstate_display,
            "execution_quality": equality,
            "action_valid_now": bool(valid),
            "suggested_limit_reference": sug,
            "suggested_limit_reason": why,
            "night_reference": _f(r.get("reference")),
            "night_ideal_low": _f(r.get("ideal_zone_low")),
            "night_ideal_high": _f(r.get("ideal_zone_high")),
            "night_ceiling": _f(r.get("acceptable_ceiling")),
            "night_above_preferred_range": _f(r.get("do_not_chase_above")),
            "night_risk_below": _f(r.get("risk_review_below")),
            "night_sell_reference": _f(r.get("sell_reference")),
            "night_sell_floor": _f(r.get("acceptable_sell_floor")),
            "night_panic_below": _f(r.get("do_not_panic_sell_below")),
            "buy_reference_reach_probability":
                _f(r.get("buy_reference_reach_probability")),
            "range_reach_confidence": r.get("range_reach_confidence"),
            "legal_limit_down": _f(r.get("legal_limit_down")),
            "legal_limit_up": _f(r.get("legal_limit_up")),
            "price_domain_status": r.get("price_domain_status"),
            "domain_validation_status": domain_status,
            "errors": ";".join(errors),
        })
    live = pd.DataFrame(rows)
    assert not {"OPEN_SHORT", "SELL_SHORT", "WATCH_SHORT"} & \
        set(live["user_action"]), "short-creation action — forbidden"
    n = len(live)
    n_fresh = int((live["quote_freshness"] == "FRESH").sum())
    market_data = ("FRESH" if n and n_fresh >= 0.8 * n else
                   "MIXED" if n and n_fresh >= 0.5 * n else "DEGRADED")
    meta = {"signal_date": signal_date, "intended": intended,
            "session_date": session_date, "mode": mode,
            "refresh_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "market_data": market_data, "book_stale": book_stale,
            "session_ok": session_ok}
    return live, meta


# ------------------------------------------------------------------ report

def _p(v, pat="{:.2f}"):
    return "n/a" if v is None or (isinstance(v, float)
                                  and not np.isfinite(v)) else pat.format(v)


def write_report(live, meta, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    d = meta["session_date"]
    csv_p = os.path.join(out_dir, f"{d}_live_execution_plan.csv")
    md_p = os.path.join(out_dir, f"{d}_live_execution_plan.md")
    live[CSV_COLS].to_csv(csv_p, index=False)

    md = [f"# AI-Quant Live Execution Refresh — {d}", "",
          f"Night plan: signal {meta['signal_date']} · intended session "
          f"{meta['intended']} · refresh {meta['refresh_time']} · mode "
          f"**{meta['mode']}**",
          f"Timing: NEXT_OPEN_TIMING_VALIDATED · market data: "
          f"**{meta['market_data']}**", "",
          "Live prices are market-state / quote references only. No "
          "orders are placed. Suggested limits are references only — "
          "order execution is not guaranteed or predicted (order-book "
          "depth and queue priority are not modeled). Range-reach "
          "percentages are the "
          "NIGHT-BEFORE historical statistics, never recalculated with "
          "today's outcome. NO VALIDATED OPEN-SHORT MODEL EXISTS.", ""]
    if meta["mode"] == "HISTORICAL_SESSION_DIAGNOSTIC":
        md += ["> **HISTORICAL_SESSION_DIAGNOSTIC** — this refresh ran "
               "against a session that is not the current live market; "
               "every action_valid_now is false by construction.", ""]
    if meta["mode"] == "REJECTED":
        md += ["> **REJECTED — LIVE_PLAN_DATE_MISMATCH / STALE_PLAN**: "
               f"plan intends {meta['intended']}, session is "
               f"{meta['session_date']}, book_stale="
               f"{meta['book_stale']}. Regenerate the nightly plan; "
               "nothing here is actionable.", ""]

    def line(r):
        parts = [f"- **{r['symbol']}** {r['user_action']} · "
                 f"{r['live_execution_state'] or 'no live state'} · "
                 f"quality {r['execution_quality']} · "
                 f"{r['quote_freshness']}"]
        if np.isfinite(r["live_price"]):
            src = r["live_price_source"]
            px = (f"market-state proxy ≈ {_p(r['live_price'])} — wait "
                  "for a valid actionable quote/trade"
                  if src == "MIDQUOTE_STATE_PROXY"
                  else f"{src} {_p(r['live_price'])}")
            parts.append(f" · {px}")
        if np.isfinite(r["suggested_limit_reference"]):
            parts.append(f" · suggested limit reference "
                         f"{_p(r['suggested_limit_reference'])} "
                         f"({r['suggested_limit_reason']})")
        else:
            parts.append(f" · {r['suggested_limit_reason']}")
        if np.isfinite(r["actual_open"]):
            parts.append(f" · open {_p(r['actual_open'])} "
                         f"[{r['open_percentile_approx']}]")
        if r["errors"]:
            parts.append(f" · **{r['errors']}**")
        return "".join(parts)

    def reach_ctx(r):
        if np.isfinite(r["buy_reference_reach_probability"]):
            return (f"    night reference {_p(r['night_reference'])}; "
                    "historically comparable next-session LOWS reached "
                    "the reference or lower "
                    f"{r['buy_reference_reach_probability']:.0%} of the "
                    "time (range reach, NOT a fill probability; "
                    f"confidence {r['range_reach_confidence']})")
        return None

    groups = [
        ("1. ACTIONABLE NOW", live[live["action_valid_now"]]),
        ("2. PRICE EXPENSIVE / WAITING TRADEOFF",
         live[(~live["action_valid_now"])
              & (live["live_execution_state"]
                 == "ABOVE_PREFERRED_EXECUTION_RANGE")
              | (live["action_valid_now"]
                 & (live["execution_quality"] == "EXPENSIVE"))]),
        ("3. RISK REVIEW",
         live[live["live_execution_state"].isin(
             ("GAPPED_THROUGH_RISK_REVIEW", "URGENT_RISK_REVIEW"))]),
        ("4. REDUCE / EXIT", live[live["user_action"].isin(SELL_ACTIONS)]),
        ("5. HOLD / NO ACTION",
         live[live["user_action"].isin(("HOLD_LONG", "NO_ACTION"))]),
        ("6. WATCH", live[live["user_action"].isin(("WATCH_LONG",
                                                    "WATCH_NEUTRAL"))]),
        ("7. SHORT POSITION REVIEW",
         live[live["user_action"].isin(("HOLD_SHORT", "REDUCE_SHORT",
                                        "BUY_TO_COVER"))]),
        ("8. NO MODEL OPINION",
         live[live["user_action"] == "NO_MODEL_OPINION"]),
        ("9. DATA / SESSION ERRORS",
         live[(live["quote_freshness"].isin(("STALE", "MISSING")))
              | (live["errors"] != "")
              | (live["signal_validity"].isin(("SESSION_MISMATCH",
                                               "STALE_PLAN")))]),
    ]
    seen_note = ("(rows may appear in several groups; group 1 contains "
                 "ONLY action_valid_now=true rows)")
    md.append(seen_note)
    for title, g in groups:
        md += ["", f"## {title}", ""]
        if title.startswith("7.") and len(g):
            md.append("**NO VALIDATED OPEN-SHORT MODEL EXISTS** — "
                      "risk-tracked existing positions only.\n")
        if not len(g):
            md.append("None.")
            continue
        for _, r in g.iterrows():
            if title.startswith("1.") and not r["action_valid_now"]:
                continue
            md.append(line(r))
            ctx = reach_ctx(r)
            if ctx and title in ("1. ACTIONABLE NOW",
                                 "2. PRICE EXPENSIVE / WAITING TRADEOFF",
                                 "6. WATCH"):
                md.append(ctx)
            if r["live_execution_state"] == \
                    "ABOVE_PREFERRED_EXECUTION_RANGE":
                md.append("    validated model signal remains intact; "
                          "the current price is unusually expensive vs "
                          "the historical next-open distribution — "
                          "manual decision")
            if r.get("domain_validation_status") == \
                    "PRICE_DOMAIN_ASSUMPTION_CONFLICT":
                md.append(
                    "    PRICE DOMAIN REVIEW REQUIRED: the observed TWSE "
                    "market price is outside the nightly normal-day "
                    "legal-domain estimate. This does NOT establish that "
                    "the market quote is illegal — a special "
                    "opening-auction reference (e.g. a corporate-action "
                    "session) may apply, and this repo cannot verify the "
                    "reference automatically. Observed price preserved "
                    "unchanged; manual review required before using the "
                    "nightly execution bands.")
    with open(md_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    shutil.copyfile(csv_p, os.path.join(out_dir,
                    "latest_live_execution_plan.csv"))
    shutil.copyfile(md_p, os.path.join(out_dir,
                    "latest_live_execution_plan.md"))
    # user-facing simplified summary (presentation layer only)
    import simplified_reports as sr
    sr.write_live_summary(live, meta, out_dir)
    return csv_p, md_p


def terminal_summary(live, meta, md_p):
    """Concise C2 console summary (never the full CSV)."""
    n = {"act": int(live["action_valid_now"].sum()),
         "exp": int((live["live_execution_state"]
                     == "ABOVE_PREFERRED_EXECUTION_RANGE").sum()),
         "risk": int(live["live_execution_state"].isin(
             ("GAPPED_THROUGH_RISK_REVIEW", "URGENT_RISK_REVIEW")).sum()),
         "sell": int(live["user_action"].isin(SELL_ACTIONS).sum()),
         "hold": int(live["user_action"].isin(("HOLD_LONG",
                                               "NO_ACTION")).sum()),
         "data": int(((live["quote_freshness"].isin(("STALE", "MISSING")))
                      | (live["errors"] != "")).sum())}
    print("AI-Quant Live Execution Refresh")
    print(f"  Session: {meta['session_date']}   Refresh: "
          f"{meta['refresh_time']}   Mode: {meta['mode']}")
    print(f"  Market data: {meta['market_data']}")
    print(f"  ACTIONABLE NOW: {n['act']}")
    print(f"  ABOVE PREFERRED EXECUTION RANGE: {n['exp']}")
    print(f"  RISK REVIEW: {n['risk']}")
    print(f"  REDUCE / EXIT: {n['sell']}")
    print(f"  HOLD / NO ACTION: {n['hold']}")
    print(f"  DATA ISSUES: {n['data']}")
    print("  今日操作表: "
          r"reports\user_actions\latest_live_execution_summary.md")
    print(r"  完整技術報告: reports\user_actions"
          r"\latest_live_execution_plan.md")
    print("  No automatic orders.")


def main(argv=None):
    import market_readiness as mr
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=PLAN_DEFAULT)
    ap.add_argument("--db", default=lms.DB_DEFAULT)
    ap.add_argument("--session-date", default=None)
    ap.add_argument("--now", default=None,
                    help='"YYYY-MM-DD HH:MM:SS" (tests/diagnostics)')
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--wait-until-ready", action="store_true",
                    help="poll the session-level readiness gate before a "
                         "normal actionable refresh")
    ap.add_argument("--max-wait-seconds", type=int,
                    default=mr.MAX_WAIT_SECONDS_DEFAULT)
    ap.add_argument("--poll-seconds", type=int,
                    default=mr.POLL_SECONDS_DEFAULT)
    a = ap.parse_args(argv)
    now = lms.parse_ts(a.now) if a.now else _dt.datetime.now()
    session = a.session_date or now.date().isoformat()
    if not os.path.isfile(a.plan):
        print(f"plan not found: {a.plan} — run "
              "research/user_next_session_plan.py first")
        return 2
    intended = str(pd.read_csv(a.plan, nrows=1)
                   ["intended_execution_date"].iloc[0])
    if not a.diagnostic:
        # session-level readiness gate (C2). Row-level gates stay in C1.
        status, detail = mr.assess(a.db, intended, now=now)
        if a.wait_until_ready and status in ("WAITING_FOR_MARKET_OPEN",
                                             "WAITING_FOR_MARKET_DATA"):
            import time as _time
            deadline = _time.monotonic() + a.max_wait_seconds
            while status in ("WAITING_FOR_MARKET_OPEN",
                             "WAITING_FOR_MARKET_DATA") and \
                    _time.monotonic() < deadline:
                print(f"[wait] {status}: {detail} — polling every "
                      f"{a.poll_seconds}s")
                _time.sleep(a.poll_seconds)
                now = _dt.datetime.now()
                status, detail = mr.assess(a.db, intended, now=now)
            session = now.date().isoformat() if a.session_date is None \
                else session
        if status != "MARKET_READY":
            print(f"MARKET_DATA_NOT_READY ({status}): {detail}")
            if status == "MARKET_CLOSED":
                print("MARKET_CLOSED — normal actionable refresh refused; "
                      "rerun with --diagnostic for a "
                      "HISTORICAL_SESSION_DIAGNOSTIC.")
            if status == "SESSION_MISMATCH":
                print("The plan's intended date may have been a "
                      "non-trading day (the intended date is a weekday "
                      "ESTIMATE, unverified for TWSE holidays). The plan "
                      "is never rolled forward automatically. To re-issue "
                      "it for the next session, run: "
                      "research/user_next_session_plan.py --nightly "
                      "--allow-current-book-recovery (manual recovery), "
                      "then rerun this refresh.")
            print("No actionable report was generated.")
            return 3
    live, meta = refresh(a.plan, a.db, session, now=now,
                         diagnostic=a.diagnostic)
    csv_p, md_p = write_report(live, meta, a.out_dir)
    terminal_summary(live, meta, md_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
