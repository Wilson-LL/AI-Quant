"""v16 Stage C1 — read-only live market state from the v15 intraday DB.

Data contract (verified against the live SQLite 2026-08-19, C1-0 audit):
intraday_quotes(symbol TEXT, timestamp TEXT exchange time
"YYYY-MM-DD HH:MM:SS", price REAL — NULL means no trade matched in the
latest window (normal TWSE MIS microstructure, v15.1), volume,
cumulative_volume, bid_price, ask_price, open/high/low REAL — session so
far, previous_close REAL, source 'TWSE_MIS'|'MOCK', collected_at TEXT
local ISO "YYYY-MM-DDTHH:MM:SS", run_id). PK (symbol,timestamp,source).

READ-ONLY. Never writes to the DB, never places orders. Midquote is a
STATE PROXY, never an execution price (v15 finding).

Freshness thresholds (pre-registered before inspecting any symbol's live
outcome; global, never per-symbol): exchange-timestamp age <= 120s FRESH,
<= 300s AGING, > 300s STALE, no row MISSING. The exchange timestamp is
authoritative for freshness (it can legitimately trail collected_at —
e.g. the 13:25-13:30 closing-auction freeze).
"""

import datetime as _dt
import os
import sqlite3

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DB_DEFAULT = os.path.join(ROOT, "research", "intraday_cache",
                          "intraday.sqlite")

FRESH_S = 120          # pre-registered
AGING_S = 300          # pre-registered
SOURCE = "TWSE_MIS"


def parse_ts(s):
    if s is None:
        return None
    s = str(s).strip().replace("T", " ")
    try:
        return _dt.datetime.fromisoformat(s[:19])
    except ValueError:
        return None


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def freshness_of(age_s):
    if age_s is None:
        return "MISSING"
    if age_s <= FRESH_S:
        return "FRESH"
    if age_s <= AGING_S:
        return "AGING"
    return "STALE"


def load_session_state(db_path, session_date, now=None, source=SOURCE):
    """{symbol: state dict} for one session date. READ-ONLY single query
    per table; duplicate protection: PK is (symbol,timestamp,source) and
    only `source` rows are read, so 'latest' is unique per symbol."""
    now = now or _dt.datetime.now()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute(
            "SELECT symbol, timestamp, price, bid_price, ask_price, "
            "open, high, low, previous_close, collected_at "
            "FROM intraday_quotes WHERE source=? AND "
            "substr(timestamp,1,10)=? ORDER BY symbol, timestamp",
            (source, str(session_date)[:10])).fetchall()
    finally:
        con.close()
    out = {}
    for (sym, ts, px, bid, ask, o, h, lo, pc, cat) in rows:
        st = out.setdefault(sym, {"symbol": sym})
        st["exchange_ts"] = ts                      # rows are ts-sorted
        st["collected_at"] = cat
        st["bid"] = _num(bid)
        st["ask"] = _num(ask)
        st["open"] = _num(o)
        st["high"] = _num(h)
        st["low"] = _num(lo)
        st["prev_close_db"] = _num(pc)
        p = _num(px)
        if p is not None:
            st["last_trade_price"] = p
            st["last_trade_ts"] = ts
    for st in out.values():
        ex = parse_ts(st.get("exchange_ts"))
        co = parse_ts(st.get("collected_at"))
        st["quote_age_seconds"] = ((now - ex).total_seconds()
                                   if ex else None)
        st["collected_age_seconds"] = ((now - co).total_seconds()
                                       if co else None)
        tr = parse_ts(st.get("last_trade_ts"))
        st["trade_age_seconds"] = ((now - tr).total_seconds()
                                   if tr else None)
        st["quote_freshness"] = freshness_of(st["quote_age_seconds"])
        st.setdefault("last_trade_price", None)
        st.setdefault("last_trade_ts", None)
    return out


def midquote(state):
    b, a = state.get("bid"), state.get("ask")
    if b is not None and a is not None:
        return (b + a) / 2.0
    return None


def actionable_price(state, side):
    """(price, provenance) per the pre-registered hierarchy.
    side='buy':  BEST_ASK -> fresh TRADE_PRICE -> MIDQUOTE_STATE_PROXY
                 -> STALE_TRADE_STATE_PROXY -> NONE.
    side='sell': BEST_BID -> fresh TRADE_PRICE -> MIDQUOTE_STATE_PROXY
                 -> STALE_TRADE_STATE_PROXY -> NONE.
    side='state' (HOLD / NO_ACTION market context, never an execution
                 recommendation): fresh TRADE_PRICE ->
                 MIDQUOTE_STATE_PROXY -> STALE_TRADE_STATE_PROXY -> NONE.
    *_STATE_PROXY values are state estimates, NEVER execution prices."""
    if state is None:
        return None, "NONE"
    if side in ("buy", "sell"):
        quoted = state.get("ask") if side == "buy" else state.get("bid")
        if quoted is not None:
            return quoted, "BEST_ASK" if side == "buy" else "BEST_BID"
    tr = state.get("last_trade_price")
    ta = state.get("trade_age_seconds")
    if tr is not None and ta is not None and ta <= AGING_S:
        return tr, "TRADE_PRICE"
    mq = midquote(state)
    if mq is not None:
        return mq, "MIDQUOTE_STATE_PROXY"
    if tr is not None:
        return tr, "STALE_TRADE_STATE_PROXY"
    return None, "NONE"
