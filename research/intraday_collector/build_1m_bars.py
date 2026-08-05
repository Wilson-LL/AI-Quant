"""v15 — aggregate collected quote snapshots into 1-minute bars.

Reads intraday_quotes for a date, buckets by exchange-timestamp minute,
derives OHLC from snapshot prices and volume_delta from cumulative-volume
differences (clipped at 0; decreases were already flagged as quality
events). Idempotent: INSERT OR REPLACE on (symbol, bar_time, source).

Usage: python research/intraday_collector/build_1m_bars.py [--date YYYY-MM-DD]
       [--db path]   # default date = latest quote date in the DB
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DEFAULT = os.path.join(ROOT, "research", "intraday_cache",
                          "intraday.sqlite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--db", default=DB_DEFAULT)
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=30)
    if a.date is None:
        row = con.execute("SELECT MAX(substr(timestamp,1,10)) "
                          "FROM intraday_quotes").fetchone()
        if not row or not row[0]:
            sys.exit("no quotes in DB")
        a.date = row[0]
    try:  # v15.1 migration for pre-patch DBs
        con.execute("ALTER TABLE intraday_1m_bars ADD COLUMN price_basis TEXT")
    except sqlite3.OperationalError:
        pass
    # TWSE MIS serves latest_trade_price='-' when no trade matched in the
    # most recent window; bid/ask stay live. Bars fall back to the midquote
    # BUT carry price_basis so consumers can distinguish TRADE_PRICE bars
    # from MIDQUOTE_FALLBACK/MIXED state proxies. Midquote bars are NOT
    # execution prices and must never be used as fills without explicit
    # spread/slippage assumptions.
    rows = con.execute(
        "SELECT symbol, timestamp, price, bid_price, ask_price, "
        "cumulative_volume, source FROM intraday_quotes "
        "WHERE substr(timestamp,1,10)=? ORDER BY symbol, timestamp",
        (a.date,)).fetchall()
    now = datetime.now().isoformat(timespec="seconds")
    bars = {}
    last_cum = {}
    n_used = 0
    for sym, ts, px, bid, ask, cum, src in rows:
        if px is not None and px > 0:
            eff, from_trade = px, True
        elif bid and ask and bid > 0 and ask > 0:
            eff, from_trade = (bid + ask) / 2.0, False
        else:
            continue
        n_used += 1
        minute = ts[:16]  # YYYY-MM-DD HH:MM
        key = (sym, minute, src)
        b = bars.get(key)
        if b is None:
            bars[key] = b = {"o": eff, "h": eff, "l": eff, "c": eff,
                             "v": 0.0, "n_trade": 0, "n_mid": 0}
        b["h"], b["l"], b["c"] = max(b["h"], eff), min(b["l"], eff), eff
        b["n_trade" if from_trade else "n_mid"] += 1
        prev = last_cum.get((sym, src))
        if cum is not None:
            if prev is not None:
                b["v"] += max(0.0, cum - prev)
            last_cum[(sym, src)] = cum
    def basis(b):
        if b["n_mid"] == 0:
            return "TRADE_PRICE"
        if b["n_trade"] == 0:
            return "MIDQUOTE_FALLBACK"
        return "MIXED"
    con.executemany(
        "INSERT OR REPLACE INTO intraday_1m_bars (symbol, bar_time, open, "
        "high, low, close, volume_delta, amount_delta, source, created_at, "
        "price_basis) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(sym, minute, b["o"], b["h"], b["l"], b["c"], b["v"], None, src,
          now, basis(b)) for (sym, minute, src), b in bars.items()])
    con.commit()
    n_sym = len({k[0] for k in bars})
    counts = {}
    for b in bars.values():
        counts[basis(b)] = counts.get(basis(b), 0) + 1
    print(f"[bars] {a.date}: {len(bars)} 1m bars across {n_sym} symbols "
          f"from {n_used} usable snapshots; basis {counts} "
          "(midquote bars are state proxies, not execution prices)")
    con.close()


if __name__ == "__main__":
    main()
