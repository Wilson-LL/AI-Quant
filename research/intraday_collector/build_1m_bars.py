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
    rows = con.execute(
        "SELECT symbol, timestamp, price, cumulative_volume, source "
        "FROM intraday_quotes WHERE substr(timestamp,1,10)=? AND price>0 "
        "ORDER BY symbol, timestamp", (a.date,)).fetchall()
    now = datetime.now().isoformat(timespec="seconds")
    bars = {}
    last_cum = {}
    for sym, ts, px, cum, src in rows:
        minute = ts[:16]  # YYYY-MM-DD HH:MM
        key = (sym, minute, src)
        b = bars.get(key)
        if b is None:
            bars[key] = b = {"o": px, "h": px, "l": px, "c": px, "v": 0.0}
        b["h"], b["l"], b["c"] = max(b["h"], px), min(b["l"], px), px
        prev = last_cum.get((sym, src))
        if cum is not None:
            if prev is not None:
                b["v"] += max(0.0, cum - prev)
            last_cum[(sym, src)] = cum
    con.executemany(
        "INSERT OR REPLACE INTO intraday_1m_bars VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(sym, minute, b["o"], b["h"], b["l"], b["c"], b["v"], None, src, now)
         for (sym, minute, src), b in bars.items()])
    con.commit()
    n_sym = len({k[0] for k in bars})
    print(f"[bars] {a.date}: {len(bars)} 1m bars across {n_sym} symbols "
          f"from {len(rows)} snapshots")
    con.close()


if __name__ == "__main__":
    main()
