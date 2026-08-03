"""v15 — collector status overview (runs, coverage, recent events).

Usage: python research/intraday_collector/status.py [--db path]
"""

import argparse
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DEFAULT = os.path.join(ROOT, "research", "intraday_cache",
                          "intraday.sqlite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_DEFAULT)
    a = ap.parse_args()
    if not os.path.exists(a.db):
        print(f"[status] no database at {a.db} — collector has never run")
        return
    con = sqlite3.connect(a.db, timeout=30)
    print(f"[status] db: {a.db} "
          f"({os.path.getsize(a.db) / 2**20:.1f} MB)")
    print("\nruns (latest 5):")
    for r in con.execute(
            "SELECT run_id, started_at, ended_at, mode, universe, n_symbols,"
            " cycles, quotes_written, events, status FROM collector_runs "
            "ORDER BY run_id DESC LIMIT 5"):
        print("  " + " | ".join(str(x) for x in r))
    print("\nquotes by date/source:")
    for r in con.execute(
            "SELECT substr(timestamp,1,10), source, COUNT(*), "
            "COUNT(DISTINCT symbol) FROM intraday_quotes "
            "GROUP BY 1, 2 ORDER BY 1 DESC LIMIT 10"):
        print(f"  {r[0]} {r[1]:9s} {r[2]:7d} quotes  {r[3]:4d} symbols")
    print("\n1m bars by date:")
    for r in con.execute(
            "SELECT substr(bar_time,1,10), COUNT(*), COUNT(DISTINCT symbol) "
            "FROM intraday_1m_bars GROUP BY 1 ORDER BY 1 DESC LIMIT 5"):
        print(f"  {r[0]} {r[1]:7d} bars  {r[2]:4d} symbols")
    print("\nrecent quality events:")
    for r in con.execute(
            "SELECT event_time, symbol, event_type, substr(detail,1,60) "
            "FROM data_quality_events ORDER BY event_id DESC LIMIT 8"):
        print("  " + " | ".join(str(x) for x in r))
    con.close()


if __name__ == "__main__":
    main()
