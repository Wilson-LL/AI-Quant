"""v15 — end-of-day data quality report for collected intraday data.

Per symbol/day: snapshot count, coverage vs expected cycles, largest gap,
quality-event counts by type, bar count. Output md goes to the gitignored
quality/ subdir (generated artifact).

Usage: python research/intraday_collector/quality_report.py [--date YYYY-MM-DD]
"""

import argparse
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DEFAULT = os.path.join(ROOT, "research", "intraday_cache",
                          "intraday.sqlite")
OUT_DIR = os.path.join(ROOT, "reports", "continuous_research",
                       "v15_intraday_collector", "quality")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--expected-cycles", type=int, default=270,
                    help="expected snapshots/symbol for a full session at "
                         "60s cadence (09:00-13:30)")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=30)
    if a.date is None:
        row = con.execute("SELECT MAX(substr(timestamp,1,10)) "
                          "FROM intraday_quotes").fetchone()
        if not row or not row[0]:
            sys.exit("no quotes in DB")
        a.date = row[0]
    q = con.execute(
        "SELECT symbol, source, COUNT(*), COUNT(DISTINCT substr(timestamp,1,16)),"
        " MIN(timestamp), MAX(timestamp) FROM intraday_quotes "
        "WHERE substr(timestamp,1,10)=? GROUP BY symbol, source",
        (a.date,)).fetchall()
    ev = dict(con.execute(
        "SELECT symbol || '|' || event_type, COUNT(*) "
        "FROM data_quality_events WHERE substr(event_time,1,10)>=? "
        "GROUP BY symbol, event_type", (a.date,)).fetchall())
    bars = dict(con.execute(
        "SELECT symbol, COUNT(*) FROM intraday_1m_bars "
        "WHERE substr(bar_time,1,10)=? GROUP BY symbol", (a.date,)).fetchall())
    etypes = ["MISSING_QUOTE", "STALE_QUOTE", "INVALID_PRICE",
              "CUMVOL_DECREASE", "VOLUME_JUMP", "FETCH_ERROR", "STORE_ERROR"]
    md = [f"# Intraday collector quality report — {a.date}", "",
          "(generated artifact — gitignored; research data only)", "",
          "| symbol | source | snapshots | distinct minutes | coverage | "
          "bars | " + " | ".join(e.split('_')[0] for e in etypes) + " |",
          "|" + "---|" * (6 + len(etypes))]
    worst = []
    for sym, src, n, mins, tmin, tmax in sorted(q):
        cov = mins / a.expected_cycles
        counts = [ev.get(f"{sym}|{e}", 0) for e in etypes]
        md.append(f"| {sym} | {src} | {n} | {mins} | {cov:.0%} | "
                  f"{bars.get(sym, 0)} | "
                  + " | ".join(str(c) for c in counts) + " |")
        if cov < 0.9 or sum(counts) > 0:
            worst.append(sym)
    md += ["", f"Symbols with <90% coverage or any event: "
           f"{', '.join(sorted(set(worst))) or 'none'}",
           f"Expected cycles/session: {a.expected_cycles} (60s cadence).",
           "Coverage on MOCK/smoke runs is expected to be tiny — this "
           "report is meaningful for real session runs only."]
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"QUALITY_{a.date}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[quality] {a.date}: {len(q)} symbol-source rows -> {p}")
    con.close()


if __name__ == "__main__":
    main()
