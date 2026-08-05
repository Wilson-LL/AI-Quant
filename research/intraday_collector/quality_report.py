"""v15.1 — end-of-day data quality report for collected intraday data.

Per symbol/day: snapshots, distinct minutes, coverage, largest pre-auction
gap, no-trade tick count (from stored rows — NOT event spam), true
invalid/stale/volume-jump events, bars by price basis. Per run: written vs
stored (dedupe) and measured cadence statistics. Output md goes to the
gitignored quality/ subdir (generated artifact).

Usage: python research/intraday_collector/quality_report.py [--date YYYY-MM-DD]
"""

import argparse
import json
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
    ap.add_argument("--expected-cycles", type=int, default=270)
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=30)
    if a.date is None:
        row = con.execute("SELECT MAX(substr(timestamp,1,10)) "
                          "FROM intraday_quotes").fetchone()
        if not row or not row[0]:
            sys.exit("no quotes in DB")
        a.date = row[0]

    q = {r[0]: r for r in con.execute(
        "SELECT symbol, COUNT(*), COUNT(DISTINCT substr(timestamp,1,16)), "
        "SUM(CASE WHEN price IS NULL AND bid_price IS NOT NULL THEN 1 "
        "ELSE 0 END), MIN(source) FROM intraday_quotes "
        "WHERE substr(timestamp,1,10)=? GROUP BY symbol", (a.date,))}
    # events are wall-clock-stamped; tie them to the runs that produced
    # this date's quotes (robust when data date != collection date, e.g.
    # mock runs)
    run_ids = [r[0] for r in con.execute(
        "SELECT DISTINCT run_id FROM intraday_quotes "
        "WHERE substr(timestamp,1,10)=?", (a.date,))]
    ph = ",".join("?" * len(run_ids)) or "NULL"
    ev = dict(con.execute(
        f"SELECT symbol || '|' || event_type, COUNT(*) "
        f"FROM data_quality_events WHERE run_id IN ({ph}) "
        f"AND symbol != '*' GROUP BY symbol, event_type",
        run_ids).fetchall())
    bars = {r[0]: (r[1], r[2], r[3]) for r in con.execute(
        "SELECT symbol, "
        "SUM(CASE WHEN price_basis='TRADE_PRICE' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN price_basis='MIDQUOTE_FALLBACK' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN price_basis='MIXED' THEN 1 ELSE 0 END) "
        "FROM intraday_1m_bars WHERE substr(bar_time,1,10)=? "
        "GROUP BY symbol", (a.date,))}
    # largest pre-auction gap per symbol (minutes between snapshots <=13:25)
    gaps = {}
    for sym in q:
        ts = [r[0] for r in con.execute(
            "SELECT timestamp FROM intraday_quotes WHERE symbol=? AND "
            "substr(timestamp,1,10)=? AND substr(timestamp,12,5) <= '13:25' "
            "ORDER BY timestamp", (sym, a.date))]
        worst = 0.0
        for i in range(1, len(ts)):
            m0 = int(ts[i - 1][11:13]) * 60 + int(ts[i - 1][14:16])
            m1 = int(ts[i][11:13]) * 60 + int(ts[i][14:16])
            worst = max(worst, m1 - m0)
        gaps[sym] = worst

    md = [f"# Intraday collector quality report — {a.date}", "",
          "(generated artifact — gitignored; research data only. Midquote-"
          "derived bars are STATE PROXIES, not execution prices — never use "
          "them as fills without explicit spread/slippage assumptions.)", ""]

    md += ["## Runs on this date", "",
           "| run | mode | cycles | written | stored | deduped | mean/median"
           "/p95/max cycle s | events |", "|---|---|---|---|---|---|---|---|"]
    for r in con.execute(
            "SELECT run_id, mode, cycles, quotes_written, events, "
            "cadence_json FROM collector_runs WHERE substr(started_at,1,10)=?"
            " ORDER BY run_id", (a.date,)):
        stored = con.execute("SELECT COUNT(*) FROM intraday_quotes WHERE "
                             "run_id=?", (r[0],)).fetchone()[0]
        cad = json.loads(r[5]) if r[5] else {}
        cs = ("/".join(str(cad.get(k, "-")) for k in
                       ("mean_cycle_s", "median_cycle_s", "p95_cycle_s",
                        "max_cycle_s")) if cad else "n/a")
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {stored} | "
                  f"{r[3] - stored} | {cs} | {r[4]} |")

    etypes = ["INVALID_PRICE", "STALE_QUOTE", "VOLUME_JUMP",
              "CUMVOL_DECREASE", "MISSING_QUOTE", "FETCH_ERROR",
              "STORE_ERROR"]
    md += ["", "## Per symbol", "",
           "| symbol | src | snaps | minutes | coverage | no-trade | gap(m) "
           "| bars T/M/X | " + " | ".join(e.split('_')[0] for e in etypes)
           + " |", "|" + "---|" * (8 + len(etypes))]
    flagged = []
    for sym in sorted(q):
        _, n, mins, null_px, src = q[sym]
        cov = mins / a.expected_cycles
        counts = [ev.get(f"{sym}|{e}", 0) for e in etypes]
        # no-trade = NULL-price rows minus the truly-invalid ones
        ntrade_null = max(0, (null_px or 0) - counts[0])
        bt = bars.get(sym, (0, 0, 0))
        md.append(f"| {sym} | {src} | {n} | {mins} | {cov:.0%} | "
                  f"{ntrade_null} | {gaps.get(sym, 0):.0f} | "
                  f"{bt[0]}/{bt[1]}/{bt[2]} | "
                  + " | ".join(str(c) for c in counts) + " |")
        if cov < 0.9 or sum(counts) > 0:
            flagged.append(sym)
    nt_total = sum(max(0, (v[3] or 0) - ev.get(f"{s}|INVALID_PRICE", 0))
                   for s, v in q.items())
    md += ["", f"Run-level NO_TRADE_TICK summary events: see "
           "data_quality_events (symbol='*'). Per-symbol no-trade counts "
           "above come from stored rows (price NULL, bid live) = "
           f"{nt_total} total — normal TWSE MIS microstructure, "
           "informational only.",
           f"Symbols <90% coverage or with true anomaly events: "
           f"{', '.join(flagged) or 'none'}",
           f"Expected cycles/session: {a.expected_cycles} (60s cadence).",
           "bars T/M/X = TRADE_PRICE / MIDQUOTE_FALLBACK / MIXED."]
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"QUALITY_{a.date}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[quality] {a.date}: {len(q)} symbols, {nt_total} no-trade "
          f"ticks, flagged: {len(flagged)} -> {p}")
    con.close()


if __name__ == "__main__":
    main()
