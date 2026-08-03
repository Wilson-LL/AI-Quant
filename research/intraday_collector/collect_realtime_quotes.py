"""v15 intraday collector MVP — realtime quote snapshots to local SQLite.

DATA COLLECTION ONLY. No orders, no broker APIs (twstock has none; this
polls the public TWSE MIS quote endpoint). Raw data stays local
(research/intraday_cache/, gitignored). Restart-safe: append-only WAL
SQLite; every run is recorded in collector_runs; stale 'running' rows from
crashed runs are marked aborted on the next start; one failing symbol or
chunk never kills the cycle.

Usage:
  python research/intraday_collector/collect_realtime_quotes.py --once
  python research/intraday_collector/collect_realtime_quotes.py --mock 5
  python research/intraday_collector/collect_realtime_quotes.py \
      [--universe book|full] [--interval 60] [--max-cycles N]
      # session loop: ONLY runs 08:55-13:35 TW on weekdays; exits otherwise.
      # Do not start the session loop without user approval.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, time as dtime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

DB_DEFAULT = os.path.join(ROOT, "research", "intraday_cache", "intraday.sqlite")
CHUNK = 20
THROTTLE_S = 1.5
SESSION_START, SESSION_END = dtime(8, 55), dtime(13, 35)
STALE_S = 180
VOLJUMP_Z = 6.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS intraday_quotes (
  symbol TEXT NOT NULL, timestamp TEXT NOT NULL, price REAL, volume REAL,
  cumulative_volume REAL, bid_price REAL, ask_price REAL, open REAL,
  high REAL, low REAL, previous_close REAL, source TEXT NOT NULL,
  collected_at TEXT NOT NULL, run_id INTEGER,
  PRIMARY KEY (symbol, timestamp, source));
CREATE TABLE IF NOT EXISTS intraday_1m_bars (
  symbol TEXT NOT NULL, bar_time TEXT NOT NULL, open REAL, high REAL,
  low REAL, close REAL, volume_delta REAL, amount_delta REAL,
  source TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (symbol, bar_time, source));
CREATE TABLE IF NOT EXISTS collector_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, ended_at TEXT,
  mode TEXT, universe TEXT, n_symbols INTEGER, interval_s REAL,
  cycles INTEGER DEFAULT 0, quotes_written INTEGER DEFAULT 0,
  events INTEGER DEFAULT 0, status TEXT);
CREATE TABLE IF NOT EXISTS data_quality_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, symbol TEXT,
  event_time TEXT, event_type TEXT, detail TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS iq_date ON intraday_quotes (timestamp);
CREATE INDEX IF NOT EXISTS ev_time ON data_quality_events (event_time);
"""


def connect(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    # restart safety: mark crashed runs
    con.execute("UPDATE collector_runs SET status='aborted', "
                "ended_at=COALESCE(ended_at, datetime('now','localtime')) "
                "WHERE status='running'")
    con.commit()
    return con


def load_universe(mode):
    import glob
    import pandas as pd
    if mode == "book":
        books = sorted(glob.glob(os.path.join(
            ROOT, "reports", "paper_trading",
            "*_blend50_band10_decision_book.csv")))
        if not books:
            sys.exit("no decision book found for --universe book")
        return sorted(set(pd.read_csv(books[-1], dtype={"symbol": str})
                          ["symbol"]))
    from data import SECTOR_MAP
    return sorted(s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf")


def prev_closes(symbols):
    import pandas as pd
    out = {}
    for s in symbols:
        p = os.path.join(ROOT, "research", "data_cache", f"{s}.csv")
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p, usecols=["date", "close"]).drop_duplicates(
                    "date", keep="last")
                out[s] = float(df["close"].iloc[-1])
            except Exception:
                pass
    return out


def _num(v):
    try:
        f = float(v)
        return f if f > 0 or f == 0 else None
    except (TypeError, ValueError):
        return None


def _event(con, run_id, symbol, etype, detail, source):
    con.execute("INSERT INTO data_quality_events "
                "(run_id, symbol, event_time, event_type, detail, source) "
                "VALUES (?,?,datetime('now','localtime'),?,?,?)",
                (run_id, symbol, etype, detail[:300], source))


def store_snapshot(con, run_id, sym, snap, pc, state, source="TWSE_MIS"):
    """Validate one snapshot; insert quote; emit quality events.
    Returns quotes_written (0/1)."""
    now = datetime.now().isoformat(timespec="seconds")
    if not snap or not snap.get("success"):
        _event(con, run_id, sym, "MISSING_QUOTE",
               str(snap.get("rtmessage", "no payload"))[:100] if snap else
               "no payload", source)
        return 0
    rt = snap.get("realtime", {})
    ts = snap.get("info", {}).get("time") or ""
    price = _num(rt.get("latest_trade_price"))
    cumv = _num(rt.get("accumulate_trade_volume"))
    tickv = _num(rt.get("trade_volume"))
    if price is None or price <= 0:
        _event(con, run_id, sym, "INVALID_PRICE",
               f"price={rt.get('latest_trade_price')!r}", source)
        # still record the row (price NULL) so gaps are analyzable
    st = state.setdefault(sym, {})
    if ts and st.get("ts") == ts and time.time() - st.get("wall", 0) > STALE_S:
        _event(con, run_id, sym, "STALE_QUOTE",
               f"exchange ts unchanged: {ts}", source)
    if st.get("ts") != ts:
        st["ts"], st["wall"] = ts, time.time()
    if cumv is not None and st.get("cumv") is not None \
            and cumv < st["cumv"]:
        _event(con, run_id, sym, "CUMVOL_DECREASE",
               f"{st['cumv']} -> {cumv}", source)
    if cumv is not None:
        st["cumv"] = cumv
    if tickv is not None:
        hist = st.setdefault("ticks", [])
        if len(hist) >= 10:
            mu = sum(hist) / len(hist)
            sd = (sum((x - mu) ** 2 for x in hist) / len(hist)) ** 0.5
            if sd > 0 and (tickv - mu) / sd > VOLJUMP_Z:
                _event(con, run_id, sym, "VOLUME_JUMP",
                       f"tick {tickv:.0f} vs mean {mu:.0f} sd {sd:.0f}",
                       source)
        hist.append(tickv)
        if len(hist) > 60:
            del hist[0]
    bid = _num((rt.get("best_bid_price") or [None])[0])
    ask = _num((rt.get("best_ask_price") or [None])[0])
    con.execute(
        "INSERT OR IGNORE INTO intraday_quotes VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sym, ts or now, price, tickv, cumv, bid, ask,
         _num(rt.get("open")), _num(rt.get("high")), _num(rt.get("low")),
         pc.get(sym), source, now, run_id))
    return 1


def in_session(now=None):
    now = now or datetime.now()
    return (now.weekday() < 5
            and SESSION_START <= now.time() <= SESSION_END)


def mock_snapshot(sym, cycle, base=100.0):
    """Deterministic synthetic snapshot; injects anomalies for testing:
    symbol ending '1' -> zero price at cycle 2; ending '2' -> frozen
    timestamp; ending '3' -> volume spike at cycle 3."""
    import numpy as np
    rng = np.random.default_rng(abs(hash((sym, cycle))) % (2 ** 31))
    px = base * (1 + 0.001 * cycle + float(rng.normal(0, 0.002)))
    tick = 50 + float(rng.integers(0, 50))
    ts_cycle = 0 if sym.endswith("2") else cycle
    if sym.endswith("1") and cycle == 2:
        px = 0.0
    if sym.endswith("3") and cycle == 3:
        tick = 99999
    return {"success": True,
            "info": {"time": f"2026-01-01 09:{ts_cycle:02d}:00"},
            "realtime": {"latest_trade_price": f"{px:.2f}",
                         "trade_volume": f"{tick:.0f}",
                         "accumulate_trade_volume": f"{(cycle + 1) * 500:.0f}",
                         "best_bid_price": [f"{px - 0.05:.2f}"],
                         "best_ask_price": [f"{px + 0.05:.2f}"],
                         "open": f"{base:.2f}", "high": f"{px + 1:.2f}",
                         "low": f"{base - 1:.2f}"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["book", "full"], default="book")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--mock", type=int, default=0,
                    help="run N synthetic cycles (no network)")
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--db", default=DB_DEFAULT)
    a = ap.parse_args()
    a.interval = max(30.0, a.interval)

    con = connect(a.db)
    mode = "mock" if a.mock else ("once" if a.once else "session")
    if a.mock:
        symbols = ["9991", "9992", "9993", "9990"]
        pc = {s: 100.0 for s in symbols}
    else:
        symbols = load_universe(a.universe)
        pc = prev_closes(symbols)
    cur = con.execute(
        "INSERT INTO collector_runs (started_at, mode, universe, n_symbols, "
        "interval_s, status) VALUES (datetime('now','localtime'),?,?,?,?,"
        "'running')", (mode, a.universe, len(symbols), a.interval))
    run_id = cur.lastrowid
    con.commit()
    print(f"[collector] run {run_id} mode={mode} symbols={len(symbols)}")

    state, cycles, written, t0 = {}, 0, 0, time.time()
    try:
        if a.mock:
            for cyc in range(a.mock):
                for s in symbols:
                    written += store_snapshot(con, run_id, s,
                                              mock_snapshot(s, cyc), pc,
                                              state, source="MOCK")
                # simulate a missing quote for coverage testing
                store_snapshot(con, run_id, "9994", None, pc, state, "MOCK")
                cycles += 1
                con.commit()
        else:
            import twstock
            while True:
                if not a.once and not in_session():
                    print("[collector] outside session window "
                          "(08:55-13:35 TW weekdays) — exiting; session "
                          "loop requires user approval to schedule")
                    break
                for i in range(0, len(symbols), CHUNK):
                    chunk = symbols[i:i + CHUNK]
                    try:
                        res = twstock.realtime.get(chunk)
                    except Exception as e:  # noqa: BLE001
                        _event(con, run_id, ",".join(chunk[:3]) + "...",
                               "FETCH_ERROR", f"{type(e).__name__}: {e}",
                               "TWSE_MIS")
                        continue
                    if len(chunk) == 1:
                        res = {chunk[0]: res}
                    for s in chunk:
                        snap = res.get(s) if isinstance(res, dict) else None
                        if isinstance(snap, dict) and "realtime" not in snap \
                                and "success" in res:
                            snap = res  # single-symbol payload shape
                        try:
                            written += store_snapshot(con, run_id, s, snap,
                                                      pc, state)
                        except Exception as e:  # noqa: BLE001
                            _event(con, run_id, s, "STORE_ERROR",
                                   f"{type(e).__name__}: {e}", "TWSE_MIS")
                    time.sleep(THROTTLE_S)
                cycles += 1
                con.commit()
                if a.once or (a.max_cycles and cycles >= a.max_cycles):
                    break
                time.sleep(max(0.0, a.interval - THROTTLE_S))
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        nev = con.execute("SELECT COUNT(*) FROM data_quality_events WHERE "
                          "run_id=?", (run_id,)).fetchone()[0]
        con.execute("UPDATE collector_runs SET ended_at="
                    "datetime('now','localtime'), cycles=?, quotes_written=?,"
                    " events=?, status=? WHERE run_id=?",
                    (cycles, written, nev, status, run_id))
        con.commit()
        con.close()
    print(f"[collector] run {run_id} {status}: {cycles} cycle(s), "
          f"{written} quotes, {nev} quality events, "
          f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
