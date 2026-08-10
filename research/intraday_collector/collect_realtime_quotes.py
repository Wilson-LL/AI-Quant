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
  source TEXT NOT NULL, created_at TEXT NOT NULL, price_basis TEXT,
  PRIMARY KEY (symbol, bar_time, source));
CREATE TABLE IF NOT EXISTS collector_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, ended_at TEXT,
  mode TEXT, universe TEXT, n_symbols INTEGER, interval_s REAL,
  cycles INTEGER DEFAULT 0, quotes_written INTEGER DEFAULT 0,
  events INTEGER DEFAULT 0, status TEXT, cadence_json TEXT);
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
    # v15.1 migrations for pre-patch DBs (no-ops on fresh databases)
    for ddl in ("ALTER TABLE intraday_1m_bars ADD COLUMN price_basis TEXT",
                "ALTER TABLE collector_runs ADD COLUMN cadence_json TEXT"):
        try:
            con.execute(ddl)
        except sqlite3.OperationalError:
            pass
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
    raw_px = rt.get("latest_trade_price")
    price = _num(raw_px)
    cumv = _num(rt.get("accumulate_trade_volume"))
    tickv = _num(rt.get("trade_volume"))
    if price is not None and price <= 0:
        price = None
        _event(con, run_id, sym, "INVALID_PRICE",
               f"non-positive price={raw_px!r}", source)
    elif price is None:
        # TWSE MIS serves '-' when no trade matched in the latest window —
        # normal microstructure, counted (not per-row event-logged) as
        # NO_TRADE_TICK; anything else unparseable is a true INVALID_PRICE
        if str(raw_px).strip() in ("-", "", "None", "null"):
            c = state.setdefault("_counters", {})
            c["no_trade"] = c.get("no_trade", 0) + 1
        else:
            _event(con, run_id, sym, "INVALID_PRICE",
                   f"unparseable price={raw_px!r}", source)
        # row is still recorded (price NULL, bid/ask live) for analyzability
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
    """Deterministic synthetic snapshot; injects the v15.1 test matrix:
    ending '1' -> '-' no-trade price (valid bid/ask) at cycle 2, malformed
    '0.00' at cycle 3; ending '2' -> frozen timestamp (stale logic is
    wall-time gated so it cannot fire in sub-second mocks); ending '3' ->
    volume spike at cycle 11 (needs >=10 history) and cumulative-volume
    decrease at cycle 12; other symbols normal."""
    import numpy as np
    rng = np.random.default_rng(abs(hash((sym, cycle))) % (2 ** 31))
    px = base * (1 + 0.001 * cycle + float(rng.normal(0, 0.002)))
    tick = 50 + float(rng.integers(0, 50))
    ts_cycle = 0 if sym.endswith("2") else cycle
    px_s = f"{px:.2f}"
    if sym.endswith("1") and cycle == 2:
        px_s = "-"                       # normal no-trade marker
    if sym.endswith("1") and cycle == 3:
        px_s = "0.00"                    # malformed -> true INVALID_PRICE
    if sym.endswith("3") and cycle == 11:
        tick = 99999                     # volume jump
    cum = (cycle + 1) * 500
    if sym.endswith("3") and cycle == 12:
        cum = 100                        # cumulative-volume decrease
    return {"success": True,
            "info": {"time": f"2026-01-01 09:{ts_cycle:02d}:00"},
            "realtime": {"latest_trade_price": px_s,
                         "trade_volume": f"{tick:.0f}",
                         "accumulate_trade_volume": f"{cum:.0f}",
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
    cycle_starts = []
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
            # The scheduled task fires slightly before the open (08:54);
            # wait for the gate instead of exiting. Bounded to 15 min so a
            # genuinely off-session launch still exits immediately below.
            now = datetime.now()
            if (not a.once and now.weekday() < 5
                    and now.time() < SESSION_START):
                wait = (datetime.combine(now.date(), SESSION_START)
                        - now).total_seconds()
                if wait <= 900:
                    print(f"[collector] {wait:.0f}s before session open "
                          "- waiting for gate")
                    time.sleep(wait)
            next_target = time.time()
            while True:
                if not a.once and not in_session():
                    print("[collector] outside session window "
                          "(08:55-13:35 TW weekdays) — exiting; session "
                          "loop requires user approval to schedule")
                    break
                cycle_starts.append(time.time())
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
                # elapsed-time-compensated pacing: one cycle per interval,
                # request/throttle time absorbed (v15.1 — measured 66.8s
                # actual cadence at nominal 60s before this fix)
                next_target += a.interval
                time.sleep(max(0.0, next_target - time.time()))
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        no_trade = state.get("_counters", {}).get("no_trade", 0)
        if no_trade:
            # ONE informational summary event per run — per-row granularity
            # lives in intraday_quotes (price NULL, bid/ask populated), so
            # NO_TRADE_TICK never drowns true anomalies in the event table
            _event(con, run_id, "*", "NO_TRADE_TICK",
                   f"n={no_trade} no-trade snapshots this run (normal TWSE "
                   "MIS behavior; counted, not per-row logged)", "TWSE_MIS"
                   if mode != "mock" else "MOCK")
        cadence = None
        if len(cycle_starts) >= 2:
            import numpy as np
            d = np.diff(cycle_starts)
            cadence = {"mean_cycle_s": round(float(d.mean()), 2),
                       "median_cycle_s": round(float(np.median(d)), 2),
                       "p95_cycle_s": round(float(np.percentile(d, 95)), 2),
                       "min_cycle_s": round(float(d.min()), 2),
                       "max_cycle_s": round(float(d.max()), 2)}
        nev = con.execute("SELECT COUNT(*) FROM data_quality_events WHERE "
                          "run_id=?", (run_id,)).fetchone()[0]
        con.execute("UPDATE collector_runs SET ended_at="
                    "datetime('now','localtime'), cycles=?, quotes_written=?,"
                    " events=?, status=?, cadence_json=? WHERE run_id=?",
                    (cycles, written, nev, status,
                     json.dumps(cadence) if cadence else None, run_id))
        con.commit()
        con.close()
    print(f"[collector] run {run_id} {status}: {cycles} cycle(s), "
          f"{written} quotes, {nev} quality events, {no_trade} no-trade "
          f"ticks, {time.time() - t0:.1f}s"
          + (f", cadence mean {cadence['mean_cycle_s']}s" if cadence else ""))


if __name__ == "__main__":
    main()
