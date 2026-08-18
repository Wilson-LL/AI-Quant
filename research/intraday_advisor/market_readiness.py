"""v16 Stage C2 — session-level market-readiness gate for the morning
live refresh.

Data-quality requirement, NOT an alpha optimization: the v15 collector
polls ~once per minute and the opening market state can still be
initializing at 09:00:00, so a normal actionable refresh must not trust
the first print. Constants are centralized here and are not tuned from
trading returns.

Session-level ONLY: individual missing/stale symbols are handled by the
C1 row-level gates; this module decides whether the SESSION is ready.
READ-ONLY on the v15 DB.
"""

import datetime as _dt
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live_market_state as lms  # noqa: E402

SESSION_OPEN = _dt.time(9, 0)          # TWSE regular session open
SESSION_CLOSE = _dt.time(13, 30)       # regular session close
MIN_AFTER_OPEN_SECONDS = 120           # pre-registered; ordinary
#                                        actionable refresh from ~09:02
POLL_SECONDS_DEFAULT = 15
MAX_WAIT_SECONDS_DEFAULT = 300


def assess(db_path, intended_date, now=None):
    """-> (status, detail). Statuses: SESSION_MISMATCH /
    WAITING_FOR_MARKET_OPEN / WAITING_FOR_MARKET_DATA / MARKET_READY /
    MARKET_CLOSED."""
    now = now or _dt.datetime.now()
    intended = str(intended_date)[:10]
    if now.date().isoformat() != intended:
        return "SESSION_MISMATCH", (f"current date {now.date()} != "
                                    f"intended session {intended}")
    open_dt = _dt.datetime.combine(now.date(), SESSION_OPEN)
    close_dt = _dt.datetime.combine(now.date(), SESSION_CLOSE)
    if now > close_dt:
        return "MARKET_CLOSED", ("regular session closed at "
                                 f"{SESSION_CLOSE}")
    gate = open_dt + _dt.timedelta(seconds=MIN_AFTER_OPEN_SECONDS)
    if now < gate:
        return "WAITING_FOR_MARKET_OPEN", (
            f"normal actionable refresh begins at {gate.time()} "
            f"(open + {MIN_AFTER_OPEN_SECONDS}s)")
    # current-session rows must exist and the newest must be fresh
    # enough under the existing C1 thresholds (session-level check)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                              timeout=30)
        try:
            row = con.execute(
                "SELECT MAX(timestamp) FROM intraday_quotes WHERE "
                "source=? AND substr(timestamp,1,10)=?",
                (lms.SOURCE, intended)).fetchone()
        finally:
            con.close()
    except sqlite3.OperationalError as e:
        return "WAITING_FOR_MARKET_DATA", f"collector DB not readable: {e}"
    latest = row[0] if row else None
    if not latest:
        return "WAITING_FOR_MARKET_DATA", ("no current-session collector "
                                           "rows yet")
    ts = lms.parse_ts(latest)
    age = (now - ts).total_seconds() if ts else None
    if age is None or age > lms.AGING_S:
        return "WAITING_FOR_MARKET_DATA", (
            f"newest collector row is {age:.0f}s old "
            f"(> {lms.AGING_S}s)" if age is not None
            else "unparseable collector timestamp")
    return "MARKET_READY", f"newest collector row {age:.0f}s old"
