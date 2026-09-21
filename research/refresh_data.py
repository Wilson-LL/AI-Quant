"""Incremental TWSE EOD cache refresh (daily collect step of the production loop).

For each cached stock, fetches only the months missing since the last cached
date (usually just the current month) and appends new rows, preserving the
`date,open,high,low,close,volume` schema the whole research stack reads.

Full-field capture: twstock also serves `turnover` (value), `transaction`
(trade count) and `change`. When --full-fields is passed, those columns are
written to a parallel cache `research/data_cache_full/<id>.csv` for the fetched
months, so a true full-field history can accumulate going forward without
breaking the frozen OHLCV snapshot.

Network-bound (~1 request per stock-month, throttled). Resumable; failures skip.

Usage:
  python research/refresh_data.py                  # refresh missing months
  python research/refresh_data.py --full-fields    # also write full-field rows
  python research/refresh_data.py --dry-run        # show what would be fetched
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CACHE_DIR, SECTOR_MAP, load_cached, cache_path  # noqa: E402

FULL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache_full")


def months_between(start, end):
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _twstock_fetch(sid):
    import twstock
    return twstock.Stock(sid).fetch


def refresh_stock(sid, today, full_fields=False, dry_run=False, throttle_s=1.5, fetch_fn=None):
    """Append NEW TAIL DATA only. Contiguity-preserving (H-DATA-INTEGRITY
    root-cause fix): months are fetched oldest-first and the loop STOPS at the
    first month that fails or comes back empty, so rows from a later month are
    never appended past an unfetched month. Previously a failed month was
    skipped, a later month advanced `last`, and every retry recomputed `need`
    from the new `last` -- the hole became permanent and the symbol was then
    reported as refreshed. Holes that already exist are repaired by
    `repair_gaps` (HISTORICAL_MISSING_INTERVALS), never by this path."""
    df = load_cached(sid)
    if df is None or df.empty:
        print(f"[{sid}] no cache — skipped (use data.build_cache for full fetch)")
        return 0, 0
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    last = df["date"].max()
    need = months_between(last, today)
    if not need:
        return 0, 0
    if dry_run:
        print(f"[{sid}] would fetch {len(need)} month(s) from {last.date()}")
        return 0, 0

    fetch = fetch_fn or _twstock_fetch(sid)
    new_rows, full_rows = [], []
    suspect_months = 0
    for (y, m) in need:
        try:
            data = fetch(y, m)
        except Exception as e:  # noqa: BLE001
            print(f"[{sid}] {y}-{m:02d} fetch failed: {type(e).__name__} — "
                  "stopping so no later month is appended past it")
            suspect_months += 1
            time.sleep(throttle_s)
            break
        if not data:
            # Every listed month has trading days, so an EMPTY payload is
            # not "nothing published" — it is the TWSE rate-limit/outage
            # signature (2026-08-24 incident: 74 symbols silently no-oped
            # this way and coverage fell to 34/108). Count it as suspect
            # so the caller can warn + retry instead of swallowing it, and
            # STOP: appending a later month would create a permanent hole.
            print(f"[{sid}] {y}-{m:02d} EMPTY response (possible rate "
                  "limit) — will retry; later months not appended")
            suspect_months += 1
            time.sleep(throttle_s)
            break
        for d in data:
            if d.close is None:
                continue
            new_rows.append({"date": pd.Timestamp(d.date), "open": d.open,
                             "high": d.high, "low": d.low, "close": d.close,
                             "volume": d.capacity})
            if full_fields:
                full_rows.append({"date": pd.Timestamp(d.date), "open": d.open,
                                  "high": d.high, "low": d.low, "close": d.close,
                                  "capacity": d.capacity, "turnover": d.turnover,
                                  "transaction": d.transaction, "change": d.change})
        time.sleep(throttle_s)

    if not new_rows:
        return 0, suspect_months
    add = pd.DataFrame(new_rows)
    add = add[add["date"] > last]
    if add.empty:
        return 0, suspect_months
    # Append only the new tail rows, keeping every existing line of the file
    # byte-identical (no full rewrite of the history).
    n_new = merge_missing_rows(cache_path(sid), add, add["date"])
    if full_fields and full_rows:
        os.makedirs(FULL_DIR, exist_ok=True)
        fp = os.path.join(FULL_DIR, f"{sid}.csv")
        fdf = pd.DataFrame(full_rows)
        if os.path.exists(fp):
            fdf = (pd.concat([pd.read_csv(fp, parse_dates=["date"]), fdf])
                     .sort_values("date").drop_duplicates("date", keep="last"))
        fdf.to_csv(fp, index=False)
    print(f"[{sid}] +{n_new} rows (last now {add['date'].max().date()})")
    return n_new, suspect_months


def backfill_stock(sid, start, dry_run=False, throttle_s=1.5, fetch_fn=None):
    """Prepend history from `start` (YYYY-MM) up to the first cached month.
    Existing cached rows are never modified (keep='last' on the cached side).
    Contiguity-preserving: months are fetched NEWEST-first (adjacent to the
    cached history) and the loop stops at the first failed/empty month, so a
    failure can never leave a hole between prepended and cached rows."""
    df = load_cached(sid)
    if df is None or df.empty:
        print(f"[{sid}] no cache — skipped")
        return 0
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    first = df["date"].min()
    start_ts = pd.Timestamp(f"{start}-01")
    end_ts = (first - pd.offsets.MonthBegin(1))
    if start_ts > end_ts:
        return 0
    need = months_between(start_ts, end_ts)
    if dry_run:
        print(f"[{sid}] would backfill {len(need)} month(s) "
              f"{need[0][0]}-{need[0][1]:02d}..{need[-1][0]}-{need[-1][1]:02d}")
        return 0

    fetch = fetch_fn or _twstock_fetch(sid)
    new_rows = []
    for (y, m) in reversed(need):
        try:
            data = fetch(y, m)
        except Exception as e:  # noqa: BLE001
            print(f"[{sid}] {y}-{m:02d} fetch failed: {type(e).__name__} — "
                  "stopping (older months not prepended past it)")
            time.sleep(throttle_s)
            break
        if not data:
            print(f"[{sid}] {y}-{m:02d} EMPTY response — stopping (older "
                  "months not prepended past it)")
            time.sleep(throttle_s)
            break
        for d in data:
            if d.close is None:
                continue
            new_rows.append({"date": pd.Timestamp(d.date), "open": d.open,
                             "high": d.high, "low": d.low, "close": d.close,
                             "volume": d.capacity})
        time.sleep(throttle_s)
    if not new_rows:
        return 0
    add = pd.DataFrame(new_rows)
    add = add[add["date"] < first]
    if add.empty:
        return 0
    out = (pd.concat([add, df], ignore_index=True)
             .sort_values("date").drop_duplicates("date", keep="last"))
    out.to_csv(cache_path(sid), index=False)
    print(f"[{sid}] backfilled +{len(add)} rows (first now {out['date'].min().date()})")
    return len(add)


# ------------------------------------------------------------ HISTORICAL_MISSING_INTERVALS repair
# A symbol being current at its newest date does NOT imply its cache is
# complete. Existing holes are first-class outstanding work: they are
# discovered against the expected market calendar (research/eod_integrity),
# fetched month by month, validated, merged WITHOUT touching existing rows,
# re-audited, and anything still missing stays explicitly outstanding in a
# resumable state file. Nothing is forward-filled, interpolated or invented.

REPAIR_MAX_ATTEMPTS = 3          # per symbol-month, across runs (resumable)
REPAIR_SUSPECT_COOLDOWN_AFTER = 3   # consecutive suspect responses -> cooldown
REPAIR_SUSPECT_ABORT_AFTER = 6      # consecutive suspect responses -> stop this run
REPAIR_STATE_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "reports", "data_integrity", "eod_gap_repair_state.json")


def classify_fetch_exception(exc):
    import eod_integrity as E
    msg = f"{type(exc).__name__} {exc}".lower()
    if any(k in msg for k in ("429", "too many", "rate", "throttl", "forbidden", "403")):
        return E.RATE_LIMIT_SUSPECT
    return E.NETWORK_ERROR


def _row_ok(r):
    """A plausible EOD row: positive finite close; any present open/high/low
    positive; high >= low when both present. (A row with a close but a missing
    open is kept, as the previous writer did, rather than dropped into a hole.)"""
    def num(k):
        try:
            v = float(r[k])
        except (TypeError, ValueError, KeyError):
            return None
        return v if v == v else None
    c = num("close")
    if c is None or c <= 0:
        return False
    o, h, lo = num("open"), num("high"), num("low")
    if any(v is not None and v <= 0 for v in (o, h, lo)):
        return False
    return not (h is not None and lo is not None and h < lo)


def rows_from_payload(data):
    rows = []
    for d in data or []:
        if getattr(d, "close", None) is None:
            continue
        rows.append({"date": pd.Timestamp(d.date), "open": d.open, "high": d.high,
                     "low": d.low, "close": d.close, "volume": d.capacity})
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def merge_missing_rows(path, add, allowed_dates):
    """Insert ONLY rows whose date is an expected-but-missing session listed in
    `allowed_dates` and is not already cached. Every existing line of the file
    is kept byte-identical (same order, same formatting, same line ending); new
    lines are inserted at their date position. Returns the number of rows
    added. Refuses to touch a file whose existing dates are not sorted."""
    with open(path, "rb") as f:
        raw = f.read()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = raw.split(eol)
    trailing = lines[-1] == b""
    if trailing:
        lines = lines[:-1]
    header, body = lines[0], lines[1:]
    have = [b.split(b",", 1)[0].decode() for b in body]
    if have != sorted(have):
        raise ValueError(f"{path}: existing dates are not sorted; refusing to merge")
    allowed = {pd.Timestamp(x).strftime("%Y-%m-%d") for x in allowed_dates}
    add = add.copy()
    add["d"] = pd.to_datetime(add["date"]).dt.strftime("%Y-%m-%d")
    add = add[add["d"].isin(allowed) & ~add["d"].isin(set(have))]
    add = add[[_row_ok(r) for _, r in add.iterrows()]] if len(add) else add
    add = add.drop_duplicates("d").sort_values("d")
    if add.empty:
        return 0
    cols = header.decode().split(",")
    fmt = add.assign(date=add["d"])[cols].to_csv(index=False, header=False, lineterminator="\n")
    new_lines = [x.encode() for x in fmt.strip("\n").split("\n")]
    import bisect
    out = list(body)
    keys = list(have)
    for nl in new_lines:
        k = nl.split(b",", 1)[0].decode()
        pos = bisect.bisect_left(keys, k)
        keys.insert(pos, k)
        out.insert(pos, nl)
    data = eol.join([header] + out) + (eol if trailing else b"")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return len(new_lines)


def _load_state(state_path):
    if state_path and os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state_path, state):
    if not state_path:
        return
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, state_path)


def repair_gaps(cache_dir, calendar, requests, state_path=REPAIR_STATE_DEFAULT, max_requests=20,
                registry_path=None,
                throttle_s=1.5, cooldown_s=120.0, fetch_factory=None, sleep_fn=time.sleep):
    """Bounded, resumable repair of HISTORICAL_MISSING_INTERVALS.
    `requests`: ordered list of (symbol, 'YYYY-MM') to attempt (priority order).
    For each: recompute the expected-but-missing sessions of that month inside
    the symbol's cached span, drop sessions the no-trade registry explicitly
    confirms (never fetched, never filled), fetch the month, keep only rows for
    the exact missing sessions, merge without touching existing rows, and
    record the state:
      FETCH_OK                     every wanted session is now present (COMPLETE)
      CONFIRMED_SYMBOL_NO_TRADE    every remaining session is registry-confirmed
                                   (e.g. an exchange-announced suspension)
      MARKET_OPEN_SYMBOL_NO_DATA   the source returned the month but not these
                                   sessions: symbol-specific no-data with
                                   SUSPENSION_STATUS_UNKNOWN. Terminal for the
                                   queue (not re-downloaded forever) and never
                                   COMPLETE; it stays an unresolved exception
      EMPTY_RESPONSE_SUSPECT / NETWORK_ERROR / RATE_LIMIT_SUSPECT /
      INVALID_PAYLOAD_SUSPECT      retryable; after REPAIR_MAX_ATTEMPTS -> UNRESOLVED
    An EMPTY payload is never read as 'month had no trading'. Every session that
    is still missing keeps breaking contiguity for the model (gap guard)."""
    import eod_integrity as E
    cal = pd.DatetimeIndex(calendar)
    state = _load_state(state_path)
    confirmed = E.load_no_trade_registry(registry_path)
    used, consecutive_suspect, log = 0, 0, []
    terminal = (E.FETCH_OK, E.CONFIRMED_SYMBOL_NO_TRADE, E.MARKET_OPEN_SYMBOL_NO_DATA, E.UNRESOLVED)
    for sym, month in requests:
        key = f"{sym}|{month}"
        st = state.get(key, {"attempts": 0, "state": None, "added": 0})
        nd = st.get("no_data_sessions") or []
        if (st["state"] == E.MARKET_OPEN_SYMBOL_NO_DATA and nd
                and all((sym, d) in confirmed for d in nd)):
            # every no-data session is now explicitly registry-confirmed: the
            # queue item is resolved (no fetch, no row, contiguity stays broken)
            st.update(state=E.CONFIRMED_SYMBOL_NO_TRADE, registry_confirmed_no_trade=list(nd),
                      note="upgraded from MARKET_OPEN_SYMBOL_NO_DATA via no-trade registry")
            st.pop("suspension_status", None)
            state[key] = st
            log.append({"symbol": sym, "month": month, **st})
            continue
        if st["state"] in terminal:
            continue
        if st["attempts"] >= REPAIR_MAX_ATTEMPTS:
            st["state"] = E.UNRESOLVED
            st["suspension_status"] = E.SUSPENSION_STATUS_UNKNOWN
            state[key] = st
            continue
        if used >= max_requests:
            break
        path = os.path.join(cache_dir, f"{sym}.csv")
        dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"])
        p = pd.Period(month, "M")
        span = cal[(cal >= dates.min()) & (cal <= dates.max())]
        missing = span[(span >= p.start_time) & (span <= p.end_time)].difference(pd.DatetimeIndex(dates))
        conf = [d for d in missing if (sym, str(d)[:10]) in confirmed]
        want = missing.difference(pd.DatetimeIndex(conf))
        st["expected_missing"] = int(len(missing))
        st["registry_confirmed_no_trade"] = [str(d)[:10] for d in conf]
        if len(want) == 0:
            st.update(state=E.CONFIRMED_SYMBOL_NO_TRADE if conf else E.FETCH_OK,
                      note="no fetch needed", remaining=0)
            state[key] = st
            log.append({"symbol": sym, "month": month, **st})
            continue
        fetch = (fetch_factory or _twstock_fetch)(sym)
        st["attempts"] += 1
        used += 1
        rec = {"rows_returned": 0, "rows_accepted": 0, "rows_rejected": 0, "rows_added": 0}
        try:
            payload = fetch(p.year, p.month)
        except Exception as exc:  # noqa: BLE001
            st["state"] = classify_fetch_exception(exc)
            payload = None
        if payload is not None and not payload:
            st["state"] = E.EMPTY_RESPONSE_SUSPECT
        if payload:
            rows = rows_from_payload(payload)
            rec["rows_returned"] = int(len(rows))
            rows = rows[(rows["date"] >= p.start_time) & (rows["date"] <= p.end_time)]
            wanted = rows[rows["date"].isin(want)]
            ok_mask = np.array([_row_ok(r) for _, r in wanted.iterrows()], dtype=bool)
            rec["rows_accepted"] = int(ok_mask.sum())
            rec["rows_rejected"] = int((~ok_mask).sum())
            rec["rows_added"] = merge_missing_rows(path, wanted[ok_mask], want) if ok_mask.any() else 0
            got = set(pd.to_datetime(wanted[ok_mask]["date"]))
            still = [d for d in want if d not in got]
            st["added"] = st.get("added", 0) + rec["rows_added"]
            if not still:
                st["state"] = E.CONFIRMED_SYMBOL_NO_TRADE if conf else E.FETCH_OK
            elif rec["rows_rejected"]:
                st["state"] = E.INVALID_PAYLOAD_SUSPECT
            elif len(rows):
                st["state"] = E.MARKET_OPEN_SYMBOL_NO_DATA
                st["suspension_status"] = E.SUSPENSION_STATUS_UNKNOWN
                st["no_data_sessions"] = [str(d)[:10] for d in still]
            else:
                st["state"] = E.EMPTY_RESPONSE_SUSPECT
            st["remaining"] = int(len(still))
        else:
            st["remaining"] = int(len(want))
        suspect = st["state"] in E.RETRYABLE_STATES
        if suspect and st["attempts"] >= REPAIR_MAX_ATTEMPTS:
            st["state"] = E.UNRESOLVED
            st["suspension_status"] = E.SUSPENSION_STATUS_UNKNOWN
        st.update(last_request=rec)
        state[key] = st
        _save_state(state_path, state)
        log.append({"symbol": sym, "month": month, **{k: v for k, v in st.items() if k != "last_request"},
                    **rec})
        consecutive_suspect = consecutive_suspect + 1 if suspect else 0
        if consecutive_suspect >= REPAIR_SUSPECT_ABORT_AFTER:
            print(f"[repair] {consecutive_suspect} consecutive suspect responses — stopping (resumable)")
            break
        sleep_fn(cooldown_s if consecutive_suspect >= REPAIR_SUSPECT_COOLDOWN_AFTER else throttle_s)
    _save_state(state_path, state)
    return log, state


def plan_requests(plan_csv, priorities=("P0",)):
    plan = pd.read_csv(plan_csv, dtype={"symbol": str})
    plan = plan[plan["priority"].isin(priorities)]
    order = {p: i for i, p in enumerate(("P0", "P1", "P2"))}
    plan = plan.assign(_o=plan["priority"].map(order)).sort_values(["_o", "symbol", "fetch_month"])
    return list(dict.fromkeys(zip(plan["symbol"], plan["fetch_month"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-fields", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--universe", nargs="*", default=None)
    ap.add_argument("--backfill-start", default=None, metavar="YYYY-MM",
                    help="prepend history from this month to the first cached "
                         "date (e.g. 2015-01); skips the forward refresh")
    ap.add_argument("--retry-cooldown", type=float, default=120.0,
                    help="base cooldown before the first retry pass; each "
                         "further pass doubles it")
    ap.add_argument("--max-retry-passes", type=int, default=3,
                    help="bounded number of retry passes over symbols "
                         "whose fetch came back empty/failed")
    ap.add_argument("--repair-gaps", action="store_true",
                    help="repair HISTORICAL_MISSING_INTERVALS from the backfill "
                         "plan (bounded, resumable); skips the tail refresh")
    ap.add_argument("--repair-plan", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "data_integrity", "eod_backfill_plan.csv"))
    ap.add_argument("--repair-priorities", nargs="*", default=["P0"])
    ap.add_argument("--max-requests", type=int, default=20)
    ap.add_argument("--repair-state", default=REPAIR_STATE_DEFAULT)
    ap.add_argument("--repair-log", default=None, help="append a per-request CSV log")
    ap.add_argument("--registry", default=None,
                    help="symbol no-trade registry CSV (default reports/data_integrity/)")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()

    ids = args.universe or sorted(SECTOR_MAP)
    today = pd.Timestamp.today().normalize()
    t0 = time.time()
    if os.path.abspath(args.cache_dir) != os.path.abspath(CACHE_DIR):
        import data as _data       # tail refresh / backfill read+write via data.cache_path
        _data.CACHE_DIR = args.cache_dir
    if args.repair_gaps:
        import eod_integrity as E
        raw = E.read_cache_dates(args.cache_dir)
        cal = E.derive_calendar({s: raw[s] for s in E.model_eligible(sorted(raw))})
        reqs = plan_requests(args.repair_plan, tuple(args.repair_priorities))
        if args.dry_run:
            print(f"[repair] would attempt {min(len(reqs), args.max_requests)} of "
                  f"{len(reqs)} symbol-month requests ({args.repair_priorities})")
            return
        log, state = repair_gaps(args.cache_dir, cal, reqs, args.repair_state,
                                 max_requests=args.max_requests, registry_path=args.registry)
        for r in log:
            print(f"[repair] {r['symbol']} {r['month']}: {r['state']} "
                  f"(expected {r.get('expected_missing')}, returned {r.get('rows_returned', 0)}, "
                  f"accepted {r.get('rows_accepted', 0)}, added {r.get('rows_added', 0)}, "
                  f"rejected {r.get('rows_rejected', 0)}, remaining {r.get('remaining')}, "
                  f"attempts {r['attempts']})")
        if args.repair_log and log:
            cols = ["symbol", "month", "state", "expected_missing", "rows_returned", "rows_accepted",
                    "rows_added", "rows_rejected", "remaining", "attempts", "suspension_status",
                    "no_data_sessions", "registry_confirmed_no_trade"]
            df = pd.DataFrame([{c: (";".join(r[c]) if isinstance(r.get(c), list) else r.get(c))
                                for c in cols} for r in log])
            df.insert(0, "run_at", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            df.to_csv(args.repair_log, mode="a", index=False,
                      header=not os.path.isfile(args.repair_log))
        outstanding = [k for k, v in state.items()
                       if v.get("state") not in (E.FETCH_OK, E.CONFIRMED_SYMBOL_NO_TRADE)]
        print(f"[repair] outstanding symbol-months: {len(outstanding)}")
        return
    if args.backfill_start:
        total = sum(backfill_stock(sid, args.backfill_start, args.dry_run)
                    for sid in ids)
        suspects = []
    elif args.dry_run:
        total = sum(refresh_stock(sid, today, args.full_fields, True)[0]
                    for sid in ids)
        suspects = []
    else:
        cooldowns = tuple(args.retry_cooldown * (2 ** i)
                          for i in range(max(0, args.max_retry_passes)))
        total, suspects = refresh_with_retries(
            ids, today, args.full_fields, cooldowns=cooldowns)
    print(f"\nRefresh done: +{total} rows across {len(ids)} stocks "
          f"in {time.time() - t0:.0f}s")
    if suspects:
        print(f"WARNING: {len(suspects)} symbol(s) still returned no data "
              "for requested months after all retry passes (possible "
              "TWSE rate limit/outage) — coverage may be PARTIAL: "
              f"{', '.join(suspects[:20])}"
              f"{' ...' if len(suspects) > 20 else ''}")


def refresh_with_retries(ids, today, full_fields=False,
                         cooldowns=(120.0, 240.0, 480.0),
                         sleep_fn=time.sleep):
    """Bounded rate-limit recovery (2026-08-24 incident): one full pass,
    then at most len(cooldowns) retry passes over still-suspect symbols
    with escalating cooldowns. A suspect = a symbol whose NEEDED month
    fetch raised or came back empty (every listed month has trading
    days, so empty = rate limit/outage, not "nothing published"); an
    up-to-date symbol is never suspect. No rows are fabricated; retries
    re-fetch from source and the date>last + drop_duplicates path makes
    them idempotent. Persistent failures stay in the returned suspect
    list (explicit final failure) and the cross-sectional coverage gate
    remains the authoritative publication block."""
    total = 0
    suspects = []
    for sid in ids:
        added, susp = refresh_stock(sid, today, full_fields)
        total += added
        if susp:
            suspects.append(sid)
    for i, cd in enumerate(cooldowns):
        if not suspects:
            break
        print(f"\n{len(suspects)} symbol(s) returned empty/failed "
              f"responses — retry pass {i + 1}/{len(cooldowns)} after "
              f"{cd:.0f}s cooldown")
        sleep_fn(cd)
        still = []
        for sid in suspects:
            added, susp = refresh_stock(sid, today, full_fields)
            total += added
            if susp:
                still.append(sid)
        suspects = still
    return total, suspects


if __name__ == "__main__":
    main()
