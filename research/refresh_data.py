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
import os
import sys
import time

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


def refresh_stock(sid, today, full_fields=False, dry_run=False, throttle_s=1.5):
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

    import twstock
    stock = twstock.Stock(sid)
    new_rows, full_rows = [], []
    suspect_months = 0
    for (y, m) in need:
        try:
            data = stock.fetch(y, m)
        except Exception as e:  # noqa: BLE001
            print(f"[{sid}] {y}-{m:02d} fetch failed: {type(e).__name__}")
            suspect_months += 1
            time.sleep(throttle_s)
            continue
        if not data:
            # Every listed month has trading days, so an EMPTY payload is
            # not "nothing published" — it is the TWSE rate-limit/outage
            # signature (2026-08-24 incident: 74 symbols silently no-oped
            # this way and coverage fell to 34/108). Count it as suspect
            # so the caller can warn + retry instead of swallowing it.
            print(f"[{sid}] {y}-{m:02d} EMPTY response (possible rate "
                  "limit) — will retry")
            suspect_months += 1
            time.sleep(throttle_s)
            continue
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
    out = (pd.concat([df, add], ignore_index=True)
             .sort_values("date").drop_duplicates("date", keep="last"))
    out.to_csv(cache_path(sid), index=False)
    if full_fields and full_rows:
        os.makedirs(FULL_DIR, exist_ok=True)
        fp = os.path.join(FULL_DIR, f"{sid}.csv")
        fdf = pd.DataFrame(full_rows)
        if os.path.exists(fp):
            fdf = (pd.concat([pd.read_csv(fp, parse_dates=["date"]), fdf])
                     .sort_values("date").drop_duplicates("date", keep="last"))
        fdf.to_csv(fp, index=False)
    print(f"[{sid}] +{len(add)} rows (last now {out['date'].max().date()})")
    return len(add), suspect_months


def backfill_stock(sid, start, dry_run=False, throttle_s=1.5):
    """Prepend history from `start` (YYYY-MM) up to the first cached month.
    Existing cached rows are never modified (keep='last' on the cached side)."""
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

    import twstock
    stock = twstock.Stock(sid)
    new_rows = []
    for (y, m) in need:
        try:
            data = stock.fetch(y, m)
        except Exception as e:  # noqa: BLE001
            print(f"[{sid}] {y}-{m:02d} fetch failed: {type(e).__name__}")
            time.sleep(throttle_s)
            continue
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
    args = ap.parse_args()

    ids = args.universe or sorted(SECTOR_MAP)
    today = pd.Timestamp.today().normalize()
    t0 = time.time()
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
