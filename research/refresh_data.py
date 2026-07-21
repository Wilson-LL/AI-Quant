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
        return 0
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    last = df["date"].max()
    need = months_between(last, today)
    if not need:
        return 0
    if dry_run:
        print(f"[{sid}] would fetch {len(need)} month(s) from {last.date()}")
        return 0

    import twstock
    stock = twstock.Stock(sid)
    new_rows, full_rows = [], []
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
            if full_fields:
                full_rows.append({"date": pd.Timestamp(d.date), "open": d.open,
                                  "high": d.high, "low": d.low, "close": d.close,
                                  "capacity": d.capacity, "turnover": d.turnover,
                                  "transaction": d.transaction, "change": d.change})
        time.sleep(throttle_s)

    if not new_rows:
        return 0
    add = pd.DataFrame(new_rows)
    add = add[add["date"] > last]
    if add.empty:
        return 0
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
    return len(add)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-fields", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--universe", nargs="*", default=None)
    args = ap.parse_args()

    ids = args.universe or sorted(SECTOR_MAP)
    today = pd.Timestamp.today().normalize()
    t0 = time.time()
    total = 0
    for sid in ids:
        total += refresh_stock(sid, today, args.full_fields, args.dry_run)
    print(f"\nRefresh done: +{total} rows across {len(ids)} stocks "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
