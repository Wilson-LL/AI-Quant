"""EOD cache trading-session integrity (H-DATA-INTEGRITY, P0 data defect).

The per-symbol EOD cache can contain missing trading sessions, including
whole missing months, and the feature pipeline builds features, 60-step
sequences and forward-return labels on each symbol's OWN rows. A hole is
therefore silently spliced over: `log_ret_1` can carry a multi-week return
as if it were one session, and every rolling lookback shifts.

This module is the single source of truth for:

* the EXPECTED market-session calendar (no authoritative TWSE holiday
  calendar exists in this repo, so it is DERIVED from cross-sectional
  coverage; see CALENDAR_METHOD);
* per-symbol gap auditing (missing sessions, duplicates, non-monotonic
  dates, off-calendar rows, full-month holes, latest contiguous run);
* the three integrity levels (A live inference, B training, C history);
* the required contiguous window, DERIVED from the feature code;
* gap flags used by the dataset / inference guard;
* the deterministic backfill plan and the data manifest.

It never modifies the cache. It never forward-fills or interpolates.

  python research/eod_integrity.py audit [--cache-dir DIR] [--out DIR]
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

SCHEMA_VERSION = "eod_integrity/1"
CALENDAR_METHOD = "CROSS_SECTIONAL_MAJORITY_v1"
CALENDAR_MIN_SHARE = 0.5     # share of symbols alive on the date that have a row
CALENDAR_MIN_SYMBOLS = 10    # absolute floor
SEQ_LEN = 60                 # production preset sequence length (train_transformer_eod.PRESETS)
EXEC_LAG, HORIZON = 1, 20    # production target: fwd_20 from close[t+1]
RECENT_TRAINING_SESSIONS = 504   # "recent training data" for backfill priority P1 (~2 years)

# fetch/repair states (refresh_data gap repair)
FETCH_OK = "FETCH_OK"                                  # all wanted sessions now present (COMPLETE)
EMPTY_RESPONSE_SUSPECT = "EMPTY_RESPONSE_SUSPECT"      # empty payload: never "no trading"
NETWORK_ERROR = "NETWORK_ERROR"
RATE_LIMIT_SUSPECT = "RATE_LIMIT_SUSPECT"
INVALID_PAYLOAD_SUSPECT = "INVALID_PAYLOAD_SUSPECT"    # rows returned but failed validation
UNRESOLVED = "UNRESOLVED"                              # retry budget exhausted
# symbol-specific no-trade semantics: MARKET OPEN does not imply EVERY STOCK HAS A ROW
MARKET_OPEN_SYMBOL_NO_DATA = "MARKET_OPEN_SYMBOL_NO_DATA"  # source served the month, not these days
CONFIRMED_SYMBOL_NO_TRADE = "CONFIRMED_NO_TRADE"           # explicit registry entry (exchange record)
REGISTRY_CONFIRMED_STATUSES = ("CONFIRMED_NO_TRADE", "CONFIRMED_SYMBOL_NO_TRADE")
SUSPENSION_STATUS_UNKNOWN = "SUSPENSION_STATUS_UNKNOWN"
FETCH_STATES = (FETCH_OK, EMPTY_RESPONSE_SUSPECT, NETWORK_ERROR, RATE_LIMIT_SUSPECT,
                INVALID_PAYLOAD_SUSPECT, UNRESOLVED, MARKET_OPEN_SYMBOL_NO_DATA,
                CONFIRMED_SYMBOL_NO_TRADE)
RETRYABLE_STATES = (EMPTY_RESPONSE_SUSPECT, NETWORK_ERROR, RATE_LIMIT_SUSPECT, INVALID_PAYLOAD_SUSPECT)
COMPLETE_STATES = (FETCH_OK,)
RESOLVED_EXCEPTION_STATES = (CONFIRMED_SYMBOL_NO_TRADE,)
UNRESOLVED_EXCEPTION_STATES = (MARKET_OPEN_SYMBOL_NO_DATA, UNRESOLVED)
NO_TRADE_REGISTRY_DEFAULT = os.path.join(ROOT, "reports", "data_integrity", "symbol_no_trade_registry.csv")


def load_no_trade_registry(path=None):
    """Explicit, curated symbol-specific no-trade sessions {(symbol, 'YYYY-MM-DD')}.
    Only entries with status CONFIRMED_SYMBOL_NO_TRADE and a non-empty source
    count. Such sessions are never fetched or filled, but they still break
    model contiguity (the gap guard treats every missing expected session as
    a break unless a model/data convention explicitly supports it)."""
    path = path or NO_TRADE_REGISTRY_DEFAULT
    if not os.path.isfile(path):
        return set()
    r = pd.read_csv(path, dtype=str).fillna("")
    r = r[r["status"].isin(REGISTRY_CONFIRMED_STATUSES) & (r["source"].str.strip() != "")]
    return set(zip(r["symbol"], r["date"]))


def missing_exceptions(state_path=None, registry_path=None):
    """{(symbol, date): category} for missing sessions with a known status:
    CONFIRMED_SYMBOL_NO_TRADE (registry), MARKET_OPEN_SYMBOL_NO_DATA /
    UNRESOLVED (repair state)."""
    out = {k: CONFIRMED_SYMBOL_NO_TRADE for k in load_no_trade_registry(registry_path)}
    if state_path and os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        for key, v in st.items():
            sym = key.split("|")[0]
            for d in v.get("no_data_sessions", []):
                out.setdefault((sym, d), MARKET_OPEN_SYMBOL_NO_DATA)
            if v.get("state") == UNRESOLVED:
                out.setdefault((sym, key.split("|")[1]), UNRESOLVED)   # month-level
    return out

# ------------------------------------------------------------ live (symbol-level) integrity policy
# Two levels (user decision 2026-09-21):
#   HARD BLOCK  any invalid symbol that is an open position (my_holdings.csv) or
#               held in the standing model book -> UNSAFE_FOR_NEW_MODEL_OUTPUT,
#               previous plan preserved.
#   DEGRADE     invalid NON-held symbols are not scored, reported as
#               DATA_INTEGRITY_FAILURE and removed from that session's model
#               cross-section; publication is allowed only if
#               valid / otherwise-eligible >= VALID_MODEL_COVERAGE_MIN.
# This is deliberately separate from the newest-date coverage gate
# (user_next_session_plan.PARTIAL_COVERAGE_MIN), which answers a different
# question (did the latest session publish for the cross-section?).
VALID_MODEL_COVERAGE_MIN = 0.99
SAFE, DEGRADED, UNSAFE = "SAFE", "DEGRADED", "UNSAFE_FOR_NEW_MODEL_OUTPUT"
CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW = "CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW"
MISSING_SESSION_IN_REQUIRED_WINDOW = "MISSING_SESSION_IN_REQUIRED_WINDOW"


def classify_live_integrity(eligible, invalid, held, min_ratio=VALID_MODEL_COVERAGE_MIN):
    """Pure policy. eligible: otherwise-eligible model symbols; invalid: symbols
    whose required inference window is not contiguous; held: open positions and
    standing-book holdings. Never 'one symbol is fine': the threshold is the
    ratio rule, and a held invalid symbol always blocks."""
    eligible, invalid, held = set(eligible), set(invalid) & set(eligible), set(held)
    valid = eligible - invalid
    ratio = len(valid) / max(len(eligible), 1)
    held_invalid = sorted(invalid & held)
    if held_invalid:
        status, reason = UNSAFE, f"held/position symbol(s) invalid: {held_invalid}"
    elif ratio < min_ratio:
        status, reason = UNSAFE, f"valid model coverage {len(valid)}/{len(eligible)} = {ratio:.2%} < {min_ratio:.0%}"
    elif invalid:
        status, reason = DEGRADED, f"{len(invalid)} non-held symbol(s) excluded; valid {len(valid)}/{len(eligible)} = {ratio:.2%}"
    else:
        status, reason = SAFE, "no invalid current model symbols"
    return {"status": status, "reason": reason, "eligible": len(eligible), "valid": len(valid),
            "valid_ratio": ratio, "threshold": min_ratio, "excluded": sorted(invalid),
            "held_invalid": held_invalid, "publication_allowed": status != UNSAFE}


def window_failure_reasons(cache_dir, symbols, calendar, asof, req=None, registry_path=None):
    """{symbol: {"reason", "missing_in_window"}} for symbols whose required
    inference window ending at `asof` misses expected sessions. A window whose
    only missing sessions are registry-confirmed no-trade days is labelled
    CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW (still invalid: a no-trade day
    breaks contiguity); anything else MISSING_SESSION_IN_REQUIRED_WINDOW."""
    req = req or required_windows()
    n = req["REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS"]
    cal = pd.DatetimeIndex(calendar)
    win = cal[cal <= pd.Timestamp(asof)][-n:]
    conf = load_no_trade_registry(registry_path)
    out = {}
    raw = read_cache_dates(cache_dir, list(symbols))
    for s, d in raw.items():
        miss = win.difference(pd.DatetimeIndex(pd.unique(d)))
        if len(miss) == 0:
            continue
        ms = [str(x)[:10] for x in miss]
        allconf = all((s, x) in conf for x in ms)
        out[s] = {"reason": CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW if allconf else MISSING_SESSION_IN_REQUIRED_WINDOW,
                  "missing_in_window": ms}
    return out


def integrity_window_excluded(pred_dir, asof):
    """Symbols the live integrity policy removed from the `asof` model
    cross-section because their required window crosses a missing session
    (reason WINDOW_CROSSES_DATA_GAP in <asof>_data_integrity.csv). Stale-tail
    names are NOT included: they were never scored before the integrity fix
    either, so they never counted toward the portfolio denominator."""
    p = os.path.join(pred_dir, f"{asof}_data_integrity.csv")
    if not os.path.isfile(p):
        return set()
    d = pd.read_csv(p, dtype={"symbol": str})
    if d.empty:
        return set()
    return set(d.loc[d["reason"] == "WINDOW_CROSSES_DATA_GAP", "symbol"])


def reference_universe_n(valid_symbols, excluded, eligible_pool=None):
    """REFERENCE_ELIGIBLE_UNIVERSE_N: the otherwise-eligible model universe
    BEFORE temporary data-integrity exclusions = valid scored names + excluded
    names that would otherwise have entered the same cross-section
    (`eligible_pool`, e.g. names with a momentum score for the blend book).
    Every denominator-based portfolio parameter (top-N, band, watch list) uses
    this N; ranks and candidates come from valid names only, so an excluded
    name's slot is filled by the next valid name. Excluded names never get a
    score of any kind."""
    valid = set(valid_symbols)
    extra = set(excluded) - valid
    if eligible_pool is not None:
        extra &= set(eligible_pool)
    return len(valid) + len(extra)


DATA_INTEGRITY_EXCLUSION = "DATA_INTEGRITY_EXCLUSION"
INTEGRITY_EXIT_CAVEAT = ("DATA_INTEGRITY_EXCLUSION; NOT AN ALPHA-DRIVEN SELL; signal_driven=false; "
                         "previous model-book name removed only because its current input window "
                         "is invalid")


def is_integrity_exit(text):
    """True for a book row whose caveats carry the DATA_INTEGRITY_EXCLUSION tag."""
    return isinstance(text, str) and text.startswith(DATA_INTEGRITY_EXCLUSION)


def held_symbols(root):
    """REAL_HELD_SYMBOLS: open positions (any side; unparseable qty still
    counts) in the canonical production holdings file <root>/my_holdings.csv.
    This is the ONLY hard-block source of the live integrity policy. The
    previous model book is model state, not the user's holdings (see
    previous_book_symbols). Returns (set, sources)."""
    hp = os.path.join(root, "my_holdings.csv")
    if not os.path.isfile(hp):
        return set(), []
    import holdings as H
    lots, _ = H.load_lots(hp)
    pos, _ = H.aggregate_positions(lots)
    syms = set(pos["symbol"].astype(str))
    return syms, [f"my_holdings.csv ({len(syms)} open positions)"]


def previous_book_symbols(root):
    """PREVIOUS_BOOK_SYMBOLS: names with target_weight > 0 in the latest
    blend50_band10 decision book: band10 / hysteresis continuity and previous
    model-portfolio state only. Never a hard-block source. Returns (set, source)."""
    pt = os.path.join(root, "reports", "paper_trading")
    books = sorted(f for f in os.listdir(pt) if f.endswith("_blend50_band10_decision_book.csv")) \
        if os.path.isdir(pt) else []
    if not books:
        return set(), None
    b = pd.read_csv(os.path.join(pt, books[-1]), dtype={"symbol": str})
    return set(b.loc[b["target_weight"] > 0, "symbol"]), books[-1]


# sample / symbol integrity statuses
WINDOW_CROSSES_DATA_GAP = "WINDOW_CROSSES_DATA_GAP"
DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"
UNKNOWN_LISTING_BOUNDARY = "UNKNOWN_LISTING_BOUNDARY"


# ------------------------------------------------------------ required window (derived from code)

def feature_lookback(feature_set="close_only", n=400):
    """Largest number of PRIOR rows any production feature needs: the index of
    the first row at which every feature is finite on a gap-free synthetic
    series. Derived by running the production feature code, not assumed."""
    from dataset_transformer_eod import FEATURE_COLS, _stock_features
    rng = np.random.default_rng(0)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"date": pd.bdate_range("2000-01-03", periods=n), "open": c, "high": c * 1.01,
                       "low": c * 0.99, "close": c, "volume": np.full(n, 1e6)})
    feats = _stock_features(df, feature_set)
    # per-stock columns only: cross-sectional placeholders are filled at panel level
    cols = [c for c in FEATURE_COLS[feature_set] if c in feats.columns and feats[c].notna().any()]
    f = feats[cols].to_numpy(float)
    ok = np.isfinite(f).all(axis=1)
    if not ok.any():
        raise ValueError(f"{feature_set}: no row with all per-stock features finite in {n} rows")
    return int(np.argmax(ok))


def required_windows(feature_set="close_only", seq_len=SEQ_LEN, exec_lag=EXEC_LAG, horizon=HORIZON):
    L = feature_lookback(feature_set)
    inference = seq_len + L                   # sessions ending at the as-of date, inclusive
    return {"feature_lookback_rows": L, "seq_len": seq_len,
            "REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS": inference,
            "label_forward_sessions": exec_lag + horizon,
            "REQUIRED_TRAINING_CONTIGUOUS_SESSIONS": inference + exec_lag + horizon,
            "derivation": (f"first sequence step needs {L} prior rows (max feature lookback, "
                           f"measured on the production feature code); a {seq_len}-step sequence "
                           f"therefore spans {L} + {seq_len} = {inference} contiguous sessions ending "
                           f"at the as-of date; a training sample additionally needs the "
                           f"{exec_lag}+{horizon} forward sessions of its label")}


# ------------------------------------------------------------ calendar

def read_cache_dates(cache_dir, symbols=None):
    """{symbol: raw date Series in FILE order} (duplicates and order preserved)."""
    out = {}
    names = symbols if symbols is not None else sorted(f[:-4] for f in os.listdir(cache_dir) if f.endswith(".csv"))
    for s in names:
        p = os.path.join(cache_dir, f"{s}.csv")
        if not os.path.isfile(p):
            continue
        d = pd.read_csv(p, usecols=["date"])["date"]
        out[s] = pd.to_datetime(d)
    return out


def derive_calendar(dates_by_symbol, min_share=CALENDAR_MIN_SHARE, min_symbols=CALENDAR_MIN_SYMBOLS):
    """Expected market sessions = dates on which at least `min_share` of the
    symbols whose cached span covers the date (first <= d <= last) have a row,
    and at least `min_symbols` symbols have a row. Ordinary weekends, holidays
    and exchange closures have no (or sparse) coverage and are never sessions.
    No weekday rule: TWSE held Saturday make-up sessions (8 of them in
    2016-2018, each traded by ~104/108 names), which are real sessions."""
    uniq = {s: pd.DatetimeIndex(pd.unique(d)).sort_values() for s, d in dates_by_symbol.items() if len(d)}
    if not uniq:
        return pd.DatetimeIndex([])
    all_d = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in uniq.values()])))
    have = pd.Series(0, index=all_d)
    alive = pd.Series(0, index=all_d)
    for v in uniq.values():
        have[v] += 1
        alive[(all_d >= v[0]) & (all_d <= v[-1])] += 1
    share = have / alive.clip(lower=1)
    keep = (share >= min_share) & (have >= min_symbols)
    return all_d[keep.to_numpy()]


def gap_before(dates, calendar):
    """For sorted unique dates: bool array, True at i if at least one expected
    session lies strictly between dates[i-1] and dates[i]."""
    d = pd.DatetimeIndex(dates)
    if len(d) == 0:
        return np.zeros(0, bool)
    cal = pd.DatetimeIndex(calendar)
    lo = np.searchsorted(cal.values, d.values[:-1], side="right")
    hi = np.searchsorted(cal.values, d.values[1:], side="left")
    return np.concatenate([[False], (hi - lo) > 0])


def contiguous_back(gap, span):
    """True at row i if rows i-span+1..i are contiguous (no gap in transitions
    i-span+2..i) AND at least span rows exist. span counts rows including i."""
    n = len(gap)
    g = np.asarray(gap, int)
    cs = np.concatenate([[0], np.cumsum(g)])
    out = np.zeros(n, bool)
    for i in range(span - 1, n):
        out[i] = (cs[i + 1] - cs[i - span + 2]) == 0 if span > 1 else True
    return out


def contiguous_forward(gap, span):
    """True at row i if rows i..i+span are contiguous (no gap in transitions
    i+1..i+span) and exist."""
    n = len(gap)
    g = np.asarray(gap, int)
    cs = np.concatenate([[0], np.cumsum(g)])
    out = np.zeros(n, bool)
    for i in range(0, n - span):
        out[i] = (cs[i + span + 1] - cs[i + 1]) == 0
    return out


# ------------------------------------------------------------ per-symbol audit

def audit_symbol(sym, raw_dates, calendar, newest, req, exceptions=None):
    raw = pd.DatetimeIndex(raw_dates)
    dup = int(raw.duplicated().sum())
    nonmono = int((np.diff(raw.values.astype("int64")) < 0).sum()) if len(raw) > 1 else 0
    d = pd.DatetimeIndex(pd.unique(raw)).sort_values()
    first, last = d[0], d[-1]
    cal = pd.DatetimeIndex(calendar)
    exp = cal[(cal >= first) & (cal <= last)]
    present = exp.intersection(d)
    missing = exp.difference(d)
    exceptions = exceptions or {}
    cat = []
    for m in missing:
        ds = str(m)[:10]
        c = exceptions.get((sym, ds)) or (UNRESOLVED if exceptions.get((sym, ds[:7])) == UNRESOLVED else None)
        cat.append(c or "MISSING_UNREPAIRED")
    extra = d.difference(cal)
    # gap intervals (maximal runs of consecutive missing expected sessions)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    mp = pos.reindex(missing).to_numpy()
    intervals = []
    if len(mp):
        start = prev = mp[0]
        for p in mp[1:]:
            if p != prev + 1:
                intervals.append((start, prev))
                start = p
            prev = p
        intervals.append((start, prev))
    iv = [{"start": str(cal[a])[:10], "end": str(cal[b])[:10], "sessions": int(b - a + 1)} for a, b in intervals]
    # full calendar months with zero rows inside the cached span
    months_exp = pd.Series(1, index=exp).groupby([exp.year, exp.month]).size()
    months_have = pd.Series(1, index=present).groupby([present.year, present.month]).size() if len(present) else pd.Series(dtype=int)
    full_months = [f"{y}-{m:02d}" for (y, m), n in months_exp.items() if (y, m) not in months_have.index]
    # latest contiguous run ending at the last cached date (in expected sessions)
    last_missing = missing.max() if len(missing) else None
    run = int(((exp > last_missing) if last_missing is not None else np.ones(len(exp), bool)).sum())
    newest = pd.Timestamp(newest)
    tail_fresh = last == newest
    need_a = req["REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS"]
    recent = cal[cal <= newest][-need_a:]
    recent_missing = int(len(recent.difference(d))) if tail_fresh else None
    level_a = bool(tail_fresh and recent_missing == 0 and len(recent) == need_a and first <= recent[0])
    # training samples that would be built under the OLD finite-only rule but cross a gap
    gb = gap_before(d, cal)
    L, S, fwd = req["feature_lookback_rows"], req["seq_len"], req["label_forward_sessions"]
    n = len(d)
    old_ok = np.zeros(n, bool)
    old_ok[L + S - 1:] = True                           # enough rows for a finite window (gap-blind)
    feat_ok = contiguous_back(gb, L + S)
    lab_ok = contiguous_forward(gb, fwd)
    feat_cross = int((old_ok & ~feat_ok).sum())
    lab_cross = int((old_ok & feat_ok & ~lab_ok & (np.arange(n) < n - fwd)).sum())
    return {
        "symbol": sym, "first_date": str(first)[:10], "last_date": str(last)[:10],
        "expected_sessions": int(len(exp)), "actual_sessions": int(len(present)),
        "missing_sessions": int(len(missing)), "duplicate_dates": dup, "non_monotonic_steps": nonmono,
        "off_calendar_rows": int(len(extra)), "gap_intervals": int(len(iv)),
        "multi_session_gaps": int(sum(1 for x in iv if x["sessions"] >= 2)),
        "largest_gap_sessions": int(max((x["sessions"] for x in iv), default=0)),
        "largest_gap_start": max(iv, key=lambda x: x["sessions"])["start"] if iv else "",
        "full_month_gaps": len(full_months), "full_month_list": ";".join(full_months),
        "missing_2025_26": int((missing >= "2025-01-01").sum()),
        "latest_contiguous_run": run, "tail_fresh": bool(tail_fresh),
        "recent_window_missing": recent_missing,
        "LEVEL_A_live_inference": "PASS" if level_a else "FAIL",
        "LEVEL_B_training_samples_feature_window_cross": feat_cross,
        "LEVEL_B_training_samples_label_window_cross": lab_cross,
        "LEVEL_C_history": "COMPLETE" if len(missing) == 0 else "INCOMPLETE",
        "missing_unrepaired": int(cat.count("MISSING_UNREPAIRED")),
        "missing_market_open_symbol_no_data": int(cat.count(MARKET_OPEN_SYMBOL_NO_DATA)),
        "missing_confirmed_symbol_no_trade": int(cat.count(CONFIRMED_SYMBOL_NO_TRADE)),
        "missing_unresolved_after_retries": int(cat.count(UNRESOLVED)),
        "listing_boundary": UNKNOWN_LISTING_BOUNDARY,
        "_intervals": iv,
    }


def model_eligible(symbols):
    from data import SECTOR_MAP
    return [s for s in symbols if SECTOR_MAP.get(s) != "etf"]


# ------------------------------------------------------------ backfill plan

def backfill_plan(audits, calendar, newest, req, confirmed=None):
    """Downloadable repair requests. Registry-confirmed no-trade sessions
    (`confirmed`, {(symbol, date)}) are NOT downloadable defects: a request
    month whose missing sessions are all confirmed is omitted (the sessions
    stay missing and keep breaking contiguity)."""
    confirmed = confirmed or set()
    cal = pd.DatetimeIndex(calendar)
    cal = cal[cal <= pd.Timestamp(newest)]
    a_start = cal[-req["REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS"]]
    p1_start = cal[-RECENT_TRAINING_SESSIONS]
    span = req["REQUIRED_TRAINING_CONTIGUOUS_SESSIONS"]
    rows = []
    for a in audits:
        for g in a["_intervals"]:
            s, e = pd.Timestamp(g["start"]), pd.Timestamp(g["end"])
            months = pd.period_range(s, e, freq="M")
            prio = "P0" if e >= a_start else ("P1" if e >= p1_start else "P2")
            for m in months:
                ms = max(s, m.start_time)
                me = min(e, m.end_time.normalize())
                miss = cal[(cal >= ms) & (cal <= me)]
                n_exp = int(len(miss))
                if n_exp and all((a["symbol"], str(d)[:10]) in confirmed for d in miss):
                    continue
                rows.append({"symbol": a["symbol"], "missing_start": g["start"], "missing_end": g["end"],
                             "interval_sessions": g["sessions"], "fetch_month": str(m),
                             "expected_sessions_in_request": n_exp, "priority": prio,
                             "recent_input_impact": bool(e >= a_start),
                             "training_impact_samples": int(min(span + g["sessions"] - 1,
                                                                a["actual_sessions"]))})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["priority", "symbol", "fetch_month"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------ manifest

def cache_hash(cache_dir, symbols):
    h = hashlib.sha256()
    for s in sorted(symbols):
        p = os.path.join(cache_dir, f"{s}.csv")
        if os.path.isfile(p):
            h.update(s.encode())
            with open(p, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def file_hashes(cache_dir):
    out = {}
    for f in sorted(os.listdir(cache_dir)):
        if f.endswith(".csv"):
            with open(os.path.join(cache_dir, f), "rb") as fh:
                b = fh.read()
            out[f] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    return out


def snapshot_cache(cache_dir, dest):
    """Byte-exact copy of every cache CSV into `dest` (must not exist), verified
    file by file. Returns the snapshot manifest. Reads the cache only."""
    import shutil
    if os.path.exists(dest):
        raise FileExistsError(f"{dest} exists; snapshots are immutable")
    before = file_hashes(cache_dir)
    os.makedirs(dest)
    for f in before:
        shutil.copy2(os.path.join(cache_dir, f), os.path.join(dest, f))
    after_src, copy = file_hashes(cache_dir), file_hashes(dest)
    if not (before == after_src == copy):
        raise RuntimeError("snapshot verification failed (cache changed during copy or copy differs)")
    syms = [f[:-4] for f in before]
    return {"snapshot_dir": os.path.abspath(dest), "source_dir": os.path.abspath(cache_dir),
            "file_count": len(before), "total_bytes": int(sum(v["bytes"] for v in before.values())),
            "aggregate_sha256": cache_hash(cache_dir, syms), "per_file": before, "verified": True}


def verify_against_snapshot(snapshot_dir, cache_dir, calendar=None):
    """Prove that every pre-repair line survives byte-identical and in order,
    and classify every added line. Returns (per-symbol rows, totals)."""
    cal = set(pd.DatetimeIndex(calendar).strftime("%Y-%m-%d")) if calendar is not None else None
    rows = []
    for f in sorted(x for x in os.listdir(snapshot_dir) if x.endswith(".csv")):
        old = open(os.path.join(snapshot_dir, f), "rb").read()
        new_p = os.path.join(cache_dir, f)
        if not os.path.isfile(new_p):
            rows.append({"symbol": f[:-4], "file_missing": True})
            continue
        new = open(new_p, "rb").read()
        eol = b"\r\n" if b"\r\n" in old else b"\n"
        ol = [x for x in old.split(eol) if x]
        nl = [x for x in new.split(eol) if x]
        j, kept, added = 0, 0, []
        for line in nl:                                  # ordered subsequence match
            if j < len(ol) and line == ol[j]:
                j += 1
                kept += 1
            else:
                added.append(line)
        modified = len(ol) - kept                        # old lines not found in order
        adates = [a.split(b",", 1)[0].decode() for a in added if a != ol[0]]
        old_dates = {x.split(b",", 1)[0].decode() for x in ol[1:]}
        unexplained = [d for d in adates if d in old_dates or (cal is not None and d not in cal)]
        rows.append({"symbol": f[:-4], "old_lines": len(ol), "new_lines": len(nl),
                     "existing_rows_modified_or_removed": int(modified), "rows_added": len(adates),
                     "unexplained_added_dates": ";".join(unexplained),
                     "byte_identical": old == new})
    df = pd.DataFrame(rows)
    tot = {"files": int(len(df)),
           "files_changed": int((~df["byte_identical"]).sum()) if len(df) else 0,
           "existing_rows_modified_or_removed": int(df["existing_rows_modified_or_removed"].sum()) if len(df) else 0,
           "rows_added": int(df["rows_added"].sum()) if len(df) else 0,
           "unexplained_added_dates": int((df["unexplained_added_dates"] != "").sum()) if len(df) else 0}
    return df, tot


def run_audit(cache_dir, out_dir, state_path=None, registry_path=None):
    from data import SECTOR_MAP
    configured = sorted(SECTOR_MAP)
    raw = read_cache_dates(cache_dir)
    cached = sorted(raw)
    eligible = [s for s in model_eligible(configured) if s in raw]
    calendar = derive_calendar({s: raw[s] for s in eligible})
    newest = calendar.max()
    req = required_windows()
    exc = missing_exceptions(state_path, registry_path)
    audits = [audit_symbol(s, raw[s], calendar, newest, req, exc) for s in cached]
    inv = pd.DataFrame([{k: v for k, v in a.items() if not k.startswith("_")} for a in audits])
    inv["model_eligible"] = inv["symbol"].isin(eligible)
    plan = backfill_plan([a for a in audits if a["symbol"] in eligible], calendar, newest, req,
                         confirmed=load_no_trade_registry(registry_path))
    os.makedirs(out_dir, exist_ok=True)
    inv.to_csv(os.path.join(out_dir, "eod_cache_gap_inventory.csv"), index=False)
    plan.to_csv(os.path.join(out_dir, "eod_backfill_plan.csv"), index=False)
    el = inv[inv["model_eligible"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cache_dir": os.path.abspath(cache_dir),
        "calendar_method": CALENDAR_METHOD,
        "calendar_rule": f"share of alive symbols with a row >= {CALENDAR_MIN_SHARE} AND >= {CALENDAR_MIN_SYMBOLS} symbols",
        "calendar_sessions": int(len(calendar)), "calendar_first": str(calendar.min())[:10],
        "calendar_last": str(newest)[:10],
        "configured_symbols": len(configured), "cached_symbols": len(cached),
        "configured_without_cache": sorted(set(configured) - set(cached)),
        "model_eligible_symbols": len(eligible),
        "required_windows": req,
        "eligible_with_any_gap": int((el["missing_sessions"] > 0).sum()),
        "eligible_with_2025_26_gap": int((el["missing_2025_26"] > 0).sum()),
        "eligible_missing_symbol_sessions": int(el["missing_sessions"].sum()),
        "eligible_full_month_gaps": int(el["full_month_gaps"].sum()),
        "eligible_largest_gap": el.sort_values("largest_gap_sessions").iloc[-1][["symbol", "largest_gap_sessions", "largest_gap_start"]].to_dict(),
        "eligible_with_duplicates": int((el["duplicate_dates"] > 0).sum()),
        "eligible_with_non_monotonic": int((el["non_monotonic_steps"] > 0).sum()),
        "eligible_off_calendar_rows": int(el["off_calendar_rows"].sum()),
        "eligible_level_a_fail": sorted(el.loc[el["LEVEL_A_live_inference"] == "FAIL", "symbol"].tolist()),
        "eligible_level_a_fail_gap": sorted(el.loc[(el["LEVEL_A_live_inference"] == "FAIL") & el["tail_fresh"], "symbol"].tolist()),
        "eligible_level_a_fail_stale_tail": sorted(el.loc[~el["tail_fresh"], "symbol"].tolist()),
        "training_samples_feature_window_cross": int(el["LEVEL_B_training_samples_feature_window_cross"].sum()),
        "training_samples_label_window_cross": int(el["LEVEL_B_training_samples_label_window_cross"].sum()),
        "backfill_requests": int(len(plan)),
        "backfill_requests_by_priority": plan["priority"].value_counts().to_dict() if len(plan) else {},
        "backfill_completion_state": "NOT_STARTED",
        "unresolved_symbols": sorted(el.loc[el["missing_sessions"] > 0, "symbol"].tolist()),
        "missing_by_status": {k: int(el[k].sum()) for k in ("missing_unrepaired",
                              "missing_market_open_symbol_no_data", "missing_confirmed_symbol_no_trade",
                              "missing_unresolved_after_retries")},
        "cache_sha256": cache_hash(cache_dir, cached),
    }
    with open(os.path.join(out_dir, "eod_data_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary, inv, plan, calendar


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit"])
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "research", "data_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "data_integrity"))
    ap.add_argument("--repair-state", default=None)
    ap.add_argument("--registry", default=None)
    a = ap.parse_args()
    s, inv, plan, cal = run_audit(a.cache_dir, a.out, a.repair_state, a.registry)
    print(json.dumps({k: v for k, v in s.items() if k not in ("unresolved_symbols",)}, indent=1, default=str))
