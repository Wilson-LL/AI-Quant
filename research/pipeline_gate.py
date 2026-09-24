"""Daily-ops pipeline gate (2026-08-24 incident fix).

daily_ops.bat calls this after each critical step. A stage passes only
when the EXPECTED dated artifact for the newest cached EOD date exists —
so a failed step can never be papered over by yesterday's artifact
(stale-artifact guard), and torch's known nonzero teardown exit codes
cannot cause false aborts (the gate keys on artifacts, not ERRORLEVEL,
for the GPU steps).

Stages:
  refresh   — newest-cache-date coverage over the non-ETF universe must
              be >= the pre-registered publication threshold (same 0.99
              policy as user_next_session_plan.PARTIAL_COVERAGE_MIN);
              blocks a partial-publication day BEFORE the GPU retrain.
  integrity — LONGITUDINAL per-symbol continuity (H-DATA-INTEGRITY),
              reported separately from the cross-sectional coverage:
              TAIL_COVERAGE, RECENT_WINDOW_INTEGRITY (blocking: every
              current symbol's required inference window must be free
              of missing sessions) and HISTORICAL_TRAINING_INTEGRITY
              (reported; the dataset gap guard excludes those samples).
  retrain   — checkpoints/transformer_eod/daily_manifest.json asof ==
              newest cache date.
  inference — reports/transformer_gpu/<newest>_{predictions.csv,
              target_book.csv,metrics.json} all exist.
  book      — reports/paper_trading/<newest>_blend50_band10_decision_
              book.csv exists.

Exit-code contract (review 2026-08-24): SUCCESS requires BOTH
(A) acceptable process termination and (B) expected-current-asof
artifact validation with mtimes AFTER the step started (--since-marker).
The ONLY tolerated nonzero exit is the known torch/CUDA teardown
behavior on this rig (documented in daily_ops.bat and project memory; no
specific numeric code was ever recorded, so the whitelist is conditioned
on evidence, not a code number): the step wrote EVERY expected dated
artifact fresh after step start and only then exited nonzero. Any other
nonzero exit — including one where current-dated artifacts exist from an
EARLIER run (stale mtimes) or where any artifact is missing — aborts.
A zero exit with stale/missing artifacts also aborts (silent no-op).

Exit 0 = pass; exit 1 = FAIL (daily_ops aborts, standing user plan is
left untouched). Read-only; never generates or deletes artifacts.
"""

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from data import SECTOR_MAP  # noqa: E402

# single source of truth for the publication threshold
from user_next_session_plan import PARTIAL_COVERAGE_MIN  # noqa: E402


def universe():
    return sorted(s for s in SECTOR_MAP if SECTOR_MAP[s] != "etf")


def newest_and_coverage(root):
    last = {}
    for s in universe():
        p = os.path.join(root, "research", "data_cache", f"{s}.csv")
        if not os.path.isfile(p):
            continue
        try:
            d = pd.read_csv(p, usecols=["date"])["date"]
            if len(d):
                last[s] = str(d.iloc[-1])[:10]
        except Exception:
            continue
    if not last:
        return None, 0, 0.0
    newest = max(last.values())
    n_at = sum(1 for v in last.values() if v == newest)
    # Denominator = the CACHE-BACKED universe (what the model can score),
    # not raw SECTOR_MAP: two universe names (2809/2888) have never had
    # cache files, so a SECTOR_MAP denominator could never reach 99%.
    return newest, n_at, n_at / max(len(last), 1)


def longitudinal_integrity(root):
    """Per-symbol LONGITUDINAL continuity (H-DATA-INTEGRITY). Deliberately
    separate from the cross-sectional newest-date coverage above: "108/108 at
    the latest date" says nothing about holes inside each symbol's history.
    Returns the three independently reported components:
      TAIL_COVERAGE                 newest-date coverage vs PARTIAL_COVERAGE_MIN
                                    (same semantics as the refresh stage)
      RECENT_WINDOW_INTEGRITY       every model-eligible symbol that is current
                                    at the newest date has all expected sessions
                                    of its REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS
                                    window (derived from the feature code), no
                                    duplicates and monotonic dates -> FAIL blocks
      HISTORICAL_TRAINING_INTEGRITY historical holes and the training samples the
                                    dataset gap guard excludes -> reported (WARN),
                                    not blocking, because no sample is built
                                    across a hole."""
    import eod_integrity as E
    cache_dir = os.path.join(root, "research", "data_cache")
    newest, n_at, ratio = newest_and_coverage(root)
    raw = E.read_cache_dates(cache_dir, universe())
    elig = [s for s in universe() if s in raw]
    cal = E.derive_calendar({s: raw[s] for s in elig})
    req = E.required_windows()
    rows = [E.audit_symbol(s, raw[s], cal, cal.max(), req) for s in elig]
    recent_bad = sorted(r["symbol"] for r in rows
                        if r["tail_fresh"] and (r["recent_window_missing"] or 0) > 0)
    structural = sorted(r["symbol"] for r in rows
                        if r["duplicate_dates"] or r["non_monotonic_steps"])
    hist = [r for r in rows if r["missing_sessions"] > 0]
    # symbol-level live policy: otherwise-eligible = model-eligible AND current
    # at the newest session (stale tails are the TAIL/refresh gates' business)
    current = [r["symbol"] for r in rows if r["tail_fresh"]]
    held, held_src = E.held_symbols(root)            # REAL_HELD only = hard-block source
    book, book_src = E.previous_book_symbols(root)    # model state; never blocks
    reasons = E.window_failure_reasons(cache_dir, recent_bad, cal, cal.max(), req)
    pol = E.classify_live_integrity(current, recent_bad, held)
    if structural:   # duplicate / non-monotonic dates = the integrity audit itself fails
        pol.update(status=E.UNSAFE, publication_allowed=False,
                   reason=f"structural date defects: {structural}")
    return {
        "TAIL_COVERAGE": {"ok": ratio >= PARTIAL_COVERAGE_MIN, "newest": newest,
                          "ratio": ratio, "threshold": PARTIAL_COVERAGE_MIN},
        "RECENT_WINDOW_INTEGRITY": {
            "ok": not recent_bad and not structural,
            "required_sessions": req["REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS"],
            "symbols_window_crosses_gap": recent_bad, "symbols_structural": structural,
            "reasons": {s: v["reason"] for s, v in reasons.items()}},
        "DATA_INTEGRITY_STATUS": pol, "held_sources": held_src,
        "_rows_tail_stale": [{"symbol": r["symbol"]} for r in rows if not r["tail_fresh"]],
        "HOLDINGS_VS_BOOK": {"REAL_HELD_SYMBOLS": sorted(held), "PREVIOUS_BOOK_SYMBOLS": sorted(book),
                             "HELD_AND_BOOK_INTERSECTION": sorted(held & book),
                             "BOOK_ONLY": sorted(book - held), "previous_book": book_src},
        "HISTORICAL_TRAINING_INTEGRITY": {
            "ok": True, "symbols_with_holes": len(hist),
            "missing_symbol_sessions": int(sum(r["missing_sessions"] for r in hist)),
            "training_samples_excluded_by_guard": int(sum(
                r["LEVEL_B_training_samples_feature_window_cross"]
                + r["LEVEL_B_training_samples_label_window_cross"] for r in rows))},
        "calendar_method": E.CALENDAR_METHOD, "calendar_last": str(cal.max())[:10],
    }


def _stage_artifacts(root, stage, newest):
    """Absolute paths of every artifact the stage must produce."""
    if stage == "retrain":
        return [os.path.join(root, "checkpoints", "transformer_eod",
                             "daily_manifest.json")]
    if stage == "inference":
        return [os.path.join(root, "reports", "transformer_gpu", n)
                for n in (f"{newest}_predictions.csv",
                          f"{newest}_target_book.csv",
                          f"{newest}_metrics.json",
                          f"{newest}_report.md")]
    if stage == "book":
        return [os.path.join(
            root, "reports", "paper_trading",
            f"{newest}_blend50_band10_decision_book.csv")]
    return []


def check(root, stage, exit_code=None, since_marker=None):
    newest, n_at, ratio = newest_and_coverage(root)
    if newest is None:
        return False, "no cached EOD data found"
    # ---- process-termination + artifact-freshness contract
    if stage in ("retrain", "inference", "book"):
        arts = _stage_artifacts(root, stage, newest)
        missing = [os.path.basename(p) for p in arts
                   if not os.path.isfile(p)]
        stale_mtime = []
        if since_marker is not None and os.path.isfile(since_marker):
            t0 = os.path.getmtime(since_marker)
            stale_mtime = [os.path.basename(p) for p in arts
                           if os.path.isfile(p)
                           and os.path.getmtime(p) < t0]
        if exit_code not in (None, 0):
            if missing or stale_mtime:
                return False, (
                    f"step exited rc={exit_code} and its expected "
                    f"{newest} artifacts are not freshly complete "
                    f"(missing={missing}, stale_mtime={stale_mtime}) — "
                    "unexpected failure, aborting (a dated artifact "
                    "from an earlier run cannot excuse a crash)")
            # every artifact freshly written after step start, then a
            # nonzero exit: the documented torch-teardown case
            print(f"[gate {stage}] WARNING: nonzero exit rc={exit_code} "
                  "tolerated — all expected artifacts were freshly "
                  "written after step start (known torch/CUDA teardown "
                  "behavior on this rig); artifact validation follows")
        else:
            if stale_mtime:
                return False, (
                    f"{stage} artifacts for {newest} predate step start "
                    f"({stale_mtime}) — the step produced no fresh "
                    "output (silent no-op / stale artifact)")
    n_cached = round(n_at / ratio) if ratio else 0
    if stage == "integrity":
        li = longitudinal_integrity(root)
        tc, rw, hi = (li["TAIL_COVERAGE"], li["RECENT_WINDOW_INTEGRITY"],
                      li["HISTORICAL_TRAINING_INTEGRITY"])
        print(f"[gate integrity] TAIL_COVERAGE: {'PASS' if tc['ok'] else 'FAIL'} "
              f"({tc['ratio']:.0%} at {tc['newest']}, threshold {tc['threshold']:.0%})")
        pol = li["DATA_INTEGRITY_STATUS"]
        print(f"[gate integrity] RECENT_WINDOW_INTEGRITY: {'PASS' if rw['ok'] else 'FAIL'} "
              f"({rw['required_sessions']}-session window; crosses gap: "
              f"{rw['symbols_window_crosses_gap'] or 'none'}; structural: "
              f"{rw['symbols_structural'] or 'none'})")
        print(f"[gate integrity] HISTORICAL_TRAINING_INTEGRITY: "
              f"{'WARN' if hi['symbols_with_holes'] else 'PASS'} "
              f"({hi['symbols_with_holes']} symbols with holes, "
              f"{hi['missing_symbol_sessions']} missing symbol-sessions, "
              f"{hi['training_samples_excluded_by_guard']} training samples excluded by the gap guard)")
        hb = li["HOLDINGS_VS_BOOK"]
        print(f"[gate integrity] REAL_HELD_SYMBOLS={len(hb['REAL_HELD_SYMBOLS'])} "
              f"PREVIOUS_BOOK_SYMBOLS={len(hb['PREVIOUS_BOOK_SYMBOLS'])} "
              f"HELD_AND_BOOK_INTERSECTION={len(hb['HELD_AND_BOOK_INTERSECTION'])} "
              f"BOOK_ONLY={len(hb['BOOK_ONLY'])} (hard-block source: REAL_HELD only; "
              f"previous book {hb['previous_book'] or 'none'})")
        for s in pol["excluded"]:
            tag = (" (REAL_HELD -> HARD BLOCK)" if s in pol["held_invalid"] else
                   " (BOOK_ONLY -> DEGRADED exclusion)" if s in hb["BOOK_ONLY"] else "")
            print(f"[gate integrity]   {s} - DATA_INTEGRITY_FAILURE / {rw['reasons'].get(s, '?')}{tag}")
        stale = sorted(r["symbol"] for r in li["_rows_tail_stale"]) if "_rows_tail_stale" in li else []
        if stale:
            print(f"[gate integrity] STALE_TAIL (no row at the newest session; counted in the "
                  f"portfolio sizing reference, not in the coverage ratio): {stale}")
        print(f"[gate integrity] DATA_INTEGRITY_STATUS: {pol['status']} - {pol['reason']} "
              f"(valid model universe {pol['valid']}/{pol['eligible']} = {pol['valid_ratio']:.2%}, "
              f"threshold {pol['threshold']:.0%}; hard-block set = REAL_HELD from "
              f"{li['held_sources'] or 'none'})")
        if not tc["ok"]:
            return False, "TAIL_COVERAGE below threshold"
        if not pol["publication_allowed"]:
            return False, (f"{pol['status']}: {pol['reason']} — do not retrain/infer/publish; "
                           "the previous plan stays in force "
                           "(repair: refresh_data.py --repair-gaps --repair-priorities P0)")
        if pol["status"] == "DEGRADED":
            return True, (f"DEGRADED: {len(pol['excluded'])} non-held symbol(s) excluded "
                          f"{pol['excluded']}; universe {pol['eligible']} -> {pol['valid']}")
        return True, "longitudinal integrity OK for current inference windows"
    if stage == "refresh":
        if ratio < PARTIAL_COVERAGE_MIN:
            return False, (f"newest EOD date {newest} covers only "
                           f"{n_at}/{n_cached} cached universe names "
                           f"({ratio:.0%} < {PARTIAL_COVERAGE_MIN:.0%}) — "
                           "partial publication suspected; do not "
                           "retrain/infer on this cross-section")
        return True, f"coverage {n_at}/{n_cached} at {newest}"
    if stage == "retrain":
        mp = os.path.join(root, "checkpoints", "transformer_eod",
                          "daily_manifest.json")
        if not os.path.isfile(mp):
            return False, "daily_manifest.json missing"
        try:
            asof = json.load(open(mp, encoding="utf-8")).get("asof")
        except Exception as e:
            return False, f"daily_manifest.json unreadable: {e}"
        if str(asof)[:10] != newest:
            return False, (f"manifest asof {asof} != newest cache date "
                           f"{newest} — retrain did not produce a fresh "
                           "model (stale artifact)")
        return True, f"manifest asof {asof}"
    if stage == "inference":
        missing = [os.path.basename(p) for p in
                   _stage_artifacts(root, stage, newest)
                   if not os.path.isfile(p)]
        if missing:
            return False, (f"inference artifacts for {newest} missing: "
                           f"{missing} — a stale dated artifact cannot "
                           "substitute for today's run")
        return True, f"inference artifacts present for {newest}"
    if stage == "book":
        p = _stage_artifacts(root, stage, newest)[0]
        if not os.path.isfile(p):
            return False, f"decision book for {newest} missing"
        return True, f"decision book present for {newest}"
    return False, f"unknown stage {stage!r}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("refresh", "integrity", "retrain", "inference",
                                      "book"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--exit-code", type=int, default=None,
                    help="the step's process exit code (ERRORLEVEL)")
    ap.add_argument("--since-marker", default=None,
                    help="marker file created at step start; artifacts "
                         "must be newer than it")
    a = ap.parse_args(argv)
    ok, msg = check(a.root, a.stage, exit_code=a.exit_code,
                    since_marker=a.since_marker)
    tag = "PASS" if ok else "FAIL"
    print(f"[gate {a.stage}] {tag}: {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
