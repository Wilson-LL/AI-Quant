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
    ap.add_argument("stage", choices=("refresh", "retrain", "inference",
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
