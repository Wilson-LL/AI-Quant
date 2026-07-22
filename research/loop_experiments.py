"""Continuous-loop GPU experiment driver (LOOP_* configs).

One config per invocation, background-friendly, crash-safe (JSON rewritten per
config; success judged by artifacts, not exit code — AMP teardown quirk).

Usage:
  python research/loop_experiments.py A1     # rank-10 target, preset B, 5 seeds
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from transformer_experiments import get_data, run_config, REPORT_DIR  # noqa: E402
from train_transformer_eod import require_cuda  # noqa: E402

OUT = os.path.join(ROOT, "reports", "continuous_research")
os.makedirs(OUT, exist_ok=True)
FULL_SEEDS = [0, 1, 2, 3, 4]


def a1():
    """tgt_rank_10 at full champion strength (preset B, 5 seeds, equal-all)."""
    require_cuda()
    out = os.path.join(OUT, "A1_results.json")
    results = {}
    data, Xg = get_data("close_only", 60)
    run_config(data, Xg, "LOOP_A1_rank10_presetB", results, out, holding=10,
               target="tgt_rank_10", horizon=10, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None)
    print("\nA1 done")


def a1b():
    """rank-10 preset B on the bear window (OOS 2021-01→) to validate the
    h10 blend finding out-of-window. Gates in A1B_preregistration.md."""
    require_cuda()
    out = os.path.join(OUT, "A1B_results.json")
    results = {}
    data, Xg = get_data("close_only", 60)
    run_config(data, Xg, "LOOP_A1B_rank10_bear2021", results, out, holding=10,
               target="tgt_rank_10", horizon=10, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None,
               oos_start="2021-01-01")
    print("\nA1B done")


def a2():
    """Champion-config rerun emitting score_std (reproducibility + confidence
    data for A2 filtering). Gates in A2_preregistration.md."""
    require_cuda()
    out = os.path.join(OUT, "A2_results.json")
    results = {}
    data, Xg = get_data("close_only", 60)
    run_config(data, Xg, "LOOP_A2_champion_std", results, out, holding=20,
               target="tgt_rank_20", horizon=20, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None)
    print("\nA2 done")


def e1():
    """close+D1.2 feature set at full champion strength. Gates in
    E1_preregistration.md (val-IC selection discipline)."""
    require_cuda()
    out = os.path.join(OUT, "E1_results.json")
    results = {}
    data, Xg = get_data("close_d12", 60)
    run_config(data, Xg, "LOOP_E1_close_d12_presetB", results, out, holding=20,
               target="tgt_rank_20", horizon=20, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None)
    print("\nE1 done")


def batch1():
    """A2 then E1 in one process (amortize startup)."""
    a2()
    e1()


def batch2():
    """Post-backfill batch (deep 2015+ cache): BEAR-DEEP champion rerun,
    B3 vol-adjusted target, B4 avoid-bottom target, A3 regularization checks.
    One close_only/seq60 dataset build serves all five configs.
    Pre-registrations: BATCH2_preregistration.md."""
    require_cuda()
    from train_transformer_eod import PRESETS
    PRESETS["B_do3"] = dict(PRESETS["B"], dropout=0.3)
    out = os.path.join(OUT, "BATCH2_results.json")
    results = {}
    data, Xg = get_data("close_only", 60)
    run_config(data, Xg, "LOOP_BEARDEEP_rank20_2021", results, out, holding=20,
               target="tgt_rank_20", horizon=20, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None,
               oos_start="2021-01-01")
    run_config(data, Xg, "LOOP_B3_voladj20", results, out, holding=20,
               target="tgt_voladj_20", horizon=20, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None)
    run_config(data, Xg, "LOOP_B4_avoidbot20", results, out, holding=20,
               target="tgt_avoid_bot_20", horizon=20, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None)
    run_config(data, Xg, "LOOP_A3_dropout30", results, out, holding=20,
               target="tgt_rank_20", horizon=20, preset="B_do3",
               refit_every=126, seeds=FULL_SEEDS, recency=None)
    run_config(data, Xg, "LOOP_A3_wd5e4", results, out, holding=20,
               target="tgt_rank_20", horizon=20, preset="B",
               refit_every=126, seeds=FULL_SEEDS, recency=None,
               weight_decay=5e-4)
    print("\nBATCH2 done")


if __name__ == "__main__":
    {"A1": a1, "A1B": a1b, "A2": a2, "E1": e1,
     "BATCH1": batch1, "BATCH2": batch2}[sys.argv[1] if len(sys.argv) > 1 else "A1"]()
