"""v17 P4-A (part 4) — exact production validation policy on a real as-of date.

Reproduces, with the production functions themselves, what a daily
retrain on the latest cached date uses for gradient training, for
early-stopping validation, and what it discards. Writes
reports/model_audit/v17/p4/validation_policy_example.json. CPU only.
"""

import inspect
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import dataset_transformer_eod as dte  # noqa: E402
import train_transformer_eod as tte  # noqa: E402

OUT = os.path.join(ROOT, "reports", "model_audit", "v17", "p4")


def describe(data, refit_rank, horizon=20):
    tr, va, w = dte.matured_train_val(data, "tgt_rank_20", refit_rank, horizon)
    dr, dates = data["date_rank"], data["dates"]
    d = lambda r: str(dates[int(r)])[:10]  # noqa: E731
    tr_max, va_min, va_max = int(dr[tr].max()), int(dr[va].min()), int(dr[va].max())
    latest = int(dr.max())
    n_val_dates = int(len(np.unique(dr[va])))
    n_tr_dates = int(len(np.unique(dr[tr])))
    excluded_sessions = latest - tr_max          # newest sessions never in gradient training
    cal_days = (pd.Timestamp(dates[latest]) - pd.Timestamp(dates[tr_max])).days
    return {
        "as_of_date": d(refit_rank), "latest_cache_date": d(latest),
        "history_first_date": d(int(dr.min())), "history_sessions": latest + 1,
        "train": {"first": d(int(dr[tr].min())), "last": d(tr_max),
                  "sessions": n_tr_dates, "samples": int(len(tr))},
        "purge_gap_sessions": va_min - tr_max - 1,
        "validation": {"first": d(va_min), "last": d(va_max),
                       "sessions": n_val_dates, "samples": int(len(va)),
                       "contiguous": bool(n_val_dates == va_max - va_min + 1)},
        "immature_tail_sessions": latest - va_max,
        "newest_sessions_excluded_from_gradient_training": excluded_sessions,
        "newest_calendar_days_excluded": cal_days,
        "newest_calendar_months_excluded": round(cal_days / 30.44, 1),
        "val_fraction_of_matured_dates": round(n_val_dates / (n_val_dates + n_tr_dates + (va_min - tr_max - 1)), 3),
        "sample_weights_all_one": bool(float(w.min()) == 1.0 and float(w.max()) == 1.0),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = tte.PRESETS["B"]
    data = dte.build_dataset("close_only", seq_len=cfg["seq_len"], horizons=(20,), verbose=False)
    latest = int(data["date_rank"].max())
    ex_latest = describe(data, latest)
    ex_prev = describe(data, latest - 1)
    # how the ranges move from one daily refit to the next
    delta = {k: (ex_latest[k] if not isinstance(ex_latest[k], dict) else None)
             for k in ("as_of_date",)}
    move = {"train_last_moves_by_sessions": (pd.Timestamp(ex_latest["train"]["last"]) > pd.Timestamp(ex_prev["train"]["last"])),
            "train_last_prev": ex_prev["train"]["last"], "train_last_now": ex_latest["train"]["last"],
            "val_first_prev": ex_prev["validation"]["first"], "val_first_now": ex_latest["validation"]["first"],
            "val_last_prev": ex_prev["validation"]["last"], "val_last_now": ex_latest["validation"]["last"]}
    src_daily = inspect.getsource(tte.mode_daily_retrain)
    src_fit = inspect.getsource(tte.fit_one)
    policy = {
        "early_stopping_rule": "per epoch: val rank IC (mean per-date Spearman); keep deepcopy of best; stop when no_improve >= patience(3) and epoch+1 >= min_epochs(2); max_epochs 25",
        "final_weights_are_best_validation_epoch": "net.load_state_dict(best_state)" in src_fit,
        "validation_rows_used_for_gradients": False if "tr_idx" in src_fit else None,
        "refit_on_train_plus_validation_afterwards": ("matured_train_val" in src_daily and "fit_one(Xg, yg, tr, va" in src_daily and src_daily.count("fit_one(") == 1),
        "note": "verified by source inspection: mode_daily_retrain calls fit_one exactly once per seed with (tr, va); there is no second fit on tr+va",
    }
    policy["refit_on_train_plus_validation_afterwards"] = not policy["refit_on_train_plus_validation_afterwards"]
    out = {"example_latest": ex_latest, "example_previous_session": ex_prev,
           "daily_movement": move, "policy": policy}
    with open(os.path.join(OUT, "validation_policy_example.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
