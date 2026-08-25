# Purged generated artifacts — 2026-08-25 repository cleanup

All purged items were **gitignored, generated, and reproducible**; no
production reference exists to any of them (verified by the Stage A
dependency audit: no import, no bat, no scheduled task, no test, no
runtime read). Committed research evidence for every purged line remains
under `reports/continuous_research/`. Purge date: 2026-08-25.

| Path | Size | Line / purpose | Why inactive | No-production-reference evidence | Reproducible from |
|---|---|---|---|---|---|
| `checkpoints/v12_big_transformer/` (5 files) | 3,305 MB | v12 big-model feasibility checkpoints (XL1/XL2 up to 1.25 GB each) | v12 CLOSED 2026-08-01: all big-model axes rejected | only referenced by `run_queue_v12_big_transformer.py` (legacy runner, unreachable from daily_ops/tasks/tests) | committed runner `research/run_queue_v12_big_transformer.py` + plan/results in `reports/continuous_research/v12_big_transformer/` |
| `checkpoints/v13_fixed_xl_feature_rich/` (2 files) | 2,381 MB | v13 fixed-XL feature-rich checkpoints | v13 CLOSED 2026-08-03: fixed-XL + feature expansion rejected | only referenced by `run_queue_v13_fixed_xl_feature_rich.py` (legacy) | committed runner + `reports/continuous_research/v13_fixed_xl_feature_rich/` |
| `checkpoints/v11_sweep/` (1 file) | 0.4 MB | v11 sweep smoke checkpoint | v11 CLOSED 2026-07-31 | none | `research/run_queue_v11.py` |
| `checkpoints/transformer_eod/smoke.pt` | 0.4 MB | one-off smoke artifact | superseded by daily_seed0..6.pt (which are KEPT) | not in `daily_seed*.pt` glob read by inference (`inference_transformer_eod.py:46`) | any smoke run |
| `reports/transformer_gpu/v12_big_transformer/` | 33 MB | v12 generated outputs (gitignored family) | line closed | gitignored (`.gitignore:34`), no reader | committed v12 runner |
| `reports/transformer_gpu/v13_fixed_xl_feature_rich/` | 26 MB | v13 generated outputs | line closed | gitignored (`.gitignore:42`), no reader | committed v13 runner |
| `reports/intraday_playbook/day_frame.csv` | 31 MB | v14 daily-bar proxy frame | v14 concluded (framework retained; frame is derived data) | gitignored; only reader is legacy `search_playbook_rules.py:22` | `research/intraday_playbook/backtest_conditional_playbook.py:107-108` (`build_day_frame()` from the EOD cache) |
| repo `__pycache__/` dirs (6, excl. `.venv`) | ~1.7 MB | Python bytecode | always regenerable | gitignored | automatic on next import |

**Explicitly NOT purged** (Stage A/approval carve-outs):
`reports/transformer_gpu/panels/` (frozen calibration panels — loaded by
production `user_next_session_plan.py:188`), production
`checkpoints/transformer_eod/daily_seed*.pt` + manifest, the intraday
SQLite + supervisor logs, all paper-trading state/history, holdings
files, current plans, and all committed research evidence.
