# EOD cache gap integrity — audit, fix design and backfill plan (H-DATA-INTEGRITY, P0)

**Branch:** `fix/eod-cache-gap-integrity`, based on `main` at `d2966f0`. It lives in a separate worktree; the production working tree is untouched. **Status:** uncommitted, for review.
**Audited cache:** the live `research/data_cache`, read-only. It ends 2026-09-11 (cache SHA-256 `52d8320c…`).
**Machine-readable:**
- `eod_cache_gap_inventory.csv` (per symbol)
- `eod_backfill_plan.csv` (per symbol-month request)
- `eod_data_manifest.json` (data version)

**Classification: PRODUCTION_DATA_INTEGRITY = UNSAFE_FOR_NEW_MODEL_OUTPUT.** Production as it runs today would score 8 symbols on inputs spliced across missing sessions. Two of them, 2883 and 3443, are held in the latest decision book (2026-09-11). It also trains on data where 8.6% of the sample windows cross a hole. With this branch deployed and only the 9 P0 requests repaired, the status becomes DEGRADED: current windows are clean, and historical holes are excluded by the guard rather than filled. It becomes SAFE after the full backfill and a new data baseline.

## 1. Root cause (confirmed mechanically)

**Tail refresh (`research/refresh_data.py`).** For each symbol, `refresh_stock` computes the months to fetch from the newest cached date. The old code then did this:

1. A month that raised an error or came back empty was skipped with `continue`.
2. A later month that succeeded was still appended, which advanced `last` past the failed month.
3. The retry pass recomputed the months to fetch from that new `last`.
4. So the failed month was never requested again. The symbol stopped being "suspect" and was reported as refreshed.

The hole became permanent. It is triggered whenever a run has to catch up two or more months, for example after missed `daily_ops` days. `backfill_stock` had the same skip-and-continue pattern.

**Gate.** `pipeline_gate.py refresh` checks only cross-sectional coverage of the NEWEST date (the 99% rule), so it could never see holes inside a symbol's history.

**Consumers.** `dataset_transformer_eod.build_dataset` and `inference_transformer_eod.py` compute features, 60-step sequences and forward-return labels on each symbol's OWN rows, so a hole is spliced straight across. `log_ret_1` then carries a multi-week return as if it were one session, and every rolling lookback shifts. There is no look-ahead: label maturity is tracked on the global calendar.

## 2. Inventory (model-eligible universe)

| item | value |
|---|---|
| Configured symbols (`SECTOR_MAP`) | 112 (2809 and 2888 have never had a cache file) |
| Cached symbols | 110 (includes ETFs 0050 and 0056) |
| **Model-eligible symbols** (non-ETF with cache) | **108** |
| Expected-session calendar | 2,850 sessions, 2015-01-05 → 2026-09-11 |
| **Symbols with any verified gap** | **68** |
| **Symbols with 2025–26 gaps** | **22** |
| **Missing expected symbol-sessions** | **2,537** |
| **Largest gap** | **3037: 230 consecutive sessions from 2017-02-02** (next: 2382, 212 from 2017-03-01) |
| **Full calendar-month holes** | **120** (2017: 25, 2018: 8, 2019: 14, 2020: 12, 2021: 7, 2022: 12, 2023: 9, 2024: 12, 2025: 14, 2026: 7) |
| Symbols with duplicate dates / non-monotonic dates / off-calendar rows | 0 / 0 / 0 |
| **Symbols whose current inference window crosses a gap** | **8: 2002, 2207, 2312, 2376, 2610, 2883, 3443, 5880** |
| Symbols with a stale tail at the newest date | 0 |
| **Training samples whose feature window crosses a gap** | **23,951 of 280,027 (8.6%)** |
| Further samples whose label window crosses a gap | 2,483 |

The 22 symbols with 2025–26 gaps: 1101, 2002, 2207, 2312, 2317, 2337, 2344, 2376, 2382, 2385, 2408, 2409, 2610, 2883, 2884, 2886, 3034, 3037, 3443, 5880, 9910, 9933.

**Calendar method and its limitation.** No authoritative TWSE holiday calendar exists locally, so the calendar is DERIVED (`CROSS_SECTIONAL_MAJORITY_v1`). A date is an expected session if at least 50% of the symbols whose cached span covers it have a row, and at least 10 symbols do.
- Weekends, holidays and exchange closures have no coverage and are never sessions.
- There is deliberately no weekday rule. The first audit pass used one and wrongly excluded 8 real Saturday make-up sessions (2016-01-30, 2016-06-04, 2016-09-10, 2017-02-18, 2017-06-03, 2017-09-30, 2018-03-31, 2018-12-22), each traded by 102–106 of 108 names.
- The method would miss a hole shared by more than half the universe on the same dates. No month in the audit has 15 or more sessions missing for more than 3 symbols.

**Listing life.** A symbol is expected to trade only between its first and last cached dates. Before the first cached date it is marked `UNKNOWN_LISTING_BOUNDARY`, and no expectation is manufactured there. A missing session is also not proven to be a fetch failure: it could be a trading suspension. The repair distinguishes the two. If the source returns the month but not those sessions, they become `CONFIRMED_NO_DATA`.

## 3. Three integrity levels

| level | requirement | exact window | today |
|---|---|---|---|
| **A: live inference safety** | Every session of the symbol's required inference window is cached. The tail is current at the newest date, with no duplicates or non-monotonic dates. | The last **191** expected sessions ending at the as-of date | **FAIL for 8 symbols** |
| **B: current training safety** | No training sample's feature lookback, 60-step sequence or label window crosses a missing expected session. | Rows t−190 … t contiguous for features; rows t … t+21 contiguous for the label, so **212 contiguous sessions** per sample | Enforced by the new dataset guard, which excludes 23,951 feature-window and 2,483 label-window crossings |
| **C: full historical completeness** | Every expected session between the symbol's first and last cached date is present. | The full cached span; the listing boundary before the first date is UNKNOWN | **68 symbols incomplete** (2,537 symbol-sessions) |

## 4. REQUIRED_INFERENCE_CONTIGUOUS_SESSIONS = 191 (derived from code)

- `eod_integrity.feature_lookback()` runs the production feature code (`dataset_transformer_eod._stock_features`, close block) on a gap-free synthetic series. It finds the first row where all ten features are finite: **row 131**.
- The binding feature is `mom_126_5 = close.shift(5) / close.shift(131) − 1`. The others need at most 60 prior rows: `vol_60` and `dist_*_60` use rolling 60, `mom_60` uses shift 60, and `log_ret_1` needs 1.
- A 60-step sequence (production `seq_len`) therefore needs **131 + 60 = 191** contiguous sessions ending at the as-of date.
- A training sample also needs exec lag 1 + horizon 20 = 21 forward sessions, so **212** in total.
- Purge (21 sessions) is a split rule, not a per-sample window.
- The regression test `test_required_window_derived_from_feature_code` pins 131 / 191 / 212.

## 5. Fix design (implemented on the branch; not deployed)

| component | change |
|---|---|
| `research/refresh_data.py`, tail | **Contiguity-preserving.** Months are fetched oldest-first and the loop **stops at the first failed or empty month**, so no later month is ever appended past a hole. The existing suspect/retry pass then resumes from the true last date. |
| `research/refresh_data.py`, prepend backfill | Same protection: months are fetched newest-first and the loop stops at the first failure. |
| `research/refresh_data.py`, `repair_gaps` (new) | **HISTORICAL_MISSING_INTERVALS** are handled separately from **NEW_TAIL_DATA**. For each queued symbol-month the repair: (1) recomputes the expected-but-missing sessions against the calendar, (2) fetches that month, (3) validates the rows (dates inside the month and exactly the missing sessions, positive OHLC, high ≥ low), (4) merges with **every existing line byte-identical** and new lines inserted at their date position, via an atomic write, (5) records a state (`FETCH_OK` / `CONFIRMED_NO_DATA` / `EMPTY_RESPONSE_SUSPECT` / `NETWORK_ERROR` / `RATE_LIMIT_SUSPECT` / `UNRESOLVED`) in a **resumable** JSON state file. An empty payload is never read as "no trading". Limits: 3 attempts per symbol-month, a cooldown after 3 consecutive suspect responses, a stop after 6 (resume later), and a per-run cap on requests. No forward-fill, no interpolation, no invented rows. CLI: `--repair-gaps --repair-priorities P0 [P1 P2] --max-requests N`. |
| `research/eod_integrity.py` (new) | Single source of truth for the calendar, the auditor, the levels, the derived windows, the gap flags, the backfill planner and the manifest. It only reads the cache. |
| `dataset_transformer_eod.build_dataset` | **Gap guard** (default on). Feature rows whose 131-row lookback crosses a hole are masked, so no window ending within 190 rows after a hole is built. `fwd_*`, `_vol20_t`, `_ddwin_20` and the barrier target are set to NaN when their window crosses a hole, *before* cross-sectional ranking. Rejected samples are recorded as `WINDOW_CROSSES_DATA_GAP` in `out["integrity"]`. **On a gap-free cache the guard changes nothing**: verified byte-identical on the live cache with the guard off, and on gap-free stocks with it on. |
| `inference_transformer_eod.py` | A symbol whose current window crosses a hole gets **no normal score**. It is listed as **`DATA_INTEGRITY_FAILURE`** (reason `WINDOW_CROSSES_DATA_GAP`, or `STALE_TAIL`) in `<asof>_data_integrity.csv`, `metrics.json["data_integrity"]` and the report. It is never silently dropped. The predictions CSV schema is unchanged. |
| `research/pipeline_gate.py integrity` (new stage) | Reports **TAIL_COVERAGE**, **RECENT_WINDOW_INTEGRITY** and **HISTORICAL_TRAINING_INTEGRITY** independently. It **fails** on low tail coverage or when any current symbol's 191-session window crosses a hole. Historical holes are reported as WARN (the guard excludes those samples). The `refresh` stage and `PARTIAL_COVERAGE_MIN = 0.99` are **unchanged**: cross-sectional coverage and longitudinal continuity stay separate gates. |
| `daily_ops.bat` | Runs `pipeline_gate.py integrity` right after the `refresh` coverage gate and before retrain, inference, book and plan. On failure it goes to the existing `:pipefail`: **no new book or plan, and the standing plan is left untouched**. |

**What the gates say on the live cache today:**
- the old `refresh` stage reports "PASS: coverage 108/108 at 2026-09-11";
- the new `integrity` stage reports TAIL_COVERAGE PASS (100%), RECENT_WINDOW_INTEGRITY FAIL (the 8 symbols) and HISTORICAL_TRAINING_INTEGRITY WARN (68 symbols, 2,537 missing symbol-sessions, 26,434 samples excluded), and the gate **FAILS**.

## 6. Backfill plan (deterministic; NOT executed)

`eod_backfill_plan.csv` has one row per symbol and calendar month to fetch: the missing interval, the expected sessions in that request, the priority, the recent-input impact, and an upper-bound training impact.

| priority | rule | requests | symbols |
|---|---|---|---|
| **P0** | gap intersects the current 191-session inference window | **9** | 2002 (2026-03), 2207 (2025-12), 2312 (2026-02), 2376 (2026-01), 2610 (2026-02), 2883 (2026-06), 3443 (2026-04), 5880 (2025-11 and 2026-06) |
| **P1** | gap intersects the most recent 504 sessions (~2 years) of training data | **22** | 1101, 2317, 2337, 2344, 2351, 2382, 2385, 2408, 2409, 2610, 2884, 2886, 2887, 3034, 3037, 3711, 9910, 9933 |
| **P2** | older history | **135** | |
| **total** | | **166 remote requests** (one TWSE month request each) | |

**Runtime estimate** (throttle 1.5 s plus about 0.5–1.0 s of source latency per request):

| scope | estimate |
|---|---|
| P0 | about 20–25 s |
| Everything | about 6–7 min |
| Worst case | 3 attempts each (498 requests, about 21 min) plus 120 s cooldowns after every 3 consecutive suspect responses, and a stop after 6 that resumes on the next run. Under sustained rate limiting, completion takes several resumable runs, roughly 30–60 min of wall-clock. |

## 7. Smoke backfill (one symbol, one month, isolated copy)

**2317, 2025-09**, a documented full-month hole. It ran on a COPY of `2317.csv` in a scratch directory with one TWSE request. The live cache file's SHA-256 was unchanged afterwards.

| check | result |
|---|---|
| Fetch state | `FETCH_OK` |
| Rows added | **21 = all 21 expected sessions of 2025-09** |
| Existing lines | byte-identical and in order |
| Price plausibility | 203.5 (2025-08-29) → 198.5 … 216.0 (September) → 219.0 (2025-10-01) |
| Second repair run | no request (resumed state), file unchanged |
| Direct re-merge of the same rows | 0 added, file unchanged |
| Gap auditor | 2317 missing 48 → **27**; full-month holes `2019-06;2025-09` → `2019-06` |
| Splice flags Aug–Nov 2025 | 1 → **0** |
| Remaining | a separate 1-session hole on 2025-07-30 still invalidates windows ending 2025-12 → 2026-04; the current 2317 window passes Level A |

## 8. Data versioning and BOOK_EQUIVALENCE semantics

- **Data manifest** (`eod_data_manifest.json`, schema `eod_integrity/1`):
  - audit timestamp and calendar method;
  - gap counts, unresolved symbols and the cache SHA-256;
  - backfill state `NOT_STARTED` (the smoke ran on an isolated copy only).
- **After the backfill the corrected cache becomes a NEW DATA BASELINE.** Results on it are not directly comparable to results on the old cache.
- **CODE_NONREGRESSION:** against the same frozen pre-repair inputs, the decision book is **BYTE_IDENTICAL**. The `test_17` regeneration uses this branch's code on the runtime inputs, and the dataset builder with the guard off is byte-identical to production on the live cache.
- **DATA_REPAIR_EXPECTED_DIVERGENCE:** after the backfill, or with the guard on over the holed cache, predictions and books **will** change, because the input data changed. That is expected and must not be tested for byte identity against old bad-data outputs.

## 9. Research invalidation map (not rerun)

| prior v17 finding | exposure | classification |
|---|---|---|
| P0 cadence parity (P0-A′, P0-A) | Paired arms share the same contaminated data (8.6% of training windows cross holes). Relative conclusions are likely robust; absolute IC levels shift. | **LIKELY_MINOR** |
| P4 validation / early stopping (P4-A, P4-B0) | Paired, same data. The validation-IC levels of gap names are affected. | **LIKELY_MINOR** |
| P4-B0-LONG (long epochs) | Paired epoch contrasts on the same data. The memorisation mechanism is unlikely to change; OOS values of gap names shift. | **LIKELY_MINOR** |
| H-RESIDUAL-SIGNAL | In 2026-H1, spliced-window rows carried 42–60% of the residual. | **MATERIAL_RERUN_REQUIRED** |
| H-FACTOR-PREMIUM | Factor ICs were computed on the calendar close matrix, so holes are NaN and names drop out, not spliced. More names after repair. | **LIKELY_MINOR** |
| Long-training factor-de-exposure attribution | Based on H-RESIDUAL-style decompositions of LONG predictions | **LIKELY_MINOR** |
| 40-day horizon observations | 41-session labels cross holes about twice as often as 21-session labels, and have never been studied carefully. | **UNKNOWN** |
| Baseline comparisons (report 11: momentum and ridge vs transformer) | The transformer side was trained and scored on spliced inputs; the baselines used calendar closes. | **LIKELY_MINOR** (level shift) |
| Champion references (CH 2.147 / BR 1.443) and all historical backtests | Every model was trained with 8.6% contaminated windows; references must be re-established on the new baseline. | **MATERIAL_RERUN_REQUIRED** |

## 10. Scope boundaries respected

- **Not done in this round:** no mass backfill, no production data mutation (the smoke used an isolated copy), no retrain, no regeneration of official plans, no change to checkpoints, architecture or scheduled tasks, no merge, no commit.
- **Corporate actions:** unadjusted prices remain a **separate** problem (H-CORPACT), untouched here. Return-adjustment semantics are unchanged.
- **Research freeze:** all model and GPU research stays frozen until the cache is repaired and re-baselined.
