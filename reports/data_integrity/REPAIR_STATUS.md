# EOD cache repair — status (2026-09-21, after P0/P1/P2, tail refresh and DATA_BASELINE_V2)

**PRODUCTION_DATA_INTEGRITY: DEGRADED.** One non-held symbol, 2207, is excluded. The valid model universe is 107/108 (99.07%, threshold 99%). No held or current-position symbol is invalid, and every exclusion is explicit.

No official daily_ops, model inference, merge or GPU research has been run. The repo root has not been switched.

## Decisions applied (user, 2026-09-21)

- **2207 / 2025-12-18 = CONFIRMED_NO_TRADE** (`symbol_no_trade_registry.csv`, evidence: TWSE MI_INDEX 2025-12-18 shows 0 shares, 0 trades, prices `--`).
  - It is never re-downloaded, filled or interpolated, and still breaks contiguity.
  - It is no longer a downloadable defect: it is removed from the backfill plan.
- **Symbol-level live policy** (`eod_integrity.classify_live_integrity`, integrity gate, inference, book):
  - HARD BLOCK: any invalid symbol that is a my_holdings.csv position or held in the standing decision book → UNSAFE_FOR_NEW_MODEL_OUTPUT; the previous plan stays in force.
  - DEGRADE: invalid non-held symbols get no score and are listed as DATA_INTEGRITY_FAILURE. They are removed from the cross-section, and z, rank, top fraction, band10 and weights are recomputed over valid names (N → N-1).
  - Publication requires valid / otherwise-eligible >= 0.99. This rule is separate from the latest-date coverage gate.

## Repair results

| stage | requests | FETCH_OK | symbol no-data | confirmed no-trade | suspect / unresolved | rows added |
|---|---|---|---|---|---|---|
| P0 | 9 | 8 | 0 | 1 (2207, upgraded, no refetch) | 0 | 149 |
| P1 | 22 | 17 | 5 | 0 | 0 | 339 |
| P2 | 134 unique (135 plan rows) | 95 | 39 | 0 | 0 | 1,934 |
| tail refresh 2026-09-14 → 2026-09-21 | 110 symbols | 110 | – | – | 0 failed or empty months | 660 |

- **P0 final:** COMPLETE_WITH_CONFIRMED_NO_TRADE_EXCEPTION. 7 formerly invalid symbols are repaired, 2207 is CONFIRMED_NO_TRADE, the current raw-window invalid count is 1, and 0 downloadable P0 requests remain.
- **Symbol no-data:** 44 symbol-months, 114 sessions, state MARKET_OPEN_SYMBOL_NO_DATA / SUSPENSION_STATUS_UNKNOWN. In each case the source served the month but not that day.
  - The 5 P1 sessions were checked read-only against TWSE MI_INDEX. All show 0 shares, 0 trades, prices `--` (`post_p1/no_data_evidence.csv`).
  - None are registered; that classification is the user's call. All 114 are historical and outside every current 191-session window.

## Diff vs the immutable V1 snapshot (`post_tail/snapshot_verification.csv`)

- Files: 110, of which 110 changed (every file got the tail).
- **Existing V1 rows modified or removed: 0.**
- Rows added: 3,082.
- Unexplained / unexpected dates: 0.

## Whole cache (108 model-eligible)

| item | V1 (Phase A) | post-P0 | post-P1 | post-P2 | V2 (post tail) |
|---|---|---|---|---|---|
| symbols with gaps | 68 | 66 | 62 | 30 | 30 |
| symbols with 2025–26 gaps | 22 | 16 | 5 | 5 | 5 |
| missing symbol-sessions | 2,537 | 2,388 | 2,049 | 115 | 115 |
| full-month holes | 120 | 112 | 95 | 0 | 0 |
| largest gap (sessions) | 230 (3037) | 230 | 230 | 9 (2337) | 9 |
| of which downloadable / unrepaired | 2,537 | 2,387 | 2,043 | 0 | 0 |
| training samples crossing a gap (feature window) | 23,951 | 23,193 | 20,281 | 7,595 | 7,601 |
| training samples crossing a gap (label window) | 2,483 | 2,357 | 2,062 | 798 | 798 |
| current-window invalid symbols | 8 | 1 (2207) | 1 | 1 | 1 (2207) |

2207's window contains 2025-12-18 until that date leaves the 191-session window, about 9 sessions after 2026-09-21. It then becomes valid and the status becomes SAFE if nothing else changes.

## DATA_BASELINE_V2 (`data_baseline_v2_manifest.json`)

| field | value |
|---|---|
| latest market date | 2026-09-21 |
| calendar | CROSS_SECTIONAL_MAJORITY_v1: 2,856 sessions, 2015-01-05 → 2026-09-21 |
| files / eligible | 110 / 108 (2809 and 2888 configured but never cached) |
| total rows | 309,281 |
| aggregate SHA-256 | `c37dc97035953502be074ce0df0e7c1f17af9c9259d691fdb0fbf9a005340a8a` |
| manifest SHA-256 | `27d5f23b8fe5c02fbf59126781469019bd52e3277567aadb97bbe76508f665be` |
| immutable copy | `C:\Users\wilso\source\code\AI-Quant-backups\eod_cache_data_baseline_v2_20260921\data_cache` (outside the repo) |
| rows added since V1 | 3,082 |
| V1 rows modified | 0 |
| verified downloadable gaps remaining | 0 |
| current invalid inference symbols | 2207 |

## Gates on V2 (release-branch code, read-only)

- REFRESH: PASS (108/108 at 2026-09-21).
- TAIL_COVERAGE: PASS.
- RECENT_WINDOW_INTEGRITY: DEGRADED (2207 — DATA_INTEGRITY_FAILURE / CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW). The gate passes with 107/108.
- HISTORICAL_TRAINING_INTEGRITY: WARN (30 symbols, 115 sessions, 8,399 guard-excluded samples).

## Integration (`release/eod-integrity-ops`, worktree `..\AI-Quant-release`, from main d2966f0; not merged)

- Commits: holdings completeness, plus 510e23c, 1f5b998, 80e148b, 618d9ed and 331fae0, plus the records commit.
- Validation: 307 tests OK, compileall OK, diff-check OK.
  - **A (frozen V1 snapshot):** the 2026-09-11 book csv and md are byte-identical (test_17).
  - **B (V2, 2207 excluded, frozen 09-11 predictions, no inference):** gate DEGRADED 107/108; 2207 is absent from book and ranking and is listed in the md as `2207 — DATA_INTEGRITY_FAILURE / CONFIRMED_NO_TRADE_IN_REQUIRED_WINDOW`.
    - The book goes from 22 to 21 held names: 2308 leaves.
    - Attribution: V2 data with 2207 kept gives the same 22 names, so the 2308 exit is the N → N-1 recompute (2308 sat at the band edge, rank 25).
  - **C (2883 gap injected into a temp copy):** gate BLOCKED (UNSAFE_FOR_NEW_MODEL_OUTPUT, held 2883). No runtime file changed.
