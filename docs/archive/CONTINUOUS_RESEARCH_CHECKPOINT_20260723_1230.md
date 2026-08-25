# Continuous Research Checkpoint — 2026-07-23 12:30

Covers 2026-07-23 01:45 → 12:30 (loop elapsed ~16h) · commits 6f0a486 → 9b9a1a7.

## 1. Experiments since last checkpoint

| ID | hypothesis | verdict |
|---|---|---|
| BF | 2015→2017 cache backfill | done — 106 stocks, +77k rows, 0 failures |
| BD | deep training improves bear window | PASS — new refs; blend 1.47/−18.7% |
| B3 | vol-adjusted target | REJECT (1.44/1.72) |
| B4 | avoid-bottom target (book + veto) | REJECT — val/OOS inversion (+.146/−.069) |
| A3 | regularization one-knob checks | champion confirmed at plateau |
| F3 | live daily cadence on 2026-07-22 data | PASS — full cycle ~11 min |
| C4 | sector-neutral scoring | REJECT (sector tilt is load-bearing) |
| WD1 | train-window-diversity ensemble | REJECT — 3y member collapses on deep cache |
| REF | deep-cache champion references | SET — blend50+band10 **2.06 / −10.7%** (B4d) |
| B5 | preset-C architecture-diversity blend | REJECT (ties champ window, loses bear) |
| ES | excess-vs-sector target @ 5 seeds | REJECT — promising-list emptied |
| RL1 | pairwise ranking loss (Edit 2) | RUNNING (dual-window, gates pre-registered) |

## 2. Leaderboard

Production candidate unchanged and twice strengthened by deeper training data:
**blend50+band10 — L/S 2.06 net60 / DD −10.7% (2023–26 deep, B4d); 1.47 / −18.7%
(2021–26 deep, B4c); 2022 ≈ flat; no losing year; bootstrap-robust; turnover 0.29.**
Now 12 challengers rejected on pre-registered dual-window gates.

## 3. Rejected this window

voladj / avoid-bottom / excess-sector targets, sector-neutral scoring,
window-diversity and architecture-diversity ensembles, close_d12 (prev. window),
confidence filtering (prev. window). Every closed line documented with numbers.

## 4. Promoted

None new — the standing candidate kept its title; its references improved with
the deep cache (1.95→2.06 champion window, 1.42→1.47 bear window).

## 5. Code changes

train_transformer_eod.py Edit 2 (optional pairwise loss; MSE path verified
byte-identical), dataset_transformer_eod.py (+2 target columns),
refresh_data.py (--backfill-start). All documented in PRODUCTION_EDIT_PLAN.md
with tests. model.py/train.py/inference.py/dataset.py untouched.

## 6. Runtime

GPU: BATCH2 (195 fits), WD1 (55), REF23 (35), BATCH3 (~110), BATCH4 (90) — all
completed on the RTX 4060 Ti, AMP, ≤ 6.3 GB VRAM. Deep cache: dataset build
~35 min/process (batched configs amortize), daily 5-seed retrain 267 s.
Daily production cycle ~11 min ≈ 65× inside the 12h budget.

## 7. Next hypotheses (queue v3)

1. RL1 pairwise loss (running) — last allowed Track-A lever.
2. F3 daily operational cadence (time-gated).
3. E2 true full-fields (calendar-gated ~2027-01).
4. Monitors: EX3 gate, 7.5% cap, preset C standalone.

## 8. Best-candidate change?

No change in identity; references upgraded (B4d/B4c supersede shallow rows).
