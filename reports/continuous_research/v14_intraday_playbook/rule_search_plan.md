# v14 Rule Search Plan (Task 6) — pre-registered before running

Restricted to what daily bars support: **gap-bin × EOD-state → open-to-
close** rules. No checkpoint rules, no stop/TP rules (path-ambiguous).

## Day-frame construction (leakage-controlled)

Signal: previous session's blend z (frozen A8 BR panel, 2021→2026 —
scores at t−1 are OOS walkforward outputs, so no lookahead). Pseudo-action
from the t−1 cross-section: TOP_Q (top 20% — BUY/HOLD-equivalent),
WATCH_BAND (20–30%), MID, BOTTOM_Q (bottom 20% — short-diagnostic).
Outcome day t: gap (known at open) and oc_ret (open→close). Decision info
set at t's open = {t−1 score, gap} — both available before any action.

## Search space (deliberately tiny — ~80 cells, not thousands)

gap_bin (10) × pseudo_action (4) × direction (long for TOP_Q/WATCH_BAND,
short-diagnostic for BOTTOM_Q; MID excluded from selection). No other
degrees of freedom (no thresholds to tune — bins are pre-registered).

## Splits and selection rules

- Train 2021-01→2023-12 · Validation 2024-01→2024-12 · OOS 2025-01→
  2026-07 (touched once, after selection is frozen).
- Cost: 30 bps round trip per day-trade (long); 2× for short-diagnostic.
- A cell is SELECTED only if, in TRAIN: n ≥ 100, |mean net oc| > 0 with
  t-stat ≥ 2, win rate consistent with sign; and in VALIDATION: same sign,
  net mean > 0, n ≥ 30. Max 5 selected rules (complexity cap, best-t
  first). Concentration guard: a rule is dropped if any single symbol
  contributes > 30% of its train PnL.
- OOS is reported for the frozen selection only, with the explicit
  expectation (12 dissociations of history) that survivorship of a cell
  through val does not guarantee OOS.

## Output

`selected_rules.json` (consumed by the playbook generator as
confidence=validated rows) + results section in backtest_results.md.
Everything labeled DAILY_BAR_PROXY_ONLY / not decision-grade.
