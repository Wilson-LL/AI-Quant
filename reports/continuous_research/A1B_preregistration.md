# Pre-registration — Cycle 7 (A1B rank-10 bear-window validation)

Registered: 2026-07-22 22:15 (before running)

Context: A1 standalone rank-10 was REJECTED (L/S 1.51 vs champion 1.91, 2023–26).
But its 50/50 D1.2 blend at h10 + band10 scored L/S 1.93 / LO 1.97 (net100 1.91,
turn 0.16) on the champion window — LO beats the promoted blend50+band10 (1.83).
Single-window; possibly blend-multiplicity luck. This cycle validates out-of-window.

Config: identical to A1 but OOS 2021-01→2026-07 (11 refits, early refits train on
2018–20 only — same handicap as BEAR_presetB_2021, so compare only within-panel).
Panel: `LOOP_A1B_rank10_bear2021`.

Gates (pre-declared):
- Compute A1B-blend (z-avg 50/50 with mom126/5) at h10+band10, L/S and LO, plus
  standalone. Compare vs blend50+band10 on BEAR_presetB_2021 (L/S 1.42 / LO 1.48,
  DD −26.4%/−30.8%) and 2022 yearly.
- PROMOTE (as alternative production book) only if A1B-blend beats blend50+band10
  on the bear window in BOTH L/S and LO net60 (margin ≥ 0.05) without materially
  worse DD (≤ +5pp absolute).
- If it only matches: classify "monitor" — the h10 blend stays a research variant.
- If it loses: reject and close the rank-10 line entirely.

Runtime estimate: ~25–40 min GPU (11 refits × 5 seeds).
