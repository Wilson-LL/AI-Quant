# Pre-registration — Cycle 16 (WD1 train-window-diversity ensemble)

Registered: 2026-07-23 06:35 (before running)

Motivation: deep-cache training raised the bear-window level (tf 1.00→1.14) but
gave up 2022 crash-adaptivity (+0.58→−0.05). Hypothesis: an ensemble of the
deep-trained model and a 3y-rolling-window model (the best-performing bounded
window from G2, val-IC 0.049) recovers part of the crash behavior without
sacrificing the deep level. This is NOT recency weighting (rejected): both
members are equal-weight within their windows; diversity is in window length.

Config: LOOP_WD_roll3y_2021 — rank-20, preset B, 5 seeds, recency={window:756},
OOS 2021-01→ on the deep cache. CPU post-pass: z-avg with BEARDEEP scores →
tf-ensemble; blend50+band10 with D1.2.

Gates (vs standing refs B3c/B4c): ensemble blend beats 1.47 full-window AND
2022 ≥ −0.05 → adopt as candidate; matches → monitor; loses → reject
(window-diversity line closed).
