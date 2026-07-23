# Pre-registration — Cycle 20 (E3 close + market-regime features)

Registered: 2026-07-23 15:30 (before running; queue v4 #1)

Motivation: the champion's one weakness is 2022 (blend ≈ flat; deep tf −0.05).
"close + market regime" is on the allowed Track-E list and was never run in G5.
Hypothesis: per-date market-state inputs (equal-weight index drawdown, market
vol, market momentum, breadth — all causal, broadcast to every name) let the
model condition its momentum exposure on regime, improving 2022 without
sacrificing level. This is feature-level regime conditioning — distinct from the
rejected blend-weight switching (C1) and exposure gates (C3), which acted at
the portfolio level on a regime-blind model.

Config: feature_set=close_regime (close_only 10 + 4 market columns), preset B,
5 seeds, rank-20, deep cache, OOS 2023-01→ and 2021-01→.
Panels LOOP_E3_regime_{2023,2021}.

Gates:
- Val-IC selection discipline as always (vs champion 0.050 on the 2023 grid).
- Standalone vs 1.91/1.91 and 1.14; blend50+band10 vs 2.06/−10.7% and
  1.47/−18.7%; special attention: 2022 yearly (blend ref −0.15) and bear DD.
- Promote only on dual-window improvement (≥ +0.05 or 2022 ≥ +0.3 with
  full-window within 0.05); partial → monitor; else reject (feature-level
  regime conditioning closed; A4 head-level variant NOT pursued after a
  feature-level failure).

Runtime: ~2h GPU (new dataset build + 5 seeds × 18 refits).
