# Research Lines Closed — Do Not Reopen Without New Evidence

Each line below was closed at a pre-registered gate with committed evidence
(scoreboard row in parentheses). "New evidence" means new data (e.g. the
~2027-01 full-field history), a new market regime in the paper ledger, or an
explicitly pre-registered re-design — not a re-run of the same idea.

## The v10 arc (2026-07-26/27)

- **14-seed ensemble (V8):** convergence confirmed but bear margin was
  parity within seed noise at 2× cost, worse bear DD; user decision KEEP 7.
- **MT multi-task heads (V10s/V10b):** the best-ever bear/2022 profile
  (1.515 / −13.0% / +0.23) collapsed on disjoint seeds (1.259 / −23.5% /
  −0.38) — seed-set luck, caught by the validation battery before adoption.
- **Distillation 7→1 (D2o):** student = 96% of ensemble val IC yet 0/6
  adoption gates (blend 1.034 vs 2.147; book overlap 0.52). Ensembling
  stabilizes the rank tails books are built from; a single student does not
  inherit that. Daily retrain (~5 min) was never a real pain point.
- **TCN trunk (V10b):** val-IC parity (0.04995) with collapsed books
  (LS 0.303) — the attention trunk's edge lives in book space.
- **Market-residualized target (V10b):** plain failure both sides (val
  0.032, OOS IC −0.002, LS −0.81); not an inversion, just no signal.
- **5d reversal (V10s):** standalone Sharpe NEGATIVE (−0.51/−1.05) — TWSE
  at this horizon shows continuation, not reversal; any blend weight hurts.
- **Input augmentation (V10s):** noise/date-dropout worsened the (already
  negligible, 0.00025) cross-set val-IC spread — the v7 seed variance lives
  in OOS book space, downstream of anything augmentation can touch.
- **Cross-sectional attention (V10s):** −23% val IC (0.037 vs 0.048);
  decisive at this trunk/universe scale. (The inverted variant A2 was
  dropped on the same evidence without spending GPU.)

## Overlays and adaptivity (v9 + earlier)

- **Regime overlays (B4, V9):** regime-conditional D1.2 downweight rejected
  at a deliberately high bar; bear-side improvement was one crash sample.
  Regime FEATURES already closed at E3 (2022 worsened). Transformer hedge
  sleeve (B3) rejected: +1.6pp DD vs the 3pp bar.
- **Exposure gates (EX1/EX2/EX3, V9-B2):** index-level gates rejected in
  v1–v4; the own-equity DD gate (EX3) was armed as a monitor and then
  formally DISARMED in v9 — the primary cell missed the DD bar on D7b.
  No crash overlay exists by decision; bear risk is handled by construction.
- **Adaptive blend weights:** closed in v1–v4, re-confirmed at book level
  by V9-B4. Reopening requires a fresh pre-registration AND new evidence —
  the parked quantile-standalone lead is the only candidate reason on file.

## Earlier closures still in force (v1–v7, see scoreboard)

Sequence axis (40<60>90>120); 9-seed saturation; ranking losses (pairwise +
listwise); risk-proxy targets (avoid-bottom, dd-adjusted — val/OOS inversion
×2); sector-excess target; rank-10 / voladj targets; close_d12 / close_range
/ close_liq / full-field features (until real full-field data ~2027-01);
sector-neutral scoring; confidence filtering (seed disagreement); recency
weighting; warm-start; rolling-window and architecture-diversity ensembles;
short holds (h5/h10); concentrated top-N books; variance-parity / inverse-vol
weighting.

## Standing meta-rules distilled from the closures

1. **No adoption on val-IC evidence alone** — four recorded dissociations
   (D1, MT, TCN, D2). Book-level dual-window validation decides.
2. **Single 7-seed draws are not decision-grade** — disjoint-seed
   replication is mandatory (v7 and MT precedents).
3. 2–3-seed screens select, never adopt; marginal val-IC edges (±0.005)
   are noise (E3).
4. Risk-proxy-flavored targets inflate val IC and invert OOS (B4/B6) —
   any new target needs an inversion tripwire.
