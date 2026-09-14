# 14 — Recommendation (revised after review)

## CURRENT_PRODUCTION_OPERATIONALLY_VALID, BUT TRUE EDGE MAGNITUDE REMAINS UNCERTAIN

### Operationally valid
- No look-ahead leakage (mechanically proven); execution timing validated;
  costs immaterial; construction on the plateau; publication gates intact;
  holdings review now complete and gated.
- Prospective ranking since 2026-07 is *consistent with* the historical IC
  and shows no decay — but it is ≈ one independent 20-session block and
  must not be called validation (06).

### Edge magnitude uncertain — the reasons, in order
1. **Selection & sampling**: ~60–70 distinct comparisons on the same
   windows; n = 42 blocks; disjoint-seed replication −0.30; the champion
   point sits at its own bootstrap p95. Deflated-Sharpe check: very likely
   > 1.0 long-only on CH (P ≈ 0.9–0.98 under 30–70 trials), less certain
   on BR (0.7–0.94); a candid central range is **CH 1.0–1.7 / BR 0.7–1.3**
   rather than 1.99 / 1.46 (05).
2. **Survivor universe**: upper-bound references only (03).
3. **Corporate actions**: MATERIAL (qualitative) seasonal label/feature bias, inferred from detected candidate prints, not a confirmed event table (02/P1).
4. **Production ≠ validated cadence**: daily full refit of the same fixed seeds unvalidated; P0 uses a paired-seed design (15).
5. **The neural model is mostly momentum**: residual IC ≈ 0.02/0.00;
   momentum alone is within one SE; the model's clearest contribution is
   diversification/drawdown, not standalone IC (11).

### Recent-loss attribution (hardened wording)
**ACTUAL_REALIZED_LOSS_ATTRIBUTION_INCOMPLETE** — no fills or entry dates
are recorded. What is measurable: the current portfolio holds 58% of
capital in model names, 22% in an unselected name, 20% in an ETF outside
scope, omits 55% of the model's target exposure, and deviates 0.61
one-way from target with a 22% single-name weight. That is sufficient to
make realized results unrepresentative of the model; it does **not**
prove how much of past losses it explains. The model's own paper path
shows a normal-range drawdown block followed by a strong rebound.

### Recommended sequence (no compute until approved)
P0 parity → P1 corporate-action data → P2 requirements → P3 40d target →
P4 recency → (P6/P7 conditional). Architecture scaling: not proposed.

### Decisions available to the user now (no code)
1. Whether the real portfolio should track the model book.
2. The pre-validated 30% hard sector cap (free insurance).
3. Adding `entry_date` to `my_holdings.csv` to enable attribution.
4. A separate reliability task for the intraday supervisor (incident record
   in `reports/continuous_research/v15_intraday_collector/INCIDENT_2026-09-14_supervisor_exit.md`).
