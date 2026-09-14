# v17 Production Audit — Executive Summary (revised 2026-09-14)

Branch `research/v17-production-audit-and-holdings` from main `d2966f0`.
Reproducible from `research/audit_v17_{signal,baselines,corporate_actions,statistics}.py`
(all CPU) + the CSV/JSON artifacts here; baseline freeze in `../v17_baseline.md`.

**Recommendation: CURRENT_PRODUCTION_OPERATIONALLY_VALID, BUT TRUE EDGE MAGNITUDE REMAINS UNCERTAIN** (14).

## Track A — done and verified
Every actual position appears first in both daily summaries with side /
cost / price / unrealized P&L (abs and %) / rank / status / action /
priority / reason, behind a `HoldingsCoverageError` gate. Real-holdings
acceptance 11 / 11 / 11. Cost basis is context only (no action rule reads
it — tested). Decision book unchanged: BOOK_EQUIVALENCE BYTE_IDENTICAL.

## Track B — findings (evidence-backed, with their limits)
| Question | Answer | Where |
|---|---|---|
| Look-ahead leakage | No — proven mechanically (5 tests) | 02 |
| Corporate actions | **MATERIAL** (qualitative) seasonal bias inferred from DETECTED candidate prints (Jul 2× baseline; ≈0.6σ label shift; est. ≈5–8% of windows — estimate, not confirmed events); no local adjustment source; data requirement specified | 02 |
| Survivorship | **PRESENT**; data requirements specified; all references = SURVIVOR-UNIVERSE UPPER-BOUND | 03 |
| Validation / multiplicity | ~60–70 distinct trials on the same 42-block window; deflated-Sharpe: edge very likely real (>1.0 CH), magnitude uncertain: candid range CH 1.0–1.7 / BR 0.7–1.3 | 05 |
| Target horizon | IC 0.067→0.087 (20d→40d); book holds for months; 40d target never trained → P3 | 04 |
| Prospective | 66 calendar days, 14 matured 20d rows ≈ **1 independent block**; 5d HAC CI includes 0; encouraging, not validation | 06 |
| Regimes | ≈0 IC in bear/narrow-breadth; 2022 negative; 2025 weakest recent year | 07 |
| Costs / timing | net60 ≈ fee-only, net100 realistic; retention 0.97; next-open validated | 08 |
| Construction | on the plateau / BR optimum; 60% semis+electronics tilt | 09 |
| Holdings vs model | current divergence large (58% in book names; 55% of target exposure omitted; 0.61 one-way deviation) — **ACTUAL_REALIZED_LOSS_ATTRIBUTION_INCOMPLETE** (no fills/entry dates) | 10 |
| Baselines | transformer ≈ 65% momentum; residual IC 0.02 / 0.00; beats ridge clearly; adds diversification more than IC | 11 |
| Production parity | daily full refit (same fixed seeds 0–6 retrained each session) never validated → **P0**, paired-seed design, ≈9 h GPU (up to 26 h) | 15 |

## Proposed research order (no GPU launched)
P0 parity (9–26 h) → P1 corporate-action data (0 GPU) → P2 universe data
spec (0) → P3 40d target (≈6 h) → P4 recency (≈7 h) → P6/P7 conditional.
Total ≈ 22 h baseline, ≤ 39 h. Architecture/seed/lookback/feature-XL
scaling: **not proposed**.

## Operational note
Intraday supervisor exited mid-session again (STATUS_CONTROL_C_EXIT,
25-min gap) — recorded for a separate reliability task; not fixed here.
