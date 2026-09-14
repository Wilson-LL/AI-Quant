# 09 — Portfolio construction & concentration (B12, B13)

## top_frac × band sweep (challengers only; `construction.csv`, net60 Sharpe, long-only)

| top_frac \ band | 0.00 | 0.10 | 0.15 | 0.20 |  | 0.00 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|---|---|---|---|
| | **CH** | | | | | **BR** | | | |
| 0.10 | 2.018 | 1.947 | 1.863 | 1.739 | | 1.314 | 1.303 | 1.349 | 1.456 |
| 0.15 | 1.650 | 1.782 | 1.825 | 1.875 | | 1.404 | 1.322 | 1.335 | 1.349 |
| **0.20** | 1.901 | **1.989** | 1.968 | 1.974 | | 1.357 | **1.455** | 1.432 | 1.422 |
| 0.25 | 1.890 | 1.863 | 1.794 | 1.679 | | 1.396 | 1.387 | 1.412 | 1.378 |
| 0.30 | 1.796 | 1.782 | 1.830 | 1.663 | | 1.258 | 1.319 | 1.308 | 1.233 |

Max drawdowns (net60): top_frac 0.10 books −33…−36% on both windows;
current (0.20, 0.10) **−23.8% CH / −26.0% BR — the best or near-best DD
cell in each grid**. net100 preserves every ordering.

**Verdict: the production construction sits on the plateau and is the
grid maximum on BR; no cell dominates it on both windows and the largest
CH "gains" (top10/band0 +0.03) come with +10 pp drawdown. B12: KEEP.**
Note this sweep is in-sample on burned windows and 40 cells were tried —
even a 0.1 edge would be uninterpretable (05).

## Sector concentration (B13)

- Top-quintile book composition (mean share): electronics 35% + semis
  25% = **60% (CH)**; 51% (BR). Max single-sector share mean 37%, p95 57%.
- Attribution (per-date Spearman on the 20d return):
  raw IC 0.067 (CH) / 0.032 (BR); **sector-neutral (within-sector) IC
  0.086 / 0.043 — higher, IR 0.41 vs 0.34**; between-sector (sector-mean
  score vs sector-mean return) IC 0.124 / 0.072, positive 67% / 60%.
- But a sector-neutralized *signal* run through the same book builder
  loses money relative to the champion (Stage-1 CPU test, long-only
  net60: CH 1.795 vs 1.989; BR 1.291 vs 1.455; L/S collapses to 0.9 / 0.5)
  — replicating the closed C4 line.

**Reading:** the model is not merely learning sector momentum — within-
sector selection is its cleanest skill — but the *returns* of the long
book depend materially on the sector tilt (semis/electronics
over-weight during a semis bull). That tilt is a risk concentration: a
semis-led drawdown would hit the book harder than its IC suggests. The
pre-validated free lever remains the 30% hard sector cap (v9-R4: costs
≤ 0.001 Sharpe) — a construction switch for the user to decide, not
research.
