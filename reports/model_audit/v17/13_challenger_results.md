# 13 — Challenger results

**No GPU challenger has been run.** The only GPU work so far is the P0-A′
methodology screen (cadence parity, not a challenger) — see 16: classified
**PARITY_WARNING** (5-session refit: IC −0.0155 vs 126-session on the same
131 dates, ~1.6 HAC SE; book-level metrics identical; stability high).

## Stage-1 CPU results (frozen panels, `stage1_holding_sector.csv`)

| Variant | CH L/O net60 | BR L/O net60 | vs champion (1.989 / 1.455) | Verdict |
|---|---|---|---|---|
| Current signal, 40-session rebalance | 1.626 | 1.212 | −0.36 / −0.24 | not a free win; motivates a 40d-*target* model, not longer holds of the 20d model |
| Current signal, 60-session rebalance | 1.288 | 1.139 | −0.70 / −0.32 | reject |
| Sector-neutralized signal, 20d | 1.795 | 1.291 | −0.19 / −0.16 | **REJECT** (replicates closed C4) |
| top_frac × band grid (40 cells) | best CH 2.018 (top10/band0, DD −33%) | best BR = champion | — | **KEEP production construction** |

All Stage-1 numbers are in-sample on burned windows and would need the
05 statistical bar to mean anything; they are used only to *order*
Stage-2 work, which is why H6 is closed and H1 is reframed as a target
change.

Champion re-verification in this pass: BOOK_EQUIVALENCE BYTE_IDENTICAL
(2026-09-11 book regenerated from frozen inputs).
