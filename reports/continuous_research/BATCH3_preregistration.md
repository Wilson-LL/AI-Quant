# Pre-registration — Cycle 17 (BATCH3 architecture-diversity blend, preset C)

Registered: 2026-07-23 08:40 (before running)

Motivation: preset C (h128) was the best drawdown profile in G9 (−10.5% at 1.58,
3 seeds). Hypothesis: adding it as a third signal — blend = ¼ z(B) + ¼ z(C) +
½ z(mom), band10 — improves DD/2022 vs the standing references without losing
full-window Sharpe. Architecture diversity ≠ the rejected horizon/window
diversity: both members use the same target/window, differing only in capacity.

Configs: preset C, 3 seeds (its G9 spec), rank-20, deep cache, OOS 2023-01→ and
2021-01→.

Pre-declared post-pass variants (all reported): C-standalone books; ¼B+¼C+½mom
band10; ½·mean(zB,zC)+½·mom band10 (equivalent weighting stated for clarity —
same thing); B-C 50/50 tf-only blend.

Gates vs standing refs (B4d: 2.06/−10.7% champion window; B4c: 1.47/−18.7% bear):
- Promote only if the 3-way blend beats BOTH references' net60 (≥ +0.05) or
  improves DD ≥ 2pp on both with Sharpe within 0.05.
- Otherwise reject and close the architecture-diversity line (single-tier C
  stays a documented DD-friendly standalone alternative).

Runtime: ~1.2h GPU (3 seeds × 18 refits + build).
