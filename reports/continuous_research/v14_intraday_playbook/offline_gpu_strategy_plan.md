# v14 Offline GPU Strategy Plan (Task 7) — ARCHITECTURE ONLY, no training

Per Task 7E and the data report: intraday data is missing, so NO model is
trained in this sprint. This document records how heavy compute would be
used once the collector has accumulated real data.

## Principle

Heavy models run BEFORE the session; the live layer is a table lookup.
Live path: read price/time → match playbook row → emit label. No GPU, no
model, milliseconds.

## Components (future, gated on collector data + separate approvals)

A. **Scenario scoring (precompute).** Nightly job enumerates the scenario
   grid (symbol × gap-bin × checkpoint × coarse price/volume state) and
   scores each cell offline; output = the playbook table. Grid size is
   ~110 × 10 × 7 × O(10²) states ≈ 10⁶ rows — trivially precomputable
   offline even with a large model, impossible live.
B. **Monte Carlo path simulation.** Learn empirical transition
   distributions (gap → first-30-min → …) from collected bars; simulate
   paths per scenario cell to estimate first-passage probabilities
   (TP-before-stop) that daily bars can never give.
C. **Conditional policy approximation.** Train an offline model
   (small first; size must re-earn its place — v11–v13 lesson) mapping
   scenario states → action values; DISTILL its outputs into the
   playbook table. The table, not the model, goes live.
D. **Distillation caveat from v12:** teacher→student distillation failed
   for the EOD rank task (tail instability). Table-export is a different,
   easier kind of distillation (finite input grid — exact enumeration,
   no generalization gap), which is why the playbook design is table-first.
E. **XL usage rule:** no 1 GB-class model unless true intraday data
   volume justifies it (v13 showed 312M params collapse on ~10⁵ samples;
   minute bars × 110 symbols × months ≈ 10⁷ rows is the first regime
   where that calculus could change — measured then, not assumed).

## Hard gates before any of this trains

Collector running + ≥3 months QA-clean data + pre-registered plan +
user approval per phase. Until then this file is the entire GPU story.
