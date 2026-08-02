# Queue v13 verdict — Fixed-XL / Feature-Rich Research

## VERDICT: KEEP_CURRENT_PRODUCTION
### Axis verdicts: REJECT_FEATURE_EXPANSION · REJECT_FIXED_XL · (short-side: moot — see below)

Closed by user 2026-08-03 after Phases 0–3 (feature inventory/specs, quality
battery, small-model screens, f2 confirmation, fixed-XL screens).

## Evidence

**Feature expansion (REJECT, decision-grade).**
- Small-model screens (3 seeds, CH; bar close_only 1.969): f1 1.420 ·
  **f2 2.101 (screen pass)** · f3 1.447 · f4 1.523 · f5 1.592 · f6 1.547.
  The cross-sectional-rank family (f3, inherited by f4–f6) is actively
  harmful: highest val ICs, broken books (dissociation #10; the Phase-1
  2-epoch val-IC 0.185 flag proved prophetic — rank features share the
  target's structure and inflate val IC mechanically).
- **f2 confirmation destroyed the screen result** (the MT pattern,
  replicated): CH 7-seed 1.601 / battery 1.682 (ref 2.147); BR 7-seed
  1.011 / battery 1.076 (ref 1.443); **2022: −0.67 / −1.22** vs baseline
  ≈ −0.15; val ICs 0.065–0.075 ABOVE baseline on every run
  (dissociations #11–12). The 3-seed 2.101 was a lucky draw. Meta-rules 2
  and 3 (screens select, batteries decide) earned their keep again.
- Leakage-clean construction verified (exact truncation test, 0.00e+00).

**Fixed XL (REJECT).** XL2 (312M params, ~1.19 GB checkpoints, produced and
saved for 10/27/44-feature inputs) **collapsed to mean-prediction at every
input width** — train loss pinned at the rank-target variance, OOS score
dispersion exactly 0.0, "blend" = the momentum leg alone (1.456) — while
the matched small S_ref trained healthily on every set (1.55–1.57
single-refit). Richer input does NOT rescue XL capacity: the collapse is an
optimization/data-scale property of the recipe (lr 3e-4, no warmup, ~150k
train samples), not information starvation. Footnote for the ledger:
XL-on-f2 logged val IC **0.167** during training — the highest ever
recorded in this project — attached to a model with constant OOS output.
- Un-run contingency (would need separate approval + pre-registration):
  XL-specific optimization (lr warmup/schedule). Given v11–v13's monotonic
  size-degradation at every trainable scale, expected value ≈ nil.

**Short-side feature research (Task 9): moot.** No feature set survived
confirmation and no XL model produces usable scores, so there is nothing
new to evaluate short legs on. v12's short-side rejection (short leg
standalone negative after 2× costs, 6/6 cells) stands unchanged.

## Standing state

Production unchanged and re-validated for the third consecutive arc:
preset B / close_only / tgt_rank_20 / 7-seed / equal history / 25-epoch
early stop / blend50+band10. Newly closed lines: feature expansion at
small scale (f1–f6, incl. the xs-rank family), fixed-XL at any tested
input width. The v13 feature library (v13_f1..f6, leakage-verified,
anchor-verified additive) remains in `dataset_transformer_eod.py` for the
~2027-01 full-field revisit — the one pre-authorized future feature line.
Standing dissociation count: **12** — the rule "never adopt on validation
IC" is now the single most-replicated finding this project has.

## Artifacts

queue_v13.json · current_feature_inventory.md · feature_family_plan.md ·
feature_set_specs.md · fixed_xl_architecture_plan.md ·
feature_quality_report.md · feature_missingness.csv ·
feature_correlation_summary.csv · small_model_feature_screen.{csv,md} ·
xl_feature_screen.{csv,md} · xl_learning_curves.csv ·
short_side_feature_report.md (mootness record) · queue_v13_results.{csv,md}
· logs/ · panels + 3×1.19 GB checkpoints (gitignored).
