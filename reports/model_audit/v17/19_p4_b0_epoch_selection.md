# 19 — P4-B0: epoch-selection diagnostic (is validation-IC early stopping informative?)

**Burned methodology diagnostic** (2026-H1, already used for research
selection) — not OOS validation, not an edge estimate. Spec frozen before
the run in `p4_validation_policy_spec.json → P4_B0`; artifacts
`p4/b0/*.json` (per fit, per epoch), `p4/b0_epoch_curves.csv`,
`p4/b0_summary.csv`, `p4/result_P4_B0.json`.

Run: 9 deterministic refit dates (2026-01-05, 01-26, 02-25, 03-26, 04-20,
05-12, 06-09, 07-01, 07-23) × seeds {0,1,2} = **27 fits × 15 epochs**,
early stopping disabled, production Policy-A split per refit, per-epoch
train loss / val IC / val MSE / subsequent 5-session OOS block IC /
prediction dispersion recorded through a research-only `predict_idx`
wrapper (OOS never influences optimisation; the net is discarded).
**0.77 h GPU** (01:30–02:17, 0 failed), launched after the production
window with no `daily_ops` having run that night (GPU idle).
Example separation (refit 2026-01-05, seed 0, from `b0_2026-01-05_*_s0*.json`):
training ends **2024-10-28**; purge 21 sessions; validation **2024-11-28 →
2025-12-02** (ends 21 sessions before the refit); OOS block = the refit's
own 5 sessions **2026-01-05 → 2026-01-09** (labels matured, cache through
2026-09-11); production rule would have selected epoch 4 on this fit.

## Mean curves across the 27 fits

| epoch | train loss | val IC | val MSE | **OOS IC** | pred std |
|---|---|---|---|---|---|
| 1 | 0.3347 | 0.075 | 0.3326 | 0.058 | 0.013 |
| 2 | 0.3332 | 0.094 | 0.3322 | 0.099 | 0.036 |
| **3** | 0.3329 | **0.107** | **0.3320** | **0.115** | 0.048 |
| 4 | 0.3326 | 0.101 | 0.3324 | 0.090 | 0.065 |
| 5 | 0.3322 | 0.093 | 0.3328 | 0.106 | 0.088 |
| 6 | 0.3317 | 0.081 | 0.3333 | 0.107 | 0.110 |
| 8 | 0.3310 | 0.059 | 0.3356 | 0.085 | 0.127 |
| 12 | 0.3295 | 0.037 | 0.3386 | 0.079 | 0.151 |
| 15 | 0.3284 | 0.036 | 0.3391 | 0.060 | 0.174 |

On average the validation curve is *right about the shape*: val IC, val
MSE and OOS IC all peak at epoch 3, then decay while train loss keeps
falling and prediction dispersion keeps growing (classic over-fit).

## Selection outcomes (27 fits; oracle = ex-post OOS-best epoch, unattainable)

| Rule | mean OOS IC | regret to oracle | beats fixed-3 on | note |
|---|---|---|---|---|
| **fixed epoch 3** (ex-ante primary comparator = historical median) | **0.1153** | **0.090** | — | pos-IC 78% |
| fixed 5 | 0.1055 | 0.099 | | pos 70% |
| fixed 2 | 0.0988 | 0.106 | | pos 81% |
| production rule (patience 3 / min 2 on the val curve) | 0.1068 | 0.098 | 30% of fits | selects mean 3.2, max 6 |
| val-best epoch (argmax over 1..15) | 0.0946 | 0.110 | 30% | selects mean 4.0, range 1–15 |
| fixed 8 / 12 | 0.085 / 0.079 | 0.120 / 0.126 | | |
| oracle | 0.2049 | 0 | | mean epoch 6.2 |

Paired, refit-clustered SE (fits of one refit share the OOS block):
production-selected − fixed 3 = **−0.0086 (SE 0.011)**; val-best − fixed
3 = −0.0207 (SE 0.013). By seed (prod − fixed 3): −0.045 / +0.002 /
−0.019. By refit (val-best − fixed 3): ≤ 0 on 7 of 9 refits (two ties).
Val-best lies within 2 epochs of the oracle in 56% of fits (median gap 2).

## Correlations

- Aggregate over all 405 epoch observations: Spearman **−0.22**, Pearson
  −0.21 (this pools refits and re-imports the cross-refit trend — reported,
  not used).
- **Within-refit (primary):** median Spearman **+0.24**, mean +0.17, IQR
  [−0.36, +0.70], positive in 63% of fits; by seed 0.23 / 0.75 / 0.11;
  by refit from −0.78 (07-01) to +0.73 (03-26) — unstable.

## Answers

- **Q1** Higher val IC predicts higher later OOS IC only weakly and
  unstably within a fit (median ρ +0.24, IQR spanning zero).
- **Q2** Val-best is near the OOS-best epoch about half the time (56%
  within 2).
- **Q3** Regret: fixed epoch 3 **0.090** < production rule 0.098 <
  val-best 0.110. Selection by validation *costs* ~0.01–0.02 OOS IC
  relative to the ex-ante fixed epoch.
- **Q4** Yes: a fixed epoch in the 3–5 range performs as well or better
  than validation-based stopping (3 best; 5 comparable; ≥ 8 clearly worse).
- **Q5** Direction holds across seeds (2 of 3 negative, 1 flat) and
  across refits (7 of 9 non-positive).

## Classification (frozen rule, applied mechanically): **EARLY_STOPPING_WEAK**

Not INFORMATIVE (production rule does not beat fixed 3; +0.010 bar not
met; 30% win rate). Not UNINFORMATIVE by the frozen rule because the
within-refit median association (+0.24) exceeds the +0.10 line — the
validation curve carries the *average* signal (peak ≈ 3) even though
per-fit argmax selection adds more noise than it removes.

## Implications

1. **For Policy D**: the calibration step should not use per-fit
   val-best epochs (noisy); an **ex-ante fixed E = 3** (from the
   historical distribution and this curve) is the better-supported
   calibration → proposed amendment "D_fixed3: train on train+val for 3
   epochs" (uses the exact `fit_fixed_epochs` mechanism; no per-fit
   selection at all).
2. **For Policies B/C**: a recent/shorter validation block might make
   per-fit selection *more* informative than the stale 263-session block,
   but B0 shows the ceiling for any per-fit selection here is low; B is
   worth one bounded test, C is lower priority.
3. **Full P4-B (243 fits, 3.6 h) is not justified as specified.** A
   reduced screen is: **D_fixed3 (81 fits, ≈1.2 h) + B_recent126 (81,
   ≈1.2 h)**, same paired design and thresholds; C and the val-best-
   calibrated D deferred. Not launched.
4. Nothing here changes production: the daily retrain keeps its
   validation-IC early stopping until a challenger passes the frozen bar
   and shadow period.
