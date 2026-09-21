# 21 — P4-B0-LONG mechanism report (frozen before OOS unblinding)

**Stage:** 2 of 4 under `P4_B0_LONG_AMENDMENT_1` (commit `c6e6a6d`). **Evaluator:** `research/p4_long_eval.py` (commit `7d5378b`).
**Nothing in this report uses OOS rank IC.** The OOS-IC field was dropped at load, and the OOS prediction array was never accessed. `oos_pred_std` is the SD of predictions on OOS-block inputs and uses no labels.
**Machine-readable:** `p4_b0_long_anchor_summary.json`, `p4_b0_long_mechanism_summary.json`.
**Scope:** more optimizer steps under the frozen recipe (batch 1024, constant LR 3e-4, AdamW wd 1e-4, no schedule). Three refits × three seeds. Refit is the unit (n = 3). All data are BURNED_DIAGNOSTIC on 2026-H1.

## Verdict

**MECHANISM_NO_PHASE_CHANGE**, by the frozen rules. Past epoch 15, training loss keeps falling and prediction dispersion keeps expanding. Validation ranking quality does not recover in the 50–100 window, and no non-IC channel shows a transition. The run moves steadily into a memorisation regime: in-sample fit rises about twelve-fold while validation rank IC ends below its epoch-15 level on one refit and barely above it on the other two.

## Stage 1 — reproduction anchor

**PARITY_PASS.** All nine LONG cells match P4-B0 for epochs 1–15 **bit-for-bit** on train loss, validation IC, validation loss and prediction dispersion. The split metadata match exactly, and all 18 SHA-256 checksums verify. The long harness is on the same trajectory as P4-B0, so the extension from epoch 16 onward is directly comparable.

## Mechanism trajectory

Mean over three refits (each the mean of three seeds). Steps per epoch are 226, 232 and 239 for the three refits.

| epoch | ≈ steps | train loss | in-sample R² | val rank IC | val MSE | val pred SD (÷ target SD) | seed agreement | epoch churn | weight norm |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.23k | 0.3347 | −0.4% | 0.045 | 0.3328 | 1.2% | 0.24 | — | 69.00 |
| 3 | 0.70k | 0.3329 | 0.1% | **0.100** | 0.3318 | 3.3% | 0.75 | 0.110 | 69.09 |
| 5 | 1.2k | 0.3322 | 0.3% | 0.076 | 0.3335 | 7.6% | 0.57 | 0.095 | 69.24 |
| 10 | 2.3k | 0.3303 | 0.9% | 0.040 | 0.3374 | 13.0% | 0.52 | 0.167 | 69.62 |
| 15 | 3.5k | 0.3284 | 1.5% | 0.026 | 0.3406 | 17.3% | 0.49 | 0.246 | 69.98 |
| 20 | 4.6k | 0.3263 | 2.1% | 0.036 | 0.3437 | 20.0% | 0.65 | 0.213 | 70.35 |
| 30 | 7.0k | 0.3205 | 3.9% | 0.055 | 0.3512 | 26.2% | 0.62 | 0.151 | 71.14 |
| 50 | 11.6k | 0.3069 | 7.9% | 0.042 | 0.3668 | 34.5% | 0.48 | 0.143 | 72.51 |
| 75 | 17.4k | 0.2885 | 13.5% | 0.028 | 0.3869 | 42.0% | 0.38 | 0.142 | 74.05 |
| 100 | 23.2k | 0.2708 | 18.8% | 0.022 | 0.4050 | 48.4% | 0.34 | 0.113 | 75.53 |

Seed agreement is the mean pairwise Spearman correlation of validation predictions across the three seeds of a refit. Epoch churn is one minus the Spearman correlation of validation predictions with the previous epoch.

Validation rank IC by refit:

| epoch | 2026-01-05 | 2026-04-20 | 2026-07-23 |
|---|---|---|---|
| 3 | 0.028 | 0.133 | 0.141 |
| 15 | 0.007 | 0.048 | 0.023 |
| 30 | 0.048 | 0.059 | 0.058 |
| 50 | 0.032 | 0.045 | 0.050 |
| 75 | 0.020 | 0.022 | 0.043 |
| 100 | 0.025 | 0.015 | 0.025 |

## Answers to the six mechanism questions (frozen rules)

**Q1 — Does train loss keep falling after epoch 15? YES: CONTINUES, no ACCELERATION.** It falls by 0.056–0.060 from epoch 15 to 100 on every refit, about 13 times the epoch 3–15 drop of 0.0045. The slope steepens gradually, from −0.00038 per epoch over 3–15 to −0.00072 per epoch over 50–100. The 50–100 slope is only 1.37× the 15–30 slope, short of the frozen factor of 2 needed to count as acceleration. In-sample R² goes from 1.5% to 18.8%. The Council's Round-1 extrapolation of about 0.299 at epoch 100 was too conservative: the model memorises faster than linearly.

**Q2 — Does validation rank IC recover materially late? NO.** The late mean (epochs 50/75/100) minus epoch 15 is +0.018, −0.021 and +0.016 by refit, a mean of +0.005, and no refit reaches the +0.02 band. At epoch 100, validation IC is 0.022 against 0.100 at epoch 3.

**Q3 — Does prediction dispersion expand, plateau or contract? EXPANDING** on all refits. Validation prediction SD grows 1.30–1.57× from epoch 50 to 100, and the label-free OOS-input dispersion grows 1.18–1.23×. It reaches 48% of the target SD by epoch 100, so the model is still under-dispersed but no longer shrunk.

**Q4 — Do gradient norms show structured learning or noisy updates? Frozen label INSTABILITY, which is an AMP artefact.** The label fires because the per-epoch mean of the pre-clip gradient norm is `inf` in 71 of 900 fit-epochs. Those are mechanical AMP loss-scaler probes, not divergence:
- They appear only after epoch 15, about eight per fit, spaced by roughly 2,000 to 2,900 optimizer steps. That matches `GradScaler`'s default of doubling its scale after 2,000 clean steps; an overflowing probe step is skipped and the scale is halved.
- No other channel has a non-finite value. Weight norm never jumps more than 0.12 between epochs, and train loss never rises more than 0.0004.
- About eight skipped steps per fit are a 0.03% loss of optimizer steps.

Setting those skipped steps aside, the frozen sub-rules give MIXED on every refit. Cross-seed agreement falls from 0.49 at epoch 15 to 0.34 at epoch 100, while churn also falls, from 0.25 to 0.11. The finite mean gradient norm rises about five-fold, from 0.07–0.09 at epochs 1–15 to 0.41 at 75–100, and never averages above 0.64. Individual steps exceed the 1.0 clip in 6% of epochs 1–15 and 69% of epochs 50–100, so clipping becomes intermittently active late. The picture is steady, increasingly seed-specific movement, which is neither convergence nor a new structured regime. The INSTABILITY label is not an input to the phase-change classification, so this artefact does not affect Q6.

**Q5 — Does weight norm show a transition? NO: SMOOTH.** Weight norm grows almost linearly from 69.98 at epoch 15 to 75.53 at epoch 100. The 50–100 growth rate is 0.74–0.84× the 15–30 rate on every refit, inside the frozen 0.5–2× band. As expected, the numerically inert weight decay does not cap growth.

**Q6 — Is there a new late-training regime? MECHANISM_NO_PHASE_CHANGE.** Q2 is NO, and none of the four non-IC transition signals fires: no acceleration, dispersion still expanding, no convergence, weight norm smooth.

## Observation outside the frozen rules, recorded before OOS unblinding

**A transient mid-run validation bump, not sustained.** On all three refits, validation rank IC rises again after the epoch-15 trough, peaks around epochs 31–40, and then decays:

- Band means are 0.034 over epochs 16–20, 0.051 over 31–40, and 0.023 over 76–100.
- Per-refit peaks after epoch 15 are 0.049, 0.067 and 0.072 at epochs 37, 31 and 40. These are maxima over 85 epochs and carry the usual max-selection bias.
- At epoch 30 the rise over epoch 15 is +0.041, +0.011 and +0.035.
- Seed agreement rises in the same window on two of three refits (0.41 to 0.68 and 0.45 to 0.63 from epoch 15 to 20; flat at 0.63 on 2026-01-05), before falling to about 0.34 on all three.
- The hump stays below the epoch-3 validation IC on two of three refits: 0.067 vs 0.133, and 0.072 vs 0.141. The exception is 2026-01-05, where it reaches 0.049 vs 0.028.

The frozen rules score only the 50–100 window against epoch 15, so the classification is unchanged. The observation is minuted here so that it cannot be retrofitted after OOS unblinding. In Stage 3, OOS checkpoints 20 and 30 are reported like every other checkpoint. **Any OOS bump there is descriptive only.** Under the original spec, no new preferred epoch may be chosen from this set, and 20–30 would be a data-selected window on a burned interval.

## Claim boundary

These results describe about 22,600–23,900 optimizer steps under the frozen recipe. They say nothing about LR decay or annealing, SWA or checkpoint averaging, or different regularisation. Those remain separate, untested hypotheses.

## Stage 3 gate

This report and its JSON are committed before any OOS rank IC is computed. The OOS stage of the evaluator refuses to run until `p4_b0_long_mechanism_summary.json` is committed and unmodified.
