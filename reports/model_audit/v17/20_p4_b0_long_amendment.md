# 20 — P4-B0-LONG: interpretation amendment recorded before unblinding

**Status:** PREREGISTERED DEVIATION. Recorded 2026-09-21 10:48 +0800, before any P4-B0-LONG artifact was opened, parsed or evaluated.
**Machine-readable version:** `p4_validation_policy_spec.json` → `P4_B0_LONG_AMENDMENT_1` (authoritative if the two differ).
**Original rule as launched:** `p4_validation_policy_spec.json` → `P4_B0_LONG` (commit `0be4c68`, unchanged by this amendment).

## Why the original rule is not the primary evidence

The blind Research Council review (Session 01, `COUNCIL_01_long_training_synthesis.md`) found six problems before any result was seen:

1. **Effective sample size is the number of refits.** Each refit date is one future market block. Seeds are replicate models inside that block. Effective n is 3, not 9 fits and not 900 fit-epochs.
2. **One block is truncated.** The 2026-07-23 refit scores a single session, so one of three clusters is one cross-section of about 108 names.
3. **NO_BENEFIT is effectively unreachable.** It needs the mean per-fit maximum OOS IC over 71 late epochs to fall below the early regime. That maximum is biased upward by about +0.11 from noise alone. The exact rule already returns POSSIBLE_RECOVERY on the clean-decay P4-B0 data for these refits.
4. **POSSIBLE_RECOVERY is the default outcome** under noisy null behaviour. It is not evidence of recovery.
5. **PROMISING has an unacceptable false-positive rate.** It fires about 9.5% of the time under the reviewed null, with near-zero power at its own +0.010 threshold.
6. **Epoch 3 was selected on this burned interval.** Its premium over the mean of epochs 2–5 is +0.0128. A "better than epoch 3" claim must clear +0.023, and the unselected early regime (mean of epochs 2–5) is reported alongside.

> **The original frozen classifier will still be reported for historical transparency, labelled ORIGINAL_PREREGISTERED_CLASSIFIER / STRUCTURALLY_FLAWED / NOT_PRIMARY_EVIDENCE. It is not primary evidence and will not determine the verdict.**

## What the experiment can and cannot show

It tests **more optimizer steps under the current frozen recipe**: batch 1024, constant LR 3e-4, AdamW with numerically inert weight decay, no schedule or warm-up, same model, features and target. That is about 22,600–23,900 steps at 100 epochs. A negative result rejects only that. It says nothing about LR decay, cosine schedules, SWA or checkpoint averaging, or different regularisation. Those are separate hypotheses and are not launched here.

## Unblinding order (frozen)

| stage | reads | gate |
|---|---|---|
| 0 | nothing | this amendment and the evaluator are committed first |
| 1 Anchor | LONG epochs 1–15 vs the nine shared P4-B0 cells: train loss, validation IC, validation loss, prediction dispersion, split metadata. **Not OOS IC.** | on LONG_HARNESS_PARITY_FAILURE, stop |
| 2 Mechanism | train loss, validation rank IC, prediction dispersion, gradient norm, weight norm, validation-prediction movement and cross-seed agreement. The OOS-IC field is dropped at load; the OOS prediction array is never accessed. | mechanism report committed before stage 3 |
| 3 OOS | full OOS trajectory, refit-clustered uncertainty, selection-aware comparison, late-maximum null, verdict; then the original classifier last | — |
| 4 Council | the same frozen evidence package to all four members | — |

## Frozen anchor tolerance

Basis: same-seed GPU reruns reproduced validation IC to four decimals in P0-A, and seed-to-seed spread in P4-B0 is about 0.02 in validation IC. A harness divergence would look like a different seed, so tolerances sit far below that spread and far above the rerun band.

| check | epochs 1–3, every cell | epochs 4–15, every cell |
|---|---|---|
| train loss, absolute | 0.0001 | 0.0005 |
| validation IC, absolute | 0.002 | 0.01 |
| validation loss, absolute | 0.0002 | 0.001 |
| prediction dispersion, relative | 5% | 25% |

Pooled over epochs 4–15: mean absolute validation-IC difference at most 0.005, and mean signed difference at most 0.003 in absolute value. Split metadata must match exactly. A pass with late drift is allowed only if per-cell epoch-4–15 bounds fail in at most 10% of cell-epochs, all at epoch 6 or later. Anything else is LONG_HARNESS_PARITY_FAILURE.

## Frozen mechanism rules

All per refit, averaging its three seeds; n = 3 refits.

- **Q1 train loss:** continues if it falls at least 0.005 from epoch 15 to 100 on all refits. Acceleration if the 50–100 slope is at least twice as steep as the 15–30 slope on two of three refits.
- **Q2 validation rank IC:** genuine late recovery if the mean of epochs 50/75/100 exceeds epoch 15 by at least 0.02 on average, on all three refits, with each of 50, 75 and 100 above epoch 15. Partial if the late mean exceeds the lowest of epochs 15/20/30 by 0.02 on two of three refits.
- **Q3 dispersion:** expanding, plateau or contracting by a ±10% band from epoch 50 to 100.
- **Q4 gradients and prediction movement:** structured, noisy, converged, instability or mixed, from epoch-to-epoch validation-prediction churn, cross-seed agreement and gradient norms.
- **Q5 weight norm:** transition if the 50–100 growth rate differs from the 15–30 rate by more than a factor of two, or changes sign.
- **Phase change:** CLEAR needs a genuine validation recovery plus a coincident non-IC change. WEAK needs a partial recovery, or a recovery without a coincident change, or two non-IC channels changing. Otherwise NO.

## Frozen OOS analysis and verdict rules

- Refit is the cluster. Point estimate is the mean of three refit values. SE is their SD over √3, with a t(2) confidence interval and a leave-one-refit-out range. Seeds are descriptive only.
- Every checkpoint is reported: 1, 3, 5, 10, 15, 20, 30, 50, 75, 100. Never only the best epoch.
- Paired difference against epoch 3 (selected, bar +0.023) and against the unselected mean of epochs 2–5.
- Any late maximum is reported next to its null: the expected maximum-minus-mean of the same window under an AR(1) noise model fitted to each fit's own window.
- Trajectory classes, from the late mean L = mean of epochs 50/75/100: A continued decay, B plateau, C partial recovery, D full recovery, E late improvement beyond the early peak. C, D and E need neighbouring-checkpoint, refit, mechanism and 2-SE support, or they are reported as an unsupported plateau.
- Verdict rules for H-EPOCH-LONG (SUPPORTED through REJECTED) are frozen in the JSON. SUPPORTED is stated in advance to be practically unattainable at n = 3 on a burned interval.
- No production change may be promoted from this experiment under any verdict.

## Integrity

SHA-256 checksums of all 18 LONG files are recorded in the JSON. They were taken at byte level without decoding. Any later change to the artifacts is detectable.
