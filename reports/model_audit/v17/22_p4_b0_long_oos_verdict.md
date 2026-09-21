# 22 — P4-B0-LONG OOS trajectory and H-EPOCH-LONG verdict

**Stage:** 3 of 4 under `P4_B0_LONG_AMENDMENT_1` (`c6e6a6d`). OOS was unblinded only after the mechanism report was committed (`93fff20`). **Evaluator:** `research/p4_long_eval.py oos`. **Machine-readable:** `p4_b0_long_oos_summary.json`.
**Evidence grade:** BURNED_DIAGNOSTIC on 2026-H1. Three refits (the independent unit), three seeds each. One refit (2026-07-23) scores a single session. Survivor universe and unadjusted prices apply throughout. Seeds are descriptive only.
**Scope:** more optimizer steps under the frozen recipe (batch 1024, constant LR 3e-4, AdamW wd 1e-4, no schedule), about 22,600–23,900 steps at epoch 100.

## Verdict

**H-EPOCH-LONG (frozen recipe: batch 1024, constant LR 3e-4 with gradient clipping, ≤ 23.9k steps): REJECTED under the preregistered rule, met at its minimum margins** (3/3 refits with 2026-07-23 at −0.014 on one session; 7/9 fits; mechanism Q2 0.002–0.004 below PARTIAL). Directional, not significant (L − E2–5 = −0.139, t(2) = −1.38, 95% CI −0.57 to +0.30); burned interval, n = 3 refits; about 81% of the magnitude comes from 2026-01-05. In plain terms, more steps under this recipe do not recover early-epoch ranking: late OOS is below the production-selected epoch on 9/9 fits, and validation shows the same. Mechanism: memorisation, in which long training strips the momentum and volatility exposure that makes up the early model's signal (Council Session 02). The OOS cost of that de-exposure depends on the factor's payoff in each block: momentum alone scored 0.33 on 2026-01-05 and −0.02 on 2026-07-23. This is the wording agreed in Council Session 02 (`COUNCIL_02_long_training_unblinded_synthesis.md`). The frozen label token stays REJECTED. The trajectory class is **A — continued decay**, and the mechanism class, frozen beforehand, is **NO_PHASE_CHANGE**.

**Sensitivity:** the label is fragile. It sits exactly on the 7-of-9-fits threshold, and its 3-of-3-refits leg depends on the single-session 2026-07-23 block at −0.014 (seed values +0.026, −0.047, −0.022). If that refit or one more fit flipped sign, the same frozen rules would return WEAKLY_CONTRADICTED. The mean effect is driven by 2026-01-05 (−0.339), the refit with the highest early IC, so continued decay cannot be fully separated from regression of an unusually strong epoch-3 block. The one-sided sign test for 3 of 3 refits is p = 0.125, and t = −1.38 on 2 degrees of freedom. The frozen "mean + 1 SE < 0" bar corresponds to only about 79% one-sided confidence at t(2).

The mechanism leg is also on a knife-edge. The frozen Q2 partial-recovery bar is +0.02 on two of three refits. 2026-01-05 reached +0.018 and 2026-07-23 reached +0.016, short by 0.002 and 0.004, which is under a quarter of the seed-level SE (about 0.02). Had Q2 been PARTIAL, the mechanism class would have been WEAK, REJECTED would have been unreachable, and the frozen rules would have returned INCONCLUSIVE.

Leave-one-refit-out, with the fit-count leg scaled from 7/9 to 5/6 (Red Team, Session 02):

| dropped refit | mean L − E2–5 | SE | fits negative | frozen label |
|---|---|---|---|---|
| none | −0.139 | 0.101 | 7/9 | REJECTED |
| 2026-04-20 | −0.177 | 0.163 | 5/6 | REJECTED |
| 2026-07-23 | −0.202 | 0.138 | 5/6 | REJECTED |
| 2026-01-05 | −0.039 | 0.025 | 4/6 | WEAKLY_CONTRADICTED |

The direction survives every drop. The REJECTED label does not survive dropping 2026-01-05.

Seed bootstrap (post-hoc; seeds resampled with replacement within each refit, 2,000 draws, frozen Q2 and verdict rules reapplied): REJECTED 51%, WEAKLY_CONTRADICTED 39%, INCONCLUSIVE 11%, and no supportive verdict in any draw. Because three-seed resampling understates reseed variance, even these shares overstate stability. The direction is robust; the choice between REJECTED and WEAKLY_CONTRADICTED is close to a coin flip.

**Deployment-relevant comparator (outside the frozen rules):** L minus the OOS IC at the production-selected epoch (epochs 1–6, chosen by the production early-stopping rule on validation) is −0.169 (SE 0.100), negative on 3/3 refits and 9/9 fits.

- Each late checkpoint (epochs 50, 75 and 100) is below epoch 3 on all three refits, and on seven or eight of nine fits.
- The late mean L (epochs 50/75/100) is below the unselected early regime (epochs 2–5) on all three refits and on seven of nine fits.
- The rejection rests on **direction consistency plus the mechanism read, not on statistical significance.** The 95% t-interval of L minus the early regime is −0.574 to +0.296, which spans zero widely at n = 3. REJECTED here means the preregistered one-SE, unanimous-direction criterion was met on a burned interval. It is not a significance test.

No production change follows. The result rejects only more steps under this recipe. It does not test LR decay or annealing, SWA or checkpoint averaging, or different regularisation.

## Raw OOS trajectory at every preregistered checkpoint

Mean OOS rank IC over refits, each refit the mean of its three seeds. SE is refit-clustered with n = 3 and t(2) = 4.303.

| epoch | ≈ steps | mean | SE | 95% t-CI | 2026-01-05 | 2026-04-20 | 2026-07-23 | refits > 0 | fits > 0 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.23k | +0.075 | 0.108 | −0.39, +0.54 | +0.283 | −0.081 | +0.023 | 2/3 | 6/9 |
| 3 | 0.70k | **+0.174** | 0.060 | −0.08, +0.43 | +0.284 | +0.158 | +0.078 | 3/3 | 9/9 |
| 5 | 1.2k | +0.151 | 0.059 | −0.11, +0.41 | +0.269 | +0.078 | +0.106 | 3/3 | 9/9 |
| 10 | 2.3k | +0.099 | 0.083 | −0.26, +0.46 | +0.223 | −0.058 | +0.133 | 2/3 | 7/9 |
| 15 | 3.5k | +0.036 | 0.035 | −0.12, +0.19 | +0.097 | −0.025 | +0.037 | 2/3 | 5/9 |
| 20 | 4.6k | +0.001 | 0.045 | −0.19, +0.19 | −0.022 | −0.063 | +0.087 | 1/3 | 3/9 |
| 30 | 7.0k | +0.011 | 0.029 | −0.11, +0.14 | −0.041 | +0.014 | +0.059 | 2/3 | 5/9 |
| 50 | 11.6k | +0.052 | 0.032 | −0.08, +0.19 | −0.011 | +0.088 | +0.078 | 2/3 | 7/9 |
| 75 | 17.4k | +0.002 | 0.026 | −0.11, +0.11 | −0.046 | +0.007 | +0.044 | 2/3 | 5/9 |
| 100 | 23.2k | **−0.020** | 0.049 | −0.23, +0.19 | −0.116 | +0.011 | +0.046 | 2/3 | 4/9 |

Epochs 1–15 reproduce P4-B0 exactly (maximum absolute OOS-IC difference 0.0). Seed means, descriptive only: epoch 3 is 0.137 / 0.187 / 0.197 for seeds 0/1/2 and L is +0.031 / −0.019 / +0.021, so all three seeds decline.

## Selection-aware comparison

Epoch 3 is labelled **SELECTED_COMPARATOR_ON_BURNED_INTERVAL**. The selection-aware bar for beating it is +0.023. The unselected early regime E2–5 is the mean of epochs 2–5.

| contrast | mean | SE | 95% t-CI | 2026-01-05 | 2026-04-20 | 2026-07-23 | refits > 0 | fits > 0 |
|---|---|---|---|---|---|---|---|---|
| L − epoch 3 | −0.162 | 0.094 | −0.57, +0.24 | −0.342 | −0.123 | −0.023 | 0/3 | 1/9 |
| L − E2–5 | −0.139 | 0.101 | −0.57, +0.30 | −0.339 | −0.064 | −0.014 | 0/3 | 2/9 |
| L − epoch 15 | −0.025 | 0.066 | −0.31, +0.26 | −0.154 | +0.061 | +0.018 | 2/3 | 3/9 |
| epoch 100 − epoch 3 | −0.193 | 0.109 | −0.66, +0.27 | −0.400 | −0.147 | −0.033 | 0/3 | 1/9 |
| epoch 50 − epoch 3 | −0.122 | 0.089 | −0.51, +0.26 | −0.295 | −0.070 | −0.001 | 0/3 | 2/9 |

No late checkpoint clears the +0.023 bar on any refit. The closest is 2026-07-23 at epoch 50, 0.001 below its own epoch 3. After epoch 5, the only refit-level values above epoch 3 are also on 2026-07-23, at epochs 10 (+0.054) and 20 (+0.009). Epoch 3's own premium over E2–5 in these three refits is +0.023 (SE 0.018, positive on 3/3), consistent with the winner's-curse haircut recorded in the amendment.

## Late-maximum null

The statistic is the maximum minus the mean over a window, per fit, refit-weighted. The null is per-fit AR(1) noise with the fit's own SD and autocorrelation, 20,000 simulations.

| window | observed max − mean | null mean | null 95th pct | null percentile of observed | raw mean max | null-corrected max | null-corrected max − epoch 3 |
|---|---|---|---|---|---|---|---|
| checkpoints 20/30/50/75/100 | 0.059 | 0.057 | 0.073 | 62nd | 0.068 | 0.011 | −0.162 |
| epochs 30–100 | 0.112 | 0.110 | 0.123 | 58th | 0.116 | 0.006 | −0.168 |

The best late epochs are about as high as noise alone predicts. A raw maximum over epochs 30–100 of 0.116 looks like partial recovery, but 0.110 of it is the expected maximum of noise around a late mean of 0.004. **The epoch-50 value of +0.052 is inside this null** and is not evidence of a late regime. One limit applies: the null is calibrated on each series' own SD and autocorrelation, so a sustained sub-window level shift would inflate the null too. The null is consistent with noise; it cannot rule out a short sustained regime, and it is not an input to the verdict.

## Trajectory class and support checks

- **Aggregate class A — continued decay:** L = 0.011 against epoch 15 = 0.036, below the 0.02 band by only 0.005. Class B would also meet the REJECTED rule. A plainer description of the aggregate curve is a collapse by about epoch 20 to a near-zero plateau, with no return to the early regime.
- **By refit the picture is mixed:** 2026-01-05 is A (L = −0.058), 2026-04-20 is C, partial recovery relative to its own epoch 15 (L = 0.035 vs −0.025), and 2026-07-23 is D (L = 0.056 vs an early regime of 0.070). The last is the single-session block.
- Support checks for any recovery class all fail. Epochs 50/75/100 are not all above epoch 15. The mechanism class is NO_PHASE_CHANGE. L minus epoch 15 is not more than 2 SE above zero.

The two refits classed C and D improve only relative to their own low epoch-15 values. They stay below their early regime. This resembles the frozen mechanism observation of a non-sustained mid-run validation bump. On OOS, those two refits are higher at epochs 30–50 than at 20 (epoch 50: +0.088 and +0.078), then fall back by 75–100 (+0.007 and +0.044 at 75). The timing differs from validation, which peaked at epochs 31–40.

## Label-free book-shape diagnostic (secondary, not a performance claim)

This is the mean per-date overlap of the top quintile of seed-averaged transformer scores at epoch k with epoch 3.

| epoch | 2026-01-05 | 2026-04-20 | 2026-07-23 |
|---|---|---|---|
| 5 | 0.78 | 0.72 | 0.64 |
| 15 | 0.53 | 0.64 | 0.68 |
| 30 | 0.35 | 0.56 | 0.68 |
| 50 | 0.38 | 0.57 | 0.59 |
| 100 | 0.42 | 0.51 | 0.59 |

By epoch 20 the late model picks a materially different top quintile, sharing about 35–68% of names with epoch 3. Longer training changes what the book would hold, not only how scores are scaled.

## Original preregistered classifier — ORIGINAL_PREREGISTERED_CLASSIFIER / STRUCTURALLY_FLAWED / NOT_PRIMARY_EVIDENCE

The original frozen rule returns **LONG_EPOCH_NO_BENEFIT.**

- **It agrees in direction with the primary verdict,** but for a weaker reason. It compares against the early regime of these three refits (0.150), which the Council had flagged as the most hypothesis-hostile sub-sample (epoch-3 OOS 0.174 here vs 0.086 on the other six P4-B0 refits).
- **The Council's prediction about this rule was wrong.** In Session 01, the members' own confidence statements put POSSIBLE_RECOVERY at 85% (Red Team, ML Researcher and Auditor in Round 1), revised to 80–90% in Round 2, and the Red Team put NO_BENEFIT at under 5%. These figures come from the round transcripts, which were not committed; the committed Session-01 synthesis says only that NO_BENEFIT was "effectively unreachable" and POSSIBLE_RECOVERY the "default". The Auditor adds, in Session 02, that the miss was avoidable: epoch 15 (0.036) and the early regime (0.150) for these nine fits were already known from P4-B0, and with those values even a plateau made NO_BENEFIT likely. The Council reasoned about the rule unconditionally instead of simulating it on the comparator values it already had. The Council's structural analysis still holds: the maximum over 71 epochs is biased by about +0.11, and NO_BENEFIT needs a true late mean below about 0.02–0.025. What the Council under-weighted was the chance that the late mean itself would fall that low. It did, to 0.004 over epochs 30–100, so the rule fired despite the bias. The classifier stays structurally flawed. It happened to give a correct direction on an extreme outcome.

## What this does and does not establish

- **Mechanical facts:** the harness reproduces P4-B0 bit-for-bit, and the LONG artifacts match the pre-unblinding checksums. The `inf` gradient norms are AMP GradScaler overflow steps, which the scaler skips (at least 71 steps, 0.034% of all steps); no other channel is non-finite.

## Erratum to the frozen mechanism report (21)

Report 21 (Q4) says the `inf` gradient norms are growth probes "spaced by roughly 2,000 to 2,900 optimizer steps". That spacing was an average and is wrong as stated. Observed gaps between `inf` epochs range from 1 to 26 epochs, about 230 to 6,000 steps. At least 20 of the 62 gaps are shorter than the 2,000 clean steps a growth probe requires after a backoff, so heavy-tailed gradient spikes also overflow; the largest finite per-step norm in an epoch reaches 10.8. The GPU Reviewer found this in Council Session 02. The conclusion is unchanged: every overflow step is skipped by `GradScaler`, train loss and weight norm change no differently in those epochs, and the label is not an input to the phase-change class. Report 21 is left as frozen; this erratum supersedes its spacing sentence.
- **Statistical evidence:** unanimous direction across refits, and seven or eight of nine fits below epoch 3 at each of epochs 50, 75 and 100. Intervals span zero at n = 3. The late maximum is consistent with a noise null, with the limit noted above.
- **Interpretation:** under this recipe, extra steps drive memorisation (in-sample R² from 1.5% to 18.8%) and a growing, increasingly seed-specific prediction spread. Neither validation nor OOS ranking gets better.
- **Not established:** anything about LR schedules, averaging or regularisation. Anything prospective. Anything about the book-level P&L of late-epoch models.

## BOOK_EQUIVALENCE

**BYTE_IDENTICAL.** The 2026-09-11 production decision book, its markdown summary and the universe scores were regenerated from frozen inputs into a temporary directory. All three match the on-disk production files byte for byte, and those files were not modified. 2026-09-11 is the latest production book: `daily_ops` is run manually and has not run since.
