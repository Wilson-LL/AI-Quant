# AI-Quant Research Council — Session 01 — Lead Synthesis (Round 3)

**Question:** Could substantially longer training — potentially 100+ epochs — improve AI-Quant OOS ranking quality despite the early peak observed around epoch 3?

**Date:** 2026-09-15. **Lead:** main session. **Members:** Quant Red Team, ML Training/Architecture Researcher, Data/Validation Auditor, GPU/Systems Reviewer (project agents under `.claude/agents/`, run as read-only subagents; Agent Teams unavailable in this runtime, so Round 2 was Lead-relayed with the verbatim Round-1 reports on disk).

**Blinding:** all rounds were run while `P4-B0-LONG` (3 refits × 3 seeds × 100 epochs) was in flight. No member opened `reports/model_audit/v17/p4/b0_long/`; the Lead has not run `--evaluate-long`. Nothing below uses the 100-epoch results. Evidence is `p4/b0_epoch_curves.csv` (27 fits × 15 epochs), `p4/b0_summary.csv`, `p4/b0/*.json`, `p4/fits/A_current_*.npz`, v12 `epoch_curves.csv`, and code.

**Evidence grade of everything here:** BURNED_DIAGNOSTIC on 2026-H1 (an interval already used ~111 times), plus MECHANICAL_CODE_PROOF for code claims. Zero PROSPECTIVE, zero PAIRED_EXPERIMENT on the question itself.

The Round-1 and Round-2 reports are summarised here with their key numbers. The verbatim transcripts lived in a temporary session directory and are intentionally not committed. Nothing was voted on.

---

## 1. AGREED FINDINGS (all four members, after debate)

**A1. The frozen P4-B0-LONG classifier cannot decide the question, in either direction.** (Auditor, Red Team, ML independently; GPU concurs.)
- `NO_BENEFIT` is effectively unreachable: `oos_late_max` is a max over 71 noisy epochs (`run_p4_validation_policy.py:578`, `:595`). Simulated bias +0.112 to +0.116, robust to the true within-fit autocorrelation (ρ up to 0.567). Running the exact rule on the clean-decay B0 data for the same three refits returns `POSSIBLE_RECOVERY`.
- `PROMISING` is a ≈9.5% false-positive coin flip under the null (full-rule simulation, between-refit SD 0.149, seed SD 0.069) with ≈0% power at the declared +0.010.
- `POSSIBLE_RECOVERY` is therefore the design's default output, not evidence. This must be minuted before unblinding.

**A2. Agreed uncertainty arithmetic for the level contrast** (Δ = mean OOS rank IC at late checkpoints − epoch 3, per fit, refit-clustered):

| quantity | value |
|---|---|
| refit-clustered SE at n = 3 refits | 0.081 |
| refit-clustered SE at n = 9 refits | 0.047 |
| refit clusters for SE = 0.010 | ≈195–200 |
| MDE at n = 3, 80% power | ≈0.23 |
| agreed contrast on the three LONG refits in B0 (ep {12,15} − ep 3) | −0.105 ± 0.051 |
| seed averaging (3 seeds) reduces SE by | ≈6% (level) / ≈13% (shape) |

`evaluate_long` (`:559-617`) computes no SE, HAC, bootstrap or clustering.

**A3. Honest comparator bar ≈ +0.023, not +0.010.** Epoch 3 was the winner of a 5-way comparison on the same 27 fits; its premium over the mean of epochs 2–5 is +0.0128 (SE 0.0091), over the mean of the five pre-registered comparators +0.0188.

**A4. One of the three LONG refits scores a single session.** `block_end = min(r0+4, e_rank)` (`:530`); `b0_2026-07-23_c5_s0_*.json` has `oos_start == oos_end`. Its within-fit SD is 46% higher than the five-session blocks. The effective n of the design is 3 blocks, one of them one cross-section, and cross-sectional SE is worse than 1/√107 because the 108 names share a market factor and ≈65% momentum correlation.

**A5. Mechanism arithmetic removes the two textbook late-recovery mechanisms.** Decoupled weight decay is mechanically confirmed (`torch/optim/adam.py:419`) and numerically inert: shrink factor 0.99930 (0.070%) over the run. No LR schedule or warm-up exists in `fit_one` (`train_transformer_eod.py:254-384`). Train loss reaches R² ≈ 1.5% of the 1/3 floor at epoch 15 and extrapolates to ≈10% at epoch 100: no interpolation, so no epoch-wise double descent. Grokking needs weight-decay pressure that is absent.

**A6. Epochs are optimizer steps at fixed batch; the result must be reported in steps at batch 1024 / constant LR 3e-4.** Corrected counts from the reused Policy-A fits (`p4/fits/A_current_*.npz` n_train 231,242 / 237,533 / 243,740): 226 / 232 / 239 steps per epoch; epoch 3 = 678–717 steps; 100 epochs = 22,600–23,900 steps. No internal batch/step confound (batch frozen, constant LR, length-invariant RNG, nine cells shared with B0). Batch optimisation comes AFTER, and probably never for this question.

**A7. The late regime is not pure noise.** Per-refit epoch slopes reproduce across disjoint seeds (Pearson +0.886 on epochs 1–15, +0.857 on 8–15; leave-one-out range +0.787 to +0.954). After removing the common epoch curve and the per-refit level, the block × epoch residual is still +0.567 seed-reproducible. 64% of within-fit epoch-deviation variance is refit-shared with lag-1 autocorrelation 0.73. Four observables reject a stationary iterate over epochs 8–15 (train loss falls in 27/27 fits, t −32.5; dispersion rises t +17.4; val MSE t +6.1; val rank IC t −2.2). The ML Researcher conceded this.

**A8. Raw validation MSE is not evidence of overfitting.** Decomposing MSE with the fixed-variance target (1/3): implied Pearson r rises +0.066 → +0.122 from epoch 3 to 15 and optimally-rescaled MSE falls 0.3319 → 0.3284. The rise in raw MSE is a dispersion artefact. Prediction dispersion 0.013 → 0.174 is 2.3% → 30% of the target's 0.577 SD: the model is un-shrinking, still 3.3× under-dispersed at epoch 15, not over-confident. The deployment-relevant instrument is per-date rank IC.

**A9. Nobody has measured epoch-N weights at N > 15 with best-validation restore disabled.** v12's epoch-expansion arms kept patience and restore (deployed ≈ epoch 13, not 35). The Red Team conceded this; it is the one legitimate reason the frozen long fits are worth having, as a mechanism probe.

**A10. Compute is not the binding resource.** The frozen run costs 1.73 GPU-h (spec's 6.9 s/epoch was exactly right; the GPU Reviewer's 8.3 s/epoch median was the production reference class). Over half of each research epoch is fixed overhead (val forward + pandas rank IC + OOS hook), not training. The highest-value next items cost zero GPU-hours.

**A11. Mechanical validity holds.** Leakage/purge/maturity asserted at run time; OOS never touches optimisation; early stopping and restore genuinely disabled; grad norm recorded pre-clip (`clip_grad.py:230`); pos_emb dominance (98.7% of the attention input at init, ratio 8.7–17×) and post-LN (`norm_first` unset) are MECHANICAL_CODE_PROOF. Two test-file defects: `unittest.main()` precedes `TestLongEpochCurve` in `tests/test_p4_epoch_diagnostic.py` (runs under discovery only) and there is no label-flip invariance test for `epoch_curve_long`.

**A12. Block selection was a zero-degree-of-freedom rule** (first / index 4 / last of nine `linspace` dates asserted at `:313`), so it cannot encode result information even though it was authored after B0 results existed. The realised sub-sample is the least favourable to the hypothesis (epoch-3 OOS IC 0.174 vs 0.086 on the other six; late-epoch mean 0.066 vs 0.071) — sampling misfortune, not selection, but any negative from these three blocks is confounded with regression to the mean.

---

## 2. FALSIFIED OR WITHDRAWN CLAIMS (by whom, on what)

| claim | author | status after Round 2 | decisive evidence |
|---|---|---|---|
| Post-epoch-6 iterate is a stationary "noise ball"; anneal/averaging recovers most lost IC | ML | FALSIFIED as stated (conceded) | A7; 64/36 variance split; ρ₁ 0.73 |
| Rising held-out MSE on 28k samples proves overfitting | ML | FALSIFIED (conceded) | A8 decomposition |
| Effective double descent (113k params vs ≈118 independent blocks) | ML | FALSIFIED (category error) | interpolation is set by training residuals, train R² 1.5% |
| Epoch-3 peak sits inside Adam β₂ warm-up (52% converged) | GPU | FALSIFIED (conceded; MECHANICAL_CODE_PROOF) | torch bias-corrects both moments; 0.516 is the divisor |
| Spec's 6.9 s/epoch is optimistic; expect ≈2.3 h | GPU | FALSIFIED (withdrawn) | LONG log: 661–732 s/fit |
| Rising dispersion = idiosyncratic fit amplification / over-confidence | Auditor, Red Team F13 (partly) | WEAKLY_CONTRADICTED | A8: 30% of target SD at epoch 15 |
| OOS IC decays monotonically past epoch 3 | Auditor | FALSIFIED (conceded) | epochs 9–11 (0.087/0.091/0.096) exceed epoch 8 (0.085) |
| ≈157 refit blocks for SE 0.010 | Auditor | narrowed → 195–200 | agreed window {12,15} |
| PROMISING base rate ≈3.7% | Red Team | withdrawn → 9.5% type-I | full-rule simulation |
| Score r0→r0+64 post hoc "at negligible cost" | Red Team | FALSIFIED as post hoc | saved `_preds.npz` cover only the 5-session block (`:530-532`, `:551`); needs re-training; 64 sessions ≈ 3.2 label events → SE ×1.8, not 10× |
| If only 3 refits, choose 01-05 / 01-26 / 07-01 | Red Team | FALSIFIED as methodology | selects on the realised outcome |
| Compute estimate optimistic (2.0–2.5 h) | Red Team F17 | FALSIFIED | 1.73 h measured |
| Block selection = selection on results (F3/F12 as misconduct) | Red Team | WEAKLY_CONTRADICTED | A12 |
| 242 steps/epoch, 725 steps, 24,160 steps | ML, Red Team, Auditor | corrected (digits only) | A6 |
| v12 late val-IC rise is restore-confounded | Auditor (R1) | withdrawn | per-epoch val_ic logged before restore |
| v12 late val-IC rise is evidence for late OOS recovery | ML | INCONCLUSIVE (Auditor: 0.10) | validation metric, 1.04 SE, and that arm had the worst book in its family (1.326 vs 1.838) |

---

## 3. UNRESOLVED DISAGREEMENTS (dissent preserved)

**D1. Is "epoch ≈3 is the useful regime, later is worse" established?**
- Auditor: YES on paired per-date **validation** rank IC (28,106 samples × 263 dates per fit): ep15 − ep3 = −0.0711, clustered SE 0.0120, t −5.9, negative on 9/9 refits; ep8 − ep3 = −0.0482, t −9.8. Val IC is a poor argmax selector but "right about the shape".
- Red Team and ML: NO on OOS (ep15 − ep3 = −0.056 ± 0.048, t −1.15; no epoch differs from 3 by >1.9 SE; noise floor 0.0103 equals the decision threshold). Red Team adds that the validation block is the instrument the Council already found uninformative for selection (H-VALSTALE) and is 14 months stale.
- Lead reading: the two sides measure different things. The shape claim on validation is strong; the OOS claim is underpowered. Deployment is judged on OOS, so the production-relevant statement remains "not established on OOS, strongly indicated on validation".
- Settles at zero GPU: val rank IC at epochs 30/50/75/100 from the retained LONG predictions across all 9 fits.

**D2. What is the block-conditional structure?**
- Red Team: convergence toward a common late-epoch solution (variance reallocation; 3/9 blocks improve materially).
- ML: block × prediction-aggressiveness interaction, not training dynamics — validation slopes are negative on 9/9 refits and correlate −0.26 with OOS slopes (wrong sign).
- Auditor: not identified; the −0.85 slope-vs-level relation is largely variance compression (cross-refit SD 0.198 → 0.124), equally predicted by "common informative solution" and "decay toward a common uninformative ranking".
- Settles at zero GPU: cross-refit rank correlation of predictions at matched late epochs together with their val IC; and corr(validation-block slope, OOS-block slope) on ≥9 refits.

**D3. Does the late trajectory carry any recoverable value (anneal / averaging)?**
- ML (revised): prediction averaging beats a single late epoch 45%, beats epoch 3 12%; anneal contrast worth testing as a paired schedule estimand.
- Red Team: <5% that the M1 rationale holds; supports the anneal arm only as a mechanism probe.
- GPU: dispersion may asymptote below the target SD (un-shrinking) — if so, late epochs are not over-fit and averaging/anneal has something to work with.
- Settles: `oos_pred_std` and `weight_norm` at 50/75/100 (already recorded); prediction averaging from saved `_preds.npz` (zero GPU).

**D4. Optimizer transient at epoch 3.** β₂ is dead. Live: pos_emb dominance (M3), post-LN without warm-up (M4), and the GPU Reviewer's replacement argument that epoch 3 is a distance-travelled point (max per-coordinate displacement 0.21 at epoch 3 vs 6.96 at epoch 100), so the peak cannot be a content-driven solution. Settles: `grad_norm_mean/max` (clip binding?) and `weight_norm` over epochs 1–15 (already recorded).

**D5. The next GPU experiment.** Four proposals, not reconcilable into one overnight window:

| proposal | GPU-h | code impact | what it decides |
|---|---|---|---|
| Zero-compute set (clustered SE, permutation null for the max, prediction averaging, anchor check with declared tolerance, slope tables) | 0 | research-only, ~40 lines | makes the existing LONG artifacts interpretable |
| GPU Tier 1: 1 refit × 3 seeds × batch 4096 × 100 epochs, LR unchanged | 0.58 | none (`batch` preset key exists) | whether the peak moves in epoch units (steps vs data exposure) |
| ML paired schedule contrast: 9 refits × 3 seeds forked at epoch 50 into constant vs cosine-to-zero over 10 epochs, plus calibration fork at epoch 3; A3 as prediction averaging | ≈3.3 (6.1 with weight-SWA) | needs an LR hook that `fit_one` does not have → research-only monkeypatch or production edit | paired SE ≈0.003–0.006; whether the late iterate has a good centre |
| Red Team LONG′: all 9 refits × 3 seeds × 100 epochs with a widened scoring window | 5.2 | scoring window must be widened before training | slope estimand with SE ≈0.05 |

The Auditor's position, endorsed by the Lead: publish the frozen long fits mechanism-only first (train loss, val rank IC, dispersion, grad/weight norms) and let those trajectories decide whether any of the GPU arms is worth its window.

---

## 4. EVIDENCE QUALITY

- Everything is BURNED_DIAGNOSTIC on 2026-H1 or MECHANICAL_CODE_PROOF. No prospective block exists (labels matured to ≈2026-08-13 were not used).
- The only historical evidence (v12) is a validation metric on a restore-selected arm with the worst book of its family: INCONCLUSIVE.
- The pre-registration is self-asserted: the `P4_B0_LONG` spec section and the launch are not git-verifiable (spec sections were committed with results, and the LONG section is uncommitted). Future preregistrations should be committed before launch.
- Multiplicity: every additional axis screened on 2026-01-05 → 07-23 adds to an interval used ≈111 times.

---

## 5. RECOMMENDED NEXT DISCRIMINATING EXPERIMENT (Lead)

**Before unblinding IC (zero GPU, research-only code, user-gated):**
1. Minute A1 as a preregistered deviation: the frozen `NO_BENEFIT / POSSIBLE_RECOVERY / PROMISING` output will be reported but labelled "structurally biased; not evidence". Add refit-clustered SE, leave-one-refit-out range, and a within-fit permutation null for `oos_late_max` to `evaluate_long`; relabel results `..._AT_BATCH_1024_CONSTANT_LR`; report steps, not epochs; state the honest bar (+0.023).
2. Tier-0 anchor check of epochs 1–15 against the nine B0 cells with a pre-declared tolerance derived from the measured non-determinism band (re-run or `tests/test_seed_reproducibility.py`).
3. Read the mechanism channels first: train loss, val rank IC, `oos_pred_std`, `grad_norm_mean/max`, `weight_norm` at 30/50/75/100. Publish those before any IC number.
4. Then the zero-GPU settlers for D1–D4 (val rank IC late; cross-refit prediction agreement; prediction averaging from saved preds; grad-norm read).
5. Move `TestLongEpochCurve` above `unittest.main()`; add a label-flip test for `epoch_curve_long`.

**GPU (only after 1–4, only if the mechanism read shows a phase change, user-gated):** GPU Tier 1 first (0.58 h, zero production-code change). The paired schedule contrast next, only with a research-only LR hook that is bit-identical when flat. LONG′ (27 fits) only after the scoring-window change is designed and preregistered. Never inside the daily_ops window.

---

## 6. PRODUCTION IMPLICATION

None. No change to `max_epochs`, patience, or seeds is supported. H-EPOCH-LONG moves from TESTING to WEAKLY_CONTRADICTED on mechanism arithmetic pending the mechanism read; "epoch 3 is specifically optimal" is NOT established on OOS and must not be written as such. The H-EARLYSTOP-WEAK finding (fixed epoch 3 ≥ production rule on B0) is unaffected but now carries the +0.0128 winner's-curse haircut. P4-B1 sequencing is unaffected.

---

## 7. THE EIGHT QUESTIONS — Council answers

1. **Plausible?** Weakly, as "epoch 3 is not a converged state" (678–717 steps; 98.7% positional input at init; displacement 0.21 per coordinate). Implausible as "a late-recovery mechanism exists in this recipe" (A5).
2. **Mechanisms?** Live: block-conditional drift (A7), pos_emb / post-LN transient (D4), un-shrinking dispersion (A8). Dead: double descent, grokking, Adam β₂ warm-up, stationary noise ball.
3. **Evidence against?** Val rank IC falls 0.107 → 0.036 with t −5.9 on 9/9 refits (validation, D1); residual-IC ceiling 0.017/−0.001; capacity scaling falsified; v12 book-level dose-response bad (INCONCLUSIVE for this question).
4. **Can the design detect a real recovery?** No (A1–A4). It is a legitimate mechanism probe and an illegitimate decision instrument.
5. **Meaningful evidence for?** A pre-declared single late checkpoint beating epoch 3 by ≥ +0.023 with clustered SE below half the effect on ≥8 non-overlapping blocks including prospective ones, plus a mechanistic correlate (val rank IC turning point, dispersion asymptote, weight-norm plateau), plus disjoint-seed replication.
6. **What falsifies it?** Val rank IC still monotonically falling at 50/75/100 on 9/9 fits; per-fit late max inside its permutation null; no phase change in norms or dispersion.
7. **Batch/steps confound?** Internal: none. External: decisive; epochs ≡ steps at batch 1024 (A6). Report in steps; do not generalise across batch.
8. **Batch before or after?** After. It would destroy the free nine-cell anchor, costs more than the diagnostic it would speed up, attacks only the ~3.3 s training half of a ~6.9 s epoch, and is not VRAM-bound.

---

## 8. Process notes for the Council protocol

- Round 1 produced four independent reports that converged on the design's power and classifier defects while disagreeing on mechanism; Round 2 falsified one headline claim from each of three members (ML noise ball, GPU β₂, Auditor monotone OOS) and one from the Red Team (PROMISING base rate), each conceded on recomputation. That is the intended behaviour.
- Subagent outputs are not persisted by the harness; the Lead must archive reports to disk before relaying. Done for this session.
- Agent Teams are unavailable in this runtime; role files load only at session start. Round 1 used the `Explore` type with role files read first; Round 2 used `SendMessage` to resume the same agents.

---

## Erratum (Council Session 02, 2026-09-21)

**A8 is FALSIFIED as computed.** A8 decomposed validation MSE using `oos_pred_std` as a proxy for validation prediction dispersion. That proxy overstates validation dispersion by 1.7 to 2.1 times. With the true validation prediction SD, available from the P4-B0-LONG retained predictions, the scale-free validation Pearson r on the nine shared fits *falls* from 0.062 at epoch 3 to 0.029 at epoch 15 and 0.021 at epoch 100. It does not rise from 0.066 to 0.122. The Auditor and the ML Researcher reproduced this independently. P4-B0 kept no validation predictions, so the 27-fit value cannot be recomputed directly; the proxy reproduces the reported rise, which identifies it as the source. Consequences:

- The reading "un-shrinking, not over-confident" loses its support. Past epoch 15 the learned signal degrades while in-sample fit rises, which is ordinary overfitting.
- The dispersion figure at epoch 15 should be 17% of the target SD (validation predictions), not 30%.
- The logical point that raw MSE alone is not proof of overfitting still stands.

See `COUNCIL_02_long_training_unblinded_synthesis.md` §2.
