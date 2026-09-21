# AI-Quant Research Council — Session 02 — Lead Synthesis (Round 3)

**Question:** Given the unblinded P4-B0-LONG evidence, are the Lead's conclusions valid, correctly scoped and correctly worded, and what, if anything, comes next?

**Date:** 2026-09-21. **Lead:** main session. **Members:** Quant Red Team, ML Training/Architecture Researcher, Data/Validation Auditor, GPU/Systems Reviewer, now loaded as native project agents from `.claude/agents/`. Round 2 was Lead-relayed. Nothing was voted on.

**Evidence package:** reports 20–22, `p4_b0_long_{anchor,mechanism,oos}_summary.json`, the per-fit CSVs, `research/p4_long_eval.py`, and the Session-01 synthesis. All four members received the same package, fingerprinted by a 15-file SHA-256 manifest. The manifest verified for every member who checked it. After the GPU Reviewer's Round 1 report, the Lead added an erratum and sensitivity sections to report 22 (uncommitted); all members saw those changes in Round 2. Report 21 stayed frozen.

**Evidence grade:** BURNED_DIAGNOSTIC on 2026-H1, plus MECHANICAL_CODE_PROOF for integrity, gating and code claims. Three refits are the independent unit, and one of them is a single session. No PROSPECTIVE evidence.

---

## 1. Agreed findings (all four members after Round 2)

**G1. The unblinding was procedurally valid.** (MECHANICAL FACT; Auditor and Red Team verified independently.)
- Commit and file-time order: the preregistration as launched (`0be4c68`) was committed first, then the amendment (`c6e6a6d`, 10:49). The evaluator followed (file time 10:57, committed `7d5378b` at 11:04). The anchor and mechanism outputs were written at 11:05, the mechanism report was committed (`93fff20`) at 11:07, and the OOS outputs were written at 11:08.
- The OOS stage refuses to run unless the mechanism summary is committed and unchanged.
- All 18 LONG artifacts match the pre-unblinding checksums.
- The 2026-09-15 run log contains no OOS IC.
- **One residual limit:** that the raw result files were not opened between 09-15 and 09-21 rests on the Lead's word, because OOS IC sits in plaintext in the artifacts. No derived file carries an earlier modification time.

**G2. The anchor is exact.** All 135 cell-epochs (epochs 1–15 × 9 fits) reproduce P4-B0 bit-for-bit, including OOS IC, despite no cuDNN determinism flags. (MECHANICAL FACT) The likely reasons are the same torch build, cuDNN and driver, benchmark mode off, seeded Philox and a single stream. Parity is not guaranteed across a nightly or driver update, so versions should be recorded in future manifests.

**G3. REJECTED is the correct output of the frozen rules.** (MECHANICAL FACT)
- The verdict logic implements the frozen text.
- The two discretionary operationalisations added in `7d5378b` do not lie on the REJECTED path; if anything, they favour INCONCLUSIVE.
- Class B would also have met REJECTED, so the verdict does not depend on the A-versus-B boundary. The A margin was only 0.005.

**G4. The label is fragile.** It holds at minimum margins on three near-threshold facts:
- mechanism Q2 was 0.002 and 0.004 below the PARTIAL bar on two refits;
- the 2026-07-23 single-session refit sits at −0.014;
- exactly 7 of 9 fits are negative.

Dropping 2026-01-05 gives WEAKLY_CONTRADICTED. A reseed would turn the mechanism leg to PARTIAL, and the verdict to INCONCLUSIVE, with probability about 12–20% (seed bootstrap from the Auditor and ML Researcher), before counting the OOS legs. The contrast is not significant: t(2) = −1.38, sign test p = 0.125, 95% CI −0.57 to +0.30. (STATISTICAL EVIDENCE)

The Lead's post-hoc seed bootstrap covers the whole verdict. It resamples three seeds with replacement within each refit, 2,000 draws, reapplying the frozen Q2 and verdict rules with the other mechanism channels held at their observed labels. Resampling three seeds understates reseed variance, so these shares are optimistic about stability.

| resampled verdict | share |
|---|---|
| REJECTED | 51% |
| WEAKLY_CONTRADICTED | 39% |
| INCONCLUSIVE | 11% |
| any supportive verdict | 0% |

The mechanism leg flips to WEAK in 11% of draws, and the refits leg reaches 3/3 in 72%. The *direction* is robust. The specific label REJECTED versus WEAKLY_CONTRADICTED is close to a coin flip.

**G5. The direction is credible across instruments.** (STATISTICAL EVIDENCE)
- Late OOS is below the **production-selected epoch** on **3/3 refits and 9/9 fits**: −0.169, SE 0.100. This is the strongest directional contrast, though it is not a frozen verdict input.
- Validation rank IC at epoch 100 is 0.022, against 0.100 at epoch 3. On two of three refits the late value is far below epoch 3.
- True scale-free validation Pearson r falls from 0.062 to 0.029 to 0.021.
- A late maximum over epochs 30–100 is consistent with the AR(1) noise null (58th–63rd percentile). That null was frozen to debias a maximum, not to detect sub-window regimes.

**G6. The mechanism is memorisation.** (STATISTICAL EVIDENCE with INTERPRETATION)
- In-sample R² rises from 1.5% to 18.8% while validation and OOS ranking decline.
- Cross-seed agreement falls, the model drifts away from its epoch-3 ranking (rank correlation 0.13–0.16 by epoch 100), and dispersion keeps expanding.
- No phase change. The transient validation hump at epochs 31–40 is real on all three refits but is not sustained, did not transfer to OOS (epochs 20 and 30: 0.001 and 0.011), and is data-selected.

**G7. Step count is not a live hypothesis. Do not run the batch-4096 test.** (INTERPRETATION, unanimous) More steps are what hurt, and no decision depends on locating the peak in steps versus epochs.

**G8. The `inf` gradient norms are GradScaler-skipped overflow steps: at least 71 steps, 0.034% of all steps.** (MECHANICAL FACT)
- They are a mix of scale-growth probes and gradient spikes. At least 20 of 62 gaps are shorter than the 2,000 clean steps a probe needs.
- They are harmless. Train loss and weight norm change no differently in those epochs, and they are not time-linked to the validation hump.
- The frozen report 21 wrongly described them as probes spaced 2,000–2,900 steps apart. An erratum is in report 22.
- Late clipping acts on tail steps only: the late mean gradient norm is 0.41, under the 1.0 clip. It limits spikes rather than creating a normalised-gradient regime.

**G9. The Session-01 blind predictions about the original classifier were wrong, and the miss was avoidable.** Members predicted POSSIBLE_RECOVERY at about 80–90%. NO_BENEFIT, put at under 5%, fired. The structural analysis was right: the maximum over 71 epochs is biased by about +0.11, as the observed null mean of 0.110 confirms. But epoch 15 (0.036) and the early regime (0.150) for these nine fits were already known from P4-B0, and the Red Team's own Session-01 derivation said NO_BENEFIT needed a late mean below about 0.025. Conditioned on those facts, NO_BENEFIT was roughly a coin flip. **Provenance:** the probability figures come from uncommitted round transcripts. They are recorded here as *attributed; uncommitted source*.

---

## 2. Falsified or withdrawn claims

| claim | from | status | decisive evidence |
|---|---|---|---|
| **Session-01 A8:** scale-free validation r *rises* from epoch 3 to 15, so the rising raw MSE is "un-shrinking, not over-confidence" | Session 01 (agreed) | **FALSIFIED as computed.** Erratum appended to the Session-01 synthesis | The MSE decomposition used `oos_pred_std` as a proxy for validation dispersion, which overstates it 1.7–2.1×. With true validation dispersion, r falls 0.062 → 0.029 (Auditor and ML Researcher reproduced this exactly). A 27-fit check is impossible because P4-B0 kept no validation predictions. |
| `inf` gradient norms are growth probes spaced 2,000–2,900 steps | Lead, report 21 | FALSIFIED on spacing; harmlessness stands | GPU Reviewer and Auditor: 20/62 gaps are under 2,000 steps |
| First clip events and first overflows coincide at epochs 16–24 on 9/9 fits | ML | FALSIFIED as an independent observation | It counted `inf` as clip events. Finite clipping already occurred at epoch 1 on 7/9 fits |
| A reseed gives INCONCLUSIVE with 20–30% probability | Red Team | Narrowed to about 12–20% | Seed bootstrap from Auditor and ML Researcher |
| Paired noise of the borderline fits is about 0.08 | Red Team | Narrowed to about 0.04–0.07 (seed); 0.08 applies only to session resampling | GPU Reviewer |
| Late regime is normalised-gradient AdamW | question put to GPU | Rejected | Late mean norm 0.41 < 1.0 |
| Headline "NOT SUPPORTED" | Red Team, R1 | Withdrawn by the Red Team | Relabelling after unblinding would be a post-hoc deviation |
| "Late maximum fully explained by noise" | Lead, report 22 | Softened to "consistent with a noise null" | Red Team and Auditor |
| The 2026-01-05 contrast measures a training-length effect | implicit in report 22 | Narrowed | See D1 |

---

## 3. Unresolved disagreements (dissent preserved)

**D1. What the OOS magnitude measures.**
- **ML Researcher and Red Team:** the size of the OOS loss is mostly factor payoff. Epoch 3 is about 80% momentum-exposed (rank exposure to 12-1 momentum 0.81/0.78/0.90 by refit). Long training strips that exposure to 0.03–0.31. Momentum alone scored OOS IC 0.329, 0.264 and −0.016 on the three blocks. So the 2026-01-05 collapse is what de-exposure costs when momentum pays, and the sign of the long-training penalty on OOS depends on the factor premium, which three blocks cannot establish.
- **Auditor:** the frozen label stays valid either way, and the production-selected comparator (9/9 fits) keeps the direction robust.
- **Lead:** both hold. The mechanism "long training de-exposes the model from the factor it learned first" is well supported. How much that costs OOS is block-dependent.
- **Settles it:** 12-1 momentum and 60-day volatility OOS IC across all nine P4-B0 blocks plus the prospective window. This is zero GPU.

**D2. Is the binding limit DATA/TARGET?**
- **ML Researcher:** yes (0.65). The epoch-3 residual validation IC beyond the ten last-step features averages 0.001.
- **Red Team:** INCONCLUSIVE. "The net matches the best single feature" fits both "the data holds nothing more" and "this recipe extracts nothing more", and the tying feature was chosen after the fact from ten.
- **Auditor:** it is an INTERPRETATION, not a finding. No data or target alternative was tested.
- **Settles it (Red Team's corrected design):** the net's paired incremental IC over a benchmark fitted only on training rows, clustered by refit, with an **equivalence** decision rule. Call it DATA/TARGET-limited only if the upper 95% bound is below 0.01. Call it non-factor signal only if the lower bound is above 0. Otherwise INCONCLUSIVE. The ML Researcher's original "residual < 0.01 or CI includes 0" rule was a default-outcome rule of the same kind as NO_BENEFIT.

**D3. Averaging: close or deprioritise?**
- **ML Researcher and Auditor:** closed as a late-regime rescue. Prediction averaging over epochs 50–100 recovers essentially nothing: +0.0015 over the mean single epoch with per-date z-scored means, +0.0004 with raw means; both constructions are stated because the exact figures depend on them. The average also fails the fair early bar (epochs 2–5: 0.102 vs about 0.049 for the 30–100 average).
- **Red Team:** deprioritised, not closed. Weight-space SWA was never measured, the evidence is post-hoc on three hostile refits, and early-window averaging is untouched.
- **Lead:** record H-WEIGHT-AVERAGING (late window) as WEAKLY_CONTRADICTED and deprioritised, with weight-SWA untested and a low prior.

**D4. Is an early anneal worth GPU time?**
- **GPU Reviewer:** yes, if gated. Anneal to zero by about 1,150 steps on all 27 P4-B0 fits, about 0.26 GPU-h, read on validation only, preceded by a λ ≡ 1 bit-parity smoke test.
- **ML Researcher:** not yet. Epoch-3 residual IC beyond the factors is about 0, so sharpening epoch 3 would mostly sharpen momentum. A moving peak also adds about +0.02 of max-selection bias unless the readout step is pre-declared.
- **Red Team:** no GPU until the equivalence ledger (D2) runs.
- **Auditor:** any arm on this interval is a burned mechanism probe and cannot be promotable evidence.
- **Lead:** gate it on D2. If D2 returns non-factor signal, the early anneal becomes a fair, cheap, validation-read probe with a pre-declared readout step.

**D5. Optimisation "not binding".**
- **ML Researcher:** OPTIMIZATION not binding.
- **GPU Reviewer:** too broad. The optimiser can fit, but the recipe's implicit regularisation (constant LR, inert weight decay) is untested, and the early anneal is exactly that test.
- **Lead:** adopt the GPU wording. Optimisation *capability* is not binding; the recipe's implicit regularisation is untested.

---

## 4. H-EPOCH-LONG verdict (recorded wording)

> **H-EPOCH-LONG (frozen recipe: batch 1024, constant LR 3e-4 with gradient clipping, ≤ 23.9k steps): REJECTED under the preregistered rule, met at its minimum margins** (3/3 refits with 2026-07-23 at −0.014 on one session; 7/9 fits; mechanism Q2 0.002–0.004 below PARTIAL). Directional, not significant (L − E2–5 = −0.139, t(2) = −1.38, 95% CI −0.57 to +0.30); burned interval, n = 3 refits; about 81% of the magnitude comes from 2026-01-05. In plain terms, more steps under this recipe do not recover early-epoch ranking: late OOS is below the production-selected epoch on 9/9 fits, and validation shows the same. Mechanism: memorisation, in which long training strips the momentum and volatility exposure that makes up the early model's signal. This says nothing about LR schedules, weight averaging or regularisation.

The frozen label token stays REJECTED. All four members agreed that replacing it after unblinding would repeat the failure the amendment was written to prevent. The plain-language gloss follows the label.

**Original frozen classifier:** LONG_EPOCH_NO_BENEFIT. ORIGINAL_PREREGISTERED_CLASSIFIER / STRUCTURALLY_FLAWED / NOT_PRIMARY_EVIDENCE. It agrees with the verdict in direction, and it fired only because the late mean (0.004) fell below the level the bias required.

**Production implication:** none. No change to epochs, patience, seeds or recipe.

---

## 5. Next hypotheses (none launched)

1. **Zero GPU, first: H-RESIDUAL-SIGNAL, the equivalence ledger (D2).** It asks whether the deployed transformer carries signal beyond the ten features it is built from. Use the existing production-selected fit predictions (P0/P4 caches, 9+ refits) rather than the P4-B0 curves, which kept no predictions. Preregister the equivalence rule before computing. Report 11's existing residual-IC figures (0.017 / −0.001) point toward "little".
2. **Zero GPU: H-FACTOR-PREMIUM (D1).** Factor OOS IC across all nine P4-B0 blocks and the prospective window. This decides whether de-exposure is a cost or a hedge.
3. **Conditional GPU (about 0.26 GPU-h): H-LR-SCHEDULE, early variant.** Anneal to zero by about 1,150 steps on the 27 P4-B0 fits, read on validation, with a pre-declared readout step and a λ ≡ 1 parity test. Run only if (1) finds non-factor signal. The late-fork anneal is not motivated.
4. **Deprioritised: H-WEIGHT-AVERAGING (late window).** Prediction-space averaging recovers essentially nothing; weight-SWA is untested and has a low prior.
5. **Not justified:** the batch-4096 step test, and 200/500/1000-epoch runs.

---

## 6. Process lessons

- **Condition blind predictions on what is already known.** The Session-01 miss came from reasoning about a rule unconditionally while the comparator values sat in P4-B0. Simulate structural critiques on the actual known inputs, and state predictions as intervals.
- **Check mechanism stories against logged timing before calling them mechanical facts.** Both the Session-01 β₂ story and the Lead's Session-02 probe story failed this test.
- **Verify proxies before building an agreed finding on them.** Session-01 A8 used OOS-input dispersion for validation dispersion. Future harnesses should log validation prediction SD directly, which P4-B0-LONG's retained predictions made possible.
- **Default-outcome decision rules recur.** NO_BENEFIT, and the first draft of the residual ledger, would both have returned their default under low power. Use equivalence bounds for "no effect" claims.
- **Blinding by attestation is a weak point.** Future runs should write outcome channels to a separate file, or encrypt them, so that blinding is mechanical.
- **Native project agents now load.** Session 02 used `.claude/agents/` types directly, and Round 2 via `SendMessage` preserved each member's context.
