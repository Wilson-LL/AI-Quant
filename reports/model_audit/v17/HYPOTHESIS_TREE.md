# AI-Quant Hypothesis Tree (v17) — maintained by the Research Council Lead

Statuses: UNTESTED / TESTING / SUPPORTED / WEAKLY_SUPPORTED / INCONCLUSIVE / WEAKLY_CONTRADICTED / REJECTED.
Evidence types: MECHANICAL_CODE_PROOF / PAIRED_EXPERIMENT / HISTORICAL_OOS / BURNED_DIAGNOSTIC / PROSPECTIVE / THEORETICAL_ONLY.
Failed hypotheses are never deleted. Every entry says whether its evidence interval is burned.

---

## H-EPOCH-LONG — long training (100+ epochs) may produce late OOS recovery / improvement
- **Status:** **REJECTED (frozen recipe only), met at minimum margins.** P4-B0-LONG, unblinded 2026-09-21 under the preregistered amendment `P4_B0_LONG_AMENDMENT_1` (`c6e6a6d`); mechanism frozen before OOS (`93fff20`); reports 20–22; Council Sessions 01 (blind) and 02 (post-unblind).
- **Scope:** batch 1024, constant LR 3e-4 with gradient clipping, AdamW wd 1e-4 (inert), no schedule, ≤ 23.9k optimizer steps. Says nothing about LR schedules, weight averaging or regularisation.
- **Supporting evidence (for the hypothesis):** none. Validation rank IC shows a transient, non-sustained hump at epochs 31–40 on all three refits (0.049/0.067/0.072) that did not transfer to OOS (epochs 20/30: 0.001/0.011). It is data-selected and cannot select an epoch.
- **Contradicting evidence:** BURNED_DIAGNOSTIC. Anchor bit-exact vs P4-B0 (epochs 1–15). Mechanism MECHANISM_NO_PHASE_CHANGE: in-sample R² 1.5% → 18.8%, true val Pearson r 0.062 → 0.021, val rank IC 0.100 (ep 3) → 0.022 (ep 100), cross-seed agreement falls, dispersion still expanding. OOS trajectory class A: epoch 3 0.174 → late mean (50/75/100) 0.011. L − early regime (epochs 2–5) −0.139, SE 0.101, 3/3 refits, 7/9 fits. L − production-selected epoch −0.169, SE 0.100, 3/3 refits, 9/9 fits. Late maxima are consistent with an AR(1) noise null. Prediction averaging over epochs 50–100 recovers ≤ 0.0015 (post-hoc).
- **Confounders / fragility:** not significant (t(2) = −1.38, 95% CI −0.57 to +0.30). The label rests on three near-threshold facts (Q2 0.002/0.004 below PARTIAL; 2026-07-23 single session at −0.014; 7/9 fits exactly). A reseed flips the mechanism leg with probability ~12–20%; a within-refit seed bootstrap of the full verdict gives REJECTED 51%, WEAKLY_CONTRADICTED 39%, INCONCLUSIVE 11%, supportive 0% (direction robust, label near coin-flip). Dropping 2026-01-05 gives WEAKLY_CONTRADICTED; that block supplies ~81% of the magnitude, and it is where momentum alone paid 0.33. The OOS cost is mostly factor de-exposure (12-1 momentum exposure ~0.8 → 0.03–0.31) priced by each block's factor payoff. Hostile subsample (epoch-3 OOS 0.174 vs 0.086 on the other six P4-B0 refits). Burned interval; blinding was by attestation.
- **Original frozen classifier:** LONG_EPOCH_NO_BENEFIT (ORIGINAL_PREREGISTERED_CLASSIFIER / STRUCTURALLY_FLAWED / NOT_PRIMARY_EVIDENCE); agrees in direction.
- **Next:** closed for the frozen recipe. Do NOT test 200/500/1000 epochs or batch 4096. Follow-ups are separate hypotheses: H-RESIDUAL-SIGNAL, H-FACTOR-PREMIUM, H-LR-SCHEDULE (early), H-WEIGHT-AVERAGING.
- **Interval burned:** YES (2026-H1). **Last updated:** 2026-09-21 (Council Session 02).

## H-RESIDUAL-SIGNAL — the deployed transformer carries ranking signal beyond its ten input features
- **Status:** **RESIDUAL_SIGNAL_WEAK (frozen, preregistered `5012d5b`, arm B, 2026-H1).** Operative reading (Council Session 03, unanimous): **no residual signal demonstrated beyond the network's own inputs.**
- **Evidence:**
  - BURNED_DIAGNOSTIC: residual IC beyond the 10 inputs +0.056 (HAC CI −0.008, +0.120; 67% of blocks positive); 69% of the Transformer's IC is carried by the linear input span.
  - Arms C and A reach PRESENT, but they share seeds, dates and labels with arm B (block correlation 0.80–0.92), and arm C falls to WEAK under df = 5.
  - HISTORICAL_OOS, post-peek (`SCHED_P0_A_BR_7s`, 2021–25, 61 windows): residual +0.014 (CI −0.012, +0.040); Transformer IC +0.021; 12-1 momentum IC +0.041.
  - Post-hoc: beyond raw-scale inputs plus 12-1 momentum, the residual is about zero (arm B −0.003, arm C +0.012, history +0.004).
  - The residual co-moves with momentum payoff (r 0.62–0.74). In 2026-H1 it concentrates in rows whose input windows splice across cache holes (see H-DATA-INTEGRITY).
- **Confounders:** momentum-extreme burned window; benchmark features built on the calendar while the network builds them on each stock's own rows; df convention; regime-variable placebo floor; calibration test run on Gaussian inputs.
- **Next:** H-RESIDUAL-HIST (zero GPU, preregistered, post-peek). Use features built the way the network sees them, raw-scale and 12-1 benchmarks, clean-name and spliced-window splits, one df convention, and the residual regressed on momentum payoff.
- **Burned:** YES (2026-H1); history is post-peek. **Updated:** 2026-09-21 (Council Session 03).

## H-FACTOR-PREMIUM — the factor the early model learns pays positively out of sample
- **Status:** **FACTOR_PREMIUM_REGIME_DEPENDENT (frozen)** for mom_126_5, 12-1 momentum and vol_60.
- **2026-H1:** all three pay strongly Jan–Apr (+0.17 to +0.30) and turn negative Jun–Jul. Prospective slice (15 dates) +0.22 to +0.26.
- **2016–2025 context:** yearly mom_126_5 IC −0.056 to +0.075.
- **Split sensitivity:** for 12-1 momentum and vol_60 the label depends on the split date (a 40th-percentile split gives WEAK); mom_126_5 is robust. No decision changes.
- **Attribution:** FACTOR_DEEXPOSURE_EXPLAINS_MOST (frozen). Met on only 2 qualifying refits, and close to an accounting identity. The P4-B0 epoch 3→15 gap tracks momentum payoff (r 0.82; permutation p optimistic).
- **Next:** long-history factor premium 2016–2026 with one df convention (zero GPU). **Updated:** 2026-09-21.

## H-DATA-INTEGRITY — EOD cache holes splice model inputs and labels
- **Status:** **SUPPORTED as a defect** (MECHANICAL_CODE_PROOF; Council Session 03).
- **Extent:** 70 of 110 cached stocks have missing sessions since 2016; 59 have runs of 5 or more; 24 have holes in 2025–26, often whole calendar months (2317 and 9910 all of Sep 2025; 2376 all of Jan 2026).
- **Cause:**
  - In `research/refresh_data.py`, a failed or empty month is skipped. A later successful month advances the cached last date, and the retry pass recomputes the needed months from that date, so the hole becomes permanent.
  - `research/pipeline_gate.py` checks only coverage of the newest date.
- **Effect:**
  - `dataset_transformer_eod.py` and `inference_transformer_eod.py` build features, windows and labels on each stock's own rows, so they splice across holes.
  - About 3% of 60-row windows, 6–9% of mom_126_5 lookbacks and about 1% of labels are affected.
  - There is no look-ahead: maturity is tracked on the calendar.
  - The live model is fed spliced inputs.
- **Next (user decision; production-relevant):** add a history gap gate, fix the refresh path, re-fetch the missing months, freeze a cache hash, then rescore or re-document. **Updated:** 2026-09-21.

## H-LR-SCHEDULE — an LR schedule changes what the model learns in the early regime
- **Status:** UNTESTED. **NOT JUSTIFIED** (Council Session 03, unanimous): every member's non-factor-residual gate fails on every benchmark and sample. The late-fork variant (epoch 50) is ruled out.
- **Conditional design (recorded, not launched):**
  - Gate: H-RESIDUAL-HIST clean-name historical residual ≥ 0.02, lower CI > 0, and a positive momentum-payoff intercept.
  - Data: a repaired cache.
  - Arms: batch 1024 fixed; control at constant LR; treatment annealed to 0 between about 725 and 1,150 steps.
  - Readout: one, pre-declared, at the end of epoch 5.
  - Sample: primary the 2021–25 walk-forward; secondary the P4-B0 refits, read on validation only.
  - Guards: raw IC must not fall; correlation with mom_126_5 must rise by no more than 0.05.
  - Parity: a research-only fork of `fit_one`, bit-identical at λ ≡ 1.
  - Cost: about 40–70 GPU-minutes.
- **Burned:** YES. **Updated:** 2026-09-21.

## H-WEIGHT-AVERAGING — SWA / checkpoint averaging recovers signal from the late regime
- **Status:** WEAKLY_CONTRADICTED (late window, prediction space, post-hoc BURNED_DIAGNOSTIC). Averaging epochs 50–100 recovers ≤ 0.0015 over the mean single epoch; the 30–100 average (~0.049) fails the early bar (0.102). Weight-space SWA is untested, with a low prior. Early-window averaging is untested.
- **Dissent:** Red Team wants DEPRIORITISED, not closed; ML Researcher and Auditor call it closed as a late-regime rescue.
- **Next:** none now. **Updated:** 2026-09-21.

## H-EARLYSTOP-WEAK — validation-IC early stopping adds little over an ex-ante fixed epoch
- **Status:** WEAKLY_SUPPORTED (BURNED_DIAGNOSTIC, P4-B0: production rule 0.107 vs fixed-3 0.115 OOS IC; wins 30% of fits; within-refit val→OOS Spearman median +0.24). Classification EARLY_STOPPING_WEAK.
- **Confounders:** 9 refits; val block 14 months stale; 15-epoch horizon.
- **Next:** P4-B1 (current early stopping vs fixed 3, 27 refits × 3 seeds) — not launched.
- **Note (2026-09-21):** the fixed-epoch-3 comparator carries a +0.0128 (P4-B0) / +0.023 (P4-B0-LONG refits) winner's-curse premium over the mean of epochs 2–5.
- **Interval burned:** YES. **Last updated:** 2026-09-15.

## H-VALSTALE — the 263-session, 14-month-stale validation block carries little selection signal
- **Status:** WEAKLY_SUPPORTED (val→OOS uninformative after detrending; see `18_validation_policy_diagnosis.md`). Original "anti-predictive" reading FALSIFIED (time-trend confound).
- **Next:** P4-B reduced screen (D_fixed3 + B_recent126). **Burned:** YES. **Updated:** 2026-09-15.

## H-CADENCE — daily full refit degrades ranking vs the validated 126-session cadence
- **Status:** WEAKLY_SUPPORTED at the model level / REJECTED at the book level (PAIRED_EXPERIMENT P0-A: ΔIC −0.0126 ± 0.0077; band10 absorbs; PARITY_WARNING; density effect INCONCLUSIVE).
- **Next:** P6 (weekly/monthly cadence) only after the validation policy is settled. **Burned:** YES. **Updated:** 2026-09-14.

## H-40D — a 40-session target improves rank quality (IC rises 20d→40d; book holds for months)
- **Status:** UNTESTED (holding the 20d model longer REJECTED as a proxy). **Next:** P3 after P1/P4. **Burned:** would use CH/BR windows (burned). **Updated:** 2026-09-14.

## H-CORPACT — unadjusted corporate actions materially bias labels/features
- **Status:** WEAKLY_SUPPORTED (candidate-detector estimate, MATERIAL qualitative). **Next:** P1 data acquisition (0 GPU). **Updated:** 2026-09-14.

## H-SURVIVOR — survivor universe inflates all historical references
- **Status:** SUPPORTED as a bias (SURVIVORSHIP_BIAS_PRESENT); magnitude UNKNOWN. **Next:** P2 point-in-time data. **Updated:** 2026-09-14.

## H-MOMENTUM-EQUIV — the neural model adds little independent IC over 126/5 momentum
- **Status:** WEAKLY_SUPPORTED (HISTORICAL_OOS on burned windows: corr(tf, mom) 0.65; residual IC 0.02 CH / 0.00 BR; blend +0.28/+0.09 Sharpe within 1 SE; diversification/drawdown benefit real). **Next:** any challenger must be judged on residual information, not raw IC. **Updated:** 2026-09-14.

## H-SCALE — larger hidden/depth/FFN, longer lookback, richer features improve OOS
- **Status:** REJECTED (v11/v12/v13, HISTORICAL_OOS at book level; XL collapse). Not to be reopened without a new mechanism-level hypothesis. **Updated:** 2026-08-03.

## H-PROSPECTIVE — the live model's ranking is intact prospectively
- **Status:** INCONCLUSIVE (PROSPECTIVE, ≈1 independent 20-session block since 2026-07-24; 5d HAC CI includes 0). **Next:** accumulate ≥3 blocks. **Updated:** 2026-09-14.
