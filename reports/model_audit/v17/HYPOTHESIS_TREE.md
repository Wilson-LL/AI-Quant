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

## H-RESIDUAL-SIGNAL — the deployed transformer carries ranking signal beyond its ten input features (DATA/TARGET vs model limit)
- **Status:** UNTESTED as preregistered. Hints: report 11 residual IC 0.017 / −0.001 beyond momentum; Session 02 post-hoc epoch-3 residual validation IC beyond the ten last-step features ≈ 0.001 on three refits (ML Researcher); epoch-3 rank exposure to 12-1 momentum 0.77–0.90.
- **Next (0 GPU):** paired per-date incremental IC of the production-selected fits (P0/P4 prediction caches, ≥ 9 refits) over a benchmark fitted only on training rows (ridge on the ten features; 12-1 momentum alone), clustered by refit. Equivalence rule, preregistered before computing: DATA/TARGET-limited only if the upper 95% bound < 0.01; non-factor signal only if the lower bound > 0; otherwise INCONCLUSIVE.
- **Burned:** would use burned blocks (mechanism probe only). **Updated:** 2026-09-21.

## H-FACTOR-PREMIUM — the factor the early model learns (12-1 momentum / 60-day vol) pays positively out of sample
- **Status:** UNTESTED. It decides whether long-training de-exposure is a cost or a hedge. Block values so far: 12-1 momentum OOS IC 0.329 / 0.264 / −0.016 on the three P4-B0-LONG blocks.
- **Next (0 GPU):** factor OOS IC across all nine P4-B0 blocks plus the prospective window from 2026-07-24. **Updated:** 2026-09-21.

## H-LR-SCHEDULE — an LR schedule changes what the model learns in the early regime
- **Status:** UNTESTED. The late-fork variant (anneal from epoch 50) is NOT motivated: it would start from a memorising iterate. Only an early variant has a rationale.
- **Next (conditional, ~0.26 GPU-h):** anneal to zero by ~1,150 steps on the 27 P4-B0 fits, read on validation, with a pre-declared readout step and a λ ≡ 1 bit-parity smoke test. Run only if H-RESIDUAL-SIGNAL finds non-factor signal. Not launched. **Burned:** YES. **Updated:** 2026-09-21.

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
