# AI-Quant Hypothesis Tree (v17) — maintained by the Research Council Lead

Statuses: UNTESTED / TESTING / SUPPORTED / WEAKLY_SUPPORTED / INCONCLUSIVE / WEAKLY_CONTRADICTED / REJECTED.
Evidence types: MECHANICAL_CODE_PROOF / PAIRED_EXPERIMENT / HISTORICAL_OOS / BURNED_DIAGNOSTIC / PROSPECTIVE / THEORETICAL_ONLY.
Failed hypotheses are never deleted. Every entry says whether its evidence interval is burned.

---

## H-EPOCH-LONG — long training (100+ epochs) may produce late OOS recovery / improvement
- **Status:** TESTING → WEAKLY_CONTRADICTED on mechanism arithmetic (Council Session 01, 2026-09-15, blind to the in-flight results). P4-B0-LONG launched 2026-09-15 09:25 under prior approval; its OOS results remain embargoed and unevaluated; the Council found its frozen classifier structurally unable to decide (see `COUNCIL_01_long_training_synthesis.md` §1 A1).
- **Claim:** OOS rank IC, which peaks at epoch ≈3 (678–717 optimizer steps at batch 1024) and decays through epoch 15, may recover or exceed the early peak at 50–100 epochs (22.6k–23.9k steps).
- **Supporting evidence:** THEORETICAL_ONLY. Nobody has measured epoch-N weights at N > 15 with best-validation restore disabled (v12 epoch arms kept patience + restore, deployed ≈ epoch 13). Epoch 3 is not a converged state (98.7% positional attention input at init; per-coordinate displacement 0.21 at epoch 3). B0's OOS decline is not statistically established (ep15 − ep3 = −0.056 ± 0.048, t −1.15). Late-epoch behaviour is block-conditional and seed-reproducible (slope corr +0.886; 3/9 refits improve).
- **Contradicting evidence:** MECHANICAL_CODE_PROOF: decoupled wd inert (0.070% shrink over the run), no LR schedule/warm-up, no interpolation (train R² ≈1.5% at ep15, ≈10% at ep100) → double descent and grokking unavailable. BURNED_DIAGNOSTIC P4-B0: per-date validation rank IC ep15 − ep3 = −0.0711 (clustered SE 0.0120, t −5.9, 9/9 refits) — strong on validation, underpowered on OOS (dissent preserved: D1). Raw val MSE rise is a dispersion artefact (rescaled MSE improves), so it is NOT evidence. HISTORICAL v12: INCONCLUSIVE (validation metric; worst book in its family).
- **Confounders:** epochs ≡ steps at fixed batch (report in steps; label BATCH_1024_CONSTANT_LR); epoch-3 comparator carries a +0.0128 winner's-curse premium (honest bar ≈ +0.023); n_eff = 3 refit blocks, one a single session; refit-clustered SE 0.081 at n=3 vs ±0.010 bar (≈195–200 clusters needed); `NO_BENEFIT` biased shut by a max over 71 epochs (+0.112); `PROMISING` ≈9.5% type-I at ≈0% power; the three LONG refits are the most hypothesis-hostile sub-sample (ep3 OOS IC 0.174 vs 0.086); 2026-H1 burned; no schedule arm; block-conditional heterogeneity breaks any "all refits agree" gate.
- **Next discriminating experiment:** (0 GPU, before unblinding IC) mechanism-first read of the retained LONG artifacts — train loss, val rank IC, `oos_pred_std`, `grad_norm`, `weight_norm` at 30/50/75/100; anchor check of epochs 1–15 vs the nine B0 cells with a declared tolerance; clustered SE + permutation null for the max added to `evaluate_long`; prediction averaging from saved preds. GPU (user-gated, only if a phase change appears): Tier 1 batch-4096 step test (0.58 h, no code change) → paired schedule contrast at epoch 50 (≈3.3 h, needs a research-only LR hook) → LONG′ on 9 refits with a widened scoring window (5.2 h).
- **Interval burned:** YES (2026-H1).
- **Last updated:** 2026-09-15 (Council Session 01).

## H-EARLYSTOP-WEAK — validation-IC early stopping adds little over an ex-ante fixed epoch
- **Status:** WEAKLY_SUPPORTED (BURNED_DIAGNOSTIC, P4-B0: production rule 0.107 vs fixed-3 0.115 OOS IC; wins 30% of fits; within-refit val→OOS Spearman median +0.24). Classification EARLY_STOPPING_WEAK.
- **Confounders:** 9 refits; val block 14 months stale; 15-epoch horizon.
- **Next:** P4-B1 (current early stopping vs fixed 3, 27 refits × 3 seeds) — not launched.
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
