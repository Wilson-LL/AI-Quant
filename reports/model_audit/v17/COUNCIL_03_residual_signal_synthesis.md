# AI-Quant Research Council — Session 03 — Lead Synthesis (Round 3)

**Question:** What unique signal does the production-like neural model add beyond simple momentum, volatility and input-factor exposure?

**Date:** 2026-09-21. **Members:** Quant Red Team, ML Training/Architecture Researcher, Data/Validation Auditor, GPU/Systems Reviewer (native project agents). Round 2 was Lead-relayed. Nothing was voted on.

**Evidence package:**
- 19 files fingerprinted by SHA-256: the preregistration (`23_…`, spec, commit `5012d5b`), the results (`24_…`, two JSONs, the `h_rf/` tables), code and tests, and Session-02 context.
- Every member verified the manifest.
- **Changes after Round 1:** the Lead corrected one sentence in report 24 (the like-for-like 12-1 absorption is 0.056 → 0.038). The Lead also verified two post-hoc findings read-only: the 2021–25 historical panel and the cache holes. These were given to all members in Round 2.

**Evidence grade:**
- Frozen results: BURNED MECHANISM DIAGNOSTIC (2026-H1, about 6 independent 21-session label windows).
- Historical panel `SCHED_P0_A_BR_7s`, 2021–25: HISTORICAL_OOS, **post-peek** and selection-burned.
- Every figure below marked *post-hoc* is exploratory.

---

## 1. The frozen labels stand; the operative reading is stricter

| hypothesis | frozen label | operative reading, agreed by all four members |
|---|---|---|
| H-RESIDUAL-SIGNAL | **RESIDUAL_SIGNAL_WEAK** (arm B) | **No residual signal is demonstrated beyond the network's own inputs.** WEAK is generous. |
| H-FACTOR-PREMIUM | **FACTOR_PREMIUM_REGIME_DEPENDENT** (all three factors) | Robust as a label for `mom_126_5`. For 12-1 momentum and `vol_60` it depends on the split date: a 40th-percentile split gives WEAK. No decision changes. |
| Long-training attribution | **FACTOR_DEEXPOSURE_EXPLAINS_MOST** | Met on only 2 qualifying refits, and close to an accounting identity (ΔIC ≈ Δexposure × payoff). It shows that early models bet on momentum. It does not show a causal training effect. |

No frozen label is replaced. The gloss follows each label.

## 2. Agreed findings

**F1. The mechanics are clean.** (MECHANICAL FACT; the Auditor and Red Team verified independently.)
- The spec was committed at 12:07:38, before the first result at 12:08:26, and the code is byte-identical to the committed version.
- Features at date t use closes up to t only.
- The ridge trains on matured rows only; on all 27 blocks the last training date + 21 sessions equals block start − 1 exactly.
- Residualisation never uses forward returns, and the additive split is exact.
- Every number reproduces, with a maximum difference of 8e-17.

**F2. The secondary arms' PRESENT labels are not corroboration.** (STATISTICAL EVIDENCE)
- Arms C and A share dates, labels and recipe with arm B. Their block residuals correlate with arm B at 0.80–0.92, and C − B is only +0.014 (SE 0.0075).
- The residual stage used t(26). The factor stage used df = 5 on the same overlapping dates. Under df = 5, arm C falls to WEAK and arm A stays PRESENT only at +0.009.
- The label also depends on the HAC lag: it is PRESENT at lag 0–1 and WEAK at lag 2 and above.
- Both conventions were preregistered, so nothing was violated. The mismatch is a design defect to fix next time.

**F3. Outside 2026-H1 the residual is small and not significant.** (HISTORICAL_OOS, post-peek; Lead reproduced.) On `SCHED_P0_A_BR_7s`, 2021–25 (1,214 dates, 61 non-overlapping 20-session blocks):

| quantity | value | 95% CI |
|---|---|---|
| Transformer IC | +0.021 | −0.025, +0.067 |
| 12-1 momentum IC | +0.041 | |
| `mom_126_5` IC | +0.026 | |
| **Residual beyond the 10 inputs** | **+0.014** | **−0.012, +0.040** |
| Residual beyond 10 inputs + 12-1 | +0.010 | |
| Placebo | +0.004 | |
| R² of the score on the inputs | 0.78 | |

- Residual by year: 2021 +0.028, 2022 −0.043, 2023 +0.006, 2024 +0.065, 2025 +0.011.
- Transformer IC by year: 2021 −0.009, 2022 −0.047, then 2023–25 about +0.05.
- 2026-H1 (0.056–0.087 depending on the arm) was the model's best stretch, inside a momentum regime roughly 3–10 times a normal year.

**F4. The residual is momentum-conditional.** Block residual IC correlates 0.62–0.74 with momentum payoff (STATISTICAL EVIDENCE). By month on arm B:

| Jan | Feb | Mar | Apr | May | Jun | Jul |
|---|---|---|---|---|---|---|
| +0.18 | +0.05 | +0.08 | +0.03 | +0.16 | −0.06 | −0.05 |

Regressed on momentum payoff, the implied residual at zero momentum payoff is +0.018 on arm B.

**F5. Most of the residual is the network's use of raw feature scale plus longer momentum.** (STATISTICAL EVIDENCE, post-hoc; the ML Researcher confirmed the Red Team's hypothesis.) These use features built the way the network sees them:

| sample | residual beyond rank inputs | + raw-scale inputs | + raw-scale inputs + 12-1 momentum |
|---|---|---|---|
| arm B | 0.070 | 0.029 | **−0.003** |
| arm C | 0.082 | 0.042 | **+0.012** |
| 2021–25 | 0.016 | 0.006 | **+0.004** |

The shrinkage far exceeds the cost of the extra regressors (seven noise columns cost only −0.006).

**F6. Factor share.**
- In the linear rank span, 69% of the Transformer's IC is factor exposure in 2026-H1 and 78% in 2021–25.
- With raw scale, 12-1 momentum and regime conditioning counted, the members' estimates run from 70–90% (GPU Reviewer) to 85–100% (Red Team, Auditor) and 95–100% (ML Researcher).
- The ML Researcher notes that "69% is a lower bound" holds in direction but not by construction; nested residual ICs are the better report. (INTERPRETATION)

**F7. The method calibration had a hole.** (MECHANICAL FACT, Red Team) The synthetic test used Gaussian inputs, so the switch to rank-Gaussian scores passed by construction. The network actually sees raw, clipped, heavy-tailed inputs. The placebo floor also moves with the regime: monthly −0.093 to +0.068, and −0.130 in the prospective slice. A single +0.010 floor understates the method's noise, and any one-window prospective residual cannot be interpreted.

**F8. A production data-integrity defect: cache holes.** (MECHANICAL FACT; the Lead verified the holes and the refresh code path; the Auditor verified the splicing.)
- **Extent.** 70 of 110 cached stocks have missing sessions since 2016, 59 have runs of 5 or more, and 24 have holes in 2025–26. They are typically whole calendar months: 2317 and 9910 are missing all of September 2025, 2376 all of January 2026, and 2610 33 sessions from August 2025 to February 2026.
- **Cause.**
  - In `research/refresh_data.py`, a month that fails or comes back empty is skipped. A later month that succeeds advances the cache's last date, and the retry pass recomputes the needed months from that new date, so the missing month is never requested again and the stock is reported as refreshed.
  - The refresh gate (`research/pipeline_gate.py`) checks only coverage of the newest date. It never checks for holes inside the history.
  - Why holes arose before 2025 is not established.
- **Effect.**
  - `dataset_transformer_eod.py` builds features, windows and labels on each stock's own rows, so it splices straight across a hole.
  - `log_ret_1` can then carry a multi-week return as if it were one day, and momentum lookbacks shift.
  - About 3% of 60-row input windows and 6–9% of `mom_126_5` lookbacks are spliced. About 1% of training labels span more than 20 sessions.
- **No look-ahead.** Label maturity is tracked on the global calendar, so there is no leakage.
- **Production.** `inference_transformer_eod.py` builds the same spliced windows, so **the live model is fed spliced inputs**.
- **Effect on the residual.**
  - In 2026-H1, rows whose input lookback spans a hole are 12.6% of rows but carry 42–60% of the residual.
  - In 2021–25 those rows carry *negative* residual, and the small historical residual sits in clean rows.
  - So this is a regime-dependent interaction with a data defect, not a reliable signal either way.

## 3. Falsified, withdrawn or corrected claims

| claim | from | status |
|---|---|---|
| Residual concentrates in cache-gap *names*; clean names ≈ 0 | ML, Round 1 | **Withdrawn as a general claim.** It is a *window* effect in 2026-H1 only; historically the clean rows carry the small residual. |
| "Much of the residual is rebuilt longer momentum" (rank-space benchmarks) | Red Team, Round 1 | Narrowed. Rank-space momentum controls remove only about 0.01–0.025. Raw-scale inputs remove most of it (F5). |
| Rebuilt momentum explains at most a third | ML, Round 1 | Holds only against rank-space benchmarks. |
| 12-1 momentum cuts the residual to +0.028 | Lead, report 24 | Corrected in report 24. The like-for-like cut is 0.056 → 0.038; 0.028 was the 12-1-only set. |
| "69% is a lower bound on factor share" | Auditor, Round 1 | Holds in direction, not by construction. |
| Arms C and A PRESENT corroborate | implicit in report 24 | Rejected (F2). |
| Wait for 3 prospective windows before concluding | Auditor, Round 1 | Withdrawn. On-disk historical panels are the cheaper sample extension, if preregistered and labelled post-peek. |
| Features "VALID" | Auditor, Round 1 | Valid against the calendar rebuild, which differs from the network's own-row inputs on 7.4% of 2026-H1 rows. |

## 4. The five required questions: Council answers

1. **Does the Transformer contain residual signal?** Not demonstrated.
   - 2026-H1: +0.056 on the primary arm, CI including zero.
   - 2021–25: +0.014, CI including zero.
   - Beyond raw-scale inputs and 12-1 momentum, roughly zero in both samples.
   - What remains in 2026-H1 is momentum-conditional and partly tied to spliced windows.
   - Unanimous.
2. **How much of its OOS value is factor exposure?** At least 69% (2026-H1) to 78% (2021–25) in the linear rank span. Effectively 85–100% once raw scale, 12-1 momentum and regime conditioning are counted; the GPU Reviewer estimates 70–90%. Historically, plain 12-1 momentum beat the Transformer (+0.041 vs +0.021).
3. **DATA/TARGET, OPTIMISATION or CAPACITY?** **DATA/TARGET**, unanimous. That covers a single regime-dominated 20-session rank target, 10 close-derived inputs, a survivor universe, unadjusted prices, and now the cache-hole defect. Validation is a co-limit (ML Researcher, Auditor). Optimisation *capability* is not binding: the optimiser keeps fitting but only memorises. Capacity is not binding: v11 larger shapes were monotonically worse, and the v12–v13 312M model collapsed.
4. **Is an LR-schedule experiment justified?** **No**, unanimous.
   - Every member's gate for a non-factor residual (≥ 0.02 with lower CI > 0) fails on every benchmark and sample.
   - On the 9 burned P4-B0 blocks it would mostly measure momentum exposure.
   - A single conditional design was agreed for later (§5).
5. **Is architecture scaling justified?** **No**, unanimous. There is no residual to capture, the network is about 78% linear in its inputs historically, and prior scaling failed.

## 5. Decision-tree outcome and next work

**Case C, with a stronger reading close to Case A.**
- The primary label is WEAK, so formally Case C applies: no architecture scaling, and grow the independent sample first.
- The post-hoc evidence (F3, F5) points toward Case A: the network is mostly a factor extractor.
- The Council recommends acting on Case A's research direction without re-labelling: targets, independent data, a point-in-time universe, corporate-action-adjusted prices, and **data integrity first**.

**Recommended order (no GPU, no launch, all user-gated):**
1. **Cache-hole integrity (production-relevant; the user decides).**
   - Add a detection gate: flag any stock-month with zero rows when 90% or more of the universe has rows. Warn on history, and fail if the hole falls in the last 13 months.
   - Fix the refresh path: re-request every month missing against the union calendar, and merge by full date dedupe.
   - Re-fetch the missing months, then freeze a cache hash.
   - Then rescore existing checkpoints where they exist, or document that old panels describe "the pipeline as it ran".
   - Cost is a few hundred month requests. It changes production inputs, so it needs the user's approval.
2. **H-RESIDUAL-HIST (zero GPU, preregistered, post-peek).** On the 2021–25 historical panels:
   - features built the way the network sees them, plus raw-scale and 12-1 benchmarks;
   - clean-name and all-name co-primaries, and a spliced-window split;
   - block HAC with **one** df convention (non-overlapping windows − 1) for both stages;
   - the residual regressed on momentum payoff, reporting the intercept;
   - a block distribution for a 10-raw-input placebo instead of a single floor.
3. **H-FACTOR-PREMIUM long history (zero GPU).** Factor ICs 2016–2026 with the same df convention, to test whether momentum and volatility carry a premium outside the 2026 regime.
4. **Conditional only, design recorded, not launched: H-LR-SCHEDULE-EARLY.**
   - **Gate:** step 2's clean-name historical residual must be ≥ 0.02 with lower CI > 0 and a positive intercept. The GPU Reviewer expects it to fail.
   - **Data:** a repaired cache.
   - **Arms:** batch 1024 fixed; control at constant LR; treatment annealed to 0 between about 725 and 1,150 steps (epochs 3 → 5).
   - **Readout:** one, pre-declared, at the end of epoch 5, with no early stopping.
   - **Sample:** primary is the 2021–25 walk-forward (10 refits × 3 seeds); secondary is the 9 P4-B0 refits, read on validation only.
   - **Metric:** the paired change in clean-name residual IC beyond the network-matching 10 inputs + 12-1, clustered by refit.
   - **Guards:** raw IC must not fall; correlation with `mom_126_5` must rise by no more than 0.05.
   - **Parity:** a research-only fork of `fit_one` must reproduce the stored P4-B0 curves at λ ≡ 1.
   - **Cost:** about 40–70 GPU-minutes.
   - Never an epoch-50 fork.
5. **Operations:** `daily_ops` is manual and was not run on 2026-08-05, 08-06, 08-12, 08-13, 08-20, 09-03, 09-10 or 09-14 → 09-18. Those gaps also create the multi-month catch-ups that trigger the cache-hole bug. Automation is the user's call.

## 6. Dissent preserved

- **Red Team:** "artefact" must not be read as "non-predictive". A spliced window is effectively longer-horizon momentum, which predicts in momentum regimes. It also ranks the cache-hole fix *above* further residual work.
- **ML Researcher:** the data-integrity item must stand as its own workstream, not a limits footnote. Any historical addendum must use features built the way the network sees them, or it will report the same mixed number.
- **GPU Reviewer:** the historical panel should be the test bed for any future LR arm. The Red Team and Auditor counter that it is corroborating context, not a deciding sample, because it is post-peek, selection-burned and survivor-biased.
- **Auditor:** the historical panel is a bound, not a new label. Every conclusion stays "burned or post-peek" until prospective shadow history reaches at least 3 non-overlapping windows.

## 7. Process lessons

- **One df convention across stages** for overlapping labels.
- **Build benchmark features exactly the way the model does.** The calendar-versus-own-row mismatch was the biggest confounder, and nobody specified it in advance.
- **Calibrate on realistic inputs.** Test raw, heavy-tailed features, not Gaussian ones. Report a *distribution* of placebo residuals, not one floor.
- **Search the disk for existing panels before designing a new experiment.** The 2021–25 walk-forward panel gave ten times the independent sample for free. It also meant the historical question became post-peek the moment a member looked.
- **Research can expose production defects.** The cache-hole bug was found by a residual study and belongs in operations, not in a footnote.
