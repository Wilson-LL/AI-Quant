# Research rerun plan after DATA_BASELINE_V2 (plan only — nothing here has been run)

> **Superseded ordering (2026-09-22):** see `RERUN_PLAN_REVIEW_20260922.md`. The steps now run in this order:
> 1. champion and simple-baseline references;
> 2. Transformer vs momentum / ridge;
> 3. H-RESIDUAL-SIGNAL;
> 4. a measured V1→V2 materiality check;
> 5. P0 / P4 reruns only if step 4 triggers them.
>
> The review also categorises each prior result as STILL_VALID, SENSITIVITY_RERUN or FULL_RERUN_REQUIRED.

**Status:** research frozen. No GPU work and no model output until the user approves this plan, integrates `release/eod-integrity-ops`, and lifts the freeze.

Every earlier research result was computed on the holed cache (DATA_BASELINE_V1), using no gap guard. The comparisons below are therefore **re-measurements on V2 data with the gap guard on**, not continuations. Each item carries four fixed conditions:
- it pins the V2 manifest hash;
- it states in advance the metric and the decision rule it answers;
- it uses the already-burned intervals only for reconstruction, never for new selection;
- it reports V1 → V2 deltas separately from any model change.

## Priority 1 — champion and simple-baseline reconstruction (gate for everything else)

- **What:** retrain the production recipe on V2. Frozen preset, epochs, early stopping, LR, cadence, target fwd_20 and blend50 + band10 are all unchanged; the gap guard is on. Evaluate with the frozen walk-forward protocol. Alongside it, compute the simple baselines (momentum 126_5 alone, equal-weight universe, ridge panel) on V2.
- **Why first:** every downstream claim is relative to the champion and the baselines, so their V2 levels must exist before anything is compared.
- **Readout:**
  - V1 vs V2 for rank IC, top-quintile spread, blend50 + band10 paper Sharpe and turnover;
  - the number of training samples the guard excluded;
  - per-symbol attribution of the change for the formerly holed names.
- **Decision rule, fixed now:**
  - If the champion's V2 metric stays within the V1 seed-dispersion band, record DATA_REPAIR_NEUTRAL.
  - Otherwise, record the shift and re-anchor all references (2.06 / 1.47) to V2.
- **Cost:** CPU for the baselines. GPU only after approval, using the standard 5-seed daily-retrain budget with no scaling.

## Priority 2 — H-RESIDUAL-SIGNAL rerun

- **What:** rerun the preregistered residual-signal / factor-premium diagnostics (commits 5012d5b and 1858cdb) unchanged, on V2 panels. The rank-Gaussian scoring fix stays.
- **Why:** the holes spliced multi-week returns into `log_ret_1` and shifted rolling lookbacks, and those are the inputs of the residual decomposition. The rerun tests whether the earlier conclusion was an artefact of the holes.
- **Decision rule:** the same preregistered thresholds as the original audit, with no re-tuning. Any sign flip is reported as a V1 data artefact.

## Priority 3 — momentum / factor baseline comparison

- **What:** compare champion vs momentum 126_5 and vs the factor baselines on identical V2 cross-sections (the same symbols per date, integrity exclusions applied identically), using HAC / cluster-robust uncertainty.
- **Why:** the question "is the model more than momentum?" must be asked on the same repaired data. Before the repair, 20,281 feature windows and 2,062 label windows crossed holes (post-P1), and those contaminated both sides unevenly.
- **Decision rule:** the difference must exceed its HAC 95% interval on the non-burned interval; otherwise it is recorded as NOT_DISTINGUISHABLE.

## Priority 4 — reassess P0 / P4 invalidation

- **What:** list every P0 (refit cadence, daily refit parity) and P4 (P4-B0-LONG long-training) conclusion. For each, state whether it depended on data the repair changed: symbols/dates touched, and the share of affected samples. Rerun only the ones that did, under their original preregistration.
- **Readout:** a table with columns conclusion | V1 evidence | touched by repair (y/n, share) | action (KEEP / RERUN / VOID).
- **Constraint:** no new hypotheses, and no scaling, LR schedule, seeds, 40d, cadence or feature changes. This step only re-validates past decisions.

## Out of scope until the freeze is lifted

GPU research of any kind; new features, targets, horizons or cadences; changes to production architecture or portfolio construction.
