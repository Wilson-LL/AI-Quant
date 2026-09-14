# 18 — P4-A: validation-policy diagnosis (and P4-B preregistration)

Sources: `research/audit_v17_validation_policy.py` → `p4/validation_policy_example.json`;
`research/audit_v17_val_oos_diagnosis.py` → `p4/val_oos_diagnosis.{csv,json}`,
`p4/val_oos_detrended.json`. All CPU, from existing P0 fits and the production
split functions. **Burned methodology diagnostic** — nothing here is OOS validation.

## 1. The current validation policy, exactly (as-of 2026-09-11, live cache)

| Item | Value |
|---|---|
| History | 2015-10-19 → 2026-09-11, 2,850 sessions |
| Gradient-training dates | 2015-10-19 → **2025-06-16** (2,355 sessions, 247,385 samples) |
| Purge gap | 21 sessions (no rows) |
| Validation block | **2025-07-16 → 2026-08-13**, 263 contiguous sessions (28,106 samples) = 10.0% of matured dates |
| Immature tail (labels not yet realized) | 21 sessions (2026-08-14 → 09-11) |
| **Newest sessions never used for gradients** | **305 sessions = 452 calendar days ≈ 14.8 months** |
| Movement per daily refit | every boundary slides by exactly one session (train_last 06-13 → 06-16 across the 09-10 → 09-11 refits; val 07-15→07-16 … 08-12→08-13) |
| Early stopping | per epoch: mean per-date Spearman IC on the validation block; keep best; stop after 3 non-improving epochs (min 2, max 25) |
| Final production weights | **the best-validation-IC epoch** (`net.load_state_dict(best_state)`) |
| Validation rows in gradients | never |
| Refit on train + validation afterwards | **NO** (verified: `mode_daily_retrain` calls `fit_one` once per seed with `(tr, va)`) |

So the precise statement is: *on any production date the newest ~305
sessions (≈ 14.8 months) contribute no gradient — 21 are immature, 21 are
purge, 263 are the validation block used only to pick the epoch.*

## 2. Why validation IC looked anti-predictive — diagnosis

Per-refit table (27 five-session refits × 3 seeds; 131 daily refits × 3):

| Association (Spearman) | 5-session | daily |
|---|---|---|
| raw: val IC vs subsequent block OOS IC (ensemble) | **−0.66** (boot 95% −0.82…−0.38) | **−0.53** (−0.63…−0.41) |
| per seed | −0.66 / −0.64 / −0.60 | −0.57 / −0.54 / −0.46 |
| trimmed (drop 5% tails) | −0.47 | −0.41 |
| **by quarter** — 2026Q1 / Q2 / Q3 | +0.07 / −0.66 / n=4 | +0.05 / −0.40 / −0.18 |
| val-IC mean by quarter | 0.067 → 0.153 → 0.152 | 0.069 → 0.154 → 0.151 |
| OOS-IC mean by quarter | 0.255 → 0.013 → −0.011 | 0.245 → 0.030 → −0.048 |
| **time-detrended partial correlation** (both series residualized on a quadratic time trend) | **+0.004** | **+0.029** |
| within-quarter demeaned | −0.20 | −0.17 |
| within-refit across seeds (pure selection test: same data, seeds differ) | +0.08 | −0.14 |
| epochs run vs OOS / vs val | −0.30 / +0.25 | (seed-level −0.13) |
| val-IC variance: between refits vs within-refit across seeds | 0.047 vs 0.011 | |
| OOS IC by val-IC tercile (low / mid / high) | 0.22 / 0.19 / **−0.09** | |
| regimes | validation blocks 22 BULL / 5 BEAR; every OOS block BULL; same-regime pairs n=22 ρ −0.51 | |

**Reading.** The strong negative correlation is almost entirely a
**time-trend confound**: as the sliding validation window moved into the
strong 2025H2–2026Q1 period, own-validation IC rose (0.07 → 0.15) while
the OOS period itself got harder (0.25 → 0.03 → −0.01). Removing the
common time trend leaves **≈ 0 association** (+0.004 / +0.03), and the
seed-level test — same refit, same data, seeds differing — is also ≈ 0.
Hypotheses A–E:

- **A / B (stale validation regime; regime mismatch): the main driver** —
  the validation block is 14 months stale and its content drifts slowly
  while the OOS regime changes faster; the correlation is between two
  trends, not between model quality and selection.
- **C (small-sample noise)**: the raw correlation is not noise (bootstrap
  CI excludes 0) but the *interpretation* was; the detrended estimate has
  wide uncertainty around 0.
- **D (seed selection noise)**: not the cause — per-seed results agree and
  within-refit seed variance (0.011) is a quarter of between-refit
  variance (0.047).
- **E (early-stopping over-selection)**: weak, indirect support only —
  refits that ran more epochs (higher val IC) had slightly worse OOS
  (ρ −0.30 at 5-session); needs the per-epoch diagnostic (P4-B0).

**Corrected statement** (supersedes the P0-A wording "refits that look
best on the holdout do worst afterwards"): *own-validation IC is
essentially uninformative about subsequent OOS IC at this granularity;
the apparent negative relationship is a shared time trend.* This still
motivates the policy screen — a holdout that is 14 months stale carries
little selection signal — but it does not show that the current rule
actively selects bad models.

## 3. Epoch-level early-stopping diagnostic

Not available from existing artifacts: production and P0 fits keep only
the best-epoch weights and per-epoch *validation* IC/train loss (fit
history), not per-epoch OOS predictions. Preregistered as **P4-B0** (9
refits × 3 seeds, early stopping disabled for 15 epochs, per-epoch val
IC / train loss / 5-session OOS block IC via a research-process-only wrap
of `predict_idx`; ~1 h GPU). Reports: correlation(val IC, OOS IC) across
epochs; regret of the validation-best epoch vs the ex-post OOS-best
epoch; OOS IC at a fixed epoch near the median stop (6). Not run.

## 4. P4-B preregistration (frozen; not launched) — `p4_validation_policy_spec.json`

Policies (5-session cadence, seeds {0,1,2}, 131 dates, everything else identical):
- **A_current** — the production split/early-stop rule; fits **reused
  from P0 arm B** (identical function, seeds, dates, config hash: 81 fits, 0 new).
- **B_recent126** — validation = most recent 126 matured sessions
  (ending 21 before the refit); train = all matured dates ending 21
  before validation; early stop on that block; best-val epoch; no refit on train+val.
- **C_recent63** — as B with 63 sessions.
- **D_calibrate_then_train_all** — E = policy B's best-validation epoch
  index + 1 (reused), then retrain the same seed from scratch on
  train + validation for E epochs with early stopping disabled
  (patience 10⁶, min = max = E). Implementation caveat documented in the
  spec: `fit_one` restores the best in-sample-IC epoch among the E
  (typically the last); an inert `restore_best=False` kwarg would make it
  exact — proposed, **not applied** (production file untouched). No
  post-refit information at any step.

Metrics/thresholds frozen in the spec (primary: subsequent rank IC + HAC
SE, positive-IC frequency, spread, time-detrended val→OOS association;
secondary: seed dispersion, refit rank stability, top-Q overlap;
portfolio check: net100, max DD, turnover). PROMISING requires ΔIC ≥
+0.005 and > 1 HAC SE with no secondary/portfolio deterioration and all
three seeds individually ≥ A; NOT_PROMISING at ΔIC ≤ −0.005 or two
failed checks; else VALIDATION_POLICY_INCONCLUSIVE.

Compute (measured 52.9 s/fit): B 81 + C 81 + D 81 = **243 fits ≈ 3.6 h**;
P4-B0 27 fits × ~130 s ≈ **1.0 h**; total **≈ 4.6 h** (≈ 5.5 h at the
p90 70 s/fit). To start only after tonight's `daily_ops` completes and
the GPU is idle.
