# Queue v12 verdict — Big Model / Recency / Epochs / Short-Side / Deep Inference

## VERDICT: KEEP_CURRENT_PRODUCTION
### Axis verdicts: REJECT_BIG_MODEL · REJECT_RECENCY_WEIGHTING · REJECT_EPOCH_EXPANSION · REJECT_SHORT_SIDE · REJECT_DEEP_INFERENCE

Closed by user 2026-08-01 after Phases 0–3 + short-side + deep-inference
(22 items done, 0 unresolved failures; L1/L3/XL1 held by the Option-A gate
— running them would interpolate between measured failures).

## Evidence per axis

**Big model (REJECT).** Size measured across four orders of magnitude:
114k (production, blend screen 1.969) → 3.8M (M2 1.730) → 5.4M (M3 1.480)
→ 312M (XL2: **collapsed to the mean** — train loss frozen at the rank-
target variance, val IC NaN; its 1.443 single-refit blend equals the
momentum leg alone, matched S_ref 1.418). The 1.19 GB checkpoint target was
achieved as an artifact (checkpoints/v12_big_transformer/, gitignored) and
certifies that ~276k samples × 10 features gives heavy capacity nothing to
learn. Feasibility engineering is real and retained: AMP + gradient
checkpointing + micro-batch/accumulation + subprocess-isolated OOM handling
(hard `AcceleratorError` OOMs poison the CUDA context — fresh process per
band is the fix).

**Recency weighting (REJECT, 10/10).** Hard windows 1y/2y/3y (0.38/0.59/
1.64), half-lives 63–504d (0.30–1.50), hybrids (0.55/1.40), calendar bands
(1.05) — ALL below the 1.969 equal-history bar, with monotonic dose-
response: less effective history → worse Sharpe, worse DD (−13% to −43%),
higher turnover, lower book overlap. Old data is load-bearing.

**Epoch expansion (REJECT, 4/4).** A perfect anti-correlation staircase:
val IC 0.0481→0.0503→0.0516→0.0628 as the book fell 1.969→1.838→1.798→
1.326. The 25-epoch/patience-3 early stop is a regularizer, not a budget
limitation. (Dissociations #8–9 on the standing ledger.)

**Short side (REJECT, 6/6 cells).** Short leg standalone NEGATIVE
everywhere (−0.64…−1.02 after 2× costs), hit rate ~49%, mean worst adverse
20d move +16–19% — on a survivorship-biased universe that flatters shorts.
Recorded lead (not a candidate): shorts act as an expensive hedge — CH
blend 100/50 gave 2.00/−17.6% vs L100 1.92/−26.7% — but fail the
independent-value and bear-window gates; adjacent to the closed B3 hedge
line.

**Deep inference (REJECT).** 97-pass uncertainty run costs 17 s, not 5 h.
MC-dropout and feature-noise stds correlate 0.89/0.77 with the free seed
std — redundant information. Seed std weakly predicts realized rank error
(+0.06/+0.07) but acting on it destroys the book (drop-high-std tercile:
CH 2.147→−1.768, BR 1.443→−0.393): the high-disagreement names ARE the
alpha engine. Confidence filtering stays closed with its sharpest evidence
yet. Optional free diagnostic (not adopted): the 17 s stability report
(UNSTABLE list) could join daily ops if ever wanted.

## Standing state after v12

Production unchanged and re-validated: preset B / close_only / tgt_rank_20 /
7 seeds / equal-weight all history / 25-epoch early stop / batch 1024 /
blend50+band10. Newly closed lines: model scale (any), training-window
truncation, recency weighting (all forms), epoch expansion, short side
(current signals), inference-time uncertainty exploitation. Flag-gated
research keys added this arc (all default-OFF, champion path anchor-verified
×3): `batch`, `min_epochs`, `keep_history`, `recency.bands`.

The one hypothesis v12 could NOT test: whether richer input information
(features / universe) changes the big-model answer — close_only's 10
features may simply be too little for any capacity to matter. That is a
new research question, not a reopening of these closed lines.

## Artifacts

queue_v12.json · manifest · results.{csv,md} · gpu_usage.csv ·
phase1_feasibility.json · recency_window_comparison.{csv,md} ·
sample_weight_diagnostics.csv · train_window_diagnostics.md ·
epoch_curves.csv · epoch_depth_report.md · phase3_xl2_reduced.json ·
short_side_report.md · short_side_metrics.csv ·
short_candidate_diagnostics.csv · short_worst_trades.csv ·
queue_v12_deep_inference_report.md · logs/ · panels + checkpoints
(gitignored).
