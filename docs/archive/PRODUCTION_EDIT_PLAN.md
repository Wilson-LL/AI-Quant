# Production Edit Plan

Status as of 2026-07-22 (RTX 4060 Ti 20h sprint):

**No production file edits planned or made.** `model.py`, `train.py`, `inference.py`,
`dataset.py` remain untouched. The transformer EOD daily-retrain system is built as
new, additive files (`dataset_transformer_eod.py`, `train_transformer_eod.py`,
`inference_transformer_eod.py`, `research/transformer_portfolio.py`,
`research/refresh_data.py`) that **import** `model.LSTM_CondTransformer` and reuse
train.py's training patterns (AdamW, gradient clipping, early stopping, checkpoint
format) without modifying them.

Rationale: zero risk to the validated D1.1 lineage and the original barrier-target
pipeline; rollback is file deletion.

If a production edit becomes necessary, this file must first be updated with:
reason, expected benefit, rollback, and tests — per sprint rules.

---

## Edit 1 — 2026-07-22 (continuous loop, cycle 8): `train_transformer_eod.py`

- **Change:** in `walkforward()`, emit one extra panel column `score_std`
  (cross-seed std of the ensemble predictions), computed from the same stacked
  tensor already built for the mean. No behavioral change to score, selection,
  training, or any consumer (all existing readers select columns by name).
- **Reason:** Track A2 (confidence/uncertainty filtering) needs per-date/name seed
  disagreement in walk-forward panels; daily inference already emits `score_std`
  but historical panels do not.
- **Expected benefit:** enables testing "drop low-confidence names" filters on
  cached panels without re-running experiments.
- **Rollback:** revert the 3-line diff (column is additive; old panels without the
  column remain valid).
- **Tests:** rerun a 1-seed fast config → `score_std` = 0 for single seed;
  multi-seed run → column present, mean/score unchanged vs pre-edit run
  (assert scores identical since mean path untouched).
- Production model.py / train.py / inference.py / dataset.py remain untouched.

## Edit 2 — 2026-07-23 (continuous loop, cycle 19): `train_transformer_eod.py`

- **Change:** optional `loss="pairwise"` mode in `fit_one()` (+ `loss` and
  `date_ranks` plumbing through `walkforward()`). Pairwise mode batches by DATE
  (same-date pairs are the only meaningful ranking pairs) and minimizes
  softplus(−(pᵢ−pⱼ)·sign(yᵢ−yⱼ)) over same-date pairs with |Δy| > 0.1.
  Default `loss="mse"` keeps the exact existing path — zero behavior change
  for all current callers.
- **Reason:** RL1 (queue v3 #1) — last untested allowed Track-A lever; the
  champion optimizes MSE on a rank target, which is not the ranking metric
  (val IC) it is selected on.
- **Expected benefit:** possible IC/IC-IR lift; if none, the line closes with
  evidence.
- **Rollback:** omit the parameter (default path byte-identical); revert diff.
- **Tests:** (1) default-path regression — a 2-epoch MSE fit before/after the
  edit produces identical val IC for the same seed; (2) pairwise smoke run
  (1 seed, 3 epochs) shows finite loss, nonzero grad flow, val IC computed.

## Edit 3 — 2026-07-24 (GPU research mode, queue v5): `train_transformer_eod.py`

- **Change:** `loss="listwise"` (ListNet top-1: CE between softmax(y/τ) and
  log-softmax(pred) per date group) reusing Edit 2's date-grouped batching.
  MSE default path untouched (dispatch only).
- **Reason:** RL2 screen in queue v5 (allowed Track-A item).
- **Rollback:** parameter default; revert diff.
- **Tests:** smoke via scheduler SMOKE config (MSE, regression-guarded by Edit
  2's identical-path check) + RL2 screen run itself (finite loss/val IC logged
  by the scheduler; a NaN/failure is recorded as status=failed, not skipped).

## Edit 4 — 2026-07-24 (GPU mode, A8/A9 verdicts): `train_transformer_eod.py`

- **Change:** `mode_daily_retrain` default seed list 5 → 7 (`list(range(7))`),
  implementing the adopted production spec. PRESETS untouched (experiments
  pass explicit seed lists). Inference loads whatever checkpoints exist.
- **Reason:** A8 dual-window win (blend 2.147/−10.6% & 1.443/−18.0%); A9
  showed the seed curve saturates at 7 (9 within tolerance, worse 2022/DD).
- **Expected benefit:** daily production ensemble matches the validated spec.
- **Rollback:** revert one line (or pass seeds explicitly).
- **Tests:** next daily retrain writes 7 checkpoints and a 7-seed train log;
  inference consumes them unchanged (it globs seed checkpoints).
