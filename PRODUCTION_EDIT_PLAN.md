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
